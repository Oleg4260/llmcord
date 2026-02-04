import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
import datetime as dt
from zoneinfo import ZoneInfo
import logging
from typing import Any, Literal, Optional
import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml
import dokuwiki as wiki

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

MAX_MESSAGE_NODES = 500

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

config = get_config()

prompt_file = open(config["prompt_file"], "r")

timezone = ZoneInfo(config["timezone"])

bot_token = config["bot_token"]
system_prompt = prompt_file.read()

httpx_client = httpx.AsyncClient()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
wiki_data = []
history_settings = {}

def download_wiki():
    global wiki_data
    try:
        wiki_data = wiki.download_all_pages(config["wiki_url"],config["wiki_token"])
        logging.info(f"Wiki data received, {len(wiki_data)} articles downloaded.")
    except Exception as e:
        logging.error(f"Could not load wiki data: {e}")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/Oleg4260/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="clear", description="Delete all of the bot's messages in the current DM channel.")
async def clear(interaction: discord.Interaction):
    """Deletes all of the bot's messages in the current DM channel."""
    if interaction.channel.type != discord.ChannelType.private:
        await interaction.response.send_message("This command can only be used in DMs.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    
    deleted_count = 0
    async for message in interaction.channel.history(limit=None):
        if message.author == discord_bot.user:
            try:
                await message.delete()
                deleted_count += 1
            except discord.HTTPException as e:
                logging.error(f"Failed to delete message {message.id}: {e}")

    await interaction.followup.send(f"Successfully deleted {deleted_count} bot message(s).")

@discord_bot.tree.command(name="history", description="Toggle use of channel history.")
async def history(interaction: discord.Interaction):
    global history_settings
    user = interaction.user

    await interaction.response.defer(ephemeral=True)
    if not config.get("read_history", False):
        await interaction.followup.send("Channel history is disabled in the configuration.", ephemeral=True)
        return

    history_settings[user.id] = not history_settings.get(user.id, True)
    await interaction.followup.send(f"Channel history {'enabled' if history_settings[user.id] else 'disabled'} for user {user.name}.")

@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))

@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]

@discord_bot.event
async def on_ready() -> None:
    logging.info(f"Logged in as {discord_bot.user}.")
    if await discord_bot.tree.sync():
        logging.info("Commands synced.")

    if config.get("use_wiki", False):
        download_wiki()

@discord_bot.event
async def on_message(new_msg) -> None:

    global msg_nodes, last_task_time, wiki_data, history_settings

    config = await asyncio.to_thread(get_config)

    if new_msg.author.id == config["webhook_id"] and config.get("use_wiki", False):
        logging.info("Webhook message received, updating wiki cache")
        download_wiki()

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    allow_dms = config.get("allow_dms", True)
    config = await asyncio.to_thread(get_config)

    permissions = config["permissions"]
    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    attachment_whitelist = ("text", "image") if accept_images else ("text")

    async def format_message(msg: discord.Message):
        """
        Given a discord.Message, return a tuple:
          (message_dict or None, node)
        message_dict is a dict ready to append to `messages` (role/content/optional name)
        node is the MsgNode object for further introspection (parent_msg, etc)
        This function fills msg_nodes cache and fetches attachments/parent as needed.
        """
        node = msg_nodes.setdefault(msg.id, MsgNode())

        async with node.lock:
            if node.text is None:
                try:
                    msg_date_local = msg.created_at.replace(tzinfo=dt.timezone.utc).astimezone(timezone)
                except Exception:
                    msg_date_local = msg.created_at

                formatted_message = f"{msg_date_local.strftime('%d.%m.%Y %H:%M')} {msg.author.name}: " + (msg.content or "")
                
                attachments = []
                for att in msg.attachments:
                    if att.content_type and any(att.content_type.startswith(x) for x in attachment_whitelist):
                        if not att.content_type.startswith("image") or len([a for a in attachments if a.content_type.startswith("image")]) < max_images:
                            attachments.append(att)

                attachment_responses = []
                if attachments:
                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in attachments])

                node.text = "\n".join(
                    ([formatted_message] if (msg.content and msg.content != discord_bot.user.mention) else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, getattr(embed.footer, 'text', None)))) for embed in msg.embeds]
                    + [component.content for component in msg.components if getattr(component, "type", None) == discord.ComponentType.text_display]
                    + [f"[{att.filename}]" for att in msg.attachments]
                    + [resp.text for att, resp in zip(attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                node.role = "assistant" if msg.author == discord_bot.user else "user"
                node.user_id = msg.author.id if node.role == "user" else None

                try:
                    if (
                        msg.reference == None
                        and discord_bot.user.mention not in (msg.content or "")
                        and (prev_msg_in_channel := ([m async for m in msg.channel.history(before=msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if msg.channel.type == discord.ChannelType.private else msg.author)
                    ):
                        node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and msg.reference == None and getattr(msg.channel, "parent", None) and msg.channel.parent.type == discord.ChannelType.text

                        parent_msg_id = None
                        if parent_is_thread_start:
                            parent_msg_id = msg.channel.id
                        else:
                            parent_msg_id = getattr(msg.reference, "message_id", None)

                        if parent_msg_id:
                            if parent_is_thread_start:
                                try:
                                    node.parent_msg = msg.channel.starter_message or await msg.channel.parent.fetch_message(parent_msg_id)
                                except discord.NotFound:
                                    logging.warning(f"Thread starter message {parent_msg_id} not found")
                                    node.parent_msg = None
                            else:
                                try:
                                    node.parent_msg = msg.reference.cached_message or await msg.channel.fetch_message(parent_msg_id)
                                except discord.NotFound:
                                    logging.warning(f"Referenced message {parent_msg_id} not found")
                                    node.parent_msg = None
                except (discord.NotFound, discord.HTTPException) as e:
                    logging.exception(f"Error fetching parent message: {e}")

            if node.images and max_images > 0:
                content = ([dict(type="text", text=(node.text[:max_text] if node.text else ""))] if (node.text and node.text[:max_text]) else []) + node.images[:max_images]
            else:
                content = node.text[:max_text] if node.text else ""

            if content != "":
                message = dict(content=content, role=node.role)
                if accept_usernames and node.user_id is not None:
                    message["name"] = str(node.user_id)
            else:
                message = None

        return message, node

    # Build reply chain (from newest to oldest)
    messages = []
    history_enabled = config.get("read_history", False) and not is_dm and history_settings.get(new_msg.author.id, True)
    curr_msg = new_msg
    oldest_chain_msg = None

    while curr_msg is not None and len(messages) < max_messages:
        msg_dict, node = await format_message(curr_msg)

        if msg_dict:
            messages.append(msg_dict)

        if node.parent_msg is None:
            oldest_chain_msg = curr_msg

        curr_msg = node.parent_msg

    # Channel history (fetched above the oldest message of reply chain)
    if history_enabled and len(messages) < max_messages and oldest_chain_msg is not None:
        try:
            channel_history = [m async for m in new_msg.channel.history(before=oldest_chain_msg, limit=(max_messages - len(messages) + 2))] # 2 system messages
            messages.append(dict(role="system", content="Channel history ends here. Current conversation (reply chain) starts from the next message. Do not refer to any previous topics unless the user does. Revert behaviour to defaults defined in the system prompt."))
            for msg in channel_history:
                msg_dict, node = await format_message(msg)
                if msg_dict:
                    messages.append(msg_dict)
            messages.append(dict(role="system", content="Channel history starts here. Below are the latest messages in the channel. Ignore them unless the user directly refers to them."))
        except Exception as e:
            logging.exception(f"Error fetching channel history: {e}")

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{messages[0]["content"]}")

    # Get info about members in the channel
    members_list = []
    if not is_dm:
        for member in new_msg.guild.members:
            if new_msg.channel.permissions_for(member).read_messages and (member == discord_bot.user or not member.bot):
                member_info = {
                    "id":member.id,
                    "username":member.name,
                    "display_name":member.display_name,
                    "global_name":member.global_name,
                    "status": str(member.status),
                    "activities": [str(activity) for activity in member.activities],
                    "roles": [
                        {"name":role.name, "id":role.id, "color":str(role.color)}
                        for role in member.roles[1:] # Remove @everyone
                    ],
                    "created_at":str(member.created_at),
                    "joined_at":str(member.joined_at)
                }
                members_list.append(member_info)
    # Make emojis list
    emojis_list = [f"<{'a' if e.animated else ''}:{e.name}:{e.id}>" for e in discord_bot.emojis]
    # Add extras to system prompt
    system_prompt_extras = [
        f"Current date and time ({str(timezone)}): {dt.datetime.now(timezone).strftime('%b %-d %Y %H:%M:%S')}",
        f"Current model: {curr_model}",
        f"Custom emojis available: {emojis_list}"
        ]
    if not is_dm:
        if not history_enabled:
            system_prompt_extras.append("Access to channel history is disabled. Only current conversation is visible.")
        system_prompt_extras.append(f"Server name: {new_msg.guild.name}, Channel name: {new_msg.channel.name}, Channel topic: {new_msg.channel.topic}")
        system_prompt_extras.append(f"Users in the channel: {members_list}")
    else:
        system_prompt_extras.append(f"You are currently in a DM channel with user {new_msg.author.name}.")
    # Add content from wiki
    if wiki_data:
        system_prompt_extras.append(f"Content of all wiki pages: {wiki_data}")
    full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
    messages.append(dict(role="system", content=full_system_prompt))

    messages = messages[::-1] # Reverse message order, because the list was generated from newer to older

    # Generate dump of chat history (for debug purposes)
    #with open("history.txt", "w") as f:
    #    for msg in messages:
    #        f.write(f"[{msg['role']}]: {msg['content']}\n")
    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    openai_kwargs = dict(model=model, messages=messages, stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
    max_message_length = 2000

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

            for content in response_contents:
                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                response_msgs.append(response_msg)

                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()

    except Exception as e: # Catch the exception and assign it to variable 'e'
        logging.exception("Error while generating response")
        error_message = f"`{e}`"
        await new_msg.reply(content=error_message, suppress_embeds=True)

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)

async def main() -> None:
    await discord_bot.start(config["bot_token"])

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass