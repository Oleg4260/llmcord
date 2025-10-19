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
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/Oleg4260/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

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
    if user.id not in history_settings:
        history_settings[user.id] = True

    await interaction.response.defer(ephemeral=True)
    if not config["read_history"]:
        await interaction.followup.send("Channel history is disabled in the configuration.", ephemeral=True)
        return

    history_settings[user.id] = not(history_settings[user.id])
    await interaction.followup.send(f"Channel history {"enabled" if history_settings[user.id] else "disabled"} for user {user.name}.")

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
    global wiki_data
    if config["use_wiki"]:
        download_wiki()
    logging.info(f"Logged in as {discord_bot.user}")
    await discord_bot.tree.sync()
    logging.info("Commands synced.")

@discord_bot.event
async def on_message(new_msg) -> None:

    global msg_nodes, last_task_time, wiki_data, history_settings

    config = await asyncio.to_thread(get_config)

    if config["use_wiki"] and new_msg.author.id == config["webhook_id"]:
        logging.info("Webhook message received, updating wiki cache")
        download_wiki()

    is_dm = new_msg.channel.type == discord.ChannelType.private
    if new_msg.author.id not in history_settings and config["read_history"]:
        history_settings[new_msg.author.id] = True

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

    extra_headers = provider_config.get("extra_headers", None)
    extra_query = provider_config.get("extra_query", None)
    extra_body = (provider_config.get("extra_body", None) or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    attachment_whitelist = ("text", "image") if accept_images else ("text")

    # Build message chain and set user warnings
    messages = []
    channel_history = []
    chain_ended = False
    history_enabled = config["read_history"] and not is_dm and history_settings[new_msg.author.id]
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                msg_date_local = curr_msg.created_at.replace(tzinfo=dt.UTC).astimezone(timezone)
                formatted_message = f"{msg_date_local.strftime('%d.%m.%Y %H:%M')} {curr_msg.author.name}: " + curr_msg.content

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in attachment_whitelist)]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([formatted_message] if curr_msg.content != discord_bot.user.mention else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                try:
                                    curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                                except discord.NotFound:
                                    logging.warning(f"Thread starter message {parent_msg_id} not found")
                                    curr_node.parent_msg = None
                                    curr_node.fetch_parent_failed = True
                            else:
                                try:
                                    curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)
                                except discord.NotFound:
                                    logging.warning(f"Referenced message {parent_msg_id} not found")
                                    curr_node.parent_msg = None
                                    curr_node.fetch_parent_failed = True

                except (discord.NotFound, discord.HTTPException) as e:
                    logging.exception(f"Error fetching parent message: {e}")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)
                messages.append(message)

            if history_enabled and not chain_ended and not curr_node.parent_msg:
                chain_ended = True
                messages.append(dict(role="system", content="Channel history ends here. Current conversation starts from the next message. Do not refer to any previous topics unless the user does. Revert behaviour to defaults defined in the system prompt."))

            if chain_ended and len(messages) < max_messages:
                if not channel_history:
                    channel_history = [msg async for msg in curr_msg.channel.history(before=curr_msg, limit=max_messages - len(messages))]
                curr_node.parent_msg = channel_history[0]
                del channel_history[0]
            
            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg
    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")
    
    if chain_ended:
        messages.append(dict(role="system", content=f"Channel history starts here. Below are the latest messages in the #{new_msg.channel.name} channel. Ignore them unless the user directly refers to them."))
    # Get info about members in the channel
    members_list = []
    if not is_dm:
        for member in new_msg.guild.members:
            if new_msg.channel.permissions_for(member).read_messages and (member == discord_bot.user or not member.bot):
                member_info = {
                    "id":member.id,
                    "name":member.name,
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
    emojis_list = [f"<{"a" if e.animated else ""}:{e.name}:{e.id}>" for e in discord_bot.emojis]
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
        system_prompt_extras.append("You are currently in a DM channel.")
    # Add content from wiki
    if config["use_wiki"]:
        system_prompt_extras.append(f"Content of all wiki pages: {wiki_data}")
    full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
    messages.append(dict(role="system", content=full_system_prompt))
    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
    extra_api_parameters = config["extra_api_parameters"]
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
