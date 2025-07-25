import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
import datetime as dt
from zoneinfo import ZoneInfo
import logging
from typing import Literal, Optional
import discord
import httpx
from openai import AsyncOpenAI
import yaml
import dokuwiki as wiki

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude", "gemini", "gemma", "llama", "pixtral", "mistral", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

MAX_MESSAGE_NODES = 500

def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


config = get_config()

prompt_file = open(config["prompt_file"], "r")

timezone = ZoneInfo(config["timezone"])

bot_token = config["bot_token"]
system_prompt = prompt_file.read()

status_message = config["status_message"] or "github.com/Oleg4260/llmcord"

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = True
activity = discord.CustomActivity(name=status_message[:128])
discord_client = discord.Client(intents=intents, activity=activity)
tree = discord.app_commands.CommandTree(discord_client)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0
wiki_data = []

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_client.event
async def on_ready():
    global wiki_data
    if config["use_wiki"]:
        wiki_data = wiki.download_all_pages(config["wiki_url"],config["wiki_token"])
        logging.info(f"Wiki data received, {len(wiki_data)} articles downloaded.")
    logging.info(f"Logged in as {discord_client.user}")
    await tree.sync()
    logging.info("Commands synced.")

@tree.command(name="clear", description="Deletes all of the bot's messages in the current DM channel.")
async def clear(interaction: discord.Interaction):
    """Deletes all of the bot's messages in the current DM channel."""
    if interaction.channel.type != discord.ChannelType.private:
        await interaction.response.send_message("This command can only be used in DMs.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    
    deleted_count = 0
    async for message in interaction.channel.history(limit=None):
        if message.author == discord_client.user:
            try:
                await message.delete()
                deleted_count += 1
            except discord.HTTPException as e:
                logging.error(f"Failed to delete message {message.id}: {e}")

    await interaction.followup.send(f"Successfully deleted {deleted_count} bot message(s).")


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time, wiki_data

    config = await asyncio.to_thread(get_config)

    if config["use_wiki"] and new_msg.author.id == config["webhook_id"]:
        logging.info("Webhook message received, updating wiki data")
        wiki_data = wiki.download_all_pages(config["wiki_url"],config["wiki_token"])
        logging.info(f"Wiki data received, {len(wiki_data)} articles downloaded.")

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_client.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    allow_dms = config["allow_dms"]
    permissions = config["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    providers = config["providers"]
    provider_slash_model = config["model"]

    provider, model = provider_slash_model.split("/", 1)
    base_url = providers[provider]["base_url"]
    api_key = providers[provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config["max_text"]
    max_images = config["max_images"] if accept_images else 0
    max_messages = config["max_messages"]

    attachment_whitelist = ("text", "image") if accept_images else ("text")

    # Build message chain and set user warnings
    messages = []
    channel_history = []
    chain_ended = False
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
                    ([formatted_message] if curr_msg.content != discord_client.user.mention else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
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

            if not chain_ended and not is_dm and config["read_history"] and curr_node.parent_msg is None:
                chain_ended = True
                messages.append(dict(role="system", content="Channel history ends here. Current message chain starts below. (Only next messages are related to the current topic.)"))

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
    if chain_ended:
        messages.append(dict(role="system", content=f"Latest channel history starts here. Below are the latest messages in the #{new_msg.channel.name} channel above the current message chain."))

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")
    # Get info about members in the channel
    members_list = []
    if not is_dm:
        for member in new_msg.guild.members:
            if new_msg.channel.permissions_for(member).read_messages:
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
    # Add extras to system prompt
    system_prompt_extras = [
        f"Current date and time ({str(timezone)}): {dt.datetime.now(timezone).strftime('%b %-d %Y %H:%M:%S')}",
        f"Custom emojis available: {str(discord_client.emojis)}"
        ]
    if not is_dm:
        system_prompt_extras.append(f"Current server name: {new_msg.guild.name}, Current channel name: {new_msg.channel.name}")
        system_prompt_extras.append(f"Current members in the channel: {members_list}")
    else:
        system_prompt_extras.append("You are currently in a DM channel.")
    # Add content from wiki
    if config["use_wiki"]:
        system_prompt_extras.append(f"Content of all wiki pages: {wiki_data}")
    full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
    messages.append(dict(role="system", content=full_system_prompt))
    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    extra_api_parameters = config["extra_api_parameters"]

    max_message_length = 2000

    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(model=model, messages=messages[::-1], stream=True, extra_body=extra_api_parameters):
                if finish_reason != None:
                    break

                finish_reason = curr_chunk.choices[0].finish_reason

                prev_content = curr_content or ""
                curr_content = curr_chunk.choices[0].delta.content or ""

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

async def main():
    await discord_client.start(bot_token)

asyncio.run(main())