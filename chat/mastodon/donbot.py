#!/usr/bin/env python3
"""
donbot.py

Chat with your persyn on Mastodon.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import random
import sys
import tempfile
import uuid

from pathlib import Path
from hashlib import sha256

from bs4 import BeautifulSoup
from mastodon import Mastodon, MastodonError, MastodonMalformedEventError, StreamListener

import requests
import spacy

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../../').resolve()))

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# Reminders
from interaction.reminders import Reminders

# Mastodon support for image posting
from chat.mastodon.login import mastodon

# Common chat library
from chat.common import Chat

persyn_config = load_config()

# Chat library
chat = Chat(persyn_config, service='mastodon')

# Coroutine reminders
reminders = Reminders()

try:
    mastodon = Mastodon(
        access_token = persyn_config.chat.mastodon.secret,
        api_base_url = persyn_config.chat.mastodon.instance
    )
except (MastodonError, AttributeError):
    raise SystemExit("Invalid credentials, run masto-login.py and try again.") #pylint: disable=raise-missing-from

log.info(f"🎺 Logged in as: {mastodon.me().url}")

def say_something_later(channel, when=1, what=None, status=None):
    ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
    reminders.cancel(channel)

    if what:
        reminders.add(channel, when, toot, args=[what, status])
    else:
        # Yadda yadda yadda
        reminders.add(channel, when, dispatch, args=[channel, '...', status])

def synthesize_image(channel, prompt, engine="stable-diffusion", style=None, model=None):
    ''' It's not AI art. It's _image synthesis_ '''
    chat.take_a_photo(channel, prompt, engine=engine, style=style, model=model, width=768, height=768, guidance=15)

    ents = chat.get_entities(prompt)
    if ents:
        chat.inject_idea(channel, ents)

def following(account_id):
    ''' Return true if we are following this account '''
    return account_id in [follower.id for follower in mastodon.account_following(id=mastodon.me().id)]

def get_text(msg):
    ''' Extract just the text from a message (no HTML or @username) '''
    return BeautifulSoup(msg, features="html.parser").text.strip().replace(f'@{mastodon.me().username} ','')

def fetch_and_post_image(url, msg):
    ''' Download the image at URL and post it to Mastodon '''
    media_ids = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            fname = f"{tmpdir}/{uuid.uuid4()}.{url[-3:]}"
            with open(fname, "wb") as f:
                for chunk in response.iter_content():
                    f.write(chunk)
            caption = chat.get_caption(url)
            media_ids.append(mastodon.media_post(fname, description=caption).id)

            resp = mastodon.status_post(
                msg,
                media_ids=media_ids,
                idempotency_key=sha256(url.encode()).hexdigest()
            )
            if not resp or 'url' not in resp:
                raise RuntimeError(resp)
            log.info(f"🎺 Posted {url}: {resp['url']}")

    except RuntimeError as err:
        log.error(f"🎺 Could not post {url}: {err}")


class TheListener(StreamListener):
    ''' Handle streaming events from Mastodon.py '''

    def __init__(self):
        self.channel = mastodon.me().url

     # def on_update(self, update):
     #      print(f"Got update: {update}")

     # def on_conversation(self, conversation):
     #      print(f"Got conversation: {conversation}")

    def on_notification(self, notification):
        ''' Handle notifications '''
        if 'status' not in notification:
            log.info("📪 Ignoring non-status notification")
            return

        log.info("📫 Notification:", notification.status.url)

        if notification.type == "favourite":
            log.info("⭐️")
            return

        if not following(notification.status.account.id):
            log.warning("📪 Not following, so ignoring:", notification.status.account.acct)
            return

        msg = get_text(notification.status.content)
        log.info(f"📬 {notification.status.account.acct}:", msg)

        dispatch(self.channel, msg, notification.status)

    def handle_heartbeat(self):
        log.debug("💓")

def paginate(text, maxlen=persyn_config.chat.mastodon.toot_length):
    ''' Break a single status string into a list of toots < the posting limit. '''
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('sentencizer')

    doc = nlp(text)
    posts = []
    post = ""
    for sent in [str(sent) for sent in doc.sents]:
        if len(sent) > maxlen:
            # edge case, just truncate it
            sent = sent[:maxlen - 1] + '…'
        if len(post) + len(sent) < maxlen - 1:
            post = f"{post} {sent}"
        else:
            posts.append(post)
            post = f"{sent}"

    if post:
        posts.append(post)

    return posts

def toot(status, to_status=None, **kwargs):
    '''
    Quick send a toot or reply.

    If status is longer than the max toot length, post a thread.

    Returns a list of all status messages posted.
    '''
    rets = []
    for post in paginate(status):
        if to_status:
            rets.append(mastodon.status_reply(to_status, post, **kwargs))
            log.info("🎺 Posted reply:", rets[-1].url)
        else:
            rets.append(mastodon.status_post(post, **kwargs))
            log.info("🎺 Posted:", rets[-1].url)
            to_status = rets[-1]

    return rets

def dispatch(channel, msg, status=None):
    ''' Handle commands and replies '''

    if msg.startswith('🎨'):
        synthesize_image(channel, msg[1:].strip(), engine="stable-diffusion")

    elif msg.startswith('🦜'):
        prompt = msg[1:].strip()
        style = chat.prompt_parrot(prompt)
        log.warning(f"🦜 {style}")
        synthesize_image(channel, prompt, engine="stable-diffusion", style=style)

    elif msg.strip() == '🤳':
        synthesize_image(
            channel,
            f"{persyn_config.id.name} takes a selfie",
            engine="stylegan2",
            model=random.choice(["ffhq", "waifu"])
        )

    else:
        if status:
            (the_reply, goals_achieved) = chat.get_reply(channel, msg, status.account.username, status.account.id)
            my_response = toot(
                the_reply,
                to_status=status
            )
        else:
            (the_reply, goals_achieved) = chat.get_reply(channel, msg, persyn_config.id.name, persyn_config.id.guid)
            my_response = toot(the_reply)

        for goal in goals_achieved:
            log.info(f"🏆 _achievement unlocked: {goal}_")

        chat.summarize_later(channel, reminders)

        if the_reply.endswith('…') or the_reply.endswith('...'):
            say_something_later(
                channel,
                when=1,
                status=my_response[-1]
            )
            return

        # 5% chance of random interjection later
        if random.random() < 0.05:
            say_something_later(
                channel,
                when=random.randint(2, 5),
                status=my_response[-1]
            )

# -=-=-=-=-=-=-

#     if ctx.attachments:
#         await handle_attachments(ctx)


#     elif ctx.content == 'help':
#         await ctx.channel.send(f"""*Commands:*
#   `...`: Let {persyn_config.id.name} keep talking without interrupting
#   `summary`: Explain it all to me very briefly.
#   `status`: Say exactly what is on {persyn_config.id.name}'s mind.
#   `nouns`: Some things worth thinking about.
#   `reflect`: {persyn_config.id.name}'s opinion of those things.
#   `daydream`: Let {persyn_config.id.name}'s mind wander on the convo.
#   `goals`: See {persyn_config.id.name}'s current goals

#   *Image generation:*
#   :art: _prompt_ : Generate a picture of _prompt_ using stable-diffusion
#   :magic_wand: _prompt_ : Generate a *fancy* picture of _prompt_ using stable-diffusion
#   :selfie: Take a selfie
# """)

#     elif ctx.content == 'status':
#         status = ("\n".join([f"> {line.strip()}" for line in chat.get_status(channel).split("\n")])).rstrip("> \n")
#         if len(status) < 2000:
#             await ctx.channel.send(status.strip())
#         else:
#             # 2000 character limit for messages
#             reply = ""
#             for line in status.split("\n"):
#                 if len(reply) + len(line) < 1999:
#                     reply = reply + line + "\n"
#                 else:
#                     await ctx.channel.send(reply)
#                     reply = line + "\n"
#             if reply:
#                 await ctx.channel.send(reply)

#     elif ctx.content == 'summary':
#         await ctx.channel.send("💭 " + chat.get_summary(channel, save=False, include_keywords=False, photo=True))

#     elif ctx.content == 'summary!':
#         await ctx.channel.send("💭 " + chat.get_summary(channel, save=True, include_keywords=True, photo=False))

#     elif ctx.content == 'nouns':
#         await ctx.channel.send("> " + ", ".join(chat.get_nouns(chat.get_status(channel))))

#     else:
#         reminders.add(channel, 0, schedule_reply, f'reply-{uuid.uuid4()}', args=[ctx])

# async def on_message(ctx):
#     ''' Default message handler. '''
#     channel = get_channel(ctx)

#     # Don't talk to yourself.
#     if it_me(ctx.author.id):
#         return

#     # Interrupt any rejoinder in progress
#     reminders.cancel(channel)

#     if ctx.author.bot:
#         log.warning(f'🤖 BOT DETECTED: {ctx.author.name} ({ctx.author.id})')
#         # 95% chance to just ignore them
#         if random.random() < 0.95:
#             return

#     # Handle commands and schedule a reply (if any)
#     await dispatch(ctx)

if __name__ == "__main__":
    listener = TheListener()

    while True:
        try:
            mastodon.stream_user(listener)
        except MastodonMalformedEventError as mastoerr:
            log.critical(f"MastodonMalformedEventError, continuing.\n{mastoerr}")
