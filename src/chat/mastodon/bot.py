#!/usr/bin/env python3
"""
mastodon/bot.py

Chat with your persyn on Mastodon.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import argparse
import os
import random
import tempfile
import uuid

from pathlib import Path
from hashlib import sha256

from mastodon import Mastodon as MastoNative, MastodonError, MastodonMalformedEventError, StreamListener
from bs4 import BeautifulSoup

import requests
import spacy

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# Reminders
from interaction.reminders import Reminders

# Common chat library
from chat.common import Chat

class Mastodon():
    ''' Wrapper for handling Mastodon calls using a Persyn config '''

    def __init__(self, config_file):
        ''' Save state but don't log in yet '''
        self.config_file = config_file
        self.cfg = None
        self.client = None
        self.chat = None

        # Coroutine reminders
        self.reminders = Reminders()

        # Spacy for basic parsing
        if self.valid_config():
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.add_pipe('sentencizer')

    def valid_config(self):
        ''' Returns True if a masto config is present '''
        if self.cfg is None:
            self.cfg = load_config(self.config_file)

        return (
            hasattr(self.cfg, 'chat') and
            hasattr(self.cfg.chat, 'mastodon') and
            hasattr(self.cfg.chat.mastodon, 'instance') and
            hasattr(self.cfg.chat.mastodon, 'secret')
        )

    def login(self):
        ''' Attempt to log into the Mastodon service '''
        if not self.valid_config():
            log.info(f"No Mastodon configuration for {self.cfg.id.name}, skipping.")
            return False

        masto_secret = Path(self.cfg.chat.mastodon.secret)
        if not masto_secret.is_file():
            raise RuntimeError(
                f"Mastodon instance specified but secret file '{masto_secret}' does not exist.\nCheck your config."
            )
        try:
            self.client = MastoNative(
                access_token = masto_secret,
                api_base_url = self.cfg.chat.mastodon.instance
            )
        except MastodonError:
            raise SystemExit("Invalid credentials, run masto-login.py and try again.") from MastodonError

        creds = self.client.account_verify_credentials()
        log.info(
            f"Logged into Mastodon as @{creds.username}@{self.cfg.chat.mastodon.instance} ({creds.display_name})"
        )

        self.chat = Chat(self.cfg, service='mastodon')

        return True

    def is_logged_in(self):
        ''' Helper to see if we're logged in '''
        return self.client is not None

    def say_something_later(self, channel, when=1, what=None, status=None):
        ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
        self.reminders.cancel(channel)

        if what:
            self.reminders.add(channel, when, self.toot, args=[what, status])
        else:
            # Yadda yadda yadda
            self.reminders.add(channel, when, self.dispatch, args=[channel, '...', status])

    def synthesize_image(self, channel, prompt, engine="stable-diffusion", style=None, model=None):
        ''' It's not AI art. It's _image synthesis_ '''
        self.chat.take_a_photo(
            channel,
            prompt,
            engine=engine,
            style=style,
            model=model,
            width=768,
            height=768,
            guidance=15
        )
        ents = self.chat.get_entities(prompt)
        if ents:
            self.chat.inject_idea(channel, ents)

    def following(self, account_id):
        ''' Return true if we are following this account '''
        return account_id in [follower.id for follower in self.client.account_following(id=self.client.me().id)]

    def get_text(self, msg):
        ''' Extract just the text from a message (no HTML or @username) '''
        return BeautifulSoup(msg, features="html.parser").text.strip().replace(f'@{self.client.me().username} ','')

    def fetch_and_post_image(self, url, msg):
        ''' Download the image at URL and post it to Mastodon '''
        if not self.client:
            log.error(f"🚫 Mastodon not configured, cannot post image: {url}, {msg}")
            return

        media_ids = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                fname = f"{tmpdir}/{uuid.uuid4()}.{url[-3:]}"
                with open(fname, "wb") as f:
                    for chunk in response.iter_content():
                        f.write(chunk)
                caption = self.chat.get_caption(url)
                media_ids.append(self.client.media_post(fname, description=caption).id)

                resp = self.client.status_post(
                    msg,
                    media_ids=media_ids,
                    idempotency_key=sha256(url.encode()).hexdigest()
                )
                if not resp or 'url' not in resp:
                    raise RuntimeError(resp)
                log.info(f"🎺 Posted {url}: {resp['url']}")

        except RuntimeError as err:
            log.error(f"🎺 Could not post {url}: {err}")

    def paginate(self, text, maxlen=None):
        ''' Break a single status string into a list of toots < the posting limit. '''
        if maxlen is None or maxlen > self.cfg.chat.mastodon.toot_length:
            maxlen = self.cfg.chat.mastodon.toot_length

        doc = self.nlp(text)
        posts = []
        post = ""
        trimmed = max(50, maxlen - 30)
        for sent in [str(sent) for sent in doc.sents]:
            if len(sent) > trimmed:
                # edge case, just truncate it
                sent = sent[:trimmed] + '…'
            # save some margin for error
            if len(post) + len(sent) < trimmed:
                post = f"{post} {sent}"
            else:
                posts.append(post)
                post = f"{sent}"

        if post:
            posts.append(post)

        return posts

    def toot(self, status, to_status=None, **kwargs):
        '''
        Quick send a toot or reply.

        If status is longer than the max toot length, post a thread.

        Returns a list of all status messages posted.
        '''
        rets = []
        for post in self.paginate(status):
            if to_status:
                rets.append(self.client.status_reply(to_status, post, **kwargs))
                log.info("🎺 Posted reply:", rets[-1].url)
            else:
                rets.append(self.client.status_post(post, **kwargs))
                log.info("🎺 Posted:", rets[-1].url)
                to_status = rets[-1]

        return rets

    def dispatch(self, channel, msg, status=None):
        ''' Handle commands and replies '''

        if msg.startswith('🎨'):
            self.synthesize_image(channel, msg[1:].strip(), engine="stable-diffusion")

        elif msg.startswith('🦜'):
            prompt = msg[1:].strip()
            style = self.chat.prompt_parrot(prompt)
            log.warning(f"🦜 {style}")
            self.synthesize_image(channel, prompt, engine="stable-diffusion", style=style)

        else:
            if status:
                (the_reply, goals_achieved) = self.chat.get_reply(
                    channel,
                    msg,
                    status.account.username,
                    status.account.id
                )
                my_response = self.toot(
                    the_reply,
                    to_status=status
                )
            else:
                (the_reply, goals_achieved) = self.chat.get_reply(channel, msg, self.cfg.id.name, self.cfg.id.guid)
                my_response = self.toot(the_reply)

            for goal in goals_achieved:
                log.info(f"🏆 _achievement unlocked: {goal}_")

            self.chat.summarize_later(channel, self.reminders)

            if the_reply.endswith('…') or the_reply.endswith('...'):
                self.say_something_later(
                    channel,
                    when=1,
                    status=my_response[-1]
                )
                return

            # 5% chance of random interjection later
            if random.random() < 0.05:
                self.say_something_later(
                    channel,
                    when=random.randint(2, 5),
                    status=my_response[-1]
                )

class TheListener(StreamListener):
    ''' Handle streaming events from Mastodon.py '''

    def __init__(self, masto):
        self.masto = masto
        self.channel = self.masto.client.me().url

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
            log.info("⭐️ by", notification.account.acct)
            return

        if notification.status.account.id == self.masto.client.me().id:
            return

        if not self.masto.following(notification.status.account.id):
            log.warning("📪 Not following, so ignoring:", notification.status.account.acct)
            return

        msg = self.masto.get_text(notification.status.content)
        log.info(f"📬 {notification.status.account.acct}:", msg)

        self.masto.dispatch(self.channel, msg, notification.status)

    def handle_heartbeat(self):
        log.debug("💓")


# -=-=-=-=-=-=-

#     if ctx.attachments:
#         await handle_attachments(ctx)


#     elif ctx.content == 'help':
#         await ctx.channel.send(f"""*Commands:*
#   `...`: Let {self.cfg.id.name} keep talking without interrupting
#   `summary`: Explain it all to me very briefly.
#   `status`: Say exactly what is on {self.cfg.id.name}'s mind.
#   `nouns`: Some things worth thinking about.
#   `reflect`: {self.cfg.id.name}'s opinion of those things.
#   `daydream`: Let {self.cfg.id.name}'s mind wander on the convo.
#   `goals`: See {self.cfg.id.name}'s current goals

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

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Mastodon chat module for Persyn'''
    )
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    # parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)


    args = parser.parse_args()

    mastodon = Mastodon(args.config_file)

    if not mastodon.valid_config():
        raise SystemExit('mastodon not defined in config, exiting.')

    if not mastodon.login():
        raise SystemExit("Invalid credentials, run masto-login.py and try again.")

    log.info(f"🎺 Logged in as: {mastodon.client.me().url}")

    listener = TheListener(mastodon)

    try:
        while True:
            try:
                mastodon.client.stream_user(listener)
            except MastodonMalformedEventError as mastoerr:
                log.critical(f"MastodonMalformedEventError, continuing.\n{mastoerr}")
            # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        print()
        raise SystemExit(0) from kbderr

if __name__ == "__main__":
    main()
