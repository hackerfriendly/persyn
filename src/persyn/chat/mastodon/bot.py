#!/usr/bin/env python3
"""
mastodon/bot.py

Chat with your persyn on Mastodon.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import logging
import os
import re
import tempfile
from typing import Union
import uuid
import json
import datetime

from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace

from mastodon import Mastodon as MastoNative, MastodonError, MastodonMalformedEventError, StreamListener
from bs4 import BeautifulSoup

import requests
import spacy

# Color logging
from persyn.utils.color_logging import log

# Reminders
from persyn.interaction.reminders import Reminders

# Common chat library
from persyn.chat.common import Chat
from persyn.utils.config import PersynConfig, load_config

def dtconverter(o) -> Union[str, None]:
    ''' Serialize datetime '''
    if isinstance(o, datetime.datetime):
        return str(o)
    return None

class Mastodon():
    ''' Wrapper for handling Mastodon calls using a Persyn config '''

    def __init__(self, persyn_config: PersynConfig) -> None:
        ''' Save state but don't log in yet '''
        self.config = persyn_config
        self.client = None
        self.chat = None

        # Coroutine reminders
        self.reminders = Reminders()

        # Spacy for basic parsing
        if self.valid_config():
            self.nlp = spacy.load(self.config.spacy.model)
            self.nlp.add_pipe('sentencizer')

    def valid_config(self):
        ''' Returns True if a masto config is present '''
        return (
            hasattr(self.config, 'chat') and
            hasattr(self.config.chat, 'mastodon') and
            hasattr(self.config.chat.mastodon, 'instance') and
            hasattr(self.config.chat.mastodon, 'secret')
        )

    def login(self) -> bool:
        ''' Attempt to log into the Mastodon service '''
        if not self.valid_config():
            log.info(f"No Mastodon configuration for {self.config.id.name}, skipping.")
            return False

        masto_secret = Path(self.config.chat.mastodon.secret)
        if not masto_secret.is_file():
            raise RuntimeError(
                f"Mastodon instance specified but secret file '{masto_secret}' does not exist.\nCheck your config."
            )
        try:
            self.client = MastoNative(
                access_token = masto_secret,
                api_base_url = self.config.chat.mastodon.instance
            )
        except MastodonError:
            raise SystemExit("Invalid credentials, run masto-login.py and try again.") from MastodonError

        creds = self.client.account_verify_credentials()
        log.info(
            f"Logged into Mastodon as @{creds.username}@{self.config.chat.mastodon.instance} ({creds.display_name})"
        )

        self.chat = Chat(persyn_config=self.config, service='mastodon')

        return True

    def is_logged_in(self) -> bool:
        ''' Helper to see if we're logged in '''
        return self.client is not None

    def say_something_later(self, channel, when=1, what=None, status=None, extra=None):
        ''' Continue the train of thought later. When is in seconds. If what, just say it. '''
        self.reminders.cancel(channel)

        if what:
            if extra:
                self.reminders.add(channel, when, self.toot, args=[what, status, json.loads(extra, default=dtconverter)])
            else:
                self.reminders.add(channel, when, self.toot, args=[what, status, extra])
        else:
            # Yadda yadda yadda
            self.reminders.add(channel, when, self.dispatch, args=[channel, '...', status, extra])

    def synthesize_image(self, channel, prompt, engine="dall-e", model=None, width=None, height=None, style=None, extra=None):
        ''' It's not AI art. It's _image synthesis_ '''
        self.chat.take_a_photo(
            channel,
            prompt,
            engine=engine,
            model=model,
            width=width,
            height=height,
            style=style,
            extra=extra
        )
        ents = self.chat.get_entities(prompt)
        if ents:
            self.chat.inject_idea(channel, ents)

    def following(self, account_id) -> bool:
        ''' Return true if we are following this account '''
        return account_id in [follower.id for follower in self.client.account_following(id=self.client.me().id)]

    def get_text(self, msg) -> str:
        ''' Extract just the text from a message (no HTML or @username) '''
        return BeautifulSoup(msg, features="lxml").text.strip().replace(f'@{self.client.me().username} ','')

    def fetch_and_post_image(self, url, msg, extra=None):
        ''' Download the image at URL and post it to Mastodon '''
        if not self.client:
            log.error(f"🚫 Mastodon not configured, cannot post image: {url}, {msg}")
            return

        if extra is None:
            extra = '{}'

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
                    idempotency_key=sha256(url.encode()).hexdigest(),
                    **json.loads(extra, default=dtconverter)
                )
                if not resp or 'url' not in resp:
                    raise RuntimeError(resp)
                log.info(f"🎺 Posted {url}: {resp['url']}")

        except RuntimeError as err:
            log.error(f"🎺 Could not post {url}: {err}")

    def paginate(self, text, maxlen=None) -> list[str]:
        ''' Break a single status string into a list of toots < the posting limit. '''
        if maxlen is None or maxlen > self.config.chat.mastodon.toot_length:
            maxlen = self.config.chat.mastodon.toot_length

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

    def toot(self, msg, to_status=None, kwargs=None) -> dict:
        '''
        Quick send a toot or reply.

        If status is longer than the max toot length, post a thread.

        Returns a list of all status messages posted.
        '''
        try:

            log.warning(f"toot(): {msg} {to_status} {kwargs}")
            if kwargs and 'visibility' not in kwargs:
                kwargs['visibility'] = 'unlisted'

            # FIXME: private chat is broken because of this monstrosity. Figure out how to get to_status here from dispatch().
            if to_status or (kwargs and 'to_status' in kwargs):
                if not to_status:
                    to_status = json.loads(kwargs['to_status'], default=dtconverter)

                kwargs['in_reply_to_id'] = to_status.id
                kwargs['visibility'] = to_status.visibility
                if to_status.visibility == 'direct' and f"@{to_status.account.acct}" not in msg:
                    if to_status.account.acct in msg:
                        msg = re.sub(rf'\b({to_status.account.acct})\b', f'@{to_status.account.acct}', msg, count=1)
                    else:
                        msg = f"@{to_status.account.acct} {msg}"
            else:
                log.warning("toot(): to_status is not available")

            log.error(f"toot(): {msg} {kwargs}")

            rets = []
            for post in self.paginate(msg):
                if to_status:
                    rets.append(self.client.status_reply(to_status, post, **kwargs))
                    log.info("🎺 Posted reply:", rets[-1].url)
                else:
                    rets.append(self.client.status_post(post, **kwargs))
                    log.info("🎺 Posted:", rets[-1].url)
                    to_status = rets[-1]

            log.error(f"toot(): returning {rets}")
            return rets
        except Exception as e:
            log.error(e)
            return {}

    def dispatch(self, channel, msg, to_status=None, extra=None) -> None:
        ''' Handle commands and replies '''
        if extra is None:
            extra = '{}'

        if to_status:
            extra = json.loads(extra, default=dtconverter)
            extra['in_reply_to_id'] = to_status.id
            extra['visibility'] = to_status.visibility
            extra['to_status'] = to_status

            if to_status.visibility == 'direct' and f'@{to_status.account.acct}' not in msg:
                if to_status.account.acct in msg:
                    msg = re.sub(rf'\b({to_status.account.acct})\b', f'@{to_status.account.acct}', msg, count=1)
                else:
                    msg = f"@{to_status.account.acct} {msg}"
            extra = json.dumps(extra, default=dtconverter)
        else:
            log.warning("dispatch(): to_status is not available")

        if msg.startswith('🎨'):
            self.synthesize_image(channel, msg[1:].strip(), engine="dall-e", extra=extra)

        elif msg.startswith('🖼️'):
            self.synthesize_image(channel, msg[1:].strip(), engine="dall-e", width=1024, height=1792, extra=extra)

        # Dispatch a "message received" event. Replies are handled by CNS.
        if to_status:
            self.chat.chat_received(channel, msg, to_status.account.username, extra)
        else:
            self.chat.chat_received(channel, msg, None, extra)

        #     if the_reply.endswith('…') or the_reply.endswith('...'):
        #         self.say_something_later(
        #             channel,
        #             when=1,
        #             status=my_response[-1]
        #         )
        #         return

        #     # 5% chance of random interjection later
        #     if random.random() < 0.05:
        #         self.say_something_later(
        #             channel,
        #             when=random.randint(2, 5),
        #             status=my_response[-1]
        #         )

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
        log.info(f"📬 {notification.status.account.acct} ({notification.status.visibility}):", msg)

        self.masto.dispatch(self.channel, msg, notification.status, json.dumps({"visibility": notification.status.visibility}))

    def handle_heartbeat(self):
        log.debug("💓")

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

    persyn_config = load_config(args.config_file)
    mastodon = Mastodon(persyn_config)

    if not mastodon.valid_config():
        raise SystemExit('mastodon not defined in config, exiting.')

    if not mastodon.login():
        raise SystemExit("Invalid credentials, run masto-login.py and try again.")

    log.info(f"🎺 Logged in as: {mastodon.client.me().url}")

    # enable logging to disk
    if hasattr(persyn_config.id, "logdir"):
        logging.getLogger().addHandler(logging.FileHandler(f"{persyn_config.id.logdir}/{persyn_config.id.name}-mastodon.log"))

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
