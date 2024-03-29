#!/usr/bin/env python3
"""
slack/bot.py

Chat with your persyn on Slack.
"""
# pylint: disable=import-error, wrong-import-position, no-member, invalid-name
import argparse
import os
import random
import re
import tempfile
import uuid
import logging

from hashlib import sha256
from pathlib import Path

import requests

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

# Reminders
from persyn.interaction.reminders import Reminders

# Mastodon support for image posting
from persyn.chat.mastodon.bot import Mastodon

# Common chat library
from persyn.chat.common import Chat

# Upload files (for image caption processing)
from persyn.dreams.dreams import upload_files

# Username cache
known_users = {}

# Known bots
known_bots = {}

# Threaded reminders
reminders = Reminders()

rs = requests.Session()

# Defined in main()
app = None
persyn_config = None
mastodon = None
chat = None

###
# Slack helper functions
###
def is_bot(user_id):
    """ Returns true if the user_id is a Slack bot """
    get_display_name(user_id)
    return known_bots[user_id]

def get_display_name(user_id):
    """ Return the user's first name if available, otherwise the display name """
    if user_id not in known_users:
        users_info = app.client.users_info(user=user_id)['user']
        try:
            profile = users_info['profile']
            known_users[user_id] = profile.get('first_name') or profile.get('display_name') or profile.get('real_name')
        except KeyError:
            known_users[user_id] = user_id

    if user_id not in known_bots:
        known_bots[user_id] = users_info['is_bot']

    return known_users[user_id]

def substitute_names(text):
    """ Substitute all <@XYZ> in text with the equivalent display name. """
    for user_id in set(re.findall(r'<@(\w+)>', text)):
        text = re.sub(f'<@{user_id}>', get_display_name(user_id), text)
    return text

def get_caption(url):
    ''' Fetch the image caption using OpenAI CLIP '''
    return chat.get_caption(url)

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Slack chat module for Persyn'''
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
    global persyn_config
    persyn_config = load_config(args.config_file)

    # enable logging to disk
    if hasattr(persyn_config.id, "logdir"):
        logging.getLogger().addHandler(logging.FileHandler(f"{persyn_config.id.logdir}/{persyn_config.id.name}-slack.log"))

    # Slack bolt App
    global app
    app = App(token=persyn_config.chat.slack.bot_token)

    # Mastodon support
    global mastodon
    mastodon = Mastodon(persyn_config)
    mastodon.login()

    # Chat library
    global chat
    chat = Chat(persyn_config=persyn_config, service=app.client.auth_test().data['url'])

    log.info(f"👖 Logged into chat.service: {chat.service}")

    # Ugh, you can't instantiate App until you have the token, which requires
    # the config to be loaded. So Slack events follow. -_-
    ###
    def say_something_later(say, channel, context, when, what=None):
        ''' Continue the train of thought later. When is in seconds. If what, just say it. '''

        reminders.cancel(channel)

        if what:
            reminders.add(channel, when, say, args=what)
        else:
            # yadda yadda yadda
            yadda = {
                'channel_id': channel,
                'user_id': context['user_id'],
                'matches': ['...']
            }
            reminders.add(channel, when, catch_all, args=[say, yadda])

    @app.message(re.compile(r"^help$", re.I))
    def help_me(say, context): # pylint: disable=unused-argument
        ''' TODO: These should really be / commands. '''
        say(f"""*Commands:*
    `...`: Let {persyn_config.id.name} keep talking without interrupting
    `summary`: Explain it all to me very briefly.
    `status`: Say exactly what is on {persyn_config.id.name}'s mind.
    `nouns`: Some things worth thinking about.
    `reflect`: {persyn_config.id.name}'s opinion of those things.
    `goals`: See {persyn_config.id.name}'s current goals

    *Image generation:*
    :art: _prompt_ : Generate a picture of _prompt_ using dall-e v3
    """)

    @app.message(re.compile(r"^goals$"))
    def goals(say, context): # pylint: disable=unused-argument
        ''' What are we doing again? '''
        service = app.client.auth_test().data['url']
        channel = context['channel_id']

        current_goals = chat.recall.list_goals(service, channel)
        if current_goals:
            for goal in current_goals:
                say(f":goal_net: {goal}")
        else:
            say("No goals. :shrug:")

    @app.message(re.compile(r"^:art:$"))
    def photo_stable_diffusion_summary(say, context): # pylint: disable=unused-argument
        ''' Take a stable diffusion photo of this conversation '''
        if not persyn_config.dreams.stable_diffusion:
            log.error('🎨 No Stable Diffusion support available, check your config.')
            return
        channel = context['channel_id']
        chat.take_a_photo(
            channel,
            chat.get_summary(channel),
            engine="dall-e",
            width=persyn_config.dreams.stable_diffusion.width,
            height=persyn_config.dreams.stable_diffusion.height,
            style=persyn_config.dreams.stable_diffusion.quality
        )

    @app.message(re.compile(r"^:art:(.+)$"))
    def dalle_picture(say, context): # pylint: disable=unused-argument
        ''' Take a picture with Dall-E '''
        if not persyn_config.dreams.dalle:
            log.error('🎨 No DALL-E support available, check your config.')
            return
        speaker_name = get_display_name(context['user_id'])
        channel = context['channel_id']
        prompt = context['matches'][0].strip()

        chat.take_a_photo(
            channel,
            prompt,
            engine="dall-e",
            width=persyn_config.dreams.dalle.width,
            height=persyn_config.dreams.dalle.height,
            style=persyn_config.dreams.dalle.quality
        )
        say(f"OK, {speaker_name}.")
        say_something_later(say, channel, context, 3, ":camera_with_flash:")
        ents = chat.get_entities(prompt)
        if ents:
            chat.inject_idea(channel, ents)

    @app.message(re.compile(r"^:frame_with_picture:(.+)$"))
    def dalle_portrait(say, context): # pylint: disable=unused-argument
        ''' Take a picture with Dall-E '''
        speaker_name = get_display_name(context['user_id'])
        channel = context['channel_id']
        prompt = context['matches'][0].strip()

        chat.take_a_photo(
            channel,
            prompt,
            engine="dall-e",
            width=1024,
            height=1792,
        )
        say(f"OK, {speaker_name}.")
        say_something_later(say, channel, context, 3, ":camera_with_flash:")
        ents = chat.get_entities(prompt)
        if ents:
            chat.inject_idea(channel, ents)

    @app.message(re.compile(r"^summary(\!)?$", re.I))
    def summarize(say, context):
        ''' Say a condensed summary of this channel '''
        save = bool(context['matches'][0])
        channel = context['channel_id']
        say("💭 " + chat.get_summary(
            channel=channel,
            convo_id=None,
            photo=True)
        )

    @app.message(re.compile(r"^(status|:question:)$", re.I))
    def status(say, context):
        ''' Say a condensed summary of this channel '''
        channel = context['channel_id']
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        say("\n".join([f"> {line}" for line in chat.get_status(channel, speaker_name).split("\n")]))

    @app.message(re.compile(r"^nouns$", re.I))
    def nouns(say, context):
        ''' Say the nouns mentioned on this channel '''
        channel = context['channel_id']
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        say("> " + ", ".join(chat.get_nouns(chat.get_status(channel, speaker_name))))

    @app.message(re.compile(r"^entities$", re.I))
    def entities(say, context):
        ''' Say the entities mentioned on this channel '''
        channel = context['channel_id']
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        say("> " + ", ".join(chat.get_entities(chat.get_status(channel, speaker_name))))

    @app.message(re.compile(r"^opinions (.*)$", re.I))
    def opine_all(say, context):
        ''' Fetch our opinion on a topic '''
        topic = context['matches'][0]
        channel = context['channel_id']

        opinions = chat.get_opinions(channel, topic, condense=False)
        if opinions:
            say('\n'.join(opinions))
        else:
            say('I have no opinion on that topic.')

    @app.message(re.compile(r"^opinion (.*)$", re.I))
    def opine(say, context):
        ''' Fetch our opinion on a topic '''
        topic = context['matches'][0]
        channel = context['channel_id']

        opinion = chat.get_opinions(channel, topic, condense=True)
        if opinion:
            say(opinion[0])
        else:
            say('I have no opinion on that topic.')

    @app.message(re.compile(r"^reflect$", re.I))
    def reflect(say, context):
        ''' Fetch our opinion on all the nouns in the channel '''
        channel = context['channel_id']
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)

        for noun in chat.get_nouns(chat.get_status(channel, speaker_name)):
            opinion = chat.get_opinions(channel, noun.lower(), condense=True)
            if not opinion:
                opinion = [chat.opinion(channel, noun.lower())]

            say(f"{noun}: {opinion[0]}")

    @app.message(re.compile(r"^:bulb:$"))
    def lights(say, context): # pylint: disable=unused-argument
        ''' Are the lights on? '''
        them = get_display_name(context['user_id'])
        channel = context['channel_id']

        if os.path.isfile('/home/rob/.u16-lights'):
            say(f'The lights are on, {them}.')
            chat.inject_idea(channel, "The lights are on.")
        else:
            say(f'The lights are off, {them}.')
            chat.inject_idea(channel, "The lights are off.")

    @app.message(re.compile(r"(.*)", re.I))
    def catch_all(say, context): # pylint: disable=unused-argument
        ''' Default message handler '''
        service = app.client.auth_test().data['url']
        channel = context['channel_id']

        log.debug(context)

        # Interrupt any rejoinder in progress
        reminders.cancel(channel)

        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        msg = substitute_names(' '.join(context['matches'])).strip()

        if is_bot(speaker_id):
            log.warning(f'🤖 BOT DETECTED ({speaker_name})')
            # 95% chance to just ignore them
            if random.random() < 0.95:
                return

        log.info(f"{speaker_name}: {msg}")

        # Dispatch a "message received" event. Replies are handled by CNS.
        chat.chat_received(channel, msg, speaker_name)

        # Interrupt any rejoinder in progress
        reminders.cancel(channel)
        reminders.cancel(channel, name='summarizer')

    @app.event("app_mention")
    def handle_app_mention_events(body, client, say): # pylint: disable=unused-argument
        ''' Reply to @mentions '''
        channel = body['event']['channel']
        speaker_name = get_display_name(body['event']['user'])
        msg = substitute_names(body['event']['text'])

        chat.get_reply(channel, msg, speaker_name, send_chat=True)

    @app.event("reaction_added")
    def handle_reaction_added_events(body, logger): # pylint: disable=unused-argument
        '''
        Handle reactions: post images to Mastodon.
        '''
        if mastodon.client is None:
            log.error("🎺 Mastodon is not configured, check your config.")
            return

        channel = body['event']['item']['channel']
        speaker_name = get_display_name(body['event']['user'])

        try:
            result = app.client.conversations_history(
                channel=channel,
                inclusive=True,
                oldest=body['event']['item']['ts'],
                limit=1
            )

            for msg in result.get('messages', []):
                # only post on the first reaction
                if 'reactions' in msg and len(msg['reactions']) == 1:
                    if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                        blk = msg['blocks'][0]
                        try:
                            tags = "\n#imagesynthesis #persyn"
                            maxlen = 497 - len(tags)
                            if len(blk['alt_text']) > maxlen:
                                toot = blk['alt_text'][:maxlen] + '...' + tags
                            else:
                                toot = blk['alt_text'] + tags
                            with tempfile.TemporaryDirectory() as tmpdir:
                                media_ids = []
                                for blk in msg['blocks']:
                                    response = rs.get(blk['image_url'], timeout=30)
                                    response.raise_for_status()
                                    fname = f"{tmpdir}/{uuid.uuid4()}.{blk['image_url'][-3:]}"
                                    with open(fname, "wb") as f:
                                        for chunk in response.iter_content():
                                            f.write(chunk)
                                    caption = get_caption(blk['image_url'])
                                    media_ids.append(mastodon.client.media_post(fname, description=caption).id)

                                resp = mastodon.client.status_post(
                                    toot,
                                    media_ids=media_ids,
                                    visibility='unlisted',
                                    idempotency_key=sha256(blk['image_url'].encode()).hexdigest()
                                )
                                if not resp or 'url' not in resp:
                                    raise RuntimeError(resp)
                                log.info(f"🎺 Posted {blk['image_url']}: {resp['url']}")

                        except RuntimeError as err:
                            log.error(f"🎺 Could not post {blk['image_url']}: {err}")
                    else:
                        log.info(f"{speaker_name} reacted {msg['reactions'][0]['name']} to: {msg['text']}")
                        chat.inject_idea(
                            channel, f"{speaker_name} reacted {msg['reactions'][0]['name']} to: {msg['text']}"
                        )

        except SlackApiError as err:
            log.error(f"Slack error: {err}")

    @app.event("reaction_removed")
    def handle_reaction_removed_events(body, logger): # pylint: disable=unused-argument
        ''' Skip for now '''
        log.info("Reaction removed event")


    @app.event("message")
    def handle_message_events(body, say):
        ''' Handle uploaded images '''
        channel = body['event']['channel']

        if 'user' not in body['event']:
            log.warning("Message event with no user. 🤷")
            return

        speaker_name = get_display_name(body['event']['user'])
        msg = substitute_names(body['event']['text'])

        if 'files' not in body['event']:
            log.warning("Message with no picture? 🤷‍♂️")
            return

        for file in body['event']['files']:

            # Download needs auth, so upload it to a public link
            log.warning(f"Downloaded: {file['url_private_download']}")

            response = rs.get(
                file['url_private_download'],
                headers={'Authorization': f'Bearer {persyn_config.chat.slack.bot_token}'}
            )
            response.raise_for_status()

            with tempfile.TemporaryDirectory() as tmpdir:
                image_id = uuid.uuid4()
                fname = str(Path(tmpdir)/f"{image_id}.jpg")
                with open(fname, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)

                upload_files([fname], persyn_config)

            log.warning(f"Uploaded: {persyn_config.dreams.upload.url_base}/{image_id}.jpg")

            chat.inject_idea(channel, f"{speaker_name} uploads a picture.")
            caption = get_caption(f"{persyn_config.dreams.upload.url_base}/{image_id}.jpg")

            if caption:
                say(caption)

                chat.inject_idea(channel, caption, verb="observes")

                if not msg.strip():
                    msg = "..."

                chat.get_reply(channel, msg, speaker_name, send_chat=True)

            else:
                say(
                    random.choice([
                        "I'm not sure.",
                        ":face_with_monocle:",
                        ":face_with_spiral_eyes:",
                        "What the...?",
                        "Um.",
                        "No idea.",
                        "Beats me."
                    ])
                )


    handler = SocketModeHandler(app, persyn_config.chat.slack.app_token)
    try:
        handler.start()
    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        print()
        raise SystemExit(0) from kbderr

if __name__ == "__main__":
    main()
