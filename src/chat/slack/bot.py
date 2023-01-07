#!/usr/bin/env python3
"""
slack/bot.py

Chat with your persyn on Slack.
"""
# pylint: disable=import-error, wrong-import-position
import argparse
import os
import random
import re
import tempfile
import uuid

from hashlib import sha256

import requests

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# Reminders
from interaction.reminders import Reminders

# Mastodon support for image posting
from chat.mastodon.bot import Mastodon

# Common chat library
from chat.common import Chat

# Username cache
known_users = {}

# Known bots
known_bots = {}

# Threaded reminders
reminders = Reminders()

# Defined in main()
app = None

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
    ''' Fetch the image caption using CLIP Interrogator '''
    log.warning("🖼  needs a caption")

    resp = requests.get(url, headers={'Authorization': f'Bearer {persyn_config.chat.slack.bot_token}'}, timeout=20)
    if not resp.ok:
        log.error(f"🖼  Could not retrieve image: {resp.text}")
        return None

    return chat.get_caption(resp.content)

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
    persyn_config = load_config(args.config_file)

    # Slack bolt App
    global app
    app = App(token=persyn_config.chat.slack.bot_token)

    # Mastodon support
    mastodon = Mastodon(args.config_file)
    mastodon.login()

    # Chat library
    chat = Chat(persyn_config, service=app.client.auth_test().data['url'])

    log.info(f"👖 Logged into chat.service: {chat.service}")

    # Ugh, you can't instantiate App until you have the token, which requires
    # the config to be loaded. So Slack events follow. -_-
    ###
    @app.message(re.compile(r"^help$", re.I))
    def help_me(say, context): # pylint: disable=unused-argument
        ''' TODO: These should really be / commands. '''
        say(f"""*Commands:*
    `...`: Let {persyn_config.id.name} keep talking without interrupting
    `summary`: Explain it all to me very briefly.
    `status`: Say exactly what is on {persyn_config.id.name}'s mind.
    `nouns`: Some things worth thinking about.
    `reflect`: {persyn_config.id.name}'s opinion of those things.
    `daydream`: Let {persyn_config.id.name}'s mind wander on the convo.
    `goals`: See {persyn_config.id.name}'s current goals

    *Image generation:*
    :art: _prompt_ : Generate a picture of _prompt_ using stable-diffusion v2
    :frame_with_picture: _prompt_ : Generate a *high quality* picture of _prompt_ using stable-diffusion v2
    :magic_wand: _prompt_ : Generate a *fancy* picture of _prompt_ using stable-diffusion v2
    """)

    @app.message(re.compile(r"^goals$"))
    def goals(say, context): # pylint: disable=unused-argument
        ''' What are we doing again? '''
        channel = context['channel_id']

        current_goals = chat.get_goals(channel)
        if current_goals:
            for goal in current_goals:
                say(f":goal_net: {goal}")
        else:
            say(":shrug:")

    @app.message(re.compile(r"^:art:$"))
    def photo_stable_diffusion_summary(say, context): # pylint: disable=unused-argument
        ''' Take a stable diffusion photo of this conversation '''
        channel = context['channel_id']
        chat.take_a_photo(
            channel,
            chat.get_summary(channel, max_tokens=30),
            engine="stable-diffusion",
            width=768,
            height=768,
            guidance=15
        )

    @app.message(re.compile(r"^:art:(.+)$"))
    def stable_diffusion_picture(say, context): # pylint: disable=unused-argument
        ''' Take a picture with stable diffusion '''
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        channel = context['channel_id']
        prompt = context['matches'][0].strip()

        chat.take_a_photo(channel, prompt, engine="stable-diffusion")
        say(f"OK, {speaker_name}.")
        say_something_later(say, channel, context, 3, ":camera_with_flash:")
        ents = chat.get_entities(prompt)
        if ents:
            chat.inject_idea(channel, ents)

    @app.message(re.compile(r"^:frame_with_picture:(.+)$"))
    def stable_diffusion_picture_hq(say, context): # pylint: disable=unused-argument
        ''' Take a picture with stable diffusion '''
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        channel = context['channel_id']
        prompt = context['matches'][0].strip()

        chat.take_a_photo(channel, prompt, engine="stable-diffusion", width=768, height=768, guidance=15)
        say(f"OK, {speaker_name}.")
        say_something_later(say, channel, context, 3, ":camera_with_flash:")
        ents = chat.get_entities(prompt)
        if ents:
            chat.inject_idea(channel, ents)

    @app.message(re.compile(r"^:magic_wand:(.+)$"))
    def prompt_parrot_picture(say, context): # pylint: disable=unused-argument
        ''' Take a picture with stable diffusion '''
        speaker_id = context['user_id']
        speaker_name = get_display_name(speaker_id)
        channel = context['channel_id']
        prompt = context['matches'][0].strip()

        parrot = chat.prompt_parrot(prompt)
        log.warning(f"🦜 {parrot}")
        chat.take_a_photo(channel, prompt, engine="stable-diffusion", style=parrot)
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
        say("💭 " + chat.get_summary(channel, save, include_keywords=False, photo=True))

    @app.message(re.compile(r"^status$", re.I))
    def status(say, context):
        ''' Say a condensed summary of this channel '''
        channel = context['channel_id']
        say("\n".join([f"> {line}" for line in chat.get_status(channel).split("\n")]))

    @app.message(re.compile(r"^nouns$", re.I))
    def nouns(say, context):
        ''' Say the nouns mentioned on this channel '''
        channel = context['channel_id']
        say("> " + ", ".join(chat.get_nouns(chat.get_status(channel))))

    @app.message(re.compile(r"^entities$", re.I))
    def entities(say, context):
        ''' Say the entities mentioned on this channel '''
        channel = context['channel_id']
        say("> " + ", ".join(chat.get_entities(chat.get_status(channel))))

    @app.message(re.compile(r"^daydream$", re.I))
    def daydream(say, context):
        ''' Let your mind wander '''
        channel = context['channel_id']
        say(f"_{persyn_config.id.name}'s mind starts to wander..._")

        ideas = chat.get_daydream(channel)

        for idea in random.sample(list(ideas), k=min(len(ideas), 5)):
            # skip eg. "4 months ago"
            if 'ago' in str(idea):
                continue

            chat.inject_idea(channel, ideas[idea])
            say(f"💭 *{idea}*: _{ideas[idea]}_")

        the_nouns = chat.get_nouns(chat.get_status(channel))
        for noun in random.sample(the_nouns, k=min(3, len(the_nouns))):
            opinion = chat.get_opinions(channel, noun.lower(), condense=True)
            if not opinion:
                opinion = [chat.judge(channel, noun.lower())]
            if opinion:
                chat.inject_idea(channel, opinion[0])
                say(f"🤔 *{noun}*: _{opinion[0]}_")

        say(f"_{persyn_config.id.name} blinks and looks around._")
        chat.summarize_later(channel, reminders, when=1)

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

        for noun in chat.get_nouns(chat.get_status(channel)):
            opinion = chat.get_opinions(channel, noun.lower(), condense=True)
            if not opinion:
                opinion = [chat.judge(channel, noun.lower())]

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
    def catch_all(say, context):
        ''' Default message handler. Prompt GPT and randomly arm a Timer for later reply. '''
        channel = context['channel_id']

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

        (the_reply, goals_achieved) = chat.get_reply(channel, msg, speaker_name, speaker_id)

        say(the_reply)

        # Interrupt any rejoinder in progress
        reminders.cancel(channel)
        reminders.cancel(channel, name='summarizer')

        for goal in goals_achieved:
            say(f"🏆 _achievement unlocked: {goal}_")

        chat.summarize_later(channel, reminders)

        if the_reply.endswith('…') or the_reply.endswith('...'):
            say_something_later(
                say,
                channel,
                context,
                when=1
            )
            return

        # 5% chance of random interjection later
        if random.random() < 0.05:
            say_something_later(
                say,
                channel,
                context,
                when=random.randint(2, 5)
            )


    @app.event("app_mention")
    def handle_app_mention_events(body, client, say): # pylint: disable=unused-argument
        ''' Reply to @mentions '''
        channel = body['event']['channel']
        speaker_id = body['event']['user']
        speaker_name = get_display_name(speaker_id)
        msg = substitute_names(body['event']['text'])

        reply, goals_achieved = chat.get_reply(channel, msg, speaker_name, speaker_id)

        say(reply)

        for goal in goals_achieved:
            say(f"🏆 _achievement unlocked: {goal}_")

    @app.event("reaction_added")
    def handle_reaction_added_events(body, logger): # pylint: disable=unused-argument
        '''
        Handle reactions: post images to Mastodon.
        '''
        if mastodon.client is None:
            log.error("🎺 Mastodon is not configured, check your config.")
            return

        channel = body['event']['item']['channel']
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
                    if msg['reactions'][0]['name'] in ['-1', 'hankey', 'no_entry', 'no_entry_sign', 'hand']:
                        if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                            log.warning("🎺 Not posting:", {msg['reactions'][0]['name']})
                            return
                        log.warning("🤯 All is forgotten.")
                        chat.forget_it(channel)
                        return
                    try:
                        req = { "service": chat.service, "channel": channel }
                        response = requests.post(f"{persyn_config.interact.url}/amnesia/", params=req, timeout=10)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as err:
                        log.critical(f"🤖 Could not post /amnesia/ to interact: {err}")
                        return

                    if 'blocks' in msg and 'image_url' in msg['blocks'][0]:
                        blk = msg['blocks'][0]
                        try:
                            if len(blk['alt_text']) > 497:
                                toot = blk['alt_text'][:497] + '...'
                            else:
                                toot = blk['alt_text']
                            with tempfile.TemporaryDirectory() as tmpdir:
                                media_ids = []
                                for blk in msg['blocks']:
                                    response = requests.get(blk['image_url'], timeout=30)
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
                                    idempotency_key=sha256(blk['image_url'].encode()).hexdigest()
                                )
                                if not resp or 'url' not in resp:
                                    raise RuntimeError(resp)
                                log.info(f"🎺 Posted {blk['image_url']}: {resp['url']}")

                        except RuntimeError as err:
                            log.error(f"🎺 Could not post {blk['image_url']}: {err}")
                    else:
                        log.error(f"🎺 Unhandled reaction {msg['reactions'][0]['name']} to: {msg['text']}")

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

        speaker_id = body['event']['user']
        speaker_name = get_display_name(speaker_id)
        msg = substitute_names(body['event']['text'])

        if 'files' not in body['event']:
            log.warning("Message with no picture? 🤷‍♂️")
            return

        for file in body['event']['files']:
            caption = get_caption(file['url_private_download'])

            if caption:
                prefix = random.choice(["I see", "It looks like", "Looks like", "Might be", "I think it's"])
                say(f"{prefix} {caption}")

                chat.inject_idea(channel, f"{speaker_name} posted a photo of {caption}")

                if not msg.strip():
                    msg = "..."

                reply, goals_achieved = chat.get_reply(channel, msg, speaker_name, speaker_id)

                say(reply)

                for goal in goals_achieved:
                    say(f"🏆 _achievement unlocked: {goal}_")
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
