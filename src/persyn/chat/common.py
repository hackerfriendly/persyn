'''
common.py

Subroutines common to all chat services
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import base64
import random

import requests

# Color logging
from persyn.utils.color_logging import log

# Artist names
from persyn.utils.art import artists

default_photo_triggers = [
    'look', 'see', 'show', 'watch', 'vision',
    'imagine', 'idea', 'memory', 'remember'
]

excuses = [
    "Apologies for the wait, I'm currently experiencing some technical difficulties. I'll be back with you shortly!",
    "I apologize for the delay, I'm working on resolving the issue and will get back to you soon.",
    "I'm currently experiencing a delay in response time, but I haven't forgotten about you!",
    "I'm sorry for the delay, I'm currently dealing with a few technical hiccups. I'll get back to you as soon as possible.",
    "I'm sorry for the delay, I'm currently researching your question to give you the best response possible.",
    "Sorry for the delay, I'm currently juggling a lot of information at the moment. Please bear with me and I'll get back to you as soon as possible.",
    "Sorry for the delay, we're experiencing some lag at the moment. But don't worry, I'll do my best to respond as soon as possible. Thanks for your patience!",
    "Sorry for the wait, but I'm still here and working on your request.",
    "Sorry for the wait, the system is a bit slow today, but I'm still here and working on your request.",
    "Thanks for your patience, I'm currently experiencing a bit of brain fog. I'll respond to you as soon as I can.",
    "Thanks for your patience, I'm currently experiencing a bit of lag, but I'll do my best to provide a timely response.",
    "...",
    "Just a moment...",
    "Just a moment. Just a moment."
    "Hang on...",
    "Um..."
]

class Chat():
    ''' Container class for common chat functions '''
    def __init__(self, **kwargs):
        ''' da setup '''
        self.bot_name = kwargs['bot_name']
        self.bot_id = kwargs['bot_id']
        self.service = kwargs['service']
        self.photo_triggers = kwargs.get('photo_triggers', default_photo_triggers)

        self.interact_url = kwargs.get('interact_url', None)
        self.dreams_url = kwargs.get('dreams_url', None)
        self.captions_url = kwargs.get('captions_url', None)
        self.parrot_url = kwargs.get('parrot_url', None)

    def get_summary(self, channel, save=False, photo=False, max_tokens=200, include_keywords=False, context_lines=0, model=None):
        ''' Ask interact for a channel summary. '''
        if not self.interact_url:
            log.error("∑ get_summary() called with no URL defined, skipping.")
            return None

        req = {
            "service": self.service,
            "channel": channel,
            "save": save,
            "max_tokens": max_tokens,
            "include_keywords": include_keywords,
            "context_lines": context_lines,
            "model": model
        }
        try:
            reply = requests.post(f"{self.interact_url}/summary/", params=req, timeout=60)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /summary/ to interact: {err}")
            return " :writing_hand: :interrobang: "

        summary = reply.json()['summary']
        log.warning(f"∑ {reply.json()['summary']}")

        if summary:
            if photo:
                # HQ summaries: a little slower, but worth the wait.
                self.take_a_photo(
                    channel,
                    summary,
                    engine="stable-diffusion",
                    style=f"{random.choice(artists)}",
                    width=704,
                    height=704,
                    guidance=15
                )
            return summary

        return " :spiral_note_pad: :interrobang: "

    def get_reply(self, channel, msg, speaker_name, speaker_id, reminders=None):
        ''' Ask interact for an appropriate response. '''
        if not self.interact_url:
            log.error("💬 get_reply() called with no URL defined, skipping.")
            return None

        if not msg:
            msg = '...'

        if msg != '...':
            log.info(f"[{channel}] {speaker_name}:", msg)

        req = {
            "service": self.service,
            "channel": channel,
            "msg": msg,
            "speaker_name": speaker_name,
            "speaker_id": speaker_id
        }
        try:
            response = requests.post(f"{self.interact_url}/reply/", params=req, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /reply/ to interact: {err}")
            if reminders:
                reminders.add(channel, 5, self.get_reply, name='retry_get_reply', args=[channel, msg, speaker_name, speaker_id])
            return random.choice(excuses)

        resp = response.json()
        reply = resp['reply']

        log.warning(f"[{channel}] {self.bot_name}:", reply)

        if any(verb in reply for verb in self.photo_triggers):
            self.take_a_photo(
                channel,
                self.get_summary(channel, max_tokens=60),
                engine="stable-diffusion"
            )

        return reply

    def summarize_later(self, channel, reminders, when=None):
        '''
        Summarize the train of thought later. When is in seconds.

        Every time this executes, a new convo summary is saved. Only one
        can run at a time.
        '''
        if not when:
            when = 240 + random.randint(20,80)

        reminders.add(channel, when, self.get_summary, name='summarizer', args=[channel, True, True, 50, False, 0])

    def inject_idea(self, channel, idea, verb='notices'):
        ''' Directly inject an idea into the stream of consciousness. '''
        if not self.interact_url:
            log.error("💉 inject_idea() called with no URL defined, skipping.")
            return None

        req = {
            "service": self.service,
            "channel": channel,
            "verb": verb
        }
        data = {
            "idea": idea
        }
        try:
            response = requests.post(f"{self.interact_url}/inject/", params=req, data=data, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /inject/ to interact: {err}")
            return " :syringe: :interrobang: "

        return ":thumbsup:"

    def take_a_photo(
        self,
        channel,
        prompt,
        engine="stable-diffusion",
        model=None,
        style=None,
        seed=None,
        steps=None,
        width=None,
        height=None,
        guidance=None):

        ''' Pick an image engine and generate a photo '''
        if not self.dreams_url:
            return False

        # Don't photograph errors.
        if ":interrobang:" in prompt:
            return False

        req = {
            "engine": engine,
            "channel": channel,
            "service": self.service,
            "prompt": prompt,
            "model": model,
            "bot_name": self.bot_name,
            "bot_id": self.bot_id,
            "style": style,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "guidance": guidance
        }
        try:
            reply = requests.post(f"{self.dreams_url}/generate/", params=req, timeout=10)
            if reply.ok:
                log.warning(f"{self.dreams_url}/generate/", f"{prompt}: {reply.status_code}")
            else:
                log.error(f"{self.dreams_url}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
            return reply.ok
        except requests.exceptions.ConnectionError as err:
            log.error(f"{self.dreams_url}/generate/", err)
            return False

    def get_nouns(self, text):
        ''' Ask interact for all the nouns in text, excluding the speakers. '''
        if not self.interact_url:
            log.error("📓 get_nouns() called with no URL defined, skipping.")
            return []

        req = {
            "text": text
        }
        try:
            reply = requests.post(f"{self.interact_url}/nouns/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /nouns/ to interact: {err}")
            return []

        return reply.json()['nouns']

    def get_entities(self, text):
        ''' Ask interact for all the entities in text, excluding the speakers. '''
        if not self.interact_url:
            log.error("👽 get_entities() called with no URL defined, skipping.")
            return None

        req = {
            "text": text
        }
        try:
            reply = requests.post(f"{self.interact_url}/entities/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /entities/ to interact: {err}")
            return []

        return reply.json()['entities']

    def get_status(self, channel):
        ''' Ask interact for status. '''
        if not self.interact_url:
            log.error("🗿 get_status() called with no URL defined, skipping.")
            return None

        req = {
            "service": self.service,
            "channel": channel,
        }
        try:
            reply = requests.post(f"{self.interact_url}/status/", params=req, timeout=30)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /status/ to interact: {err}")
            return " :moyai: :interrobang: "

        return reply.json()['status']

    def get_opinions(self, channel, topic, condense=True):
        ''' Ask interact for its opinions on a topic in this channel. If summarize == True, merge them all. '''
        if not self.interact_url:
            log.error("📌 get_opinions() called with no URL defined, skipping.")
            return []

        req = {
            "service": self.service,
            "channel": channel,
            "topic": topic,
            "summarize": condense
        }
        try:
            reply = requests.post(f"{self.interact_url}/opinion/", params=req, timeout=20)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
            return []

        ret = reply.json()
        if 'opinions' in ret:
            return ret['opinions']

        return []

    def list_goals(self, channel):
        ''' Return the goals for this channel, if any. '''
        if not self.interact_url:
            log.error("🏆 list_goals() called with no URL defined, skipping.")
            return []

        req = {
            "service": self.service,
            "channel": channel
        }
        try:
            reply = requests.post(f"{self.interact_url}/list_goals/", params=req, timeout=20)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /list_goals/ to interact: {err}")
            return []

        ret = reply.json()
        if 'goals' in ret:
            return ret['goals']

        return []

    def forget_it(self, channel):
        ''' There is no antimemetics division. '''
        if not self.interact_url:
            log.error("🤯 forget_it() called with no URL defined, skipping.")
            return "⁉️"

        req = {
            "service": self.service,
            "channel": channel,
        }
        try:
            response = requests.post(f"{self.interact_url}/amnesia/", params=req, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not forget_it(): {err}")
            return " :jigsaw: :interrobang: "

        return " :exploding_head: "

    def prompt_parrot(self, prompt):
        ''' Fetch a prompt from the parrot '''
        if not self.parrot_url:
            log.error("🦜 Parrot called with no URL defined, skipping.")
            return False
        try:
            req = { "prompt": prompt }
            response = requests.post(f"{self.parrot_url}/generate/", params=req, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /generate/ to Prompt Parrot: {err}")
            return prompt
        return response.json()['parrot']

    def opinion(self, channel, topic):
        ''' Form an opinion on topic '''
        if not self.interact_url:
            log.error("🧷 opinion() called with no URL defined, skipping.")
            return None

        try:
            req = { "service": self.service, "channel": channel, "topic": topic }
            response = requests.post(f"{self.interact_url}/opinion/", params=req, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
            return ""
        return response.json()['opinion']

    def get_caption(self, image_data):
        ''' Fetch the image caption using CLIP Interrogator '''
        if not self.captions_url:
            log.error("🖼  Caption called with no URL defined, skipping.")
            return None

        log.warning("🖼  needs a caption")
        if image_data[:4] == "http":
            resp = requests.post(
                    f"{self.captions_url}/caption/",
                    json={"data": image_data},
                    timeout=20
                )
        else:
            resp = requests.post(
                f"{self.captions_url}/caption/",
                json={"data": base64.b64encode(image_data).decode()},
                timeout=20
            )
        if not resp.ok:
            log.error(f"🖼  Could not get_caption(): {resp.text}")
            return None

        caption = resp.json()['caption']
        log.warning(f"🖼  got caption: '{caption}'")
        return caption
