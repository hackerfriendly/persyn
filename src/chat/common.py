'''
common.py

Subroutines common to all chat services
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import base64
import random

import requests

# Color logging
from utils.color_logging import log

# Artist names
from utils.art import artists

class Chat():
    ''' Container class for common chat functions '''
    def __init__(self, config, service=None):
        ''' da setup '''
        self.config = config
        self.service = service

        if not hasattr(self.config.chat, 'photo_triggers'):
            self.config.chat.photo_triggers = [
                'look', 'see', 'show', 'watch', 'vision',
                'imagine', 'idea', 'memory', 'remember'
            ]

    def can_dream(self):
        ''' Return True if dreams are configured '''
        return self.config.get("dreams")

    def get_summary(self, channel, save=False, photo=False, max_tokens=200, include_keywords=False, context_lines=0):
        ''' Ask interact for a channel summary. '''
        req = {
            "service": self.service,
            "channel": channel,
            "save": save,
            "max_tokens": max_tokens,
            "include_keywords": include_keywords,
            "context_lines": context_lines
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/summary/", params=req, timeout=30)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /summary/ to interact: {err}")
            return " :writing_hand: :interrobang: "

        summary = reply.json()['summary']
        log.warning(f"∑ {reply.json()['summary']}")

        if summary:
            if photo:
                # HQ summaries: slower, but worth the wait.
                self.take_a_photo(channel, summary, engine="stable-diffusion", style=f"{random.choice(artists)}", width=768, height=768, guidance=15)
            return summary

        return " :spiral_note_pad: :interrobang: "

    def get_reply(self, channel, msg, speaker_name, speaker_id):
        ''' Ask interact for an appropriate response. '''
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
            response = requests.post(f"{self.config.interact.url}/reply/", params=req, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /reply/ to interact: {err}")
            return (" :speech_balloon: :interrobang: ", [])

        resp = response.json()
        reply = resp['reply']
        goals_achieved = resp['goals_achieved']

        log.warning(f"[{channel}] {self.config.id.name}:", reply)
        if goals_achieved:
            log.warning(f"[{channel}] {self.config.id.name}:", f"🏆 {goals_achieved}")

        if any(verb in reply for verb in self.config.chat.photo_triggers):
            self.take_a_photo(
                channel,
                self.get_summary(channel, max_tokens=30),
                engine="stable-diffusion"
            )

        return (reply, goals_achieved)

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
        req = {
            "service": self.service,
            "channel": channel,
            "idea": idea,
            "verb": verb
        }
        try:
            response = requests.post(f"{self.config.interact.url}/inject/", params=req, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /inject/ to interact: {err}")
            return " :syringe: :interrobang: "

        return response.json()['status']

    def take_a_photo(self, channel, prompt, engine=None, model=None, style=None, seed=None, steps=None, width=None, height=None, guidance=None):
        ''' Pick an image engine and generate a photo '''
        if not self.can_dream():
            return False

        if not engine:
            engine = random.choice(self.config.dreams.all_engines)

        req = {
            "engine": engine,
            "channel": channel,
            "service": self.service,
            "queue": getattr(self.config.cns, "sqs_queue", None),
            "prompt": prompt,
            "model": model,
            "bot_name": self.config.id.name,
            "style": style,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "guidance": guidance
        }
        reply = requests.post(f"{self.config.dreams.url}/generate/", params=req, timeout=10)
        if reply.ok:
            log.warning(f"{self.config.dreams.url}/generate/", f"{prompt}: {reply.status_code}")
        else:
            log.error(f"{self.config.dreams.url}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
        return reply.ok

    def get_nouns(self, text):
        ''' Ask interact for all the nouns in text, excluding the speakers. '''
        req = {
            "text": text
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/nouns/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /nouns/ to interact: {err}")
            return []

        return reply.json()['nouns']
        # return [e for e in reply.json()['nouns'] if e not in speakers()]

    def get_entities(self, text):
        ''' Ask interact for all the entities in text, excluding the speakers. '''
        req = {
            "text": text
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/entities/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /entities/ to interact: {err}")
            return []

        return reply.json()['entities']
        # return [e for e in reply.json()['entities'] if e not in speakers()]

    def get_daydream(self, channel):
        ''' Ask interact to daydream about this channel. '''
        req = {
            "service": self.service,
            "channel": channel,
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/daydream/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /daydream/ to interact: {err}")
            return []

        return reply.json()['daydream']

    def get_status(self, channel):
        ''' Ask interact for status. '''
        req = {
            "service": self.service,
            "channel": channel,
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/status/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /status/ to interact: {err}")
            return " :moyai: :interrobang: "

        return reply.json()['status']

    def get_opinions(self, channel, topic, condense=True):
        ''' Ask interact for its opinions on a topic in this channel. If summarize == True, merge them all. '''
        req = {
            "service": self.service,
            "channel": channel,
            "topic": topic,
            "summarize": condense
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/opinion/", params=req, timeout=20)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
            return []

        ret = reply.json()
        if 'opinions' in ret:
            return ret['opinions']

        return []
        # return [e for e in reply.json()['nouns'] if e not in speakers()]

    def get_goals(self, channel):
        ''' Return the goals for this channel, if any. '''
        req = {
            "service": self.service,
            "channel": channel
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/get_goals/", params=req, timeout=20)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /get_goals/ to interact: {err}")
            return []

        ret = reply.json()
        if 'goals' in ret:
            return ret['goals']

        return []
        # return [e for e in reply.json()['nouns'] if e not in speakers()]

    def forget_it(self, channel):
        ''' There is no antimemetics division. '''
        req = {
            "service": self.service,
            "channel": channel,
        }
        try:
            response = requests.post(f"{self.config.interact.url}/amnesia/", params=req, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not forget_it(): {err}")
            return " :jigsaw: :interrobang: "

        return " :exploding_head: "

    def prompt_parrot(self, prompt):
        ''' Fetch a prompt from the parrot '''
        if not self.can_dream():
            return False
        try:
            req = { "prompt": prompt }
            response = requests.post(f"{self.config.dreams.parrot.url}/generate/", params=req, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /generate/ to Prompt Parrot: {err}")
            return prompt
        return response.json()['parrot']

    def judge(self, channel, topic):
        ''' Form an opinion on topic '''
        try:
            req = { "service": self.service, "channel": channel, "topic": topic }
            response = requests.post(f"{self.config.interact.url}/judge/", params=req, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /judge/ to interact: {err}")
            return ""
        return response.json()['opinion']

    def get_caption(self, image_data):
        ''' Fetch the image caption using CLIP Interrogator '''
        log.warning("🖼  needs a caption")

        if not self.can_dream():
            return None

        if image_data[:4] == "http":
            resp = requests.post(
                    f"{self.config.dreams.captions.url}/caption/",
                    json={"data": image_data},
                    timeout=20
                )
        else:
            resp = requests.post(
                f"{self.config.dreams.captions.url}/caption/",
                json={"data": base64.b64encode(image_data).decode()},
                timeout=20
            )
        if not resp.ok:
            log.error(f"🖼  Could not get_caption(): {resp.text}")
            return None

        caption = resp.json()['caption']
        log.warning(f"🖼  got caption: '{caption}'")
        return caption
