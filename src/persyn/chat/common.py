'''
common.py

Subroutines common to all chat services
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import random
from typing import Optional, Union

import requests

from openai import OpenAI

# Color logging
from persyn.utils.color_logging import log

# Long and short term memory
from persyn.interaction.memory import Recall
from persyn.utils.config import PersynConfig

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

rs = requests.Session()

class Chat():
    ''' Container class for common chat functions '''
    def __init__(self, persyn_config: PersynConfig, service: str):
        ''' Container class for common chat functions. Pass the persyn config and the calling chat service. '''
        self.persyn_config = persyn_config
        self.service=service

        self.bot_name=persyn_config.id.name # type: ignore
        self.bot_id=persyn_config.id.guid # type: ignore
        self.interact_url=persyn_config.interact.url # type: ignore
        self.dreams_url=persyn_config.dreams.url # type: ignore

        self.photo_triggers = default_photo_triggers
        self.recall = Recall(persyn_config)

        self.oai_client = OpenAI(
            api_key=persyn_config.completion.openai_api_key, # type: ignore
            organization=persyn_config.completion.openai_org # type: ignore
        )

    def get_summary(
        self,
        channel: str,
        convo_id: Optional[str] = None,
        photo: Optional[bool] = False,
        extra: Optional[str] = None,
        final: Optional[bool] = False
        ) -> str:
        ''' Ask interact for a channel summary. '''
        if not self.interact_url:
            log.error("∑ get_summary() called with no interact_url defined, skipping.")
            return ""

        req = {
            "service": self.service,
            "channel": channel,
            "convo_id": convo_id,
            "final": final
        }
        try:
            reply = rs.post(f"{self.interact_url}/summary/", params=req, timeout=60)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /summary/ to interact: {err}")
            return " :writing_hand: :interrobang: "

        summary = reply.json()['summary']
        log.warning(f"∑: » {reply.json()['summary']} «")

        if summary:
            if self.dreams_url and photo:
                self.take_a_photo(
                    channel,
                    summary,
                    engine="dall-e",
                    width=self.persyn_config.dreams.dalle.width, # type: ignore
                    height=self.persyn_config.dreams.dalle.height, # type: ignore
                    style=self.persyn_config.dreams.dalle.quality, # type: ignore
                    extra=extra
                )
            return summary

        return " :spiral_note_pad: :interrobang: "

    def get_reply(
        self,
        channel: str,
        msg: str,
        speaker_name: str,
        reminders=None,
        send_chat: Optional[bool] = True,
        extra: Optional[str] = None
        ) -> str:
        ''' Ask interact for an appropriate response. '''
        if not self.interact_url:
            log.error("💬 get_reply() called with no interact_url defined, skipping.")
            return ""

        if not msg:
            msg = '...'

        if msg != '...':
            log.info(f"[{channel}] {speaker_name}:", msg)

        req = {
            "service": self.service,
            "channel": channel,
            "msg": msg,
            "speaker_name": speaker_name,
            "send_chat": send_chat,
            "extra": extra
        }
        try:
            response = rs.post(f"{self.interact_url}/reply/", params=req, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /reply/ to interact: {err}")
            if reminders:
                reminders.add(channel, 5, self.get_reply, name='retry_get_reply', args=[channel, msg, speaker_name])
            return random.choice(excuses)

        resp = response.json()
        reply = resp['reply']

        log.warning(f"[{channel}] {self.bot_name}:", reply)

        if self.dreams_url and reply and any(verb in reply for verb in self.photo_triggers):
            self.take_a_photo(
                channel,
                self.get_summary(channel),
                engine="dall-e",
                width=self.persyn_config.dreams.dalle.width, # type: ignore
                height=self.persyn_config.dreams.dalle.height, # type: ignore
                style=self.persyn_config.dreams.dalle.quality, # type: ignore
                extra=extra
            )

        return reply

    def summarize_later(self, channel, reminders, when=None):
        '''
        Summarize the train of thought later. When is in seconds.

        Every time this executes, a new convo summary is saved. Only one
        can run at a time.

        This is typically handled automatically by cns.py.
        '''
        if not when:
            when = 240 + random.randint(20,80)

        reminders.add(channel, when, self.get_summary, name='summarizer', args=[channel, True, True, 50, False, 0])

    def inject_idea(self, channel, idea, verb='notices'):
        ''' Directly inject an idea into the stream of consciousness. '''
        if not self.interact_url:
            log.error("💉 inject_idea() called with no interact_url defined, skipping.")
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
            response = rs.post(f"{self.interact_url}/inject/", params=req, data=data, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /inject/ to interact: {err}")
            return " :syringe: :interrobang: "

        return ":thumbsup:"

    def take_a_photo(
        self,
        channel,
        prompt,
        engine="dall-e",
        model="dall-e-3",
        style="standard",
        seed=None,
        steps=None,
        width=None,
        height=None,
        guidance=None,
        extra=None):

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
            "guidance": guidance,
            "extra": extra
        }
        try:
            reply = rs.post(f"{self.dreams_url}/generate/", params=req, timeout=10)
            if reply.ok:
                log.warning(f"{self.dreams_url}/generate/", f"{prompt}: {reply.status_code}")
            else:
                log.error(f"{self.dreams_url}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
            return reply.ok
        except requests.exceptions.ConnectionError as err:
            log.error(f"{self.dreams_url}/generate/: {err}")
            return False

    def get_nouns(self, text):
        ''' Ask interact for all the nouns in text, excluding the speakers. '''
        if not self.interact_url:
            log.error("📓 get_nouns() called with no interact_url defined, skipping.")
            return []

        req = {
            "text": text
        }
        try:
            reply = rs.post(f"{self.interact_url}/nouns/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /nouns/ to interact: {err}")
            return []

        return reply.json()['nouns']

    def get_entities(self, text):
        ''' Ask interact for all the entities in text, excluding the speakers. '''
        if not self.interact_url:
            log.error("👽 get_entities() called with no interact_url defined, skipping.")
            return None

        req = {
            "text": text
        }
        try:
            reply = rs.post(f"{self.interact_url}/entities/", params=req, timeout=10)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /entities/ to interact: {err}")
            return []

        return reply.json()['entities']

    def get_status(self, channel, speaker_name):
        ''' Ask interact for status. '''
        if not self.interact_url:
            log.error("🗿 get_status() called with no interact_url defined, skipping.")
            return None

        req = {
            "service": self.service,
            "channel": channel,
            "speaker_name": speaker_name
        }
        try:
            reply = rs.post(f"{self.interact_url}/status/", params=req, timeout=30)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /status/ to interact: {err}")
            return " :moyai: :interrobang: "

        return reply.json()['status']

    def get_opinions(self, channel, topic, condense=True):
        ''' Ask interact for its opinions on a topic in this channel. If summarize == True, merge them all. '''
        if not self.interact_url:
            log.error("📌 get_opinions() called with no interact_url defined, skipping.")
            return []

        req = {
            "service": self.service,
            "channel": channel,
            "topic": topic,
            "summarize": condense
        }
        try:
            reply = rs.post(f"{self.interact_url}/opinion/", params=req, timeout=20)
            reply.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
            return []

        ret = reply.json()
        if 'opinions' in ret:
            return ret['opinions']

        return []

    def opinion(self, channel, topic):
        ''' Form an opinion on topic '''
        if not self.interact_url:
            log.error("🧷 opinion() called with no interact_url defined, skipping.")
            return None

        try:
            req = { "service": self.service, "channel": channel, "topic": topic }
            response = rs.post(f"{self.interact_url}/opinion/", params=req, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            log.critical(f"🤖 Could not post /opinion/ to interact: {err}")
            return ""
        return response.json()['opinion']

    def get_caption(self, image_url):
        ''' Fetch the image caption using OpenAI gpt-4-vision (aka CLIP) '''
        log.warning("🖼  needs a caption")

        response = self.oai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=200,
        )

        caption = self.recall.lm.trim(response.choices[0].message.content)
        log.warning(f"🖼️  {caption}")
        return caption

    def chat_received(self, channel, msg, speaker_name, extra=None):
        ''' Dispatch a ChatReceived event. Extra is optional JSON for service options (Discord DM, Masto visibility, etc.) '''
        if not self.interact_url:
            log.error("💬 chat_received() called with no interact_url defined, skipping.")
            return None

        req = {
            "service": self.service,
            "channel": channel,
            "speaker_name": speaker_name,
            "msg": msg,
            "extra": extra
        }
        try:
            reply = rs.post(f"{self.interact_url}/chat_received/", params=req, timeout=60)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /chat_received/ to interact: {err}")
            return " 💬 :interrobang: "

        return True
