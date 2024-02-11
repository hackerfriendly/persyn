#!/usr/bin/env python3
'''
cns.py

The central nervous system. Listen for events on the event bus and inject results into interact.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member, unused-wildcard-import
import argparse
import asyncio
import logging
import os
import datetime as dt
from time import sleep

from typing import Optional, Union

import requests

from bs4 import BeautifulSoup

from scheduler import Scheduler

from persyn.langchain.zim import ZimWrapper

# Autobus, forked from https://github.com/schuyler/autobus
from persyn import autobus

# Common chat library
from persyn.chat.common import Chat
from persyn.chat.simple import slack_msg, discord_msg, mastodon_msg

# Mastodon support for image posting
from persyn.chat.mastodon.bot import Mastodon

# Time
from persyn.interaction.chrono import get_cur_ts, elapsed

# Long and short term memory
from persyn.interaction.memory import Recall

# Goals
from persyn.interaction.goals import Goal

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Message classes
from persyn.interaction.messages import *  # pylint: disable=wildcard-import

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import PersynConfig, load_config

cns = None
schedule = Scheduler(n_threads=0)

class CNS:
    ''' Container class for the Central Nervous System '''

    def __init__(self, persyn_config: PersynConfig) -> None:
        self.config = persyn_config
        self.mastodon = Mastodon(persyn_config)
        self.recall = Recall(persyn_config)
        self.goal = Goal(self.config)
        self.lm = LanguageModel(persyn_config)
        self.datasources = {}
        self.rs = requests.Session()

        if self.config.get('zim'):
            for cfgtool in self.config.zim: # type: ignore
                log.info("💿 Loading Zim:", cfgtool)
                self.datasources[cfgtool] = ZimWrapper(path=self.config.zim.get(cfgtool).path) # type: ignore

        self.mastodon.login()

    def send_chat(self, service: str, channel: str, msg: str, images: Optional[list[str]] = None, extra: Optional[str] = None) -> None:
        ''' Send a chat message to a service + channel '''

        if 'slack.com' in service:
            func = slack_msg
        elif service == 'discord':
            func = discord_msg
        elif service == 'mastodon':
            func = mastodon_msg
        else:
            log.critical(f"Unknown service: {service}")
            return

        chat = Chat(persyn_config=self.config, service=service)
        try:
            func(self.config, chat, channel, msg, images, extra)
        except Exception as err:
            log.error(f"💬 Could not send chat to {service}|{channel}: {err}")

    def say_something(self, event: SendChat) -> None:
        ''' Send a message to a service + channel '''
        log.debug(f'SendChat received: {event.service} {event.channel} {event.msg} {event.images} {event.extra}')
        self.send_chat(service=event.service, channel=event.channel, msg=event.msg, images=event.images, extra=event.extra)

    def chat_received(self, event: ChatReceived) -> None:
        ''' Somebody is talking to us '''

        # Receiving new chat resets the elaborations counter
        convo_id = self.recall.get_last_convo_id(event.service, event.channel)
        if convo_id:
            self.recall.set_convo_meta(convo_id, "elaborations", '0')

        chat = Chat(persyn_config=self.config, service=event.service)

        start = get_cur_ts()
        log.warning("💬 chat_received")

        # TODO: Decide whether to delay reply, or to reply at all?

        chat.get_reply(
            channel=event.channel,
            msg=event.msg,
            speaker_name=event.speaker_name,
            send_chat=True,
            extra=event.extra
        )

        # Time for self-examination.

        # TODO: schedule jobs instead of publishing more messages

        # Update emotional state
        vc = VibeCheck(
            service=event.service,
            channel=event.channel,
        )
        autobus.publish(vc)

        # Do some research
        wp = Wikipedia(
            service=event.service,
            channel=event.channel,
            focus=event.msg
        )
        autobus.publish(wp)

        log.warning("💬 chat_received done in:", f"{elapsed(start, get_cur_ts()):0.2f} sec")

    def new_idea(self, event: Idea) -> None:
        ''' Inject a new idea '''
        chat = Chat(persyn_config=self.config, service=event.service)
        chat.inject_idea(
            channel=event.channel,
            idea=event.idea,
            verb=event.verb
        )

    def cns_summarize_channel(self, event: Summarize) -> str:
        ''' Summarize the channel '''
        chat = Chat(persyn_config=self.config, service=event.service)

        summary = chat.get_summary(
            channel=event.channel,
            convo_id=event.convo_id,
            photo=event.photo,
            extra=None,
            final=event.final
        )

        if event.send_chat:
            self.send_chat(service=event.service, channel=event.channel, msg=summary)

        return summary

    def elaborate(self, event: Elaborate) -> None:
        '''
        Continue the train of thought up to 5 times, checking Claude each time to see if we should continue.
        If no convo_id is available, do nothing.
        If context is present, ask Claude whether we should continue (given the context) before elaborating.
        Otherwise, elaborate immediately.
        '''
        log.warning(f"elaborate(): {event.service} {event.channel} {event.context[:100]}…")

        if event.convo_id:
            convo_id = event.convo_id
        else:
            convo_id = self.recall.get_last_convo_id(event.service, event.channel)

        if convo_id is None:
            log.warning("🤷‍♀️ No convo, nothing to elaborate.")
            return None

        # This should never happen, but is here as a safety valve in case Claude gets loquatious.
        if self.recall.incr_convo_meta(convo_id, "elaborations") > 5:
            log.warning("✋ Too many elaborations, stopping.")
            return None

        if event.context:
            reply = self.recall.lm.ask_claude(
                query=event.context,
                prefix=f"In the following dialog, does {self.config.id.name} have anything else to add? You must answer ONLY yes or no, and nothing else. If you are not sure, make your best guess:",
            )
            log.warning(reply)
            if 'yes' not in reply.lower():
                log.warning("🤔 Claude says there is no need to elaborate.")
                return None

        log.warning("🤔 Elaborating...")
        chat = Chat(persyn_config=self.config, service=event.service)
        chat.get_reply(
            channel=event.channel,
            msg='...',
            speaker_name=self.config.id.name
        )
        return None

    def vibe_check(self, event: VibeCheck) -> None:
        ''' Run sentiment analysis on ourselves. '''
        convo_id = self.recall.get_last_convo_id(event.service, event.channel)
        if convo_id is None:
            log.warning("😑 No convo, nothing to feel.")
            return
        summary = self.recall.fetch_summary(convo_id)
        if summary:
            feels = self.lm.ask_claude(
                summary,
                prefix=f"""In the following text, choose three words that best describe how {self.config.id.name} feels. You MUST include ONLY three comma separated words with no other preface or commentary. If you are not sure, make your best guess based on the information provided. The three words are: """ # type: ignore
            )
        else:
            feels = "nothing in particular"
        self.recall.set_convo_meta(convo_id, "feels", feels)
        log.warning("😄 Feeling:", feels)


    def check_wikipedia(self, event: Wikipedia) -> None:
        ''' Extract concepts from the text and ask Claude for further reading.'''

        convo = self.recall.fetch_convo(event.service, event.channel)
        dialog = cns.recall.fetch_dialog(event.service, event.channel, convo_id=convo.id)
        last_lines = '\n'.join(dialog.split('\n')[-10:])

        log.info("🌍 Extract concepts from text:", last_lines)

        # Fetch and filter concepts
        concepts = self.recall.lm.extract_entities(last_lines)
        concepts = concepts.difference({self.config.id.name, self.recall.fetch_convo_meta(convo.id, 'initiator')})

        if concepts:
            log.warning(f"🌍 Extracted concepts: {concepts}")
        else:
            log.warning("🌍 No concepts extracted. What are we even talking about?")
            return

        if event.focus:
            focus = f", paying close attention to '{event.focus}'"
        else:
            focus = ''

        reply = self.recall.lm.ask_claude(
            prefix=f"In the following dialog:\n{last_lines}\nWhich Wikipedia pages would be most useful to learn about these concepts? You must reply ONLY with a comma-separated list of the three most important pages that {self.config.id.name} should read, and nothing else.",
            query=str(concepts)
        )
        keywords = self.recall.lm.cleanup_keywords(reply)
        log.warning("🌍 Claude suggests further reading:", str(keywords))

        if not keywords:
            log.warning("🌍 No keywords, nothing to do.")
            return

        summaries = []
        for kw in keywords:
            page = self.datasources['Wikipedia'].run(kw)
            if page:
                summaries.append(
                    self.recall.lm.summarize_text(
                        text=str(page),
                        summarizer=f"Summarize this Wikipedia page{focus}:\n",
                        final=True
                    )
                )

        if not summaries:
            log.warning("🌍 No summaries, nothing to do.")
            return

        log.warning("🌍 Injecting summaries:", str(summaries))
        chat = Chat(persyn_config=self.config, service=event.service)
        for summary in summaries:
            chat.inject_idea(
                channel=event.channel,
                idea=summary,
                verb='recalls'
            )

    def text_from_url(self, url: str, selector: Optional[str] = 'body') -> str:
        ''' Return just the text from url. You probably want a better selector than <body>. '''
        try:
            article = self.rs.get(url, timeout=30)
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.error(f"🗞️ Could not fetch article {url}: {err}")
            return ''

        soup = BeautifulSoup(article.text, features="lxml")
        story = []
        for line in soup.select_one(selector).text.split('\n'): # type: ignore
            if not line:
                continue
            story.append(line)

        return '\n'.join(story)

    def read_web(self, event: Web) -> None:
        ''' Read a web page '''
        log.info("🕸️ Reading web:", event.url)
        sleep(5)
        log.info("🕸️ Still reading web:", event.url)
        sleep(5)
        log.info("🕸️ Done reading web:", event.url)
        return None

        # if self.config.web.get(urlparse(event.url).netloc, None):
        #     cfg = self.config.web.get(urlparse(event.url).netloc) # type: ignore
        #     selector = cfg.get('selector', 'body')
        #     stop = cfg.get('stop', [])
        # else:
        #     selector = 'body'
        #     stop = []

        # chat = Chat(persyn_config=self.config, service=event.service)
        # log.debug(self.text_from_url(event.url, selector))

        # if not event.reread and self.recall.have_read(event.service, event.channel, event.url):
        #     log.info("🕸️ Already read:", event.url)
        #     chat.inject_idea(
        #         channel=event.channel,
        #         idea=f"and doesn't need to re-read it: {event.url}",
        #         verb="already read this article"
        #     )
        #     reply = chat.get_reply(
        #         channel=event.channel,
        #         msg='...',
        #         speaker_name=event.bot_name
        #     )
        #     self.services[self.get_service(event.service)](self.config, chat, event.channel, event.bot_name, reply)
        #     return

        # self.recall.add_news(event.service, event.channel, event.url, "web page")

        # body = self.text_from_url(event.url, selector)

        # if not body:
        #     log.error("🗞️ Got empty body from", event.url)
        #     return

        # log.info("📰 Reading", event.url)

        # prompt = "To briefly summarize this article:"
        # max_reply_length = 300
        # done = False
        # for chunk in self.completion.paginate(body, prompt=prompt, max_reply_length=max_reply_length):
        #     for stop_word in stop:
        #         if stop_word in chunk:
        #             log.warning("📰 Stopping here:", stop_word)
        #             chunk = chunk[:chunk.find(stop_word)]
        #             done = True

        #     summary = self.completion.get_summary(
        #         text=chunk,
        #         summarizer=prompt
        #     )

        #     chat.inject_idea(
        #         channel=event.channel,
        #         idea=f"{summary} {event.url}",
        #         verb="saw on the web"
        #     )
        #     self.services[self.get_service(event.service)](self.config, chat, event.channel, event.bot_name, f"{summary} {event.url}")

        #     if done:
        #         return

    def read_news(self, event: News) -> None:
        ''' Check our RSS feed. Read the first unread article. '''
        log.info("🗞️  Reading news feed:", event.url)
        sleep(5)
        log.info("🗞️  Still reading news feed:", event.url)
        sleep(5)
        log.info("🗞️  Done reading news feed:", event.url)
        return None
        # log.info("🗞️  Reading news feed:", event.url)
        # try:
        #     page = rs.get(event.url, timeout=30)
        # except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
        #     log.error(f"🗞️  Could not fetch RSS feed {event.url}", err)
        #     return

        # feed = BeautifulSoup(page.text, "xml")
        # for item in feed.find_all('item'):
        #     item_url = item.find('link').text
        #     if self.recall.have_read(event.service, event.channel, item_url):
        #         log.info("🗞️  Already read:", item_url)
        #         continue

        #     item_event = Web(
        #         service=event.service,
        #         channel=event.channel,
        #         bot_name=event.bot_name,
        #         bot_id=event.bot_id,
        #         url=item_url,
        #         reread=False
        #     )
        #     await read_web(item_event)
        #     # only one at a time
        #     return

    def reflect_on(self, event: Reflect) -> None:
        ''' Reflect on recent events. Inspired by Stanford's Smallville, https://arxiv.org/abs/2304.03442 '''
        convo_id = event.convo_id or cns.recall.get_last_convo_id(event.service, event.channel)
        dialog = cns.recall.fetch_dialog(event.service, event.channel, convo_id=convo_id)

        qa = cns.lm.reflect(dialog)
        if qa is None:
            log.warning("🪩  No reply from lm, nothing to reflect.")
            return

        for question, answers in qa.items():
            log.warning(f"🪩  {question}", str(answers))
            closest = self.goal.find_related_goals(
                event.service,
                event.channel,
                text=question,
                threshold=self.config.memory.relevance / 2, # be strict
                size=1
            )
            if closest:
                log.warning("🥅  Closest goal:", closest[0].content)
                goal_id = closest[0].goal_id
            else:
                log.warning("🥅  New goal:", question)
                goal_id = self.goal.add(event.service, event.channel, question)

            for answer in answers:
                log.warning("⚽️  Adding goal action:", answer)
                self.goal.add_action(goal_id, answer)

        log.warning("🪩  Done reflecting.")

    def generate_photo(self, event: Photo) -> None:
        ''' Generate a photo '''
        chat = Chat(persyn_config=self.config, service=event.service)

        chat.take_a_photo(
            channel=event.channel,
            prompt=event.prompt,
            width=event.size[0],
            height=event.size[1]
        )

    def plan_your_day(self) -> None:
        ''' Make a schedule of actions for the next part of the day '''
        log.info("📅 Time to make a schedule")
        # TODO: use LangChain to decide on the actions to take for the next interval, and inject as an idea.

    def run(self):
        ''' Main event loop '''
        log.info(f"⚡️ {self.config.id.name}'s CNS is online") # type: ignore

        schedule.cyclic(dt.timedelta(hours=4), self.plan_your_day)

        try:
            autobus.run(url=self.config.cns.redis, namespace=self.config.id.guid) # type: ignore
        except KeyboardInterrupt as kbderr:
            print()
            raise SystemExit(0) from kbderr


# Autobus subscriptions. These must be top-level functions.

@autobus.subscribe(SendChat)
async def sendchat_event(event):
    ''' Dispatch SendChat event w/ optional images. '''
    log.debug("SendChat received", event)

    schedule.once(dt.timedelta(seconds=0), cns.say_something, kwargs={'event':event})

    # Possibly elaborate
    event = Elaborate(
        service=event.service,
        channel=event.channel,
        context=cns.recall.fetch_dialog(event.service, event.channel)
    )
    schedule.once(dt.timedelta(seconds=1), cns.elaborate, kwargs={'event':event})

@autobus.subscribe(ChatReceived)
async def chatreceived_event(event):
    ''' Dispatch ChatReceived event '''
    log.debug("ChatReceived received", event)
    schedule.once(dt.timedelta(seconds=0), cns.chat_received, kwargs={'event':event})

@autobus.subscribe(Idea)
async def idea_event(event):
    ''' Dispatch idea event. '''
    log.debug("Idea received", event)
    schedule.once(dt.timedelta(seconds=0), cns.new_idea, kwargs={'event':event})

@autobus.subscribe(Summarize)
async def summarize_event(event):
    ''' Dispatch summarize event. '''
    log.debug("Summarize received", event)
    schedule.once(dt.timedelta(seconds=0), cns.cns_summarize_channel, kwargs={'event':event})

@autobus.subscribe(Elaborate)
async def elaborate_event(event):
    ''' Dispatch elaborate event. '''
    log.debug("Elaborate received", event)
    schedule.once(dt.timedelta(seconds=3), cns.elaborate, kwargs={'event':event})

# @autobus.subscribe(CheckGoals)
# async def check_goals_event(event):
#     ''' Dispatch CheckGoals event. '''
#     log.debug("CheckGoals received", event)
#     await cns.goals_achieved(event) # type: ignore

# @autobus.subscribe(AddGoal)
# async def goals_event(event):
#     ''' Dispatch AddGoal event. '''
#     log.debug("AddGoal received", event)
#     await cns.add_goal(event) # type: ignore

@autobus.subscribe(VibeCheck)
async def feels_event(event):
    ''' Dispatch VibeCheck event. '''
    log.debug("VibeCheck received", event)
    schedule.once(dt.timedelta(seconds=3), cns.vibe_check, kwargs={'event':event})

@autobus.subscribe(News)
async def news_event(event):
    ''' Dispatch News event. '''
    log.debug("News received", event)
    schedule.once(dt.timedelta(seconds=0), cns.read_news, kwargs={'event':event})

@autobus.subscribe(Web)
async def web_event(event):
    ''' Dispatch Web event. '''
    log.debug("Web received", event)
    schedule.once(dt.timedelta(seconds=0), cns.read_web, kwargs={'event':event})

@autobus.subscribe(Reflect)
async def reflect_event(event):
    ''' Dispatch Reflect event. '''
    log.debug("Reflect received", event)
    schedule.once(dt.timedelta(seconds=1), cns.reflect_on, kwargs={'event':event})

@autobus.subscribe(Photo)
async def photo_event(event):
    ''' Dispatch Photo event. '''
    log.debug("Photo received", event)
    schedule.once(dt.timedelta(seconds=2), cns.generate_photo, kwargs={'event':event})

@autobus.subscribe(Wikipedia)
async def wikipedia_event(event):
    ''' Dispatch Wikipedia event. '''
    log.debug("Wikipedia", event)
    schedule.once(dt.timedelta(seconds=1), cns.check_wikipedia, kwargs={'event':event})

# Autobus scheduled events. These must also be top-level functions.

@autobus.schedule(autobus.every(5).seconds)
async def auto_summarize() -> None:
    ''' Automatically summarize conversations when they expire. '''
    convos = cns.recall.list_convo_ids(expired=False) # type: ignore
    for convo_id, meta in convos.items():
        if cns.recall.convo_expired(convo_id=convo_id): # type: ignore
            log.debug(f"{convo_id} expired.")

        remaining = cns.config.memory.conversation_interval - elapsed(cns.recall.id_to_timestamp(cns.recall.get_last_message_id(convo_id)), get_cur_ts()) # type: ignore
        if remaining >= 5:
            log.info(f"💓 Active convo: {convo_id} (expires in {int(remaining)} seconds)")

    expired_convos = cns.recall.list_convo_ids(expired=True, after=4) # type: ignore
    for convo_id, meta in expired_convos.items():
        log.info(f"💔 Convo expired: {convo_id}")
        if len(cns.recall.fetch_summary(convo_id)) > 10:
            log.info("🪩  Reflecting:", convo_id)
            event = Reflect(
                service=meta['service'],
                channel=meta['channel'],
                convo_id=convo_id,
                send_chat=True
            )
            autobus.publish(event)

        event = Summarize(
            service=meta['service'],
            channel=meta['channel'],
            convo_id=convo_id,
            photo=False,
            send_chat=False,
            final=True
        )
        autobus.publish(event)

@autobus.schedule(autobus.every(1).seconds)
async def run_schedule() -> None:
    ''' Run the schedule '''
    # log.warning("Running jobs", str(schedule))
    schedule.exec_jobs()

def main() -> None:
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Persyn central nervous system. Run one server for each bot.'''
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

    if not hasattr(persyn_config, 'cns'):
        raise SystemExit('cns not defined in config, exiting.')

    # enable logging to disk
    if hasattr(persyn_config.id, "logdir"):
        logging.getLogger().addHandler(logging.FileHandler(f"{persyn_config.id.logdir}/{persyn_config.id.name}-cns.log")) # type: ignore

    mastodon = Mastodon(persyn_config)
    mastodon.login()

    global cns
    cns = CNS(persyn_config)

    cns.run()

if __name__ == '__main__':
    main()
