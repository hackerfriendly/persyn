#!/usr/bin/env python3
'''
cns.py

The central nervous system. Listen for events on the event bus and inject results into interact.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member, unused-wildcard-import
import argparse
import logging
import os
import asyncio

import requests

from spacy.lang.en.stop_words import STOP_WORDS
from urllib.parse import urlparse

# just-in-time Wikipedia
import wikipedia
from wikipedia.exceptions import WikipediaException

from bs4 import BeautifulSoup

from Levenshtein import ratio

# Autobus, forked from https://github.com/schuyler/autobus
from persyn import autobus

# Common chat library
from persyn.chat.common import Chat
from persyn.chat.simple import slack_msg, discord_msg

# Mastodon support for image posting
from persyn.chat.mastodon.bot import Mastodon

# Time
from persyn.interaction.chrono import ago, get_cur_ts, elapsed

# Long and short term memory
from persyn.interaction.memory import Recall

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Message classes
from persyn.interaction.messages import *  # pylint: disable=wildcard-import

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

# Defined in main()
mastodon = None
persyn_config = None
recall = None
completion = None

wikicache = {}

rs = requests.Session()

def mastodon_msg(_, chat, channel, bot_name, msg, images):  # pylint: disable=unused-argument
    ''' Post images to Mastodon '''
    if images:
        for image in images:
            mastodon.fetch_and_post_image(
                f"{persyn_config.dreams.upload.url_base}/{image}", f"{msg}\n#imagesynthesis #persyn"
            )
    else:
        # TODO: This can't respond to a specific thread, need to patch through to_status
        mastodon.toot(msg)

services = {
    'slack': slack_msg,
    'discord': discord_msg,
    'mastodon': mastodon_msg
}

def get_service(service):
    ''' Find the correct service for the dispatcher '''
    if 'slack.com' in service:
        return 'slack'
    if service in services:
        return service

    log.critical(f"Unknown service: {service}")
    return None

async def say_something(event):
    ''' Send a message to a service + channel '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, event.msg, event.images)

async def chat_received(event):
    ''' Somebody is talking to us '''
    chat = Chat(persyn_config=persyn_config, service=event.service)

    start = get_cur_ts()
    log.warning("💬 chat_received")

    # convo_id = recall.convo_id(event.service, event.channel)
    # if convo_id not in recall.list_convos():
    #     log.warning(f"😈 Rogue convo_id {convo_id}, adding to list of active convos")
    #     recall.redis.sadd(recall.active_convos_prefix, f"{event.service}|{event.channel}|{convo_id}")

    # TODO: Give it a few seconds. Ideally, value to be chosen by an interval model for perfect timing.

    # TODO: Decide whether to delay reply, or to reply at all?

    the_reply = chat.get_reply(
        channel=event.channel,
        speaker_name=event.speaker_name,
        msg=event.msg,
        send_chat=True
    )

    # Time for self-examination.

    # Update emotional state
    vc = VibeCheck(
        service=event.service,
        channel=event.channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid
    )
    autobus.publish(vc)

    # if len(recall.convo(event.service, event.channel, convo_id, verb='dialog')) > 5:
    #     # Check facts
    #     fc = FactCheck(
    #         service=event.service,
    #         channel=event.channel,
    #         bot_name=persyn_config.id.name,
    #         bot_id=persyn_config.id.guid,
    #         convo_id=None,
    #         room=None
    #     )
    #     autobus.publish(fc)


    # TODO: Should this be a priority queue?

    # Dispatch an event to gather facts

        # # Facts and opinions
        # self.gather_facts(service, channel, entities)

    #     self.check_goals(service, channel, convo)

    log.warning("💬 chat_received done in:", f"{elapsed(start, get_cur_ts()):0.2f} sec")

async def new_idea(event):
    ''' Inject a new idea '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    chat.inject_idea(
        channel=event.channel,
        idea=event.idea,
        verb=event.verb
    )

async def summarize_channel(event):
    ''' Summarize the channel '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    summary = chat.get_summary(
        channel=event.channel,
        convo_id=event.convo_id,
        save=True,
        photo=event.photo,
        max_tokens=event.max_tokens,
        model=persyn_config.completion.reasoning_model
    )
    if event.send_chat:
        services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, summary)

async def elaborate(event):
    ''' Continue the train of thought '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    chat.get_reply(
        channel=event.channel,
        msg='...',
        speaker_name=event.bot_name
    )

async def opine(event):
    ''' Recall opinions of entities (if any). Form a new opinion if none is found. '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    log.info(f"🙆‍♂️ Opinion time for {len(event.entities)} entities on {event.service} | {event.channel}")
    for entity in event.entities:
        if not entity.strip() or entity in STOP_WORDS:
            continue

        opinions = recall.surmise(event.service, event.channel, entity)
        if opinions:
            log.warning(f"🙋‍♂️ Opinions about {entity}: {len(opinions)}")
            if len(opinions) == 1:
                opinion = opinions[0]
            else:
                opinion = completion.nlp(completion.get_summary(
                    text='\n'.join(opinions),
                    summarizer=f"Briefly state {event.bot_name}'s opinion about {entity} from {event.bot_name}'s point of view, and convert pronouns and verbs to the first person."
                )).text

            chat.inject_idea(
                channel=event.channel,
                idea=opinion,
                verb=f"thinks about {entity}"
            )

        else:
            log.warning(f"💁‍♂️ Forming an opinion about {entity}")
            opinion = completion.get_opinions(recall.convo(event.service, event.channel), entity)
            recall.judge(
                event.service,
                event.channel,
                entity,
                opinion,
                recall.convo_id(event.service, event.channel)
            )
            chat.inject_idea(
                channel=event.channel,
                idea=opinion,
                verb=f"thinks about {entity}"
            )

async def add_goal(event):
    ''' Add a new goal '''
    if not event.goal.strip():
        return

    # Don't repeat yourself
    goals = recall.list_goals(event.service, event.channel) or ['']
    for goal in goals:
        if ratio(goal, event.goal) > 0.6:
            log.warning(f'🏅 We already have a goal like "{event.goal}", skipping.')
            return

    log.info("🥇 New goal:", event.goal)
    recall.add_goal(event.service, event.channel, event.goal)

async def check_feels(event):
    ''' Run sentiment analysis on ourselves. '''
    convo_id = recall.get_last_convo_id(event.service, event.channel)
    if convo_id is None:
        log.warning("😑 No convo, nothing to feel.")
        return
    feels = completion.summarize_text(
        recall.fetch_summary(convo_id),
        summarizer=f"""In the following text, these three words best describe {persyn_config.id.name}'s emotional state. You MUST include only three comma separated words:"""
    )
    recall.set_convo_meta(convo_id, "feels", feels)
    log.warning("😄 Feeling:", feels)

async def check_facts(event):
    ''' Ask for a second opinion about our side of the conversation. '''
    if not event.room:
        event.room = '\n'.join(recall.convo(event.service, event.channel))
    if not event.convo_id:
        event.convo_id = recall.convo_id(event.service, event.channel)

    facts = completion.fact_check(event.room)
    if facts:
        recall.save_convo_line(
            service=event.service,
            channel=event.channel,
            msg=facts,
            speaker_name=event.bot_name,
            convo_id=event.convo_id,
            verb='realizes'
        )
        log.warning("🧠 Thinking:", facts)

async def build_knowledge_graph(event, max_opinions=3):
    ''' Build the knowledge graph. '''
    pass
    # triples = completion.generate_triples(event.convo)
    # log.warning(f'📉 Saving {len(triples)} triples to the knowledge graph')
    # recall.triples_to_kg(triples)

    # # Recall any relevant opinions about subjects and predicates
    # so = set()
    # for triple in triples:
    #     so.add(triple[0])
    #     so.add(triple[2])

    # await opine(
    #     Opine(
    #         service=event.service,
    #         channel=event.channel,
    #         bot_name=event.bot_name,
    #         bot_id=event.bot_id,
    #         entities=random.sample(list(so), k=min(max_opinions, len(so)))
    #     )
    # )

async def find_goals(event):
    ''' Interrogate the conversation, looking for goals '''

    preamble = f"-----\nIn the previous dialog, does {event.bot_name} express any desires or goals? "
    prompt = preamble + """
Answer in the first person and in JSON format using the following template with no other text or explanation:

{
  goals: ["LIST", "OF", "GOALS"]
}

If no goals or desires are expressed, return an empty JSON list in this format, with no other text:

{
  goals: []
}

Your response MUST return valid JSON.
"""

async def goals_achieved(event):
    ''' Have we achieved our goals? '''
    chat = Chat(persyn_config=persyn_config, service=event.service)

    for goal in event.goals:
        goal_achieved = completion.get_summary(
            event.convo,
            summarizer=f"Q: True or False: {persyn_config.id.name} achieved the goal of {goal}.\nA:"
        )

        log.warning(f"🧐 Did we achieve our goal? {goal_achieved}")
        if 'true' in goal_achieved.lower():
            log.warning(f"🏆 Goal achieved: {goal}")
            services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, f"🏆 _achievement unlocked: {goal}_")
            recall.achieve_goal(event.service, event.channel, goal)
        else:
            log.warning(f"🚫 Goal not yet achieved: {goal}")

    # # Any new goals?
    # summary = completion.nlp(completion.get_summary(
    #     text=event.convo,
    #     summarizer=f"In a few words, {persyn_config.id.name}'s overall goal is:",
    #     max_tokens=100
    # ))

    # # 1 sentence max please.
    # the_goal = ' '.join([s.text for s in summary.sents][:1])

    # log.warning("🥅 Potential goal:", the_goal)

    # # some goals are too easy
    # for taboo in ['remember', 'learn']:
    #     if taboo in the_goal:
    #         return

    # new_goal = AddGoal(
    #     bot_name=persyn_config.id.name,
    #     bot_id=persyn_config.id.guid,
    #     service=event.service,
    #     channel=event.channel,
    #     goal=the_goal
    # )
    # await add_goal(new_goal)

def text_from_url(url, selector='body'):
    ''' Return just the text from url. You probably want a better selector than <body>. '''
    try:
        article = rs.get(url, timeout=30)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
        log.error(f"🗞️ Could not fetch article {url}", err)
        return ''

    soup = BeautifulSoup(article.text, features="lxml")
    story = []
    for line in soup.select_one(selector).text.split('\n'):
        if not line:
            continue
        story.append(line)

    return '\n'.join(story)

async def read_web(event):
    ''' Read a web page '''
    if persyn_config.web.get(urlparse(event.url).netloc, None):
        cfg = persyn_config.web.get(urlparse(event.url).netloc)
        selector = cfg.get('selector', 'body')
        stop = cfg.get('stop', [])
    else:
        selector = 'body'
        stop = []

    chat = Chat(persyn_config=persyn_config, service=event.service)
    log.debug(text_from_url(event.url, selector))

    if not event.reread and recall.have_read(event.service, event.channel, event.url):
        log.info("🕸️ Already read:", event.url)
        chat.inject_idea(
            channel=event.channel,
            idea=f"and doesn't need to re-read it: {event.url}",
            verb="already read this article"
        )
        reply = chat.get_reply(
            channel=event.channel,
            msg='...',
            speaker_name=event.bot_name
        )
        services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, reply)
        return

    recall.add_news(event.service, event.channel, event.url, "web page")

    body = text_from_url(event.url, selector)

    if not body:
        log.error("🗞️ Got empty body from", event.url)
        return

    log.info("📰 Reading", event.url)

    prompt = "To briefly summarize this article:"
    max_reply_length = 300
    done = False
    for chunk in completion.paginate(body, prompt=prompt, max_reply_length=max_reply_length):
        for stop_word in stop:
            if stop_word in chunk:
                log.warning("📰 Stopping here:", stop_word)
                chunk = chunk[:chunk.find(stop_word)]
                done = True

        summary = completion.get_summary(
            text=chunk,
            summarizer=prompt
        )

        chat.inject_idea(
            channel=event.channel,
            idea=f"{summary} {event.url}",
            verb="saw on the web"
        )
        services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, f"{summary} {event.url}")

        if done:
            return

async def read_news(event):
    ''' Check our RSS feed. Read the first unread article. '''
    log.info("🗞️  Reading news feed:", event.url)
    try:
        page = rs.get(event.url, timeout=30)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
        log.error(f"🗞️  Could not fetch RSS feed {event.url}", err)
        return

    feed = BeautifulSoup(page.text, "xml")
    for item in feed.find_all('item'):
        item_url = item.find('link').text
        if recall.have_read(event.service, event.channel, item_url):
            log.info("🗞️  Already read:", item_url)
            continue

        item_event = Web(
            service=event.service,
            channel=event.channel,
            bot_name=event.bot_name,
            bot_id=event.bot_id,
            url=item_url,
            reread=False
        )
        await read_web(item_event)
        # only one at a time
        return

async def reflect_on(event):
    ''' Reflect on recent events. Inspired by Stanford's Smallville, https://arxiv.org/abs/2304.03442 '''
    log.warning("🪩  Reflecting...")

    convo = '\n'.join(recall.convo(event.service, event.channel, feels=True))
    convo_id = event.convo_id or recall.convo_id(event.service, event.channel)
    chat = Chat(persyn_config=persyn_config, service=event.service)

    """
    Given only the dialog above, what are the three most salient high-level question that can be asked about Anna?

    What three actions can Anna take to answer those questions?

    Please convert pronouns and verbs to the first person, and format your reply using JSON in the following format:

    {
    "questions": ["THE QUESTIONS", "AS A LIST"],
    "actions": ["THE ACTIONS", "AS A LIST"]
    }

    Your response should only include JSON, no other text. Your response MUST return valid JSON.
    """

    questions = completion.get_reply(
        f"""{convo}
Given only the information above, what are three most salient high-level questions I can answer about the people in the statements?
Questions only, no answers. Please convert pronouns and verbs to the first person.
"""
    ).split('?')

    log.warning("🪩 ", questions)

    # Answer each question, supplemented by relevant memories.
    for question in questions:
        question = question.strip().strip('"\'')
        if len(question) < 10:
            if question:
                log.warning("⁉️  Bad question:", question)
            continue

        log.warning("❓ ", question)

        ranked = recall.find_related_convos(
            event.service, event.channel,
            query=convo,
            size=5,
            current_convo_id=convo_id,
            threshold=persyn_config.memory.relevance * 1.4,
            any_convo=True
        )

        visited = []
        context = [convo]
        for hit in ranked:
            if hit.convo_id not in visited:
                if hit.service == 'import_service':
                    log.info("📚 Hit found from import:", hit.channel)
                the_summary = recall.get_summary_by_id(hit.convo_id)
                # Hit a sentence? Inject the summary and the sentence.
                if the_summary and the_summary not in convo:
                    context.append(f"""
                        {persyn_config.id.name} remembers that {ago(recall.id_to_timestamp(hit.convo_id))} ago,
                        f"{the_summary.summary} From that conversation, {hit.msg}"""
                    )
                # No summary? Just inject the sentence.
                else:
                    context.append(f"""
                        {persyn_config.id.name} remembers that {ago(recall.id_to_timestamp(hit.convo_id))} ago, {hit.msg}"""
                    )
                visited.append(hit.convo_id)
                log.info(f"🧵 Related convo {hit.convo_id} ({float(hit.score):0.3f}):", hit.msg[:50] + "...")

        prompt = '\n'.join(context) + f"""
{persyn_config.id.name} asks: {question}?
Respond with the best possible answer from {persyn_config.id.name}'s point of view.
Don't use proper names, and convert all pronouns and verbs to the first person.
"""
        log.warning("✏️", question)

        answer = completion.get_reply(prompt)
        log.warning("❗️", answer)

        # Inject the question and answer.
        chat.inject_idea(
            channel=event.channel,
            idea=f"{question}? {answer}",
            verb="reflects"
        )

    if event.send_chat:
        await elaborate(event)

    log.warning("🪩  Done reflecting.")

def generate_photo(event):
    ''' Generate a photo '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    chat.take_a_photo(event.channel, event.prompt, width=event.size[0], height=event.size[1])

@autobus.subscribe(SendChat)
async def sendchat_event(event):
    ''' Dispatch SendChat event w/ optional images. '''
    log.debug("SendChat received", event)
    await say_something(event)

@autobus.subscribe(ChatReceived)
async def chatreceived_event(event):
    ''' Dispatch ChatReceived event '''
    log.debug("ChatReceived received", event)
    await chat_received(event)

@autobus.subscribe(Idea)
async def idea_event(event):
    ''' Dispatch idea event. '''
    log.debug("Idea received", event)
    await new_idea(event)

@autobus.subscribe(Summarize)
async def summarize_event(event):
    ''' Dispatch summarize event. '''
    log.debug("Summarize received", event)
    await summarize_channel(event)

@autobus.subscribe(Elaborate)
async def elaborate_event(event):
    ''' Dispatch elaborate event. '''
    log.debug("Elaborate received", event)
    await elaborate(event)

@autobus.subscribe(Opine)
async def opine_event(event):
    ''' Dispatch opine event. '''
    log.debug("Opine received", event)
    await opine(event)

@autobus.subscribe(CheckGoals)
async def check_goals_event(event):
    ''' Dispatch CheckGoals event. '''
    log.debug("CheckGoals received", event)
    await goals_achieved(event)

@autobus.subscribe(AddGoal)
async def goals_event(event):
    ''' Dispatch AddGoal event. '''
    log.debug("AddGoal received", event)
    await add_goal(event)

@autobus.subscribe(VibeCheck)
async def feels_event(event):
    ''' Dispatch VibeCheck event. '''
    log.debug("VibeCheck received", event)
    # asyncio.sleep(3)
    await check_feels(event)

@autobus.subscribe(FactCheck)
async def facts_event(event):
    ''' Dispatch FactCheck event. '''
    log.debug("FactCheck received", event)
    await check_facts(event)

@autobus.subscribe(KnowledgeGraph)
async def kg_event(event):
    ''' Dispatch KnowledgeGraph event. '''
    log.debug("KnowledgeGraph received", event)
    await build_knowledge_graph(event)

@autobus.subscribe(News)
async def news_event(event):
    ''' Dispatch News event. '''
    log.debug("News received", event)
    await read_news(event)

@autobus.subscribe(Web)
async def web_event(event):
    ''' Dispatch Web event. '''
    log.debug("Web received", event)
    await read_web(event)

@autobus.subscribe(Reflect)
async def reflect_event(event):
    ''' Dispatch Reflect event. '''
    log.debug("Reflect received", event)
    await reflect_on(event)

@autobus.subscribe(Photo)
async def photo_event(event):
    ''' Dispatch Reflect event. '''
    log.debug("Photo received", event)
    await generate_photo(event)

##
# recurring events
##
@autobus.schedule(autobus.every(5).seconds)
async def auto_summarize():
    ''' Automatically summarize conversations when they expire. '''
    convos = recall.list_convo_ids()

    if convos:
        for convo_id in convos:
            remaining = persyn_config.memory.conversation_interval - elapsed(recall.id_to_timestamp(recall.get_last_message_id(convo_id)), get_cur_ts())
            log.info(f"💓 Active convo: {convo_id} (expires in {int(remaining)} seconds)")

    # for key in convos:
    #     (service, channel, convo_id) = key.split('|')
    #     # TODO: Also check if the convo is too long, even if it hasn't expired

    #     # it should be stale and have more in it than a new_convo marker
    #     if recall.expired(service, channel) and recall.get_last_message(service, channel).verb != 'new_convo':
    #         log.warning("💓 Convo expired:", key)

    #         if len(recall.convo(service, channel, convo_id, verb='dialog')) > 3:
    #             log.info("🪩  Reflecting:", convo_id)
    #             event = Reflect(
    #                 bot_name=persyn_config.id.name,
    #                 bot_id=persyn_config.id.guid,
    #                 service=service,
    #                 channel=channel,
    #                 send_chat=True,
    #                 convo_id=convo_id
    #             )
    #             autobus.publish(event)

    #             log.info("💓 Summarizing:", convo_id)
    #             event = Summarize(
    #                 bot_name=persyn_config.id.name,
    #                 bot_id=persyn_config.id.guid,
    #                 service=service,
    #                 channel=channel,
    #                 convo_id=convo_id,
    #                 photo=True,
    #                 max_tokens=30,
    #                 send_chat=False
    #             )
    #             autobus.publish(event)

@autobus.schedule(autobus.every(6).hours)
async def plan_your_day():
    ''' Make a schedule of actions for the next part of the day '''
    log.info("📅 Time to make a schedule")
    # TODO: use LangChain to decide on the actions to take for the next interval, and inject as an idea.

def main():
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
    global persyn_config
    persyn_config = load_config(args.config_file)

    if not hasattr(persyn_config, 'cns'):
        raise SystemExit('cns not defined in config, exiting.')

    # enable logging to disk
    if hasattr(persyn_config.id, "logdir"):
        logging.getLogger().addHandler(logging.FileHandler(f"{persyn_config.id.logdir}/{persyn_config.id.name}-cns.log"))

    global mastodon
    mastodon = Mastodon(args.config_file)
    mastodon.login()

    global recall
    recall = Recall(persyn_config)

    global completion
    completion = LanguageModel(config=persyn_config)

    log.info(f"⚡️ {persyn_config.id.name}'s CNS is online")

    try:
        autobus.run(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)

    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        print()
        raise SystemExit(0) from kbderr

if __name__ == '__main__':
    main()
