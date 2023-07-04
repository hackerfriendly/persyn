#!/usr/bin/env python3
'''
cns.py

The central nervous system. Listen for events on the event bus and inject results into interact.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member, unused-wildcard-import
import os
import argparse

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

def mastodon_msg(_, chat, channel, bot_name, caption, images):  # pylint: disable=unused-argument
    ''' Post images to Mastodon '''
    for image in images:
        mastodon.fetch_and_post_image(
            f"{persyn_config.dreams.upload.url_base}/{image}", f"{caption}\n#imagesynthesis #persyn"
        )

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
        model=persyn_config.completion.summary_model
    )
    if event.send_chat:
        services[get_service(event.service)](persyn_config, chat, event.channel, event.bot_name, summary)

async def elaborate(event):
    ''' Continue the train of thought '''
    chat = Chat(persyn_config=persyn_config, service=event.service)
    chat.get_reply(
        channel=event.channel,
        msg='...',
        speaker_name=event.bot_name,
        speaker_id=event.bot_id
    )

async def opine(event):
    ''' Recall opinions of entities (if any) '''
    chat = Chat(persyn_config=persyn_config, service=event.service)

    for entity in event.entities:
        if not entity.strip() or entity in STOP_WORDS:
            continue

        opinions = recall.opine(event.service, event.channel, entity)
        if opinions:
            log.warning(f"🙋‍♂️ Opinions about {entity}: {len(opinions)}")
            if len(opinions) == 1:
                opinion = opinions[0]
            else:
                opinion = completion.nlp(completion.get_summary(
                    text='\n'.join(opinions),
                    summarizer=f"{event.bot_name}'s opinion about {entity} can be briefly summarized as:",
                    max_tokens=75
                )).text

            chat.inject_idea(
                channel=event.channel,
                idea=opinion,
                verb=f"thinks about {entity}"
            )

async def wikipedia_summary(event):
    ''' Summarize some wikipedia pages '''
    chat = Chat(persyn_config=persyn_config, service=event.service)

    for entity in event.entities:
        if not entity.strip() or entity in STOP_WORDS:
            continue

        log.warning(f'📚 Look up "{entity}" on Wikipedia')

        entity = entity.strip().lower()

        # Missing? Look it up.
        # None? Ignore it.
        # Present? Use it.
        if entity in wikicache and wikicache[entity] is not None:
            log.warning(f'🤑 wiki cache hit: "{entity}"')
        else:
            wiki = None
            try:
                if wikipedia.page(entity, auto_suggest=False).original_title.lower() != entity.lower():
                    log.warning(f"❎ no exact match found for {entity}")
                    continue

                log.warning(f"✅ found {entity}")
                wiki = wikipedia.summary(entity, sentences=3)

                summary = completion.nlp(completion.get_summary(
                    text=f"This Wikipedia article:\n{wiki}",
                    summarizer="Can be briefly summarized as: ",
                    max_tokens=75
                ))
                # 3 sentences max please.
                wikicache[entity] = ' '.join([s.text for s in summary.sents][:3])

            except WikipediaException:
                log.warning(f"❎ no unambiguous wikipedia entry found for {entity}")
                wikicache[entity] = None
                continue

        if entity in wikicache and wikicache[entity] is not None:
            chat.inject_idea(event.channel, wikicache[entity], verb="recalls")


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
    feels = completion.get_feels(event.room)
    recall.save_convo_line(
        service=event.service,
        channel=event.channel,
        msg=feels,
        speaker_name=event.bot_name,
        speaker_id=event.bot_id,
        convo_id=event.convo_id,
        verb='feels'
    )
    log.warning("😄 Feeling:", feels)

async def build_knowledge_graph(event):
    ''' Build the knowledge graph. '''
    triples = completion.model.generate_triples(event.convo)
    log.warning(f'📉 Saving {len(triples)} triples to the knowledge graph')
    recall.triples_to_kg(triples)

async def goals_achieved(event):
    ''' Have we achieved our goals? '''
    chat = Chat(persyn_config=persyn_config, service=event.service)

    for goal in event.goals:
        goal_achieved = completion.get_summary(
            event.convo,
            summarizer=f"Q: True or False: {persyn_config.id.name} achieved the goal of {goal}.\nA:",
            max_tokens=10
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
        article = requests.get(url, timeout=30)
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
        log.error(f"🗞️ Could not fetch article {url}", err)
        return ''

    soup = BeautifulSoup(article.text, 'html')
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
            speaker_name=event.bot_name,
            speaker_id=event.bot_id
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
            summarizer=prompt,
            max_tokens=max_reply_length
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
        page = requests.get(event.url, timeout=30)
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

@autobus.subscribe(SendChat)
async def chat_event(event):
    ''' Dispatch chat event w/ optional images. '''
    log.debug("SendChat received", event)
    await say_something(event)

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

@autobus.subscribe(Wikipedia)
async def wiki_event(event):
    ''' Dispatch wikipedia event. '''
    log.debug("Wikipedia received", event)
    await wikipedia_summary(event)

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
    await check_feels(event)

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

##
# recurring events
##
@autobus.schedule(autobus.every(10).seconds)
async def auto_summarize():
    ''' Automatically summarize conversations when they expire. '''
    convos = [convo.decode() for convo in recall.list_convos()]

    if convos:
        log.info("💓 Active convos:", convos)

    for key in convos:
        (service, channel, convo_id) = key.split('|')
        # it should be stale and have more in it than a new_convo marker
        if recall.expired(service, channel) and recall.get_last_message(service, channel).verb != 'new_convo':
            log.warning("💓 Convo expired:", key)

            # Remove it from the convo list
            recall.redis.srem(f"{recall.active_convos_prefix}", key)

            if len(recall.convo(service, channel, convo_id, verb='dialog')) > 3:
                log.info("💓 Summarizing:", convo_id)
                event = Summarize(
                    bot_name=persyn_config.id.name,
                    bot_id=persyn_config.id.guid,
                    service=service,
                    channel=channel,
                    convo_id=convo_id,
                    photo=True,
                    max_tokens=30,
                    send_chat=False
                )
                autobus.publish(event)

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
