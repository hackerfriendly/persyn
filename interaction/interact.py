'''
interact.py

A REST API for the limbic system.
'''
import os
import random
import json

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

# just-in-time Wikipedia
import wikipedia
from wikipedia.exceptions import WikipediaException

from spacy.lang.en.stop_words import STOP_WORDS

# Prompt completion
from completion import LanguageModel

# text-to-speech
from voice import tts

# Long and short term memory
from memory import Recall

# Time handling
from chrono import natural_time, ago, today

# Color logging
from color_logging import ColorLog

log = ColorLog()

# These are all defined in config/*.conf
BOT_NAME = os.environ['BOT_NAME']
BOT_ID = os.environ['BOT_ID']
BOT_ENGINE = os.environ.get('BOT_ENGINE', 'openai')
BOT_VOICE = os.environ.get('BOT_VOICE', 'USA')

# How are we feeling today?
feels = {'current': "nothing in particular", 'goals': ["to learn about something new"]}

# Pick a language model for completion
completion = LanguageModel(engine=BOT_ENGINE, bot_name=BOT_NAME)

# FastAPI
app = FastAPI()

# Elasticsearch memory
recall = Recall(
    bot_name=BOT_NAME,
    bot_id=BOT_ID,
    url=os.environ['ELASTIC_URL'],
    auth_name=os.environ['ELASTIC_USER'],
    auth_key=os.environ.get('ELASTIC_KEY', None),
    index_prefix=os.environ.get('ELASTIC_INDEX_PREFIX', BOT_NAME.lower()),
    conversation_interval=600, # ten minutes
    verify_certs=True
)

# local Wikipedia cache
wikicache = {}

def dialog(convo):
    '''
    Return only the words actually spoken in a convo
    TODO: Convo lines should be better structured, but it would require overhauling memory.py
    '''
    return [
        line for line in convo
        if not line.startswith(f"{BOT_NAME} remembers")
        and not line.startswith(f"{BOT_NAME} recalls")
        and not line.startswith(f"{BOT_NAME} thinks")
    ]

def summarize_convo(service, channel, save=True, max_tokens=200, include_keywords=False, context_lines=0):
    '''
    Generate a summary of the current conversation for this channel.
    Also generate and save opinions about detected topics.
    If save == True, save it to long term memory.
    Returns the text summary.
    '''
    summaries, convo = recall.load(service, channel, summaries=3)
    if not convo:
        if not summaries:
            summaries = [ f"{BOT_NAME} isn't sure what is happening." ]

        # No convo? summarize the summaries
        convo = summaries

    spoken = dialog(convo)
    log.warning(f"∑ summarizing convo: {json.dumps(spoken)}")

    summary = completion.get_summary(
        text='\n'.join(convo),
        summarizer="To briefly summarize this conversation,",
        max_tokens=max_tokens
    )
    keywords = completion.get_keywords(summary)

    if save:
        recall.summary(service, channel, summary, keywords)

    for topic in keywords:
        recall.judge(
            service,
            channel,
            topic,
            completion.get_opinions(summary, topic)
        )

    if include_keywords:
        return summary + f"\nKeywords: {keywords}"

    if context_lines:
        return "\n".join(convo[-context_lines:] + [summary])

    return summary

def choose_reply(prompt, convo, goals):
    ''' Choose the best reply from a list of possibilities '''
    if not convo:
        convo = [f'{BOT_NAME} changes the subject.']

    scored = completion.get_replies(
        prompt=prompt,
        convo=convo,
        goals=goals
    )

    if not scored:
        log.warning("🤨 No surviving replies, try again.")
        scored = completion.get_replies(
            prompt=prompt,
            convo=convo,
            goals=goals
        )

    # Uh-oh. Just keep it brief.
    if not scored:
        log.warning("😳 No surviving replies, one last try.")
        scored = completion.get_replies(
            prompt=generate_prompt([], convo[-4:]),
            convo=convo,
            goals=goals
        )

    if not scored:
        log.warning("😩 No surviving replies, I give up.")
        log.info("🤷‍♀️ Choice: none available")
        return ":shrug:"

    for item in sorted(scored.items()):
        log.warning(f"{item[0]:0.2f}:", item[1])

    idx = random.choices(list(sorted(scored)), weights=list(sorted(scored)))[0]
    reply = scored[idx]
    log.info(f"✅ Choice: {idx:0.2f}", reply)

    return reply

def gather_memories(service, channel, entities, summaries, convo):
    ''' Take a trip down memory lane '''
    search_term = ' '.join(entities)
    log.warning(f"ℹ️ look up '{search_term}' in memories")

    for memory in recall.remember(service, channel, search_term, summaries=5):
        # Don't repeat yourself, loopy-lou.
        if memory['text'] in summaries or memory['text'] in '\n'.join(convo):
            continue

        # Stay on topic
        prompt = '\n'.join(convo + [f"{BOT_NAME} remembers that {ago(memory['timestamp'])} ago: " + memory['text']])
        on_topic = completion.get_summary(
            prompt,
            summarizer="Q: True or False: this memory relates to the earlier conversation.\nA:",
            max_tokens=10)

        log.warning(f"🧐 Are we on topic? {on_topic}")
        if 'true' not in on_topic.lower():
            log.warning(f"🚫 Irrelevant memory discarded: {memory}")
            continue

        log.warning(f"🐘 Memory found: {memory}")
        inject_idea(service, channel, memory['text'], f"remembers that {ago(memory['timestamp'])} ago")
        break

def gather_facts(service, channel, entities):
    ''' Gather facts (from Wikipedia) and opinions (from memory) '''
    for entity in entities:
        if entity == '' or entity in STOP_WORDS:
            continue

        opinions = recall.opine(service, channel, entity)
        if opinions:
            log.warning(f"🙋‍♂️ Opinions about {entity}: {len(opinions)}")
            if len(opinions) == 1:
                opinion = opinions[0]
            else:
                opinion = completion.nlp(completion.get_summary(
                    text='\n'.join(opinions),
                    summarizer=f"{BOT_NAME}'s opinion about {entity} can be briefly summarized as:",
                    max_tokens=75
                )).text

            if opinion not in recall.stm.get_bias(service, channel):
                recall.stm.add_bias(service, channel, opinion)
                inject_idea(service, channel, opinion, f"thinks about {entity}")

        log.warning(f'❇️ look up "{entity}" on Wikipeda')

        if entity in wikicache:
            log.warning(f'🤑 wiki cache hit: "{entity}"')
        else:
            wiki = None
            try:
                if wikipedia.page(entity).original_title.lower() != entity.lower():
                    log.warning("❎ no exact match found")
                    continue

                log.warning("✅ found it.")
                wiki = wikipedia.summary(entity, sentences=3)

                summary = completion.nlp(completion.get_summary(
                    text=f"This Wikipedia article:\n{wiki}",
                    summarizer="Can be briefly summarized as: ",
                    max_tokens=75
                ))
                # 2 sentences max please.
                wikicache[entity] = ' '.join([s.text for s in summary.sents][:2])

            except WikipediaException:
                log.warning("❎ no unambigous wikipedia entry found")
                wikicache[entity] = None
                continue

            if entity in wikicache and wikicache[entity] is not None:
                inject_idea(service, channel, wikicache[entity])

def check_goals(service, channel, convo):
    ''' Have we achieved our goals? '''
    achieved = []

    if feels['goals']:
        for goal in feels['goals']:
            goal_achieved = completion.get_summary(
                '\n'.join(convo),
                summarizer=f"Q: True or False: {BOT_NAME} achieved the goal of {goal}.\nA:",
                max_tokens=10
            )

            log.warning(f"🧐 Did we achieve our goal? {goal_achieved}")
            if 'true' not in goal_achieved.lower():
                log.warning(f"🚫 Goal not yet achieved: {goal}")
                continue

            log.warning(f"🏆 Goal achieved: {goal}")
            achieved.append(goal)
            feels['goals'].remove(goal)

    summary = completion.nlp(completion.get_summary(
        text='\n'.join(convo),
        summarizer=f"{BOT_NAME}'s goal is",
        max_tokens=100
    ))
    # 1 sentence max please.
    recall.add_goal(service, channel, ' '.join([s.text for s in summary.sents][:1]))

    return achieved

def get_reply(service, channel, msg, speaker_name, speaker_id): # pylint: disable=too-many-locals
    '''
    Get the best reply for the given channel. Saves to recall memory.

    Returns the best reply, and any goals achieved.
    '''
    feels['goals'] = recall.get_goals(service, channel)

    if recall.expired(service, channel):
        summarize_convo(service, channel, save=True, context_lines=2)

    if msg != '...':
        recall.save(service, channel, msg, speaker_name, speaker_id)
        tts(msg)

    # Load summaries and conversation
    summaries, convo = recall.load(service, channel, summaries=2)
    convo_length = len(convo)
    last_sentence = None

    if convo:
        last_sentence = convo[-1]

    # Ruminate a bit
    entities = extract_entities(msg)

    if entities:
        log.warning(f"🆔 extracted entities: {entities}")
    else:
        entities = completion.get_keywords(convo)
        log.warning(f"🆔 extracted keywords: {entities}")

    if entities:
        # Memories
        gather_memories(service, channel, entities, summaries, convo)

        # Facts and opinions (interleaved)
        gather_facts(service, channel, entities)

    # Goals
    achieved = check_goals(service, channel, convo)

    # If our mind was wandering, remember the last thing that was said.
    if convo_length != len(recall.load(service, channel, summaries=0)[1]):
        inject_idea(service, channel, last_sentence)

    prompt = generate_prompt(summaries, convo)

    # Is this just too much to think about?
    if len(prompt) > completion.max_prompt_length:
        log.warning("🥱 get_reply(): prompt too long, summarizing.")
        summarize_convo(service, channel, save=True, max_tokens=100)
        summaries, _ = recall.load(service, channel, summaries=3)
        prompt = generate_prompt(summaries, convo[-3:])

    reply = choose_reply(prompt, convo, feels['goals'])

    recall.save(service, channel, reply, BOT_NAME, BOT_ID)

    tts(reply, voice=BOT_VOICE)
    feels['current'] = completion.get_feels(f'{prompt} {reply}')

    log.warning("😄 Feeling:", feels['current'])

    return (reply, achieved)

def default_prompt_prefix():
    ''' The default prompt prefix '''
    if feels['goals']:
        goals = f"""\n{BOT_NAME}'s goals include {', '.join(feels['goals'])}."""
    else:
        goals = ""

    return f"""It is {natural_time()} on {today()}. {BOT_NAME} is feeling {feels['current']}.{goals}"""

def generate_prompt(summaries, convo):
    ''' Generate the model prompt '''
    newline = '\n'

    return f"""{default_prompt_prefix()}
{newline.join(summaries)}
{newline.join(convo)}
{BOT_NAME}:"""

def get_status(service, channel):
    ''' status report '''
    paragraph = '\n\n'
    newline = '\n'
    summaries, convo = recall.load(service, channel, summaries=2)
    return f"""{default_prompt_prefix()}

{paragraph.join(summaries)}

{newline.join(convo)}
"""

def amnesia(service, channel):
    ''' forget it '''
    return recall.forget(service, channel)

def extract_nouns(text):
    ''' return a list of all nouns (except pronouns) in text '''
    nlp = completion.nlp(text)
    nouns = {n.text.strip() for n in nlp.noun_chunks if n.text.strip() != BOT_NAME for t in n if t.pos_ != 'PRON'}
    return list(nouns)

def extract_entities(text):
    ''' return a list of all entities in text '''
    nlp = completion.nlp(text)
    return list({n.text.strip() for n in nlp.ents if n.text.strip() != BOT_NAME})

def daydream(service, channel):
    ''' Chew on recent conversation '''
    paragraph = '\n\n'
    newline = '\n'
    summaries, convo = recall.load(service, channel, summaries=5)

    reply = {}
    entities = extract_entities(paragraph.join(summaries) + newline.join(convo))

    for entity in random.sample(entities, k=3):
        if entity == '' or entity in STOP_WORDS:
            continue

        if entity in wikicache:
            log.warning(f"🤑 wiki cache hit: {entity}")
            reply[entity] = wikicache[entity]
        else:
            try:
                hits = wikipedia.search(entity)
                if hits:
                    try:
                        wiki = wikipedia.summary(hits[0:1], sentences=3)
                        summary = completion.nlp(completion.get_summary(
                            text=f"This Wikipedia article:\n{wiki}",
                            summarizer="Can be summarized as: ",
                            max_tokens=100
                        ))
                        # 2 sentences max please.
                        reply[entity] = ' '.join([s.text for s in summary.sents][:2])
                        wikicache[entity] = reply[entity]
                    except WikipediaException:
                        continue

            except WikipediaException:
                continue

    log.warning("💭 daydream entities:")
    log.warning(reply)
    return reply

def inject_idea(service, channel, idea, verb="recalls"):
    ''' Directly inject an idea into recall memory. '''
    if recall.expired(service, channel):
        summarize_convo(service, channel, save=True, context_lines=2)

    recall.save(service, channel, idea, f"{BOT_NAME} {verb}", BOT_ID)

    log.warning(f"🤔 {verb}:", idea)
    return "🤔"

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Interact server. Try /docs"}

@app.post("/reply/")
async def handle_reply(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    msg: str = Query(..., min_length=1, max_length=5000),
    speaker_id: str = Query(..., min_length=1, max_length=255),
    speaker_name: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''

    if not msg.strip():
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least one non-space character."
        )

    (reply, achieved) = get_reply(service, channel, msg, speaker_name, speaker_id)
    return {
        "reply": reply,
        "goals_achieved": achieved
    }

@app.post("/summary/")
async def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    save: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(200),
    include_keywords: Optional[bool] = Query(False),
    context_lines: Optional[int] = Query(0)
    ):
    ''' Return the reply '''
    return {
        "summary": summarize_convo(service, channel, save, max_tokens, include_keywords, context_lines)
    }

@app.post("/status/")
async def handle_status(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "status": get_status(service, channel)
    }

@app.post("/amnesia/")
async def handle_amnesia(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "amnesia": amnesia(service, channel)
    }

@app.post("/nouns/")
async def handle_nouns(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    return {
        "nouns": extract_nouns(text)
    }

@app.post("/entities/")
async def handle_entities(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    return {
        "entities": extract_entities(text)
    }

@app.post("/daydream/")
async def handle_daydream(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "daydream": daydream(service, channel)
    }

@app.post("/inject/")
async def handle_inject(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    idea: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Inject an idea into the stream of consciousness '''
    inject_idea(service, channel, idea)

    return {
        "status": get_status(service, channel)
    }

@app.post("/opinion/")
async def handle_opinion(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    topic: str = Query(..., min_length=1, max_length=16384),
    speaker_id: Optional[str] = Query(None, min_length=1, max_length=36),
    size: Optional[int] = Query(10),
    summarize: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(50)
    ):
    ''' Get our opinion about topic '''

    opinions = recall.opine(service, channel, topic, speaker_id, size)

    if summarize:
        if not opinions:
            return { "opinions": [] }

        return {
            "opinions": [
                completion.get_summary(
                    text='\n'.join(opinions),
                    summarizer="To briefly summarize,",
                    max_tokens=max_tokens
                )
            ]
        }

    # If not summarizing, just return them all
    return {
        "opinions": opinions
    }

@app.post("/add_goal/")
async def add_goal(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    goal: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Add a short-term goal in a given context '''
    recall.add_goal(service, channel, goal)
    return {
        "goals": recall.get_goals(service, channel)
    }

@app.post("/get_goals/")
async def get_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Fetch the current short-term goals for a given context '''
    return {
        "goals": recall.get_goals(service, channel)
    }
