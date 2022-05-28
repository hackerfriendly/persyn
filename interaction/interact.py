'''
interact.py

A REST API for the limbic system.
'''
import os
import random

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

import wikipedia

# Prompt completion
from gpt import GPT

# text-to-speech
from voice import tts

# Emotions courtesy of Dr. Noonian Soong
from feels import get_feels

# Long and short term memory
from memory import Recall

# Time handling
from chrono import natural_time

# Color logging
from color_logging import ColorLog

log = ColorLog()

# These are all defined in config/*.conf
BOT_NAME = os.environ["BOT_NAME"]
BOT_ID = os.environ["BOT_ID"]
BOT_VOICE = os.environ.get('BOT_VOICE', 'USA')

# Minimum completion reply quality. Lower numbers get more dark + sleazy.
MINIMUM_QUALITY_SCORE = float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))

# Temperature. 0.0 == repetitive, 1.0 == chaos
TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.99))

# How are we feeling today?
feels = {'current': get_feels("")}

# GPT-3 for completion
completion = GPT(bot_name=BOT_NAME, min_score=MINIMUM_QUALITY_SCORE)

# FastAPI
app = FastAPI()

# Elasticsearch memory
recall = Recall(
    bot_name=BOT_NAME,
    bot_id=BOT_ID,
    url=os.environ['ELASTIC_URL'],
    auth_name=os.environ["BOT_NAME"],
    auth_key=os.environ.get('ELASTIC_KEY', None),
    index_prefix=os.environ.get('ELASTIC_INDEX_PREFIX', BOT_NAME.lower()),
    conversation_interval=600, # ten minutes
    verify_certs=True
)

def summarize_convo(service, channel, save=True, max_tokens=200):
    '''
    Generate a GPT summary of the current conversation for this channel.
    If save == True, save it to long term memory.
    Returns the text summary.
    '''
    summaries, convo = recall.load(service, channel, summaries=1)
    if not convo:
        return '\n'.join(summaries)

    summary = completion.get_summary(
        text='\n'.join(convo),
        summarizer="To briefly summarize this conversation, ",
        max_tokens=max_tokens
    )
    if save:
        recall.summary(service, channel, summary)
    return summary

def choose_reply(prompt, convo):
    ''' Choose the best reply from a list of possibilities '''

    # TODO: If no replies survive, try again?
    scored = completion.get_replies(
        prompt=prompt,
        convo=convo,
        temperature=TEMPERATURE
    )

    if not scored:
        log.warning("🤨 No surviving replies, try again.")
        scored = completion.get_replies(
            prompt=prompt,
            convo=convo,
            temperature=TEMPERATURE
        )

    if not scored:
        log.warning("😩 No surviving replies, I give up.")
        return ":shrug:"

    for item in sorted(scored.items()):
        log.warning(f"{item[0]:0.2f}:", item[1])

    idx = random.choices(list(sorted(scored)), weights=list(sorted(scored)))[0]
    reply = scored[idx]
    log.info(f"✅ Choice: {idx:0.2f}", reply)

    return reply

def get_reply(service, channel, msg, speaker_name, speaker_id):
    ''' Get the best reply for the given channel. Saves to recall memory. '''
    if recall.expired(service, channel):
        summarize_convo(service, channel, save=True)

    if msg != '...':
        recall.save(service, channel, msg, speaker_name, speaker_id)
        tts(msg)

    # Ruminate a bit
    for entity in extract_entities(msg):
        try:
            hits = wikipedia.search(entity)
            if hits:
                wiki = wikipedia.summary(random.choice(hits), sentences=3)
                summary = completion.nlp(completion.get_summary(
                    text=f"This Wikipeda article:\n{wiki}",
                    summarizer="Can be summarized as: ",
                    max_tokens=100
                ))
                # 2 sentences max please.
                inject_idea(service, channel, ' '.join([s.text for s in summary.sents][:2]))
        except wikipedia.exceptions.WikipediaException:
            continue

    # Load summaries and conversation
    summaries, convo = recall.load(service, channel, summaries=2)

    newline = '\n'
    prefix = "" # TODO: more contextual motivations go here

    prompt = f"""{prefix}It is {natural_time()}. {BOT_NAME} is feeling {feels['current']['text']}.

{newline.join(summaries)}
{newline.join(convo)}
{BOT_NAME}:"""

    reply = choose_reply(prompt, convo)

    recall.save(service, channel, reply, BOT_NAME, BOT_ID)

    tts(reply, voice=BOT_VOICE)
    feels['current'] = get_feels(f'{prompt} {reply}')

    log.warning("😄 Feeling:", feels['current'])

    return reply

def get_status(service, channel):
    ''' status report '''
    paragraph = '\n\n'
    newline = '\n'
    summaries, convo = recall.load(service, channel, summaries=2)
    return f'''It is {natural_time()}. {BOT_NAME} is feeling {feels['current']['text']}.

{paragraph.join(summaries)}

{newline.join(convo)}
'''

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

    # TODO: Wikipedia is slow. Cache these.
    for entity in random.sample(entities, k=3):
        try:
            hits = wikipedia.search(entity)
            if hits:
                wiki = wikipedia.summary(hits, sentences=3)
                summary = completion.nlp(completion.get_summary(
                    text=f"This Wikipeda article:\n{wiki}",
                    summarizer="Can be summarized as: ",
                    max_tokens=100
                ))
                # 2 sentences max please.
                reply[entity] = ' '.join([s.text for s in summary.sents][:2])

        except wikipedia.exceptions.WikipediaException:
            continue

    log.warning("💭 daydream entities:")
    log.warning(reply)
    return reply

def inject_idea(service, channel, idea):
    ''' Directly inject an idea into recall memory. '''
    if recall.expired(service, channel):
        summarize_convo(service, channel, save=True)

    recall.save(service, channel, idea, f"{BOT_NAME} thinks", BOT_ID)

    log.warning("🤔 Thinking:", idea)
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

    return {
        "reply": get_reply(service, channel, msg, speaker_name, speaker_id)
    }

@app.post("/summary/")
async def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    save: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(200),
    ):
    ''' Return the reply '''
    return {
        "summary": summarize_convo(service, channel, save, max_tokens)
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
