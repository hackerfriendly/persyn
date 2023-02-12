#!/usr/bin/env python3
'''
interact_server.py

A REST API for the limbic system.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import os
import argparse
import asyncio

from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

import uvicorn

# The event bus
import autobus

# Interaction routines
from interaction.interact import Interact

# Message classes
from interaction.messages import Opine, Wikipedia

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# FastAPI
app = FastAPI()

# Initialize interact in main()
persyn_config = None
interact = None

@app.get("/", status_code=302)
def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.post("/reply/")
def handle_reply(
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
        "reply": interact.get_reply(service, channel, msg, speaker_name, speaker_id)
    }

@app.post("/summary/")
def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    save: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(200),
    include_keywords: Optional[bool] = Query(False),
    context_lines: Optional[int] = Query(0)
    ):
    ''' Return the reply '''
    return {
        "summary": interact.summarize_convo(service, channel, save, max_tokens, include_keywords, context_lines)
    }

@app.post("/status/")
def handle_status(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "status": interact.get_status(service, channel)
    }

@app.post("/amnesia/")
def handle_amnesia(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "amnesia": interact.amnesia(service, channel)
    }

@app.post("/nouns/")
def handle_nouns(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    return {
        "nouns": interact.extract_nouns(text)
    }

@app.post("/entities/")
def handle_entities(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    return {
        "entities": interact.extract_entities(text)
    }

@app.post("/inject/")
def handle_inject(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    idea: str = Query(..., min_length=1, max_length=16384),
    verb: Optional[str] = Query('recalls', min_length=1, max_length=16384),
    ):
    ''' Inject an idea into the stream of consciousness '''
    interact.inject_idea(service, channel, idea, verb)

    return {
        "status": interact.get_status(service, channel)
    }

@app.post("/opinion/")
def handle_opinion(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    topic: str = Query(..., min_length=1, max_length=16384),
    speaker_id: Optional[str] = Query(None, min_length=1, max_length=36),
    size: Optional[int] = Query(10),
    summarize: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(50)
    ):
    ''' Get our opinion about topic '''

    opinions = interact.opine(service, channel, topic, speaker_id, size)

    if summarize:
        if not opinions:
            return { "opinions": [] }

        return {
            "opinions": [
                interact.completion.get_summary(
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
def add_goal(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    goal: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Add a short-term goal in a given context '''
    interact.add_goal(service, channel, goal)
    return {
        "goals": interact.get_goals(service, channel)
    }

@app.post("/get_goals/")
def get_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Fetch the current short-term goals for a given context '''
    return {
        "goals": interact.get_goals(service, channel)
    }

@app.post("/opine/")
async def opine(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    entities: List[str] = Query(...)
):
    ''' Ask the autobus to gather opinions about entities '''

    event = Opine(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        entities=entities
    )
    autobus.publish(event)

    return {
        "success": True
    }

@app.post("/wikipedia/")
async def wiki(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    entities: List[str] = Query(..., min_length=1, max_length=255)
):
    ''' Summarize some Wikipedia pages '''

    event = Wikipedia(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        entities=entities
    )
    autobus.publish(event)

    return {
        "success": True
    }

async def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Persyn interact_server. Run one server for each bot.'''
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
    global interact
    interact = Interact(persyn_config)

    log.info(f"💃 {persyn_config.id.name}'s interact server starting up")

    uvicorn_config = uvicorn.Config(
        'interaction.interact_server:app',
        host=persyn_config.interact.hostname,
        port=persyn_config.interact.port,
        workers=persyn_config.interact.workers,
        reload=False,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    try:
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        await uvicorn_server.serve()
    finally:
        await autobus.stop()

def launch():
    ''' asyncio wrapper to allow launching from pyproject.toml scripts '''
    asyncio.run(main())

if __name__ == '__main__':
    launch()
