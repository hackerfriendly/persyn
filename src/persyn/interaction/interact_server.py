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
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.responses import RedirectResponse

import uvicorn

# The event bus
import autobus

# Interaction routines
from persyn.interaction.interact import Interact

# Message classes
from persyn.interaction.messages import SendChat, Opine, Wikipedia, CheckGoals, VibeCheck, News, KnowledgeGraph, Web

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

_executor = ThreadPoolExecutor(8)

async def in_thread(func, args):
    ''' Run a function in its own thread and await the result '''
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, func, *args)

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

    ret = await asyncio.gather(in_thread(
        interact.get_reply, [service, channel, msg, speaker_name, speaker_id]
    ))
    return {
        "reply": ret[0]
    }

@app.post("/summary/")
async def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    save: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(200),
    include_keywords: Optional[bool] = Query(False),
    context_lines: Optional[int] = Query(0),
    model: Optional[str] = Query(None, min_length=1, max_length=64)
):
    ''' Return the reply '''
    ret = await asyncio.gather(in_thread(
        interact.summarize_convo, [service, channel, save, max_tokens, include_keywords, context_lines, model]
    ))
    return {
        "summary": ret[0]
    }

@app.post("/status/")
async def handle_status(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    ret = await asyncio.gather(in_thread(
        interact.get_status, [service, channel]
    ))
    return {
        "status": ret[0]
    }

@app.post("/amnesia/")
async def handle_amnesia(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    ret = await asyncio.gather(in_thread(
        interact.amnesia, [service, channel]
    ))
    return {
        "amnesia": ret[0]
    }

@app.post("/nouns/")
async def handle_nouns(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    ret = await asyncio.gather(in_thread(
        interact.extract_nouns, [text]
    ))
    return {
        "nouns": ret[0]
    }

@app.post("/entities/")
async def handle_entities(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    ret = await asyncio.gather(in_thread(
        interact.extract_entities, [text]
    ))
    return {
        "entities": ret[0]
    }

@app.post("/inject/")
async def handle_inject(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    idea: str = Form(..., min_length=1, max_length=65535),
    verb: Optional[str] = Query('recalls', min_length=1, max_length=255),
):
    ''' Inject an idea into the stream of consciousness '''
    await asyncio.gather(in_thread(
        interact.inject_idea, [service, channel, idea, verb]
    ))
    return {
        "success": True
    }

@app.post("/opinion/")
async def handle_opinion(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    topic: str = Query(..., min_length=1, max_length=16384),
    size: Optional[int] = Query(10),
    summarize: Optional[bool] = Query(True),
    max_tokens: Optional[int] = Query(50)
    ):
    ''' Get our opinion about topic '''

    ret = await asyncio.gather(in_thread(
        interact.surmise, [service, channel, topic, size]
    ))

    opinions = ret[0]

    if summarize:
        if not opinions:
            return { "opinions": [] }

        ret = await asyncio.gather(in_thread(
            interact.completion.get_summary, ['\n'.join(opinions), "To briefly summarize,", max_tokens]
        ))

        return {
            "opinions": [
                ret[0]
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
    ''' Add a goal in a given context '''
    ret = []
    if goal.strip():
        ret = await asyncio.gather(in_thread(
            interact.add_goal, [service, channel, goal.strip()]
        ))

    return {
        "goals": ret[0]
    }

@app.post("/get_goals/")
async def get_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    size: Optional[int] = Query(10),
):
    ''' Fetch the current goals for a given context '''
    ret = await asyncio.gather(in_thread(
        interact.get_goals, [service, channel, None, size]
    ))

    return {
        "goals": ret
    }

@app.post("/list_goals/")
async def list_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    size: Optional[int] = Query(10),
):
    ''' List the current goals for a given context '''
    ret = await asyncio.gather(in_thread(
        interact.list_goals, [service, channel, size]
    ))

    return {
        "goals": ret
    }

@app.post("/check_goals/")
async def check_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    convo: str = Query(..., max_length=65535),
    goals: List[str] = Query(...)
):
    ''' Ask the autobus check whether goals have been achieved '''
    event = CheckGoals(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        convo=convo,
        goals=goals
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/send_msg/")
async def send_msg(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    msg: str = Query(..., min_length=1, max_length=65535)
):
    ''' Send a chat message via the autobus '''
    event = SendChat(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        msg=msg
    )
    autobus.publish(event)

    return {
        "success": True
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

@app.post("/vibe_check/")
async def vibe_check(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    convo_id: str = Query(..., min_length=1, max_length=255),
    room: str = Form(..., max_length=65535),
):
    ''' Ask the autobus to vibe check the room '''
    event = VibeCheck(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        convo_id=convo_id,
        room=room
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/build_graph/")
async def build_graph(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    convo_id: str = Query(..., min_length=1, max_length=255),
    convo: str = Form(..., max_length=65535),
):
    ''' Add to this convo to the knowledge graph '''
    event = KnowledgeGraph(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        convo_id=convo_id,
        convo=convo
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/read_news/")
async def read_news(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    url: str = Query(..., min_length=9, max_length=4096),
):
    ''' Doomscrolling on the autobus '''
    event = News(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        url=url
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/read_url/")
async def read_url(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    url: str = Query(..., min_length=9, max_length=4096),
    reread: Optional[bool] = Query(False),
):
    ''' Let's surf the interwebs... on the autobus! '''
    event = Web(
        service=service,
        channel=channel,
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        url=url,
        reread=reread
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
        'persyn.interaction.interact_server:app',
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
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == '__main__':
    launch()
