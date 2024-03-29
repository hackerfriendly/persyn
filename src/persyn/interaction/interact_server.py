#!/usr/bin/env python3
'''
interact_server.py

A REST API for the limbic system.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import asyncio
import logging
import os

from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query, Form, BackgroundTasks
from fastapi.responses import RedirectResponse

import uvicorn

# The event bus
from persyn import autobus

# Fast send chat
from persyn.chat.simple import send_msg

# Interaction routines
from persyn.interaction.interact import Interact

# Message classes
from persyn.interaction.messages import ChatReceived, CheckGoals, VibeCheck, News, Web

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
    background_tasks: BackgroundTasks,
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    msg: str = Query(..., min_length=1, max_length=65535),
    speaker_name: str = Query(..., min_length=1, max_length=255),
    send_chat: Optional[bool] = Query(True),
    extra: Optional[str] = Query(None, min_length=1, max_length=65535),
    ) -> dict[str, str]:
    ''' Get a reply to a message posted to a channel '''

    if not msg.strip():
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least one non-space character."
        )

    background_tasks.add_task(interact.retort, service, channel, msg, speaker_name, send_chat, extra)

    return {
        "reply": "queued"
    }

@app.post("/chat_received/")
async def handle_chat_received(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    speaker_name: str = Query(..., min_length=1, max_length=255),
    msg: str = Query(..., min_length=1, max_length=65535),
    extra: Optional[str] = Query(None, min_length=1, max_length=65535),
    ):
    ''' Notify CNS that a message was received '''

    if not msg.strip():
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least one non-space character."
        )

    event = ChatReceived(
        service=service,
        channel=channel,
        speaker_name=speaker_name,
        msg=msg,
        extra=extra
    )
    autobus.publish(event)

    return {
        "success": True
    }

@app.post("/summary/")
async def handle_summary(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    convo_id: Optional[str] = Query(None, min_length=1, max_length=255),
    final: Optional[bool] = Query(False)
):
    ''' Return the most recent summary from convo_id '''
    ret = await asyncio.gather(in_thread(
        interact.summarize_channel, [service, channel, convo_id, final]
    ))
    return {
        "summary": ret[0]
    }

@app.post("/status/")
async def handle_status(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    speaker_name: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Get the channel status, as it would be seen by /reply/ '''
    ret = await asyncio.gather(in_thread(
        interact.status, [service, channel, speaker_name]
    ))
    return {
        "status": ret[0]
    }

@app.post("/nouns/")
async def handle_nouns(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Extract nouns from a string '''
    ret = await asyncio.gather(in_thread(
        interact.lm.extract_nouns, [text]
    ))
    return {
        "nouns": ret[0]
    }

@app.post("/entities/")
async def handle_entities(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Extract entities from a string '''
    ret = await asyncio.gather(in_thread(
        interact.lm.extract_entities, [text]
    ))
    return {
        "entities": ret[0]
    }

@app.post("/inject/")
async def handle_inject(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    idea: str = Form(..., min_length=1, max_length=2e6),
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
    summarize: Optional[bool] = Query(True)
    ):
    ''' Get our opinion about a topic '''

    ret = await asyncio.gather(in_thread(
        interact.surmise, [service, channel, topic, size]
    ))

    opinions = ret[0]

    if summarize:
        if not opinions:
            return { "opinions": [] }

        ret = await asyncio.gather(in_thread(
            interact.completion.get_summary, ['\n'.join(opinions), "To briefly summarize,"]
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
async def handle_add_goal(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    goal: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Add a goal to a channel '''
    ret = []
    if goal.strip():
        ret = await asyncio.gather(in_thread(
            interact.add_goal, [service, channel, goal.strip()]
        ))

    return {
        "goals": ret[0]
    }

@app.post("/get_goals/")
async def handle_get_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    size: Optional[int] = Query(10),
):
    ''' Fetch the current goals for a given channel '''
    ret = await asyncio.gather(in_thread(
        interact.get_goals, [service, channel, None, size]
    ))

    return {
        "goals": ret
    }

@app.post("/list_goals/")
async def handle_list_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    size: Optional[int] = Query(10),
):
    ''' List the current goals for a channel '''
    ret = await asyncio.gather(in_thread(
        interact.list_goals, [service, channel, size]
    ))

    return {
        "goals": ret
    }

@app.post("/check_goals/")
async def handle_check_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    convo: str = Query(..., max_length=65535),
    goals: List[str] = Query(...)
):
    ''' Check whether goals have been achieved for this channel, via the autobus '''
    event = CheckGoals(
        service=service,
        channel=channel,
        convo=convo,
        goals=goals
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/send_msg/")
async def handle_send_msg(
    background_tasks: BackgroundTasks,
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    msg: str = Query(..., min_length=1, max_length=65535),
    extra: Optional[str] = Query(None, min_length=1, max_length=65535),
):
    ''' Send a chat message immediately '''

    background_tasks.add_task(send_msg, persyn_config, service, channel, msg, None, extra) # type: ignore

    return {
        "success": True
    }


@app.post("/vibe_check/")
async def handle_vibe_check(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255)
):
    ''' Ask the autobus to vibe check the room '''
    event = VibeCheck(
        service=service,
        channel=channel,
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/read_news/")
async def handle_read_news(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    url: str = Query(..., min_length=9, max_length=4096),
):
    ''' Doomscrolling on the autobus '''
    event = News(
        service=service,
        channel=channel,
        url=url
    )
    autobus.publish(event)

    return {
        "success": True
    }


@app.post("/read_url/")
async def handle_read_url(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    url: str = Query(..., min_length=9, max_length=4096),
    reread: Optional[bool] = Query(False),
):
    ''' Let's surf the interwebs... on the autobus! '''
    event = Web(
        service=service,
        channel=channel,
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

    # enable logging to disk
    if hasattr(persyn_config.id, "logdir"):
        logging.getLogger().addHandler(logging.FileHandler(f"{persyn_config.id.logdir}/{persyn_config.id.name}-interact.log"))

    log.info(f"💃🕺 {persyn_config.id.name}'s interact server starting up")

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
