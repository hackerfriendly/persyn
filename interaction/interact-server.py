'''
interact-server.py

A REST API for the limbic system.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import sys

from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

from interact import Interact

# Color logging
# from utils.color_logging import log

# Bot config
from utils.config import load_config

interact = Interact(load_config())

# FastAPI
app = FastAPI()

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

    (reply, achieved) = interact.get_reply(service, channel, msg, speaker_name, speaker_id)
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
        "summary": interact.summarize_convo(service, channel, save, max_tokens, include_keywords, context_lines)
    }

@app.post("/status/")
async def handle_status(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "status": interact.get_status(service, channel)
    }

@app.post("/amnesia/")
async def handle_amnesia(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "amnesia": interact.amnesia(service, channel)
    }

@app.post("/nouns/")
async def handle_nouns(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    return {
        "nouns": interact.extract_nouns(text)
    }

@app.post("/entities/")
async def handle_entities(
    text: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Return the reply '''
    return {
        "entities": interact.extract_entities(text)
    }

@app.post("/daydream/")
async def handle_daydream(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Return the reply '''
    return {
        "daydream": interact.daydream(service, channel)
    }

@app.post("/inject/")
async def handle_inject(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    idea: str = Query(..., min_length=1, max_length=16384),
    ):
    ''' Inject an idea into the stream of consciousness '''
    interact.inject_idea(service, channel, idea)

    return {
        "status": interact.get_status(service, channel)
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
async def add_goal(
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
async def get_goals(
    service: str = Query(..., min_length=1, max_length=255),
    channel: str = Query(..., min_length=1, max_length=255),
    ):
    ''' Fetch the current short-term goals for a given context '''
    return {
        "goals": interact.get_goals(service, channel)
    }
