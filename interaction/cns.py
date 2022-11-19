#!/usr/bin/env python3
'''
cns.py

The central nervous system. Listen for events and inject them into interact.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import json
import sys
# import uuid

from pathlib import Path

import boto3
import requests

from botocore.exceptions import ClientError

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

# Common chat library
from chat.common import Chat

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

persyn_config = load_config()

sqs = boto3.resource('sqs', region_name=persyn_config.id.aws_region)

try:
    queue = sqs.get_queue_by_name(QueueName=persyn_config.id.sqs_queue)
except ClientError as sqserr:
    try:
        queue = sqs.create_queue(
            QueueName=persyn_config.id.sqs_queue,
            Attributes={
                'DelaySeconds': '0',
                'MessageRetentionPeriod': '345600'
            }
        )
    except ClientError as sqserr:
        raise RuntimeError from sqserr

def post_image_to_slack(chat, channel, prompt, images, bot_name):
    ''' Post the image URL to Slack '''

    # Posting multiple images in a single block doesn't seem to be possible from a bot. Hmm.
    blocks = []
    url = ""
    for i, image in enumerate(images):
        url = f"{persyn_config.dreams.upload.url_base}/{image}"
        blocks.append(
            {
                "type": "image",
                "title": {
                    "type": "plain_text",
                    "text": prompt if i == 0 else " "
                },
                "image_url" : url,
                "alt_text": prompt
            }
        )
    req = {
        "token": persyn_config.chat.slack.bot_token,
        "channel": channel,
        "username": bot_name,
        "text": prompt,
        "blocks": json.dumps(blocks)
    }

    try:
        reply = requests.post('https://slack.com/api/chat.postMessage', data=req)
        reply.raise_for_status()
        log.info(f"⚡️ Posted image to Slack as {bot_name}")
    except requests.exceptions.RequestException as err:
        log.critical(f"⚡️ Could not post image to Slack: {err}")

    chat.inject_idea(channel, f"{persyn_config.id.name} posted a photo of {chat.get_caption(url)}")

def post_image_to_discord(chat, channel, prompt, images, bot_name):
    ''' Post the image URL to Discord '''
    req = {
        "username": persyn_config.id.name,
        "avatar_url": getattr(persyn_config.id, "avatar", "https://hackerfriendly.com/pub/anna/anna.png")
    }
    embeds = []
    url = ""
    for image in images:
        url =  f"{persyn_config.dreams.upload.url_base}/{image}"
        embeds.append(
            {
                "description": prompt or "Untitled",
                "image": {
                    "url": url
                }
            }
        )

    req['embeds'] = embeds

    try:
        reply = requests.post(persyn_config.chat.discord.webhook, json=req)
        reply.raise_for_status()
        log.info(f"⚡️ Posted image to Discord as {bot_name}")
    except requests.exceptions.RequestException as err:
        log.critical(f"⚡️ Could not post image to Discord: {err}")

    chat.inject_idea(channel, f"{persyn_config.id.name} posted a photo of {chat.get_caption(url)}")

def image_ready(event):
    ''' An image has been generated '''

    chat = Chat(persyn_config, service=event['service'])

    if 'slack.com' in event['service']:
        post_image_to_slack(chat, event['channel'], event['caption'], event['images'], event['bot_name'])
    elif 'discord' in event['service']:
        post_image_to_discord(chat, event['channel'], event['caption'], event['images'], event['bot_name'])
    else:
        log.error(f"Unknown service {event['service']}, cannot post photo")
        return

# def new_idea(msg):
    # ''' Inject a new idea '''
    # chat.inject_idea(
    #     channel=msg['channel'],
    #     idea=f"an image of '{msg['caption']}' was posted to {persyn_config.dreams.upload.url_base}/{msg['guid']}.jpg",
    #     verb="notices"
    # )

# Map all event types to the relevant functions
events = {
    'image-ready': image_ready
}

if __name__ == '__main__':
    while True:
        for sqsm in queue.receive_messages(WaitTimeSeconds=20):
            log.info(f"⚡️ {sqsm.body}")

            try:
                msg = json.loads(sqsm.body)

            except json.JSONDecodeError as e:
                log.critical(f"Bad json, skipping message: {sqsm.body}")
                continue

            finally:
                sqsm.delete()

            try:
                events[msg['event_type']](msg)

            except AttributeError:
                log.critical(f"⚡️ Unknown event type: {msg['event_type']}")
