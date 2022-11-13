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
except ClientError as error:
    try:
        queue = sqs.create_queue(
            QueueName=persyn_config.id.sqs_queue,
            Attributes={
                'DelaySeconds': '0',
                'MessageRetentionPeriod': '345600'
            }
        )
    except ClientError as error:
        raise RuntimeError(error)

def post_to_slack(channel, prompt, images, bot_name):
    ''' Post the image URL to Slack '''

    # Posting multiple images in a single block doesn't seem to be possible from a bot. Hmm.
    blocks = []
    for i, image in enumerate(images):
        blocks.append(
            {
                "type": "image",
                "title": {
                    "type": "plain_text",
                    "text": prompt if i == 0 else " "
                },
                "image_url" : f"{persyn_config.dreams.upload.url_base}/{image}",
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

def image_ready(msg):
    ''' An image has been generated '''
    chat = Chat(persyn_config, service=msg['service'])

    if 'slack.com' in msg['service']:
        post_to_slack(msg['channel'], msg['caption'], msg['images'], msg['bot_name'])
    else:
        log.error(f"Unknown service {msg['service']}, cannot post photo")


def new_idea(msg):
    pass
    # chat.inject_idea(
    #     channel=msg['channel'],
    #     idea=f"an image of '{msg['caption']}' was posted to {persyn_config.dreams.upload.url_base}/{msg['guid']}.jpg",
    #     verb="notices"
    # )

while True:
    for message in queue.receive_messages(WaitTimeSeconds=20):
        log.info(f"⚡️ {message.body}")

        try:
            msg = json.loads(message.body)

            if msg['event_type'] == 'image-ready':
                image_ready(msg)

            else:
                log.critical(f"⚡️ Unknown event type: {msg['event_type']}")

        except (json.JSONDecodeError, AttributeError) as err:
            log.critical(f"Bad json, skipping message: {message.body}")

        message.delete()
