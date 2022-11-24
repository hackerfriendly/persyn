#!/usr/bin/env python3
'''
cns.py

The central nervous system. Listen for events and inject them into interact.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import json
import sys
# import uuid

from pathlib import Path

import boto3

from botocore.exceptions import ClientError

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

# Common chat library
from chat.common import Chat
from chat.simple import slack_msg, discord_msg

# Mastodon support for image posting
from chat.mastodon.login import mastodon

if mastodon:
    from chat.mastodon.donbot import fetch_and_post_image

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

def mastodon_msg(chat, channel, bot_name, caption, images): # pylint: disable=unused-argument
    ''' Post images to Mastodon '''
    if not mastodon:
        return

    for image in images:
        fetch_and_post_image(f"{persyn_config.dreams.upload.url_base}/{image}", f"{caption}\n#imagesynthesis #persyn")

def image_ready(event, service):
    ''' An image has been generated '''
    chat = Chat(persyn_config, service=event['service'])
    services[service](chat, event['channel'], event['bot_name'], event['caption'], event['images'])

def say_something(event, service):
    ''' Send a message to a service + channel '''
    chat = Chat(persyn_config, service=event['service'])
    services[service](chat, event['channel'], event['bot_name'], event['message'])

# def new_idea(msg):
    # ''' Inject a new idea '''
    # chat.inject_idea(
    #     channel=msg['channel'],
    #     idea=f"an image of '{msg['caption']}' was posted to {persyn_config.dreams.upload.url_base}/{msg['guid']}.jpg",
    #     verb="notices"
    # )

# Map all event types to the relevant functions
events = {
    'image-ready': image_ready,
    'say': say_something
}

services = {
    'slack': slack_msg,
    'discord': discord_msg,
    'mastodon': mastodon_msg
}

if __name__ == '__main__':
    log.info(f"⚡️ {persyn_config.id.name}'s CNS is online")
    while True:
        for sqsm in queue.receive_messages(WaitTimeSeconds=20):
            log.info(f"⚡️ {sqsm.body}")

            try:
                msg = json.loads(sqsm.body)

                if 'slack.com' in msg['service']:
                    chat_service = 'slack'
                elif msg['service'] == 'discord':
                    chat_service = 'discord'
                elif msg['service'] == 'mastodon':
                    chat_service = 'mastodon'
                else:
                    log.critical(f"Unknown service {msg['service']}, skipping message: {sqsm.body}")
                    continue

            except json.JSONDecodeError as e:
                log.critical(f"Bad json, skipping message: {sqsm.body}")
                continue

            finally:
                sqsm.delete()

            try:
                events[msg['event_type']](msg, chat_service)

            except AttributeError:
                log.critical(f"⚡️ Unknown event type: {msg['event_type']}")
