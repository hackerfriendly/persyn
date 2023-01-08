#!/usr/bin/env python3
'''
cns.py

The central nervous system. Listen for events and inject them into interact.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import json
import os
import argparse
# import uuid

import boto3

from botocore.exceptions import ClientError

# Common chat library
from chat.common import Chat
from chat.simple import slack_msg, discord_msg

# Mastodon support for image posting
from chat.mastodon.donbot import Mastodon

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

# Defined in main()
mastodon = None
persyn_config = None

def mastodon_msg(_, chat, channel, bot_name, caption, images): # pylint: disable=unused-argument
    ''' Post images to Mastodon '''
    for image in images:
        mastodon.fetch_and_post_image(
            f"{persyn_config.dreams.upload.url_base}/{image}", f"{caption}\n#imagesynthesis #persyn"
        )

def image_ready(event, service):
    ''' An image has been generated '''
    chat = Chat(persyn_config, service=event['service'])
    services[service](persyn_config, chat, event['channel'], event['bot_name'], event['caption'], event['images'])

def say_something(event, service):
    ''' Send a message to a service + channel '''
    chat = Chat(persyn_config, service=event['service'])
    services[service](persyn_config, chat, event['channel'], event['bot_name'], event['message'])

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

    sqs = boto3.resource('sqs', region_name=persyn_config.cns.aws_region)

    try:
        queue = sqs.get_queue_by_name(QueueName=persyn_config.cns.sqs_queue)
    except ClientError:
        try:
            queue = sqs.create_queue(
                QueueName=persyn_config.cns.sqs_queue,
                Attributes={
                    'DelaySeconds': '0',
                    'MessageRetentionPeriod': '345600'
                }
            )
        except ClientError as sqserr:
            raise RuntimeError from sqserr

    log.info(f"⚡️ {persyn_config.id.name}'s CNS is online")

    try:
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

                except json.JSONDecodeError as err:
                    log.critical(f"Bad json, skipping message: {sqsm.body}", err)
                    continue

                finally:
                    sqsm.delete()

                try:
                    events[msg['event_type']](msg, chat_service)

                except AttributeError:
                    log.critical(f"⚡️ Unknown event type: {msg['event_type']}")
    # Exit gracefully on ^C (so the wrapper script while loop continues)
    except KeyboardInterrupt as kbderr:
        print()
        raise SystemExit(0) from kbderr

if __name__ == '__main__':
    main()
