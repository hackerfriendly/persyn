#!/usr/bin/env python3
"""
summarize.py

Summarize a channel via the event bus
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

from persyn import autobus

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import Summarize

import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Summarize a chat channel.'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('--convo-id', type=str, help='Convo ID (optional)')
    parser.add_argument('--photo', action='store_true', help='Take a photo of the summary (default: False)')
    parser.add_argument('--send-chat', action='store_true', help='Send the summary to the channel (default: False)')
    parser.add_argument('--final', action='store_true', help='Save the final summary (default: False)')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = Summarize(
        service=args.service,
        channel=args.channel,
        convo_id=args.convo_id,
        photo=args.photo,
        send_chat=args.send_chat,
        final=args.final
    )

    async def inject_idea():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(inject_idea())

if __name__ == '__main__':
    main()
