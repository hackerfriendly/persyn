#!/usr/bin/env python3
"""
elaborate.py

Continue the train of thought on a channel via the event bus
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

# Bot config
from utils.config import load_config

from interaction.messages import Elaborate

import autobus
import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Continue the train of thought on a chat channel.'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = Elaborate(
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        service=args.service,
        channel=args.channel,
    )

    async def inject_idea():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(inject_idea())

if __name__ == '__main__':
    main()
