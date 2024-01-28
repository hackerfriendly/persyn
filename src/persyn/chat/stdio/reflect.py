#!/usr/bin/env python3
"""
reflect.py

Reflect on recent events
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

import asyncio

from persyn import autobus

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import Reflect

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Reflect on recent events.'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('--send-chat', action='store_true', help='Comment on what was learned')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = Reflect(
        service=args.service,
        channel=args.channel,
        send_chat=args.send_chat
    )

    async def inject_idea():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(inject_idea())

if __name__ == '__main__':
    main()
