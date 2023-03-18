#!/usr/bin/env python3
"""
read_url.py

Read page on the web.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

# Bot config
from utils.config import load_config

from interaction.messages import Web

import autobus
import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Read a page on the web. Requires the service and channel where it will be read.'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('url', type=str, help='URL to read')
    parser.add_argument('--reread', action='store_true', help='Read the page even if it has been seen before.')
    # parser.add_argument('--selector', type=str, default='body', help='HTML selector for the main story (default: body)')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = Web(
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        service=args.service,
        channel=args.channel,
        url=args.url,
        reread=args.reread
    )

    async def inject_idea():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(inject_idea())

if __name__ == '__main__':
    main()
