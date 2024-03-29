#!/usr/bin/env python3
"""
read_url.py

Read an RSS feed.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

from persyn import autobus

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import News

import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Read an RSS feed. Requires the service and channel where it will be read.'''
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
    # parser.add_argument('--selector', type=str, default='body', help='HTML selector for the main story (default: body)')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = News(
        service=args.service,
        channel=args.channel,
        url=args.url
    )

    async def read_rss():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(read_rss())

if __name__ == '__main__':
    main()
