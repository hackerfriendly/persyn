#!/usr/bin/env python3
"""
inject_idea.py

Send an idea to the autobus.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

from persyn import autobus

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import Idea

import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Inject an idea into a running Persyn'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('idea', type=str, help='Idea to inject')
    parser.add_argument('--verb', type=str, help='Verb for this idea (default: thinks)', default='thinks')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = Idea(
        service=args.service,
        channel=args.channel,
        idea=args.idea,
        verb=args.verb
    )

    async def inject_idea():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(inject_idea())

if __name__ == '__main__':
    main()
