#!/usr/bin/env python3
"""
add_goal.py

Add a goal to a channel via the event bus
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

from persyn import autobus

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import AddGoal

import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Give a bot a goal on a specific chat channel.'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('goal', type=str, help='The goal')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = AddGoal(
        service=args.service,
        channel=args.channel,
        goal=args.goal
    )

    async def add_goal():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(add_goal())

if __name__ == '__main__':
    main()
