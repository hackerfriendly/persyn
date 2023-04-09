#!/usr/bin/env python3
"""
send_chat.py

Speak a message on a channel via the event bus.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import SendChat

import autobus
import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Say something on a Persyn channel'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('msg', type=str, help='Message to say')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = SendChat(
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        service=args.service,
        channel=args.channel,
        msg=args.msg
    )

    async def inject_idea():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(inject_idea())

if __name__ == '__main__':
    main()
