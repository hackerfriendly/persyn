#!/usr/bin/env python3
"""
opine.py

Recall some opinions using the autobus.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.messages import Opine

import autobus
import asyncio

# Defined in main
interact = None

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Force a Persyn to recall some opinions '''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('service', type=str, help='Chat service (discord, mastodon, or a Slack URL)')
    parser.add_argument('channel', type=str, help='Chat channel id')
    parser.add_argument('entities', nargs='+', help='Entities to form an opinion about')
    parser.add_argument('--verb', type=str, help='Verb for this opinion (default: has an opinion about)', default='has an opinion about')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    event = Opine(
        bot_name=persyn_config.id.name,
        bot_id=persyn_config.id.guid,
        service=args.service,
        channel=args.channel,
        entities=list(args.entities),
    )

    async def opine():
        await autobus.start(url=persyn_config.cns.redis, namespace=persyn_config.id.guid)
        autobus.publish(event)
        await asyncio.sleep(0.1)
        await autobus.stop()

    asyncio.run(opine())

if __name__ == '__main__':
    main()
