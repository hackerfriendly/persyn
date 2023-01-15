#!/usr/bin/env python3
"""
persyn.py

Chat with your persyn on the command line.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import argparse
import os
import random
import sys
import tempfile
import uuid

from pathlib import Path
from hashlib import sha256

from mastodon import Mastodon as MastoNative, MastodonError, MastodonMalformedEventError, StreamListener
from bs4 import BeautifulSoup

import requests
import spacy

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

from interaction.interact import Interact

import fileinput

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Import text into Persyn'''
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to bot config (default: use $PERSYN_CONFIG)',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('--title', type=str, help='A title for this document (required)')
    parser.add_argument('--author', type=str, help='Author of this document (required)')
    parser.add_argument('--convo_id', type=str, help='convo_id (if not specified, generate one)', default=None)
    parser.add_argument('--archetypes', action='store_true', help='Convert entities to archetypes (default: False)', default=False)
    parser.add_argument('files', nargs='?', type=str, help='One or more text files to import')

    args = parser.parse_args()

    persyn_config = load_config(args.config)
    interact = Interact(persyn_config)

    with fileinput.FileInput(files=args.files, mode='r') as f:
        for line in f:
            if not line.strip():
                continue

            assert interact.recall.ltm.save_relationship_graph(
                    service='import_service',
                    channel=f"{args.author.replace('|', '_')}|{args.title}",
                    convo_id=args.convo_id,
                    text=line,
                    include_archetypes=False
            ) == 'created'

if __name__ == '__main__':
    main()
