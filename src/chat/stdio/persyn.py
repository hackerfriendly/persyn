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

parser = argparse.ArgumentParser(
    description='''Import text into Persyn'''
)
parser.add_argument(
    '--config',
    type=str,
    help='Path to bot config (default: use $PERSYN_CONFIG)',
    default=os.environ.get('PERSYN_CONFIG', None)
)
parser.add_argument('--service', type=str, help='service', default='import_service')
parser.add_argument('--channel', type=str, help='channel', default='import_channel')
parser.add_argument('--convo_id', type=str, help='convo_id', default='convo_id')

args = parser.parse_args()

persyn_config = load_config(args.config)
interact = Interact(persyn_config)

for line in fileinput.input():
    print(line, end='')

    assert interact.recall.ltm.save_relationship_graph(
            args.service,
            args.channel,
            args.convo_id,
            line
    )['result'] == 'created'
