#!/usr/bin/env python3
"""
import_text.py

Import text files. Creates relationship graphs and optionally summarizes as it imports.
Summarization is run through the completion model, so use with caution.

Text should be preformatted to remove line breaks within paragraphs.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name, no-member
import argparse
import os
import re

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.interact import Interact

# Defined in main
interact = None

def save_imported_convo(service, channel, convo_id, text, author, summarize=False):
    ''' Save the imported conversation '''
    log.debug('👉', text)

    if summarize:
        interact.recall.save(service, channel, text, author, None, verb="wrote")
        interact.summarize_convo(service, channel, save=True, dialog_only=False)

    else:
        raise NotImplementedError('Not implemented yet.')

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
    parser.add_argument(
        '--summarize',
        action='store_true',
        help='Summarize text using completion (default: False, might cost $)',
        default=False
    )
    parser.add_argument('files', nargs='+', help='One or more text files to import')

    args = parser.parse_args()

    persyn_config = load_config(args.config)
    global interact
    interact = Interact(persyn_config)

    service = 'import_service'
    channel = f"{args.author.replace('|', '_')}|{args.title}"

    # Make a cup of tea and settle down in your favorite chair.
    if args.summarize:
        interact.recall.save(service, channel, f"{args.title} by {args.author}",
                             persyn_config.id.name, persyn_config.id.guid, verb="reads")

    for file in args.files:
        with open(file, mode='r', encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if not line or not re.search('[a-zA-Z]', line):
                    continue

                convo = ' '.join(lines)
                if interact.completion.toklen(convo + line) + 1 > interact.completion.max_prompt_length():
                    save_imported_convo(
                        service, channel, args.convo_id, convo, args.author, summarize=args.summarize
                    )
                    lines = [line]
                else:
                    lines.append(line)

            if lines:
                save_imported_convo(service, channel, args.convo_id, convo, args.author, summarize=args.summarize)

        log.info(f"📚 imported: {file}")

if __name__ == '__main__':
    main()
