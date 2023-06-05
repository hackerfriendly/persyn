#!/usr/bin/env python3
"""
import_text.py

Import text files. Creates relationship graphs and optionally summarizes as it imports.
Summarization is run through the completion model, so use with caution.

Text should be preformatted to remove line breaks within paragraphs.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, no-member, invalid-name
import argparse
import os
import re

# Color logging
from persyn.utils.color_logging import log

# Bot config
from persyn.utils.config import load_config

from persyn.interaction.interact import Interact

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
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Summarization model (default: gpt-3.5-turbo)')
    parser.add_argument(
        '--summarize',
        action='store_true',
        help='Summarize text using completion (default: False, slow and potentially 💸)',
        default=False
    )
    parser.add_argument('files', nargs='+', help='One or more text files to import')

    args = parser.parse_args()

    persyn_config = load_config(args.config)

    interact = Interact(persyn_config)

    service = 'import_service'
    channel = f"{args.author.replace('|', '_')}|{args.title}"

    # Make a cup of tea and settle down in your favorite chair.
    doc = interact.recall.save_convo_line(
        service,
        channel,
        f"{args.title} by {args.author}",
        persyn_config.id.name,
        persyn_config.id.guid,
        verb="reads"
    )
    log.info("📚 New convo_id:", doc.convo_id)

    for file in args.files:
        with open(file, mode='r', encoding='utf-8') as f:
            lines = []
            for line in f:
                line = line.strip()
                if not line or not re.search('[a-zA-Z]', line):
                    continue
                lines.append(line)

        page = []
        for sent in interact.completion.nlp(' '.join(lines)).sents:
            log.debug(sent.text)

            interact.recall.save_convo_line(
                service,
                channel,
                msg=sent.text,
                speaker_name=args.author,
                speaker_id=None,
                convo_id=interact.recall.convo_id(service, channel),
                verb="wrote"
            )

            if interact.completion.toklen(' '.join(page) + sent.text) + 1 > interact.completion.max_prompt_length():
                if args.summarize:
                    log.debug("Maximum length reached, summarizing.")

                    interact.summarize_convo(
                        service,
                        channel,
                        save=True,
                        dialog_only=False,
                        model=args.model,
                        save_kg=False
                    )

                page = [sent.text]
            else:
                page.append(sent.text)

        if page and args.summarize:
            log.debug("Final summary.")
            interact.summarize_convo(
                service,
                channel,
                save=True,
                dialog_only=False,
                model=args.model,
                save_kg=False
            )

        log.info(f"📚 Imported: {file}")

if __name__ == '__main__':
    main()
