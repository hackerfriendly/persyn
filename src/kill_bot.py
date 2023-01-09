#!/usr/bin/env python3
'''
Launch a persyn bot using tmux
'''
import argparse

from subprocess import run

# Color logging
from utils.color_logging import log

def main():
    ''' Main event '''
    parser = argparse.ArgumentParser(
        description='''Terminate a Persyn bot by its session name.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tmux', type=str, help="Path to tmux", default='/usr/bin/tmux')
    parser.add_argument('session',
        type=str,
        help="Name of the tmux session",
        default=None
    )
    # parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    log.warning(f"💀 Killing {args.session}")

    run([args.tmux, 'kill-session', '-t', f'={args.session}'], check=False)

if __name__ == '__main__':
    main()
