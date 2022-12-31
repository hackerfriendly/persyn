#!./interaction/env/bin/python3
'''
Launch a persyn bot using tmux
'''
#pylint: disable=no-member, wrong-import-position
import os
import sys
import argparse

from pathlib import Path
from subprocess import run, CalledProcessError

# Add persyn root to sys.path
sys.path.insert(0, str(Path(__file__).resolve()))

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

def run_cmd(cmd):
    return run(
        cmd,
        shell=False,
        check=True,
        env=os.environ.copy()
    )

def tmux_is_running(session_name):
    return run([args.tmux, 'has-session', '-t', session_name], shell=False, check=False, capture_output=True).returncode == 0

def run_tmux_cmd(session_name, cmd):

    running = tmux_is_running(session_name)
    tmux = ' '.join([
            args.tmux,
            'split-pane' if running else 'new-session',
            '-t' if running else '-s',
            session_name,
            '-d'
         ]
    )

    return run(f"""{tmux} 'while :; do {' '.join(cmd)} ; sleep 1; done'""",
        shell=True,
        check=True,
        capture_output=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Launch local services for your Persyn bot.''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        help='Path to bot config, or use $PERSYN_CONFIG if set.',
        default=os.environ.get('PERSYN_CONFIG', None)
    )
    parser.add_argument('--tmux', type=str, help="Path to tmux", default='/usr/bin/tmux')
    parser.add_argument('--session', type=str, help="Name of the tmux session (use the bot's name if None)", default=None)
    # parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()
    cfg = load_config(args.config_file)

    session_name = args.session or cfg.id.name.lower().replace(' ', '')

    if tmux_is_running(session_name):
        raise SystemExit(f'Session {session_name} already exists. Attach with: tmux attach -t {session_name}')

    log.info(f"🤖 Starting services for {cfg.id.name}")
    if hasattr(cfg, 'interact') and hasattr(cfg.interact, 'workers'):
        log.info("🧠 Starting interact-server")
        run_tmux_cmd(session_name, ['interaction/interact-server.py', args.config_file])

    if hasattr(cfg, 'cns'):
        log.info("⚡️ Starting cns")
        run_tmux_cmd(session_name, ['interaction/cns.py', args.config_file])

    if hasattr(cfg, 'chat'):
        if hasattr(cfg.chat, 'slack'):
            log.info("👖 Starting slack")
            run_tmux_cmd(session_name, ['chat/slack/slack.py', args.config_file])

        if hasattr(cfg.chat, 'discord'):
            log.info("🙀 Starting discord")
            run_tmux_cmd(session_name, ['chat/discord/discord-bot.py', args.config_file])

        if hasattr(cfg.chat, 'mastodon'):
            log.info("🎺 Starting mastodon")
            run_tmux_cmd(session_name, ['chat/mastodon/donbot.py', args.config_file])

    # TODO: dreams, etc.

    log.info(f"\n{cfg.id.name} is running. Attach with: tmux attach -t {session_name}")