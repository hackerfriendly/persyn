#!/usr/bin/env python3
'''
Launch a persyn bot using tmux
'''
#pylint: disable=no-member, wrong-import-position
import os
import argparse

from subprocess import run

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

def tmux_is_running(session, tmux):
    ''' True if tmux session exists '''
    return run(
        [tmux, 'has-session', '-t', f'={session}'],
        shell=False,
        check=False,
        capture_output=True
    ).returncode == 0

def run_tmux_cmd(session, cmd, tmux):
    ''' Start a new tmux session if needed, then add panes for each cmd '''
    running = tmux_is_running(session, tmux)
    tmux = ' '.join([
            tmux,
            'split-pane' if running else 'new-session',
            '-t' if running else '-s',
            session,
            '-d'
         ]
    )

    return run(f"""{tmux} 'while :; do {' '.join(cmd)} ; sleep 1; done'""",
        shell=True,
        check=True,
        capture_output=True,
        env=os.environ.copy()
    )

def main():
    ''' Main event '''
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
    parser.add_argument('--session',
        type=str,
        help="Name of the tmux session (use the bot's name if None)",
        default=None
    )
    # parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()
    cfg = load_config(args.config_file)

    session_name = args.session or cfg.id.name.lower().replace(' ', '')

    # iTerm's fantastic tmux integration.
    # https://iterm2.com/documentation-tmux-integration.html
    ccmode = ' -CC' if os.environ.get('LC_TERMINAL', '') == 'iTerm2' else ''

    if tmux_is_running(session_name, args.tmux):
        raise SystemExit(f'Session {session_name} already exists. Attach with 👉 tmux{ccmode} attach -t {session_name}')

    log.info(f"🤖 Starting services for {cfg.id.name}")
    if hasattr(cfg, 'interact') and hasattr(cfg.interact, 'workers'):
        log.info("🧠 Starting interact_server")
        run_tmux_cmd(session_name, ['interact', args.config_file], args.tmux)

    if hasattr(cfg, 'cns'):
        log.info("⚡️ Starting cns")
        run_tmux_cmd(session_name, ['cns', args.config_file], args.tmux)

    if hasattr(cfg, 'chat'):
        if hasattr(cfg.chat, 'slack'):
            log.info("👖 Starting slack")
            run_tmux_cmd(session_name, ['slack', args.config_file], args.tmux)

        if hasattr(cfg.chat, 'discord'):
            log.info("🙀 Starting discord")
            run_tmux_cmd(session_name, ['discord', args.config_file], args.tmux)

        if hasattr(cfg.chat, 'mastodon'):
            log.info("🎺 Starting mastodon")
            run_tmux_cmd(session_name, ['mastodon', args.config_file], args.tmux)

    if hasattr(cfg, 'dreams'):
        if hasattr(cfg.dreams, 'workers'):
            log.info("😴 Starting dreams server")
            run_tmux_cmd(session_name, ['dreams', args.config_file], args.tmux)

        if hasattr(cfg.dreams, 'stable_diffusion') and hasattr(cfg.dreams.stable_diffusion, 'workers'):
            log.info("🎨 Starting stable_diffusion")
            run_tmux_cmd(session_name, ['stable_diffusion', args.config_file], args.tmux)

    # TODO: dreams, captions, sdd, parrot, voice

    log.info(f"\n{cfg.id.name} is running. Attach with 👉 tmux{ccmode} attach -t {session_name}")

if __name__ == '__main__':
    main()
