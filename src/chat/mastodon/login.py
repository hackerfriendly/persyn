"""
login.py

Log into Mastodon, if available. Fetches instance and secret from the persyn config.

Exports a mastodon object ready to use, or None if unconfigured.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
from pathlib import Path

from mastodon import Mastodon as MastoNative, MastodonError

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

__all__ = [ 'Mastodon' ]

class Mastodon(cfg=None):
    ''' Wrapper for logging into Mastodon '''
    def __init__(self, cfg):
        ''' Log into Mastodon using credentials from a persyn config file. '''
        self.cfg = cfg
        self.client = None

    def login(self):
        ''' Attempt to log into the Mastodon service '''
        cfg = load_config(self.cfg)

        if not hasattr(cfg.chat, 'mastodon'):
            log.info(f"No Mastodon configuration for {cfg.id.name}, skipping.")
            return False

        masto_instance = getattr(cfg.chat.mastodon, 'instance', None)
        if masto_instance:
            masto_secret = Path(getattr(cfg.chat.mastodon, 'secret', ''))
            if not masto_secret.is_file():
                raise RuntimeError(
                    f"Mastodon instance specified but secret file '{masto_secret}' does not exist.\nCheck your config."
                )
            try:
                self.client = MastoNative(
                    access_token = masto_secret,
                    api_base_url = masto_instance
                )
            except MastodonError:
                raise SystemExit("Invalid credentials, run masto-login.py and try again.") from MastodonError

        creds = self.client.account_verify_credentials()
        log.info(
            f"Logged into Mastodon as @{creds.username}@{masto_instance} ({creds.display_name})"
        )
        return True

    def is_logged_in(self):
        ''' Helper to see if we're logged in '''
        return self.client is not None
