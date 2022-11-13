"""
login.py

Log into Mastodon, if available. Fetches instance and secret from the persyn config.

Exports a mastodon object ready to use, or None if unconfigured.
"""
# pylint: disable=import-error, wrong-import-position, wrong-import-order, invalid-name
import sys

from pathlib import Path

from mastodon import Mastodon, MastodonError

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../../').resolve()))

# Color logging
from utils.color_logging import log

# Bot config
from utils.config import load_config

__all__ = [ 'mastodon' ]

CFG = load_config()

mastodon = None

try:
    masto_instance = getattr(CFG.chat.mastodon, 'instance', None)
    if masto_instance:
        masto_secret = Path(getattr(CFG.chat.mastodon, 'secret', ''))
        if not masto_secret.is_file():
            raise RuntimeError(
                f"Mastodon instance specified but secret file '{masto_secret}' does not exist.\nCheck your config."
            )
        try:
            mastodon = Mastodon(
                access_token = masto_secret,
                api_base_url = masto_instance
            )
        except MastodonError:
            raise SystemExit("Invalid credentials, run masto-login.py and try again.") from MastodonError

    creds = mastodon.account_verify_credentials()
    log.info(
        f"Logged into Mastodon as @{creds.username}@{masto_instance} ({creds.display_name})"
    )
except AttributeError:
    log.info(f"No Mastodon configuration for {CFG.id.name}, skipping.")
