#!/usr/bin/env python3

from mastodon import Mastodon, MastodonError
from pathlib import Path
from getpass import getpass

config_dir = Path(f'{Path.home()}/.config/persyn.io/')
config_dir.mkdir(parents=True, exist_ok=True)

instance = input("Instance address (eg. mastodon.social): ")
email = input("Email: ")

client_secret = config_dir / f'{email}@{instance}.client.secret'
user_secret = config_dir / f'{email}@{instance}.user.secret'

if not client_secret.exists():
     Mastodon.create_app(
          'Persyn.io',
          api_base_url = f'https://{instance}',
          to_file = client_secret
     )

if not user_secret.exists():
     mastodon = Mastodon(
         client_id = client_secret,
         api_base_url = instance
     )

     try:
          access_token = mastodon.log_in(
              email,
              getpass("Password: "),
              to_file = user_secret
          )
     except MastodonError:
          raise SystemExit("Invalid credentials, try again.")

mastodon = Mastodon(
    access_token = user_secret,
    api_base_url = f'https://{instance}'
)

creds = mastodon.account_verify_credentials()

print(f"Logged in as: {creds.display_name}, @{creds.username}@{instance}")
