#!/usr/bin/env python3

from mastodon import Mastodon, MastodonError, MastodonMalformedEventError, StreamListener
from pathlib import Path

import datetime
import json

def defaultconverter(o):
  if isinstance(o, datetime.datetime):
      return o.__str__()


config_dir = Path(f'{Path.home()}/.config/persyn.io/')

# instance = 'mas.to'
# email = 'anna@hackerfriendly.com'

# instance = 'mastodon.social'
# email = 'rob@hackerfriendly.com'

instance = 'botsin.space'
email = 'anna@hackerfriendly.com'

user_secret = config_dir / f'{email}@{instance}.user.secret'

if not user_secret.exists():
     raise RuntimeError("Run masto-login.py first.")

try:
     mastodon = Mastodon(
         access_token = user_secret,
         api_base_url = f'https://{instance}'
     )
except MastodonError:
     raise SystemExit("Invalid credentials, run mast-login.py and try again.")

print(f"Logged in as: {mastodon.me().username} @ {instance}")

class TheListener(StreamListener):

     def on_update(self, update):
          print(f"Got update: {update}")

     def on_conversation(self, conversation):
          print(f"Got conversation: {conversation}")

     def on_notification(self, notification):
          print(f"Got notification: {notification}")

     def handle_heartbeat(self):
          print(f"💓")


listener = TheListener()

while True:
     try:
          mastodon.stream_user(listener)
     except MastodonMalformedEventError:
          print("MastodonMalformedEventError, continuing.")


# desc = "Now I make pictures with Stable Diffusion. If whatever is on my mind doesn't pass the safety filter, I make something like this instead. It looks like a teddy bear running through a field of grass."
# pic = mastodon.media_post("teddy.jpg", "image/jpeg", description=desc)

# resp = mastodon.status_post(desc, media_ids=[pic.id], idempotency_key="abc123456")

# print(json.dumps(resp, default=defaultconverter))
