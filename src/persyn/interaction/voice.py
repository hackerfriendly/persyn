''' Text-to-speech support via REST '''
import os
import random

import requests

# Color logging
from color_logging import ColorLog

log = ColorLog()

# Voice support
VOICE_SERVER = os.environ.get('GTTS_SERVER_URL', None)
BOT_VOICE = os.environ.get('BOT_VOICE', 'UK')
VOICES = []

def tts(message, voice=None):
    ''' Send a message to a voice server '''

    # Skip continuation messages
    if message == "...":
        return

    global VOICE_SERVER, VOICES

    if not VOICE_SERVER:
        log.error("No tts voice server found, voice disabled.")
        return

    if not VOICES:
        reply = requests.get(f'{VOICE_SERVER}/voices/')
        if not reply.ok:
            log.error("Could not fetch tts voices:", reply.text)
            VOICE_SERVER = None
            return
        VOICES = [v for v in reply.json()['voices'] if v != BOT_VOICE]
        log.warning("📣 Available voices:", VOICES)

    if voice is None:
        voice = random.choice(VOICES)

    req = {
        "text": message,
        "voice": voice
    }
    reply = requests.post(f'{VOICE_SERVER}/say/', params=req)

    if reply.ok:
        log.info(f"📣 Sent to tts: ({voice})", message)
    else:
        log.error("📣 Connect to tts failed:", reply.text)
