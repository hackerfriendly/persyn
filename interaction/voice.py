''' Text-to-speech support via REST '''
import os
import random

import requests

from color_logging import debug, info, warning, error, critical # pylint: disable=unused-import

# Voice support
VOICE_SERVER = os.environ['GTTS_SERVER_URL']
DEFAULT_VOICE = os.environ.get('DEFAULT_VOICE', 'UK')
VOICES = []

def tts(message, voice=None):
    ''' Send a message to a voice server '''

    # Skip continuation messages
    if message == "...":
        return

    global VOICE_SERVER, VOICES

    if not VOICE_SERVER:
        error("No tts voice server found, voice disabled.")
        return

    if not VOICES:
        reply = requests.get(f'{VOICE_SERVER}/voices/')
        if not reply.ok:
            error("Could not fetch tts voices:", reply.text)
            VOICE_SERVER = None
            return
        VOICES = [v for v in reply.json()['voices'] if v != DEFAULT_VOICE]
        warning("📣 Available voices:", VOICES)

    if voice is None:
        voice = random.choice(VOICES)

    req = {
        "text": message,
        "voice": voice
    }
    reply = requests.post(f'{VOICE_SERVER}/say/', params=req)

    if reply.ok:
        info(f"📣 Sent to tts: ({voice})", message)
    else:
        error("📣 Connect to tts failed:", reply.text)
