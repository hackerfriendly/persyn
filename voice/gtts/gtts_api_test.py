'''
gTTS voice tests

Source the bot config first and ensure the gtts_api.py server is running.
'''
import os
import requests

from gtts_api import VOICES

def test_import():
    ''' Some voices must exist '''
    print(VOICES)
    assert len(VOICES) > 0

def test_all_voices():
    ''' Say hello in your native language '''
    for voice in VOICES:
        req = {
            "text": f"Hello from {voice}",
            "voice": voice
        }
        reply = requests.post(f'{os.environ["GTTS_SERVER_URL"]}/say/', json=req)
        assert reply.ok
