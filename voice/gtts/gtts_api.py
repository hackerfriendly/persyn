'''
gtts.py

A REST API for generating Persyn voices via Google TTS.
'''
from io import BytesIO
from multiprocessing import Process, Queue
from subprocess import run, CalledProcessError
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from gtts import gTTS

app = FastAPI()

VOICES = {
    "Australia": "com.au",
    "UK": "co.uk",
    "USA": "com",
    "Canada": "ca",
    "India": "co.in",
    "Ireland": "ie",
    "South Africa": "co.za"
}

class Speaker(Process):
    ''' Queue processor '''
    def __init__(self, inq):
        super().__init__()
        self.inq = inq

    @staticmethod
    def speak(voice, text):
        ''' Say the magic words '''
        try:
            print(f"({voice}):", text)
            tts = gTTS(text, lang='en', tld=VOICES[voice])
            mp3 = BytesIO()
            tts.write_to_fp(mp3)
            run(
                ["mpg123", "-"],
                input=mp3.getvalue(),
                shell=False,
                check=False,
                capture_output=True,
                timeout=60
            )
            mp3.close()

        except CalledProcessError as procerr:
            raise HTTPException(
                status_code=500,
                detail="Could not play audio, see console for error."
            ) from procerr

    def run(self):
        ''' speak speak '''
        while True:
            data = self.inq.get()
            if data is None:
                break

            (voice, text) = data
            self.speak(voice, text)

### main ###

voiceq = Queue()
speaker = Speaker(voiceq)
speaker.start()

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Google Text To Speech server. Try /docs"}

@app.get("/voices/")
async def voices():
    ''' List all available voices '''
    return {
        "voices": list(VOICES),
        "success": True
    }

@app.post("/say/")
async def say(
    text: str = Query(..., min_length=1, max_length=5000),
    voice: Optional[str] = Query("USA", max_length=32)
    ):
    ''' Generate with gTTS and pipe to audio. '''

    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Choose one of: {', '.join(list(VOICES))}"
        )

    if not any(c.isalnum() for c in text):
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least one alphanumeric character."
        )

    # Simple filtering goes here
    text = text.replace('*','')

    if not text.strip():
        return {
            "voice": voice,
            "success": False,
            "reason": "empty text"
        }

    voiceq.put([voice, text.strip()])

    return {
        "voice": voice,
        "success": True
    }
