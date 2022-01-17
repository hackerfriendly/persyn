'''
gtts.py

A REST API for generating Persyn voices via Google TTS.
'''
from subprocess import run, CalledProcessError
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, HTTPException, Body
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

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Google Text To Speech server. Try /docs"}

@app.post("/say/")
async def say(
    text: str = Body(..., min_length=1, max_length=5000),
    voice: Optional[str] = Body("USA", max_length=32)):
    ''' Generate with gTTS and pipe to audio. '''

    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Choose one of: {', '.join(list(VOICES))}"
        )

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
            capture_output=True
        )
        mp3.close()

    except CalledProcessError as procerr:
        raise HTTPException(
            status_code=500,
            detail="Could not play audio, see console for error."
        ) from procerr

    return {
        "voice": voice,
        "success": True
    }
