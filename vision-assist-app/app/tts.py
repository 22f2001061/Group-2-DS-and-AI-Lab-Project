import os, hashlib
from pydub import AudioSegment
from gtts import gTTS

CACHE_DIR = "tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _text_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()

class TTS:
    def __init__(self, lang='en'):
        self.lang = lang

    def synthesize(self, text):
        key = _text_hash(text)
        path = os.path.join(CACHE_DIR, f"{key}.mp3")
        if os.path.exists(path):
            return AudioSegment.from_mp3(path)

        tts = gTTS(text=text, lang=self.lang, slow=False)
        tts.save(path)
        return AudioSegment.from_mp3(path)
