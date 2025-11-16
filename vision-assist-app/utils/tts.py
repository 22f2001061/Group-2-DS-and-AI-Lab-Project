import os
import tempfile
import json
from gtts import gTTS
import pyttsx3
from config.settings import TTS_CACHE_DIR
from utils.helpers import text_hash


def text_to_audio_bytes_with_cache(text: str):
    key = text_hash(text)
    cache_wav = os.path.join(TTS_CACHE_DIR, f"{key}.wav")

    if os.path.exists(cache_wav):
        return open(cache_wav, "rb").read(), "audio/wav"

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        data = open(tmp, "rb").read()
        open(cache_wav, "wb").write(data)
        os.remove(tmp)
        return data, "audio/wav"
    except Exception:
        pass

    try:
        tts = gTTS(text=text, lang="en", slow=False)
        fd, tmp_mp3 = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        tts.save(tmp_mp3)
        data = open(tmp_mp3, "rb").read()
        os.remove(tmp_mp3)
        return data, "audio/mpeg"
    except Exception:
        pass

    return b"", "application/octet-stream"
