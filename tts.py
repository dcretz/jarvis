"""
Text-to-speech for Jarvis using Microsoft Edge TTS.

Free, no account needed, natural-sounding voices.
Install: pip install edge-tts pygame

Voices to try (set VOICE constant):
  English (US): en-US-AriaNeural, en-US-GuyNeural, en-US-JennyNeural
  English (UK): en-GB-RyanNeural, en-GB-SoniaNeural
  Romanian:     ro-RO-EmilNeural, ro-RO-AlinaNeural

Run `edge-tts --list-voices` for the full list.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading

# === CONFIG ===
VOICE = "en-US-GuyNeural"   # default voice
SPEED = "+10%"              # talk a bit faster than default (jarvis is brisk)

# === STATE ===
_speech_queue: queue.Queue[str | None] = queue.Queue()
_tts_thread: threading.Thread | None = None
_tts_available: bool | None = None  # cached on first check


def _check_available() -> bool:
    """Lazy-check that dependencies are installed."""
    global _tts_available
    if _tts_available is not None:
        return _tts_available
    try:
        import edge_tts  # noqa: F401
        import pygame    # noqa: F401
        _tts_available = True
    except ImportError as e:
        print(f"⚠️  TTS disabled — missing dependency: {e}")
        print("   To enable: pip install edge-tts pygame")
        _tts_available = False
    return _tts_available


async def _synthesize(text: str) -> bytes:
    """Generate MP3 audio bytes from text. Async because edge-tts is async."""
    import edge_tts
    communicate = edge_tts.Communicate(text, VOICE, rate=SPEED)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _play_mp3_bytes(audio: bytes):
    """Play MP3 bytes synchronously using pygame."""
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(io.BytesIO(audio))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)


def _tts_worker():
    """Background worker: pulls text from queue, synthesizes, plays."""
    while True:
        text = _speech_queue.get()
        if text is None:  # shutdown sentinel
            break
        try:
            audio = asyncio.run(_synthesize(text))
            _play_mp3_bytes(audio)
        except Exception as e:
            print(f"⚠️  TTS error: {e}")


def start():
    """Start the TTS worker thread. Idempotent."""
    global _tts_thread
    if not _check_available():
        return
    if _tts_thread is not None and _tts_thread.is_alive():
        return
    _tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="tts-worker")
    _tts_thread.start()


def speak(text: str):
    """Queue text to be spoken. Non-blocking. Safe to call before start()."""
    if not text or not text.strip():
        return
    if not _check_available():
        return
    _speech_queue.put(text)


def stop():
    """Signal the worker to exit. Drops pending speech."""
    if _tts_thread is not None and _tts_thread.is_alive():
        _speech_queue.put(None)