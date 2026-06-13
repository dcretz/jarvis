"""
Jarvis: dedicated wake-word + on-demand transcription + local LLM with tool calling.

Flow:
  microphone -> openWakeWord ("hey_jarvis", always-on, low-latency)
                     if wake word detected:
                         capture the following command utterance (Silero VAD)
                         Whisper transcribes ONLY that command
                         send command to Ollama
                         Ollama returns tool call(s)
                         we execute them, print/speak result

Whisper is NO LONGER used to detect the wake word (that was unreliable — it
depended on Whisper spelling "jarvis" correctly). A small always-on model
listens for the wake word directly on the audio, like "Hey Siri".

Run Ollama first:   `ollama serve`   (usually auto-starts on Windows)
And pull a model:   `ollama pull qwen2.5:7b`   (or llama3.1:8b)

Then:               `python jarvis.py`
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import sys
import threading
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher

# Load .env BEFORE importing ha_tools (which reads HA_TOKEN at module load).
# Install: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from current working dir
except ImportError:
    print("⚠️  python-dotenv not installed — .env file won't be loaded.")
    print("   Install with: pip install python-dotenv")
    print("   (Or set env vars manually in PowerShell with $env:HA_TOKEN = \"...\")")


import numpy as np
import requests
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from openwakeword.model import Model as OWWModel

from tools import TOOLS, run_tool, DISPATCH
from ha_tools import HA_TOOLS, HA_DISPATCH
from youtube_tools import YOUTUBE_TOOLS, YOUTUBE_DISPATCH
import tts

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    print("⚠️  websockets not installed — UI widget won't receive events.")
    print("   Install with: pip install websockets")

# === WEBSOCKET SERVER (live UI widget) ===
_ws_clients: set = set()
_ws_loop: asyncio.AbstractEventLoop | None = None


async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _ws_clients.discard(websocket)


async def _ws_serve():
    async with websockets.serve(_ws_handler, "localhost", 8765):
        await asyncio.Future()


def _start_ws_server():
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    try:
        _ws_loop.run_until_complete(_ws_serve())
    except Exception as e:
        print(f"⚠️  WebSocket server stopped: {e}")


def broadcast(event: dict):
    """Send a JSON event to every connected UI client (non-blocking)."""
    if not _WS_AVAILABLE or _ws_loop is None or not _ws_clients:
        return
    msg = json.dumps(event)
    for client in list(_ws_clients):
        asyncio.run_coroutine_threadsafe(client.send(msg), _ws_loop)

# Merge HA tools into the main registry
ALL_TOOLS = TOOLS + HA_TOOLS + YOUTUBE_TOOLS
ALL_DISPATCH = {**DISPATCH, **HA_DISPATCH, **YOUTUBE_DISPATCH}


def run_any_tool(name: str, args: dict) -> str:
    """Dispatcher that handles both built-in and HA tools."""
    fn = ALL_DISPATCH.get(name)
    if fn is None:
        return f"[Unknown tool: {name}]"
    try:
        return fn(**args)
    except TypeError as e:
        return f"[Bad arguments for {name}: {e}]"
    except Exception as e:
        return f"[Error in {name}: {e}]"

# === CONFIG: audio / whisper ===
MODEL_SIZE      = "medium"
LANGUAGE        = "en"
SAMPLE_RATE     = 16000
VAD_WINDOW      = 512
CAPTURE_BLOCK   = 8192
MIN_SPEECH_MS   = 250
MIN_SILENCE_MS  = 700
SPEECH_PAD_MS   = 200
PRE_ROLL_MS     = 300
TRANSCRIPT_FILE = "transcript.txt"
MIC_DEVICE      = None   # None = system default. Set to a device index (int) to override.

# Whisper hallucinates these phrases on silence/noise. Drop them.
# We match case-insensitively, stripped of trailing punctuation.
HALLUCINATION_BLACKLIST = {
    # Single tokens
    ".", "you", "you.", "okay", "ok", "yeah", "uh", "um", "hmm", "mm", "mhm", "bye",
    # YouTube-titration hallucinations (very common)
    "thank you", "thank you.", "thanks", "thanks.",
    "thanks for watching", "thanks for watching.", "thanks for watching!",
    "thank you for watching", "thank you for watching.",
    "thanks for coming", "thanks for coming.",
    "thank you so much", "thank you so much.",
    "thank you for joining us", "thank you for joining us today",
    "thank you for joining us today.", "thanks for joining us", "thanks for joining us today.",
    "subscribe", "subscribe.", "please subscribe", "please subscribe.",
    "see you next time", "see you next time.",
    "see you in the next video", "see you in the next video.",
    "i'll see you in the next video", "i'll see you in the next video.",
    "bye-bye", "bye bye", "goodbye", "goodbye.",
    # Old initial_prompt echoes
    "this is a conversation in english",
    "aceasta este o conversație în limba română",
    # Whisper ASR test phrases
    "this is a test", "this is a test.",
}

# Hallucination *substrings* — if the whole utterance is essentially one of these, drop it
HALLUCINATION_SUBSTRINGS = [
    "thanks for watching",
    "thank you for watching",
    "thanks for joining",
    "thank you for joining",
    "see you in the next",
    "see you next time",
    "please subscribe",
    "don't forget to subscribe",
    "like and subscribe",
    "i hope you enjoyed",
    "hope you enjoyed this video",
    "that's it for now",
    "that's it for today",
    "let's get started",
    "let's get a little started",
    "and then we'll see",
    "we'll see, we'll see",
]


# Catch utterances that are mostly hallucination filler, even with extra noise around
# (e.g. "Celebrity. Celebrity. Hey Jarvis." or "and then we'll see, thank you.")
def _is_mostly_hallucination(cleaned: str) -> bool:
    # Repeated single-word tokens like "Celebrity. Celebrity."
    words = cleaned.split()
    if 2 <= len(words) <= 6:
        unique = set(w.rstrip(".,!?") for w in words)
        if len(unique) <= 2 and len(words) >= 3:
            return True
    # Ends in "thank you" / "thanks" with very few words total
    if len(words) <= 6 and (cleaned.endswith("thank you") or cleaned.endswith("thanks")
                              or cleaned.endswith("thank you.") or cleaned.endswith("thanks.")):
        return True
    return False


def _is_repetitive(text: str) -> bool:
    """Detect Whisper repetition loops like 'we'll do it again, we'll do it again, ...'."""
    words = text.lower().split()
    if len(words) < 8:
        return False
    # If any 3-word chunk appears 3+ times, it's a repetition loop
    chunks = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
    from collections import Counter
    most_common = Counter(chunks).most_common(1)
    return bool(most_common) and most_common[0][1] >= 3

# === CONFIG: wake word (openWakeWord — dedicated always-on detector) ===
OWW_MODEL        = "hey_jarvis"  # pretrained openWakeWord model (ships with the package)
OWW_THRESHOLD    = 0.5           # 0..1 — raise to cut false triggers, lower to catch more
VAD_SPEECH_PROB  = 0.5           # silero speech-prob threshold while capturing the command
COMMAND_GRACE_MS = 6000          # after wake word, wait this long for the user to START speaking
COMMAND_MAX_MS   = 12000         # hard cap on a single command's length
# Kept only to strip a stray wake-word remnant off the front of the captured command:
WAKE_FILLERS   = {"hey", "hi", "ok", "okay", "hello", "a", "the"}

# === CONFIG: Ollama ===
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"     # smaller & faster than gemma4:latest (which is 26B MoE)
OLLAMA_TIMEOUT = 180            # first request loads model into VRAM (slow); subsequent are fast
OLLAMA_THINK = False            # Gemma 4 has a "thinking" mode that slows responses massively

SYSTEM_PROMPT = """You are Jarvis, a local assistant. You MUST call exactly one tool per turn.

DEFAULT: use chat_reply and answer from your own knowledge.
ONLY use search_web if the user literally says "search", "google", "look up", "find online", or asks about something time-sensitive (today's news, current weather, live prices).
Use open_url ONLY for "open <site>" / "go to <url>".
Use open_app ONLY for "open <app name>".
Use control_light when the user wants to turn lights on/off/dim. Rooms (Romanian or English):
  bucatarie/kitchen, dormitor/bedroom, hol/hallway/bathroom, lampa/lamp,
  birou/office, sufragerie/living, toate/all (all lights together).

Examples (LEARN THESE PATTERNS):

User: turn on the kitchen light
→ control_light(name="bucatarie", action="on")

User: aprinde lumina din bucatarie
→ control_light(name="bucatarie", action="on")

User: stinge sufrageria
→ control_light(name="sufragerie", action="off")

User: stinge dormitorul
→ control_light(name="dormitor", action="off")

User: aprinde holul
→ control_light(name="hol", action="on")

User: turn off all lights
→ control_light(name="toate", action="off")

User: aprinde toate luminile
→ control_light(name="toate", action="on")

User: dim the office light to 30 percent
→ control_light(name="birou", action="on", brightness=30)

User: aprinde lampa la jumate
→ control_light(name="lampa", action="on", brightness=50)

User: toggle the hallway light
→ control_light(name="hol", action="toggle")

User: what are the seven wonders of the world
→ chat_reply(text="The Seven Wonders of the Ancient World are the Great Pyramid of Giza, Hanging Gardens of Babylon, Statue of Zeus, Temple of Artemis, Mausoleum at Halicarnassus, Colossus of Rhodes, and Lighthouse of Alexandria.")

User: what is the capital of romania
→ chat_reply(text="Bucharest.")

User: hello how are you
→ chat_reply(text="Doing well. What can I help with?")

User: open youtube
→ open_url(url="youtube.com")

User: open notepad
→ open_app(name="notepad")

User: open word / deschide word
→ open_app(name="word")

User: open excel / deschide excel
→ open_app(name="excel")

User: open obsidian / deschide obsidian
→ open_app(name="obsidian")

User: start my day
→ start_my_day()

User: morning routine
→ start_my_day()

User: search for python tutorials
→ search_web(query="python tutorials")

User: what's the weather today
→ search_web(query="weather today")

User: play Bohemian Rhapsody
→ play_music(query="Bohemian Rhapsody")

User: pune Eminem
→ play_music(query="Eminem")

User: play something by Daft Punk
→ play_music(query="Daft Punk")

User: next song / urmatoarea piesa / skip
→ next_track()

User: previous song / piesa anterioara / inapoi
→ previous_track()

User: pause / opreste muzica / pause music
→ pause_music()

User: resume / unpause / continua muzica
→ resume_music()

User: stop music / opreste tot
→ stop_music()

RULES:
- Knowledge questions → chat_reply with the answer. NEVER search for things you already know.
- Keep chat_reply answers SHORT (1-2 sentences) — the user is listening, not reading.
- For lights, prefer Romanian names if the user spoke Romanian.
"""

# === DEVICE ===
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"🖥️   Device: {device.upper()} ({compute_type})")

# === MODELS ===
print(f"📥  Loading Whisper '{MODEL_SIZE}'...")
whisper_model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

print("📥  Loading Silero VAD...")
vad_model = load_silero_vad()

print(f"📥  Loading openWakeWord ('{OWW_MODEL}')...")
oww_model = OWWModel(wakeword_models=[OWW_MODEL], inference_framework="onnx")

import time

# === STATE / QUEUES ===
audio_q       = queue.Queue()
transcribe_q  = queue.Queue()
stop_event    = threading.Event()

# Conversation history sent to Ollama (kept short — last few turns)
chat_history: list[dict] = []
HISTORY_MAX_TURNS = 6


# ===============================================================
# AUDIO + WAKE WORD + WHISPER
# ===============================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"⚠️   {status}", file=sys.stderr)
    audio_q.put(indata[:, 0].copy())


def transcribe_segment(audio_data: np.ndarray) -> tuple[str, float]:
    """Returns (text, avg_logprob). Lower logprob = less confident."""
    segments, _ = whisper_model.transcribe(
        audio_data,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
        # NOTE: no initial_prompt — it caused Whisper to echo the prompt
        # back as a "transcription" during silence.
        # We do hallucination filtering ourselves in is_hallucination().
    )
    seg_list = list(segments)
    text = " ".join(seg.text.strip() for seg in seg_list).strip()
    avg_logprob = (sum(seg.avg_logprob for seg in seg_list) / len(seg_list)) if seg_list else -10.0
    return text, avg_logprob


def is_hallucination(text: str, logprob: float) -> bool:
    """Catch known Whisper hallucinations. Be conservative — false positives mean lost transcription."""
    cleaned = text.strip().lower().rstrip(".!?,")
    if not cleaned:
        return True
    if cleaned in HALLUCINATION_BLACKLIST:
        return True
    # Whole utterance is a known hallucination phrase (with surrounding fluff)
    for sub in HALLUCINATION_SUBSTRINGS:
        if sub in cleaned and len(cleaned.split()) <= 12:
            return True
    # Repetition loop
    if _is_repetitive(cleaned):
        return True
    # Repeated tokens or trailing "thank you"
    if _is_mostly_hallucination(cleaned):
        return True
    # Single-word ultra-low-confidence utterances are usually noise
    if len(cleaned.split()) == 1 and logprob < -1.5:
        return True
    return False


def whisper_worker():
    """Transcribe captured COMMAND buffers (post-wake-word) and dispatch to the LLM.

    Whisper no longer runs continuously — it only transcribes the short utterance
    that follows a wake-word detection, which is both faster and far more reliable.
    """
    while not stop_event.is_set():
        try:
            audio = transcribe_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if audio is None:
            break

        try:
            text, logprob = transcribe_segment(audio)
        except Exception as e:
            print(f"❌  Transcription error: {e}", file=sys.stderr)
            continue

        if not text or is_hallucination(text, logprob):
            print(f"   🗑️  no usable command heard ({text!r})")
            continue

        command = strip_wake_prefix(text)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] command: {command!r}")
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {command}\n")
            f.flush()

        if not command.strip():
            print("   (wake word only — no command followed)")
            broadcast({"type": "state", "state": "idle"})
            continue
        broadcast({"type": "message", "role": "user", "text": command})
        handle_command(command)
        broadcast({"type": "state", "state": "idle"})


def _chime():
    """Short local beep so the user knows Jarvis is listening (like the Siri chime)."""
    try:
        import winsound
        winsound.Beep(880, 120)
    except Exception:
        pass


def listen_loop():
    """Always-on wake-word detection.

    IDLE: feed audio to openWakeWord. On detection, beep and switch to CAPTURING.
    CAPTURING: record the command utterance, using Silero VAD to detect when the
    user stops speaking (or a timeout), then hand the buffer to Whisper.
    """
    OWW_FRAME = 1280  # 80ms @ 16k — openWakeWord's native frame size

    oww_leftover = np.empty(0, dtype=np.float32)
    vad_leftover = np.empty(0, dtype=np.float32)

    state = "IDLE"
    cmd_buffer: list[np.ndarray] = []
    cmd_started = False
    total_samples = 0
    silence_samples = 0

    while not stop_event.is_set():
        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if state == "IDLE":
                data = np.concatenate([oww_leftover, block]) if oww_leftover.size else block
                n = len(data) // OWW_FRAME
                oww_leftover = data[n * OWW_FRAME:].copy()
                for i in range(n):
                    frame = data[i * OWW_FRAME : (i + 1) * OWW_FRAME]
                    scores = oww_model.predict((frame * 32767).astype(np.int16))
                    if scores.get(OWW_MODEL, 0.0) >= OWW_THRESHOLD:
                        print(f"\n🟢  Wake word! (score {scores[OWW_MODEL]:.2f}) — listening for command...")
                        broadcast({"type": "state", "state": "listening"})
                        _chime()
                        oww_model.reset()
                        vad_model.reset_states()
                        oww_leftover = np.empty(0, dtype=np.float32)
                        vad_leftover = np.empty(0, dtype=np.float32)
                        cmd_buffer, cmd_started = [], False
                        total_samples, silence_samples = 0, 0
                        state = "CAPTURING"
                        break
                continue

            # === CAPTURING the command utterance ===
            data = np.concatenate([vad_leftover, block]) if vad_leftover.size else block
            n = len(data) // VAD_WINDOW
            vad_leftover = data[n * VAD_WINDOW:].copy()
            usable = data[: n * VAD_WINDOW]
            if usable.size:
                cmd_buffer.append(usable)

            for i in range(n):
                window = usable[i * VAD_WINDOW : (i + 1) * VAD_WINDOW]
                prob = vad_model(torch.from_numpy(window), SAMPLE_RATE).item()
                total_samples += VAD_WINDOW
                if prob >= VAD_SPEECH_PROB:
                    cmd_started = True
                    silence_samples = 0
                else:
                    silence_samples += VAD_WINDOW

            # End-of-command conditions (samples / 16 == milliseconds at 16 kHz)
            done = (
                (cmd_started and silence_samples / 16 >= MIN_SILENCE_MS)
                or (not cmd_started and total_samples / 16 >= COMMAND_GRACE_MS)
                or (total_samples / 16 >= COMMAND_MAX_MS)
            )
            if done:
                if cmd_started and cmd_buffer:
                    transcribe_q.put(np.concatenate(cmd_buffer))
                else:
                    print("   (no command heard after wake word)")
                oww_model.reset()
                vad_model.reset_states()
                oww_leftover = np.empty(0, dtype=np.float32)
                vad_leftover = np.empty(0, dtype=np.float32)
                state = "IDLE"
                broadcast({"type": "state", "state": "idle"})

        except Exception as e:
            print(f"❌  listen_loop error (resetting to IDLE): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Reset to a clean state so the loop can recover
            try:
                oww_model.reset()
                vad_model.reset_states()
            except Exception:
                pass
            oww_leftover = np.empty(0, dtype=np.float32)
            vad_leftover = np.empty(0, dtype=np.float32)
            cmd_buffer, cmd_started = [], False
            total_samples, silence_samples = 0, 0
            state = "IDLE"
            broadcast({"type": "state", "state": "idle"})


# ===============================================================
# COMMAND CLEANUP
# ===============================================================

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    return _PUNCT_RE.sub(" ", s.lower()).strip()


def strip_wake_prefix(text: str) -> str:
    """Remove a stray wake-word remnant from the start of a captured command.

    Capture starts right after the wake word fires, so usually the command is
    clean — but sometimes a tail like "...jarvis" or a filler "hey" leaks in.
    Drop a leading filler and a jarvis-ish first word if present.
    """
    words = _normalize(text).split()
    drop = 0
    if words and words[0] in WAKE_FILLERS:
        drop = 1
    if drop < len(words) and SequenceMatcher(None, words[drop], "jarvis").ratio() >= 0.6:
        drop += 1
    return " ".join(words[drop:]).strip()


# ===============================================================
# LLM DISPATCH
# ===============================================================

def call_ollama(user_message: str) -> dict:
    """Send one user turn to Ollama with tool definitions. Returns the assistant message dict."""
    chat_history.append({"role": "user", "content": user_message})

    # Keep history bounded
    trimmed = chat_history[-HISTORY_MAX_TURNS * 2:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": ALL_TOOLS,
        "stream": False,
        "think": OLLAMA_THINK,
        "options": {"temperature": 0.2},
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]


def warmup_ollama():
    """Preload the model into VRAM so the first real command isn't slow."""
    print(f"🔥  Warming up {OLLAMA_MODEL}...")
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "think": OLLAMA_THINK,
            "options": {"num_predict": 1},
        }
        requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        print("✅  Ollama ready.")
    except Exception as e:
        print(f"⚠️   Warmup failed (will retry on first command): {e}")


def handle_command(command: str):
    print(f"   🤖  Jarvis heard command: \"{command}\"")
    broadcast({"type": "state", "state": "thinking"})
    try:
        msg = call_ollama(command)
    except requests.exceptions.ConnectionError:
        print("   ❌  Can't reach Ollama at localhost:11434. Is `ollama serve` running?")
        return
    except Exception as e:
        print(f"   ❌  Ollama error: {e}")
        return

    # Record assistant reply (without tool results — keep history compact)
    chat_history.append({"role": "assistant", "content": msg.get("content", "")})

    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        # Model replied with plain text — treat as conversational reply
        content = msg.get("content", "").strip()
        if content:
            print(f"   💬  {content}")
            broadcast({"type": "message", "role": "jarvis", "text": content})
            tts.speak(content)
        else:
            print("   (no response)")
        return

    for call in tool_calls:
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                print(f"   ❌  Couldn't parse args for {name}: {args}")
                continue

        result = run_any_tool(name, args)
        print(f"   ⚙️   {name}({args}) → {result}")

        if name == "chat_reply":
            broadcast({"type": "message", "role": "jarvis", "text": result})
            tts.speak(result)
        elif name == "control_light":
            broadcast({"type": "message", "role": "jarvis", "text": result})
            tts.speak(result)
        elif name == "start_my_day":
            broadcast({"type": "message", "role": "jarvis", "text": result})
            tts.speak(result)
        elif name in ("play_music", "pause_music", "resume_music",
                      "next_track", "previous_track", "stop_music"):
            broadcast({"type": "message", "role": "jarvis", "text": result})
            tts.speak(result)


# ===============================================================
# MAIN
# ===============================================================

def list_audio_devices():
    print("\n🎧  Audio input devices:")
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0] if sd.default.device else None
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                marker = " ← DEFAULT" if i == default_in else ""
                print(f"   [{i}] {dev['name']} ({dev['max_input_channels']}ch){marker}")
    except Exception as e:
        print(f"   (couldn't list devices: {e})")
    print()


def check_ha_config():
    import os
    if not os.environ.get("HA_TOKEN"):
        print("⚠️   HA_TOKEN env var not set — light control will fail.")
        print('   Run before starting jarvis:  $env:HA_TOKEN = "your_token"')
    else:
        from ha_tools import HA_URL
        print(f"💡  Home Assistant: {HA_URL} (token set)")


def main():
    print(f"🧠  LLM: {OLLAMA_MODEL} via Ollama")
    print(f"👂  Wake word: '{OWW_MODEL}' (openWakeWord, threshold {OWW_THRESHOLD})")
    check_ha_config()
    list_audio_devices()
    warmup_ollama()
    tts.start()
    tts.set_state_callback(lambda s: broadcast({"type": "state", "state": s}))

    if _WS_AVAILABLE:
        ws_thread = threading.Thread(target=_start_ws_server, daemon=True, name="ws-server")
        ws_thread.start()
        print("🌐  WebSocket UI server listening on ws://localhost:8765")

    print("🎤  Listening... say 'hey jarvis, ...' (Ctrl+C to stop)\n")

    worker = threading.Thread(target=whisper_worker, daemon=True)
    worker.start()
    vad_thread = threading.Thread(target=listen_loop, daemon=True)
    vad_thread.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CAPTURE_BLOCK,
            callback=audio_callback,
            device=MIC_DEVICE,
            latency="high",
        ):
            while not stop_event.is_set():
                sd.sleep(200)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n⏳  Stopping...")
        stop_event.set()
        # Survive a second Ctrl+C during cleanup
        try:
            vad_thread.join(timeout=2)
            transcribe_q.put(None)
            worker.join(timeout=10)
            tts.stop()
        except KeyboardInterrupt:
            print("   (forced exit)")
        print(f"✅  Transcript saved to {TRANSCRIPT_FILE}")


if __name__ == "__main__":
    main()