"""
Jarvis: live transcription + wake-word + local LLM with tool calling.

Flow:
  microphone -> VAD -> Whisper -> transcript file (always)
                                -> wake-word check
                                     if "hey jarvis" detected:
                                         send rest of utterance to Ollama
                                         Ollama returns tool call(s)
                                         we execute them, print result

Run Ollama first:   `ollama serve`   (usually auto-starts on Windows)
And pull a model:   `ollama pull qwen2.5:7b`   (or llama3.1:8b)

Then:               `python jarvis.py`
"""

from __future__ import annotations

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
from silero_vad import VADIterator, load_silero_vad

from tools import TOOLS, run_tool, DISPATCH
from ha_tools import HA_TOOLS, HA_DISPATCH
import tts

# Merge HA tools into the main registry
ALL_TOOLS = TOOLS + HA_TOOLS
ALL_DISPATCH = {**DISPATCH, **HA_DISPATCH}


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

# === CONFIG: wake word ===
# Whisper often hears "Jarvis" as "Jervis", "Charlie's", "Travis" etc. — fuzzy match catches most.
WAKE_PHRASES   = ["hey jarvis", "hey jervis", "hi jarvis", "okay jarvis", "ok jarvis"]
WAKE_FUZZY     = 0.75   # 0..1 — lower = more permissive
FOLLOWUP_WINDOW_S = 8.0  # seconds we'll listen for a command after a bare wake word

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

User: start my day
→ start_my_day()

User: morning routine
→ start_my_day()

User: search for python tutorials
→ search_web(query="python tutorials")

User: what's the weather today
→ search_web(query="weather today")

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
vad_iterator = VADIterator(
    vad_model,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=MIN_SILENCE_MS,
    speech_pad_ms=SPEECH_PAD_MS,
)

import time

# === STATE / QUEUES ===
audio_q       = queue.Queue()
transcribe_q  = queue.Queue()
stop_event    = threading.Event()

# Conversation history sent to Ollama (kept short — last few turns)
chat_history: list[dict] = []
HISTORY_MAX_TURNS = 6

# Follow-up state: when user says just "hey jarvis" we wait for next utterance
_awaiting_followup_until: float = 0.0


# ===============================================================
# AUDIO + VAD + WHISPER  (unchanged from live_transcribe.py)
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

        if not text:
            continue
        if is_hallucination(text, logprob):
            # Keep this — useful to see what's getting filtered
            print(f"   🗑️  filtered: {text!r}")
            continue

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {text}")
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")
            f.flush()

        # === Wake-word check ===
        global _awaiting_followup_until
        command = extract_command(text)
        if command is not None:
            handle_command(command)
        elif time.time() < _awaiting_followup_until:
            # We're inside the follow-up window after a bare "hey jarvis"
            print(f"   ↪️  treating as follow-up command")
            _awaiting_followup_until = 0.0
            handle_command(text)


def vad_loop():
    pre_roll_samples = int(PRE_ROLL_MS * SAMPLE_RATE / 1000)
    pre_roll = deque(maxlen=pre_roll_samples)

    speech_buffer: list[np.ndarray] = []
    is_speaking = False
    leftover = np.empty(0, dtype=np.float32)

    while not stop_event.is_set():
        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        data = np.concatenate([leftover, block]) if leftover.size else block
        n_windows = len(data) // VAD_WINDOW
        usable = n_windows * VAD_WINDOW
        leftover = data[usable:].copy()
        data = data[:usable]

        for i in range(n_windows):
            window = data[i * VAD_WINDOW : (i + 1) * VAD_WINDOW]
            pre_roll.extend(window)

            if is_speaking:
                speech_buffer.append(window)

            speech_dict = vad_iterator(torch.from_numpy(window), return_seconds=False)
            if speech_dict is None:
                continue

            if "start" in speech_dict and not is_speaking:
                is_speaking = True
                speech_buffer = [np.array(pre_roll, dtype=np.float32)]

            if "end" in speech_dict and is_speaking:
                is_speaking = False
                if speech_buffer:
                    full = np.concatenate(speech_buffer)
                    if len(full) / SAMPLE_RATE * 1000 >= MIN_SPEECH_MS:
                        transcribe_q.put(full)
                speech_buffer = []


# ===============================================================
# WAKE-WORD DETECTION
# ===============================================================

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    return _PUNCT_RE.sub(" ", s.lower()).strip()


def extract_command(text: str) -> str | None:
    """
    If `text` starts with (or fuzzy-matches) a wake phrase, return the remainder.
    Otherwise return None.
    """
    norm = _normalize(text)
    if not norm:
        return None

    for phrase in WAKE_PHRASES:
        # Exact prefix
        if norm.startswith(phrase):
            rest = norm[len(phrase):].strip()
            return rest if rest else "(no command)"

        # Fuzzy match on the first N words
        n_words = len(phrase.split())
        head = " ".join(norm.split()[:n_words])
        ratio = SequenceMatcher(None, head, phrase).ratio()
        if ratio >= WAKE_FUZZY:
            rest = " ".join(norm.split()[n_words:]).strip()
            return rest if rest else "(no command)"

    return None


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
    global _awaiting_followup_until
    print(f"   🤖  Jarvis heard command: \"{command}\"")
    if command == "(no command)":
        print(f"   (listening for follow-up for {FOLLOWUP_WINDOW_S}s...)")
        _awaiting_followup_until = time.time() + FOLLOWUP_WINDOW_S
        return

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

        # Speak the result for chat_reply (full answer) and for everything else (short ack)
        if name == "chat_reply":
            tts.speak(result)  # result == the reply text itself
        elif name == "control_light":
            tts.speak(result)  # "Turned on bucatarie."
        elif name == "start_my_day":
            tts.speak(result)  # spoken morning briefing
        # For open_url / open_app / search_web, we don't speak (browser opening is its own feedback)


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
    print(f"👂  Wake phrases: {WAKE_PHRASES}")
    check_ha_config()
    list_audio_devices()
    warmup_ollama()
    tts.start()
    print("🎤  Listening... say 'hey jarvis, ...' (Ctrl+C to stop)\n")

    worker = threading.Thread(target=whisper_worker, daemon=True)
    worker.start()
    vad_thread = threading.Thread(target=vad_loop, daemon=True)
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