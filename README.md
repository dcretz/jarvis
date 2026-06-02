# 🤖 Jarvis — Local Voice Assistant

A personal always-listening voice assistant that runs **100% locally** on your PC.  
Say **"Hey Jarvis"** and it wakes up, understands you, and responds with a natural voice — no cloud, no subscriptions.

---

## ✨ Features

| Capability | Details |
|---|---|
| 🎙️ Always-on wake word | Listens for **"Hey Jarvis"** (+ fuzzy variants: "Hey Jervis", "Hi Jarvis", etc.) |
| 🗣️ Speech-to-text | `faster-whisper` on GPU (CUDA) — fast and accurate |
| 🧠 Local LLM | **Ollama** with `gemma4:e4b` — fully offline, no API keys |
| 🔊 Text-to-speech | **Edge TTS** (`en-US-GuyNeural`) — natural Microsoft neural voice |
| 💡 Smart home | Controls lights via **Home Assistant** REST API |
| 🌐 Browser control | Opens URLs and searches the web in Brave |
| 📱 App launcher | Opens desktop apps by voice |
| 📅 Morning routine | "Start my day" — opens dashboard + Obsidian + reads Bitcoin update |
| 📝 Live transcript | Every utterance saved to `transcript.txt` |

---

## 🏗️ Architecture

```
Microphone (always on)
    │
    ▼
[Wake-word — fuzzy match on "Hey Jarvis"]
    │  (detected → confirmation beep)
    ▼
[Recorder + Silero VAD]  ←── stops when you go silent
    │
    ▼
[faster-whisper on GPU]  →  text
    │
    ▼
[Ollama LLM + tool calling]  →  action / reply
    │
    ├── 💡 control_light()   →  Home Assistant REST API
    ├── 🌐 open_url()        →  Brave browser
    ├── 📱 open_app()        →  subprocess / Windows start
    ├── 🔍 search_web()      →  Google / YouTube / DuckDuckGo
    ├── 📅 start_my_day()    →  dashboard + Obsidian + BTC price
    └── 💬 chat_reply()      →  conversational answer
              │
              ▼
        [Edge TTS]  →  Speakers
              │
              ▼
        (back to wake-word listening)
```

---

## 📋 Requirements

- Windows 10/11
- Python **3.11**
- NVIDIA GPU with **8 GB+ VRAM** (for fast Whisper + LLM)
- [Ollama](https://ollama.com) installed and running
- [Brave Browser](https://brave.com) (optional — falls back to default browser)
- [Obsidian](https://obsidian.md) (optional — used in morning routine)
- Home Assistant instance (optional — for light control)

---

## 🚀 Setup

### 1. Clone / download the project

```
C:\Users\diobe\Documents\jarvis\
```

### 2. Create a virtual environment

```powershell
python -m venv venv311
.\venv311\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install faster-whisper silero-vad sounddevice torch torchvision torchaudio `
            --index-url https://download.pytorch.org/whl/cu121
pip install requests edge-tts pygame python-dotenv numpy
```

### 4. Pull the Ollama model

```powershell
ollama pull gemma4:e4b
```

> **First run** will be slow (~30s) while the model loads into VRAM. Subsequent responses are fast.

### 5. Configure Home Assistant (optional)

Create a `.env` file in the project folder:

```env
HA_URL=http://100.118.165.65:8123
HA_TOKEN=your_long_lived_access_token_here
```

To generate a token: Home Assistant → Profile → Long-Lived Access Tokens → Create.

### 6. Run Jarvis

```powershell
python jarvis.py
```

---

## 🗣️ Voice Commands

### General

| Say | Result |
|---|---|
| `Hey Jarvis, what is the capital of France?` | Answers from LLM knowledge |
| `Hey Jarvis, open YouTube` | Opens youtube.com in Brave |
| `Hey Jarvis, open Notepad` | Launches Notepad |
| `Hey Jarvis, search for Python tutorials` | Opens Google search |
| `Hey Jarvis, start my day` | Opens dashboard + Obsidian + reads BTC update |

### Smart home (Romanian / English)

| Say | Result |
|---|---|
| `Hey Jarvis, aprinde lumina din bucatarie` | Turns on kitchen light |
| `Hey Jarvis, turn off all lights` | Turns off everything |
| `Hey Jarvis, dim the office to 30 percent` | Sets birou to 30% brightness |
| `Hey Jarvis, stinge dormitorul` | Turns off bedroom lights |
| `Hey Jarvis, toggle the hallway` | Toggles hol lights |

### Follow-up window
After a bare **"Hey Jarvis"** (no command), Jarvis listens for **8 seconds** for a follow-up — you don't need to say "Hey Jarvis" again.

---

## 🏠 Smart Home Rooms

| Voice name | Aliases | HA entities |
|---|---|---|
| `bucatarie` | kitchen | `light.bucatarie_bucatarie` |
| `dormitor` | bedroom | 2 bulbs |
| `hol` | hallway, baie, bathroom, intrare | 2 entities |
| `lampa` | lamp | `light.lampa_lampa` |
| `birou` | office, desk | `light.sufragerie_birou_*` |
| `sufragerie` | living, living room | 3 bulbs |
| `toate` | all, everything | all of the above |

---

## 📁 Project Structure

```
jarvis/
├── jarvis.py        # Main entry point — audio loop, wake-word, LLM dispatch
├── tools.py         # Tool implementations: open_url, open_app, search_web,
│                    #   chat_reply, start_my_day
├── ha_tools.py      # Home Assistant light control
├── tts.py           # Text-to-speech (Edge TTS + pygame)
├── transcript.txt   # Auto-generated live transcript
├── .env             # HA_URL + HA_TOKEN (not committed)
└── venv311/         # Python virtual environment
```

---

## ⚙️ Configuration

Key constants at the top of `jarvis.py`:

| Constant | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `"medium"` | Whisper model size (`tiny` → `large-v3`) |
| `LANGUAGE` | `"en"` | Transcription language |
| `OLLAMA_MODEL` | `"gemma4:e4b"` | LLM model via Ollama |
| `WAKE_PHRASES` | `["hey jarvis", ...]` | Wake-word variants |
| `WAKE_FUZZY` | `0.75` | Fuzzy match threshold (lower = more permissive) |
| `FOLLOWUP_WINDOW_S` | `8.0` | Seconds to wait for follow-up after bare wake word |
| `CAPTURE_BLOCK` | `8192` | Audio buffer size (larger = fewer overflows) |
| `MIC_DEVICE` | `None` | Microphone device index (`None` = system default) |

To change the TTS voice, edit `tts.py`:
```python
VOICE = "en-US-GuyNeural"   # or en-GB-RyanNeural, ro-RO-EmilNeural, etc.
```
Run `edge-tts --list-voices` for all available voices.

---

## ➕ Adding New Tools

1. Write a function in `tools.py` (or `ha_tools.py` for HA-related things)
2. Add its JSON schema to the `TOOLS` list
3. Register it in `DISPATCH`
4. Add an example to `SYSTEM_PROMPT` in `jarvis.py`

Example skeleton:
```python
# tools.py
def my_tool(param: str) -> str:
    # do something
    return "Done."

TOOLS.append({
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does and when to use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "..."},
            },
            "required": ["param"],
        },
    },
})

DISPATCH["my_tool"] = my_tool
```

---

## 🐛 Troubleshooting

**`⚠️ input overflow`**  
Audio buffer overflowing (usually when GPU is busy). Already fixed with `CAPTURE_BLOCK=8192` and `latency="high"`. If it still appears, try increasing `CAPTURE_BLOCK` to `16384`.

**`❌ Can't reach Ollama`**  
Run `ollama serve` in a separate terminal, or make sure the Ollama service is running.

**Jarvis doesn't react to wake word**  
- Check mic is working: audio levels in the transcript log
- Lower `WAKE_FUZZY` (e.g. `0.65`) to be more permissive
- Add your specific mishearing to `WAKE_PHRASES` in `jarvis.py`

**Light control fails**  
- Check `HA_TOKEN` is set in `.env`
- Verify Home Assistant is reachable at `HA_URL`
- Entity IDs are in `ha_tools.py` → `ROOMS` dict

**No sound from TTS**  
```powershell
pip install edge-tts pygame
```
Make sure speakers/headphones are set as default audio output in Windows.

---

## 🗺️ Roadmap

- [ ] **v2** — More PC commands (volume, window management, clipboard)
- [ ] **v3** — Conversation memory across sessions
- [ ] **v4** — Custom wake-word training
- [ ] **v5** — Home Assistant sensors read-out (temperature, presence, etc.)
