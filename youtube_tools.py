"""
YouTube Music controls for Jarvis.

Tools:
  play_music(query)  - search YT Music and open the direct watch URL (auto-plays)
  pause_music()      - toggle pause via 'k' shortcut
  resume_music()     - toggle resume via 'k' shortcut
  next_track()       - Shift+N
  previous_track()   - Shift+P
  stop_music()       - pause via 'k' (no true "stop" in browser YT Music)
"""

from __future__ import annotations

import time
import urllib.parse

from tools import _open_in_browser

YTMUSIC_BASE = "https://music.youtube.com"


def _focus_ytmusic() -> bool:
    """Bring a Brave window whose title contains 'YouTube Music' to the foreground."""
    try:
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle("YouTube Music")
        if not windows:
            return False
        win = windows[0]
        win.restore()
        win.activate()
        time.sleep(0.35)
        return True
    except Exception:
        return False


def _send_key(key: str, shift: bool = False) -> bool:
    if not _focus_ytmusic():
        return False
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        if shift:
            pyautogui.hotkey("shift", key)
        else:
            pyautogui.press(key)
        return True
    except Exception:
        return False


# === Tool implementations ===

def play_music(query: str) -> str:
    """Search YouTube Music and open the direct watch URL for the first song result."""
    try:
        from ytmusicapi import YTMusic
        yt = YTMusic()
        results = yt.search(query, filter="songs", limit=5)
        for hit in results:
            video_id = hit.get("videoId")
            if video_id:
                url = f"{YTMUSIC_BASE}/watch?v={video_id}"
                _open_in_browser(url, "brave")
                title = hit.get("title", query)
                artists = ", ".join(a["name"] for a in hit.get("artists", []))
                desc = f"{title} by {artists}" if artists else title
                return f"Playing {desc} on YouTube Music."
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: open search page
    q = urllib.parse.quote_plus(query)
    url = f"{YTMUSIC_BASE}/search?q={q}&filter=songs"
    _open_in_browser(url, "brave")
    return f"Opened YouTube Music search for '{query}'."


def pause_music() -> str:
    if _send_key("k"):
        return "Paused."
    return "Couldn't find a YouTube Music window — is it open in Brave?"


def resume_music() -> str:
    if _send_key("k"):
        return "Resumed."
    return "Couldn't find a YouTube Music window — is it open in Brave?"


def next_track() -> str:
    if _send_key("n", shift=True):
        return "Playing next track."
    return "Couldn't find a YouTube Music window — is it open in Brave?"


def previous_track() -> str:
    if _send_key("p", shift=True):
        return "Playing previous track."
    return "Couldn't find a YouTube Music window — is it open in Brave?"


def stop_music() -> str:
    if _send_key("k"):
        return "Stopped."
    return "Couldn't find a YouTube Music window — is it open in Brave?"


# === Schema for Ollama tool calling ===

YOUTUBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Search YouTube Music for a song or artist and start playing it. "
                "Use when the user says 'play [song/artist]', 'pune [piesa/artist]', "
                "'asculta [piesa]', or any variant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song title, artist name, or both (e.g. 'Bohemian Rhapsody', 'Eminem', 'Lose Yourself Eminem').",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_music",
            "description": "Pause YouTube Music playback. Use when user says 'pause', 'opreste muzica', 'stop playing'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_music",
            "description": "Resume YouTube Music playback. Use when user says 'resume', 'unpause', 'continua muzica', 'play again'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_track",
            "description": "Skip to the next song on YouTube Music. Use when user says 'next', 'next song', 'urmatoarea', 'skip'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "previous_track",
            "description": "Go back to the previous song on YouTube Music. Use when user says 'previous', 'back', 'piesa anterioara', 'inapoi'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_music",
            "description": "Stop YouTube Music playback entirely. Use when user says 'stop music', 'stop the music', 'opreste tot'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

YOUTUBE_DISPATCH = {
    "play_music":      play_music,
    "pause_music":     pause_music,
    "resume_music":    resume_music,
    "next_track":      next_track,
    "previous_track":  previous_track,
    "stop_music":      stop_music,
}
