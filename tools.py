"""
Tools that Jarvis can call.

Each tool is a normal Python function. The TOOLS list at the bottom describes
them in the JSON-schema format Ollama expects for tool calling.

Add new tools by:
  1. Writing the function.
  2. Adding a JSON schema entry to TOOLS.
  3. Registering it in DISPATCH at the bottom.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path

import requests

# === Browser detection (Windows) ===

# Common Brave install paths on Windows. First one that exists wins.
# You can override by setting BRAVE_PATH environment variable, or by editing this list.
BRAVE_CANDIDATES = [
    # Standard installer locations
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "BraveSoftware/Brave-Browser/Application/brave.exe",
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "BraveSoftware/Brave-Browser/Application/brave.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware/Brave-Browser/Application/brave.exe",
    # Beta / Nightly / Dev channels
    Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware/Brave-Browser-Beta/Application/brave.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware/Brave-Browser-Nightly/Application/brave.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "BraveSoftware/Brave-Browser-Dev/Application/brave.exe",
    # Portable / unusual installs
    Path(r"C:/Users") / os.environ.get("USERNAME", "") / "AppData/Local/BraveSoftware/Brave-Browser/Application/brave.exe",
]


def _find_brave() -> str | None:
    # 1. Explicit env var override
    env_path = os.environ.get("BRAVE_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    # 2. Try known paths
    for p in BRAVE_CANDIDATES:
        if p.is_file():
            return str(p)
    # 3. PATH lookup
    return shutil.which("brave") or shutil.which("brave.exe")


def _open_in_browser(url: str, browser: str = "brave") -> str:
    """Open URL. Uses Brave if requested and found; otherwise default browser."""
    if browser.lower() == "brave":
        brave = _find_brave()
        if brave:
            subprocess.Popen([brave, url])
            return f"Opened {url} in Brave."
        # Brave requested but not found → fall back with a note
        webbrowser.open(url)
        return f"Brave not found; opened {url} in default browser."
    webbrowser.open(url)
    return f"Opened {url} in default browser."


# === Tool implementations ===

def open_url(url: str, browser: str = "brave") -> str:
    """Open an arbitrary URL. Adds https:// if missing."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return _open_in_browser(url, browser)


# Map of friendly app names → how to launch them on Windows.
# Extend freely. Values can be:
#   - a string: passed to `start` via shell (works for installed apps registered with Windows)
#   - a list: argv passed to subprocess.Popen directly
APP_LAUNCHERS: dict[str, str | list[str]] = {
    "notepad":  ["notepad.exe"],
    "calculator": ["calc.exe"],
    "calc":     ["calc.exe"],
    "explorer": ["explorer.exe"],
    "cmd":      ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "spotify":  "spotify:",                  # URI scheme — works if Spotify desktop is installed
    "obsidian": [str(Path(os.environ.get("LOCALAPPDATA", r"C:\Users\diobe\AppData\Local")) / "Programs" / "Obsidian" / "Obsidian.exe")],
    "word":     [r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE"],
    "excel":    [r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE"],
    "powerpoint": [r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE"],
    "outlook":  [r"C:\Program Files (x86)\Microsoft Office\root\Office16\OUTLOOK.EXE"],
    "vscode":   ["code"],                    # requires `code` on PATH (VS Code installer option)
    "code":     ["code"],
    "brave":    None,                        # resolved at call time
    "chrome":   ["chrome.exe"],
    "firefox":  ["firefox.exe"],
}


def open_app(name: str) -> str:
    """Open a local application by friendly name."""
    key = name.strip().lower()
    launcher = APP_LAUNCHERS.get(key)

    if key == "brave":
        brave = _find_brave()
        if not brave:
            return "Brave not found on this system."
        subprocess.Popen([brave])
        return "Opened Brave."

    if launcher is None:
        # Last resort: hand it to Windows `start` which will resolve from registry / Start menu
        try:
            subprocess.Popen(["cmd", "/c", "start", "", key], shell=False)
            return f"Asked Windows to open '{name}'."
        except Exception as e:
            return f"Couldn't open '{name}': {e}"

    try:
        if isinstance(launcher, list):
            subprocess.Popen(launcher)
        else:
            # URI scheme or shell string
            os.startfile(launcher) if not launcher.endswith(":") else subprocess.Popen(["cmd", "/c", "start", "", launcher])
        return f"Opened {name}."
    except FileNotFoundError:
        return f"'{name}' is not installed or not on PATH."
    except Exception as e:
        return f"Couldn't open '{name}': {e}"


def search_web(query: str, engine: str = "google") -> str:
    """Open a search results page in Brave."""
    q = urllib.parse.quote_plus(query)
    if engine.lower() == "youtube":
        url = f"https://www.youtube.com/results?search_query={q}"
    elif engine.lower() == "duckduckgo":
        url = f"https://duckduckgo.com/?q={q}"
    else:
        url = f"https://www.google.com/search?q={q}"
    return _open_in_browser(url, "brave")


def chat_reply(text: str) -> str:
    """No action — just relay a conversational answer back to the user."""
    return text


DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:3001/")


def start_my_day() -> str:
    """Morning routine: open dashboard + Obsidian, then speak Bitcoin 24h summary."""
    # These are fire-and-forget (subprocess.Popen under the hood) — both launch immediately
    open_url(DASHBOARD_URL)
    open_app("obsidian")

    # Fetch Bitcoin price from CoinGecko (free public API, no key needed)
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "bitcoin",
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
            timeout=8,
        )
        resp.raise_for_status()
        btc = resp.json()["bitcoin"]
        price = btc["usd"]
        change = btc.get("usd_24h_change", 0.0)
        direction = "up" if change >= 0 else "down"
        return (
            f"Good morning! Opening your dashboard and Obsidian. "
            f"Bitcoin is at ${price:,.0f}, "
            f"{direction} {abs(change):.1f}% in the last 24 hours."
        )
    except Exception as e:
        return (
            f"Good morning! Opening your dashboard and Obsidian. "
            f"Couldn't fetch Bitcoin data: {e}"
        )


# === Schema for Ollama tool calling ===

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a website URL in the user's browser (Brave by default).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL or domain, e.g. 'youtube.com' or 'https://github.com'"},
                    "browser": {"type": "string", "enum": ["brave", "default"], "default": "brave"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open a local desktop application by name (e.g. 'notepad', 'spotify', 'vscode', 'calculator').",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Friendly app name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web and open the results page. Use engine='youtube' for YouTube searches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "engine": {"type": "string", "enum": ["google", "youtube", "duckduckgo"], "default": "google"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_reply",
            "description": "Use this when the user just wants to chat or get information that doesn't require an action. Put your reply in the 'text' field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Your conversational reply to the user."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_my_day",
            "description": (
                "Morning routine: opens the user's personal dashboard and Obsidian, "
                "then reads out Bitcoin's 24-hour price performance. "
                "Use when the user says 'start my day', 'morning routine', 'good morning', or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


DISPATCH = {
    "open_url":      open_url,
    "open_app":      open_app,
    "search_web":    search_web,
    "chat_reply":    chat_reply,
    "start_my_day":  start_my_day,
}


def run_tool(name: str, args: dict) -> str:
    fn = DISPATCH.get(name)
    if fn is None:
        return f"[Unknown tool: {name}]"
    try:
        return fn(**args)
    except TypeError as e:
        return f"[Bad arguments for {name}: {e}]"
    except Exception as e:
        return f"[Error in {name}: {e}]"