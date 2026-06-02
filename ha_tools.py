"""
Home Assistant integration for Jarvis.

Setup:
  Create a `.env` file in this folder with:
    HA_URL=http://100.118.165.65:8123
    HA_TOKEN=your_long_lived_access_token_here

  Or set them as PowerShell env vars before running:
    $env:HA_TOKEN = "..."
"""

from __future__ import annotations

import os
from typing import Any

import requests

# Load .env if present (no-op if python-dotenv isn't installed or .env missing).
# This makes the module work standalone, not only when imported from jarvis.py.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# === CONFIG ===
HA_URL   = os.environ.get("HA_URL",   "http://homeassistant.local:8123")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
HA_TIMEOUT = 5  # seconds


# === ROOM STRUCTURE ===
# Logical rooms -> entity IDs from your HA setup.
# Multiple entities = controlled together.
ROOMS: dict[str, list[str]] = {
    "bucatarie": [
        "light.bucatarie_bucatarie",
    ],
    "dormitor": [
        "light.dormitor_1_dormitor_1",
        "light.dormitor_1_dormitor_1_2",
    ],
    "hol": [
        "light.hol_baie_hol_baie",
        "light.hol_intrare_hol_intrare",
    ],
    "lampa": [
        "light.lampa_lampa",
    ],
    "birou": [
        "light.sufragerie_birou_sufragerie_birou",
    ],
    "sufragerie": [
        "light.sufragerie_1_sufragerie_1",
        "light.sufragerie_2_sufragerie_birou",
        "light.sufragerie_3_sufragerie_3",
    ],
}

# All lights = every entity from every room
ALL_LIGHT_ENTITIES = [eid for entities in ROOMS.values() for eid in entities]


# Voice aliases -> canonical room name.
ALIASES: dict[str, str] = {
    # Bucatarie
    "bucatarie": "bucatarie", "bucătărie": "bucatarie",
    "kitchen": "bucatarie",

    # Dormitor
    "dormitor": "dormitor", "dormitorul": "dormitor",
    "bedroom": "dormitor",

    # Hol (single concept: includes baie + intrare entities)
    "hol": "hol", "holul": "hol",
    "baie": "hol", "baia": "hol",
    "intrare": "hol", "intrarea": "hol",
    "hallway": "hol", "bathroom": "hol", "entrance": "hol",

    # Lampa
    "lampa": "lampa", "lamp": "lampa",

    # Birou
    "birou": "birou", "biroul": "birou",
    "office": "birou", "desk": "birou",

    # Sufragerie
    "sufragerie": "sufragerie", "sufrageria": "sufragerie",
    "living": "sufragerie", "living room": "sufragerie",
}

# Special group: all lights
ALL_ALIASES = {
    "all", "all lights", "everything",
    "toate", "toate luminile", "tot", "totul",
}


# === HA REST API ===

def _headers() -> dict:
    if not HA_TOKEN:
        raise RuntimeError("HA_TOKEN environment variable is not set.")
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _call_service(domain: str, service: str, data: dict) -> bool:
    url = f"{HA_URL}/api/services/{domain}/{service}"
    try:
        resp = requests.post(url, headers=_headers(), json=data, timeout=HA_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️  HA request failed: {e}")
        return False


def _resolve(name: str) -> tuple[str, list[str]]:
    """
    Resolve a user-provided room name to (canonical_name, entity_ids).
    Returns ('', []) if unknown.
    """
    key = name.strip().lower()

    # 1. All-lights special case
    if key in ALL_ALIASES:
        return ("toate luminile", ALL_LIGHT_ENTITIES)

    # 2. Direct alias hit
    if key in ALIASES:
        canonical = ALIASES[key]
        return (canonical, ROOMS[canonical])

    # 3. Substring match (handles "in bucatarie", "din dormitor", etc.)
    for alias, canonical in ALIASES.items():
        if alias in key or key in alias:
            return (canonical, ROOMS[canonical])

    return ("", [])


# === Tool implementation ===

def control_light(name: str, action: str, brightness: int | None = None) -> str:
    """Control a light by friendly room name."""
    canonical, entities = _resolve(name)
    if not entities:
        return (
            f"I don't know a room called '{name}'. "
            f"Try: bucatarie, dormitor, hol, lampa, birou, sufragerie, or toate."
        )

    action = action.lower().strip()
    if action not in ("on", "off", "toggle"):
        return f"Unknown action '{action}'. Use on, off, or toggle."

    service = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}[action]
    data: dict[str, Any] = {"entity_id": entities}

    if action == "on" and brightness is not None:
        data["brightness_pct"] = max(1, min(100, brightness))

    ok = _call_service("light", service, data)
    if not ok:
        return f"Couldn't reach Home Assistant at {HA_URL}."

    count = len(entities)
    target = canonical + (f" ({count} lights)" if count > 1 else "")
    if action == "on" and brightness is not None:
        return f"Turned on {target} at {brightness}%."
    if action == "on":
        return f"Turned on {target}."
    if action == "off":
        return f"Turned off {target}."
    return f"Toggled {target}."


# === Tool schema for the LLM ===

HA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "control_light",
            "description": (
                "Turn lights on or off, or set brightness, in a specific room. "
                "Rooms (use these exact names): "
                "bucatarie (kitchen), dormitor (bedroom - has 2 bulbs), "
                "hol (hallway - includes bathroom area), lampa (a single lamp), "
                "birou (office), sufragerie (living room - has 3 bulbs), "
                "or 'toate' / 'all' for all lights at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Room name. One of: bucatarie, dormitor, hol, lampa, birou, sufragerie, toate.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "toggle"],
                        "description": "What to do with the light.",
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Optional brightness 1-100 (percent). Only with action='on'.",
                    },
                },
                "required": ["name", "action"],
            },
        },
    },
]


HA_DISPATCH = {
    "control_light": control_light,
}