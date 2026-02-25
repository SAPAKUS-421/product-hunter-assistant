import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

APP_NAME = "Product Hunter Assistant (USA Market)"
REQUEST_URL = "https://api.rainforestapi.com/request"
CATEGORIES_URL = "https://api.rainforestapi.com/categories"

STORAGE_DIR = Path(os.environ.get("PRODUCT_HUNTER_STORAGE", "/tmp/product_hunter_assistant"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STORAGE_DIR / "state.json"
CACHE_DIR = STORAGE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Risk / restriction keyword filters (exclude these from results) ---
RISK_KEYWORDS = [
    # Supplements / ingestibles
    "supplement", "vitamin", "capsule", "softgel", "tablet", "gummy", "probiotic", "creatine",
    "protein powder", "weight loss", "keto", "detox", "herbal", "dietary",

    # Medical / tests / healthcare
    "blood test", "test kit", "diagnostic", "medical", "medicine", "pharmacy", "clinical",
    "covid", "glucose", "insulin", "thermometer", "bp monitor", "pregnancy test",

    # Hazmat-ish / chemicals / flammables
    "aerosol", "spray can", "propane", "butane", "fuel", "flammable", "corrosive",
    "acid", "bleach", "solvent", "paint thinner", "hazmat", "toxic",

    # Weapon-like
    "knife", "dagger", "sword", "machete", "weapon", "ammo", "munition", "gun",

    # Brand-heavy cue words (not perfect, but helps)
    "yeti", "vtech", "fellowes", "carlyle", "nature made", "rit dye", "wonka",
]
