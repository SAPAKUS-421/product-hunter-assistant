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

# Broad heuristics (auto-exclude these during scanning)
RISK_KEYWORDS = {
    "brand-heavy": [
        "apple", "samsung", "sony", "nintendo", "disney", "lego", "dyson", "yeti", "stanley", "kitchenaid",
        "keurig", "breville", "bosch", "dewalt", "milwaukee", "makita",
    ],
    "supplement": [
        "supplement", "vitamin", "capsule", "tablet", "softgel", "gummies", "omega", "probiotic", "creatine",
        "whey", "collagen", "ashwagandha",
    ],
    "medical/test": [
        "test kit", "diagnostic", "glucose", "blood pressure", "bp monitor", "pulse oximeter", "covid", "pregnancy",
    ],
    "hazmat-ish": [
        "aerosol", "propane", "butane", "fuel", "flammable", "corrosive", "acid", "ble
