import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Product Hunter Assistant (USA Market)",
    page_icon="🧭",
    layout="wide",
)


# =========================
# Secrets helper
# =========================
def get_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        import os

        return str(os.getenv(key, default)).strip()


RAINFOREST_API_KEY = get_secret("RAINFOREST_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")  # optional
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


# =========================
# Constants
# =========================
RAINFOREST_BASE = "https://api.rainforestapi.com"

DEFAULT_AMAZON_DOMAIN = "amazon.com"


# Risk filters (exclude by default)
# Keep this as simple keyword gates; it’s intentionally conservative.
RISK_KEYWORDS = [
    # Supplements / health claims
    "supplement",
    "vitamin",
    "creatine",
    "whey",
    "protein powder",
    "fat burner",
    "weight loss",
    "keto",
    "collagen",
    "testosterone",
    "male enhancement",
    "libido",
    "detox",
    "cleanse",
    "probiotic",
    "pre workout",
    "pre-workout",
    "bcaa",
    # Medical / tests / devices
    "medical",
    "medicine",
    "drug",
    "diagnostic",
    "test kit",
    "rapid test",
    "covid",
    "glucose",
    "blood",
    "bp monitor",
    "blood pressure",
    "thermometer",
    "stethoscope",
    "pregnancy test",
    # Hazmat-ish / chemicals / flammables
    "flammable",
    "corrosive",
    "acid",
    "aerosol",
    "propane",
    "butane",
    "fuel",
    "solvent",
    "bleach",
    "pesticide",
    "insecticide",
    "herbicide",
    "toxic",
    # Weapon-like
    "knife",
    "dagger",
    "machete",
    "sword",
    "switchblade",
    "tactical",
    "ammo",
    "gun",
    "firearm",
    "holster",
    "crossbow",
    "arrowhead",
    "pepper spray",
    # Brand-heavy signals (not perfect, but reduces brand/IP risk)
    "disney",
    "pokemon",
    "nintendo",
    "marvel",
    "star wars",
    "lego",
    "hello kitty",
]


# =========================
# Session state init
# =========================
def init_state() -> None:
    st.session_state.setdefault("requests_used", 0)
    st.session_state.setdefault("api_cache", {})  # key -> (timestamp, json)
    st.session_state.setdefault("category_results", [])  # last categories search
    st.session_state.setdefault("selected_category_obj", None)
    st.session_state.setdefault("my_shortlist", [])  # list of dict rows


init_state()


# =========================
# Utilities
# =========================
def norm_text(s: str) -> str:
    return (s or "").strip().lower()


def extract_price_value(p: Any) -> Optional[float]:
    """
    Rainforest often returns:
      price: { "value": 24.99, "currency": "USD", ... }
    or sometimes "price": 24.99
    """
    if p is None:
        return None
    if isinstance(p, (int, float)):
        return float(p)
    if isinstance(p, dict):
        v = p.get("value")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    return None


def amazon_dp_url(amazon_domain: str, asin: str) -> str:
    return f"https://www.{amazon_domain}/dp/{asin}"


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flag_risk(title: str, brand: str = "") -> List[str]:
    t = norm_text(title)
    b = norm_text(brand)
    hits = []
    for kw in RISK_KEYWORDS:
        kw_n = norm_text(kw)
        if kw_n and (kw_n in t or kw_n in b):
            hits.append(kw)
    return hits


def cache_key(url: str, params: Dict[str, Any]) -> str:
    # stable cache key: url + sorted params
    parts = [url] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "||".join(parts)

def redact_api_key(text: str) -> str:
    """Redact api_key=... inside any URL/text."""
    if not text:
        return text
    return re.sub(r"(api_key=)([^&\s]+)", r"\1****REDACTED****", text)


def safe_cache_key(url: str, params: Dict[str, Any]) -> str:
    """Cache key without storing sensitive api_key value."""
    safe_params = dict(params)
    if "api_key" in safe_params:
        safe_params["api_key"] = "REDACTED"
    parts = [url] + [f"{k}={safe_params[k]}" for k in sorted(safe_params.keys())]
    return "||".join(parts)
    
def rainforest_get_json(
    endpoint: str,
    params: Dict[str, Any],
    force_refresh: bool,
    count_request: bool = True,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Small in-session cache to avoid wasting trial requests.
    If force_refresh is False and cached result exists, returns cached (no request count).
    """
    url = f"{RAINFOREST_BASE}{endpoint}"
    params2 = dict(params)

    if "api_key" not in params2:
        params2["api_key"] = RAINFOREST_API_KEY

    key = safe_cache_key(url, params2)
    if not force_refresh and key in st.session_state["api_cache"]:
        _, cached = st.session_state["api_cache"][key]
        return cached

    def rainforest_get_json(
    endpoint: str,
    params: Dict[str, Any],
    force_refresh: bool,
    count_request: bool = True,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Safe Rainforest caller:
    - Never leaks api_key in error messages
    - Detects 'temporarily suspended' messages and shows a clean instruction
    - Uses in-session cache without storing real api_key in cache key
    """
    url = f"{RAINFOREST_BASE}{endpoint}"
    params2 = dict(params)

    if "api_key" not in params2:
        params2["api_key"] = RAINFOREST_API_KEY

    # Use SAFE cache key (redacts api_key)
    key = safe_cache_key(url, params2)
    if not force_refresh and key in st.session_state["api_cache"]:
        _, cached = st.session_state["api_cache"][key]
        return cached

    try:
    r = requests.get(url, params=params2, timeout=timeout)
except requests.RequestException as e:
    raise RuntimeError(f"Network error contacting Rainforest: {e.__class__.__name__}")

# Try to parse JSON even on HTTP errors
data = None
try:
    data = r.json()
except Exception:
    data = None

# If Rainforest returns structured request_info, use it
if isinstance(data, dict) and "request_info" in data:
    ri = data.get("request_info") or {}
    success = ri.get("success", True)
    msg = str(ri.get("message") or "").strip()

    if success is False:
        lower = msg.lower()

        if "temporarily suspended" in lower:
            raise RuntimeError(
                "Rainforest says your account is temporarily suspended. "
                "Open Rainforest Dashboard → API Playground to confirm, then contact support to reinstate."
            )

        # Generic structured error
        raise RuntimeError(f"Rainforest error: {redact_api_key(msg)}")

# If still an HTTP error and no structured JSON, raise a redacted URL error
if r.status_code >= 400:
    raise RuntimeError(f"Rainforest HTTP {r.status_code}. URL: {redact_api_key(r.url)}")

# Success path
if not isinstance(data, dict):
    data = {}

st.session_state["api_cache"][key] = (time.time(), data)
if count_request:
    st.session_state["requests_used"] += 1
return data

def estimate_requests_scan(
    scan_mode: str,
    list_pages: int,
    deep_dive_n: int,
    include_summarization: bool,
) -> int:
    # Bestsellers/search pages are 1 request each
    est = max(1, list_pages)

    if scan_mode == "Deep dive":
        # product request is ~1 each; if include_summarization_attributes is enabled,
        # Rainforest docs indicate it costs extra credits (commonly 2 total for that request).
        # We conservatively estimate 2 per product when enabled.
        per_product = 2 if include_summarization else 1
        est += deep_dive_n * per_product

    return est


# =========================
# Rainforest adapters
# =========================
def categories_search_bestsellers(
    amazon_domain: str,
    search_term: str,
    force_refresh: bool,
) -> List[Dict[str, Any]]:
    """
    Categories API (bestsellers).
    Endpoint commonly: /categories
    Params per TrajectData docs: type=bestsellers, amazon_domain, search_term
    """
    if not search_term.strip():
        return []

    data = rainforest_get_json(
        endpoint="/categories",
        params={
            "type": "bestsellers",
            "amazon_domain": amazon_domain,
            "search_term": search_term.strip(),
        },
        force_refresh=force_refresh,
        count_request=True,
    )

    cats = data.get("categories") or []
    # ensure dicts
    return [c for c in cats if isinstance(c, dict)]


def build_list_url_from_category(cat_obj: Optional[Dict[str, Any]], amazon_domain: str) -> Optional[str]:
    """
    Prefer the url from categories response if present.
    Otherwise, try to build from category_id like "bestsellers_17871150011".
    """
    if cat_obj and isinstance(cat_obj, dict):
        u = cat_obj.get("url")
        if isinstance(u, str) and u.startswith("http"):
            return u

        cid = cat_obj.get("id") or cat_obj.get("category_id")
        if isinstance(cid, str) and "bestsellers_" in cid:
            node = cid.split("bestsellers_", 1)[1]
            node = re.sub(r"[^\d]", "", node)  # digits only
            if node:
                return f"https://www.{amazon_domain}/Best-Sellers/zgbs/?node={node}"

    return None


def list_url_for_list_type(base_url: Optional[str], amazon_domain: str, list_type: str, fallback_node: Optional[str]) -> Optional[str]:
    """
    Converts Best Sellers category URL into other bestseller-type pages using node parameter.
    This is more robust than trying to rewrite paths.

    list_type options:
      - Best Sellers
      - Trending (Movers & Shakers)
      - New Releases
      - Most Wished For
      - Most Gifted
    """
    list_type = list_type.strip()

    node = fallback_node
    if not node and base_url:
        # try to extract node=12345 from url
        m = re.search(r"node=(\d+)", base_url)
        if m:
            node = m.group(1)

    # If we have a base_url, Best Sellers can just use it.
    if list_type == "Best Sellers":
        return base_url

    # Other lists: prefer node-based URLs (works even if base_url is weird)
    if not node:
        return None

    if list_type == "Trending (Movers & Shakers)":
        return f"https://www.{amazon_domain}/gp/movers-and-shakers/?node={node}"
    if list_type == "New Releases":
        return f"https://www.{amazon_domain}/gp/new-releases/?node={node}"
    if list_type == "Most Wished For":
        return f"https://www.{amazon_domain}/gp/most-wished-for/?node={node}"
    if list_type == "Most Gifted":
        return f"https://www.{amazon_domain}/gp/most-gifted/?node={node}"

    return base_url


def fetch_bestsellers(
    amazon_domain: str,
    url: str,
    page: int,
    force_refresh: bool,
) -> List[Dict[str, Any]]:
    """
    Product Data API: type=bestsellers with url & page
    Endpoint: /request
    """
    data = rainforest_get_json(
        endpoint="/request",
        params={
            "type": "bestsellers",
            "amazon_domain": amazon_domain,
            "url": url,
            "page": page,
        },
        force_refresh=force_refresh,
        count_request=True,
    )
    items = data.get("bestsellers") or []
    return [x for x in items if isinstance(x, dict)]


def fetch_search(
    amazon_domain: str,
    search_term: str,
    page: int,
    force_refresh: bool,
) -> List[Dict[str, Any]]:
    """
    Product Data API: type=search with search_term & page
    Endpoint: /request
    """
    data = rainforest_get_json(
        endpoint="/request",
        params={
            "type": "search",
            "amazon_domain": amazon_domain,
            "search_term": search_term.strip(),
            "page": page,
        },
        force_refresh=force_refresh,
        count_request=True,
    )
    items = data.get("search_results") or []
    return [x for x in items if isinstance(x, dict)]


def fetch_product(
    amazon_domain: str,
    asin: str,
    include_summarization_attributes: bool,
    force_refresh: bool,
) -> Dict[str, Any]:
    """
    Product Data API: type=product with asin
    """
    params = {
        "type": "product",
        "amazon_domain": amazon_domain,
        "asin": asin,
    }
    if include_summarization_attributes:
        params["include_summarization_attributes"] = "true"

    data = rainforest_get_json(
        endpoint="/request",
        params=params,
        force_refresh=force_refresh,
        count_request=True,
    )
    return data if isinstance(data, dict) else {}


# =========================
# Scoring / ranking
# =========================
@dataclass
class ProfitConfig:
    fee_pct: float
    fixed_fee: float
    shipping: float
    packaging: float
    sourcing_discount_pct: float
    profit_goal: float
    hide_below_goal: bool
    target_price_min: Optional[float]
    target_price_max: Optional[float]


def estimate_profitability(sell_price: Optional[float], cfg: ProfitConfig) -> Dict[str, Optional[float]]:
    if sell_price is None or sell_price <= 0:
        return {
            "sell_price_used": None,
            "buy_price_est": None,
            "fees_est": None,
            "profit_est": None,
            "roi_est": None,
            "max_buy_price": None,
        }

    # Optionally clamp sell price into a user range (ranking only)
    if cfg.target_price_min is not None:
        sell_price = max(sell_price, cfg.target_price_min)
    if cfg.target_price_max is not None:
        sell_price = min(sell_price, cfg.target_price_max)

    fees = (cfg.fee_pct / 100.0) * sell_price + cfg.fixed_fee
    buy_price_est = sell_price * (1.0 - cfg.sourcing_discount_pct / 100.0)
    profit = sell_price - fees - cfg.shipping - cfg.packaging - buy_price_est
    roi = None
    if buy_price_est and buy_price_est > 0:
        roi = profit / buy_price_est

    max_buy = sell_price - fees - cfg.shipping - cfg.packaging - cfg.profit_goal

    return {
        "sell_price_used": round(sell_price, 2),
        "buy_price_est": round(buy_price_est, 2),
        "fees_est": round(fees, 2),
        "profit_est": round(profit, 2),
        "roi_est": round(roi, 3) if roi is not None else None,
        "max_buy_price": round(max_buy, 2),
    }


def overall_score(row: Dict[str, Any]) -> float:
    """
    Simple, practical score:
      - Profit (bigger is better)
      - Demand (ratings_total + rating)
      - Risk penalty if any
    """
    profit = row.get("profit_est")
    rating = row.get("rating") or 0
    ratings_total = row.get("ratings_total") or 0
    risk_count = len(row.get("risk_flags") or [])

    # profit component
    p = 0.0
    try:
        p = float(profit) if profit is not None else 0.0
    except Exception:
        p = 0.0

    # demand component (log)
    d = 0.0
    try:
        d = math.log10(max(1, int(ratings_total))) + (float(rating) / 5.0)
    except Exception:
        d = 0.0

    # risk penalty
    penalty = 0.0
    if risk_count > 0:
        penalty = 2.0 + 0.5 * risk_count

    return round((0.60 * p) + (0.40 * d) - penalty, 3)


# =========================
# Sidebar UI
# =========================
st.sidebar.title("🧭 Controls")

st.sidebar.markdown("### A) Platform")
amazon_domain = st.sidebar.selectbox(
    "Amazon domain",
    options=["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de", "amazon.fr", "amazon.it", "amazon.es"],
    index=0,
)

force_refresh = st.sidebar.checkbox("Force refresh (bypass cache)", value=False)

st.sidebar.markdown("### B) Scan Type")
scan_source = st.sidebar.selectbox(
    "Scan source",
    options=[
        "Category lists (recommended)",
        "Keyword search (optional)",
    ],
    index=0,
)

list_type = st.sidebar.selectbox(
    "List type",
    options=[
        "Best Sellers",
        "Trending (Movers & Shakers)",
        "New Releases",
        "Most Wished For",
        "Most Gifted",
    ],
    index=0,
    help="These are all Amazon’s top-selling style lists. Trending ≈ Movers & Shakers.",
)

st.sidebar.markdown("### C) Category Finder (recommended)")
cat_search_term = st.sidebar.text_input(
    "Search term for categories",
    value=st.session_state.get("cat_search_term", "Home & Kitchen"),
    placeholder="Example: Home & Kitchen / Pet Supplies / Tools / Sports / Leather",
)
st.session_state["cat_search_term"] = cat_search_term

col_c1, col_c2 = st.sidebar.columns([1, 1])
with col_c1:
    do_cat_search = st.button("Search categories")
with col_c2:
    # keep the checkbox here for clarity; used above in API cache bypass as well
    st.caption("Costs ~1 request when you search")

category_options: List[Dict[str, Any]] = st.session_state.get("category_results", [])

cat_error_box = st.sidebar.empty()

if do_cat_search:
    if not RAINFOREST_API_KEY:
        cat_error_box.error("Missing RAINFOREST_API_KEY in Streamlit Secrets.")
    else:
        try:
            cats = categories_search_bestsellers(amazon_domain, cat_search_term, force_refresh=force_refresh)
            if not cats:
                st.session_state["category_results"] = []
                cat_error_box.warning("No categories found. Try a different search term.")
            else:
                st.session_state["category_results"] = cats
                category_options = cats
                cat_error_box.success(f"Found {len(cats)} category matches.")
        except Exception as e:
            st.session_state["category_results"] = []
            cat_error_box.error(f"Category search failed: {redact_api_key(str(e))}")

# Build selectbox options safely (show full id so duplicates don’t look identical)
def cat_label(c: Dict[str, Any]) -> str:
    name = c.get("name") or c.get("title") or "Unknown"
    cid = c.get("id") or c.get("category_id") or "no_id"
    return f"{name} | {cid}"

selected_cat = None
if category_options:
    labels = [cat_label(c) for c in category_options]
    picked = st.sidebar.selectbox(
        "Pick a category",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        index=0,
        key="picked_category_index",
    )
    selected_cat = category_options[int(picked)]
    st.session_state["selected_category_obj"] = selected_cat

    # show copyable category id + URL
    selected_id = selected_cat.get("id") or selected_cat.get("category_id") or ""
    selected_url = selected_cat.get("url") or ""

    st.sidebar.markdown("**Selected category id (copyable)**")
    st.sidebar.code(str(selected_id) if selected_id else "—")

    if selected_url:
        st.sidebar.markdown("**Selected list URL (copyable)**")
        st.sidebar.code(str(selected_url))

else:
    st.sidebar.info("Use **Search categories** to find the correct bestsellers category list for your niche.")

st.sidebar.markdown("### D) Product keyword (optional)")
product_keyword = st.sidebar.text_input(
    "Keyword (only used for Keyword search scan)",
    value="",
    placeholder="Example: kitchen organizer / microfiber cloth / dog toy",
)

st.sidebar.markdown("### E) Scan size + request planning")
scan_items = st.sidebar.slider("How many items to scan", min_value=10, max_value=100, value=30, step=10)

scan_mode = st.sidebar.radio(
    "Scan mode",
    options=["Fast scan", "Deep dive"],
    index=0,
    help="Fast scan uses list pages only. Deep dive fetches extra product details for your shortlist candidates.",
)

deep_dive_n = 0
include_summarization = False
if scan_mode == "Deep dive":
    deep_dive_n = st.sidebar.slider("Deep dive on top N candidates", min_value=3, max_value=15, value=6, step=1)
    include_summarization = st.sidebar.checkbox(
        "Include Amazon 'customers say' summarization attributes (costs extra Rainforest credits)",
        value=False,
    )

st.sidebar.markdown("### F) Target selling price range (ranking-only)")
use_price_range = st.sidebar.checkbox("Use a target selling price range", value=False)
target_min = target_max = None
if use_price_range:
    cpr1, cpr2 = st.sidebar.columns(2)
    with cpr1:
        target_min = st.number_input("Min ($)", min_value=0.0, value=15.0, step=1.0)
    with cpr2:
        target_max = st.number_input("Max ($)", min_value=0.0, value=35.0, step=1.0)
    if target_max and target_min and target_max < target_min:
        st.sidebar.warning("Max should be ≥ Min. (Ranking only, but still.)")

st.sidebar.markdown("### G) Profitability assumptions (ranking-only)")
fee_pct = st.sidebar.slider("Marketplace fee %", min_value=0.0, max_value=25.0, value=15.0, step=0.5)
fixed_fee = st.sidebar.number_input("Fixed fee per order ($)", min_value=0.0, value=0.30, step=0.05)
shipping_cost = st.sidebar.number_input("Shipping cost ($)", min_value=0.0, value=4.50, step=0.25)
packaging_cost = st.sidebar.number_input("Packaging cost ($)", min_value=0.0, value=0.50, step=0.25)
sourcing_discount = st.sidebar.slider(
    "Expected sourcing discount vs selling price (%)",
    min_value=0.0,
    max_value=80.0,
    value=35.0,
    step=1.0,
    help="This is NOT a scan filter. It just estimates buy cost for ranking.",
)
profit_goal = st.sidebar.number_input(
    "Desired profit per unit ($) (goal)",
    min_value=0.0,
    value=8.00,
    step=0.5,
    help="This is NOT a scan filter. It’s used for ranking + optional hiding below goal.",
)
hide_below_goal = st.sidebar.checkbox("Hide items below profit goal", value=False)

st.sidebar.markdown("### H) Risk filters")
exclude_risky = st.sidebar.checkbox(
    "Exclude risk-flagged items (brand-heavy / supplement / medical/test / hazmat / weapon-like)",
    value=True,
)

st.sidebar.markdown("### Keys status")
st.sidebar.write(("✅" if RAINFOREST_API_KEY else "❌") + " RAINFOREST_API_KEY")
st.sidebar.write(("✅" if OPENAI_API_KEY else "⚪") + " OPENAI_API_KEY (optional)")
st.sidebar.caption(f"Model (optional): {OPENAI_MODEL}")

# Request estimation
# Pages: assume ~50 items per page
items_per_page = 50
list_pages = max(1, math.ceil(scan_items / items_per_page))
est_requests = estimate_requests_scan(
    scan_mode=scan_mode,
    list_pages=list_pages,
    deep_dive_n=deep_dive_n,
    include_summarization=include_summarization,
)
st.sidebar.info(
    f"Estimated requests this scan may cost: **~{est_requests}**  \n"
    f"Requests used this session: **{st.session_state['requests_used']}**"
)


# =========================
# Main UI
# =========================
st.title("🧭 Product Hunter Assistant (USA Market)")
st.caption(
    "This app pulls Amazon category list pages via Rainforest, ranks opportunities (profitability is ranking-only), "
    "excludes risky items by default, and generates sourcing leads + eBay sold-price workflows."
)

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_scan = st.button("🚀 Run scan", type="primary")
with run_col2:
    st.write("")

tabs = st.tabs(["✅ Shortlist candidates", "⭐ My Shortlist", "🧰 Workflows & Lead Gen"])


# =========================
# Scan execution
# =========================
def normalize_items(raw_items: List[Dict[str, Any]], amazon_domain: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for it in raw_items:
        asin = it.get("asin") or it.get("product_asin") or it.get("parent_asin")
        title = it.get("title") or it.get("name") or ""
        brand = it.get("brand") or ""
        rank = it.get("rank")
        rating = it.get("rating")
        ratings_total = it.get("ratings_total") or it.get("ratings_total_count") or it.get("reviews_total") or 0
        price = extract_price_value(it.get("price"))
        if price is None:
            price = extract_price_value(it.get("price_current"))

        risk_flags = flag_risk(title, brand)

        rows.append(
            {
                "title": title,
                "asin": asin,
                "brand": brand,
                "rank": rank,
                "rating": rating,
                "ratings_total": ratings_total,
                "amazon_price": price,
                "amazon_url": amazon_dp_url(amazon_domain, asin) if asin else "",
                "risk_flags": risk_flags,
            }
        )
    return rows


def run_fast_scan() -> pd.DataFrame:
    # Build category URL
    cat_obj = st.session_state.get("selected_category_obj")
    base_url = build_list_url_from_category(cat_obj, amazon_domain)

    fallback_node = None
    if cat_obj:
        cid = cat_obj.get("id") or ""
        if isinstance(cid, str) and "bestsellers_" in cid:
            fallback_node = re.sub(r"[^\d]", "", cid.split("bestsellers_", 1)[1])

    list_url = list_url_for_list_type(base_url, amazon_domain, list_type, fallback_node)

    if scan_source == "Category lists (recommended)":
        if not list_url:
            raise ValueError("No category URL selected. Use Category Finder → Pick a category, then Run scan.")
        # Pull pages
        all_items: List[Dict[str, Any]] = []
        for p in range(1, list_pages + 1):
            items = fetch_bestsellers(amazon_domain, list_url, page=p, force_refresh=force_refresh)
            all_items.extend(items)

        all_items = all_items[:scan_items]
        rows = normalize_items(all_items, amazon_domain)

    else:
        # Keyword search
        if not product_keyword.strip():
            raise ValueError("Keyword search selected, but keyword is empty. Enter a keyword (sidebar).")
        all_items = []
        for p in range(1, list_pages + 1):
            items = fetch_search(amazon_domain, product_keyword, page=p, force_refresh=force_refresh)
            all_items.extend(items)
        all_items = all_items[:scan_items]
        rows = normalize_items(all_items, amazon_domain)

    cfg = ProfitConfig(
        fee_pct=fee_pct,
        fixed_fee=fixed_fee,
        shipping=shipping_cost,
        packaging=packaging_cost,
        sourcing_discount_pct=sourcing_discount,
        profit_goal=profit_goal,
        hide_below_goal=hide_below_goal,
        target_price_min=target_min if use_price_range else None,
        target_price_max=target_max if use_price_range else None,
    )

    enriched: List[Dict[str, Any]] = []
    for r in rows:
        # Risk exclusion
        if exclude_risky and r["risk_flags"]:
            continue

        est = estimate_profitability(r.get("amazon_price"), cfg)
        r.update(est)
        r["overall_score"] = overall_score(r)
        enriched.append(r)

    df = pd.DataFrame(enriched)

    if df.empty:
        return df

    # Optional hide below goal
    if hide_below_goal:
        df = df[df["profit_est"].fillna(-9999) >= float(profit_goal)]

    # Sort
    df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    return df


def deep_dive_products(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    top = df.head(int(deep_dive_n)).copy()
    details = []
    for _, row in top.iterrows():
        asin = row.get("asin")
        if not asin:
            continue
        pdata = fetch_product(
            amazon_domain=amazon_domain,
            asin=str(asin),
            include_summarization_attributes=include_summarization,
            force_refresh=force_refresh,
        )

        # Pull a few useful fields
        product = pdata.get("product") or {}
        bullets = product.get("feature_bullets") or product.get("feature_bullet") or []
        bullet_text = " | ".join([str(b) for b in bullets[:5]]) if isinstance(bullets, list) else ""

        customers_say = product.get("customers_say") or product.get("summarization_attributes") or None
        customers_say_text = ""
        if customers_say is not None:
            customers_say_text = str(customers_say)[:600]

        details.append(
            {
                "asin": asin,
                "feature_bullets": bullet_text,
                "customers_say": customers_say_text,
            }
        )

    if details:
        ddf = pd.DataFrame(details)
        merged = top.merge(ddf, on="asin", how="left")
        # Put deep fields at end
        return merged

    return top


# =========================
# Rendering
# =========================
with tabs[0]:
    st.subheader("✅ Shortlist candidates (from scan)")
    st.caption("Click **Run scan** to generate results. Profitability settings affect ranking only unless you turn on hiding below goal.")

    if run_scan:
        if not RAINFOREST_API_KEY:
            st.error("Missing RAINFOREST_API_KEY. Add it to Streamlit Secrets.")
        else:
            try:
                df = run_fast_scan()

                if df.empty:
                    st.warning(
                        "No products returned after exclusions.\n\n"
                        "Try:\n"
                        "- Pick a different category in Category Finder\n"
                        "- Switch List type back to Best Sellers\n"
                        "- Turn OFF risk exclusion temporarily (to test)\n"
                        "- Increase scan size (30–50)\n"
                    )
                    st.session_state["last_df"] = None
                else:
                    # Deep dive if selected
                    if scan_mode == "Deep dive":
                        df = deep_dive_products(df)

                    st.session_state["last_df"] = df

                    st.success(f"Found {len(df)} ranked candidates. Requests used so far: {st.session_state['requests_used']}")

                    # Show table
                    show_cols = [
                        "overall_score",
                        "title",
                        "asin",
                        "amazon_price",
                        "sell_price_used",
                        "buy_price_est",
                        "fees_est",
                        "profit_est",
                        "roi_est",
                        "max_buy_price",
                        "rating",
                        "ratings_total",
                        "risk_flags",
                        "amazon_url",
                    ]
                    show_cols = [c for c in show_cols if c in df.columns]

                    st.dataframe(df[show_cols], use_container_width=True)

                    # Add-to-shortlist UI
                    st.markdown("### ⭐ Add items to My Shortlist")
                    options = df["asin"].astype(str).tolist()
                    picked_asins = st.multiselect("Select ASINs to add", options=options, default=[])

                    if st.button("Add selected to My Shortlist"):
                        cur = st.session_state["my_shortlist"]
                        cur_asins = {x.get("asin") for x in cur if isinstance(x, dict)}
                        for asin in picked_asins:
                            row = df[df["asin"].astype(str) == str(asin)].head(1)
                            if row.empty:
                                continue
                            rdict = row.iloc[0].to_dict()
                            if rdict.get("asin") not in cur_asins:
                                cur.append(rdict)
                        st.session_state["my_shortlist"] = cur
                        st.success(f"Added {len(picked_asins)} items (new only).")

                    # Export scan CSV
                    st.markdown("### ⬇️ Export scan results")
                    st.download_button(
                        "Download scan results CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="scan_results.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Scan failed: {redact_api_key(str(e))}")
                st.session_state["last_df"] = None
    else:
        st.info("Click **Run scan** to generate results.")


with tabs[1]:
    st.subheader("⭐ My Shortlist")
    st.caption("This persists within your session. For Streamlit Cloud ‘true persistence’, export/import CSV.")

    cur = st.session_state.get("my_shortlist", [])
    if not cur:
        st.info("No items yet. Add items from the Shortlist candidates tab.")
    else:
        sdf = pd.DataFrame(cur)
        # Keep a readable column set
        keep = [
            "overall_score",
            "title",
            "asin",
            "amazon_price",
            "profit_est",
            "roi_est",
            "max_buy_price",
            "rating",
            "ratings_total",
            "risk_flags",
            "amazon_url",
        ]
        keep = [c for c in keep if c in sdf.columns]
        st.dataframe(sdf[keep], use_container_width=True)

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Clear My Shortlist"):
                st.session_state["my_shortlist"] = []
                st.success("Cleared.")
        with c2:
            st.download_button(
                "Download My Shortlist CSV",
                data=sdf.to_csv(index=False).encode("utf-8"),
                file_name="my_shortlist.csv",
                mime="text/csv",
            )
        with c3:
            uploaded = st.file_uploader("Import My Shortlist CSV", type=["csv"])
            if uploaded is not None:
                try:
                    idf = pd.read_csv(uploaded)
                    st.session_state["my_shortlist"] = idf.to_dict(orient="records")
                    st.success(f"Imported {len(idf)} rows into My Shortlist.")
                except Exception as e:
                    st.error(redact_api_key(str(e)))


with tabs[2]:
    st.subheader("🧰 Workflows & Lead Gen")
    st.caption("These are copy/paste workflows to reduce wasted API requests and move faster to sourcing & validation.")

    st.markdown("### 1) eBay sold-price check (manual workflow)")
    st.code(
        "Use these searches (copy/paste into Google) for a quick sold-price reality check:\n"
        "1) site:ebay.com sold \"<product name>\" \n"
        "2) site:ebay.com \"<asin>\" sold\n"
        "3) \"<product name>\" \"sold items\" ebay\n"
        "Then open eBay → filter: Sold items / Completed items / US-only.\n",
        language="text",
    )

    st.markdown("### 2) Supplier lead generator (US wholesalers first)")
    st.code(
        "For each shortlisted product title, run these Google searches:\n"
        "- \"<product>\" wholesale USA\n"
        "- \"<product>\" distributor USA\n"
        "- \"<product>\" bulk supplier USA\n"
        "- site:faire.com \"<product>\"\n"
        "- site:thomasnet.com \"<product>\"\n"
        "- site:tundra.com \"<product>\"\n"
        "\nIf you’re sourcing from Pakistan/Alibaba, also run:\n"
        "- \"<product>\" manufacturer Pakistan\n"
        "- \"<product>\" Alibaba\n",
        language="text",
    )

    st.markdown("### 3) Bundle ideas generator (simple heuristic)")
    st.write(
        "Bundle rule-of-thumb: pair the main item with a **consumable**, a **replacement part**, or a **storage/organizer**.\n"
        "Example: kitchen organizer → labels / liners / microfiber cloths."
    )

    st.markdown("### 4) Request discipline (how to not burn 100 credits)")
    st.write(
        "- Use **Category Finder** only when you’re switching niches. Cache will prevent repeat costs unless you Force refresh.\n"
        "- Default scans: **30 items** (usually 1 page) → quick signal.\n"
        "- Only use **Deep dive** on top **3–6** candidates.\n"
        "- Keep risk exclusion ON during exploration; turn OFF only when debugging ‘why no results’."
    )


# Footer
st.caption(f"Session requests used: {st.session_state['requests_used']}")
