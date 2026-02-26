import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# ==========================================
# Page config
# ==========================================
st.set_page_config(
    page_title="Product Hunter Assistant (USA Market)",
    page_icon="🧭",
    layout="wide",
)


# ==========================================
# Secrets helper
# ==========================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        import os
        return str(os.getenv(key, default)).strip()


RAINFOREST_API_KEY = get_secret("RAINFOREST_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")  # optional
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


# ==========================================
# Constants
# ==========================================
RAINFOREST_BASE = "https://api.rainforestapi.com"

ITEMS_PER_PAGE = 50  # typical bestsellers/search page size


# ==========================================
# Session state init
# ==========================================
def init_state() -> None:
    st.session_state.setdefault("requests_used", 0)
    st.session_state.setdefault("api_cache", {})  # safe cache key -> (timestamp, json)
    st.session_state.setdefault("category_results", [])
    st.session_state.setdefault("selected_category_obj", None)
    st.session_state.setdefault("last_df", None)
    st.session_state.setdefault("my_shortlist", [])


init_state()


# ==========================================
# Safety: redact keys in errors + safe caching
# ==========================================
def redact_api_key(text: str) -> str:
    """Redact api_key=... inside any URL/text."""
    if not text:
        return text
    return re.sub(r"(api_key=)([^&\s]+)", r"\1****REDACTED****", str(text))


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
    Safe Rainforest caller:
    - Never leaks api_key in error messages
    - Detects 'temporarily suspended' messages and shows a clean instruction
    - Uses in-session cache without storing real api_key in cache key
    """
    if not RAINFOREST_API_KEY:
        raise RuntimeError("Missing RAINFOREST_API_KEY in Streamlit Secrets.")

    url = f"{RAINFOREST_BASE}{endpoint}"
    params2 = dict(params)

    if "api_key" not in params2:
        params2["api_key"] = RAINFOREST_API_KEY

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

    # Structured message handling
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
            raise RuntimeError(f"Rainforest error: {redact_api_key(msg)}")

    # Generic HTTP error fallback
    if r.status_code >= 400:
        raise RuntimeError(f"Rainforest HTTP {r.status_code}. URL: {redact_api_key(r.url)}")

    if not isinstance(data, dict):
        data = {}

    st.session_state["api_cache"][key] = (time.time(), data)
    if count_request:
        st.session_state["requests_used"] += 1

    return data


# ==========================================
# Risk exclusion (default ON)
# ==========================================
RISK_GROUPS = {
    "brand-heavy": [
        "disney", "pokemon", "nintendo", "marvel", "star wars", "lego", "hello kitty",
        "yeti", "vtech", "fellowes", "honda",
    ],
    "supplement": [
        "supplement", "vitamin", "creatine", "whey", "protein powder", "fat burner",
        "weight loss", "keto", "collagen", "testosterone", "detox", "cleanse", "probiotic",
        "pre workout", "pre-workout", "bcaa",
    ],
    "medical-test": [
        "medical", "medicine", "drug", "diagnostic", "test kit", "rapid test", "covid",
        "glucose", "blood", "bp monitor", "blood pressure", "thermometer", "stethoscope",
        "pregnancy test",
    ],
    "hazmat-ish": [
        "flammable", "corrosive", "acid", "aerosol", "propane", "butane", "fuel",
        "solvent", "bleach", "pesticide", "insecticide", "herbicide", "toxic",
        "lithium", "battery",
    ],
    "weapon-like": [
        "knife", "dagger", "machete", "sword", "switchblade", "tactical", "ammo",
        "gun", "firearm", "holster", "crossbow", "arrowhead", "pepper spray",
    ],
}


def risk_flags(title: str, brand: str = "") -> List[str]:
    t = (title or "").lower()
    b = (brand or "").lower()
    flags = []
    for group, kws in RISK_GROUPS.items():
        for kw in kws:
            if kw in t or kw in b:
                flags.append(group)
                break
    if "™" in (title or "") or "®" in (title or ""):
        flags.append("trademarked-text")
    # unique
    out = []
    for f in flags:
        if f not in out:
            out.append(f)
    return out


# ==========================================
# Categories API (STANDARD) + filter Renewed
# ==========================================
def categories_search_standard(domain: str, search_term: str, force_refresh: bool, include_renewed: bool) -> List[Dict[str, Any]]:
    if not search_term.strip():
        return []

    data = rainforest_get_json(
        endpoint="/categories",
        params={
            "domain": domain,          # correct parameter name for /categories
            "type": "standard",        # real category tree
            "search_term": search_term.strip(),
        },
        force_refresh=force_refresh,
        count_request=True,
    )

    cats = data.get("categories") or []
    seen = set()
    out = []

    for c in cats:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if not cid or cid in seen:
            continue

        name = str(c.get("name") or c.get("title") or c.get("path") or "").strip()
        name_lower = name.lower()

        if not include_renewed and "renewed" in name_lower:
            continue

        seen.add(cid)
        out.append(c)

    return out


def category_label(c: Dict[str, Any]) -> str:
    name = str(c.get("path") or c.get("name") or c.get("title") or "Category").strip()
    cid = str(c.get("id") or "").strip()
    return f"{name} | {cid}"


def build_node_from_category(cat_obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cat_obj or not isinstance(cat_obj, dict):
        return None
    cid = str(cat_obj.get("id") or "").strip()
    node = re.sub(r"[^\d]", "", cid)
    return node if node else None


def list_url_for_node(amazon_domain: str, node: str, list_type: str) -> str:
    list_type = list_type.strip()
    if list_type == "Best Sellers":
        return f"https://www.{amazon_domain}/Best-Sellers/zgbs/?node={node}"
    if list_type == "Trending (Movers & Shakers)":
        return f"https://www.{amazon_domain}/gp/movers-and-shakers/?node={node}"
    if list_type == "New Releases":
        return f"https://www.{amazon_domain}/gp/new-releases/?node={node}"
    if list_type == "Most Wished For":
        return f"https://www.{amazon_domain}/gp/most-wished-for/?node={node}"
    if list_type == "Most Gifted":
        return f"https://www.{amazon_domain}/gp/most-gifted/?node={node}"
    # fallback
    return f"https://www.{amazon_domain}/Best-Sellers/zgbs/?node={node}"


# ==========================================
# Product list fetchers
# ==========================================
def fetch_bestsellers(url: str, page: int, force_refresh: bool) -> List[Dict[str, Any]]:
    """
    IMPORTANT: When using `url`, do NOT include `amazon_domain` (Rainforest rejects the combo).
    """
    data = rainforest_get_json(
        endpoint="/request",
        params={
            "type": "bestsellers",
            "url": url,
            "page": page,
        },
        force_refresh=force_refresh,
        count_request=True,
    )
    items = (
        data.get("bestsellers")
        or data.get("best_sellers")
        or data.get("items")
        or []
    )
    return [x for x in items if isinstance(x, dict)]


def fetch_search(amazon_domain: str, search_term: str, page: int, force_refresh: bool) -> List[Dict[str, Any]]:
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
    items = data.get("search_results") or data.get("results") or []
    return [x for x in items if isinstance(x, dict)]


def normalize_items(raw_items: List[Dict[str, Any]], amazon_domain: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, it in enumerate(raw_items, start=1):
        asin = it.get("asin") or it.get("product_asin") or it.get("parent_asin")
        title = it.get("title") or it.get("name") or ""
        brand = it.get("brand") or ""
        rank = it.get("rank") or idx
        rating = it.get("rating")
        ratings_total = it.get("ratings_total") or it.get("reviews_total") or it.get("ratings") or 0

        price = None
        if isinstance(it.get("price"), dict):
            price = it["price"].get("value")
        elif it.get("price") is not None:
            price = it.get("price")

        try:
            price = float(price) if price is not None else None
        except Exception:
            price = None

        flags = risk_flags(title, brand)

        rows.append({
            "rank": rank,
            "title": title,
            "asin": asin,
            "brand": brand,
            "amazon_price": price,
            "rating": rating,
            "ratings_total": ratings_total,
            "risk_flags": ", ".join(flags) if flags else "",
            "amazon_url": f"https://www.{amazon_domain}/dp/{asin}" if asin else "",
        })
    return rows


# ==========================================
# Request planning
# ==========================================
def estimate_requests(scan_items: int, deep_mode: str, deep_n: int) -> int:
    list_pages = max(1, math.ceil(scan_items / ITEMS_PER_PAGE))
    est = list_pages
    if deep_mode != "Off":
        est += deep_n  # 1 extra call per item (product details or reviews)
    return est


# ==========================================
# UI
# ==========================================
st.title("🧭 Product Hunter Assistant (USA Market)")
st.caption("Discover what’s selling by category or keyword. Risky items excluded by default. No profit filters.")


# Sidebar
st.sidebar.title("Controls")

amazon_domain = st.sidebar.selectbox(
    "Amazon domain",
    ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de", "amazon.fr", "amazon.it", "amazon.es"],
    index=0,
)

force_refresh = st.sidebar.checkbox("Force refresh (bypass cache)", value=False)

scan_source = st.sidebar.selectbox(
    "Scan source",
    ["Category lists (recommended)", "Keyword search (optional)"],
    index=0,
)

list_type = st.sidebar.selectbox(
    "List type",
    ["Best Sellers", "Trending (Movers & Shakers)", "New Releases", "Most Wished For", "Most Gifted"],
    index=0,
)

st.sidebar.markdown("### Category Finder (STANDARD)")
include_renewed = st.sidebar.checkbox("Include Renewed categories", value=False)
cat_search_term = st.sidebar.text_input(
    "Category search term",
    value=st.session_state.get("cat_search_term", "Kitchen & Dining"),
    placeholder="Try: Kitchen & Dining / Storage & Organization / Pet Supplies / Tools",
)
st.session_state["cat_search_term"] = cat_search_term

if st.sidebar.button("Search categories"):
    try:
        cats = categories_search_standard(
            domain=amazon_domain,
            search_term=cat_search_term,
            force_refresh=force_refresh,
            include_renewed=include_renewed,
        )
        st.session_state["category_results"] = cats
        if not cats:
            st.sidebar.warning("No categories found. Try a different term (Kitchen / Storage / Pet / Tools).")
        else:
            st.sidebar.success(f"Found {len(cats)} categories.")
    except Exception as e:
        st.sidebar.error(redact_api_key(str(e)))

category_options = st.session_state.get("category_results", [])
selected_cat = None
if category_options:
    labels = [category_label(c) for c in category_options]
    picked_idx = st.sidebar.selectbox(
        "Pick a category",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        index=0,
        key="picked_category_index",
    )
    selected_cat = category_options[int(picked_idx)]
    st.session_state["selected_category_obj"] = selected_cat

    node = build_node_from_category(selected_cat) or ""
    st.sidebar.markdown("**Selected node (copyable)**")
    st.sidebar.code(node if node else "—")

    if node:
        st.sidebar.markdown("**Generated list URL (copyable)**")
        st.sidebar.code(list_url_for_node(amazon_domain, node, list_type))
else:
    st.sidebar.info("Search and pick a category to scan.")

st.sidebar.markdown("### Keyword (optional)")
keyword = st.sidebar.text_input(
    "Keyword for search scan",
    value="",
    placeholder="e.g., magnetic calendar / kitchen organizer / dog toy",
)

st.sidebar.markdown("### Scan size + planning")
scan_items = st.sidebar.slider("How many items to scan", min_value=10, max_value=100, value=50, step=10)

deep_mode = st.sidebar.selectbox(
    "Deep dive",
    ["Off", "Product details", "Critical reviews"],
    index=0,
    help="Optional. Costs extra requests. Use after you see a good shortlist.",
)
deep_n = 0
if deep_mode != "Off":
    deep_n = st.sidebar.slider("Deep dive top N", min_value=3, max_value=15, value=5, step=1)

st.sidebar.markdown("### Risk filters")
exclude_risky = st.sidebar.checkbox(
    "Exclude risk-flagged items",
    value=True,
    help="brand-heavy / supplement / medical-test / hazmat-ish / weapon-like",
)

price_filter_on = st.sidebar.checkbox("Filter by price range (optional)", value=False)
min_price = max_price = None
if price_filter_on:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        min_price = st.number_input("Min price", min_value=0.0, value=0.0, step=1.0)
    with c2:
        max_price = st.number_input("Max price", min_value=0.0, value=200.0, step=1.0)

est = estimate_requests(scan_items, deep_mode, deep_n)
st.sidebar.info(
    f"Estimated requests this scan: ~{est}\n\n"
    f"Requests used this session: {st.session_state['requests_used']}"
)

st.sidebar.markdown("### Keys status")
st.sidebar.write(("✅" if RAINFOREST_API_KEY else "❌") + " RAINFOREST_API_KEY")
st.sidebar.write(("✅" if OPENAI_API_KEY else "⚪") + " OPENAI_API_KEY (optional)")


# Main controls
run_scan = st.button("🚀 Run scan", type="primary")
tabs = st.tabs(["✅ Results", "⭐ My Shortlist", "🧰 Workflows"])


def run_scan_logic() -> pd.DataFrame:
    # Get items
    if scan_source == "Category lists (recommended)":
        cat_obj = st.session_state.get("selected_category_obj")
        node = build_node_from_category(cat_obj)
        if not node:
            raise RuntimeError("Pick a category first (Category Finder → Search → Pick).")

        list_url = list_url_for_node(amazon_domain, node, list_type)

        list_pages = max(1, math.ceil(scan_items / ITEMS_PER_PAGE))
        all_items: List[Dict[str, Any]] = []
        for p in range(1, list_pages + 1):
            all_items.extend(fetch_bestsellers(list_url, page=p, force_refresh=force_refresh))

        all_items = all_items[:scan_items]
        rows = normalize_items(all_items, amazon_domain)

    else:
        if not keyword.strip():
            raise RuntimeError("Keyword search selected, but keyword is empty.")
        list_pages = max(1, math.ceil(scan_items / ITEMS_PER_PAGE))
        all_items = []
        for p in range(1, list_pages + 1):
            all_items.extend(fetch_search(amazon_domain, keyword, page=p, force_refresh=force_refresh))
        all_items = all_items[:scan_items]
        rows = normalize_items(all_items, amazon_domain)

    # Apply exclusions / filters
    out = []
    for r in rows:
        if exclude_risky and r.get("risk_flags"):
            continue

        if price_filter_on and r.get("amazon_price") is not None:
            if min_price is not None and r["amazon_price"] < float(min_price):
                continue
            if max_price is not None and r["amazon_price"] > float(max_price):
                continue

        out.append(r)

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # Sort: lowest rank first if rank is numeric-ish
    def _rank_val(x):
        try:
            return int(str(x).replace(",", ""))
        except Exception:
            return 10**9

    df["__rank"] = df["rank"].apply(_rank_val)
    df = df.sort_values("__rank", ascending=True).drop(columns=["__rank"]).reset_index(drop=True)

    return df


with tabs[0]:
    st.subheader("✅ Results")

    if run_scan:
        try:
            df = run_scan_logic()
            st.session_state["last_df"] = df

            if df.empty:
                st.warning(
                    "No results returned.\n\n"
                    "Try:\n"
                    "- Turn OFF risk exclusion (test)\n"
                    "- Turn OFF price filter\n"
                    "- Use a different category search term (Kitchen & Dining / Storage & Organization)\n"
                    "- Switch scan source to Keyword search with a specific keyword\n"
                )
            else:
                st.success(f"Found {len(df)} items. Requests used this session: {st.session_state['requests_used']}")

                show_cols = ["rank", "title", "asin", "amazon_price", "rating", "ratings_total", "risk_flags", "amazon_url"]
                show_cols = [c for c in show_cols if c in df.columns]
                st.dataframe(df[show_cols], use_container_width=True)

                st.download_button(
                    "Download results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="scan_results.csv",
                    mime="text/csv",
                )

                st.markdown("### Add to My Shortlist")
                options = df["asin"].fillna("").astype(str).tolist()
                picked = st.multiselect("Pick ASINs", options=options, default=[])
                if st.button("Add selected"):
                    cur = st.session_state["my_shortlist"]
                    cur_asins = {x.get("asin") for x in cur if isinstance(x, dict)}
                    for asin in picked:
                        if not asin:
                            continue
                        row = df[df["asin"].astype(str) == asin].head(1)
                        if row.empty:
                            continue
                        d = row.iloc[0].to_dict()
                        if d.get("asin") not in cur_asins:
                            cur.append(d)
                    st.session_state["my_shortlist"] = cur
                    st.success("Added to shortlist (new only).")

        except Exception as e:
            st.error(f"Scan failed: {redact_api_key(str(e))}")
    else:
        st.info("Pick a category (or keyword) and click **Run scan**.")


with tabs[1]:
    st.subheader("⭐ My Shortlist")
    cur = st.session_state.get("my_shortlist", [])
    if not cur:
        st.info("Empty. Add items from Results.")
    else:
        sdf = pd.DataFrame(cur)
        keep = ["rank", "title", "asin", "amazon_price", "rating", "ratings_total", "risk_flags", "amazon_url"]
        keep = [c for c in keep if c in sdf.columns]
        st.dataframe(sdf[keep], use_container_width=True)

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Clear shortlist"):
                st.session_state["my_shortlist"] = []
                st.success("Cleared.")
        with c2:
            st.download_button(
                "Download shortlist CSV",
                data=sdf.to_csv(index=False).encode("utf-8"),
                file_name="my_shortlist.csv",
                mime="text/csv",
            )
        with c3:
            up = st.file_uploader("Import shortlist CSV", type=["csv"])
            if up is not None:
                try:
                    idf = pd.read_csv(up)
                    st.session_state["my_shortlist"] = idf.to_dict(orient="records")
                    st.success(f"Imported {len(idf)} rows.")
                except Exception as e:
                    st.error(redact_api_key(str(e)))


with tabs[2]:
    st.subheader("🧰 Workflows")

    st.markdown("### eBay sold-price validation (manual)")
    st.code(
        "In eBay:\n"
        "Search product → Filters → Show only → Sold items + Completed items.\n"
        "Compare sold prices across multiple listings.\n",
        language="text",
    )

    st.markdown("### Supplier lead generator (US wholesalers first)")
    df = st.session_state.get("last_df")
    titles = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        titles = df["title"].head(5).fillna("").tolist()

    if titles:
        for t in titles:
            t = str(t).strip()
            if not t:
                continue
            st.markdown(f"**{t[:90]}**")
            st.code(
                "\n".join([
                    f'wholesale "{t}" USA',
                    f'distributor "{t}" USA',
                    f'bulk supplier "{t}" USA',
                    f'site:faire.com "{t}"',
                    f'site:tundra.com "{t}"',
                    f'site:thomasnet.com "{t}"',
                ]),
                language="text",
            )
    else:
        st.info("Run a scan first to generate supplier searches from the top results.")

    st.markdown("### Trial-usage discipline (don’t burn 100 requests)")
    st.write(
        "- Use Category Finder only when changing niches.\n"
        "- Scan 50 items (usually 1 request per list page).\n"
        "- Use Keyword search for narrow product hunting.\n"
        "- Keep risk exclusion ON during exploration.\n"
    )

st.caption(f"Session requests used (approx): {st.session_state['requests_used']}")
