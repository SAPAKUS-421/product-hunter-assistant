import os
import re
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# Optional OpenAI (only used if user enables AI insights)
try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:
    OpenAI = None

APP_TITLE = "Product Hunter Assistant (USA Market)"
RF_REQUEST_URL = "https://api.rainforestapi.com/request"
RF_CATEGORIES_URL = "https://api.rainforestapi.com/categories"


# -------------------------
# Helpers: secrets + UI
# -------------------------
def get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then env vars."""
    try:
        val = st.secrets.get(key, default)
    except Exception:
        val = os.getenv(key, default)
    return str(val).strip()


def money(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).replace(",", "").strip()
        s = re.sub(r"[^\d.]", "", s)
        return float(s) if s else None
    except Exception:
        return None


def extract_price(item: Dict[str, Any]) -> Optional[float]:
    # Rainforest responses vary by endpoint. Try common shapes.
    for key in ["price", "buybox_price", "current_price", "amazon_price"]:
        if key in item:
            val = item.get(key)
            if isinstance(val, dict):
                return money(val.get("value") or val.get("raw") or val.get("amount"))
            return money(val)
    # Sometimes: item["prices"]["value"]
    if isinstance(item.get("prices"), dict):
        return money(item["prices"].get("value") or item["prices"].get("raw"))
    return None


def extract_request_credits(api_json: Dict[str, Any]) -> Optional[int]:
    info = api_json.get("request_info") or api_json.get("requestInfo") or {}
    # Common keys in Traject/Rainforest docs
    for k in ["credits_used_this_request", "creditsUsedThisRequest", "credits_used"]:
        if k in info:
            try:
                return int(info[k])
            except Exception:
                pass
    return None


# -------------------------
# Risk flags (simple but useful)
# -------------------------
RISK_KEYWORDS = {
    "supplement_or_ingestible": [
        "vitamin", "supplement", "softgel", "capsule", "tablet", "multivitamin",
        "omega", "probiotic", "gummy", "nutrition", "dietary"
    ],
    "medical_or_test": ["test kit", "blood type", "diagnostic", "medical", "otc"],
    "hazmat_or_restricted": ["battery", "lithium", "aerosol", "spray", "flammable", "propane"],
    "weapon_like": ["knife", "dagger", "machete", "sword"],
}

# A small “brand-heavy” heuristic list (you can expand over time)
BRAND_HEAVY = [
    "yeti", "vtech", "nature made", "carlyle", "wonka", "rit", "gum",
    "fellowes", "honda"
]


def risk_flags(title: str) -> List[str]:
    t = (title or "").lower()
    flags = []

    # Brand-heavy heuristic
    for b in BRAND_HEAVY:
        if b in t:
            flags.append("brand-heavy")
            break

    for group, words in RISK_KEYWORDS.items():
        for w in words:
            if w in t:
                flags.append(group.replace("_", "-"))
                break

    # Amazon/eBay IP risk heuristic: contains ® or ™
    if "®" in (title or "") or "™" in (title or ""):
        flags.append("trademarked-text")

    # Deduplicate
    out = []
    for f in flags:
        if f not in out:
            out.append(f)
    return out


# -------------------------
# Rainforest API calls (cached)
# -------------------------
def _rf_get(url: str, params: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def rf_bestsellers(api_key: str, amazon_domain: str, category_id: str, cache_bust: str) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "type": "bestsellers",
        "amazon_domain": amazon_domain,
        "category_id": category_id,
    }
    return _rf_get(RF_REQUEST_URL, params=params)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def rf_reviews(api_key: str, amazon_domain: str, asin: str, cache_bust: str) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "type": "reviews",
        "amazon_domain": amazon_domain,
        "asin": asin,
        "review_stars": "all_critical",
        "sort_by": "most_recent",
    }
    return _rf_get(RF_REQUEST_URL, params=params)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def rf_search_categories(api_key: str, domain: str, cat_type: str, search_term: str, cache_bust: str) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "domain": domain,
        "type": cat_type,          # bestsellers / standard / deals / etc.
        "search_term": search_term
    }
    return _rf_get(RF_CATEGORIES_URL, params=params)


# -------------------------
# AI (optional)
# -------------------------
def ai_gap_insights(openai_api_key: str, model: str, title: str, critical_reviews: List[str]) -> str:
    if OpenAI is None:
        return "OpenAI library not installed. Add `openai>=1.0.0` to requirements.txt"
    client = OpenAI(api_key=openai_api_key)

    review_blob = "\n".join([f"- {r[:350]}" for r in critical_reviews[:10]])

    prompt = f"""
You are helping an e-commerce reseller hunt profitable products.

Product: {title}

Critical review snippets:
{review_blob}

Return:
1) Top 3 concrete complaints (very specific)
2) 3 bundle ideas that increase AOV (practical)
3) 3 sourcing angles for USA wholesalers (not Amazon/eBay)
4) A 1-paragraph “how to position this listing differently” angle for eBay/Amazon

Be concise and concrete. Avoid hype.
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# Profit math
# -------------------------
def compute_financials(
    sell_price: float,
    fee_pct: float,
    fixed_fee: float,
    ship_cost: float,
    pack_cost: float,
    desired_profit: float,
) -> Dict[str, float]:
    marketplace_fee = (sell_price * (fee_pct / 100.0)) + fixed_fee
    max_buy = sell_price - marketplace_fee - ship_cost - pack_cost - desired_profit
    est_profit = sell_price - marketplace_fee - ship_cost - pack_cost  # before buy cost
    return {
        "marketplace_fee": round(marketplace_fee, 2),
        "max_buy_price": round(max_buy, 2),
        "profit_before_buy": round(est_profit, 2),
    }


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🧭 {APP_TITLE}")
st.caption("Pull Amazon bestsellers via Rainforest, score opportunities for resale/wholesale, and generate sourcing leads. (Optional AI review-gap insights)")

RF_KEY = get_secret("RAINFOREST_API_KEY")
OA_KEY = get_secret("OPENAI_API_KEY")
OA_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []  # list[dict]
if "rf_requests_estimate_last" not in st.session_state:
    st.session_state["rf_requests_estimate_last"] = 0
if "rf_requests_actual_total" not in st.session_state:
    st.session_state["rf_requests_actual_total"] = 0


# ---- Sidebar controls ----
with st.sidebar:
    st.header("A) Source")

    amazon_domain = st.selectbox("Amazon domain", ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de"], index=0)

    # Category ID input stored in session_state so we can set it from the finder
    if "category_id" not in st.session_state:
        st.session_state["category_id"] = ""

    st.text_input("Rainforest category id (BESTSELLERS category id)", key="category_id", placeholder="Use the Category Finder below")

    bestsellers_to_scan = st.slider("How many bestsellers to scan", 5, 50, 30, step=5)
    force_refresh = st.checkbox("Force refresh (bypass cache)", value=False)

    st.divider()
    st.subheader("Category Finder (recommended)")
    st.caption("Use this to find the correct BESTSELLERS category_id. This costs ~1 request per search.")
    cat_search_term = st.text_input("Search term", value="Home & Kitchen")
    cat_type = st.selectbox("Category type", ["bestsellers", "standard", "new releases", "movers and shakers", "most wished for", "gift ideas", "deals"], index=0)

    if st.button("Search categories"):
        if not RF_KEY:
            st.error("Add RAINFOREST_API_KEY in Streamlit Secrets first.")
        else:
            bust = str(datetime.utcnow()) if force_refresh else "cache"
            try:
                cat_json = rf_search_categories(RF_KEY, amazon_domain, cat_type, cat_search_term, bust)
                used = extract_request_credits(cat_json) or 1
                st.session_state["rf_requests_actual_total"] += used

                cats = cat_json.get("categories", [])[:50]
                if not cats:
                    st.warning("No categories found. Try a broader term (e.g., 'Kitchen', 'Storage', 'Pet').")
                else:
                    # Show selection
                    options = [
                        f"{c.get('id')}  |  {c.get('path')}"
                        for c in cats
                        if c.get("id") and c.get("path")
                    ]
                    pick = st.selectbox("Pick a category_id", options)
                    if pick:
                        st.session_state["category_id"] = pick.split("|")[0].strip()
                        st.success("Category ID filled above. Now run scan.")
            except Exception as e:
                st.error(f"Category search failed: {e}")

    st.divider()
    st.header("B) Your selling assumptions")

    sell_price_mode = st.selectbox("Selling price for profit math", ["Use Amazon price (rough)", "Use a fixed target price"], index=0)
    target_sell_price = st.number_input("Target selling price ($)", min_value=5.0, max_value=500.0, value=24.99, step=1.0)

    fee_pct = st.slider("Marketplace fee % (eBay/Amazon etc.)", 5, 25, 15)
    fixed_fee = st.number_input("Fixed fee per order ($)", min_value=0.0, max_value=10.0, value=0.30, step=0.10)

    ship_cost = st.number_input("Shipping cost ($)", min_value=0.0, max_value=50.0, value=4.50, step=0.50)
    pack_cost = st.number_input("Packaging cost ($)", min_value=0.0, max_value=10.0, value=0.50, step=0.10)

    desired_profit = st.number_input("Desired profit per unit ($) (used to compute Max Buy Price)", min_value=1.0, max_value=100.0, value=8.00, step=1.0)

    st.divider()
    st.header("C) Scan mode + request planning")

    scan_mode = st.radio("Mode", ["Fast scan (bestsellers only)", "Deep dive (bestsellers + critical reviews)"], index=0)

    enable_ai = st.checkbox("Enable AI review-gap insights (costs OpenAI + extra review calls)", value=False)
    top_n_reviews = 3
    if scan_mode.startswith("Deep") or enable_ai:
        top_n_reviews = st.slider("Fetch critical reviews for top N products", 1, 10, 3)

    # Estimated requests
    est_requests = 1  # bestsellers call
    if scan_mode.startswith("Deep"):
        est_requests += top_n_reviews  # one reviews call per ASIN
    if enable_ai and not scan_mode.startswith("Deep"):
        # AI needs reviews; enforce
        est_requests += top_n_reviews

    st.session_state["rf_requests_estimate_last"] = est_requests
    st.info(f"Estimated Rainforest requests for next 'Run scan': ~{est_requests}")

    st.divider()
    st.header("Keys status")
    st.write(("✅ " if RF_KEY else "❌ ") + "RAINFOREST_API_KEY")
    st.write(("✅ " if OA_KEY else "⚠️ ") + "OPENAI_API_KEY (optional)")
    st.caption(f"Model: {OA_MODEL}")

    st.divider()
    st.caption(f"Approx. Rainforest requests consumed (from responses we captured): {st.session_state['rf_requests_actual_total']}")


# ---- Main actions ----
run = st.button("🚀 Run scan", type="primary")

if run:
    if not RF_KEY:
        st.error("Missing RAINFOREST_API_KEY. Add it in Streamlit → Manage app → Settings → Secrets.")
        st.stop()

    category_id = (st.session_state.get("category_id") or "").strip()
    if not category_id:
        st.error("Please set a valid BESTSELLERS category_id. Use the Category Finder in the sidebar.")
        st.stop()

    bust = str(datetime.utcnow()) if force_refresh else "cache"

    with st.spinner("Pulling Amazon bestsellers from Rainforest…"):
        try:
            bs_json = rf_bestsellers(RF_KEY, amazon_domain, category_id, bust)
            used = extract_request_credits(bs_json) or 1
            st.session_state["rf_requests_actual_total"] += used
        except requests.HTTPError as e:
            st.error(f"Rainforest HTTP error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Rainforest error: {e}")
            st.stop()

    items = bs_json.get("bestsellers", []) or []
    if not items:
        st.warning(
            "No bestsellers returned. This almost always means the category_id is not a valid BESTSELLERS category.\n\n"
            "Use the Category Finder: set type=bestsellers, search 'Home & Kitchen', pick the numeric id, then run scan again."
        )
        st.session_state["last_df"] = None
        st.stop()

    rows: List[Dict[str, Any]] = []
    for it in items[:bestsellers_to_scan]:
        title = it.get("title") or ""
        asin = it.get("asin") or ""
        rank = it.get("rank") or None

        ap = extract_price(it)
        sell_price = ap if (sell_price_mode.startswith("Use Amazon") and ap) else float(target_sell_price)

        fin = compute_financials(
            sell_price=sell_price,
            fee_pct=float(fee_pct),
            fixed_fee=float(fixed_fee),
            ship_cost=float(ship_cost),
            pack_cost=float(pack_cost),
            desired_profit=float(desired_profit),
        )

        flags = risk_flags(title)
        rows.append({
            "rank": rank,
            "title": title,
            "asin": asin,
            "amazon_price": round(ap, 2) if ap else None,
            "used_sell_price": round(sell_price, 2),
            "marketplace_fee": fin["marketplace_fee"],
            "max_buy_price": fin["max_buy_price"],
            "risk_flags": ", ".join(flags) if flags else "",
        })

    df = pd.DataFrame(rows)

    # Sort: rank ascending (rank 1 is best). Also push empty ranks down.
    if "rank" in df.columns:
        df["rank_sort"] = df["rank"].fillna(10**9)
        df = df.sort_values(["rank_sort"], ascending=True).drop(columns=["rank_sort"])

    st.session_state["last_df"] = df


# ---- Display results ----
st.subheader("✅ Shortlist candidates (from bestsellers)")
df = st.session_state.get("last_df")

if df is None or df.empty:
    st.info("Click **Run scan** to generate results.")
else:
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Filters")
        hide_risky = st.checkbox("Hide items with risk flags", value=True)
        min_sell = st.number_input("Min used sell price", value=10.0, step=1.0)
        max_sell = st.number_input("Max used sell price", value=60.0, step=1.0)

    show_df = df.copy()
    show_df = show_df[(show_df["used_sell_price"] >= float(min_sell)) & (show_df["used_sell_price"] <= float(max_sell))]
    if hide_risky:
        show_df = show_df[show_df["risk_flags"].fillna("") == ""]

    with col1:
        st.dataframe(show_df, use_container_width=True)

        st.markdown("### Add to My Shortlist")
        options = [
            f"{r['asin']} | {str(r['title'])[:80]}"
            for _, r in show_df.head(30).iterrows()
            if r.get("asin")
        ]
        picks = st.multiselect("Pick products to shortlist", options)
        if st.button("➕ Add selected to Shortlist"):
            existing = {x.get("asin") for x in st.session_state["shortlist"]}
            for p in picks:
                asin = p.split("|")[0].strip()
                row = show_df[show_df["asin"] == asin].head(1)
                if not row.empty and asin not in existing:
                    st.session_state["shortlist"].append(row.iloc[0].to_dict())
            st.success("Added to Shortlist.")

    st.divider()

    # ---- Shortlist persistence (download/upload) ----
    st.subheader("📌 My Shortlist (export/import to persist across sessions)")
    sl = pd.DataFrame(st.session_state["shortlist"]) if st.session_state["shortlist"] else pd.DataFrame()

    cA, cB = st.columns([2, 1])
    with cA:
        if sl.empty:
            st.info("Your shortlist is empty. Add items from the results above.")
        else:
            st.dataframe(sl, use_container_width=True)

    with cB:
        if not sl.empty:
            st.download_button(
                "Download Shortlist CSV",
                sl.to_csv(index=False).encode("utf-8"),
                file_name="my_shortlist.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download Shortlist JSON",
                json.dumps(st.session_state["shortlist"], indent=2).encode("utf-8"),
                file_name="my_shortlist.json",
                mime="application/json"
            )

        up = st.file_uploader("Import Shortlist JSON", type=["json"])
        if up is not None:
            try:
                data = json.loads(up.read().decode("utf-8"))
                if isinstance(data, list):
                    st.session_state["shortlist"] = data
                    st.success("Imported shortlist.")
                else:
                    st.error("Invalid file: expected a JSON list.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    st.divider()

    # ---- Deep dive: reviews + AI ----
    if (st.sidebar and (st.session_state["rf_requests_estimate_last"] >= 2)):
        st.subheader("🔍 Deep Dive (critical reviews)")

    do_deep = st.sidebar and (scan_mode.startswith("Deep") or enable_ai)

    if do_deep:
        if df.empty:
            st.stop()

        top_df = df.copy()
        # Prefer clean items first (no flags), then best rank
        top_df["has_flags"] = top_df["risk_flags"].fillna("").apply(lambda x: 1 if x else 0)
        top_df["rank_sort"] = top_df["rank"].fillna(10**9)
        top_df = top_df.sort_values(["has_flags", "rank_sort"], ascending=[True, True]).drop(columns=["has_flags", "rank_sort"])
        top_df = top_df.head(top_n_reviews)

        if not RF_KEY:
            st.warning("Missing Rainforest key.")
        else:
            for _, r in top_df.iterrows():
                asin = r.get("asin")
                title = r.get("title")
                if not asin:
                    continue

                with st.expander(f"Deep Dive: {title[:90]}  ({asin})"):
                    bust = str(datetime.utcnow()) if force_refresh else "cache"
                    try:
                        rev_json = rf_reviews(RF_KEY, amazon_domain, asin, bust)
                        used = extract_request_credits(rev_json) or 1
                        st.session_state["rf_requests_actual_total"] += used
                    except Exception as e:
                        st.error(f"Review fetch failed: {e}")
                        continue

                    reviews = rev_json.get("reviews", []) or []
                    crit = [x.get("body", "") for x in reviews if x.get("body")]
                    if not crit:
                        st.info("No critical reviews returned for this ASIN (or API limits).")
                    else:
                        st.markdown("**Recent critical review snippets:**")
                        for t in crit[:6]:
                            st.write("• " + t[:400])

                    if enable_ai:
                        if not OA_KEY:
                            st.warning("AI is enabled but OPENAI_API_KEY is missing in Secrets.")
                        else:
                            with st.spinner("Generating AI insights…"):
                                try:
                                    txt = ai_gap_insights(OA_KEY, OA_MODEL, title, crit[:10])
                                    st.markdown("### AI Insights")
                                    st.write(txt)
                                except Exception as e:
                                    st.error(f"AI failed: {e}")

    st.divider()

    # ---- Practical workflows: eBay + suppliers + bundles ----
    st.subheader("🧩 Reseller workflows (copy/paste)")

    st.markdown("### 1) eBay sold-price check (manual)")
    st.code(
        "Use eBay Completed + Sold listings to validate demand & price.\n"
        "Search template:\n"
        "  site:ebay.com \"YOUR PRODUCT KEYWORDS\" (Sold listings)\n"
        "In eBay UI:\n"
        "  Search → Filters → Show only → Sold items + Completed items",
        language="text"
    )

    st.markdown("### 2) Supplier lead generator (US wholesalers first)")
    top_titles = [t for t in df["title"].head(5).tolist() if isinstance(t, str) and t.strip()]
    if top_titles:
        for t in top_titles:
            kw = re.sub(r"\s+", " ", t)[:80]
            st.markdown(f"**{kw}**")
            st.code(
                f"wholesale \"{kw}\" USA\n"
                f"distributor \"{kw}\" USA\n"
                f"bulk supplier \"{kw}\" USA\n"
                f"site:faire.com \"{kw}\"\n"
                f"site:tundra.com \"{kw}\"\n"
                f"site:thomasnet.com \"{kw}\"\n"
                f"site:wholesalecentral.com \"{kw}\"",
                language="text"
            )

    st.markdown("### 3) Bundle ideas (non-AI quick prompts)")
    st.code(
        "Bundle rules of thumb:\n"
        "- Pair a 'core item' with 2–3 consumables/accessories\n"
        "- Aim bundle retail: $24.99–$39.99\n"
        "- Make it easy: a complete set (starter kit angle)\n"
        "\n"
        "Examples:\n"
        "- Magnetic tape + label stickers + mini scissors\n"
        "- Fridge calendar + extra markers + magnetic eraser\n"
        "- Tool handle grip + work gloves + small tape measure",
        language="text"
    )
