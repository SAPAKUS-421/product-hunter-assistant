import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st

# ============================================================
# Product Hunter Assistant (USA Market)
# ============================================================

st.set_page_config(page_title="Product Hunter Assistant (USA Market)", layout="wide")

# -------------------------
# Secrets
# -------------------------
def get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit Secrets first, then env vars."""
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return str(os.getenv(key, default)).strip()

RAINFOREST_API_KEY = get_secret("RAINFOREST_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

RF_REQUEST_URL = "https://api.rainforestapi.com/request"
RF_CATEGORIES_URL = "https://api.rainforestapi.com/categories"

# -------------------------
# Risk filters (default ON)
# -------------------------
RISK_KEYWORDS = [
    # Brand / counterfeit / IP risk (not exhaustive)
    "disney", "marvel", "pokemon", "nintendo", "sony", "microsoft", "apple", "samsung",
    "lego", "barbie", "nerf", "hot wheels", "hasbro", "mattel",

    # Supplements / ingestibles / regulated
    "supplement", "vitamin", "omega", "probiotic", "creatine", "protein", "capsule",
    "softgel", "tablet", "gummies", "edible", "dietary", "herbal", "homeopathic",

    # Medical / test kits
    "blood", "test kit", "glucose", "diabetes", "pregnancy", "covid", "diagnostic",
    "medical", "thermometer", "stethoscope", "syringe", "needle",

    # Hazmat-ish / chemicals
    "aerosol", "propane", "butane", "fuel", "flammable", "corrosive", "acid", "bleach",
    "solvent", "paint thinner", "pesticide", "insecticide", "herbicide", "fertilizer",
    "lithium battery", "li-ion", "li ion",

    # Weapon-like
    "knife", "dagger", "sword", "machete", "pepper spray", "stun gun", "taser", "ammo",
    "firearm", "gun", "rifle", "pistol",

    # Adult
    "sex", "vibrator",
]

def normalize_domain(domain: str) -> str:
    return domain.strip().lower().replace("https://", "").replace("http://", "")

def risk_reasons(title: str) -> List[str]:
    t = (title or "").lower()
    reasons = [k for k in RISK_KEYWORDS if k in t]
    seen = set()
    out = []
    for r in reasons:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out

def parse_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(",", "")
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else None

def pick_title(item: Dict[str, Any]) -> str:
    return str(item.get("title") or item.get("name") or "").strip()

def pick_asin(item: Dict[str, Any]) -> str:
    return str(item.get("asin") or "").strip()

# -------------------------
# Rainforest helpers (cached)
# -------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def rf_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def estimate_requests(scan_mode: str, n_items: int, deep_reviews_n: int) -> int:
    base = 1  # one request for the main scan call
    if scan_mode == "Deep dive":
        base += min(n_items, deep_reviews_n)  # 1 request per item for reviews
    return base

def extract_categories(data: Any) -> List[Dict[str, str]]:
    if data is None:
        return []
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("categories") or data.get("results") or data.get("data") or []
    else:
        raw = []
    out: List[Dict[str, str]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("category_id") or "").strip()
        name = str(c.get("name") or c.get("title") or c.get("path") or "").strip()
        if cid:
            out.append({"id": cid, "name": name or cid})
    return out

def fetch_bestsellers(api_key: str, domain: str, category_id: str) -> List[Dict[str, Any]]:
    params = {
        "api_key": api_key,
        "type": "bestsellers",
        "amazon_domain": domain,
        "category_id": category_id,
    }
    data = rf_get(RF_REQUEST_URL, params)
    return data.get("bestsellers", []) or []

def fetch_critical_reviews(api_key: str, domain: str, asin: str, max_reviews: int = 25) -> List[str]:
    params = {
        "api_key": api_key,
        "type": "reviews",
        "amazon_domain": domain,
        "asin": asin,
        "review_stars": "all_critical",
        "sort_by": "most_recent",
    }
    data = rf_get(RF_REQUEST_URL, params)
    reviews = data.get("reviews", []) or []
    texts: List[str] = []
    for r in reviews[:max_reviews]:
        body = (r.get("body") or "").strip()
        if body:
            texts.append(body)
    return texts

# -------------------------
# Optional OpenAI (safe)
# -------------------------
def ai_gap_insights(title: str, critical_reviews: List[str]) -> str:
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY missing. Add it in Streamlit Secrets to enable AI insights."
    if not critical_reviews:
        return "No critical reviews found to analyze."
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return "OpenAI package not installed. Add `openai` to requirements.txt and redeploy."

    client = OpenAI(api_key=OPENAI_API_KEY)
    text = "\n\n".join(critical_reviews[:20])
    prompt = (
        f"Analyze these critical reviews for an Amazon product.\n\n"
        f"TITLE: {title}\n\n"
        f"CRITICAL REVIEWS:\n{text}\n\n"
        "Output:\n"
        "1) Top 3 specific complaints (bullets)\n"
        "2) One improved product blueprint (5-7 bullets)\n"
        "3) 2 bundle ideas (bullets)\n"
        "4) 2 listing angles (bullets)\n"
        "Keep it practical for resale/wholesale. Avoid illegal/medical claims."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# App wrapper (prevents blank pages)
# -------------------------
def main() -> None:
    st.title("🧭 Product Hunter Assistant (USA Market)")
    st.caption(
        "This app pulls Amazon Category Best Sellers (best-selling products in that category). "
        "Use Category Finder to get the correct bestsellers category_id. "
        "Risky items are excluded by default."
    )

    if not RAINFOREST_API_KEY:
        st.error("RAINFOREST_API_KEY is missing. Add it in Streamlit Secrets, then reboot.")
        st.stop()

    # session_state init (prevents your category picker bug)
    if "cat_search_results" not in st.session_state:
        st.session_state.cat_search_results = []
    if "category_id_input" not in st.session_state:
        st.session_state.category_id_input = ""

    with st.sidebar:
        st.header("A) Platform")
        domain = normalize_domain(
            st.selectbox("Amazon domain", ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de"], index=0)
        )

        st.header("B) Category Finder (recommended)")
        st.caption("Find the correct BESTSELLERS category_id (costs ~1 request per search).")
        search_term = st.text_input(
            "Search term for categories",
            value="",
            placeholder="e.g., Home & Kitchen / Pet / Tools",
        )

        col1, col2 = st.columns(2)
        do_cat_search = col1.button("Search categories")
        force_refresh = col2.checkbox("Force refresh", value=False, help="Bypass cache (may use extra requests).")

        if force_refresh:
            st.cache_data.clear()

        if do_cat_search:
            if not search_term.strip():
                st.warning("Type a category search term first (e.g., Home & Kitchen).")
            else:
                try:
                    params = {
                        "api_key": RAINFOREST_API_KEY,
                        "domain": domain,
                        "type": "bestsellers",
                        "search_term": search_term.strip(),
                    }
                    data = rf_get(RF_CATEGORIES_URL, params)
                    cats = extract_categories(data)
                    st.session_state.cat_search_results = cats
                    if not cats:
                        st.warning("No categories returned. Try: Kitchen / Pet / Tools / Storage.")
                except Exception as e:
                    st.error(f"Category search failed: {e}")

        def on_pick_category() -> None:
            sel = st.session_state.get("pick_cat_selectbox", "")
            if sel and "|" in sel:
                st.session_state.category_id_input = sel.split("|")[-1].strip()

        if st.session_state.cat_search_results:
            options = [f"{c['name']}  |  {c['id']}" for c in st.session_state.cat_search_results]
            st.selectbox(
                "Pick a category_id",
                options=options,
                index=0,
                key="pick_cat_selectbox",
                on_change=on_pick_category,
            )

        st.text_input(
            "Category id (BESTSELLERS)",
            key="category_id_input",
            help="Paste category_id from Category Finder (often like bestsellers_###########).",
        )

        st.header("C) Scan size + request planning")
        n_items = st.slider("How many items to scan", min_value=5, max_value=50, value=15, step=5)

        scan_mode = st.radio(
            "Scan mode",
            ["Fast scan", "Deep dive"],
            index=0,
            help="Deep dive fetches critical reviews for top items (extra requests).",
        )
        deep_reviews_n = st.slider("Deep dive: reviews for top N items", 1, 15, 5, 1, disabled=(scan_mode != "Deep dive"))
        enable_ai = st.checkbox("AI review-gap insights (OpenAI)", value=False, disabled=(scan_mode != "Deep dive"))

        exclude_risk = st.checkbox(
            "Exclude risk-flagged items (brand-heavy / supplements / medical-test / hazmat-ish / weapon-like)",
            value=True,
        )

        st.header("D) Target selling price range (optional)")
        price_range = st.slider("Price range filter", 0, 200, (0, 200), step=5)

        st.header("E) Profitability assumptions (NOT scan filters)")
        fee_pct = st.slider("Marketplace fee % (rough)", 5, 25, 15, 1)
        fixed_fee = st.number_input("Fixed fee per order ($)", 0.0, 10.0, 0.30, 0.05)
        ship_cost = st.number_input("Shipping cost ($)", 0.0, 50.0, 4.50, 0.50)
        pack_cost = st.number_input("Packaging cost ($)", 0.0, 10.0, 0.50, 0.10)
        desired_profit = st.number_input("Desired profit per unit ($)", 0.0, 100.0, 8.0, 0.50)

        est = estimate_requests("Deep dive" if scan_mode == "Deep dive" else "Fast scan", n_items, deep_reviews_n)
        st.info(f"Estimated Rainforest requests for next Run scan: **{est}** (Category Finder searches not included).")

        st.header("Keys status")
        st.write(("✅" if bool(RAINFOREST_API_KEY) else "❌") + " RAINFOREST_API_KEY")
        st.write(("✅" if bool(OPENAI_API_KEY) else "⚪") + " OPENAI_API_KEY (optional)")
        st.caption(f"Model: {OPENAI_MODEL}")

    run = st.button("🚀 Run scan", type="primary")

    def compute_profit_cols(sell_price: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if sell_price is None:
            return None, None, None
        fee_amt = sell_price * (fee_pct / 100.0)
        total_cost = fee_amt + fixed_fee + ship_cost + pack_cost + desired_profit
        max_buy = max(0.0, sell_price - total_cost)
        return round(fee_amt, 2), round(max_buy, 2), round(total_cost, 2)

    if not run:
        st.subheader("✅ Shortlist candidates (from bestsellers)")
        st.info("Click **Run scan** to generate results.")
        return

    category_id = st.session_state.get("category_id_input", "").strip()
    if not category_id:
        st.warning("Pick a category_id using Category Finder, then click Run scan.")
        st.stop()

    with st.spinner("Pulling category best sellers..."):
        try:
            items = fetch_bestsellers(RAINFOREST_API_KEY, domain, category_id)
        except Exception as e:
            st.error(f"Rainforest bestsellers request failed: {e}")
            st.stop()

    if not items:
        st.warning("No bestsellers returned. Usually the category_id is wrong. Use Category Finder to pick a BESTSELLERS category_id.")
        st.stop()

    rows: List[Dict[str, Any]] = []
    filtered_out = 0

    for idx, item in enumerate(items[:n_items], start=1):
        title = pick_title(item)
        asin = pick_asin(item)
        reasons = risk_reasons(title)

        if exclude_risk and reasons:
            filtered_out += 1
            continue

        p = None
        if isinstance(item.get("price"), dict):
            p = parse_price(item["price"].get("value") or item["price"].get("raw"))
        else:
            p = parse_price(item.get("price") or item.get("price_raw") or item.get("price_value"))

        fee_amt, max_buy, total_cost = compute_profit_cols(p)

        rows.append(
            {
                "rank": item.get("rank", idx),
                "title": title,
                "asin": asin,
                "amazon_price": p,
                "fee_est($)": fee_amt,
                "max_buy_price($)": max_buy,
                "assumption_total_cost($)": total_cost,
                "risk_reasons": ", ".join(reasons) if reasons else "",
                "source_category_id": category_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No items left after filtering. Try another category, or temporarily disable risk filter.")
        st.stop()

    min_p, max_p = price_range
    df["amazon_price"] = pd.to_numeric(df["amazon_price"], errors="coerce")
    df_show = df[(df["amazon_price"].isna()) | ((df["amazon_price"] >= min_p) & (df["amazon_price"] <= max_p))].copy()

    st.subheader("✅ Shortlist candidates")
    if exclude_risk and filtered_out:
        st.caption(f"Filtered out {filtered_out} items due to risk keywords.")

    st.dataframe(df_show, use_container_width=True)

    st.download_button(
        "Download CSV shortlist",
        df_show.to_csv(index=False).encode("utf-8"),
        file_name="product_shortlist.csv",
        mime="text/csv",
    )

    st.subheader("🧩 Supplier lead generator (US wholesalers first)")
    st.write("Copy/paste these Google queries:")
    top_titles = df_show["title"].head(5).tolist()
    for t in top_titles:
        st.code(
            "\n".join(
                [
                    f"wholesale \"{t}\" USA",
                    f"distributor \"{t}\" USA",
                    f"bulk supplier \"{t}\" USA",
                    f"\"{t}\" private label USA",
                    f"site:faire.com \"{t}\"",
                    f"site:thomasnet.com \"{t}\"",
                ]
            ),
            language="text",
        )

    st.subheader("🟦 eBay sold-price check (manual workflow)")
    st.write("eBay → search product → enable **Sold items** + **Completed items** → compare sold prices vs Max Buy Price.")
    st.code("\n".join([f"eBay sold listings \"{t}\"" for t in top_titles]), language="text")

    if scan_mode == "Deep dive":
        st.subheader("🔍 Deep dive (critical reviews)")
        top_for_reviews = df_show.head(deep_reviews_n).to_dict("records")
        for rec in top_for_reviews:
            title = rec["title"]
            asin = rec["asin"]
            if not asin:
                st.info(f"Skipping reviews (missing ASIN): {title[:60]}")
                continue
            with st.spinner(f"Fetching critical reviews: {title[:60]}..."):
                try:
                    crit = fetch_critical_reviews(RAINFOREST_API_KEY, domain, asin, max_reviews=25)
                except Exception as e:
                    st.warning(f"Could not fetch reviews for {asin}: {e}")
                    continue

            st.markdown(f"**{title}**  \nASIN: `{asin}`  \nCritical reviews fetched: {len(crit)}")
            if enable_ai:
                with st.spinner("Running AI gap analysis..."):
                    st.write(ai_gap_insights(title, crit))
            else:
                st.caption("AI insights disabled.")

    with st.expander("How to use your 100-request trial without wasting it"):
        st.markdown(
            """
- Think of **1 API call = 1 request**. A single **Run scan** can cost **1+** requests depending on Deep Dive.
- Start wide: **Fast scan + 15 items** across many categories.
- Go deep only on winners: Deep dive **3–5 items** for review-gap insights.
- Use Category Finder sparingly: search broad terms, then reuse the found category_id.
            """
        )

# Catch-all exception -> shows on page (prevents blank)
try:
    main()
except Exception as e:
    st.error("The app crashed while rendering. The exact error is shown below.")
    st.exception(e)
