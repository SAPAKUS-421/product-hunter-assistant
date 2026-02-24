import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# Helpers: Secrets + Safety
# =========================

def get_secret(key: str, default: str = "") -> str:
    """Fetch from Streamlit secrets first, then env vars."""
    try:
        val = str(st.secrets.get(key, default)).strip()
    except Exception:
        val = str(os.getenv(key, default)).strip()
    return val


RAINFOREST_API_KEY = get_secret("RAINFOREST_API_KEY", "")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = re.sub(r"[^\d.]", "", s)
    try:
        return float(s) if s else None
    except Exception:
        return None


# =========================
# Rainforest API
# =========================

RAINFOREST_ENDPOINT = "https://api.rainforestapi.com/request"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def rf_bestsellers(amazon_domain: str, category_id: str, api_key: str) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "type": "bestsellers",
        "amazon_domain": amazon_domain,
        "category_id": category_id,
    }
    r = requests.get(RAINFOREST_ENDPOINT, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=60 * 30)
def rf_reviews(amazon_domain: str, asin: str, api_key: str, limit: int = 40) -> Dict[str, Any]:
    params = {
        "api_key": api_key,
        "type": "reviews",
        "amazon_domain": amazon_domain,
        "asin": asin,
        "sort_by": "most_recent",
        "review_stars": "all_critical",
        "max_page": 1,  # Rainforest handles paging; keep it light
    }
    r = requests.get(RAINFOREST_ENDPOINT, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()
    # Keep it bounded
    if isinstance(data.get("reviews"), list):
        data["reviews"] = data["reviews"][:limit]
    return data


def parse_bestsellers(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = data.get("bestsellers") or data.get("best_sellers") or []
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append(it)
    return out


def extract_price(item: Dict[str, Any]) -> Optional[float]:
    # Rainforest sometimes uses price.value or price.raw / current_price.value
    candidates = [
        _safe_get(item, "price", "value"),
        _safe_get(item, "price", "raw"),
        _safe_get(item, "current_price", "value"),
        _safe_get(item, "current_price", "raw"),
        item.get("price"),
        item.get("current_price"),
    ]
    for c in candidates:
        v = _to_float(c)
        if v is not None and v > 0:
            return v
    return None


def extract_rating_reviews(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[int]]:
    rating = _to_float(item.get("rating") or _safe_get(item, "rating", "value"))
    reviews = item.get("ratings_total") or item.get("reviews_total") or item.get("total_reviews")
    try:
        reviews_int = int(str(reviews).replace(",", "")) if reviews is not None else None
    except Exception:
        reviews_int = None
    return rating, reviews_int


# =========================
# Scoring (Product Hunting)
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_log(x: float, x_min: float, x_max: float) -> float:
    """0..1 log normalization."""
    x = clamp(x, x_min, x_max)
    a = math.log1p(x_min)
    b = math.log1p(x_max)
    return 0.0 if b == a else (math.log1p(x) - a) / (b - a)


def demand_score(rank: Optional[int], rating: Optional[float], reviews: Optional[int]) -> float:
    # Lower rank = more demand. Use log scaling.
    if rank is None:
        r = 0.35
    else:
        # rank 1..100000 -> map (1 high demand)
        r = 1.0 - normalize_log(rank, 1, 100000)

    rat = 0.55 if rating is None else clamp((rating - 3.0) / 2.0, 0.0, 1.0)  # 3.0->0, 5.0->1
    rev = 0.45 if reviews is None else normalize_log(reviews, 1, 50000)

    # Weighted: rank most important
    return clamp(0.55 * r + 0.25 * rat + 0.20 * rev, 0.0, 1.0) * 100.0


def competition_score(reviews: Optional[int], rating: Optional[float]) -> float:
    """
    Higher score = easier competition (we want LOW entrenched review moat).
    Fewer reviews generally = less entrenched competition.
    """
    if reviews is None:
        rev_moat = 0.55
    else:
        # 1..50000 -> moat strength
        moat = normalize_log(reviews, 1, 50000)  # 0 low moat, 1 high moat
        rev_moat = 1.0 - moat

    # Very high rating can indicate strong incumbents; slightly lower might be opportunity.
    if rating is None:
        rat_opportunity = 0.55
    else:
        rat_opportunity = 1.0 - clamp((rating - 4.2) / 0.8, 0.0, 1.0)  # 4.2..5.0 -> incumbency

    return clamp(0.75 * rev_moat + 0.25 * rat_opportunity, 0.0, 1.0) * 100.0


def profit_math(
    sell_price: float,
    buy_cost: float,
    fee_pct: float,
    fee_fixed: float,
    ship_cost: float,
    pack_cost: float,
    misc_cost: float,
) -> Tuple[float, float]:
    fees = (sell_price * (fee_pct / 100.0)) + fee_fixed
    profit = sell_price - (buy_cost + fees + ship_cost + pack_cost + misc_cost)
    margin = 0.0 if sell_price <= 0 else (profit / sell_price) * 100.0
    return profit, margin


def profit_score(profit: float, margin_pct: float) -> float:
    """
    0..100
    - reward positive profit and healthy margins
    - penalize negative profit hard
    """
    if profit <= 0:
        return clamp(20 + profit, 0, 25)  # negative profit drives score down
    # profit $: 0..20, margin%: 0..40 typical
    p = clamp(profit / 20.0, 0.0, 1.0)
    m = clamp(margin_pct / 40.0, 0.0, 1.0)
    return clamp((0.55 * p + 0.45 * m) * 100.0, 0.0, 100.0)


def overall_score(demand: float, profit_s: float, competition: float) -> float:
    # Demand + Profit heavy; Competition as tie-breaker.
    return clamp(0.45 * demand + 0.40 * profit_s + 0.15 * competition, 0.0, 100.0)


# =========================
# Optional OpenAI (AI Gap Analysis)
# =========================

def openai_analyze(product_title: str, critical_reviews: List[str], model: str) -> str:
    """
    Works with both new OpenAI python SDK and legacy openai.
    If OpenAI is not configured or fails, returns a readable error string.
    """
    text = "\n\n".join([f"- {t[:600]}" for t in critical_reviews[:18]])
    prompt = f"""
You are a ruthless product researcher focused on USA marketplace selling.

Product: {product_title}

Below are recent CRITICAL reviews. Extract actionable insights for improving or differentiating:
{text}

Return:
1) Top 5 complaints (very specific).
2) "Opportunity blueprint" (what exact improvements to make).
3) Packaging / bundling ideas for higher AOV.
4) Biggest risk flags (returns, breakage, compliance, IP, gating).
Keep it concise, structured, and practical.
""".strip()

    # Try new SDK first
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    # Fallback: legacy SDK
    try:
        import openai  # type: ignore
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=700,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI analysis failed: {e}"


# =========================
# UI
# =========================

st.set_page_config(page_title="Product Hunter Assistant (USA)", layout="wide")

st.title("🧭 Product Hunter Assistant (USA Market)")
st.caption(
    "Pull Amazon bestsellers via Rainforest, score opportunities for resale/wholesale, and (optionally) extract gaps from critical reviews."
)

with st.sidebar:
    st.header("A) Source")
    amazon_domain = st.selectbox("Amazon domain", ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de"], index=0)
    category_id = st.text_input("Rainforest category_id", value="electronics", help="Example: electronics, home_and_kitchen, toys_and_games (depends on Rainforest).")
    scan_n = st.slider("How many bestsellers to scan", min_value=5, max_value=60, value=15, step=1)

    st.divider()
    st.header("B) Selling assumptions")
    sell_price_mode = st.selectbox(
        "Selling price for profit math",
        ["Use Amazon price (rough)", "Use my target selling price"],
        index=0,
    )
    target_sell = st.number_input("Target selling price ($)", min_value=1.0, value=24.99, step=0.50)
    fee_pct = st.slider("Marketplace fee % (eBay/Amazon etc.)", min_value=5.0, max_value=25.0, value=15.0, step=0.5)
    fee_fixed = st.number_input("Fixed fee per order ($)", min_value=0.0, value=0.30, step=0.05)

    ship_cost = st.number_input("Shipping cost ($)", min_value=0.0, value=4.50, step=0.25)
    pack_cost = st.number_input("Packaging cost ($)", min_value=0.0, value=0.50, step=0.10)
    misc_cost = st.number_input("Misc cost ($) (labels/returns buffer)", min_value=0.0, value=0.75, step=0.25)

    st.divider()
    st.header("C) Buy-cost model (COGS estimate)")
    buy_cost_mode = st.selectbox(
        "How to estimate buy cost",
        ["Use Amazon price as buy cost (worst case)", "Use % of Amazon price", "Manual buy cost (same for all)"],
        index=1,
    )
    buy_pct = st.slider("Buy cost % of Amazon price", min_value=10, max_value=95, value=60, step=1)
    manual_buy = st.number_input("Manual buy cost ($)", min_value=0.10, value=8.00, step=0.25)

    st.divider()
    st.header("D) AI (optional)")
    use_ai = st.toggle("Use AI review-gap analysis (costs OpenAI tokens)", value=False)
    ai_top_k = st.slider("AI analyze top K results", min_value=1, max_value=8, value=3, step=1)
    reviews_limit = st.slider("Critical reviews to fetch per ASIN", min_value=10, max_value=80, value=35, step=5)

    st.divider()
    st.caption("Keys are read from Streamlit Secrets: RAINFOREST_API_KEY, OPENAI_API_KEY (optional), OPENAI_MODEL (optional).")


col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    st.subheader("Run")
    run_scan = st.button("🚀 Run scan", type="primary")

with col_right:
    with st.expander("✅ What the 'Install' button is", expanded=False):
        st.write(
            "That **Install** button is just your browser offering to install this Streamlit page as an app shortcut (PWA). "
            "It’s optional. You can ignore it safely."
        )


# =========================
# Run Scan Logic (ONLY executes when button clicked)
# =========================

if run_scan:
    if not RAINFOREST_API_KEY:
        st.error("Missing RAINFOREST_API_KEY. Go to Streamlit → Manage app → Settings → Secrets and add it.")
        st.stop()

    st.info("Pulling Amazon bestsellers from Rainforest…")

    try:
        data = rf_bestsellers(amazon_domain, category_id.strip(), RAINFOREST_API_KEY)
    except requests.HTTPError as e:
        # Better message for common 401/403
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            st.error("Rainforest returned 401 Unauthorized. This usually means your RAINFOREST_API_KEY is wrong or not active.")
        else:
            st.error(f"Rainforest HTTP error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Rainforest request failed: {e}")
        st.stop()

    items = parse_bestsellers(data)
    if not items:
        st.warning("No bestsellers returned. Try another category_id.")
        st.session_state["last_df"] = None
        st.stop()

    items = items[: int(scan_n)]

    rows: List[Dict[str, Any]] = []

    for it in items:
        asin = it.get("asin") or it.get("ASIN")
        title = it.get("title") or it.get("name") or ""
        rank = it.get("rank")
        try:
            rank_int = int(rank) if rank is not None else None
        except Exception:
            rank_int = None

        amazon_price = extract_price(it)
        rating, reviews = extract_rating_reviews(it)

        # Selling price assumption
        if sell_price_mode == "Use Amazon price (rough)":
            sell_price = float(amazon_price) if amazon_price else float(target_sell)
        else:
            sell_price = float(target_sell)

        # Buy-cost assumption
        if buy_cost_mode == "Use Amazon price as buy cost (worst case)":
            buy_cost = float(amazon_price) if amazon_price else float(manual_buy)
        elif buy_cost_mode == "Use % of Amazon price":
            base = float(amazon_price) if amazon_price else float(manual_buy)
            buy_cost = base * (float(buy_pct) / 100.0)
        else:
            buy_cost = float(manual_buy)

        profit, margin = profit_math(
            sell_price=sell_price,
            buy_cost=buy_cost,
            fee_pct=float(fee_pct),
            fee_fixed=float(fee_fixed),
            ship_cost=float(ship_cost),
            pack_cost=float(pack_cost),
            misc_cost=float(misc_cost),
        )

        d_score = demand_score(rank_int, rating, reviews)
        c_score = competition_score(reviews, rating)
        p_score = profit_score(profit, margin)
        o_score = overall_score(d_score, p_score, c_score)

        flags = []
        if profit < 3:
            flags.append("Low $ profit")
        if margin < 15:
            flags.append("Low margin")
        if rating is not None and rating < 4.1:
            flags.append("Rating risk")
        if reviews is not None and reviews > 8000:
            flags.append("Strong incumbents")
        if title and any(w in title.lower() for w in ["vitamin", "supplement", "blood", "test", "medical"]):
            flags.append("Regulated/gated risk")

        rows.append(
            {
                "overall_score": round(o_score, 2),
                "demand_score": round(d_score, 2),
                "profit_score": round(p_score, 2),
                "competition_score": round(c_score, 2),
                "title": title,
                "asin": asin,
                "rank": rank_int,
                "amazon_price": round(amazon_price, 2) if amazon_price else None,
                "assumed_sell_price": round(sell_price, 2),
                "assumed_buy_cost": round(buy_cost, 2),
                "est_profit_$": round(profit, 2),
                "est_margin_%": round(margin, 2),
                "rating": rating,
                "reviews": reviews,
                "risk_flags": "; ".join(flags),
            }
        )

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if df.empty:
        st.warning("No products returned. Try another category_id or reduce scan size, then click Run scan again.")
        st.session_state["last_df"] = None
        st.stop()

    # Sort safely
    if "overall_score" in df.columns:
        df = df.sort_values("overall_score", ascending=False)
    else:
        # fallback
        for c in ["profit_score", "demand_score"]:
            if c in df.columns:
                df = df.sort_values(c, ascending=False)
                break

    st.session_state["last_df"] = df
    st.success("Scan complete. Results are below 👇")


# =========================
# DISPLAY (always safe)
# =========================

df_show = st.session_state.get("last_df", None)

st.subheader("✅ Shortlist")
if df_show is None or not isinstance(df_show, pd.DataFrame) or df_show.empty:
    st.info("Click **Run scan** to generate results.")
else:
    st.dataframe(df_show, use_container_width=True)

    st.download_button(
        "Download CSV shortlist",
        df_show.to_csv(index=False).encode("utf-8"),
        file_name="product_shortlist.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🔎 US Sourcing lead generator (wholesalers first)")

    st.write("For the top products, copy/paste these searches into Google (avoids Amazon/eBay as sources):")

    top_titles = df_show["title"].head(6).fillna("").tolist()

    for t in top_titles:
        t = t.strip()
        if not t:
            continue

        st.markdown(f"**{t}**")
        st.code(
            "\n".join(
                [
                    f'wholesale "{t}" USA',
                    f'"{t}" distributor USA',
                    f'"{t}" bulk supplier USA',
                    f'site:faire.com "{t}"',
                    f'site:tundra.com "{t}"',
                    f'site:thomasnet.com "{t}"',
                    f'site:wholesalecentral.com "{t}"',
                    f'site:catalog.com "{t}"',
                ]
            ),
            language="text",
        )

    st.divider()
    st.subheader("🧠 AI review-gap analysis (optional)")

    if use_ai:
        if not OPENAI_API_KEY:
            st.warning("AI is ON but OPENAI_API_KEY is missing in Secrets. Add it or turn AI OFF.")
        else:
            # Let user choose which rows to analyze
            max_rows = min(int(ai_top_k), len(df_show))
            st.write(f"AI will analyze critical reviews for the top **{max_rows}** results (by overall_score).")

            for i in range(max_rows):
                row = df_show.iloc[i]
                asin = row.get("asin")
                title = row.get("title", "")

                if not asin or not isinstance(asin, str):
                    continue

                with st.expander(f"AI: {title} (ASIN: {asin})", expanded=(i == 0)):
                    try:
                        rev_data = rf_reviews(amazon_domain, asin, RAINFOREST_API_KEY, limit=int(reviews_limit))
                        reviews_list = rev_data.get("reviews", []) if isinstance(rev_data, dict) else []
                        bodies = []
                        for r in reviews_list:
                            if isinstance(r, dict) and r.get("body"):
                                bodies.append(str(r["body"]))
                        if not bodies:
                            st.info("No critical reviews returned for this ASIN (or API plan limit).")
                            continue

                        ai_text = openai_analyze(title, bodies, OPENAI_MODEL)
                        st.markdown(ai_text)
                    except requests.HTTPError as e:
                        st.error(f"Rainforest reviews HTTP error: {e}")
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
    else:
        st.info("Turn ON AI in the sidebar if you want the app to analyze critical reviews and suggest improvements/bundles.")

st.divider()
with st.expander("📌 Notes (important for real-world selling)", expanded=False):
    st.write(
        "- Some items are **gated / restricted** on Amazon (supplements, medical tests, certain brands). Treat those as higher risk.\n"
        "- For eBay, brand/IP can also cause issues (YETI, Disney, etc.).\n"
        "- This tool is a **scanner**. You still confirm suppliers, MAP policies, authenticity, and return rates before buying inventory."
    )
