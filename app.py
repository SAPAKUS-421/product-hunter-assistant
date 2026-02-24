import os
import math
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import streamlit as st

# Optional AI (won't break if openai isn't installed)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Product Hunter Assistant (USA Market)", layout="wide")


# -----------------------------
# Secrets helpers
# -----------------------------
def get_secret(key: str, default: str = "") -> str:
    """Reads Streamlit Secrets first, then environment variables."""
    try:
        if key in st.secrets:
            return str(st.secrets.get(key, default)).strip()
    except Exception:
        pass
    return str(os.getenv(key, default)).strip()


RAINFOREST_API_KEY = get_secret("RAINFOREST_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")


# -----------------------------
# Rainforest API helpers
# -----------------------------
RF_ENDPOINT = "https://api.rainforestapi.com/request"


def rf_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Safe Rainforest API GET with good error messages."""
    if not RAINFOREST_API_KEY:
        raise RuntimeError("Missing RAINFOREST_API_KEY in Streamlit Secrets.")

    params = dict(params)
    params["api_key"] = RAINFOREST_API_KEY

    r = requests.get(RF_ENDPOINT, params=params, timeout=45)
    # Helpful error text on 401
    if r.status_code == 401:
        raise RuntimeError(
            "Rainforest returned 401 Unauthorized. Your RAINFOREST_API_KEY is missing/incorrect, "
            "or you pasted the placeholder text."
        )
    r.raise_for_status()
    return r.json()


def parse_price_value(obj: Any) -> Optional[float]:
    """
    Rainforest sometimes returns price like:
      price: { value: 19.99, currency: 'USD' }
    or other shapes. We'll try common patterns.
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for k in ["value", "raw", "amount"]:
                if k in obj and obj[k] is not None:
                    return float(str(obj[k]).replace("$", "").strip())
        if isinstance(obj, str):
            return float(obj.replace("$", "").strip())
    except Exception:
        return None
    return None


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# -----------------------------
# AI helper (optional)
# -----------------------------
def ai_gap_analysis(product_title: str, critical_text: str) -> str:
    """Summarize top complaints + improvement blueprint."""
    if not OPENAI_API_KEY:
        return "OpenAI key missing. Add OPENAI_API_KEY in Secrets to enable AI insights."
    if OpenAI is None:
        return "openai package not installed. Add 'openai' to requirements.txt to enable AI insights."

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You are a practical product research assistant for USA e-commerce resellers.

Product: {product_title}

Critical feedback snippets:
{critical_text}

Return:
1) Top 3 specific complaints (bullet points).
2) A simple 'better product blueprint' (bullet points).
3) Any red flags for reselling (brand gating / safety / compliance) in 1-2 bullets.
Keep it short and actionable.
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# UI
# -----------------------------
st.title("🧭 Product Hunter Assistant (USA Market)")
st.caption(
    "Pull Amazon bestsellers via Rainforest, score opportunities for resale/wholesale, "
    "and generate sourcing leads. (Optional AI review-gap insights.)"
)

# Sidebar controls
st.sidebar.header("A) Source")
amazon_domain = st.sidebar.selectbox("Amazon domain", ["amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de"], index=0)
category_id = st.sidebar.text_input("Rainforest category_id", value="electronics")
scan_n = st.sidebar.slider("How many bestsellers to scan", min_value=5, max_value=50, value=15, step=1)

st.sidebar.header("B) Your selling assumptions")
marketplace = st.sidebar.selectbox("Marketplace you plan to sell on (for fee math)", ["Amazon", "eBay", "Etsy"], index=0)

sell_price_mode = st.sidebar.selectbox(
    "Selling price for profit math",
    ["Use Amazon price (rough)", "Use my target selling price"],
    index=0,
)

target_price = st.sidebar.number_input("Target selling price ($)", min_value=1.0, value=24.99, step=0.50)

fee_pct_default = 15.0 if marketplace == "Amazon" else (13.0 if marketplace == "eBay" else 10.0)
fee_pct = st.sidebar.slider(f"Marketplace fee % ({marketplace})", min_value=0, max_value=30, value=int(fee_pct_default))
fixed_fee = st.sidebar.number_input("Fixed fee per order ($)", min_value=0.0, value=0.30, step=0.10)
shipping_cost = st.sidebar.number_input("Shipping cost ($)", min_value=0.0, value=4.50, step=0.50)
packaging_cost = st.sidebar.number_input("Packaging cost ($)", min_value=0.0, value=0.50, step=0.10)

st.sidebar.header("C) Sourcing assumptions")
expected_discount = st.sidebar.slider(
    "Expected sourcing discount vs selling price (%)",
    min_value=5,
    max_value=80,
    value=35,
    help="Rough guess: if you can source at 35% below your selling price, what does profit look like?",
)

desired_profit = st.sidebar.number_input(
    "Desired profit per unit ($) (used to compute Max Buy Price)",
    min_value=0.0,
    value=3.00,
    step=0.50,
)

st.sidebar.header("D) Optional AI")
use_ai = st.sidebar.checkbox("Enable AI review-gap insights (costs OpenAI + Rainforest review credits)", value=False)
ai_top_n = st.sidebar.slider("Fetch critical reviews for top N products", min_value=0, max_value=10, value=3, step=1)

# Key status
st.sidebar.markdown("---")
st.sidebar.subheader("Keys status")
st.sidebar.write(("✅" if RAINFOREST_API_KEY else "❌") + " RAINFOREST_API_KEY")
st.sidebar.write(("✅" if OPENAI_API_KEY else "⚠️") + " OPENAI_API_KEY (optional)")
if OPENAI_MODEL:
    st.sidebar.write(f"Model: `{OPENAI_MODEL}`")


# -----------------------------
# Run scan (IMPORTANT: rows only exists inside this block)
# -----------------------------
run_scan = st.button("🚀 Run scan")

if run_scan:
    with st.spinner("Pulling Amazon bestsellers from Rainforest..."):
        try:
            data = rf_get(
                {
                    "type": "bestsellers",
                    "amazon_domain": amazon_domain,
                    "category_id": category_id,
                }
            )
        except Exception as e:
            st.error(f"Rainforest error: {e}")
            st.stop()

    bestsellers = data.get("bestsellers", []) or data.get("best_sellers", []) or []

    if not bestsellers:
        st.warning("No bestsellers returned. Try another category_id (example: 'electronics', 'home_and_kitchen').")
        st.stop()

    rows: List[Dict[str, Any]] = []

    # Process items
    for item in bestsellers[:scan_n]:
        title = item.get("title") or item.get("name") or ""
        asin = item.get("asin") or ""
        rank = item.get("rank") or item.get("position") or None
        link = item.get("link") or item.get("url") or ""
        brand = item.get("brand") or item.get("brand_name") or ""

        # Price extraction (best effort)
        amazon_price = (
            parse_price_value(item.get("price"))
            or parse_price_value(safe_get(item, ["prices", "price"]))
            or parse_price_value(safe_get(item, ["prices", 0, "price"]))  # sometimes list
            or parse_price_value(item.get("price_raw"))
        )

        # Reviews/rating extraction (best effort)
        rating = (
            safe_get(item, ["rating"])
            or safe_get(item, ["reviews", "rating"])
            or safe_get(item, ["customer_reviews", "rating"])
        )
        reviews_count = (
            safe_get(item, ["reviews_total"])
            or safe_get(item, ["reviews", "total"])
            or safe_get(item, ["customer_reviews", "total_reviews"])
            or 0
        )
        try:
            reviews_count = int(str(reviews_count).replace(",", "").strip())
        except Exception:
            reviews_count = 0

        # Selling price used for profit math
        if sell_price_mode == "Use Amazon price (rough)" and amazon_price:
            sell_price = float(amazon_price)
        else:
            sell_price = float(target_price)

        # Cost guess based on expected discount
        est_buy_price = sell_price * (1 - expected_discount / 100.0)

        # Fee math
        est_fee = sell_price * (fee_pct / 100.0) + float(fixed_fee)
        est_profit = sell_price - est_fee - float(shipping_cost) - float(packaging_cost) - est_buy_price
        est_margin_pct = (est_profit / sell_price * 100.0) if sell_price > 0 else 0.0

        # Max buy price to still hit desired profit
        max_buy_price = sell_price - est_fee - float(shipping_cost) - float(packaging_cost) - float(desired_profit)

        # Scoring
        # demand_score: higher when rank is closer to 1
        try:
            r = int(rank) if rank is not None else None
        except Exception:
            r = None
        if r is None:
            demand_score = 50.0
        else:
            # map 1..scan_n into 100..0-ish
            demand_score = max(0.0, 100.0 * (1.0 - (r - 1) / max(1, (scan_n - 1))))

        # competition_score: higher when review count is high
        comp_raw = math.log10(reviews_count + 1)
        competition_score = min(100.0, comp_raw / 5.0 * 100.0)  # rough normalization

        # margin_score: cap at 0..100
        margin_score = max(0.0, min(100.0, (est_margin_pct + 20.0) * 2.0))  # shifts negative margins down

        # risk penalty (very simple)
        risk_flags = []
        risk_penalty = 0.0
        if brand and brand.lower() not in ["generic", "unbranded"]:
            risk_flags.append("Brand/gating risk")
            risk_penalty += 8.0
        if sell_price < 10:
            risk_flags.append("Low price (fees hurt margin)")
            risk_penalty += 5.0

        # overall score
        overall_score = (
            0.45 * demand_score
            + 0.35 * margin_score
            + 0.20 * (100.0 - competition_score)
            - risk_penalty
        )

        rows.append(
            {
                "overall_score": round(overall_score, 2),
                "title": title,
                "asin": asin,
                "rank": rank,
                "brand": brand,
                "amazon_price": round(float(amazon_price), 2) if amazon_price else None,
                "sell_price_used": round(sell_price, 2),
                "expected_buy_price": round(est_buy_price, 2),
                "max_buy_price_for_target_profit": round(max_buy_price, 2),
                "est_profit": round(est_profit, 2),
                "est_margin_%": round(est_margin_pct, 2),
                "rating": rating,
                "reviews_count": reviews_count,
                "competition_score": round(competition_score, 2),
                "demand_score": round(demand_score, 2),
                "risk_flags": ", ".join(risk_flags) if risk_flags else "",
                "link": link,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        st.warning("No products processed. Try another category_id or reduce scan size.")
        st.stop()

    df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    st.session_state["last_df"] = df

    st.success(f"Done. Scanned {len(df)} products from category_id='{category_id}' on {amazon_domain}.")


# -----------------------------
# Display section (SAFE: uses session_state only)
# -----------------------------
st.subheader("✅ Shortlist")
st.write("This is your **ranked list**. Use it to pick the top candidates worth sourcing + testing first.")

df_last = st.session_state.get("last_df")

if df_last is None:
    st.info('Click **"Run scan"** to generate results.')
else:
    st.dataframe(df_last, use_container_width=True)

    st.download_button(
        "Download CSV shortlist",
        df_last.to_csv(index=False).encode("utf-8"),
        file_name="product_shortlist.csv",
        mime="text/csv",
    )

    st.subheader("🔎 Sourcing lead generator (US wholesalers first)")
    st.write("For the top products, copy/paste these into Google (or your supplier sites).")

    top_titles = df_last["title"].head(5).tolist()
    for t in top_titles:
        st.markdown(f"**{t}**")
        st.code(
            "\n".join(
                [
                    f'"{t}" wholesale USA',
                    f'"{t}" distributor USA',
                    f'"{t}" bulk supplier USA',
                    f'site:faire.com "{t}"',
                    f'site:tundra.com "{t}"',
                    f'site:thomasnet.com "{t}"',
                ]
            ),
            language="text",
        )

    if use_ai and ai_top_n > 0:
        st.subheader("🧠 AI review-gap insights (optional)")
        st.write("This pulls **critical reviews** for top products and summarizes gaps. This can consume Rainforest credits.")

        for i in range(min(ai_top_n, len(df_last))):
            asin = str(df_last.loc[i, "asin"])
            title = str(df_last.loc[i, "title"])

            if not asin or asin.lower() == "nan":
                continue

            with st.spinner(f"Fetching critical reviews for: {title[:60]}..."):
                try:
                    reviews_json = rf_get(
                        {
                            "type": "reviews",
                            "amazon_domain": amazon_domain,
                            "asin": asin,
                            "review_stars": "all_critical",
                            "sort_by": "most_recent",
                        }
                    )
                except Exception as e:
                    st.warning(f"Could not fetch reviews for {asin}: {e}")
                    continue

            reviews = reviews_json.get("reviews", []) or []
            snippets = []
            for r in reviews[:20]:
                body = r.get("body") or r.get("review") or ""
                if body:
                    snippets.append(body.strip())

            critical_text = "\n\n".join(snippets) if snippets else "No review text returned."

            with st.spinner("Running AI summary..."):
                try:
                    analysis = ai_gap_analysis(title, critical_text)
                except Exception as e:
                    analysis = f"AI error: {e}"

            with st.expander(f"AI insights: {title}"):
                st.write(analysis)
