import math
import requests
import pandas as pd
import streamlit as st
import os

# ---------------------------
# Secrets (Cloud-ready)
# Streamlit Community Cloud: paste these in Advanced settings → Secrets
# ---------------------------
def get_secret(key: str, default: str = "") -> str:
    # st.secrets is the recommended way on Streamlit Cloud
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return os.getenv(key, default).strip()

RAINFOREST_API_KEY = get_secret("RAINFOREST_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

RF_REQUEST_ENDPOINT = "https://api.rainforestapi.com/request"

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def rf_get(params: dict) -> dict:
    """Rainforest API request wrapper (cached)."""
    if not RAINFOREST_API_KEY:
        raise ValueError("Missing RAINFOREST_API_KEY (add it in Streamlit Secrets).")
    q = dict(params)
    q["api_key"] = RAINFOREST_API_KEY
    r = requests.get(RF_REQUEST_ENDPOINT, params=q, timeout=40)
    r.raise_for_status()
    return r.json()

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(str(x).replace("$", "").replace(",", "").strip())
    except Exception:
        return default

def demand_score_from_rank(rank: int, cap: int = 100) -> float:
    """0–100. Lower rank = higher demand score."""
    if not rank or rank <= 0:
        return 0.0
    r = min(int(rank), cap)
    return round(100 * (1 - (r - 1) / (cap - 1)), 2)

def competition_score_from_reviews(reviews: int) -> float:
    """0–100. Higher reviews = higher competition score (log-scaled)."""
    if not reviews or reviews <= 0:
        return 0.0
    return round(min(math.log10(reviews + 1) / 5, 1) * 100, 2)

def risk_flags(title: str) -> list:
    """Basic risk screening (you should still validate marketplace rules)."""
    t = (title or "").lower()
    flags = []
    # High-compliance / restriction-prone
    risky = ["vitamin", "supplement", "blood test", "test kit", "medical", "drug", "diagnostic"]
    if any(w in t for w in risky):
        flags.append("Compliance-risk (supplement/medical/test-like)")
    # Weapon-like keywords (avoid when starting)
    weaponish = ["knife", "dagger", "sword", "tactical", "weapon"]
    if any(w in t for w in weaponish):
        flags.append("Restricted-risk (weapon-like)")
    return flags

def extract_price_from_bestseller_item(it: dict) -> float:
    # Rainforest returns price in different shapes depending on endpoint
    price_obj = it.get("price") or {}
    # common fields seen in Rainforest-type responses
    return (
        safe_float(price_obj.get("value"), 0.0)
        or safe_float(price_obj.get("raw"), 0.0)
        or safe_float(it.get("price_value"), 0.0)
        or safe_float(it.get("price"), 0.0)
    )

def ai_gap_analysis(title: str, critical_text: str) -> str:
    if not OPENAI_API_KEY:
        return "AI is OFF (missing OPENAI_API_KEY)."

    # New OpenAI python SDK style
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You are a USA e-commerce product researcher.

Product: {title}

Critical review text:
{critical_text}

Return:
1) Top 3 specific complaints (bullets)
2) A better product blueprint (features/materials/packaging)
3) A low-risk differentiation angle (no medical claims, no exaggerated promises)
Keep it concise.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Product Hunter (USA)", layout="wide")
st.title("🧭 Product Hunter Assistant (USA Market)")

with st.sidebar:
    st.header("A) Source")
    amazon_domain = st.selectbox("Amazon domain", ["amazon.com", "amazon.ca", "amazon.co.uk"], index=0)
    category_id = st.text_input("Rainforest category_id", value="electronics")
    limit = st.slider("How many bestsellers to scan", 5, 50, 15)

    st.header("B) Your selling assumptions")
    selling_price_mode = st.selectbox("Selling price for profit math", ["Use Amazon price (rough)", "I will type a target selling price"], index=0)
    target_sell_price = st.number_input("Target selling price ($)", min_value=0.0, value=24.99, step=1.0)

    marketplace_fee_pct = st.slider("Marketplace fee % (eBay/Amazon etc.)", 5, 25, 15)
    fixed_fee = st.number_input("Fixed fee per order ($)", min_value=0.0, value=0.30, step=0.05)
    shipping_cost = st.number_input("Shipping cost ($)", min_value=0.0, value=4.50, step=0.50)
    packaging_cost = st.number_input("Packaging cost ($)", min_value=0.0, value=0.50, step=0.25)

    st.header("C) COGS model (inventory cost)")
    cogs_mode = st.selectbox("COGS method", ["Assume % of selling price", "I will type COGS later"], index=0)
    cogs_pct = st.slider("Assumed COGS %", 10, 90, 55)

    st.header("D) Optional AI")
    use_ai = st.checkbox("Run AI gap analysis (costs OpenAI)", value=False)
    ai_top_n = st.slider("AI analyze top N products", 1, 5, 3)

run = st.button("🚀 Run scan")

if run:
    if not RAINFOREST_API_KEY:
        st.error("Add RAINFOREST_API_KEY in Streamlit Secrets, then rerun.")
        st.stop()

    st.info("Pulling Amazon bestsellers from Rainforest…")
    try:
        data = rf_get({
            "type": "bestsellers",
            "amazon_domain": amazon_domain,
            "category_id": category_id
        })
    except Exception as e:
        st.error(f"Rainforest error: {e}")
        st.stop()

    items = data.get("bestsellers", [])[:limit]
    rows = []

    for it in items:
        title = it.get("title") or it.get("name") or ""
        asin = it.get("asin") or ""
        rank = it.get("rank") or 0

        amazon_price = extract_price_from_bestseller_item(it)

        sell_price = amazon_price if selling_price_mode == "Use Amazon price (rough)" else float(target_sell_price)
        fee_cost = sell_price * (marketplace_fee_pct / 100.0) + fixed_fee

        cogs = None
        if cogs_mode == "Assume % of selling price":
            cogs = sell_price * (cogs_pct / 100.0)

        total_cost = fee_cost + shipping_cost + packaging_cost + (cogs or 0.0)
        profit = sell_price - total_cost
        margin = (profit / sell_price * 100.0) if sell_price > 0 else 0.0

        # Competition proxy (if present)
        reviews_count = it.get("ratings_total") or it.get("reviews_total") or it.get("reviews") or 0

        demand_score = demand_score_from_rank(int(rank) if rank else 0)
        comp_score = competition_score_from_reviews(int(reviews_count) if reviews_count else 0)

        flags = risk_flags(title)

        # Overall score (simple but useful)
        # Demand high, competition low, margin high, penalize risks
        overall = demand_score * 0.45 + (100 - comp_score) * 0.30 + max(min(margin, 50), 0) * 0.25
        if flags:
            overall -= 10

        rows.append({
            "overall_score": round(overall, 2),
            "title": title,
            "asin": asin,
            "rank": rank,
            "amazon_price": round(amazon_price, 2),
            "used_sell_price": round(sell_price, 2),
            "est_profit_$": round(profit, 2),
            "est_margin_%": round(margin, 2),
            "demand_score": demand_score,
            "competition_score": comp_score,
            "risk_flags": "; ".join(flags)
        })

    df = pd.DataFrame(rows).sort_values("overall_score", ascending=False)
    st.subheader("✅ Shortlist (sorted by overall_score)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV shortlist",
        df.to_csv(index=False).encode("utf-8"),
        file_name="product_shortlist.csv",
        mime="text/csv"
    )

    st.subheader("🔎 Sourcing lead generator (US wholesalers first)")
    st.write("For the top products, copy/paste these into Google:")
    for t in df["title"].head(5).tolist():
        st.markdown(f"**{t}**")
        st.code(
            f'wholesale "{t}" USA\n'
            f'"{t}" distributor USA\n'
            f'"{t}" bulk supplier USA\n'
            f'site:faire.com "{t}"\n'
            f'site:tundra.com "{t}"\n'
            f'site:thomasnet.com "{t}"\n',
            language="text"
        )

    if use_ai:
        if not OPENAI_API_KEY:
            st.warning("AI is ON but OPENAI_API_KEY is missing in Secrets.")
        else:
            st.subheader("🧠 AI Gap Analysis (top picks)")
            for _, r in df.head(ai_top_n).iterrows():
                st.markdown(f"### {r['title']}")
                # Pull critical reviews (best-effort)
                critical_text = ""
                try:
                    rev = rf_get({
                        "type": "reviews",
                        "amazon_domain": amazon_domain,
                        "asin": r["asin"],
                        "review_stars": "all_critical",
                        "sort_by": "most_recent"
                    })
                    texts = []
                    for rr in (rev.get("reviews", [])[:15]):
                        body = (rr.get("body") or "").strip()
                        if body:
                            texts.append(body)
                    critical_text = "\n\n".join(texts)[:8000]
                except Exception:
                    critical_text = "No critical reviews available from endpoint; continue without reviews."

                st.write(ai_gap_analysis(r["title"], critical_text))
