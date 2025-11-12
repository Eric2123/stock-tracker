# app.py - ULTIMATE FINAL - LIVE TICKER + AUTO REFRESH + TARGET ALERTS
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
from gnews import GNews
import time

# AUTO REFRESH EVERY 60 SECONDS
st.autorefresh(interval=60_000, key="data_refresh")

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Stock Tracker Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
.main { padding: 1rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0; text-align: center; }
.stButton>button { background-color: #1e88e5; color: white; border-radius: 8px; }
.blink { animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR UPLOAD -------------------
st.sidebar.header("Upload pythonmaster.xlsx")
uploaded_file = st.sidebar.file_uploader("Choose file", type=["xlsx"])

if not uploaded_file:
    st.error("Please upload your Excel file!")
    st.stop()

# ------------------- THEME TOGGLE -------------------
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
bg_color = "#1a1a1a" if theme == "Dark" else "white"
fg_color = "white" if theme == "Dark" else "black"
line_color = "white" if theme == "Dark" else "black"

# Apply theme
plt.rcParams['text.color'] = fg_color
plt.rcParams['axes.labelcolor'] = fg_color
plt.rcParams['xtick.color'] = fg_color
plt.rcParams['ytick.color'] = fg_color
plt.rcParams['axes.edgecolor'] = line_color

# ------------------- LIVE TICKER IN SIDEBAR -------------------
st.sidebar.markdown("### LIVE MARKET INDICES")
nifty_ph = st.sidebar.empty()
sensex_ph = st.sidebar.empty()

@st.cache_data(ttl=10)  # Update every 10 seconds
def get_live_indices():
    try:
        nifty = yf.Ticker("^NSEI").history(period="1d")["Close"].iloc[-1]
        sensex = yf.Ticker("^BSESN").history(period="1d")["Close"].iloc[-1]
        return round(nifty, 2), round(sensex, 2)
    except:
        return None, None

nifty_price, sensex_price = get_live_indices()
if nifty_price and sensex_price:
    nifty_ph.metric("NIFTY 50", f"₹{nifty_price:,.0f}", delta="LIVE")
    sensex_ph.metric("SENSEX", f"₹{sensex_price:,.0f}", delta="LIVE")
else:
    nifty_ph.warning("Fetching live data...")
    sensex_ph.warning("Fetching live data...")

# ------------------- PROCESS DATA -------------------
@st.cache_data(show_spinner=False)
def process_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = df.columns.str.strip()

    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Auto Publishing"]
    if "Index" not in df.columns:
        df["Index"] = "Unknown"
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date of Publishing"])

    results = []
    progress = st.progress(0)
    status = st.empty()
    today = datetime.today()
    three_m = today - timedelta(days=90)
    six_m = today - timedelta(days=180)

    for i, row in df.iterrows():
        company = row["Company Name"]
        ticker = str(row["Ticker"]).strip()
        if not ticker.endswith((".BO", ".NS")):
            ticker += ".BO"

        status.text(f"Fetching {company}...")
        try:
            data = yf.Ticker(ticker).history(period="1y")
            if data.empty: continue
            data.index = data.index.tz_localize(None)
            current = data["Close"].iloc[-1]
            price_3m = data[data.index >= three_m]["Close"].iloc[0] if not data[data.index >= three_m].empty else None
            price_6m = data[data.index >= six_m]["Close"].iloc[0] if not data[data.index >= six_m].empty else None

            pct = ((current - row["Record Price"]) / row["Record Price"]) * 100
            results.append({
                "Date of Publishing": row["Date of Publishing"].date(),
                "Company Name": company,
                "Ticker": ticker,
                "Index": row.get("Index", "Unknown"),
                "Record Price": row["Record Price"],
                "Current Price": round(current, 2),
                "target Price": row["Target Price"],
                "Price 3M": round(price_3m, 2) if price_3m else None,
                "Price 6M": round(price_6m, 2) if price_6m else None,
                "Absolute Current Price (%)": round(pct, 2),
            })
        except:
            continue
        progress.progress((i + 1) / len(df))

    status.empty()
    progress.empty()
    final = pd.DataFrame(results)
    final["Diff"] = final["Current Price"] - final["Record Price"]
    final["Percent Change"] = (final["Diff"] / final["Record Price"] * 100).round(2)
    final["Distance from Target (%)"] = ((final["Current Price"] - final["target Price"]) / final["target Price"] * 100).round(2)
    return final

with st.spinner("Processing stocks..."):
    df = process_data(uploaded_file)

st.success(f"Processed {len(df)} stocks!")

# ------------------- SIDEBAR: STOCK SELECTION + FILTERS -------------------
st.sidebar.markdown("### Select Stocks for Trends")
selected_companies = st.sidebar.multiselect(
    "Choose companies",
    options=df["Company Name"].unique(),
    default=df["Company Name"].head(3).tolist()
)

if not selected_companies:
    st.sidebar.warning("Please select at least one company")
    selected_companies = df["Company Name"].head(1).tolist()

# Time Period Filter
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
cutoff = datetime(1900, 1, 1)
today = datetime.today()
if period == "Last 3 Months": cutoff = today - timedelta(days=90)
elif period == "Last 6 Months": cutoff = today - timedelta(days=180)
elif period == "Last 1 Year": cutoff = today - timedelta(days=365)

filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

# CSV Download
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Full Results (CSV)", csv, "BSE_Stock_Report.csv", "text/csv")

# ------------------- TABS -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Performance", "Sentiment"])

with tab1:
    st.header("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Stocks", len(df))
    with col2: st.metric("Avg Return", f"{df['Percent Change'].mean():+.2f}%")
    with col3:
        top = df.loc[df["Percent Change"].idxmax()]
        st.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.2f}%")

    if "Index" in df.columns and df["Index"].nunique() > 1:
        idx_count = df["Index"].value_counts().reset_index()
        fig_pie = px.pie(idx_count, names="Index", values="count", title="Stocks by Index", hole=0.4)
        fig_pie.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=fg_color)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Performance Table")
    display = filtered[["Company Name", "Current Price", "target Price", "Percent Change", "Distance from Target (%)"]]
    styled = display.style.format({
        "Current Price": "₹{:.2f}",
        "target Price": "₹{:.2f}",
        "Percent Change": "{:+.2f}%",
        "Distance from Target (%)": "{:+.2f}%"
    }).bar(subset=["Percent Change"], color=['#90EE90', '#FFB6C1'])
    st.dataframe(styled, use_container_width=True)

with tab2:
    st.header("Stock Trends & Price Tracker")
    if not selected_companies:
        st.info("Select companies in the sidebar")
    else:
        for company in selected_companies:
            row = df[df["Company Name"] == company].iloc[0]
            with st.container():
                st.markdown(f"### {company}")

                # TARGET HIT ALERT
                if row["Current Price"] >= row["target Price"] * 0.95:
                    st.error(f"TARGET ALMOST HIT! {company} is at ₹{row['Current Price']:,} | Target: ₹{row['target Price']:,}")
                elif row["Current Price"] >= row["target Price"]:
                    st.success(f"TARGET ACHIEVED! {company} HIT ₹{row['Current Price']:,}")

                # Trend Chart
                hist = yf.download(row["Ticker"], period="6mo")
                if not hist.empty:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    fig.patch.set_facecolor(bg_color)
                    ax.set_facecolor(bg_color)
                    ax.plot(hist.index, hist["Close"], color="#00d4ff", linewidth=2.5, label="Close Price")
                    ax.axhline(row["target Price"], color="orange", linestyle="--", linewidth=2, label=f"Target ₹{row['target Price']}")
                    ax.grid(True, alpha=0.3, color=line_color)
                    for spine in ax.spines.values():
                        spine.set_color(line_color)
                    ax.set_title(f"{company} - 6 Month Trend", color=fg_color, fontsize=16)
                    ax.legend(facecolor=bg_color, edgecolor=line_color, labelcolor=fg_color)
                    st.pyplot(fig)

                    # Price Tracker
                    fig2, ax2 = plt.subplots(figsize=(12, 2))
                    fig2.patch.set_facecolor(bg_color)
                    ax2.set_facecolor(bg_color)
                    prices = [row["Record Price"], row["Current Price"], row["target Price"]]
                    labels = ["Record", "Current", "Target"]
                    colors = ["red", "#1e88e5", "green"]
                    for p, l, c in zip(prices, labels, colors):
                        ax2.scatter(p, 0, color=c, s=200, zorder=5, edgecolors=line_color, linewidth=2)
                        ax2.text(p, 0.15, f"{l}\n₹{p:,}", ha="center", va="bottom", fontweight="bold", fontsize=10, color=fg_color)
                    ax2.axhline(y=0, color=line_color, linewidth=1.5, alpha=0.8)
                    ax2.set_xlim(min(prices)*0.9, max(prices)*1.1)
                    ax2.set_ylim(-0.5, 0.5)
                    ax2.axis("off")
                    ax2.set_title(f"{company} - Price Tracker", color=fg_color, pad=20)
                    st.pyplot(fig2)
                else:
                    st.warning(f"No price data for {company}")
                st.markdown("---")

# [REST OF TAB3 AND TAB4 SAME AS BEFORE - NO CHANGE]
with tab3:
    st.header("Performance Analysis")
    st.bar_chart(filtered.set_index("Company Name")["Percent Change"].sort_values(ascending=False))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Gainers")
        st.dataframe(filtered.nlargest(5, "Percent Change")[["Company Name", "Percent Change"]], use_container_width=True)
    with col2:
        st.subheader("Top 5 Losers")
        st.dataframe(filtered.nsmallest(5, "Percent Change")[["Company Name", "Percent Change"]], use_container_width=True)

    st.subheader("Performance Heatmap")
    values = filtered["Absolute Current Price (%)"].fillna(0)
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    cols, rows = 6, (len(values) + 5) // 6
    fig, ax = plt.subplots(figsize=(16, rows * 1.8))
    fig.patch.set_facecolor(bg_color)
    for i, (comp, val) in enumerate(zip(filtered["Company Name"], values)):
        r, c = divmod(i, cols)
        color = plt.get_cmap("RdYlGn")(norm(val))
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color, edgecolor="white"))
        ax.text(c + 0.5, rows - r - 0.5, f"{comp}\n{val:+.1f}%", ha="center", va="center", fontsize=9, color="black")
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    ax.set_title("Stock Performance Heatmap", color=fg_color, fontsize=16, pad=20)
    st.pyplot(fig)

with tab4:
    st.header("Market Sentiment")
    try:
        news = GNews(language='en', country='IN', max_results=8)
        items = news.get_news("BSE NSE India stock market")
        sentiments = []
        for item in items:
            pol = TextBlob(item['title']).sentiment.polarity
            label = "Positive" if pol > 0.1 else "Negative" if pol < -0.1 else "Neutral"
            sentiments.append(pol)
            st.write(f"• {item['title']}")
            st.write(f"   → **{label}** ({pol:+.2f})")
        avg = np.mean(sentiments)
        if avg > 0.1: st.success(f"Overall: Positive ({avg:+.2f})")
        elif avg < -0.1: st.error(f"Overall: Negative ({avg:+.2f})")
        else: st.info(f"Overall: Neutral ({avg:+.2f})")
    except:
        st.warning("News not available")

st.sidebar.success("LIVE TICKER + AUTO REFRESH + TARGET ALERTS = GOD MODE")
