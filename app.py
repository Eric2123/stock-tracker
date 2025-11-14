# app.py - FINAL NO-EXCEL VERSION - CLIENT-READY - PRIVATE + ALL FEATURES
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
import base64
from io import BytesIO

# ==================== PASSWORD PROTECTION ====================
st.markdown("<h2 style='text-align:center;color:#00d4ff;'>Enter Password</h2>", unsafe_allow_html=True)
password = st.text_input("Password", type="password", placeholder="Secret key")

# CHANGE THIS TO YOUR PASSWORD
SECRET_PASSWORD = "stockking123"  # CHANGE NOW!

if password != SECRET_PASSWORD:
    st.error("Access Denied.")
    st.stop()

# ==================== AUTO REFRESH ====================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
elapsed = time.time() - st.session_state.last_refresh
if elapsed >= 60:
    st.session_state.last_refresh = time.time()
    st.rerun()
else:
    st.sidebar.caption(f"Auto-refresh in {60 - int(elapsed)}s")

# ==================== PAGE CONFIG + HEADER ====================
st.set_page_config(page_title="Stock Tracker Pro", page_icon="Chart increasing", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<div style="text-align:center;padding:20px;background:linear-gradient(90deg,#1e88e5,#00d4ff);border-radius:15px;margin-bottom:20px;">
    <h1 style="color:white;margin:0;animation:glow 2s infinite alternate;">STOCK TRACKER PRO</h1>
    <p style="color:white;margin:5px;">Client-Ready • No Upload • Instant Access</p>
</div>
<style>@keyframes glow {from{text-shadow:0 0 10px #00d4ff;}to{text-shadow:0 0 30px #00ff00;}}</style>
""", unsafe_allow_html=True)

# ==================== USER + WATCHLIST ====================
if 'user' not in st.session_state: st.session_state.user = "Client"
if 'watchlist' not in st.session_state: st.session_state.watchlist = []

user = st.sidebar.text_input("Your Name", value=st.session_state.user)
if user != st.session_state.user:
    st.session_state.user = user
    st.sidebar.success(f"Welcome, {user}!")

st.sidebar.markdown("### Star Watchlist")
add_watch = st.sidebar.text_input("Add stock")
if st.sidebar.button("Add"):
    if 'df' in locals() and add_watch in df["Company Name"].values:
        if add_watch not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_watch)
            st.sidebar.success(f"{add_watch} added!")
    else:
        st.sidebar.error("Not found")

for w in st.session_state.watchlist:
    st.sidebar.success(f"Star {w}")

# ==================== HARDCODED DATA (NO UPLOAD!) ====================
@st.cache_data
def load_hardcoded_data():
    data = {
        "Company Name": ["Reliance", "TCS", "HDFC Bank", "Infosys", "Bharti Airtel", "ICICI Bank"],
        "Ticker": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS", "ICICIBANK.NS"],
        "Record Price": [2800, 3500, 1500, 1700, 900, 800],
        "Target Price": [3200, 4000, 1800, 2000, 1100, 950],
        "Date of Publishing": ["2025-01-15", "2025-02-10", "2025-03-01", "2025-01-20", "2025-02-28", "2025-03-15"],
        "Index": ["Large Cap", "Large Cap", "Large Cap", "Large Cap", "Large Cap", "Large Cap"]
    }
    df = pd.DataFrame(data)
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"])
    return df

df_raw = load_hardcoded_data()

# ==================== FETCH LIVE PRICES ====================
@st.cache_data(ttl=60)
def fetch_live_prices(df):
    results = []
    for _, row in df.iterrows():
        try:
            current = yf.Ticker(row["Ticker"]).history(period="1d")["Close"].iloc[-1]
            results.append({
                "Company Name": row["Company Name"],
                "Ticker": row["Ticker"],
                "Record Price": row["Record Price"],
                "Current Price": round(current, 2),
                "target Price": row["Target Price"],
                "Index": row["Index"],
                "Date of Publishing": row["Date of Publishing"].date()
            })
        except: pass
    final_df = pd.DataFrame(results)
    final_df["Percent Change"] = ((final_df["Current Price"] - final_df["Record Price"]) / final_df["Record Price"] * 100).round(2)
    final_df["Distance from Target (%)"] = ((final_df["Current Price"] - final_df["target Price"]) / final_df["target Price"] * 100).round(2)
    return final_df

with st.spinner("Loading live stock data..."):
    df = fetch_live_prices(df_raw)

st.success(f"Live data loaded for {len(df)} stocks!")

# ==================== THEME + LIVE INDICES ====================
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
bg_color = "#1a1a1a" if theme == "Dark" else "white"
fg_color = "white" if theme == "Dark" else "black"
line_color = "white" if theme == "Dark" else "black"

plt.rcParams.update({'text.color': fg_color, 'axes.labelcolor': fg_color, 'xtick.color': fg_color, 'ytick.color': fg_color,
                     'axes.edgecolor': line_color, 'figure.facecolor': bg_color, 'axes.facecolor': bg_color})

@st.cache_data(ttl=15)
def get_indices():
    try:
        n = yf.Ticker("^NSEI").history(period="1d")["Close"].iloc[-1]
        s = yf.Ticker("^BSESN").history(period="1d")["Close"].iloc[-1]
        return round(n, 2), round(s, 2)
    except: return 25000, 82000

nifty, sensex = get_indices()
st.sidebar.metric("NIFTY 50", f"₹{nifty:,.0f}")
st.sidebar.metric("SENSEX", f"₹{sensex:,.0f}")

# ==================== FILTERS ====================
selected_companies = st.sidebar.multiselect("Select for Trends", df["Company Name"], default=st.session_state.watchlist[:3])
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months"])
cutoff = datetime.today() - timedelta(days=90) if period == "Last 3 Months" else datetime(1900,1,1)
filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Report", csv, "Stock_Report.csv", "text/csv")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab_port = st.tabs(["Overview", "Trends", "Performance", "Sentiment", "Portfolio"])

with tab1:
    st.header("Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Stocks", len(df))
    with col2: st.metric("Avg Gain", f"{df['Percent Change'].mean():+.1f}%")
    with col3:
        top = df.loc[df["Percent Change"].idxmax()]
        st.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.1f}%")
    fig_pie = px.pie(df["Index"].value_counts(), names=df["Index"].value_counts().index, hole=0.4)
    fig_pie.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=fg_color)
    st.plotly_chart(fig_pie, use_container_width=True)

    disp = filtered[["Company Name", "Current Price", "target Price", "Percent Change"]]
    st.dataframe(disp.style.format({"Current Price": "₹{:.0f}", "target Price": "₹{:.0f}", "Percent Change": "{:+.1f}%"}), use_container_width=True)

with tab2:
    st.header("Trends")
    for company in selected_companies:
        row = df[df["Company Name"] == company].iloc[0]
        st.subheader(company)
        if row["Current Price"] >= row["target Price"]:
            st.success("TARGET HIT!")
        hist = yf.download(row["Ticker"], period="6mo")
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(hist.index, hist["Close"], color="#00d4ff", linewidth=3)
            ax.axhline(row["target Price"], color="orange", linestyle="--")
            ax.set_title(f"{company} - 6M Trend", color=fg_color)
            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor=bg_color)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            wa_url = f"https://wa.me/?text={company}%20at%20₹{row['Current Price']:,}"
            st.markdown(f'<a href="{wa_url}" target="_blank"><button style="background:#25D366;color:white;padding:10px;border-radius:10px;">Share on WhatsApp</button></a>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(12,2))
            for p, l, c in zip([row["Record Price"], row["Current Price"], row["target Price"]], ["Record", "Current", "Target"], ["red", "#1e88e5", "green"]):
                ax2.scatter(p, 0, color=c, s=200)
                ax2.text(p, 0.15, f"{l}\n₹{p:,}", ha="center", color=fg_color, fontweight="bold")
            ax2.axis("off")
            st.pyplot(fig2)
        st.markdown("---")

with tab3:
    st.header("Performance")
    st.bar_chart(filtered.set_index("Company Name")["Percent Change"])
    col1, col2 = st.columns(2)
    with col1: st.dataframe(filtered.nlargest(3, "Percent Change")[["Company Name", "Percent Change"]])
    with col2: st.dataframe(filtered.nsmallest(3, "Percent Change")[["Company Name", "Percent Change"]])

with tab4:
    st.header("Sentiment")
    try:
        news = GNews(language='en', country='IN', max_results=5)
        items = news.get_news("Nifty")
        for item in items:
            with st.expander(item['title']):
                if 'image' in item: st.image(item['image'])
                st.caption(item['published date'])
    except: st.warning("News unavailable")

with tab_port:
    st.header("Portfolio")
    stock = st.selectbox("Stock", df["Company Name"])
    row = df[df["Company Name"] == stock].iloc[0]
    shares = st.number_input("Shares", 1, 1000, 100)
    value = shares * row["Current Price"]
    profit = (row["Current Price"] - row["Record Price"]) * shares
    st.metric("Value", f"₹{value:,.0f}")
    st.metric("Profit", f"₹{profit:,.0f}", delta=f"{profit/(row['Record Price']*shares)*100:+.1f}%")

st.sidebar.success("CLIENT-READY • NO UPLOAD")
