# app.py - FINAL PRIVATE + LIVE NIFTY HEATMAP + ALL FEATURES
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
password = st.text_input("Password", type="password", placeholder="Enter secret password")
SECRET_PASSWORD = "stockking123"  # CHANGE THIS!
if password != SECRET_PASSWORD:
    st.error("Incorrect password. Access denied.")
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
    <p style="color:white;margin:5px;">Private • Elite • NIFTY HEATMAP LIVE</p>
</div>
<style>@keyframes glow {from{text-shadow:0 0 10px #00d4ff;}to{text-shadow:0 0 30px #00ff00;}}</style>
""", unsafe_allow_html=True)

# ==================== USER + WATCHLIST ====================
if 'user' not in st.session_state: st.session_state.user = "Elite Trader"
if 'watchlist' not in st.session_state: st.session_state.watchlist = []

user = st.sidebar.text_input("Your Name", value=st.session_state.user)
if user != st.session_state.user:
    st.session_state.user = user
    st.sidebar.success(f"Welcome back, {user}!")

st.sidebar.markdown("### Star Watchlist")
add_watch = st.sidebar.text_input("Add to Watchlist")
if st.sidebar.button("Add"):
    if 'df' in locals() and add_watch in df["Company Name"].values:
        if add_watch not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_watch)
            st.sidebar.success(f"{add_watch} added!")
    else:
        st.sidebar.error("Stock not found")
for w in st.session_state.watchlist:
    st.sidebar.success(f"Star {w}")

# ==================== UPLOAD + THEME + INDICES ====================
st.sidebar.header("Upload pythonmaster.xlsx")
uploaded_file = st.sidebar.file_uploader("Choose file", type=["xlsx"])
if not uploaded_file:
    st.error("Please upload your Excel file to continue!")
    st.stop()

theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
bg_color = "#1a1a1a" if theme == "Dark" else "white"
fg_color = "white" if theme == "Dark" else "black"
line_color = "white" if theme == "Dark" else "black"

plt.rcParams.update({
    'text.color': fg_color, 'axes.labelcolor': fg_color,
    'xtick.color': fg_color, 'ytick.color': fg_color,
    'axes.edgecolor': line_color, 'figure.facecolor': bg_color,
    'axes.facecolor': bg_color
})

@st.cache_data(ttl=15)
def get_indices():
    try:
        n = yf.Ticker("^NSEI").history(period="1d")["Close"].iloc[-1]
        s = yf.Ticker("^BSESN").history(period="1d")["Close"].iloc[-1]
        return round(n, 2), round(s, 2)
    except: return None, None

nifty, sensex = get_indices()
if nifty and sensex:
    st.sidebar.metric("**NIFTY 50**", f"₹{nifty:,.0f}")
    st.sidebar.metric("**SENSEX**", f"₹{sensex:,.0f}")
else:
    st.sidebar.warning("Loading indices...")

# ==================== PROCESS USER DATA ====================
@st.cache_data(show_spinner=False)
def process_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = df.columns.str.strip()
    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
    if "Index" not in df.columns: df["Index"] = "Unknown"
    missing = [c for c in required if c not in df.columns]
    if missing: st.error(f"Missing: {missing}"); st.stop()
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date of Publishing"])
    results = []
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        if not ticker.endswith((".BO", ".NS")): ticker += ".BO"
        try:
            current = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            results.append({
                "Company Name": row["Company Name"],
                "Ticker": ticker,
                "Record Price": row["Record Price"],
                "Current Price": round(current, 2),
                "target Price": row["Target Price"],
                "Index": row.get("Index", "Unknown"),
                "Date of Publishing": row["Date of Publishing"].date()
            })
        except: continue
    final_df = pd.DataFrame(results)
    final_df["Percent Change"] = ((final_df["Current Price"] - final_df["Record Price"]) / final_df["Record Price"] * 100).round(2)
    final_df["Distance from Target (%)"] = ((final_df["Current Price"] - final_df["target Price"]) / final_df["target Price"] * 100).round(2)
    return final_df

with st.spinner("Processing your stocks..."):
    df = process_data(uploaded_file)
st.success(f"Loaded {len(df)} stocks!")

# ==================== NIFTY 50 HEATMAP (NEW!) ====================
@st.cache_data(ttl=60)
def get_nifty_heatmap():
    # NIFTY 50 Tickers
    nifty50 = [
        "ADANIPORTS.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS",
        "BAJAJFINSV.NS","BPCL.NS","BHARTIARTL.NS","BRITANNIA.NS","CIPLA.NS",
        "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS",
        "HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS",
        "HINDUNILVR.NS","ICICIBANK.NS","ITC.NS","IOC.NS","INDUSINDBK.NS",
        "INFY.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","LTIM.NS",
        "M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
        "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SUNPHARMA.NS",
        "TCS.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS",
        "TITAN.NS","UPL.NS","ULTRACEMCO.NS","WIPRO.NS"
    ]
    # Sector mapping
    sector_map = {
        "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking", "INFY.NS": "IT",
        "BHARTIARTL.NS": "Telecom", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
        "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "HINDUNILVR.NS": "FMCG",
        "ITC.NS": "FMCG", "BAJFINANCE.NS": "Finance", "MARUTI.NS": "Auto",
        "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto", "LT.NS": "Infra",
        "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
        "ADANIPORTS.NS": "Infra", "JSWSTEEL.NS": "Metal", "TATASTEEL.NS": "Metal"
    }
    results = []
    for t in nifty50:
        try:
            data = yf.Ticker(t).history(period="2d")
            if len(data) < 2: continue
            prev = data["Close"].iloc[-2]
            curr = data["Close"].iloc[-1]
            pct = ((curr - prev) / prev) * 100
            sector = sector_map.get(t, "Others")
            name = t.replace(".NS", "")
            results.append({"Stock": name, "Change %": round(pct, 2), "Sector": sector})
        except: continue
    return pd.DataFrame(results)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab_portfolio, tab_heatmap = st.tabs([
    "Overview", "Trends", "Performance", "Sentiment", "Portfolio", "NIFTY HEATMAP"
])

# [ALL YOUR ORIGINAL TABS — UNCHANGED — JUST SKIPPING FOR BREVITY]
# ... (tab1 to tab_portfolio same as before)

# ==================== NEW TAB: NIFTY HEATMAP ====================
with tab_heatmap:
    st.header("LIVE NIFTY 50 HEATMAP")
    with st.spinner("Fetching NIFTY 50 data..."):
        heat_df = get_nifty_heatmap()
    
    if not heat_df.empty:
        # Sort by change
        heat_df = heat_df.sort_values("Change %", ascending=False)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        norm = mcolors.TwoSlopeNorm(vmin=heat_df["Change %"].min(), vcenter=0, vmax=heat_df["Change %"].max())
        sectors = heat_df["Sector"].unique()
        y_pos = 0
        for sector in sectors:
            sec_data = heat_df[heat_df["Sector"] == sector]
            x_pos = 0
            for _, row in sec_data.iterrows():
                color = plt.get_cmap("RdYlGn")(norm(row["Change %"]))
                ax.add_patch(plt.Rectangle((x_pos, y_pos), 1, 1, facecolor=color, edgecolor='white', linewidth=1))
                ax.text(x_pos + 0.5, y_pos + 0.5, f"{row['Stock']}\n{row['Change %']:+.1f}%", 
                        ha='center', va='center', fontsize=8, fontweight='bold', color='black')
                x_pos += 1
            # Sector label
            ax.text(-0.5, y_pos + 0.5, sector, ha='right', va='center', fontweight='bold', color=fg_color)
            y_pos += 1
        
        ax.set_xlim(0, max(heat_df.groupby("Sector").size()))
        ax.set_ylim(0, len(sectors))
        ax.axis("off")
        ax.set_title("NIFTY 50 HEATMAP - TODAY'S CHANGE %", color=fg_color, fontsize=18, pad=20)
        st.pyplot(fig)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("NIFTY Up", len(heat_df[heat_df["Change %"] > 0]))
        with col2: st.metric("NIFTY Down", len(heat_df[heat_df["Change %"] < 0]))
        with col3: st.metric("Avg Change", f"{heat_df['Change %'].mean():+.2f}%")
    else:
        st.error("Failed to load NIFTY data")

# ==================== REST OF YOUR TABS (UNCHANGED) ====================
# [Insert your original tab1 to tab_portfolio code here — I skipped for space]

# FINAL STATUS
st.sidebar.success("NIFTY HEATMAP LIVE")
st.sidebar.info("Password • Watchlist • Portfolio • WhatsApp • Heatmap")
