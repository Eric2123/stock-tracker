# app.py - FINAL FIX - UPLOAD FIRST, THEN PASSWORD
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

# ==================== STEP 1: UPLOAD FILE FIRST ====================
st.sidebar.header("Upload pythonmaster.xlsx")
uploaded_file = st.sidebar.file_uploader("Choose file", type=["xlsx"])

if not uploaded_file:
    st.info("Please upload your Excel file to unlock the dashboard.")
    st.stop()

# ==================== STEP 2: NOW ASK FOR PASSWORD ====================
if 'authenticated' not in st.session_state:
    st.markdown("<h2 style='text-align:center;color:#00d4ff;'>Enter Password to Unlock</h2>", unsafe_allow_html=True)
    password = st.text_input("Password", type="password", placeholder="Enter secret password")
    SECRET_PASSWORD = "stockking123"  # CHANGE THIS!

    if st.button("Unlock Dashboard"):
        if password == SECRET_PASSWORD:
            st.session_state.authenticated = True
            st.success("Access Granted!")
            st.rerun()
        else:
            st.error("Wrong password. Try again.")
            st.stop()
    else:
        st.stop()

# ==================== USER + WATCHLIST (AFTER AUTH) ====================
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

# ==================== THEME + INDICES ====================
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

# ==================== PROCESS DATA ====================
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
    final_df["Absolute Current Price (%)"] = final_df["Percent Change"]
    return final_df

with st.spinner("Processing your stocks..."):
    df = process_data(uploaded_file)
st.success(f"Loaded {len(df)} stocks for {st.session_state.user}!")

# ==================== REST OF YOUR CODE (TABS, HEATMAP, ETC.) ====================
# [INSERT ALL YOUR TABS FROM BEFORE — INCLUDING NIFTY HEATMAP]

# Example: Just one tab to confirm it's working
tab1, tab_heatmap = st.tabs(["Overview", "NIFTY HEATMAP"])

with tab1:
    st.header("Dashboard Overview")
    st.write(df[["Company Name", "Current Price", "Percent Change"]].head())

with tab_heatmap:
    st.header("LIVE NIFTY 50 HEATMAP")
    st.write("NIFTY Heatmap coming soon...")

st.sidebar.success("APP UNLOCKED")
