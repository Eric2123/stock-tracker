# app.py - FINAL QUALSCORE EDITION - LOGO + ALERTS + P&L + FREE CHATBOX + AI PREDICTION
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
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ==================== PASSWORD PROTECTION ====================
st.markdown("<h2 style='text-align:center;color:#00d4ff;'>Enter Password</h2>", unsafe_allow_html=True)
password = st.text_input("Password", type="password", placeholder="Enter secret password")
SECRET_PASSWORD = "stockking123" # CHANGE THIS!
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

# ==================== PAGE CONFIG + QUALSCORE LOGO ====================
st.set_page_config(page_title="QualSCORE", page_icon="Chart increasing", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<div style="text-align:center;padding:20px;background:linear-gradient(90deg,#1e88e5,#00d4ff);border-radius:15px;margin-bottom:20px;">
    <h1 style="color:white;margin:0;font-size:40px;animation:glow 2s infinite alternate;">QualSCORE</h1>
    <p style="color:white;margin:5px;font-size:18px;">FUNDAMENTAL, TECHNICAL, QUALITATIVE</p>
</div>
<style>
@keyframes glow {
    from {text-shadow:0 0 10px #00d4ff;}
    to {text-shadow:0 0 30px #00ff00;}
}
</style>
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

theme = st.sidebar.radio("Theme", ["Light", "Dark"],
