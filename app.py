# app.py - ADVANCED UI/UX QUALSCORE EDITION - ENHANCED AESTHETICS + NEW FEATURES
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from textblob import TextBlob
from gnews import GNews
import time
import base64
from io import BytesIO
from sklearn.linear_model import LinearRegression
import streamlit.components.v1 as components  # For custom HTML/JS if needed
import smtplib
from email.mime.text import MIMEText
import os

# ==================== EMAIL ALERT CONFIG ====================
EMAIL_SENDER = "loboe173@gmail.com"
EMAIL_PASSWORD = "PASTE_YOUR_16_DIGIT_APP_PASSWORD_HERE"
ALERT_EMAILS = ["eric.l@qualscore.in"]

ALERT_LOG_FILE = "email_alert_log.csv"


# ==================== EMAIL ALERT FUNCTIONS ====================
def send_email(subject, body):
    from email.mime.text import MIMEText
    import smtplib

    msg = MIMEText(body)
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(ALERT_EMAILS)
    msg["Subject"] = subject

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, ALERT_EMAILS, msg.as_string())


def load_alert_log():
    import os
    import pandas as pd

    if os.path.exists(ALERT_LOG_FILE):
        return pd.read_csv(ALERT_LOG_FILE)
    return pd.DataFrame(columns=["Ticker", "Alert"])


def alert_already_sent(ticker, alert):
    log = load_alert_log()
    return ((log["Ticker"] == ticker) & (log["Alert"] == alert)).any()


def save_alert(ticker, alert):
    log = load_alert_log()
    log.loc[len(log)] = [ticker, alert]
    log.to_csv(ALERT_LOG_FILE, index=False)


def run_email_alerts(df):
    for _, row in df.iterrows():
        price = row["Current Price"]
        target = row["target Price"]
        ticker = row["Ticker"]
        company = row["Company Name"]

        conditions = {
            "15% BELOW TARGET": price <= target * 0.85,
            "5% BELOW TARGET": price <= target * 0.95,
            "TARGET HIT": price >= target,
            "5% ABOVE TARGET": price >= target * 1.05,
        }

        for alert, hit in conditions.items():
            if hit and not alert_already_sent(ticker, alert):
                subject = f"🚨 QUALSCORE ALERT — {alert}"
                body = f"""
Stock: {company}
Ticker: {ticker}

Current Price: ₹{price:.2f}
Target Price: ₹{target:.2f}

Alert Triggered: {alert}

— QualSCORE Automated Alert System
"""
                send_email(subject, body)
                save_alert(ticker, alert)
# ==================== HARD-CODED STOCK MASTER ====================
STOCK_MASTER = [
    {"Date of Publishing":"10-05-2024","Company Name":"Thomas Cook (India) Ltd","Ticker":"THOMASCOOK.BO","Index":"Microcap","Record Price":201,"Target Price":316},
    {"Date of Publishing":"20-05-2024","Company Name":"SBI Cards & Payment Services Ltd","Ticker":"SBICARD.BO","Index":"Large Cap","Record Price":715,"Target Price":1094},
    {"Date of Publishing":"31-05-2024","Company Name":"Va Tech Wabag Ltd","Ticker":"WABAG.BO","Index":"SmallCap","Record Price":980,"Target Price":1413},
    {"Date of Publishing":"10-06-2024","Company Name":"AGI Greenpac Ltd","Ticker":"AGI.BO","Index":"Microcap","Record Price":716,"Target Price":990},
    {"Date of Publishing":"18-07-2024","Company Name":"West Coast Paper Mills Ltd","Ticker":"WSTCSTPAPR.BO","Index":"Microcap","Record Price":678,"Target Price":953},
    {"Date of Publishing":"30-06-2024","Company Name":"Anthony Waste Handling Cell Ltd","Ticker":"AWHCL.BO","Index":"Microcap","Record Price":511,"Target Price":843},
    {"Date of Publishing":"22-07-2024","Company Name":"Narayana Hrudayalaya Ltd","Ticker":"NH.BO","Index":"Midcap","Record Price":1242,"Target Price":1650},
    {"Date of Publishing":"14-08-2024","Company Name":"A.K. Capital Services Ltd","Ticker":"AKCAPIT.BO","Index":"Microcap","Record Price":1051,"Target Price":1725},
    {"Date of Publishing":"21-08-2024","Company Name":"Ashok Leyland Ltd","Ticker":"ASHOKLEY.BO","Index":"Large Cap","Record Price":126,"Target Price":177.5},
    {"Date of Publishing":"22-08-2024","Company Name":"Signpost India Ltd","Ticker":"SIGNPOST.BO","Index":"Microcap","Record Price":252,"Target Price":454},
    {"Date of Publishing":"10-09-2024","Company Name":"Action Construction Equipment Ltd","Ticker":"ACE.BO","Index":"SmallCap","Record Price":1260,"Target Price":1510},
    {"Date of Publishing":"20-10-2024","Company Name":"Rupa & Company Ltd","Ticker":"RUPA.BO","Index":"Microcap","Record Price":280,"Target Price":390},
    {"Date of Publishing":"20-10-2024","Company Name":"Dollar Industries Ltd","Ticker":"DOLLAR.BO","Index":"Microcap","Record Price":537,"Target Price":775},
    {"Date of Publishing":"30-09-2024","Company Name":"Ather Energy Ltd","Ticker":"ATHERENERG.BO","Index":"Midcap","Record Price":336,"Target Price":525},
    {"Date of Publishing":"28-10-2024","Company Name":"LG Balakrishnan & Bros Ltd","Ticker":"LGBBROSLTD.BO","Index":"Microcap","Record Price":1265,"Target Price":1925},
    {"Date of Publishing":"26-11-2024","Company Name":"Avenue Supermarts Ltd","Ticker":"DMART.BO","Index":"Mega Cap","Record Price":3614,"Target Price":2676},
    {"Date of Publishing":"26-11-2024","Company Name":"Ethos Ltd","Ticker":"ETHOSLTD.BO","Index":"SmallCap","Record Price":2954,"Target Price":3479},
    {"Date of Publishing":"26-11-2024","Company Name":"Redington Ltd","Ticker":"REDINGTON.BO","Index":"SmallCap","Record Price":196,"Target Price":313},
    {"Date of Publishing":"27-11-2024","Company Name":"IndiaMART Inter Ltd","Ticker":"INDIAMART.BO","Index":"SmallCap","Record Price":2371,"Target Price":3212},
    {"Date of Publishing":"13-12-2024","Company Name":"GE Shipping Company Ltd","Ticker":"GESHIP.BO","Index":"SmallCap","Record Price":1078,"Target Price":2147},
    {"Date of Publishing":"16-12-2024","Company Name":"EMS Ltd","Ticker":"EMSLIMITED.BO","Index":"Microcap","Record Price":865,"Target Price":1185},
    {"Date of Publishing":"06-01-2025","Company Name":"Fedbank Financial Services Ltd","Ticker":"FEDFINA.BO","Index":"Microcap","Record Price":103,"Target Price":158},
    {"Date of Publishing":"20-01-2025","Company Name":"South Indian Bank Ltd","Ticker":"SOUTHBANK.BO","Index":"SmallCap","Record Price":26.5,"Target Price":55},
    {"Date of Publishing":"03-02-2025","Company Name":"IndusInd Bank Ltd","Ticker":"INDUSINDBK.BO","Index":"Midcap","Record Price":1015,"Target Price":1825},
    {"Date of Publishing":"01-03-2025","Company Name":"Amara Raja Energy & Mobility Ltd","Ticker":"ARE&M.BO","Index":"Midcap","Record Price":993,"Target Price":1575},
    {"Date of Publishing":"24-03-2025","Company Name":"Hyundai Motor India Ltd","Ticker":"HYUNDAI.BO","Index":"Large Cap","Record Price":1700,"Target Price":2163},
    {"Date of Publishing":"08-04-2025","Company Name":"PNB Gilts Ltd","Ticker":"PNBGILTS.BO","Index":"Microcap","Record Price":91.1,"Target Price":145},
    {"Date of Publishing":"05-04-2025","Company Name":"Concord Enviro Systems Ltd","Ticker":"CEWATER.BO","Index":"Microcap","Record Price":540,"Target Price":982},
    {"Date of Publishing":"07-04-2025","Company Name":"BEML Ltd","Ticker":"BEML.BO","Index":"SmallCap","Record Price":2765,"Target Price":4700},
    {"Date of Publishing":"15-04-2025","Company Name":"Mahanagar Gas Ltd","Ticker":"MGL.BO","Index":"SmallCap","Record Price":1278,"Target Price":2052},
    {"Date of Publishing":"13-04-2025","Company Name":"Jio Financial Services Ltd","Ticker":"JIOFIN.BO","Index":"Large Cap","Record Price":255,"Target Price":415},
    {"Date of Publishing":"17-04-2025","Company Name":"WPIL Ltd","Ticker":"WPIL.BO","Index":"Microcap","Record Price":455,"Target Price":775},
    {"Date of Publishing":"02-05-2025","Company Name":"Kirloskar Brothers Ltd","Ticker":"KIRLOSBROS.BO","Index":"SmallCap","Record Price":1702,"Target Price":2300},
    {"Date of Publishing":"27-05-2025","Company Name":"Vardhman Textiles Ltd","Ticker":"VTL.BO","Index":"SmallCap","Record Price":495,"Target Price":750},
    {"Date of Publishing":"06-06-2025","Company Name":"ICICI Lombard General Insurance Co Ltd","Ticker":"ICICIGI.BO","Index":"Large Cap","Record Price":2009,"Target Price":2885},
    {"Date of Publishing":"18-06-2025","Company Name":"TTK Prestige Ltd","Ticker":"TTKPRESTIG.BO","Index":"SmallCap","Record Price":620,"Target Price":840},
    {"Date of Publishing":"22-06-2025","Company Name":"Galaxy Surfactants Ltd","Ticker":"GALAXYSURF.BO","Index":"SmallCap","Record Price":2530,"Target Price":3300},
    {"Date of Publishing":"07-06-2025","Company Name":"Updater Services Ltd","Ticker":"UDS.BO","Index":"Microcap","Record Price":305,"Target Price":530},
    {"Date of Publishing":"03-07-2025","Company Name":"RateGain Travel Technologies Ltd","Ticker":"RATEGAIN.BO","Index":"SmallCap","Record Price":464,"Target Price":760},
    {"Date of Publishing":"06-07-2025","Company Name":"IGI Ltd","Ticker":"IGIL.BO","Index":"SmallCap","Record Price":381,"Target Price":590},
    {"Date of Publishing":"20-07-2025","Company Name":"Hindalco Industries Ltd","Ticker":"HINDALCO.BO","Index":"Large Cap","Record Price":676,"Target Price":1118},
    {"Date of Publishing":"11-08-2025","Company Name":"Jindal Saw Ltd","Ticker":"JINDALSAW.BO","Index":"SmallCap","Record Price":204,"Target Price":387},
    {"Date of Publishing":"31-08-2025","Company Name":"Hero MotoCorp Ltd","Ticker":"HEROMOTOCO.BO","Index":"Large Cap","Record Price":5089,"Target Price":7200},
    {"Date of Publishing":"31-08-2025","Company Name":"Geojit Financial Services Ltd","Ticker":"GEOJITFSL.BO","Index":"Microcap","Record Price":71.3,"Target Price":218},
    {"Date of Publishing":"31-08-2025","Company Name":"Indian Energy Exchange Ltd","Ticker":"IEX.BO","Index":"SmallCap","Record Price":140,"Target Price":215},
    {"Date of Publishing":"01-09-2025","Company Name":"Tata Consultancy Services Ltd","Ticker":"TCS.BO","Index":"Mega Cap","Record Price":3110,"Target Price":3900},
    {"Date of Publishing":"04-09-2025","Company Name":"Eicher Motors Ltd","Ticker":"EICHERMOT.BO","Index":"Large Cap","Record Price":6435,"Target Price":8085},
    {"Date of Publishing":"01-10-2025","Company Name":"Vikram Solar","Ticker":"VIKRAMSOLR.BO","Index":"SmallCap","Record Price":316,"Target Price":563},
    {"Date of Publishing":"15-10-2025","Company Name":"MPS Ltd","Ticker":"MPSLTD.BO","Index":"Microcap","Record Price":2248,"Target Price":2996},
    {"Date of Publishing":"28-10-2025","Company Name":"Jindal Stainless","Ticker":"JSL.BO","Index":"Midcap","Record Price":803,"Target Price":975},
    {"Date of Publishing":"06-11-2025","Company Name":"D-Link India Ltd","Ticker":"DLINKINDIA.BO","Index":"Microcap","Record Price":445,"Target Price":735},
    {"Date of Publishing":"06-11-2025","Company Name":"Mallcom India Ltd","Ticker":"MALLCOM.BO","Index":"Microcap","Record Price":1436,"Target Price":2500},
    {"Date of Publishing":"10-01-2026","Company Name":"Brookfield India Real Estate Trust","Ticker":"BIRET.BO","Index":"Midcap","Record Price":334,"Target Price":425},
    {"Date of Publishing":"10-01-2026","Company Name":"Mindspace Business Parks REIT","Ticker":"MINDSPACE.BO","Index":"Midcap","Record Price":490,"Target Price":500},
    {"Date of Publishing":"10-01-2026","Company Name":"Embassy Office Parks REIT","Ticker":"EMBASSY.BO","Index":"Midcap","Record Price":439,"Target Price":460},
    {"Date of Publishing":"10-01-2026","Company Name":"Nexus Select Trust","Ticker":"NXST.BO","Index":"SmallCap","Record Price":160,"Target Price":175},
    {"Date of Publishing":"15-01-2026","Company Name":"Jupiter Life Line Hospitals Ltd","Ticker":"JLHL.BO","Index":"SmallCap","Record Price":1366,"Target Price":1650},
    {"Date of Publishing":"18-01-2026","Company Name":"AGI Greenpac Ltd (Rework)","Ticker":"AGI.BO","Index":"Microcap","Record Price":670,"Target Price":812.5}
]
# ==================== EMAIL ALERT CONFIG ====================
EMAIL_SENDER = "loboe173@gmail.com"
EMAIL_PASSWORD = "xctm ziaq azmo dviq"
ALERT_EMAILS = ["eric.l@qualscore.in"]
ALERT_LOG_FILE = "email_alert_log.csv"
 
# ==================== RUN EMAIL ALERTS ====================
run_email_alerts(df)

            

@st.cache_data
def load_master_data():
    df = pd.DataFrame(STOCK_MASTER)
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True)
    return df

MASTER_DF = load_master_data()

# ==================== SAFE DF INITIALIZATION ====================
df = MASTER_DF.copy()

# ==================== PREMIUM FINTECH UI THEME ====================
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #F6F8FB;
    color: #1C1C1C;
}

/* Main container */
.main .block-container {
    padding: 2.5rem 3rem;
    max-width: 1400px;
}

/* Headings */
h1, h2, h3 {
    color: #1F3A5F;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* Cards */
.metric-card {
    background: #FFFFFF;
    padding: 1.25rem;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    border: 1px solid #E5E7EB;
}

/* Metrics */
.stMetric {
    background: #FFFFFF;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
}

/* Buttons */
.stButton > button {
    background-color: #1F3A5F;
    color: #FFFFFF;
    border-radius: 10px;
    padding: 0.45rem 1.2rem;
    font-weight: 500;
    border: none;
}
.stButton > button:hover {
    background-color: #162C48;
}

/* Tables */
.dataframe {
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
}

/* Expanders */
.stExpander {
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
}

/* Alerts */
.alert-card {
    background-color: #F1F5F9;
    border-left: 5px solid #4C6EF5;
    padding: 1rem;
    border-radius: 10px;
}

/* Chat bubbles */
.stChatMessage {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 500;
    color: #6B7280;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1F3A5F;
    border-bottom: 3px solid #4C6EF5;
}
</style>
""", unsafe_allow_html=True)


# ==================== PASSWORD PROTECTION ====================
st.markdown("<h2 style='text-align:center;color:#00d4ff; class='glow-text'>Enter Password</h2>", unsafe_allow_html=True)
password = st.text_input("Password", type="password", placeholder="Enter secret password")
SECRET_PASSWORD = "admin"  # CHANGE THIS!
if password != SECRET_PASSWORD:
    st.error("❌ Incorrect password. Access denied.")
    st.stop()

# ==================== AUTO REFRESH WITH PROGRESS BAR ====================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
elapsed = time.time() - st.session_state.last_refresh
refresh_progress = min(elapsed / 60, 1.0)
st.sidebar.progress(refresh_progress)
if elapsed >= 60:
    st.session_state.last_refresh = time.time()
    st.rerun()
else:
    st.sidebar.caption(f"🔄 Auto-refresh in {60 - int(elapsed)}s")

# ==================== PAGE CONFIG + ENHANCED QUALSCORE LOGO ====================
st.set_page_config(page_title="QualSCORE", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<div style="
background:#FFFFFF;
padding:30px;
border-radius:18px;
border:1px solid #E5E7EB;
margin-bottom:30px;
box-shadow:0 6px 30px rgba(0,0,0,0.04);">

<h1 style="margin:0;color:#1F3A5F;">QualSCORE</h1>
<p style="margin:6px 0 0;color:#6B7280;">
Equity research internal stock dashboard
</p>

</div>
""", unsafe_allow_html=True)

# ==================== USER + WATCHLIST + SEARCH FEATURE ====================
if 'user' not in st.session_state: 
    st.session_state.user = "Elite Trader"
if 'watchlist' not in st.session_state: 
    st.session_state.watchlist = []
user = st.sidebar.text_input("👤 Your Name", value=st.session_state.user)
if user != st.session_state.user:
    st.session_state.user = user
    st.sidebar.success(f"👋 Welcome back, {user}!")

# New Feature: Quick Search for Companies
st.sidebar.markdown("🔍 **Quick Search**")
search_query = st.sidebar.text_input("Search Company")
if search_query:
    matches = df[df["Company Name"].str.contains(search_query, case=False, na=False)]["Company Name"].tolist()
    if matches:
        selected_companies = st.sidebar.multiselect("Search Results", matches, default=matches[:3])
    else:
        st.sidebar.warning("No matches found.")

st.sidebar.markdown("⭐ **Star Watchlist**")
add_watch = st.sidebar.text_input("Add to Watchlist")
if st.sidebar.button("Add ⭐"):
    if 'df' in locals() and add_watch in df["Company Name"].values:
        if add_watch not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_watch)
            st.sidebar.success(f"⭐ {add_watch} added!")
    else:
        st.sidebar.error("Stock not found")
for w in st.session_state.watchlist:
    st.sidebar.success(f"⭐ {w}")

# ==================== UPLOAD + ENHANCED THEME + LAYOUT + INDICES ====================

# Enhanced Themes with Better Presets
theme_presets = st.sidebar.selectbox("🎨 Theme Preset", ["Dark Nebula"])
theme_presets == "Dark Nebula"
bg_color = "#F6F8FB"
plot_bg = "#FFFFFF"
fg_color = "#1C1C1C"
line_color = "#4C6EF5"
primary_color = "#1F3A5F"

# Layout Toggle
full_width = st.sidebar.checkbox("🌐 Full Width Layout", value=True)

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
    except: 
        return None, None

nifty, sensex = get_indices()
if nifty and sensex:
    st.sidebar.metric("📊 **NIFTY 50**", f"₹{nifty:,.0f}")
    st.sidebar.metric("📈 **SENSEX**", f"₹{sensex:,.0f}")
else:
    st.sidebar.warning("⏳ Loading indices...")

# New Feature: Manual Refresh Button
if st.sidebar.button("🔄 Refresh Data Now"):
    st.cache_data.clear()
    st.rerun()

# ==================== DATA PROCESSING WITH PROGRESS ====================
@st.cache_data(show_spinner=False)
def process_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
    if "Index" not in df.columns: 
        df["Index"] = "Unknown"
    missing = [c for c in required if c not in df.columns]
    if missing: 
        st.error(f"❌ Missing columns: {missing}"); 
        st.stop()
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date of Publishing"])
    results = []
    progress_bar = st.progress(0)
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        ticker = str(row["Ticker"]).strip()
        if not ticker.endswith((".BO", ".NS")): 
            ticker += ".BO"
        try:
            current = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            
            # Compute Volatility (std dev of last 30 days returns)
            hist_30d = yf.Ticker(ticker).history(period="1mo")
            if not hist_30d.empty:
                returns_30d = hist_30d["Close"].pct_change().dropna()
                volatility = returns_30d.std() * np.sqrt(252) * 100  # Annualized
            else:
                volatility = 0.0
            
            # Compute Beta vs Nifty
            hist_stock = yf.Ticker(ticker).history(period="1y")
            hist_nifty = yf.Ticker("^NSEI").history(period="1y")
            if not hist_stock.empty and not hist_nifty.empty:
                # Flatten columns if MultiIndex
                if isinstance(hist_stock.columns, pd.MultiIndex):
                    hist_stock.columns = hist_stock.columns.droplevel(1)
                if isinstance(hist_nifty.columns, pd.MultiIndex):
                    hist_nifty.columns = hist_nifty.columns.droplevel(1)
                returns_stock = hist_stock["Close"].pct_change().dropna()
                returns_nifty = hist_nifty["Close"].pct_change().dropna()
                # Align by index
                common_idx = returns_stock.index.intersection(returns_nifty.index)
                if len(common_idx) > 1:
                    X = returns_nifty.loc[common_idx].values.reshape(-1, 1)
                    y = returns_stock.loc[common_idx].values
                    model = LinearRegression().fit(X, y)
                    beta = model.coef_[0]
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            results.append({
                "Company Name": row["Company Name"],
                "Ticker": ticker,
                "Record Price": row["Record Price"],
                "Current Price": round(current, 2),
                "target Price": row["Target Price"],
                "Index": row.get("Index", "Unknown"),
                "Date of Publishing": row["Date of Publishing"].date(),
                "Volatility (%)": round(volatility, 2),
                "Beta": round(beta, 2)
            })
        except: 
            continue
        progress_bar.progress((i + 1) / total)
    final_df = pd.DataFrame(results)
    final_df["Percent Change"] = ((final_df["Current Price"] - final_df["Record Price"]) / final_df["Record Price"] * 100).round(2)
    final_df["Distance from Target ($)"] = ((final_df["Current Price"] - final_df["target Price"]) / final_df["target Price"] * 100).round(2)
    final_df["Absolute Current Price ($)"] = final_df["Percent Change"]
    return final_df

with st.spinner("🔄 Processing your stocks... Hold on for advanced insights!"):
    df = process_data(MASTER_DF)
st.success(f"✅ Processed {len(df)} stocks for {st.session_state.user}!")

# ==================== RUN EMAIL ALERTS ====================
run_email_alerts(df)

st.success(f"✅ Processed {len(df)} stocks for {st.session_state.user}!")

# New Feature: Quick Export to PDF (simulated via download)
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("📥 Download Report (CSV)", csv, "Stock_Report.csv", "text/csv")
# For PDF, use simple base64, but keep CSV for now

# ==================== FILTERS ====================
st.sidebar.markdown("🔧 **Filters**")
selected_companies = st.sidebar.multiselect(
    "Choose companies", df["Company Name"].unique(),
    default=(st.session_state.watchlist + list(df["Company Name"].head(3).tolist()))[:3]
)
if not selected_companies: 
    selected_companies = df["Company Name"].head(1).tolist()

period = st.sidebar.selectbox("📅 Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
cutoff = datetime(1900, 1, 1)
if period == "Last 3 Months": 
    cutoff = datetime.today() - timedelta(days=90)
elif period == "Last 6 Months": 
    cutoff = datetime.today() - timedelta(days=180)
elif period == "Last 1 Year": 
    cutoff = datetime.today() - timedelta(days=365)
filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab_strategy, tab_portfolio, tab_chat = st.tabs([
    "📊 Overview", "📈 Trends", "🏆 Performance", "📰 Sentiment",
    "🧭 Strategy 2026", "💼 Portfolio", "🤖 Chat"
])

# TAB 1: OVERVIEW - ENHANCED WITH CARDS
with tab1:
    st.header("🚀 Dashboard Overview")
    # New Feature: Insight Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">Total Stocks<br><h2 style="margin:0;">' + str(len(df)) + '</h2></div>', unsafe_allow_html=True)
    with col2:
        avg_return = f"{df['Percent Change'].mean():+.2f}%"
        color = "green" if df['Percent Change'].mean() > 0 else "red"
        st.markdown(f'<div class="metric-card">Avg Return<br><h2 style="margin:0;color:{color};">{avg_return}</h2></div>', unsafe_allow_html=True)
    with col3:
        top = df.loc[df["Percent Change"].idxmax()]
        st.markdown(f'<div class="metric-card">Top Gainer<br><h2 style="margin:0;">{top["Company Name"]}</h2><p style="margin:0;">{top["Percent Change"]:+.2f}%</p></div>', unsafe_allow_html=True)
    
    # Quick Stats Cards
    with st.expander("⚠️ Quick Risk Stats", expanded=False):
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            avg_vol = df["Volatility (%)"].mean()
            st.metric("Avg Volatility", f"{avg_vol:.2f}%")
        with col_r2:
            avg_beta = df["Beta"].mean()
            st.metric("Avg Beta", f"{avg_beta:.2f}")
        with col_r3:
            high_risk = len(df[df["Volatility (%)"] > 30])
            st.metric("High Risk Stocks", high_risk)
    
    if df["Index"].nunique() > 1:
        fig_pie = px.pie(df["Index"].value_counts().reset_index(), names="Index", values="count", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues)
        fig_pie.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=fg_color)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("📋 Performance Table")
    disp = filtered[["Company Name", "Current Price", "target Price", "Percent Change", "Distance from Target ($)", "Volatility (%)", "Beta"]]
    styled = disp.style.format({
        "Current Price": "₹{:.2f}", "target Price": "₹{:.2f}",
        "Percent Change": "{:+.2f}%", "Distance from Target ($)": "{:+.2f}%",
        "Volatility (%)": "{:.2f}%"
    }).bar(subset=["Percent Change"], color=['#4caf50', '#f44336'])
    st.dataframe(styled, use_container_width=True)

# TAB 2: TRENDS + TARGET ALERT - ENHANCED ALERTS
with tab2:
    st.header("📈 Stock Trends & Price Tracker")
    for company in selected_companies:
        row = df[df["Company Name"] == company].iloc[0]
        st.markdown(f"### {company}")

        if row["Current Price"] >= row["target Price"]:
            st.success(f"🎯 TARGET HIT! {company} reached ₹{row['Current Price']:,}")
            alert = f"QUALSCORE ALERT: {company} HIT TARGET! Current: ₹{row['Current Price']:,} | Target: ₹{row['target Price']:,}"
            wa_link = f"https://wa.me/?text={alert.replace(' ', '%20')}"
            st.markdown(f'''
            <a href="{wa_link}" target="_blank">
                <button style="background:#25D366;color:white;padding:10px 20px;border:none;border-radius:10px;font-weight:bold;box-shadow:0 2px 4px rgba(0,0,0,0.2);">
                    📱 Send WhatsApp Alert
                </button>
            </a>
            ''', unsafe_allow_html=True)
            st.markdown('<div class="alert-card">🚨 New Feature: Target Achieved! Consider rebalancing.</div>', unsafe_allow_html=True)
        elif row["Current Price"] >= row["target Price"] * 0.95:
            st.error(f"⚠️ NEAR TARGET! Only ₹{row['target Price'] - row['Current Price']:.0f} away!")

        publish_date = row["Date of Publishing"]
        start_str = publish_date.strftime('%Y-%m-%d')
        end_str = datetime.now().strftime('%Y-%m-%d')
        hist = yf.download(row["Ticker"], start=start_str, end=end_str)
        if not hist.empty:
            # Flatten MultiIndex columns if present
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.droplevel(1)
            # Interactive Plotly Chart - Reset index to make Date a column
            hist_plot = hist.reset_index()
            fig = px.line(hist_plot, x='Date', y='Close', title=f"{company} - Trend from {publish_date.strftime('%Y-%m-%d')}",
                          labels={'Close': 'Price (₹)', 'Date': 'Date'})
            fig.add_hline(y=row["Record Price"], line_dash="solid", line_color="orange",
                          annotation_text=f"Record ₹{row['Record Price']:.2f}", annotation_position="top left")
            fig.add_hline(y=row["target Price"], line_dash="dash", line_color="orange",
                          annotation_text=f"Target ₹{row['target Price']:.2f}", annotation_position="top right")
            publish_dt = pd.to_datetime(publish_date)
            fig.add_scatter(x=[publish_dt], y=[row["Record Price"]], mode="markers", marker=dict(color="red", size=10),
                            name="Buy Date")
            fig.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font_color=fg_color,
                hovermode="x unified",
                legend=dict(bgcolor=plot_bg, font_color=fg_color)
            )
            fig.update_xaxes(gridcolor=line_color)
            fig.update_yaxes(gridcolor=line_color)
            st.plotly_chart(fig, use_container_width=True)

            wa_msg = f"*{company}* is at ₹{row['Current Price']:,} | Target: ₹{row['target Price']:,}"
            wa_url = f"https://wa.me/?text={wa_msg.replace(' ', '%20')}"
            st.markdown(f'<a href="{wa_url}" target="_blank"><button style="background:#25D366;color:white;padding:10px 20px;border:none;border-radius:10px;">📱 Share on WhatsApp</button></a>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(12, 2))
            prices = [row["Record Price"], row["Current Price"], row["target Price"]]
            labels = ["Record", "Current", "Target"]
            colors = ["red", primary_color, "green"]
            for p, l, c in zip(prices, labels, colors):
                ax2.scatter(p, 0, color=c, s=200, edgecolors=line_color, linewidth=2)
                ax2.text(p, 0.15, f"{l}\n₹{p:,}", ha="center", va="bottom", fontweight="bold", color=fg_color)
            ax2.axhline(0, color=line_color, linewidth=1.5)
            ax2.set_xlim(min(prices)*0.9, max(prices)*1.1)
            ax2.set_ylim(-0.5, 0.5)
            ax2.axis("off")
            ax2.set_title(f"{company} - Price Tracker", color=fg_color)
            st.pyplot(fig2)
        else:
            st.warning(f"⚠️ No historical data available for {company} from {start_str}")
        st.markdown("---")

# TAB 3: PERFORMANCE - ENHANCED HEATMAP
# ==================== TAB 3: PERFORMANCE ====================
with tab3:
    st.header("🏆 Performance Analysis")

    # ----- PERFORMANCE BAR CHART (ORDER LOCKED, CLEAN LABELS) -----
    perf_df = filtered[["Company Name", "Percent Change"]].reset_index(drop=True)

    perf_df["Short Name"] = perf_df["Company Name"].apply(
        lambda x: x[:14] + "..." if len(x) > 14 else x
    )

    colors = [
        "#4C6EF5" if v >= 0 else "#F44336"
        for v in perf_df["Percent Change"]
    ]

    fig_perf = go.Figure(
        go.Bar(
            x=perf_df["Short Name"],
            y=perf_df["Percent Change"],
            marker_color=colors,
            customdata=perf_df["Company Name"],
            hovertemplate="<b>%{customdata}</b><br>% Change: %{y:.2f}%<extra></extra>"
        )
    )

    fig_perf.update_layout(
        title="Performance vs Record Price",
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font_color=fg_color,
        xaxis=dict(
            title="Company",
            tickangle=-60,
            categoryorder="array",
            categoryarray=perf_df["Short Name"]
        ),
        yaxis_title="% Change from Record Price",
        showlegend=False
    )

    fig_perf.add_hline(
        y=0,
        line_dash="dash",
        line_color="#9CA3AF"
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    # ----- TABLES -----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Top 5 Gainers")
        st.dataframe(
            filtered.nlargest(5, "Percent Change")[["Company Name", "Percent Change"]],
            use_container_width=True
        )

    with col2:
        st.subheader("📉 Top 5 Losers")
        st.dataframe(
            filtered.nsmallest(5, "Percent Change")[["Company Name", "Percent Change"]],
            use_container_width=True
        )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Top 5 Gainers")
        st.dataframe(filtered.nlargest(5, "Percent Change")[["Company Name", "Percent Change"]], use_container_width=True)
    with col2:
        st.subheader("📉 Top 5 Losers")
        st.dataframe(filtered.nsmallest(5, "Percent Change")[["Company Name", "Percent Change"]], use_container_width=True)
    st.subheader("🔥 Performance Heatmap")
    values = filtered["Absolute Current Price ($)"].fillna(0)
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    cols, rows = 6, (len(values) + 5) // 6
    fig, ax = plt.subplots(figsize=(16, rows * 1.8))
    for i, (comp, val) in enumerate(zip(filtered["Company Name"], values)):
        r, c = divmod(i, cols)
        color = plt.get_cmap("RdYlGn")(norm(val))
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color, edgecolor="white"))
        ax.text(c + 0.5, rows - r - 0.5, f"{comp}\n{val:+.1f}%", ha="center", va="center", fontsize=9, color="black")
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    ax.set_title("Performance Heatmap", color=fg_color, fontsize=16)
    st.pyplot(fig)

# TAB 4: SENTIMENT - ENHANCED WITH ICONS
with tab4:
    st.header("📰 Market Sentiment")
    try:
        news = GNews(language='en', country='IN', max_results=8)
        items = news.get_news("Indian stock market")
        sentiments = []
        for item in items:
            pol = TextBlob(item['title']).sentiment.polarity
            label = "Positive" if pol > 0.1 else "Negative" if pol < -0.1 else "Neutral"
            icon = "😊" if pol > 0.1 else "😞" if pol < -0.1 else "😐"
            sentiments.append(pol)
            with st.expander(f"{icon} {item['title']}"):
                st.write(f"**Source:** {item['publisher']['title']}")
                if 'image' in item and item['image']:
                    st.image(item['image'], use_column_width=True)
                st.write(f"→ **{label}** ({pol:+.2f})")
        avg = np.mean(sentiments)
        if avg > 0.1: 
            st.success(f"🌟 Overall: Positive ({avg:+.2f})")
        elif avg < -0.1: 
            st.error(f"😟 Overall: Negative ({avg:+.2f})")
        else: 
            st.info(f"⚖️ Overall: Neutral ({avg:+.2f})")
    except:
        st.warning("📰 News temporarily unavailable. Check connection.")

# ==================== TAB: STRATEGY 2026 ====================
with tab_strategy:
    st.header("🧭 Portfolio Strategy — 2026 Vision")

    st.markdown("""
    This section represents our **forward-looking portfolio construction philosophy**.
    It is **not performance-based**, but a **strategic allocation framework** designed
    for long-term wealth creation.
    """)

    # Target allocation (hard-coded mandate)
    strategy_data = pd.DataFrame({
        "Market Cap Segment": [
            "Mega & Large Cap",
            "Mid Cap",
            "Small Cap",
            "Micro Cap"
        ],
        "Target Allocation (%)": [25, 20, 40, 15]
    })

    col1, col2 = st.columns([2, 3])

    # ---- PIE CHART ----
    with col1:
        fig = px.pie(
            strategy_data,
            names="Market Cap Segment",
            values="Target Allocation (%)",
            hole=0.55,
            color_discrete_sequence=[
                "#1F3A5F",  # Navy (Large)
                "#4C6EF5",  # Blue (Mid)
                "#2FB344",  # Green (Small)
                "#ADB5BD"   # Grey (Micro)
            ]
        )
        fig.update_layout(
            showlegend=True,
            paper_bgcolor=plot_bg,
            plot_bgcolor=plot_bg,
            font_color=fg_color
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- EXPLANATION ----
    with col2:
        st.subheader("📌 Allocation Rationale")

        st.markdown("""
        **Mega & Large Cap (25%)**  
        → Stability, governance quality, downside protection  

        **Mid Cap (20%)**  
        → Earnings acceleration, scalable businesses  

        **Small Cap (40%)**  
        → Primary alpha generation, early growth capture  

        **Micro Cap (15%)**  
        → Optionality, deep value, asymmetric upside  

        ---
        📈 **Objective:**  
        Balance **capital protection** with **aggressive growth**, while maintaining
        diversification across market-cap cycles.
        """)

    st.divider()

    # ---- BAR VIEW (EXECUTIVE FRIENDLY) ----
    st.subheader("📊 Target Allocation Summary")

    bar_fig = px.bar(
        strategy_data,
        x="Market Cap Segment",
        y="Target Allocation (%)",
        text="Target Allocation (%)",
        color="Market Cap Segment",
        color_discrete_sequence=[
            "#1F3A5F",
            "#4C6EF5",
            "#2FB344",
            "#ADB5BD"
        ]
    )
    bar_fig.update_traces(texttemplate='%{text}%', textposition='outside')
    bar_fig.update_layout(
        showlegend=False,
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font_color=fg_color,
        yaxis_title="Allocation (%)",
        xaxis_title=""
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    st.info("🧠 This allocation acts as a **strategic compass**, not a short-term trading model.")


# TAB 5: PORTFOLIO P&L - ENHANCED WITH CHART
with tab_portfolio:
    st.header(f"💼 {st.session_state.user}'s Portfolio")
    stock = st.selectbox("Select Stock", df["Company Name"])
    row = df[df["Company Name"] == stock].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        shares = st.number_input("📊 Shares Owned", min_value=1, value=100)
    with col2:
        buy_price = st.number_input("💰 Your Buy Price", value=float(row["Record Price"]))
    current_value = shares * row["Current Price"]
    profit = (row["Current Price"] - buy_price) * shares
    profit_pct = (profit / (buy_price * shares)) * 100 if buy_price > 0 else 0
    st.metric("💎 Current Value", f"₹{current_value:,.0f}")
    st.metric("💹 Profit/Loss", f"₹{profit:,.0f}", delta=f"{profit_pct:+.1f}%")

    # New Feature: Simple P&L Bar Chart
    fig_pnl, ax_pnl = plt.subplots(figsize=(8, 4))
    categories = ['Investment', 'Current Value', 'Profit/Loss']
    values = [buy_price * shares, current_value, profit]
    colors = ['#f44336' if v < 0 else '#4caf50' for v in values]
    ax_pnl.bar(categories, values, color=colors)
    ax_pnl.set_title(f"{stock} - P&L Breakdown", color=fg_color)
    st.pyplot(fig_pnl)

# ==================== SUPER SMART FREE CHATBOX - ENHANCED WITH EMOJIS ====================
with tab_chat:
    st.header("🤖 QualSCORE AI Assistant — 100% FREE & Super Smart")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("💭 Ask anything: 'Best stock?', 'Target hit?', 'Nifty?', 'Reliance?'"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        p = prompt.lower().strip()
        reply = "💡 Ask me anything about your stocks! Try: 'Best gainer?' or 'Reliance status'."

        matched = False
        for _, row in df.iterrows():
            if row["Company Name"].lower() in p or row["Ticker"].lower().replace(".ns","").replace(".bo","") in p:
                status = "🎯 TARGET HIT!" if row["Current Price"] >= row["target Price"] else "⚠️ NEAR TARGET!" if row["Current Price"] >= row["target Price"]*0.95 else "✅ On Track"
                reply = f"**{row['Company Name']}**\n💰 Current: ₹{row['Current Price']:,}\n🎯 Target: ₹{row['target Price']:,}\n📊 Gain: {row['Percent Change']:+.2f}%\n📈 Status: **{status}**"
                matched = True
                break

        if not matched:
            if any(x in p for x in ["best", "top", "gainer"]):
                top = df.loc[df["Percent Change"].idxmax()]
                reply = f"🏆 TOP GAINER: **{top['Company Name']}** +{top['Percent Change']:+.2f}% 🔥"
            elif any(x in p for x in ["worst", "loser"]):
                bot = df.loc[df["Percent Change"].idxmin()]
                reply = f"📉 WORST: **{bot['Company Name']}** {bot['Percent Change']:+.2f}% 😞"
            elif "target hit" in p:
                hits = df[df["Current Price"] >= df["target Price"]]["Company Name"].tolist()
                reply = f"🎯 TARGET HITS: {', '.join(hits) if hits else 'None yet! Keep watching!'}"
            elif "nifty" in p or "sensex" in p:
                reply = f"📊 NIFTY 50: ₹{nifty:,.0f}\n📈 SENSEX: ₹{sensex:,.0f}"
            elif "profit" in p or "portfolio" in p:
                reply = "💼 Go to Portfolio tab → enter shares & buy price → see your profit! 📈"

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# FINAL STATUS - ENHANCED FOOTER
st.sidebar.success("🚀 QUALSCORE ACTIVE")
st.sidebar.info("🔐 Password • ⭐ Watchlist • 💼 P&L • 📱 WhatsApp Alerts • 🤖 FREE CHAT • 🔍 Search")
st.markdown("---")
st.markdown(f"© 2025 QualSCORE | Last Update: {datetime.now().strftime('%Y-%m-%d')}")
