# app.py - FINAL PRIVATE LEGEND - PASSWORD + ORIGINAL + 7 FEATURES
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

# ==================== PASSWORD PROTECTION (CHANGE THIS!) ====================
st.markdown("<h2 style='text-align:center;color:#00d4ff;'>Enter Password</h2>", unsafe_allow_html=True)
password = st.text_input("Password", type="password", placeholder="Enter secret password")

# CHANGE THIS TO YOUR OWN PASSWORD
SECRET_PASSWORD = "stockking123"  # CHANGE THIS NOW!

if password != SECRET_PASSWORD:
    st.error("Incorrect password. Access denied.")
    st.stop()

# ==================== AUTO REFRESH ====================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
elapsed = time.time() - st.session_state.last_refresh
refresh_in = 60 - int(elapsed)
if elapsed >= 60:
    st.session_state.last_refresh = time.time()
    st.rerun()
else:
    st.sidebar.caption(f"Auto-refresh in {refresh_in}s")

# ==================== PAGE CONFIG + ANIMATED HEADER ====================
st.set_page_config(
    page_title="Stock Tracker Pro",
    page_icon="Chart increasing",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# Stock Tracker Pro\nPrivate Elite Dashboard"}
)

st.markdown("""
<div style="text-align:center;padding:20px;background:linear-gradient(90deg,#1e88e5,#00d4ff);border-radius:15px;margin-bottom:20px;">
    <h1 style="color:white;margin:0;animation:glow 2s infinite alternate;">STOCK TRACKER PRO</h1>
    <p style="color:white;margin:5px;font-size:18px;">Private. Elite. Legendary.</p>
</div>
<style>@keyframes glow {from{text-shadow:0 0 10px #00d4ff;}to{text-shadow:0 0 30px #00ff00;}}</style>
""", unsafe_allow_html=True)

# ==================== MULTI-USER + WATCHLIST ====================
if 'user' not in st.session_state:
    st.session_state.user = "Elite Trader"
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

user = st.sidebar.text_input("Your Name", value=st.session_state.user)
if user != st.session_state.user:
    st.session_state.user = user
    st.sidebar.success(f"Welcome back, {user}!")

st.sidebar.markdown("### Star Your Watchlist")
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

# ==================== SIDEBAR UPLOAD + THEME + LIVE INDICES ====================
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

st.sidebar.markdown("### LIVE MARKET")
@st.cache_data(ttl=15)
def get_indices():
    try:
        nifty = yf.Ticker("^NSEI").history(period="1d")["Close"].iloc[-1]
        sensex = yf.Ticker("^BSESN").history(period="1d")["Close"].iloc[-1]
        return round(nifty, 2), round(sensex, 2)
    except:
        return None, None

nifty, sensex = get_indices()
if nifty and sensex:
    st.sidebar.metric("**NIFTY 50**", f"₹{nifty:,.0f}")
    st.sidebar.metric("**SENSEX**", f"₹{sensex:,.0f}")
else:
    st.sidebar.warning("Loading live indices...")

# ==================== DATA PROCESSING ====================
@st.cache_data(show_spinner=False)
def process_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = df.columns.str.strip()
    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
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
            pct = ((current - row["Record Price"]) / row["Record Price"]) * 100
            results.append({
                "Date of Publishing": row["Date of Publishing"].date(),
                "Company Name": company,
                "Ticker": ticker,
                "Index": row.get("Index", "Unknown"),
                "Record Price": row["Record Price"],
                "Current Price": round(current, 2),
                "target Price": row["Target Price"],
                "Absolute Current Price (%)": round(pct, 2),
            })
        except:
            continue
        progress.progress((i + 1) / len(df))
    status.empty()
    progress.empty()
    final_df = pd.DataFrame(results)
    final_df["Percent Change"] = ((final_df["Current Price"] - final_df["Record Price"]) / final_df["Record Price"] * 100).round(2)
    final_df["Distance from Target (%)"] = ((final_df["Current Price"] - final_df["target Price"]) / final_df["target Price"] * 100).round(2)
    return final_df

with st.spinner("Processing your stocks..."):
    df = process_data(uploaded_file)
st.success(f"Processed {len(df)} stocks for {st.session_state.user}!")

# ==================== FILTERS ====================
st.sidebar.markdown("### Select Stocks for Trends")
selected_companies = st.sidebar.multiselect(
    "Choose companies",
    options=df["Company Name"].unique(),
    default=(st.session_state.watchlist + list(df["Company Name"].head(3).tolist()))[:3]
)
if not selected_companies:
    selected_companies = df["Company Name"].head(1).tolist()

period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
cutoff = datetime(1900, 1, 1)
if period == "Last 3 Months": cutoff = datetime.today() - timedelta(days=90)
elif period == "Last 6 Months": cutoff = datetime.today() - timedelta(days=180)
elif period == "Last 1 Year": cutoff = datetime.today() - timedelta(days=365)
filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Full Report (CSV)", csv, "Stock_Report.csv", "text/csv")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab_portfolio = st.tabs(["Overview", "Trends", "Performance", "Sentiment", "Portfolio"])

with tab1:
    st.header("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Stocks", len(df))
    with col2: st.metric("Avg Return", f"{df['Percent Change'].mean():+.2f}%")
    with col3:
        top = df.loc[df["Percent Change"].idxmax()]
        st.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.2f}%")
    if df["Index"].nunique() > 1:
        fig_pie = px.pie(df["Index"].value_counts().reset_index(), names="Index", values="count", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues)
        fig_pie.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=fg_color)
        st.plotly_chart(fig_pie, use_container_width=True)
    st.subheader("Performance Table")
    disp = filtered[["Company Name", "Current Price", "target Price", "Percent Change", "Distance from Target (%)"]]
    styled = disp.style.format({
        "Current Price": "₹{:.2f}",
        "target Price": "₹{:.2f}",
        "Percent Change": "{:+.2f}%",
        "Distance from Target (%)": "{:+.2f}%"
    }).bar(subset=["Percent Change"], color=['#90EE90', '#FFB6C1'])
    st.dataframe(styled, use_container_width=True)

with tab2:
    st.header("Stock Trends & Price Tracker")
    for company in selected_companies:
        row = df[df["Company Name"] == company].iloc[0]
        st.markdown(f"### {company}")
        if row["Current Price"] >= row["target Price"]:
            st.success(f"TARGET HIT! {company} reached ₹{row['Current Price']:,}")
        elif row["Current Price"] >= row["target Price"] * 0.95:
            st.error(f"NEAR TARGET! Only ₹{row['target Price'] - row['Current Price']:.0f} away!")
        hist = yf.download(row["Ticker"], period="6mo")
        if not hist.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(hist.index, hist["Close"], color="#00d4ff", linewidth=2.5)
            ax.axhline(row["target Price"], color="orange", linestyle="--", linewidth=2, label=f"Target ₹{row['target Price']}")
            ax.grid(True, alpha=0.3, color=line_color)
            ax.set_title(f"{company} - 6 Month Trend", color=fg_color, fontsize=16)
            ax.legend(facecolor=bg_color, labelcolor=fg_color)
            st.pyplot(fig)

            # WHATSAPP SHARE
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor=bg_color)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            wa_msg = f"*{company}* is at ₹{row['Current Price']:,} | Target: ₹{row['target Price']:,}"
            wa_url = f"https://wa.me/?text={wa_msg.replace(' ', '%20')}"
            st.markdown(f'<a href="{wa_url}" target="_blank"><button style="background:#25D366;color:white;padding:10px 20px;border:none;border-radius:10px;">Share on WhatsApp</button></a>', unsafe_allow_html=True)

            # PRICE TRACKER
            fig2, ax2 = plt.subplots(figsize=(12, 2))
            prices = [row["Record Price"], row["Current Price"], row["target Price"]]
            labels = ["Record", "Current", "Target"]
            colors = ["red", "#1e88e5", "green"]
            for p, l, c in zip(prices, labels, colors):
                ax2.scatter(p, 0, color=c, s=200, edgecolors=line_color, linewidth=2)
                ax2.text(p, 0.15, f"{l}\n₹{p:,}", ha="center", va="bottom", fontweight="bold", color=fg_color)
            ax2.axhline(0, color=line_color, linewidth=1.5)
            ax2.set_xlim(min(prices)*0.9, max(prices)*1.1)
            ax2.set_ylim(-0.5, 0.5)
            ax2.axis("off")
            ax2.set_title(f"{company} - Price Tracker", color=fg_color)
            st.pyplot(fig2)
        st.markdown("---")

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
    st.subheader("Heatmap")
    values = filtered["Absolute Current Price (%)"].fillna(0)
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

with tab4:
    st.header("Market Sentiment")
    try:
        news = GNews(language='en', country='IN', max_results=8)
        items = news.get_news("Indian stock market")
        sentiments = []
        for item in items:
            pol = TextBlob(item['title']).sentiment.polarity
            label = "Positive" if pol > 0.1 else "Negative" if pol < -0.1 else "Neutral"
            sentiments.append(pol)
            with st.expander(item['title']):
                st.write(f"**Source:** {item['publisher']['title']}")
                if 'image' in item and item['image']:
                    st.image(item['image'], use_column_width=True)
                st.write(f"→ **{label}** ({pol:+.2f})")
        avg = np.mean(sentiments)
        if avg > 0.1: st.success(f"Overall: Positive ({avg:+.2f})")
        elif avg < -0.1: st.error(f"Overall: Negative ({avg:+.2f})")
        else: st.info(f"Overall: Neutral ({avg:+.2f})")
    except:
        st.warning("News temporarily unavailable")

with tab_portfolio:
    st.header(f"{st.session_state.user}'s Portfolio")
    stock = st.selectbox("Select Stock", df["Company Name"])
    row = df[df["Company Name"] == stock].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        shares = st.number_input("Shares Owned", min_value=1, value=100)
    with col2:
        buy_price = st.number_input("Your Buy Price", value=float(row["Record Price"]))
    value = shares * row["Current Price"]
    profit = (row["Current Price"] - buy_price) * shares
    st.metric("Current Value", f"₹{value:,.0f}")
    st.metric("Profit/Loss", f"₹{profit:,.0f}", delta=f"{(profit/(buy_price*shares)*100):+.1f}%")

# FINAL STATUS
st.sidebar.success("PRIVATE APP ACTIVE")
st.sidebar.info("Password Protected • Secret Link Only • Elite Access")
