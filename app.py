# app.py - FINAL QUALSCORE EDITION - LOGO + ALERTS + P&L + FREE CHATBOX
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

# ==================== PASSWORD PROTECTION ====================
st.markdown("<h2 style='text-align:center;color:#00d4ff;'>Enter Password</h2>", unsafe_allow_html=True)
password = st.text_input("Password", type="password", placeholder="Enter secret password")
SECRET_PASSWORD = "admin" # CHANGE THIS!
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
@keyframes glow {from{text-shadow:0 0 10px #00d4ff;}to{text-shadow:0 0 30px #00ff00;}}
.stMetric > div > div > div {background: linear-gradient(90deg, var(--primary), var(--secondary)); border-radius: 10px; padding: 10px;}
.stExpander {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
.card {background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.2);}
.pulse {animation: pulse 2s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(0,212,255,0.7);} 70% {box-shadow: 0 0 0 10px rgba(0,212,255,0);} 100% {box-shadow: 0 0 0 0 rgba(0,212,255,0);}}
</style>
""", unsafe_allow_html=True)

# ==================== DYNAMIC HEADER WITH LIVE TICKER ====================
header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
with header_col1:
    # Live Ticker Simulation (Nifty/Sensex with delta)
    nifty_change = (nifty - yf.Ticker("^NSEI").history(period="2d")["Close"].iloc[-2]) if nifty else 0
    sensex_change = (sensex - yf.Ticker("^BSESN").history(period="2d")["Close"].iloc[-2]) if sensex else 0
    st.markdown(f"""
    <div class="pulse card" style="background: linear-gradient(90deg, #00d4ff, #1e88e5); color: white; text-align: center;">
        <h3>📈 Live Market Ticker</h3>
        <p>NIFTY 50: ₹{nifty:,.0f} {'🟢' if nifty_change > 0 else '🔴'}{nifty_change:+.0f}</p>
        <p>SENSEX: ₹{sensex:,.0f} {'🟢' if sensex_change > 0 else '🔴'}{sensex_change:+.0f}</p>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    # User Avatar with Quick P&L Summary
    total_pnl = df["Percent Change"].sum() if not df.empty else 0
    st.markdown(f"""
    <div class="card" style="text-align: center; background: {bg_color}; color: {fg_color};">
        <h4>👤 {st.session_state.user}</h4>
        <p>Portfolio P&L: {total_pnl:+.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    # Quick Search Bar
    search_query = st.text_input("🔍 Quick Search Stock", placeholder="e.g., Reliance")
    if search_query:
        matching = df[df["Company Name"].str.contains(search_query, case=False, na=False)]
        if not matching.empty:
            st.success(f"Found: {matching['Company Name'].tolist()}")
        else:
            st.info("No matches—try another!")

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

# ==================== UPLOAD + THEME + LAYOUT + INDICES ====================
st.sidebar.header("Upload pythonmaster.xlsx")
uploaded_file = st.sidebar.file_uploader("Choose file", type=["xlsx"])
if not uploaded_file:
    st.error("Please upload your Excel file to continue!")
    st.stop()

# Customizable Themes
theme_presets = st.sidebar.selectbox("Theme Preset", ["Dark", "Light", "Bullish Green", "Bearish Red"])
if theme_presets == "Dark":
    bg_color = "#1a1a1a"
    fg_color = "white"
    line_color = "white"
    plot_bg = "#1a1a1a"
    primary_color = "#00d4ff"
elif theme_presets == "Light":
    bg_color = "white"
    fg_color = "black"
    line_color = "black"
    plot_bg = "white"
    primary_color = "#1e88e5"
elif theme_presets == "Bullish Green":
    bg_color = "#0f2e1a"
    fg_color = "#90EE90"
    line_color = "#90EE90"
    plot_bg = "#0f2e1a"
    primary_color = "#00ff00"
else: # Bearish Red
    bg_color = "#2e0f0f"
    fg_color = "#FFB6C1"
    line_color = "#FFB6C1"
    plot_bg = "#2e0f0f"
    primary_color = "#ff0000"

# Layout Toggle
full_width = st.sidebar.checkbox("Full Width Layout", value=True)
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

# ==================== DATA PROCESSING ====================
@st.cache_data(show_spinner=False)
def process_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = df.columns.str.strip()
    # FIX: Convert price columns to numeric to avoid str-float subtraction errors
    price_cols = ["Record Price", "Target Price"]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
    if "Index" not in df.columns: df["Index"] = "Unknown"
    missing = [c for c in required if c not in df.columns]
    if missing: st.error(f"Missing: {missing}"); st.stop()
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date of Publishing", "Record Price", "Target Price"])  # Drop if prices are NaN
    results = []
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip()
        if not ticker.endswith((".BO", ".NS")): ticker += ".BO"
        try:
            current = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
           
            # Compute Volatility (std dev of last 30 days returns)
            hist_30d = yf.Ticker(ticker).history(period="1mo")
            if not hist_30d.empty:
                returns_30d = hist_30d["Close"].pct_change().dropna()
                volatility = returns_30d.std() * np.sqrt(252) * 100 # Annualized
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
                "Record Price": float(row["Record Price"]),  # Ensure float
                "Current Price": round(float(current), 2),
                "target Price": float(row["Target Price"]),  # Ensure float
                "Index": row.get("Index", "Unknown"),
                "Date of Publishing": row["Date of Publishing"].date(),
                "Volatility (%)": round(volatility, 2),
                "Beta": round(beta, 2)
            })
        except: continue
    final_df = pd.DataFrame(results)
    final_df["Percent Change"] = ((final_df["Current Price"] - final_df["Record Price"]) / final_df["Record Price"] * 100).round(2)
    final_df["Distance from Target ($)"] = ((final_df["Current Price"] - final_df["target Price"]) / final_df["target Price"] * 100).round(2)
    final_df["Absolute Current Price ($)"] = final_df["Percent Change"]
    return final_df

with st.spinner("Processing your stocks..."):
    df = process_data(uploaded_file)
st.success(f"Processed {len(df)} stocks for {st.session_state.user}!")

# ==================== FILTERS ====================
st.sidebar.markdown("### Select Stocks for Trends")
selected_companies = st.sidebar.multiselect(
    "Choose companies", df["Company Name"].unique(),
    default=(st.session_state.watchlist + list(df["Company Name"].head(3).tolist()))[:3]
)
if not selected_companies: selected_companies = df["Company Name"].head(1).tolist()

period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
cutoff = datetime(1900, 1, 1)
if period == "Last 3 Months": cutoff = datetime.today() - timedelta(days=90)
elif period == "Last 6 Months": cutoff = datetime.today() - timedelta(days=180)
elif period == "Last 1 Year": cutoff = datetime.today() - timedelta(days=365)
filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Report", csv, "Stock_Report.csv", "text/csv")

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab_portfolio, tab_chat = st.tabs([
    "Overview", "Trends", "Performance", "Sentiment", "Portfolio", "Chat"
])

# TAB 1: OVERVIEW (Enhanced with Card-Based Layout)
with tab1:
    st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">Dashboard Overview</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Stocks", len(df))
    with col2: st.metric("Avg Return", f"{df['Percent Change'].mean():+.2f}%")
    with col3:
        top = df.loc[df["Percent Change"].idxmax()]
        st.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.2f}%")
   
    # Quick Stats Cards (Now in Aesthetic Cards)
    with st.expander("Quick Risk Stats", expanded=False):
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            avg_vol = df["Volatility (%)"].mean()
            st.markdown(f'<div class="card pulse" style="background: linear-gradient(90deg, #ff6b6b, #ee5a24); color: white; text-align: center;">Avg Volatility: {avg_vol:.2f}%</div>', unsafe_allow_html=True)
        with col_r2:
            avg_beta = df["Beta"].mean()
            st.markdown(f'<div class="card" style="background: linear-gradient(90deg, #4834d4, #686de0); color: white; text-align: center;">Avg Beta: {avg_beta:.2f}</div>', unsafe_allow_html=True)
        with col_r3:
            high_risk = len(df[df["Volatility (%)"] > 30])
            st.markdown(f'<div class="card" style="background: linear-gradient(90deg, #f093fb, #f5576c); color: white; text-align: center;">High Risk Stocks: {high_risk}</div>', unsafe_allow_html=True)
   
    if df["Index"].nunique() > 1:
        fig_pie = px.pie(df["Index"].value_counts().reset_index(), names="Index", values="count", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues)
        fig_pie.update_layout(paper_bgcolor=plot_bg, plot_bgcolor=plot_bg, font_color=fg_color)
        st.plotly_chart(fig_pie, use_container_width=True)
   
    st.subheader("Performance Table")
    disp = filtered[["Company Name", "Current Price", "target Price", "Percent Change", "Distance from Target ($)", "Volatility (%)", "Beta"]]
    styled = disp.style.format({
        "Current Price": "₹{:.2f}", "target Price": "₹{:.2f}",
        "Percent Change": "{:+.2f}%", "Distance from Target ($)": "{:+.2f}%",
        "Volatility (%)": "{:.2f}%"
    }).bar(subset=["Percent Change"], color=['#90EE90', '#FFB6C1'])
    st.dataframe(styled, use_container_width=True)

# TAB 2: TRENDS + TARGET ALERT (Unchanged - Kept All Charts)
with tab2:
    st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">Stock Trends & Price Tracker</div>', unsafe_allow_html=True)
    for company in selected_companies:
        row = df[df["Company Name"] == company].iloc[0]
        st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">### {company}</div>', unsafe_allow_html=True)
        if row["Current Price"] >= row["target Price"]:
            st.success(f"TARGET HIT! {company} reached ₹{row['Current Price']:,}")
            alert = f"QUALSCORE ALERT: {company} HIT TARGET! Current: ₹{row['Current Price']:,} | Target: ₹{row['target Price']:,}"
            wa_link = f"https://wa.me/?text={alert.replace(' ', '%20')}"
            st.markdown(f'''
            <a href="{wa_link}" target="_blank">
                <button style="background:#25D366;color:white;padding:10px 20px;border:none;border-radius:10px;font-weight:bold;">
                    Send WhatsApp Alert
                </button>
            </a>
            ''', unsafe_allow_html=True)
        elif row["Current Price"] >= row["target Price"] * 0.95:
            # FIX: Ensure subtraction uses floats
            diff = float(row["target Price"]) - float(row["Current Price"])
            st.error(f"NEAR TARGET! Only ₹{diff:.0f} away!")
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
            st.markdown(f'<a href="{wa_url}" target="_blank"><button style="background:#25D366;color:white;padding:10px 20px;border:none;border-radius:10px;">Share on WhatsApp</button></a>', unsafe_allow_html=True)
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
            st.warning(f"No historical data available for {company} from {start_str}")
        st.markdown("---")

# TAB 3: PERFORMANCE (Unchanged - Kept All Charts)
with tab3:
    st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">Performance Analysis</div>', unsafe_allow_html=True)
    st.bar_chart(filtered.set_index("Company Name")["Percent Change"].sort_values(ascending=False))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Gainers")
        st.dataframe(filtered.nlargest(5, "Percent Change")[["Company Name", "Percent Change"]], use_container_width=True)
    with col2:
        st.subheader("Top 5 Losers")
        st.dataframe(filtered.nsmallest(5, "Percent Change")[["Company Name", "Percent Change"]], use_container_width=True)
    st.subheader("Heatmap")
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

# TAB 4: SENTIMENT (Unchanged - Kept All Elements)
with tab4:
    st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">Market Sentiment</div>', unsafe_allow_html=True)
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

# TAB 5: PORTFOLIO P&L (Enhanced with Drag-and-Drop Builder Simulation)
with tab_portfolio:
    st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">{st.session_state.user}\'s Portfolio Builder</div>', unsafe_allow_html=True)
    
    # Drag-and-Drop Simulation: Two Columns - Watchlist (Left) and Portfolio (Right)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📋 Watchlist (Drag From Here)")
        watchlist_options = st.multiselect("Available Stocks", df["Company Name"].tolist(), default=st.session_state.watchlist)
        if st.button("➡️ Add to Portfolio", key="add_to_port"):
            if 'portfolio' not in st.session_state:
                st.session_state.portfolio = []
            for stock in watchlist_options:
                if stock not in st.session_state.portfolio:
                    st.session_state.portfolio.append(stock)
            st.rerun()
    
    with col_right:
        st.subheader("💼 My Portfolio (Drop Here)")
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        portfolio_options = st.multiselect("Portfolio Holdings", st.session_state.portfolio, disabled=True, default=st.session_state.portfolio)
        if st.button("⬅️ Remove from Portfolio", key="remove_from_port"):
            removed = [stock for stock in st.session_state.portfolio if stock not in portfolio_options]
            st.session_state.portfolio = [stock for stock in st.session_state.portfolio if stock in portfolio_options]
            st.rerun()
        
        # Allocation Pie Chart (Auto-Calcs Based on Selected)
        if st.session_state.portfolio:
            alloc_df = df[df["Company Name"].isin(st.session_state.portfolio)]
            if not alloc_df.empty:
                fig_alloc = px.pie(alloc_df, names="Company Name", values="Current Price", hole=0.4,
                                   color_discrete_sequence=px.colors.sequential.Viridis)
                fig_alloc.update_layout(title="Portfolio Allocation by Value")
                st.plotly_chart(fig_alloc, use_container_width=True)
        
        # Projected Returns (Simple Linear Extrapolation)
        if st.session_state.portfolio:
            proj_returns = []
            for stock_name in st.session_state.portfolio:
                row = df[df["Company Name"] == stock_name].iloc[0]
                hist_proj = yf.download(row["Ticker"], period="3mo")
                if not hist_proj.empty:
                    if isinstance(hist_proj.columns, pd.MultiIndex):
                        hist_proj.columns = hist_proj.columns.droplevel(1)
                    hist_proj['Returns'] = hist_proj['Close'].pct_change()
                    recent_trend = hist_proj['Returns'].tail(10).mean() * 30  # 30-day projection
                    proj_returns.append({"Stock": stock_name, "Projected Monthly Return": round(recent_trend * 100, 2)})
            if proj_returns:
                proj_df = pd.DataFrame(proj_returns)
                st.subheader("📊 Projected Returns (Next Month)")
                st.dataframe(proj_df)
    
    # Legacy P&L Calculator (Kept for Compatibility)
    if st.session_state.portfolio:
        stock = st.selectbox("Select Stock for Detailed P&L", st.session_state.portfolio)
    else:
        stock = st.selectbox("Select Stock", df["Company Name"])
    row = df[df["Company Name"] == stock].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        shares = st.number_input("Shares Owned", min_value=1, value=100)
    with col2:
        buy_price = st.number_input("Your Buy Price", value=float(row["Record Price"]))
    current_value = shares * row["Current Price"]
    profit = (row["Current Price"] - buy_price) * shares
    profit_pct = (profit / (buy_price * shares)) * 100 if buy_price > 0 else 0
    st.metric("Current Value", f"₹{current_value:,.0f}")
    st.metric("Profit/Loss", f"₹{profit:,.0f}", delta=f"{profit_pct:+.1f}%")

# ==================== SUPER SMART FREE CHATBOX (ONLY THIS PART CHANGED) ====================
with tab_chat:
    st.markdown(f'<div class="card" style="background: {bg_color}; color: {fg_color};">QualSCORE AI Assistant — 100% FREE & Super Smart</div>', unsafe_allow_html=True)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask anything: 'Best stock?', 'Target hit?', 'Nifty?', 'Reliance?'"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        p = prompt.lower().strip()
        reply = "Ask me anything about your stocks!"
        matched = False
        for _, row in df.iterrows():
            if row["Company Name"].lower() in p or row["Ticker"].lower().replace(".ns","").replace(".bo","") in p:
                status = "TARGET HIT!" if row["Current Price"] >= row["target Price"] else "NEAR TARGET!" if row["Current Price"] >= row["target Price"]*0.95 else "On Track"
                reply = f"**{row['Company Name']}**\nCurrent: ₹{row['Current Price']:,}\nTarget: ₹{row['target Price']:,}\nGain: {row['Percent Change']:+.2f}%\nStatus: **{status}**"
                matched = True
                break
        if not matched:
            if any(x in p for x in ["best", "top", "gainer"]):
                top = df.loc[df["Percent Change"].idxmax()]
                reply = f"TOP GAINER: **{top['Company Name']}** +{top['Percent Change']:+.2f}%"
            elif any(x in p for x in ["worst", "loser"]):
                bot = df.loc[df["Percent Change"].idxmin()]
                reply = f"WORST: **{bot['Company Name']}** {bot['Percent Change']:+.2f}%"
            elif "target hit" in p:
                hits = df[df["Current Price"] >= df["target Price"]]["Company Name"].tolist()
                reply = f"TARGET HIT: {', '.join(hits) if hits else 'None yet!'}"
            elif "nifty" in p or "sensex" in p:
                reply = f"NIFTY 50: ₹{nifty:,.0f}\nSENSEX: ₹{sensex:,.0f}"
            elif "profit" in p or "portfolio" in p:
                reply = "Go to Portfolio tab → enter shares & buy price → see your profit!"
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# FINAL STATUS
st.sidebar.success("QUALSCORE ACTIVE")
st.sidebar.info("Password • Watchlist • P&L • WhatsApp Alerts • FREE CHAT")
