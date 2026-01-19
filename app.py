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

@st.cache_data
def load_master_data():
    df = pd.DataFrame(STOCK_MASTER)
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    return df.dropna(subset=["Date of Publishing"])

MASTER_DF = load_master_data()
# ────────────────────────────────────────────────
#  PROFESSIONAL DARK THEME + TYPOGRAPHY
# ────────────────────────────────────────────────

st.set_page_config(page_title="QualSCORE • Professional", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: #0d1117;
        color: #c9d1d9;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1450px !important;
    }
    h1 { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px; color: #ffffff !important; }
    h2 { font-size: 1.8rem; font-weight: 600; color: #e6edf3 !important; }
    h3 { font-size: 1.4rem; font-weight: 600; color: #c9d1d9 !important; }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 10px;
        padding: 6px;
        gap: 4px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important;
        border-radius: 6px !important;
        padding: 10px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #ffffff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    }
    .metric-card {
        background: linear-gradient(145deg, #1f2937, #111827);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #30363d;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .positive { color: #3fb950 !important; }
    .negative { color: #f85149 !important; }
    
    hr {
        border-color: #30363d !important;
        margin: 2rem 0;
    }
    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #2ea043;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  PASSWORD + SIDEBAR
# ────────────────────────────────────────────────

if "password_ok" not in st.session_state:
    st.session_state.password_ok = False

col_pw = st.columns([1,2,1])[1]
with col_pw:
    st.markdown("<h3 style='text-align:center'>QualSCORE Access</h3>", unsafe_allow_html=True)
    pw = st.text_input("Enter password", type="password", key="pw_input")
    if st.button("Login", use_container_width=True):
        if pw == "admin":           # ← change this in production
            st.session_state.password_ok = True
            st.rerun()
        else:
            st.error("Incorrect password")

if not st.session_state.password_ok:
    st.stop()

# ────────────────────────────────────────────────
#  HEADER
# ────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding:1.8rem 0 1.2rem;">
    <h1>QualSCORE</h1>
    <p style="color:#8b949e; font-size:1.15rem; margin-top:0.4rem;">
        Fundamental • Technical • Qualitative Analysis • Professional Edition
    </p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
#  SIDEBAR (cleaner)
# ────────────────────────────────────────────────

with st.sidebar:
    st.header("Dashboard Controls")
    
    user_name = st.text_input("Your Name", value=st.session_state.get("user_name", "Analyst"))
    st.session_state.user_name = user_name
    
    st.markdown("**Watchlist**")
    watchlist_input = st.multiselect(
        "Starred companies",
        options=MASTER_DF["Company Name"].tolist(),
        default=st.session_state.get("watchlist", []),
        placeholder="Select to highlight"
    )
    st.session_state.watchlist = watchlist_input

    st.markdown("---")
    st.caption("Data refresh")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ────────────────────────────────────────────────
#  DATA PROCESSING (your original logic preserved)
# ────────────────────────────────────────────────

@st.cache_data(ttl=300)
def process_data(df_master):
    results = []
    progress = st.progress(0)
    total = len(df_master)
    
    for i, row in df_master.iterrows():
        ticker = str(row["Ticker"]).strip()
        if not ticker.endswith((".BO", ".NS")):
            ticker += ".BO"
        try:
            current = yf.Ticker(ticker).history(period="5d")["Close"].iloc[-1]
            
            hist_30d = yf.Ticker(ticker).history(period="1mo")
            volatility = hist_30d["Close"].pct_change().std() * np.sqrt(252) * 100 if not hist_30d.empty else 0
            
            hist_stock = yf.Ticker(ticker).history(period="1y")
            hist_nifty = yf.Ticker("^NSEI").history(period="1y")
            beta = 1.0
            if not hist_stock.empty and not hist_nifty.empty:
                ret_s = hist_stock["Close"].pct_change().dropna()
                ret_n = hist_nifty["Close"].pct_change().dropna()
                common = ret_s.index.intersection(ret_n.index)
                if len(common) > 10:
                    X = ret_n.loc[common].values.reshape(-1,1)
                    y = ret_s.loc[common].values
                    beta = LinearRegression().fit(X, y).coef_[0]
            
            results.append({
                "Company Name": row["Company Name"],
                "Ticker": ticker,
                "Record Price": row["Record Price"],
                "Current Price": round(current, 2),
                "Target Price": row["Target Price"],
                "Index": row.get("Index", "Unknown"),
                "Date of Publishing": row["Date of Publishing"].date(),
                "Volatility (%)": round(volatility, 1),
                "Beta": round(beta, 2)
            })
        except:
            pass
        progress.progress((i+1)/total)
    
    final = pd.DataFrame(results)
    final["Percent Change"] = ((final["Current Price"] - final["Record Price"]) / final["Record Price"] * 100).round(1)
    final["Distance from Target (%)"] = ((final["Current Price"] - final["Target Price"]) / final["Target Price"] * 100).round(1)
    return final

with st.spinner("Loading market data..."):
    df = process_data(MASTER_DF)

# ────────────────────────────────────────────────
#  TABS (your original structure — only styling improved)
# ────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab_port, tab_chat = st.tabs([
    "Overview", "Trends & Alerts", "Performance", "Sentiment", "Portfolio", "AI Chat"
])

# ───── Overview ─────
with tab1:
    st.subheader("Market Overview")
    
    cols = st.columns(4)
    with cols[0]:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric("Total Stocks", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[1]:
        avg_ret = df["Percent Change"].mean()
        st.metric("Average Return", f"{avg_ret:+.1f}%", delta_color="normal" if avg_ret >= 0 else "inverse")
    
    with cols[2]:
        top_g = df.loc[df["Percent Change"].idxmax()]
        st.metric("Top Gainer", f"{top_g['Percent Change']:+.1f}%", f"{top_g['Company Name']}")
    
    with cols[3]:
        tgt_hit = len(df[df["Current Price"] >= df["Target Price"]])
        st.metric("Target Hits", tgt_hit)
    
    st.markdown("### Quick Risk Profile")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1: st.metric("Avg Volatility", f"{df['Volatility (%)'].mean():.1f}%")
    with col_r2: st.metric("Avg Beta", f"{df['Beta'].mean():.2f}")
    with col_r3: st.metric("High Risk (>35% vol)", len(df[df["Volatility (%)"] > 35]))
    
    if df["Index"].nunique() > 1:
        fig_pie = px.pie(df, names="Index", hole=0.45, title="Stocks by Index")
        fig_pie.update_layout(
            plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
            font_color="#c9d1d9", height=380
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("All Stocks")
    styled = df.style.format({
        "Current Price": "₹{:,.1f}",
        "Target Price": "₹{:,.1f}",
        "Percent Change": "{:+.1f}%",
        "Distance from Target (%)": "{:+.1f}%",
        "Volatility (%)": "{:.1f}%"
    }).bar(subset=["Percent Change"], color=["#f85149", "#3fb950"], vmin=-50, vmax=150)
    st.dataframe(styled, use_container_width=True)

# ───── Trends ───── (your original full logic kept)
with tab2:
    st.subheader("Stock Trends & Target Tracker")
    
    selected_comp = st.multiselect(
        "Select companies to display",
        options=df["Company Name"].tolist(),
        default=st.session_state.watchlist[:4] if st.session_state.watchlist else df["Company Name"].head(4).tolist()
    )
    
    if not selected_comp:
        st.info("Select at least one company")
    else:
        for company in selected_comp:
            row = df[df["Company Name"] == company].iloc[0]
            with st.expander(f"{company} • Current: ₹{row['Current Price']:,.1f}", expanded=len(selected_comp)<=3):
                if row["Current Price"] >= row["Target Price"]:
                    st.success(f"🎯 TARGET ACHIEVED — ₹{row['Current Price']:,.1f}")
                elif row["Current Price"] >= row["Target Price"] * 0.95:
                    st.warning(f"Approaching target — {row['Target Price'] - row['Current Price']:.0f} away")
                
                start_date = row["Date of Publishing"].strftime("%Y-%m-%d")
                hist = yf.download(row["Ticker"], start=start_date)
                
                if not hist.empty:
                    fig = px.line(hist.reset_index(), x="Date", y="Close",
                                  title=f"Price since {start_date}")
                    fig.add_hline(y=row["Record Price"], line_dash="dot", line_color="#f9c851",
                                  annotation_text=f"Entry ₹{row['Record Price']:,.1f}")
                    fig.add_hline(y=row["Target Price"], line_dash="dash", line_color="#3fb950",
                                  annotation_text=f"Target ₹{row['Target Price']:,.1f}")
                    fig.update_layout(
                        height=480, plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                        font_color="#c9d1d9", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Your original price tracker mini chart
                fig2, ax2 = plt.subplots(figsize=(10, 2.2))
                prices = [row["Record Price"], row["Current Price"], row["Target Price"]]
                labels = ["Entry", "Current", "Target"]
                colors = ["#f85149", "#58a6ff", "#3fb950"]
                for p, lbl, c in zip(prices, labels, colors):
                    ax2.scatter(p, 0, s=180, color=c, edgecolors="#c9d1d9", linewidth=2)
                    ax2.text(p, 0.18, f"{lbl}\n₹{p:,.0f}", ha="center", va="bottom", color="#e6edf3")
                ax2.axhline(0, color="#30363d", lw=1.5)
                ax2.set_xlim(min(prices)*0.92, max(prices)*1.08)
                ax2.set_ylim(-0.4, 0.6)
                ax2.axis("off")
                st.pyplot(fig2)

# ───── Performance, Sentiment, Portfolio, Chat ─────
# (kept exactly your original logic — only wrapped metrics in cards where it makes sense)

with tab3:
    st.subheader("Performance Ranking")
    st.bar_chart(df.set_index("Company Name")["Percent Change"].sort_values(ascending=False))
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 Gainers**")
        st.dataframe(df.nlargest(5, "Percent Change")[["Company Name", "Percent Change"]])
    with c2:
        st.markdown("**Top 5 Losers**")
        st.dataframe(df.nsmallest(5, "Percent Change")[["Company Name", "Percent Change"]])
    
    # Your original heatmap
    st.subheader("Performance Heatmap")
    values = df["Percent Change"].fillna(0)
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    cols, rows = 6, (len(values) + 5) // 6
    fig, ax = plt.subplots(figsize=(16, max(3, rows * 1.6)))
    for i, (name, val) in enumerate(zip(df["Company Name"], values)):
        r, c = divmod(i, cols)
        color = plt.get_cmap("RdYlGn")(norm(val))
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, facecolor=color, edgecolor="#0d1117"))
        ax.text(c + 0.5, rows - r - 0.5, f"{name}\n{val:+.1f}%", ha="center", va="center",
                fontsize=9, color="black" if abs(val) < 60 else "white")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    st.pyplot(fig)

with tab4:
    # Your original news + sentiment logic (unchanged except better container)
    st.subheader("Market Sentiment — India")
    try:
        news = GNews(language='en', country='IN', max_results=8)
        items = news.get_news("Indian stock market")
        sentiments = []
        for item in items:
            pol = TextBlob(item['title']).sentiment.polarity
            label = "Positive" if pol > 0.12 else "Negative" if pol < -0.12 else "Neutral"
            icon = "🟢" if pol > 0.12 else "🔴" if pol < -0.12 else "⚪"
            sentiments.append(pol)
            with st.expander(f"{icon} {item['title']}"):
                st.caption(item['publisher']['title'])
                st.write(f"Sentiment: **{label}** ({pol:+.2f})")
        if sentiments:
            avg = np.mean(sentiments)
            if avg > 0.1:
                st.success(f"Overall market sentiment: Positive ({avg:+.2f})")
            elif avg < -0.1:
                st.error(f"Overall market sentiment: Negative ({avg:+.2f})")
            else:
                st.info(f"Overall market sentiment: Neutral ({avg:+.2f})")
    except:
        st.warning("News feed currently unavailable")

with tab_port:
    st.subheader(f"Portfolio Calculator — {st.session_state.user_name}")
    stock = st.selectbox("Stock", df["Company Name"])
    row = df[df["Company Name"] == stock].iloc[0]
    
    c1, c2 = st.columns(2)
    with c1:
        shares = st.number_input("Number of shares", min_value=1, value=100, step=10)
    with c2:
        buy_price = st.number_input("Your purchase price (₹)", value=float(row["Record Price"]), step=0.5)
    
    curr_val = shares * row["Current Price"]
    pnl = (row["Current Price"] - buy_price) * shares
    pnl_pct = (pnl / (buy_price * shares)) * 100 if buy_price > 0 else 0
    
    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Current Position Value", f"₹{curr_val:,.0f}")
    with mc2:
        st.metric("Profit / Loss", f"₹{pnl:,.0f}", delta=f"{pnl_pct:+.1f}%",
                  delta_color="normal" if pnl >= 0 else "inverse")
    
    # Your original P&L bar
    fig_pnl, ax_pnl = plt.subplots(figsize=(9, 4))
    cats = ["Invested", "Current Value", "P&L"]
    vals = [buy_price * shares, curr_val, pnl]
    cols = ["#f85149" if v < 0 else "#3fb950" for v in vals]
    ax_pnl.bar(cats, vals, color=cols)
    ax_pnl.axhline(0, color="#6e7681", lw=1)
    ax_pnl.set_title(f"P&L Breakdown — {stock}")
    ax_pnl.set_facecolor("#161b22")
    fig_pnl.set_facecolor("#0d1117")
    for t in ax_pnl.get_xticklabels() + ax_pnl.get_yticklabels():
        t.set_color("#c9d1d9")
    st.pyplot(fig_pnl)

with tab_chat:
    # Your original chat logic (kept 100%)
    st.subheader("QualSCORE Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about any stock, performance, targets…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        reply = "Ask me about your stocks!"
        p = prompt.lower()
        found = False
        
        for _, r in df.iterrows():
            if r["Company Name"].lower() in p or r["Ticker"].lower().replace(".bo","") in p:
                status = "🎯 TARGET HIT" if r["Current Price"] >= r["Target Price"] else \
                         "Approaching target" if r["Current Price"] >= r["Target Price"]*0.95 else "On track"
                reply = f"**{r['Company Name']}**\nCurrent: ₹{r['Current Price']:,.1f}\nTarget: ₹{r['Target Price']:,.1f}\nGain: {r['Percent Change']:+.1f}%\nStatus: **{status}**"
                found = True
                break
        
        if not found:
            if any(w in p for w in ["best", "top", "gainer"]):
                top = df.loc[df["Percent Change"].idxmax()]
                reply = f"Top performer: **{top['Company Name']}** {top['Percent Change']:+.1f}%"
            elif any(w in p for w in ["worst", "loser"]):
                bot = df.loc[df["Percent Change"].idxmin()]
                reply = f"Worst performer: **{bot['Company Name']}** {bot['Percent Change']:+.1f}%"
            elif "target" in p and "hit" in p:
                hits = df[df["Current Price"] >= df["Target Price"]]["Company Name"].tolist()
                reply = "Target achieved: " + (", ".join(hits) if hits else "None yet")
        
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

st.markdown("---")
st.caption("QualSCORE Professional • Internal Use • Last update: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
