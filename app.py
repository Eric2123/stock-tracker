# app.py - FINAL PERFECTION - CONTRAST FIXED - DARK/LIGHT MODE GOD MODE
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

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Stock Tracker Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
.main { padding: 1rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0; text-align: center; }
.stButton>button { background-color: #1e88e5; color: white; border-radius: 8px; }
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
line_color = "white" if theme == "Dark" else "black"  # AUTO CONTRAST LINE

# Apply theme to matplotlib
plt.rcParams['text.color'] = fg_color
plt.rcParams['axes.labelcolor'] = fg_color
plt.rcParams['xtick.color'] = fg_color
plt.rcParams['ytick.color'] = fg_color
plt.rcParams['axes.edgecolor'] = line_color

# ------------------- PROCESS DATA -------------------
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
        record = row["Record Price"]
        target = row["Target Price"]

        status.text(f"Fetching {company}...")
        try:
            data = yf.Ticker(ticker).history(period="1y")
            if data.empty: continue
            data.index = data.index.tz_localize(None)
            current = data["Close"].iloc[-1]
            price_3m = data[data.index >= three_m]["Close"].iloc[0] if not data[data.index >= three_m].empty else None
            price_6m = data[data.index >= six_m]["Close"].iloc[0] if not data[data.index >= six_m].empty else None

            pct = ((current - record) / record) * 100
            pct_3m = ((price_3m - record) / record) * 100 if price_3m else None
            pct_6m = ((price_6m - record) / record) * 100 if price_6m else None

            results.append({
                "Date of Publishing": row["Date of Publishing"].date(),
                "Company Name": company,
                "Ticker": ticker,
                "Index": row.get("Index", "Unknown"),
                "Record Price": record,
                "Current Price": round(current, 2),
                "target Price": target,
                "Price 3M": round(price_3m, 2) if price_3m else None,
                "Price 6M": round(price_6m, 2) if price_6m else None,
                "Absolute Current Price (%)": round(pct, 2),
                "Absolute 3M Price (%)": round(pct_3m, 2) if pct_3m else None,
                "Absolute 6M Price (%)": round(pct_6m, 2) if pct_6m else None
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

with st.spinner("Processing your stocks..."):
    df = process_data(uploaded_file)

st.success(f"Successfully processed {len(df)} stocks!")

# ------------------- DOWNLOAD CSV -------------------
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Full Results (CSV)", csv, "BSE_Stock_Report.csv", "text/csv")

# ------------------- FILTERS -------------------
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
cutoff = datetime(1900, 1, 1)
today = datetime.today()
if period == "Last 3 Months": cutoff = today - timedelta(days=90)
elif period == "Last 6 Months": cutoff = today - timedelta(days=180)
elif period == "Last 1 Year": cutoff = today - timedelta(days=365)

filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

# ------------------- TABS -------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Performance", "Sentiment"])

with tab1:
    st.header("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Stocks", len(df))
    with col2: st.metric("Average Return", f"{df['Percent Change'].mean():+.2f}%")
    with col3:
        top = df.loc[df["Percent Change"].idxmax()]
        st.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.2f}%")

    # Pie Chart
    if "Index" in df.columns and df["Index"].nunique() > 1:
        idx_count = df["Index"].value_counts().reset_index()
        fig_pie = px.pie(idx_count, names="Index", values="count", title="Stocks by Index", hole=0.4)
        fig_pie.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=fg_color)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Performance Table
    st.subheader("Performance Table")
    display_cols = ["Company Name", "Current Price", "target Price", "Percent Change", "Distance from Target (%)"]
    styled = filtered[display_cols].style.format({
        "Current Price": "₹{:.2f}",
        "target Price": "₹{:.2f}",
        "Percent Change": "{:+.2f}%",
        "Distance from Target (%)": "{:+.2f}%"
    }).bar(subset=["Percent Change"], color=['#90EE90', '#FFB6C1'])
    st.dataframe(styled, use_container_width=True)

with tab2:
    st.header("Stock Trends & Price Tracker")
    # ADD THIS BLOCK - STOCK SELECTION IS BACK!
st.subheader("Select Stocks to View")
selected_companies = st.multiselect(
    "Choose companies for trends & tracker",
    options=df["Company Name"].unique(),
    default=df["Company Name"].head(3).tolist()  # Default 3 stocks
)

if not selected_companies:
    st.info("Select at least one company above ↑")
    st.stop()
THEN REPLACE THE
    company = st.selectbox("Select Company", df["Company Name"].unique(), key="trend_select")
    row = df[df["Company Name"] == company].iloc[0]

    # Trend Chart
    hist = yf.download(row["Ticker"], period="6mo")
    if not hist.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        ax.plot(hist.index, hist["Close"], label="Close Price", color="#00d4ff", linewidth=2.5)
        ax.axhline(row["target Price"], color="orange", linestyle="--", linewidth=2, label=f"Target ₹{row['target Price']}", alpha=0.9)

        ax.grid(True, alpha=0.3, color=line_color)
        for spine in ax.spines.values():
            spine.set_color(line_color)

        ax.set_title(f"{company} - 6 Month Trend", color=fg_color, fontsize=16, pad=20)
        ax.legend(facecolor=bg_color, edgecolor=line_color, labelcolor=fg_color)
        st.pyplot(fig)

        # Price Tracker - FIXED CONTRAST
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
        ax2.set_title(f"{company} - Price Tracker", color=fg_color, pad=20, fontsize=14)
        st.pyplot(fig2)

with tab3:
    st.header("Performance Analysis")
    st.subheader("Return Performance")
    chart_data = filtered.set_index("Company Name")["Percent Change"].sort_values(ascending=False)
    st.bar_chart(chart_data)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Gainers")
        st.dataframe(filtered.nlargest(5, "Percent Change")[["Company Name", "Percent Change", "Current Price"]], use_container_width=True)
    with col2:
        st.subheader("Top 5 Losers")
        st.dataframe(filtered.nsmallest(5, "Percent Change")[["Company Name", "Percent Change", "Current Price"]], use_container_width=True)

    st.subheader("Performance Heatmap")
    values = filtered["Absolute Current Price (%)"].fillna(0)
    companies = filtered["Company Name"]
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    cols = 6
    rows = (len(values) + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(16, rows * 1.8))
    fig.patch.set_facecolor(bg_color)

    for i, (comp, val) in enumerate(zip(companies, values)):
        row, col = divmod(i, cols)
        color = plt.get_cmap("RdYlGn")(norm(val))
        ax.add_patch(plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor="white"))
        ax.text(col + 0.5, rows - row - 0.5, f"{comp}\n{val:+.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="black")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    ax.set_title("Stock Performance Heatmap", fontsize=16, color=fg_color, pad=20)
    st.pyplot(fig)

with tab4:
    st.header("Market Sentiment")
    st.markdown("Latest news sentiment for Indian stock market")
    try:
        google_news = GNews(language='en', country='IN', max_results=8)
        news_items = google_news.get_news("BSE NSE stock market India")
        sentiments = []
        for item in news_items:
            title = item['title']
            polarity = TextBlob(title).sentiment.polarity
            label = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
            sentiments.append(polarity)
            st.write(f"• {title}")
            st.write(f"   → **{label}** ({polarity:+.2f})")
        avg = np.mean(sentiments) if sentiments else 0
        if avg > 0.1: st.success(f"Overall Sentiment: Positive ({avg:+.2f})")
        elif avg < -0.1: st.error(f"Overall Sentiment: Negative ({avg:+.2f})")
        else: st.info(f"Overall Sentiment: Neutral ({avg:+.2f})")
    except:
        st.warning("News fetch failed. Try again later.")

st.sidebar.success("CONTRAST FIXED - DARK MODE PERFECT")
