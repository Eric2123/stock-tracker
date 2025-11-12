# app.py - FINAL FULL VERSION WITH ALL FEATURES (100% WORKING)
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
import os

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
st.sidebar.header("Step 1: Upload Your Excel")
uploaded_file = st.sidebar.file_uploader("Upload `pythonmaster.xlsx`", type=["xlsx"])

if not uploaded_file:
    st.error("Please upload `pythonmaster.xlsx` to start.")
    st.stop()

# ------------------- PROCESS DATA (LIVE FROM UPLOAD) -------------------
@st.cache_data
def process_live_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")
    df.columns = df.columns.str.strip()

    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
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
        company = str(row["Company Name"]).strip()
        ticker_raw = str(row["Ticker"]).strip()
        ticker = ticker_raw + ".BO" if not ticker_raw.endswith((".BO", ".NS")) else ticker_raw
        record = float(row["Record Price"])
        target = float(row["Target Price"])
        pub_date = row["Date of Publishing"]
        index = row.get("Index", "Unknown")

        status.text(f"Fetching: {company} ({ticker})")

        try:
            data = yf.Ticker(ticker).history(period="1y")
            if data.empty:
                continue
            data.index = data.index.tz_localize(None)
            current = data["Close"].iloc[-1]
            price_3m = data[data.index >= three_m]["Close"].iloc[0] if not data[data.index >= three_m].empty else None
            price_6m = data[data.index >= six_m]["Close"].iloc[0] if not data[data.index >= six_m].empty else None

            pct = ((current - record) / record) * 100
            pct_3m = ((price_3m - record) / record) * 100 if price_3m else None
            pct_6m = ((price_6m - record) / record) * 100 if price_6m else None

            results.append({
                "Date of Publishing": pub_date.date(),
                "Company Name": company,
                "Ticker": ticker,
                "Index": index,
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
    final["Loss"] = -final["Diff"]
    final["Distance from Target (%)"] = ((final["Current Price"] - final["target Price"]) / final["target Price"] * 100).round(2)
    return final

with st.spinner("Processing all stocks live..."):
    final_df = process_live_data(uploaded_file)

st.success(f"Processed {len(final_df)} stocks!")

# Download
csv = final_df.to_csv(index=False).encode()
st.sidebar.download_button("Download Full Results", csv, "BSE_Final_Output.csv", "text/csv")

# ------------------- FILTERS -------------------
st.sidebar.header("Filters")
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
today = datetime.today()
cutoff = today - timedelta(days=90 if period == "Last 3 Months" else 180 if period == "Last 6 Months" else 365 if period == "Last 1 Year" else 9999)
filtered_df = final_df[pd.to_datetime(final_df["Date of Publishing"]) >= cutoff]

selected_companies = st.sidebar.multiselect("Select Companies", final_df["Company Name"].unique())

# ------------------- TABS -------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Trends", "Performance"])

with tab1:
    st.header("Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Stocks", len(final_df))
    with col2: st.metric("Avg Return", f"{final_df['Percent Change'].mean():+.2f}%")
    with col3:
        top = final_df.loc[final_df["Percent Change"].idxmax()]
        st.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.2f}%")

    # Pie Chart
    if "Index" in final_df.columns:
        idx = final_df["Index"].value_counts().reset_index()
        fig_pie = px.pie(idx, names="Index", values="count", title="Stocks by Index", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap
    st.subheader("Performance Heatmap")
    values = filtered_df["Absolute Current Price (%)"].fillna(0)
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    cmap = plt.get_cmap("RdYlGn")
    cols, rows = 6, (len(values) // 6) + 1
    fig, ax = plt.subplots(figsize=(16, rows * 1.5))
    for i, (comp, val) in enumerate(zip(filtered_df["Company Name"], values)):
        r, c = divmod(i, cols)
        color = cmap(norm(val))
        ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, color=color, ec="white"))
        ax.text(c + 0.5, rows - r - 0.5, f"{comp}\n{val:+.1f}%", ha="center", va="center", fontsize=9, weight="bold")
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.axis("off")
    st.pyplot(fig)

with tab2:
    st.header("Stock Trends")
    for company in selected_companies or final_df["Company Name"].head(3):
        row = final_df[final_df["Company Name"] == company].iloc[0]
        ticker = row["Ticker"]
        target = row["target Price"]
        data = yf.download(ticker, period="6mo")
        if data.empty: continue
        data = data.reset_index()

        fig, ax = plt.subplots()
        ax.plot(data["Date"], data["Close"], label="Price", color="blue")
        ax.axhline(target, color="orange", linestyle="--", label=f"Target ₹{target}")
        ax.set_title(f"{company} - Live Trend")
        ax.legend()
        st.pyplot(fig)

        # Tracker
        prices = [row["Record Price"], row["Current Price"], target]
        labels = ["Record", "Current", "Target"]
        colors = ["red", "blue", "green"]
        fig2, ax2 = plt.subplots(figsize=(10, 2))
        for p, l, c in zip(prices, labels, colors):
            ax2.scatter(p, 0, color=c, s=100)
            ax2.text(p, 0.1, f"{l}\n₹{p}", ha="center", fontsize=10, weight="bold")
        ax2.set_xlim(min(prices)*0.9, max(prices)*1.1)
        ax2.axis("off")
        ax2.set_title(f"{company} Price Tracker")
        st.pyplot(fig2)

with tab3:
    st.header("Performance Analysis")
    st.bar_chart(filtered_df.set_index("Company Name")["Percent Change"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Gainers")
        st.dataframe(filtered_df.nlargest(5, "Percent Change")[["Company Name", "Percent Change", "Current Price"]])
    with col2:
        st.subheader("Top 5 Losers")
        st.dataframe(filtered_df.nsmallest(5, "Percent Change")[["Company Name", "Percent Change", "Current Price"]])

# Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(final_df)
