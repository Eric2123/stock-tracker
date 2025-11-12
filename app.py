# app.py
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

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Stock Tracker Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main { padding: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
    color: #333;
}
.stTabs [aria-selected="true"] {
    background-color: #1e88e5;
    color: white;
}
.sidebar .sidebar-content { background-color: #f8f9fa; }
.stButton>button {
    background-color: #1e88e5;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
}
.stButton>button:hover { background-color: #1565c0; }
.stDataFrame { border: 1px solid #ddd; border-radius: 8px; }
h1, h2, h3 { font-family: 'Arial', sans-serif; }
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# File Upload (Replaces Colab upload)
# -----------------------------
st.sidebar.header("Upload Your Excel File")
uploaded_file = st.sidebar.file_uploader("Upload pythonmaster.xlsx", type=["xlsx"])

if not uploaded_file:
    st.error("Please upload 'pythonmaster.xlsx' to continue.")
    st.stop()

# Load data
@st.cache_data
def load_and_process_data(uploaded_file):
    uploaded_file.seek(0)  # CRITICAL: Reset file pointer
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Convert Date of Publishing
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True)

    # Time periods
    today = datetime.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)

    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df.iterrows():
        company = row["Company Name"]
        ticker = str(row["Ticker"]).strip()
        index = row.get("Index", "N/A")
        record_price = row["Record Price"]
        target_price = row["Target Price"]
        pub_date = row["Date of Publishing"]

        status_text.text(f"Processing {company} ({ticker})...")
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty or "Close" not in hist:
                continue

            hist.index = hist.index.tz_localize(None)

            current_price = hist["Close"].iloc[-1]
            hist_3m = hist[hist.index >= three_months_ago]
            price_3m = hist_3m["Close"].iloc[0] if not hist_3m.empty else None
            hist_6m = hist[hist.index >= six_months_ago]
            price_6m = hist_6m["Close"].iloc[0] if not hist_6m.empty else None

            irr_current = ((current_price - record_price) / record_price) * 100 if current_price else None
            irr_3m = ((price_3m - record_price) / record_price) * 100 if price_3m else None
            irr_6m = ((price_6m - record_price) / record_price) * 100 if price_6m else None

            results.append({
                "Date of Publishing": pub_date.date(),
                "Company Name": company,
                "Ticker": ticker,
                "Index": index,
                "Record Price": record_price,
                "Current Price": round(current_price, 2),
                "target Price": target_price,
                "Price 3M": round(price_3m, 2) if price_3m else None,
                "Price 6M": round(price_6m, 2) if price_6m else None,
                "Absolute Current Price (%)": round(irr_current, 2) if irr_current else None,
                "Absolute 3M Price (%)": round(irr_3m, 2) if irr_3m else None,
                "Absolute 6M Price (%)": round(irr_6m, 2) if irr_6m else None
            })
        except Exception as e:
            continue

        progress_bar.progress((idx + 1) / len(df))

    status_text.text("Processing Complete!")
    final_df = pd.DataFrame(results)
    progress_bar.empty()
    status_text.empty()
    return final_df

# Process data
with st.spinner("Analyzing stocks... This may take 1-2 minutes."):
    final_df = load_and_process_data(uploaded_file)

st.success("Data processed successfully!")

# Save output for download
output_csv = final_df.to_csv(index=False).encode()

# Download button
st.sidebar.download_button(
    label="Download Results as CSV",
    data=output_csv,
    file_name="BSE_Final_Output.csv",
    mime="text/csv"
)

# -----------------------------
# Theme & Filters
# -----------------------------
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
bg_color = "#1a1a1a" if theme_choice == "Dark" else "white"
fg_color = "white" if theme_choice == "Dark" else "black"

today = pd.to_datetime("today")
time_period = st.sidebar.selectbox("Select Time Period", ["Last 3 Months", "Last 6 Months", "Last 1 Year", "All Time"])

if time_period == "Last 3 Months":
    start_date = today - timedelta(days=90)
elif time_period == "Last 6 Months":
    start_date = today - timedelta(days=180)
elif time_period == "Last 1 Year":
    start_date = today - timedelta(days=365)
else:
    start_date = final_df["Date of Publishing"].min()

filtered_time_df = final_df[pd.to_datetime(final_df["Date of Publishing"]) >= start_date]

selected_companies = st.sidebar.multiselect(
    "Select Companies",
    options=final_df["Company Name"].dropna().unique(),
    default=final_df["Company Name"].dropna().unique()[:3]
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Heatmaps", "Performance"])

with tab1:
    st.header("Stock Performance Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stocks", len(final_df))
    with col2:
        avg_change = final_df["Absolute Current Price (%)"].mean()
        st.metric("Avg Change", f"{avg_change:.2f}%" if pd.notna(avg_change) else "N/A")
    with col3:
        top = final_df.loc[final_df["Absolute Current Price (%)"].idxmax()]
        st.metric("Top Gainer", top["Company Name"] if 'top' in locals() else "N/A")

    # Heatmap 1: Current vs Record
    st.subheader("Heatmap: Record vs Current Price")
    values = final_df["Absolute Current Price (%)"].fillna(0).tolist()
    companies = final_df["Company Name"].tolist()
    norm = mcolors.TwoSlopeNorm(vmin=min(values), vcenter=0, vmax=max(values))
    cmap = plt.get_cmap("RdYlGn")
    cols = 6
    rows = int(np.ceil(len(companies) / cols))
    fig, ax = plt.subplots(figsize=(16, rows * 1.8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    for i, (company, val) in enumerate(zip(companies, values)):
        row, col = divmod(i, cols)
        color = cmap(norm(val))
        rect = plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor="white")
        ax.add_patch(rect)
        ax.text(col + 0.5, rows - row - 0.5, f"{company}\n{val:.1f}%", ha="center", va="center", fontsize=9, color="black")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")
    plt.title("Stock Performance (Record → Current)", color=fg_color, fontsize=16)
    st.pyplot(fig)

with tab2:
    st.header("3M & 6M Performance Heatmaps")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Last 3 Months")
        values = final_df["Absolute 3M Price (%)"].fillna(0).tolist()
        norm = mcolors.TwoSlopeNorm(vmin=min(values), vcenter=0, vmax=max(values))
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        for i, (company, val) in enumerate(zip(companies, values)):
            row, col = divmod(i, cols)
            color = cmap(norm(val))
            rect = plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor="white")
            ax.add_patch(rect)
            ax.text(col + 0.5, rows - row - 0.5, f"{company}\n{val:.1f}%", ha="center", va="center", fontsize=8)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.axis("off")
        plt.title("3M Performance", color=fg_color)
        st.pyplot(fig)

    with col2:
        st.subheader("Last 6 Months")
        values = final_df["Absolute 6M Price (%)"].fillna(0).tolist()
        norm = mcolors.TwoSlopeNorm(vmin=min(values), vcenter=0, vmax=max(values))
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        for i, (company, val) in enumerate(zip(companies, values)):
            row, col = divmod(i, cols)
            color = cmap(norm(val))
            rect = plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor="white")
            ax.add_patch(rect)
            ax.text(col + 0.5, rows - row - 0.5, f"{company}\n{val:.1f}%", ha="center", va="center", fontsize=8)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.axis("off")
        plt.title("6M Performance", color=fg_color)
        st.pyplot(fig)

with tab3:
    st.header("Top Gainers & Losers")

    # Top Gainers
    gainers = final_df.nlargest(10, "Absolute Current Price (%)")[["Company Name", "Absolute Current Price (%)"]]
    losers = final_df.nsmallest(10, "Absolute Current Price (%)")[["Company Name", "Absolute Current Price (%)"]]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Gainers")
        st.dataframe(gainers.style.format({"Absolute Current Price (%)": "{:.2f}%"}), use_container_width=True)
    with col2:
        st.subheader("Top 10 Losers")
        st.dataframe(losers.style.format({"Absolute Current Price (%)": "{:.2f}%"}), use_container_width=True)

    # Price Tracker Charts
    st.subheader("Price Tracker (Record → Current → Target)")
    os.makedirs("charts", exist_ok=True)
    for _, row in final_df.iterrows():
        company = row["Company Name"]
        record = row["Record Price"]
        current = row["Current Price"]
        target = row["target Price"]
        if pd.isna(company): continue
        prices = [record, current, target]
        labels = ["Record", "Current", "Target"]
        colors = ["red", "black", "green"]
        valid_prices = [p for p in prices if pd.notna(p)]
        if len(valid_prices) < 2: continue
        min_p, max_p = min(valid_prices), max(valid_prices)
        fig, ax = plt.subplots(figsize=(8, 2))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.hlines(0, min_p - 5, max_p + 5, color=fg_color)
        for p, label, color in zip(prices, labels, colors):
            if pd.notna(p):
                ax.scatter(p, 0, color=color, s=100)
                ax.text(p, 0, f" {label}\n ₹{p:.0f}", ha="center", va="bottom", fontsize=9, color=color)
        ax.set_xlim(min_p - 10, max_p + 10)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.set_title(company, color=fg_color, fontsize=10)
        st.pyplot(fig)
        plt.close(fig)

# Final Data Table
st.sidebar.markdown("---")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Processed Data")
    st.dataframe(final_df, use_container_width=True)
