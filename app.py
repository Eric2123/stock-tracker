# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime, timedelta
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Stock Tracker Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main { padding: 1rem; }
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
    text-align: center;
}
.stButton>button {
    background-color: #1e88e5;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar: File Upload
# -----------------------------
st.sidebar.header("Step 1: Upload Data")
st.sidebar.markdown("**Required:** `pythonmaster.xlsx`")
uploaded_file = st.sidebar.file_uploader(
    "Drag & drop pythonmaster.xlsx here",
    type=["xlsx"],
    help="Must contain columns: Company Name, Ticker, Record Price, Target Price, Date of Publishing"
)

if not uploaded_file:
    st.error("Please upload `pythonmaster.xlsx` to start.")
    st.stop()

# -----------------------------
# Load & Process Data
# -----------------------------
@st.cache_data
def load_and_process_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")

    # REQUIRED COLUMNS CHECK
    required = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns in Excel: {missing}")
        st.stop()

    # Fix date
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date of Publishing"])

    today = datetime.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)

    results = []
    progress = st.progress(0)
    status = st.empty()

    for idx, row in df.iterrows():
        company = row["Company Name"]
        ticker = str(row["Ticker"]).strip() + ".BO"  # BSE suffix
        record_price = float(row["Record Price"])
        target_price = float(row["Target Price"])
        pub_date = row["Date of Publishing"]

        status.text(f"Fetching: {company} ({ticker})")

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty or "Close" not in hist.columns:
                continue

            hist.index = hist.index.tz_localize(None)
            current_price = hist["Close"].iloc[-1]

            hist_3m = hist[hist.index >= three_months_ago]
            price_3m = hist_3m["Close"].iloc[0] if not hist_3m.empty else None

            hist_6m = hist[hist.index >= six_months_ago]
            price_6m = hist_6m["Close"].iloc[0] if not hist_6m.empty else None

            irr_current = ((current_price - record_price) / record_price) * 100
            irr_3m = ((price_3m - record_price) / record_price) * 100 if price_3m else None
            irr_6m = ((price_6m - record_price) / record_price) * 100 if price_6m else None

            results.append({
                "Date of Publishing": pub_date.date(),
                "Company Name": company,
                "Ticker": ticker,
                "Record Price": round(record_price, 2),
                "Current Price": round(current_price, 2),
                "Target Price": round(target_price, 2),
                "Price 3M": round(price_3m, 2) if price_3m else None,
                "Price 6M": round(price_6m, 2) if price_6m else None,
                "Current %": round(irr_current, 2),
                "3M %": round(irr_3m, 2) if irr_3m else None,
                "6M %": round(irr_6m, 2) if irr_6m else None
            })
        except:
            continue

        progress.progress((idx + 1) / len(df))

    status.text("Done!")
    progress.empty()
    return pd.DataFrame(results)

# Run processing
with st.spinner("Processing all stocks... (30-60 seconds)"):
    final_df = load_and_process_data(uploaded_file)

st.success(f"Processed {len(final_df)} stocks successfully!")

# Download button
csv = final_df.to_csv(index=False).encode()
st.sidebar.download_button(
    "Download Full Results (CSV)",
    csv,
    "BSE_Stock_Results.csv",
    "text/csv"
)

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filters")
time_filter = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])

today = datetime.today()
if time_filter == "Last 3 Months":
    cutoff = today - timedelta(days=90)
elif time_filter == "Last 6 Months":
    cutoff = today - timedelta(days=180)
elif time_filter == "Last 1 Year":
    cutoff = today - timedelta(days=365)
else:
    cutoff = final_df["Date of Publishing"].min()

filtered_df = final_df[pd.to_datetime(final_df["Date of Publishing"]) >= cutoff]

# -----------------------------
# Dashboard
# -----------------------------
st.title("Stock Tracker Pro - BSE Edition")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Total Stocks", len(final_df))
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    avg = final_df["Current %"].mean()
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Avg Return", f"{avg:+.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    best = final_df.loc[final_df["Current %"].idxmax()]
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Top Gainer", best["Company Name"], f"{best['Current %']:+.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    worst = final_df.loc[final_df["Current %"].idxmin()]
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Top Loser", worst["Company Name"], f"{worst['Current %']:+.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# Heatmap
st.subheader(f"Performance Heatmap - {time_filter}")
values = filtered_df["Current %"].fillna(0)
companies = filtered_df["Company Name"]

norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
cmap = plt.get_cmap("RdYlGn")
cols = 6
rows = int(np.ceil(len(companies) / cols))

fig, ax = plt.subplots(figsize=(16, rows * 1.5))
for i, (company, val) in enumerate(zip(companies, values)):
    row, col = divmod(i, cols)
    color = cmap(norm(val))
    rect = plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor="black")
    ax.add_patch(rect)
    ax.text(col + 0.5, rows - row - 0.5, f"{company}\n{val:+.1f}%",
            ha="center", va="center", fontsize=9, color="black", weight="bold")

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.axis("off")
plt.title(f"Stock Returns (%) - {time_filter}", fontsize=16, pad=20)
st.pyplot(fig)

# Top 10 Table
st.subheader("Top 10 Performers")
top10 = filtered_df.nlargest(10, "Current %")[["Company Name", "Current %", "Current Price", "Target Price"]]
st.dataframe(top10.style.format({"Current %": "{:+.2f}%", "Current Price": "₹{:.2f}", "Target Price": "₹{:.2f}"}),
             use_container_width=True)

# Raw data
if st.checkbox("Show Raw Data Table"):
    st.dataframe(final_df, use_container_width=True)
