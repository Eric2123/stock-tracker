# app.py - FINAL WORKING VERSION (Tested LIVE on Streamlit Cloud)
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="BSE Stock Tracker", layout="wide")
st.title("BSE Stock Tracker Pro")

# -----------------------------
# Sidebar Upload
# -----------------------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload `pythonmaster.xlsx`", type="xlsx")

if not uploaded_file:
    st.warning("Please upload your `pythonmaster.xlsx` file to continue.")
    st.stop()

# -----------------------------
# Process Data
# -----------------------------
@st.cache_data
def process_data(file):
    file.seek(0)
    df = pd.read_excel(file, engine="openpyxl")

    # FIX: Ensure exact column names
    df.columns = df.columns.str.strip()
    required_cols = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # Convert date safely
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date of Publishing", "Record Price", "Target Price"])

    today = datetime.today()
    three_m_ago = today - timedelta(days=90)
    six_m_ago = today - timedelta(days=180)

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, row in df.iterrows():
        company = str(row["Company Name"]).strip()
        ticker_raw = str(row["Ticker"]).strip()
        ticker = ticker_raw + ".BO" if not ticker_raw.endswith((".BO", ".NS")) else ticker_raw
        record = float(row["Record Price"])
        target = float(row["Target Price"])
        pub_date = row["Date of Publishing"]

        status.text(f"Fetching {company} ({ticker})...")
        
        try:
            data = yf.Ticker(ticker).history(period="1y")
            if data.empty or "Close" not in data.columns:
                continue
            data.index = data.index.tz_localize(None)

            current = data["Close"].iloc[-1]
            price_3m = data[data.index >= three_m_ago]["Close"].iloc[0] if not data[data.index >= three_m_ago].empty else None
            price_6m = data[data.index >= six_m_ago]["Close"].iloc[0] if not data[data.index >= six_m_ago].empty else None

            pct_current = round(((current - record) / record) * 100, 2)
            pct_3m = round(((price_3m - record) / record) * 100, 2) if price_3m else None
            pct_6m = round(((price_6m - record) / record) * 100, 2) if price_6m else None

            results.append({
                "Date of Publishing": pub_date.date(),
                "Company Name": company,
                "Ticker": ticker,
                "Record Price": round(record, 2),
                "Current Price": round(current, 2),
                "Target Price": round(target, 2),
                "Price 3M": round(price_3m, 2) if price_3m else None,
                "Price 6M": round(price_6m, 2) if price_6m else None,
                "Current %": pct_current,
                "3M %": pct_3m,
                "6M %": pct_6m
            })
        except:
            pass

        progress.progress((i + 1) / len(df))

    status.empty()
    progress.empty()
    return pd.DataFrame(results)

# Run
with st.spinner("Analyzing stocks..."):
    final_df = process_data(uploaded_file)

if final_df.empty:
    st.error("No valid data found. Check your Excel file.")
    st.stop()

st.success(f"Processed {len(final_df)} stocks!")

# Download
csv = final_df.to_csv(index=False).encode()
st.sidebar.download_button("Download Results", csv, "BSE_Results.csv", "text/csv")

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filters")
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
today = datetime.today()

if period == "Last 3 Months":
    cutoff = today - timedelta(days=90)
elif period == "Last 6 Months":
    cutoff = today - timedelta(days=180)
elif period == "Last 1 Year":
    cutoff = today - timedelta(days=365)
else:
    cutoff = pd.to_datetime("1900-01-01")

# FIX: Safe date filtering
final_df["Date of Publishing"] = pd.to_datetime(final_df["Date of Publishing"])
filtered_df = final_df[final_df["Date of Publishing"] >= cutoff]

# -----------------------------
# Dashboard
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Stocks", len(final_df))
with c2:
    avg = filtered_df["Current %"].mean()
    st.metric("Avg Return", f"{avg:+.1f}%" if not pd.isna(avg) else "N/A")
with c3:
    top = filtered_df.loc[filtered_df["Current %"].idxmax()]
    st.metric("Top Gainer", top["Company Name"], f"{top['Current %']:+.1f}%")

# Heatmap
st.subheader(f"Performance Heatmap - {period}")
vals = filtered_df["Current %"].fillna(0)
norm = mcolors.TwoSlopeNorm(vmin=vals.min(), vcenter=0, vmax=vals.max())
cmap = plt.get_cmap("RdYlGn")
n = len(vals)
cols = 6
rows = (n // cols) + 1

fig, ax = plt.subplots(figsize=(14, rows * 1.2))
for i, (company, val) in enumerate(zip(filtered_df["Company Name"], vals)):
    r, c = divmod(i, cols)
    color = cmap(norm(val))
    ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, color=color, ec="black"))
    ax.text(c + 0.5, rows - r - 0.5, f"{company}\n{val:+.1f}%", ha="center", va="center", fontsize=8, weight="bold")

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.axis("off")
st.pyplot(fig)

# Table
st.subheader("Top 10 Performers")
top10 = filtered_df.nlargest(10, "Current %")[["Company Name", "Current %", "Current Price", "Target Price"]]
st.dataframe(top10.style.format({"Current %": "{:+.2f}%"}), use_container_width=True)

# Raw data
if st.checkbox("Show Full Data"):
    st.dataframe(final_df)
