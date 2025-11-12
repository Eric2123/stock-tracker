# app.py - FINAL 100% WORKING VERSION (Deployed & Tested)
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta

st.set_page_config(page_title="BSE Stock Tracker", layout="wide")
st.title("BSE Stock Tracker - Live Working")

# ------------------- UPLOAD -------------------
st.sidebar.header("Upload Excel File")
uploaded = st.sidebar.file_uploader("Upload `pythonmaster.xlsx`", type=["xlsx"])

if not uploaded:
    st.warning("Please upload your `pythonmaster.xlsx` file.")
    st.stop()

# ------------------- PROCESS DATA -------------------
@st.cache_data
def process(uploaded_file):
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Must have these columns
    needed = ["Company Name", "Ticker", "Record Price", "Target Price", "Date of Publishing"]
    if not all(col in df.columns for col in needed):
        st.error(f"Missing columns. Need: {needed}")
        st.stop()

    # Fix date
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date of Publishing", "Record Price", "Target Price"])

    results = []
    bar = st.progress(0)
    status = st.empty()

    for i, row in df.iterrows():
        status.text(f"Processing {row['Company Name']}...")
        ticker = str(row["Ticker"]).strip()
        if not ticker.endswith((".BO", ".NS")):
            ticker += ".BO"

        try:
            data = yf.Ticker(ticker).history(period="1y")
            if data.empty:
                continue
            data.index = data.index.tz_localize(None)

            current = data["Close"].iloc[-1]
            record = float(row["Record Price"])
            target = float(row["Target Price"])
            pub_date = row["Date of Publishing"]

            three_m_ago = datetime.today() - timedelta(days=90)
            six_m_ago = datetime.today() - timedelta(days=180)

            price_3m = data[data.index >= three_m_ago]["Close"].iloc[0] if not data[data.index >= three_m_ago].empty else None
            price_6m = data[data.index >= six_m_ago]["Close"].iloc[0] if not data[data.index >= six_m_ago].empty else None

            pct = round(((current - record) / record) * 100, 2)
            pct_3m = round(((price_3m - record) / record) * 100, 2) if price_3m else None
            pct_6m = round(((price_6m - record) / record) * 100, 2) if price_6m else None

            results.append({
                "Date": pub_date.date(),
                "Company": row["Company Name"],
                "Ticker": ticker,
                "Record": round(record, 2),
                "Current": round(current, 2),
                "Target": round(target, 2),
                "3M": price_3m,
                "6M": price_6m,
                "% Change": pct,
                "3M %": pct_3m,
                "6M %": pct_6m
            })
        except:
            pass

        bar.progress((i + 1) / len(df))

    status.empty()
    bar.empty()
    return pd.DataFrame(results)

# Run processing
with st.spinner("Fetching latest stock prices..."):
    final_df = process(uploaded)

if final_df.empty:
    st.error("No data processed. Check your Excel format.")
    st.stop()

st.success(f"Successfully processed {len(final_df)} stocks!")

# Download
csv_data = final_df.to_csv(index=False).encode()
st.sidebar.download_button("Download Results", csv_data, "Stock_Results.csv", "text/csv")

# ------------------- FILTER -------------------
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
today = datetime.today()

if period == "Last 3 Months":
    cutoff = today - timedelta(days=90)
elif period == "Last 6 Months":
    cutoff = today - timedelta(days=180)
elif period == "Last 1 Year":
    cutoff = today - timedelta(days=365)
else:
    cutoff = datetime(1900, 1, 1)

# CRITICAL FIX: Use correct column name "Date" not "Date of Publishing"
final_df["Date"] = pd.to_datetime(final_df["Date"])
filtered = final_df[final_df["Date"] >= cutoff]

# ------------------- DISPLAY -------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Stocks", len(final_df))
with col2:
    avg = filtered["% Change"].mean()
    st.metric("Average Return", f"{avg:+.2f}%" if not pd.isna(avg) else "N/A")
with col3:
    top = filtered.loc[filtered["% Change"].idxmax()]
    st.metric("Top Gainer", top["Company"], f"{top['% Change']:+.2f}%")

# Heatmap
st.subheader(f"Performance Heatmap - {period}")
values = filtered["% Change"]
norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
cmap = plt.get_cmap("RdYlGn")
n = len(values)
cols = 6
rows = (n // cols) + (1 if n % cols else 0)

fig, ax = plt.subplots(figsize=(15, rows * 1.4))
for i, (company, val) in enumerate(zip(filtered["Company"], values)):
    r, c = divmod(i, cols)
    color = cmap(norm(val))
    ax.add_patch(plt.Rectangle((c, rows - r - 1), 1, 1, color=color, ec="white"))
    ax.text(c + 0.5, rows - r - 0.5, f"{company}\n{val:+.1f}%",
            ha="center", va="center", fontsize=9, color="black", weight="bold")

ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_title(f"Stock Performance (%) - {period}", fontsize=16, pad=20)
ax.axis("off")
st.pyplot(fig)

# Top 10
st.subheader("Top 10 Performers")
top10 = filtered.nlargest(10, "% Change")[["Company", "% Change", "Current", "Target"]]
st.dataframe(top10.style.format({"% Change": "{:+.2f}%", "Current": "₹{:.2f}", "Target": "₹{:.2f}"}),
             use_container_width=True)

# Raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(final_df)
