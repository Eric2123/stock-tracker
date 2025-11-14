# app.py - Streamlit app (cleaned from Colab notebook)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# optional plotting libs - use if available
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

# optional yfinance - if not installed / blocked, app will still work with uploaded CSV
try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Stock Tracker", layout="wide")

st.title("Stock Tracker — Upload file or use yfinance (if available)")

st.markdown(
    """
    **Upload** your Excel (or CSV) file with columns:
    - `Company Name`
    - `Ticker`
    - `Index` (optional)
    - `Record Price`
    - `Target Price`
    - `Date of Publishing` (d/m/Y or parseable)
    
    Or, if you prefer, the app will attempt to fetch prices from Yahoo Finance when `yfinance` is available.
    """
)

uploaded_file = st.file_uploader("Upload Excel or CSV file (pythonmaster.xlsx or pythonmaster.csv)", type=["xlsx", "csv"])
use_yfinance = st.checkbox("Try to fetch live prices with yfinance (only if installed)", value=False)

# helper to load uploaded file
@st.cache_data
def load_input_file(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None
    return df

df_input = load_input_file(uploaded_file)

if df_input is None:
    st.info("Upload a file to begin. You can also commit a CSV to the repo and the app can read it.")
    st.stop()

# normalize column names (common fixes)
df = df_input.copy()
df.columns = [c.strip() for c in df.columns]
# try to detect index-like column if missing
if "Index" not in df.columns:
    for col in df.columns:
        if "index" in col.lower() or "unnamed" in col.lower():
            df.rename(columns={col: "Index"}, inplace=True)
            break

# parse dates
df["Date of Publishing"] = pd.to_datetime(df.get("Date of Publishing", pd.NaT), dayfirst=True, errors="coerce")

# prepare time windows
today = datetime.today()
three_months_ago = today - timedelta(days=90)
six_months_ago = today - timedelta(days=180)

# processing button
if st.button("Run Processing"):
    st.info("Processing rows — this can take time if yfinance is enabled and many tickers are present.")
    results = []
    for _, row in st.experimental_data_editor(df, num_rows="dynamic").iterrows():
        company = row.get("Company Name")
        ticker = str(row.get("Ticker", "")).strip()
        index = row.get("Index", "")
        record_price = row.get("Record Price")
        target_price = row.get("Target Price")
        pub_date = row.get("Date of Publishing")

        # skip invalid rows
        if not company or not ticker:
            continue

        st.write(f"Processing: {company} ({ticker})")
        # default values
        current_price = None
        price_3m = None
        price_6m = None

        # try live fetch if requested and yfinance available
        if use_yfinance and yf is not None:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if not hist.empty and "Close" in hist:
                    hist.index = hist.index.tz_localize(None)
                    current_price = float(hist["Close"].iloc[-1])
                    hist_3m = hist[hist.index >= three_months_ago]
                    hist_6m = hist[hist.index >= six_months_ago]
                    price_3m = float(hist_3m["Close"].iloc[0]) if not hist_3m.empty else None
                    price_6m = float(hist_6m["Close"].iloc[0]) if not hist_6m.empty else None
                else:
                    st.write(f"Warning: no history for {ticker}")
            except Exception as e:
                st.write(f"yfinance error for {ticker}: {e}")

        # if no live data, try to read from columns if present
        # e.g., if input file has "Current Price", "Price 3M", "Price 6M"
        if current_price is None:
            if "Current Price" in df.columns and not pd.isna(row.get("Current Price")):
                current_price = row.get("Current Price")
        if price_3m is None and "Price 3M" in df.columns:
            price_3m = row.get("Price 3M")
        if price_6m is None and "Price 6M" in df.columns:
            price_6m = row.get("Price 6M")

        # Compute percentage changes safely
        try:
            irr_current = ((current_price - record_price) / record_price) * 100 if pd.notna(record_price) and pd.notna(current_price) else None
        except Exception:
            irr_current = None
        try:
            irr_3m = ((price_3m - record_price) / record_price) * 100 if pd.notna(record_price) and pd.notna(price_3m) else None
        except Exception:
            irr_3m = None
        try:
            irr_6m = ((price_6m - record_price) / record_price) * 100 if pd.notna(record_price) and pd.notna(price_6m) else None
        except Exception:
            irr_6m = None

        results.append({
            "Date of Publishing": pd.to_datetime(pub_date).date() if not pd.isna(pub_date) else None,
            "Company Name": company,
            "Ticker": ticker,
            "Index": index,
            "Record Price": record_price,
            "Current Price": round(current_price, 2) if current_price is not None else None,
            "target Price": target_price,
            "Price 3M": round(price_3m, 2) if price_3m is not None else None,
            "Price 6M": round(price_6m, 2) if price_6m is not None else None,
            "Absolute Current Price (%)": round(irr_current, 2) if irr_current is not None else None,
            "Absolute 3M Price (%)": round(irr_3m, 2) if irr_3m is not None else None,
            "Absolute 6M Price (%)": round(irr_6m, 2) if irr_6m is not None else None
        })

    if len(results) == 0:
        st.warning("No valid rows processed.")
        st.stop()

    final_df = pd.DataFrame(results)

    st.success("Processing complete — results below.")
    st.dataframe(final_df.head(50), use_container_width=True)

    # prepare CSV download
    csv_bytes = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv_bytes, file_name="BSE_Final_Output.csv", mime="text/csv")

    # show heatmap if matplotlib available
    if plt is not None:
        try:
            vals = final_df["Absolute Current Price (%)"].dropna().tolist()
            companies = final_df["Company Name"].tolist()
            if len(vals) > 0:
                norm = mcolors.TwoSlopeNorm(vmin=min(vals), vcenter=0, vmax=max(vals))
                cmap = plt.get_cmap("RdYlGn")
                cols = 7
                rows = int(np.ceil(len(companies) / cols))
                fig, ax = plt.subplots(figsize=(12, max(4, rows * 1.2)))
                for i, (company, val) in enumerate(zip(companies, final_df["Absolute Current Price (%)"])):
                    row = i // cols
                    col = i % cols
                    color = cmap(norm(val)) if pd.notna(val) else (0.8, 0.8, 0.8)
                    rect = plt.Rectangle((col, rows - row - 1), 1, 1, facecolor=color, edgecolor="black")
                    ax.add_patch(rect)
                    ax.text(col + 0.5, rows - row - 0.5,
                            f"{company}\n{'' if pd.isna(val) else f'{val:.2f}%'}",
                            ha="center", va="center", fontsize=8, wrap=True)
                ax.set_xlim(0, cols)
                ax.set_ylim(0, rows)
                ax.axis("off")
                st.pyplot(fig)
        except Exception as e:
            st.write("Could not render heatmap:", e)
    else:
        st.info("matplotlib not available — skipping heatmap.")

    st.balloons()
else:
    st.info("Click **Run Processing** after uploading your file and (optionally) enabling yfinance.")
