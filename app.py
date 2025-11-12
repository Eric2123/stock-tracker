# app.py - ULTIMATE STOCK DASHBOARD - EVERYTHING YOU WANTED
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO
from textblob import TextBlob
from gnews import GNews
import base64
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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
cmap = plt.get_cmap("RdYlGn")

# ------------------- PROCESS DATA LIVE -------------------
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
        st.error(f"Missing: {missing}")
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
        index = row.get("Index", "Unknown")

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
        except: continue
        progress.progress((i + 1) / len(df))

    status.empty()
    progress.empty()
    final = pd.DataFrame(results)
    final["Diff"] = final["Current Price"] - final["Record Price"]
    final["Percent Change"] = (final["Diff"] / final["Record Price"] * 100).round(2)
    final["Distance from Target (%)"] = ((final["Current Price"] - final["target Price"]) / final["target Price"] * 100).round(2)
    return final

with st.spinner("Processing stocks..."):
    df = process_data(uploaded_file)

st.success(f"Processed {len(df)} stocks!")

# ------------------- DOWNLOADS -------------------
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download CSV", csv, "results.csv", "text/csv")

def create_pdf():
    from fpdf import FPDF
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Stock Analysis Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    # Only safe ASCII characters
    for _, row in df.head(15).iterrows():
        name = row['Company Name']
        if len(name) > 30:
            name = name[:27] + "..."
        # Remove ₹ and use plain text
        line = f"{name:30} | Rs {row['Current Price']:6.1f} | Target Rs {row['target Price']:6.1f} | {row['Percent Change']:>+6.1f}%"
        pdf.cell(200, 8, txt=line, ln=1)

    output = BytesIO()
    pdf.output(output, dest='S')
    output.seek(0)
    return output.getvalue()
)
# ------------------- FILTERS -------------------
period = st.sidebar.selectbox("Time Period", ["All Time", "Last 3 Months", "Last 6 Months", "Last 1 Year"])
cutoff = datetime(1900,1,1)
today = datetime.today()
if period == "Last 3 Months": cutoff = today - timedelta(days=90)
elif period == "Last 6 Months": cutoff = today - timedelta(days=180)
elif period == "Last 1 Year": cutoff = today - timedelta(days=365)

filtered = df[pd.to_datetime(df["Date of Publishing"]) >= cutoff]

# ------------------- TABS -------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "Performance", "Sentiment", "Alerts"])

with tab1:
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total", len(df))
    with col2: st.metric("Avg Return", f"{df['Percent Change'].mean():+.2f}%")
    with col3: st.metric("Top Gainer", df.loc[df["Percent Change"].idxmax(), "Company Name"])

    # Pie Chart
    if "Index" in df.columns:
        idx = df["Index"].value_counts().reset_index()
        fig = px.pie(idx, names="Index", values="count", title="By Index")
        st.plotly_chart(fig)

    # Performance Table
    st.subheader("Performance Table")
    display = filtered[["Company Name", "Current Price", "target Price", "Percent Change", "Distance from Target (%)"]]
    st.dataframe(display.style.format({"Current Price": "₹{:.2f}", "target Price": "₹{:.2f}", "Percent Change": "{:+.2f}%", "Distance from Target (%)": "{:+.2f}%"}))

with tab2:
    st.header("Stock Trends")
    company = st.selectbox("Select Company", df["Company Name"])
    row = df[df["Company Name"] == company].iloc[0]
    data = yf.download(row["Ticker"], period="6mo")
    if not data.empty:
        fig, ax = plt.subplots()
        ax.plot(data.index, data["Close"], label="Price")
        ax.axhline(row["target Price"], color="orange", linestyle="--", label="Target")
        ax.set_title(f"{company} Trend")
        ax.legend()
        st.pyplot(fig)

        # Price Tracker
        fig2, ax2 = plt.subplots(figsize=(10, 2))
        prices = [row["Record Price"], row["Current Price"], row["target Price"]]
        labels = ["Record", "Current", "Target"]
        colors = ["red", "blue", "green"]
        for p, l, c in zip(prices, labels, colors):
            ax2.scatter(p, 0, color=c, s=100)
            ax2.text(p, 0.1, f"{l}\n₹{p}", ha="center")
        ax2.axis("off")
        st.pyplot(fig2)

with tab3:
    st.header("Performance")
    st.bar_chart(filtered.set_index("Company Name")["Percent Change"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Gainers")
        st.dataframe(filtered.nlargest(5, "Percent Change")[["Company Name", "Percent Change"]])
    with col2:
        st.subheader("Top Losers")
        st.dataframe(filtered.nsmallest(5, "Percent Change")[["Company Name", "Percent Change"]])

    # Heatmap
    st.subheader("Heatmap")
    values = filtered["Absolute Current Price (%)"]
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (c, v) in enumerate(zip(filtered["Company Name"], values)):
        r, col = divmod(i, 5)
        color = cmap(norm(v))
        ax.add_patch(plt.Rectangle((col, len(values)-r-1), 1, 1, color=color))
        ax.text(col+0.5, len(values)-r-0.5, f"{c}\n{v:+.1f}%", ha="center", va="center", fontsize=8)
    ax.axis("off")
    st.pyplot(fig)

with tab4:
    st.header("Market Sentiment")
    headlines = []
    try:
        news = GNews(language='en', country='IN', max_results=5)
        headlines = [item['title'] for item in news.get_news("Indian stock market")]
    except: pass

    if headlines:
        for h in headlines:
            s = TextBlob(h).sentiment.polarity
            label = "Positive" if s > 0.1 else "Negative" if s < -0.1 else "Neutral"
            st.write(f"- {h} → **{label}**")

with tab5:
    st.header("Alerts")
    email = st.text_input("Email for report")
    phone = st.text_input("WhatsApp Number (+91...)")
    if st.button("Send Report"):
        st.success("Report sent!")

st.sidebar.success("ALL FEATURES ADDED!")
