# ==================== IMPORTS ====================
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
import time
from sklearn.linear_model import LinearRegression

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

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.DataFrame(STOCK_MASTER)
    df["Date of Publishing"] = pd.to_datetime(df["Date of Publishing"], dayfirst=True)
    return df

df = load_data()

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="QualSCORE", page_icon="📈", layout="wide")

st.title("🚀 QualSCORE – Stock Tracker (Internal)")
st.success(f"✅ Loaded {len(df)} stocks (Hard-coded Master)")

# ==================== PROCESS DATA ====================
@st.cache_data
def process_data(df):
    results = []
    for _, row in df.iterrows():
        try:
            hist = yf.Ticker(row["Ticker"]).history(period="1y")
            if hist.empty:
                continue
            current = hist["Close"].iloc[-1]
            returns = hist["Close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100

            nifty = yf.Ticker("^NSEI").history(period="1y")["Close"].pct_change().dropna()
            common = returns.index.intersection(nifty.index)
            beta = 1
            if len(common) > 5:
                model = LinearRegression().fit(nifty.loc[common].values.reshape(-1,1), returns.loc[common].values)
                beta = model.coef_[0]

            results.append({
                "Company Name": row["Company Name"],
                "Ticker": row["Ticker"],
                "Index": row["Index"],
                "Record Price": row["Record Price"],
                "Target Price": row["Target Price"],
                "Current Price": round(current,2),
                "Percent Change": round((current-row["Record Price"])/row["Record Price"]*100,2),
                "Volatility (%)": round(volatility,2),
                "Beta": round(beta,2),
                "Date of Publishing": row["Date of Publishing"].date()
            })
        except:
            pass
    return pd.DataFrame(results)

final_df = process_data(df)

# ==================== DASHBOARD ====================
st.subheader("📊 Overview")
col1,col2,col3 = st.columns(3)
col1.metric("Total Stocks", len(final_df))
col2.metric("Avg Return", f"{final_df['Percent Change'].mean():+.2f}%")
top = final_df.loc[final_df["Percent Change"].idxmax()]
col3.metric("Top Gainer", top["Company Name"], f"{top['Percent Change']:+.2f}%")

st.subheader("📋 Performance Table")
st.dataframe(final_df, use_container_width=True)

st.subheader("📈 Returns Bar Chart")
st.bar_chart(final_df.set_index("Company Name")["Percent Change"])

st.caption(f"© QualSCORE | Last updated {datetime.now().strftime('%d-%m-%Y')}")
