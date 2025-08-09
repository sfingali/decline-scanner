# stock_decline_scanner_streamlit.py

"""
Decline & No-Rebound Scanner
- Multiple indices (S&P500, NASDAQ-100, FTSE100/250, DAX 40, CAC 40, EURO STOXX 50, Nikkei 225, ASX 200, TSX 60, NIFTY 50, Sensex 30, Hang Seng)
- Scans all tickers for decline over a window, max rebound, RSI, optional fundamentals
- Benchmarks vs SPDR sector ETFs or sector average
- Stores last scan in session_state so you can adjust filters without rescanning
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import concurrent.futures
import time
import io
import math
import ta
import re

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Decline & No-Rebound Scanner", layout="wide")
st.title("📉 Decline & No-Rebound Scanner — index-wide, ETF benchmarked")

# -----------------------
# Session state init
# -----------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None   # DataFrame of last scan results (per-ticker metrics)
if "errors" not in st.session_state:
    st.session_state.errors = []         # List of (ticker, error)
if "scan_params" not in st.session_state:
    st.session_state.scan_params = None  # Dict of last scan parameters

# -----------------------
# Utilities
# -----------------------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df.columns are simple strings (handles MultiIndex gracefully)."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                parts = [str(x) for x in col if x is not None and str(x) != "nan"]
                name = " ".join(parts).strip()
            else:
                name = str(col)
            new_cols.append(name)
        df = df.copy()
        df.columns = new_cols
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df

def first_table_with_any_columns(tables, wanted_substrings_lower):
    """Return the first table whose flattened columns contain any of the wanted substrings."""
    for t in tables:
        df = flatten_columns(t.copy())
        cols_lower = [c.lower() for c in df.columns]
        if any(any(ws in c for c in cols_lower) for ws in wanted_substrings_lower):
            return df
    return None

def ensure_suffix(ticker: str, suffix: str) -> str:
    t = str(ticker).strip()
    if t.endswith(f".{suffix}"):
        return t
    if "." in t:
        return t
    return f"{t}.{suffix}"

def hk_code_format(code: str) -> str:
    """Format HK code with zero-padding (e.g., 700 -> 0700)."""
    c = re.sub(r"[^0-9]", "", str(code))
    if len(c) >= 5:
        return c
    return c.zfill(4)

def jp_code_format(code: str) -> str:
    """Format JP code to 4-digit if numeric."""
    c = re.sub(r"[^0-9A-Za-z]", "", str(code))
    if c.isdigit() and len(c) < 4:
        c = c.zfill(4)
    return c

# -----------------------
# Index getters (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def get_sp500_tickers():
    gh_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    try:
        df = pd.read_csv(gh_url)
        df = flatten_columns(df.rename(columns={"Symbol": "Ticker"}))
        df["Ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False)
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        return df[["Ticker", "Sector"]]
    except Exception:
        wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["symbol", "ticker"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        elif "Ticker" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "GICS Sector" in df.columns:
            df = df.rename(columns={"GICS Sector": "Sector"})
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False)
        return df[["Ticker", "Sector"]]

@st.cache_data(show_spinner=False)
def get_ftse100_tickers():
    gh_url = "https://raw.githubusercontent.com/datasets/ftse-100-companies/master/data/constituents.csv"
    try:
        df = flatten_columns(pd.read_csv(gh_url))
        if "EPIC" in df.columns:
            df = df.rename(columns={"EPIC": "Ticker"})
        elif "Ticker" not in df.columns:
            raise RuntimeError("Unexpected FTSE CSV format")
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "L"))
        return df[["Ticker", "Sector"]]
    except Exception:
        wiki = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["epic", "ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "EPIC" in df.columns:
            df = df.rename(columns={"EPIC": "Ticker"})
        elif "Ticker" in df.columns:
            pass
        elif "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        else:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "L"))
        return df[["Ticker", "Sector"]]

@st.cache_data(show_spinner=False)
def get_ftse250_tickers():
    wiki = "https://en.wikipedia.org/wiki/FTSE_250_Index"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["epic", "ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "EPIC" in df.columns:
            df = df.rename(columns={"EPIC": "Ticker"})
        elif "Ticker" not in df.columns and "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        elif "Ticker" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "L"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_nasdaq100_tickers():
    try:
        wiki = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Ticker" not in df.columns:
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "Ticker"})
            else:
                df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_dax40_tickers():
    wiki = "https://en.wikipedia.org/wiki/DAX"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Ticker" not in df.columns:
            if "Ticker symbol" in df.columns:
                df = df.rename(columns={"Ticker symbol": "Ticker"})
            elif "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "Ticker"})
            else:
                df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "DE"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_cac40_tickers():
    wiki = "https://en.wikipedia.org/wiki/CAC_40"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Ticker" not in df.columns:
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "Ticker"})
            else:
                df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "PA"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_eurostoxx50_tickers():
    wiki = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Ticker" not in df.columns:
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "Ticker"})
            else:
                df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str)
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_nikkei225_tickers():
    wiki = "https://en.wikipedia.org/wiki/Nikkei_225"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["code", "ticker", "symbol"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Code" in df.columns:
            df = df.rename(columns={"Code": "Ticker"})
        elif "Ticker" not in df.columns:
            if "Symbol" in df.columns:
                df = df.rename(columns={"Symbol": "Ticker"})
            else:
                df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower() or "industry" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda x: ensure_suffix(jp_code_format(x), "T"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_asx200_tickers():
    wiki = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["code", "ticker", "symbol"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        candidates = [c for c in df.columns if c.lower() in ("asx code", "code", "ticker", "symbol")]
        if candidates:
            df = df.rename(columns={candidates[0]: "Ticker"})
        else:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "AX"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_tsx60_tickers():
    wiki = "https://en.wikipedia.org/wiki/S%26P/TSX_60"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        elif "Ticker" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "TO"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_nifty50_tickers():
    wiki = "https://en.wikipedia.org/wiki/NIFTY_50"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        elif "Ticker" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "NS"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_sensex30_tickers():
    wiki = "https://en.wikipedia.org/wiki/BSE_SENSEX"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["ticker", "symbol", "code", "company"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        elif "Code" in df.columns:
            df = df.rename(columns={"Code": "Ticker"})
        else:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda t: ensure_suffix(t, "BO"))
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

@st.cache_data(show_spinner=False)
def get_hang_seng_tickers():
    wiki = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    try:
        tables = pd.read_html(requests.get(wiki, timeout=15).text)
        df = first_table_with_any_columns(tables, ["code", "ticker", "symbol"])
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Sector"])
        df = flatten_columns(df)
        if "Code" in df.columns:
            df = df.rename(columns={"Code": "Ticker"})
        elif "Symbol" in df.columns:
            df = df.rename(columns={"Symbol": "Ticker"})
        else:
            df = df.rename(columns={df.columns[0]: "Ticker"})
        if "Sector" not in df.columns:
            for c in df.columns:
                if "sector" in c.lower():
                    df = df.rename(columns={c: "Sector"})
                    break
        if "Sector" not in df.columns:
            df["Sector"] = np.nan
        df["Ticker"] = df["Ticker"].astype(str).apply(lambda x: f"{hk_code_format(x)}.HK")
        return df[["Ticker", "Sector"]]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Sector"])

# -----------------------
# Sector -> ETF mapping
# -----------------------
SECTOR_TO_ETF = {
    "Information Technology": "XLK",
    "Technology": "XLK",
    "Financials": "XLF",
    "Financial": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Discretionary Services": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "Communication Services": "XLC",
    "Communications": "XLC",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
}

def map_sector_to_etf(sector_name: str):
    if not isinstance(sector_name, str) or sector_name.strip() == "":
        return None
    if sector_name in SECTOR_TO_ETF:
        return SECTOR_TO_ETF[sector_name]
    for k, v in SECTOR_TO_ETF.items():
        if k.lower() in sector_name.lower() or sector_name.lower() in k.lower():
            return v
    keywords = {
        "tech": "XLK",
        "financ": "XLF",
        "health": "XLV",
        "consumer": "XLY",
        "communi": "XLC",
        "indust": "XLI",
        "energy": "XLE",
        "utilit": "XLU",
        "material": "XLB",
        "real": "XLRE",
        "estate": "XLRE",
        "staple": "XLP"
    }
    for k, v in keywords.items():
        if k in sector_name.lower():
            return v
    return None

# -----------------------
# Price download and ETF decline
# -----------------------
def try_variants_download(ticker, start, end, auto_adjust=False):
    """Try '.' and '-' variants for Yahoo tickers."""
    variants = [ticker]
    if "." in ticker:
        variants.append(ticker.replace(".", "-"))
    if "-" in ticker:
        variants.append(ticker.replace("-", "."))
    variants = list(dict.fromkeys(variants))
    for sym in variants:
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=auto_adjust)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns and df["Close"].dropna().size > 0:
                return df, sym
        except Exception:
            pass
    return None, None

@st.cache_data(show_spinner=False)
def fetch_etf_decline(etf_ticker: str, decline_days: int, auto_adjust=False):
    """Decline % for ETF over decline_days (latest vs decline_days ago)."""
    end = datetime.now().date()
    start = end - timedelta(days=int(decline_days * 1.5 + 30))
    df, _ = try_variants_download(etf_ticker, start=start, end=end + timedelta(days=1), auto_adjust=auto_adjust)
    if df is None:
        return np.nan
    close = df["Close"].astype(float).dropna()
    if len(close) <= decline_days:
        return np.nan
    prior = close.shift(decline_days).iloc[-1]
    latest = close.iloc[-1]
    if prior == 0 or pd.isna(prior):
        return np.nan
    return (latest / prior - 1) * 100

# -----------------------
# Per-ticker analysis
# -----------------------
def analyze_one(ticker: str,
                sector: str,
                decline_days: int,
                rebound_check: bool,
                rebound_lookahead_days: int,
                fetch_fundamentals: bool,
                auto_adjust_prices=False):
    """Analyze a single ticker; return dict with metrics or error dict."""
    if not isinstance(ticker, str) or ticker.strip() == "":
        return None
    end = datetime.now().date()
    start = end - timedelta(days=int(decline_days * 1.5 + 60))
    df, used_sym = try_variants_download(ticker, start=start, end=end + timedelta(days=1), auto_adjust=auto_adjust_prices)
    if df is None:
        return {"Ticker": ticker, "Error": "no price data"}
    if "Close" not in df.columns:
        return {"Ticker": ticker, "Error": "no Close column"}

    close = df["Close"].astype(float).dropna()
    if len(close) <= decline_days:
        return {"Ticker": ticker, "Error": f"not enough history ({len(close)} rows)"}
    close = pd.Series(np.asarray(close), index=close.index)

    prior_price = close.shift(decline_days).iloc[-1]
    latest_price = close.iloc[-1]
    if prior_price == 0 or pd.isna(prior_price):
        return {"Ticker": ticker, "Error": "invalid prior price"}

    decline_pct = (latest_price / prior_price - 1) * 100

    # Max rebound within the last `rebound_lookahead_days` inside the decline window
    recent = close.tail(decline_days).reset_index(drop=True)
    max_rebound_pct = 0.0
    if rebound_check and len(recent) > 1:
        L = len(recent)
        look = max(1, int(rebound_lookahead_days))
        for i in range(L):
            local_min = recent.iloc[i]
            if pd.isna(local_min) or local_min == 0:
                continue
            j_end = min(L, i + look + 1)
            local_max_after = recent.iloc[i:j_end].max()
            rebound = (local_max_after / local_min - 1) * 100
            if rebound > max_rebound_pct:
                max_rebound_pct = rebound

    # RSI
    try:
        rsi_series = ta.momentum.RSIIndicator(close).rsi()
        latest_rsi = float(rsi_series.iloc[-1])
    except Exception:
        latest_rsi = np.nan

    # Fundamentals
    pe = np.nan
    dividend_yield = np.nan
    sector_name = sector if isinstance(sector, str) and sector.strip() != "" else None
    if fetch_fundamentals:
        try:
            tk = yf.Ticker(used_sym if used_sym else ticker)
            info = tk.info or {}
            pe = info.get("trailingPE", info.get("forwardPE", np.nan))
            dy = info.get("dividendYield", None)
            if dy is not None:
                dividend_yield = dy * 100
            if not sector_name:
                sector_name = info.get("sector", None)
        except Exception:
            pass

    # Avg volume
    avg_vol = None
    if "Volume" in df.columns:
        try:
            avg_vol = int(df["Volume"].dropna().tail(decline_days).mean())
        except Exception:
            avg_vol = None

    return {
        "Ticker": ticker,
        "Used Symbol": used_sym if used_sym else ticker,
        "Sector": sector_name,
        "Decline %": round(decline_pct, 2),
        "Max Rebound %": round(max_rebound_pct, 2),
        "Latest RSI": round(latest_rsi, 2) if not pd.isna(latest_rsi) else np.nan,
        "P/E": round(pe, 2) if not (pe is None or (isinstance(pe, float) and math.isnan(pe))) else np.nan,
        "Dividend %": round(dividend_yield, 2) if not (isinstance(dividend_yield, float) and math.isnan(dividend_yield)) else np.nan,
        "Avg Volume": avg_vol,
    }

# -----------------------
# Index registry
# -----------------------
INDEX_GETTERS = {
    "S&P 500": get_sp500_tickers,
    "NASDAQ-100": get_nasdaq100_tickers,
    "FTSE 100": get_ftse100_tickers,
    "FTSE 250": get_ftse250_tickers,
    "DAX 40": get_dax40_tickers,
    "CAC 40": get_cac40_tickers,
    "EURO STOXX 50": get_eurostoxx50_tickers,
    "Nikkei 225": get_nikkei225_tickers,
    "S&P/ASX 200": get_asx200_tickers,
    "S&P/TSX 60": get_tsx60_tickers,
    "NIFTY 50": get_nifty50_tickers,
    "BSE Sensex 30": get_sensex30_tickers,
    "Hang Seng Index": get_hang_seng_tickers,
}

# -----------------------
# Sidebar: scan settings (does not auto-run)
# -----------------------
with st.sidebar:
    st.header("Scan settings")
    index_choice = st.selectbox("Index to scan", options=list(INDEX_GETTERS.keys()), index=0)
    decline_days = st.number_input("Decline period (days)", min_value=5, max_value=1825, value=90)
    rebound_check_days = st.number_input("Rebound look-ahead days (for Max Rebound %)", min_value=3, max_value=365, value=30)
    do_rebound_check = st.checkbox("Compute max rebound within decline window", value=True)
    fetch_fundamentals = st.checkbox("Fetch fundamentals (P/E, dividend) — slower", value=True)
    auto_adjust_prices = st.checkbox("Auto-adjust prices (splits/dividends)", value=False)
    use_etf_benchmark = st.selectbox("Benchmark vs:", options=["Sector ETF (SPDR XL*)", "Sector average", "None"], index=0)
    max_workers = st.slider("Parallel workers (network load)", min_value=2, max_value=20, value=8)
    st.caption("Changing settings does not auto-run. Click Rescan to update.")
    run = st.button("Run / Rescan now")

# Parameters that change scan results (benchmark choice is applied on display)
current_scan_params = {
    "index_choice": index_choice,
    "decline_days": int(decline_days),
    "rebound_check_days": int(rebound_check_days),
    "do_rebound_check": bool(do_rebound_check),
    "fetch_fundamentals": bool(fetch_fundamentals),
    "auto_adjust_prices": bool(auto_adjust_prices),
}

settings_changed = (st.session_state.scan_params is not None and
                    st.session_state.scan_params != current_scan_params)

# -----------------------
# Run scan only when requested
# -----------------------
if run:
    st.info(f"Fetching tickers for {index_choice}...")
    getter = INDEX_GETTERS.get(index_choice)
    tickers_df = getter() if getter else pd.DataFrame(columns=["Ticker", "Sector"])
    if tickers_df.empty:
        st.error("Failed to retrieve tickers for the selected index.")
    else:
        tickers_df = tickers_df.rename(columns={tickers_df.columns[0]: "Ticker"})
        if "Sector" not in tickers_df.columns:
            tickers_df["Sector"] = np.nan

        tickers = tickers_df["Ticker"].astype(str).tolist()
        sectors_for_ticker = dict(zip(tickers_df["Ticker"].astype(str).tolist(), tickers_df["Sector"].tolist()))

        st.write(f"Found {len(tickers)} tickers. Starting parallel scan...")
        progress = st.progress(0)
        status_text = st.empty()

        results = []
        errors = []
        total = len(tickers)
        completed = 0
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_ticker = {
                ex.submit(
                    analyze_one,
                    ticker,
                    sectors_for_ticker.get(ticker, None),
                    int(decline_days),
                    bool(do_rebound_check),
                    int(rebound_check_days),
                    bool(fetch_fundamentals),
                    bool(auto_adjust_prices),
                ): ticker for ticker in tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                tk = future_to_ticker[future]
                try:
                    res = future.result()
                    if res is None:
                        errors.append((tk, "no data"))
                    elif isinstance(res, dict) and res.get("Error"):
                        errors.append((tk, res.get("Error")))
                    else:
                        results.append(res)
                except Exception as e:
                    errors.append((tk, str(e)))
                completed += 1
                progress.progress(int(completed / total * 100))
                status_text.text(f"Completed {completed}/{total} — last: {tk}")

        elapsed = time.time() - start_time
        progress.empty()
        status_text.empty()
        st.success(f"Scan finished in {elapsed:.0f} seconds. Results: {len(results)} succeeded, {len(errors)} errors.")

        # Store results in session for reuse
        st.session_state.results_df = pd.DataFrame(results)
        st.session_state.errors = errors
        st.session_state.scan_params = current_scan_params

# -----------------------
# Display last scan (persisted), apply benchmark + filters without rescanning
# -----------------------
if st.session_state.results_df is None:
    st.info("No scan in memory yet. Set your scan settings and click 'Run / Rescan now'.")
else:
    if settings_changed:
        st.warning("Scan settings have changed since the last run. Filters below apply to the previous results. Click 'Run / Rescan now' to refresh the data.")

    # Work on a copy of stored results
    df_results = st.session_state.results_df.copy()

    # Compute benchmark columns based on current selection (no rescan needed)
    if use_etf_benchmark == "Sector average":
        sector_avg = df_results.groupby("Sector")["Decline %"].mean().to_dict()
        df_results["Sector ETF Decline %"] = df_results["Sector"].map(lambda s: sector_avg.get(s, np.nan))
        df_results["Sector Relative %"] = (df_results["Decline %"] - df_results["Sector ETF Decline %"]).round(2)
    elif use_etf_benchmark == "Sector ETF (SPDR XL*)":
        unique_sectors = sorted(set([s for s in df_results["Sector"].dropna().unique().tolist()]))
        etf_declines = {}
        for sec in unique_sectors:
            etf = map_sector_to_etf(sec)
            etf_declines[sec] = fetch_etf_decline(etf, int(decline_days), auto_adjust=bool(auto_adjust_prices)) if etf else np.nan
        df_results["Sector ETF Decline %"] = df_results["Sector"].map(lambda s: etf_declines.get(s, np.nan))
        df_results["Sector Relative %"] = (df_results["Decline %"] - df_results["Sector ETF Decline %"]).round(2)
    else:  # None
        df_results["Sector ETF Decline %"] = np.nan
        df_results["Sector Relative %"] = np.nan

    # Column order and sort (lower = worse vs benchmark)
    display_cols = [
        "Ticker", "Sector", "Decline %", "Max Rebound %", "Latest RSI",
        "P/E", "Dividend %", "Avg Volume", "Sector ETF Decline %", "Sector Relative %"
    ]
    for c in display_cols:
        if c not in df_results.columns:
            df_results[c] = np.nan
    df_results = df_results[display_cols].sort_values(by="Sector Relative %", ascending=True, na_position="last").reset_index(drop=True)

    # Summary and errors
    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers scanned", len(df_results))
    c2.metric("Successful", len(df_results))
    c3.metric("Errors", len(st.session_state.errors))
    if st.session_state.errors:
        with st.expander("Sample errors (click to expand)"):
            st.dataframe(pd.DataFrame(st.session_state.errors[:200], columns=["Ticker", "Error"]))

    st.subheader("All results (sorted by Sector Relative % — lower = more underperformed vs benchmark)")
    st.dataframe(df_results, height=450, use_container_width=True)

    # -----------------------
    # Filter UI (does not rescan). Uses a form with Apply button.
    # -----------------------
    st.subheader("Filter & download (applies to the results above; does not rescan)")
    default_min_decline = float(df_results["Decline %"].min()) if not df_results.empty else -10.0
    default_min_sector_rel = float(df_results["Sector Relative %"].min()) if not df_results.empty else -100.0

    with st.form("filters_form", clear_on_submit=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        min_decline = fc1.number_input("Min Decline % (enter 20 to mean ≤ -20%)", value=abs(default_min_decline), step=1.0)
        max_rebound_allow = fc2.number_input("Max allowed Max-Rebound %", value=100.0, step=1.0)
        max_rsi = fc3.number_input("Max RSI (0 disables)", value=0.0, min_value=0.0, step=1.0)
        min_sector_rel = fc4.number_input("Min Sector Relative % (allow negatives; lower = worse)", value=float(default_min_sector_rel), step=1.0)
        include_unknown_sector = st.checkbox("Include rows without sector/benchmark (NaN Sector Relative %)", value=True)
        apply_filters = st.form_submit_button("Apply filters")

    # Apply filters (interpreting positive min_decline as decline magnitude)
    filtered = df_results.copy()
    # Positive input means "down at least X%"
    threshold = -abs(min_decline)
    filtered = filtered[filtered["Decline %"] <= threshold]
    filtered = filtered[filtered["Max Rebound %"] <= max_rebound_allow]
    if max_rsi > 0:
        filtered = filtered[filtered["Latest RSI"] <= max_rsi]
    if use_etf_benchmark != "None":
        sr = filtered["Sector Relative %"]
        if include_unknown_sector:
            filtered = filtered[(sr.fillna(float("-inf")) >= min_sector_rel)]
        else:
            filtered = filtered[sr.notna() & (sr >= min_sector_rel)]

    st.write(f"Filtered results: {len(filtered)} tickers")
    st.dataframe(filtered, height=320, use_container_width=True)

    # CSV download
    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button("Download filtered CSV", csv_buf.getvalue(), "decline_scan_filtered.csv", "text/csv")

    # Chart previews
    st.subheader("Chart previews (select tickers)")
    selection = st.multiselect("Choose tickers to preview", options=filtered["Ticker"].tolist(), default=filtered["Ticker"].tolist()[:5], max_selections=20)
    if selection:
        for tk in selection:
            st.markdown(f"**{tk}**")
            try:
                dfc, used = try_variants_download(
                    tk,
                    start=datetime.now().date() - timedelta(days=max(365, int(decline_days))),
                    end=datetime.now().date() + timedelta(days=1),
                    auto_adjust=bool(auto_adjust_prices),
                )
                if dfc is None or dfc.empty:
                    st.write("No price data to show.")
                    continue
                close = dfc["Close"].astype(float).dropna()
                chart_df = pd.DataFrame({
                    "Close": close,
                    "SMA50": close.rolling(window=50, min_periods=1).mean()
                })
                st.line_chart(chart_df.tail(min(len(chart_df), max(180, int(decline_days)))))
            except Exception as e:
                st.write("Chart error:", e)
