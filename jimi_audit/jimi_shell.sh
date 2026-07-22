#!/bin/bash
cd /root/.openclaw/workspace/jimi_audit
source venv/bin/activate
ipython --no-banner -i << 'IPYTHON_EOF'
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os

DATA_DIR = "/root/.openclaw/workspace/jimi_audit/data"
DERIV_DIR = f"{DATA_DIR}/derivatives_history"
REPORT_DIR = "/root/.openclaw/workspace/jimi_audit/reports"

def load_ohlcv():
    df = pd.read_csv(f"{DATA_DIR}/eth_15m_extended.csv")
    df["timestamp"] = pd.to_datetime(df["Open time"])
    return df.sort_values("timestamp").reset_index(drop=True)

def load_deriv():
    df = pd.read_csv(f"{DERIV_DIR}/derivatives_collected.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df.sort_values("timestamp").reset_index(drop=True)

def merge(ohlcv=None, deriv=None):
    if ohlcv is None: ohlcv = load_ohlcv()
    if deriv is None: deriv = load_deriv()
    m = pd.merge_asof(ohlcv, deriv[["timestamp","oi","ls_ratio","funding_rate"]],
                       on="timestamp", direction="backward", tolerance=pd.Timedelta("2h"))
    m["oi_roc"] = m["oi"].pct_change(4, fill_method=None)
    m["fwd_ret_16"] = m["Close"].shift(-16) / m["Close"] - 1
    m["vol_20"] = m["Close"].pct_change().rolling(20).std()
    return m

print("Ready. Objects: load_ohlcv(), load_deriv(), merge()")
print("Variables: pd, np, stats, plt, DATA_DIR, DERIV_DIR, REPORT_DIR")
IPYTHON_EOF
