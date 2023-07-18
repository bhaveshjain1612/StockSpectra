import yfinance as yf
from datetime import date
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import time
import tqdm
from tqdm import tqdm
from datetime import datetime
import pytz
from dateutil.relativedelta import relativedelta
import streamlit as st
import subprocess
import sys
from streamlit_autorefresh import st_autorefresh

def determine_market_trend(row):
    if row['Price Change (%)'] > 0 and row['Volume Change (%)'] > 0:
        return 'Strong Uptrend'
    elif row['Price Change (%)'] < 0 and row['Volume Change (%)'] > 0:
        return 'Strong Downtrend'
    elif row['Price Change (%)'] > 0 and row['Volume Change (%)'] < 0:
        return 'Weak Uptrend'
    elif row['Price Change (%)'] < 0 and row['Volume Change (%)'] < 0:
        return 'Weak Downtrend'
    else:
        return 'No Trend'
    
def determine_risk_level(beta):
    if str(beta) == "Not Found":
        return 'Unknown'
    else:
        if float(beta) == 0:
            return 'No Risk'
        elif 0 < float(beta) < 0.75:
            return 'Low Risk'
        elif 0.75 <= float(beta) < 1.2:
            return 'Moderate Risk'
        elif float(beta) >= 1.2:
            return 'High Risk'
        else:
            return 'Unknown'
        
def determine_recommendation(row):
    ##"Latest macd","Latest macd_ema_9","macd_change_5d"
    macd = row["Latest macd"]
    signal = row["Latest macd_ema_9"]
    trend = row['macd_change_5d']
    
    if macd>0:
        if macd>=signal:
            if trend>0:
                return "Strong Buy"
            elif trend<0:
                return "Wait/ Hold"
        elif macd<signal:
            if trend<0:
                return "Strong Sell"
            elif trend>0:
                return "Wait/ Hold"
    elif macd<0:
        if macd>=signal:
            if trend>0:
                return "Buy"
            elif trend<0:
                return "Wait/ Hold"
        elif macd<signal:
            if trend<0:
                return "Sell"
            elif trend>0:
                return "Wait/ Hold"
    else:
        return 'Wait/ Hold'