import streamlit as st
import pandas as pd
import time
import subprocess
import sys
from streamlit_autorefresh import st_autorefresh
import numpy as np
import plotly.graph_objects as go
import pytz
from indepth_utils import *

st.set_page_config(page_title="Detailed Analysis", layout = "wide")

def load_data(file_path):
    return pd.read_csv(file_path)

def generate_firmo(dataframe):    
    col1, col2 = st.columns(2)
    
    col1.write("Sector")
    col1.subheader(dataframe["Sector"].values[0])
    
    col2.write("Industry")
    col2.subheader(dataframe["Industry"].values[0])
    
    # Display company details
    st.subheader("Description")
    st.write(dataframe["Description"].values[0])
    st.write("Website: "+dataframe["Website"].values[0])
    
def generate_stock(data, symbol):
    
    file_name = "backend_data/historical/"+symbol+"_hist.csv"
    
    historical_data = pd.read_csv(file_name)
    
    #intial KPIS
    col1, col2, col3 = st.columns([1,1,2])
    latest_close= historical_data[historical_data['Trading Day']==1].Close.values[0]
    second_latest_close= historical_data[historical_data['Trading Day']==2].Close.values[0]
    latest_volume= historical_data[historical_data['Trading Day']==1].Volume.values[0]
    
    
    change_abs  = round(latest_close-second_latest_close,2)
    col1.metric(label ="Close",value=round(latest_close,2), delta=str(str(change_abs)+ " ( "+ str(data['Close_change_1d'].values[0]).strip("-")+" %)"))
    col1.metric(label ="Volume",value=round(latest_volume,2))
    
    #charting and filters
    
    ma_priod_list = [5,10,15,20,25,30,40,50,75,100,150,200]
    
    historical_with_ma = generate_ma_signals(historical_data,ma_priod_list)
    historical_with_ma['date_only'] = pd.to_datetime(historical_with_ma['date_only'], format='%Y-%m-%d').astype('datetime64[ns]')
    
    #stock_market_holidays
    holiday_list = [
    "2023-01-26",#	Republic Day
    "2023-03-07",#	Holi
    "2023-03-30",#	Ram Navami
    "2023-04-04",#	Mahavir Jayanti
    "2023-04-07",#	Good Friday
    "2023-04-14",#	Dr. Baba Saheb Ambedkar Jayanti
    "2023-05-01",#	Maharashtra Day
    "2023-06-29",#	Bakri Id
    "2023-08-15",#	Independence Day
    "2023-09-19",#	Ganesh Chaturthi
    "2023-10-02",#	Mahatma Gandhi Jayanti
    "2023-10-24",#	Dussehra
    "2023-11-14",#	Diwali-Balipratipada
    "2023-11-27",#	Gurunanak Jayanti
    "2023-12-25",#	Christmas
    ]
    
    ma_durations = [5,10,15,20,25,30,40,50,75,100,150,200]
    col1, col2, col3 = st.columns(3)
         
    sma_filter = col2.multiselect('Select SMAs to show',ma_durations)
    ema_filter = col3.multiselect('Select EMAs to show',ma_durations)   
    filter_interval = col1.selectbox('Select a Time Interval', ["1 Day", "5 Days", "1 Month","3 Months", "6 Months", "1 Year", "2 Years"])
    
    if filter_interval: 
        if filter_interval == "1 Day":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(1, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(1, "D")]
            else:
                historical_sample = historical_with_ma
        elif filter_interval == "5 Days":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(5, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(5, "D")]
            else:
                historical_sample = historical_with_ma
        elif filter_interval == "1 Month":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(31, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(1, "M")]
            else:
                historical_sample = historical_with_ma
        elif filter_interval == "3 Months":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(93, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(3, "M")]
            else:
                historical_sample = historical_with_ma
        elif filter_interval == "6 Months":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(183, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(6, "M")]
            else:
                historical_sample = historical_with_ma
        elif filter_interval == "1 Year":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(366, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(1, "Y")]
            else:
                historical_sample = historical_with_ma
        elif filter_interval == "2 Years":
            if historical_with_ma.date_only.values[-1] <= historical_with_ma.date_only.values[0] - np.timedelta64(736, "D"):
                historical_sample = historical_with_ma[historical_with_ma['date_only'] >historical_with_ma['date_only'][0]-np.timedelta64(2, "Y")]
            else:
                historical_sample = historical_with_ma
            
            
    selected_mas = []
    #creating filtering names   for sma
    for i in sma_filter:
        selected_mas.append("sma_"+str(i)+"_trace")
            
    #creating filtering names   for ema   :                         
    for i in ema_filter:
        selected_mas.append("ema_"+str(i)+"_trace")
    
    fig = generate_charts(historical_sample,selected_mas,holiday_list)
    
    st.plotly_chart(fig,theme="streamlit", use_container_height=True, use_container_width=True, height=800)
    

def show_all(data):
    symbol = st.sidebar.text_input("Enter stock Ticker")
    if symbol:
        if symbol.upper()[-3:] == ".NS":
            symbol = symbol.upper()
        else:
            symbol = symbol.upper()+".NS"
            
        df = data[data["Symbol"]==symbol.upper()].reset_index().drop(['index'],axis=1)
        company_name = df["Name"].values[0]
        company_symbol = df["Symbol"].values[0].split(".NS")[0]
        updated_date = data['Latest created_on'].values[0].split(" ")[0]
        updated_time = data['Latest created_on'].values[0].split(" ")[1][:5]
        
        title = company_name+" ("+company_symbol+")"
        st.title(title)
        st.write(' '.join(["Data updated on:",updated_date,updated_time]))

        tab1, tab2, tab3 = st.tabs(["About", "Stock", "Financials"])

        with tab1:
            try:
                generate_firmo(df)
            except:
                st.warning("Stock Ticker is invalid or company is no longer be traded")
            
        with tab2:
            ##try:
            generate_stock(df, company_symbol)
            ##except:
            #st.warning("Stock Ticker is invalid or company is no longer be traded")

def main():

    data = load_data("database.csv")
    #try:
    show_all(data)
    #except:
    #    st.warning("Stock Ticker is invalid or company is no longer be traded")

if __name__ == "__main__":

    main() 