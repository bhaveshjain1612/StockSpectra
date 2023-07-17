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
    
    ma_priod_list = [5,10,15,20,25,30,40,50,75,100,150,200]
    historical_with_ma = RSI(ADX(generate_ma_signals(historical_data,ma_priod_list),14),14)
    historical_with_ma['date_only'] = pd.to_datetime(historical_with_ma['date_only'], format='%Y-%m-%d').astype('datetime64[ns]')
    
    week52_data = week_52(historical_with_ma)
    dividend_split_data = dividends_splits(historical_with_ma)
    
    
    #intial KPIS
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    latest_close= historical_data[historical_data['Trading Day']==1].Close.values[0]
    second_latest_close= historical_data[historical_data['Trading Day']==2].Close.values[0]
    latest_volume= historical_data[historical_data['Trading Day']==1].Volume.values[0]
    second_latest_volume= historical_data[historical_data['Trading Day']==2].Volume.values[0]
    latest_High= historical_data[historical_data['Trading Day']==1].High.values[0]
    latest_Low= historical_data[historical_data['Trading Day']==1].Low.values[0]   
    
    change_abs_p = round(latest_close-second_latest_close,2)
    change_abs_v = round(latest_volume-second_latest_volume,2)
    
    col1.metric(label ="Close",value=round(latest_close,2), delta=str(str(change_abs_p)+ " ( "+ str(data['Close_change_1d'].values[0]).strip("-")+" %)"))
    col2.metric(label ="Volume",value=round(latest_volume,2), delta=str(str(change_abs_v)+ " ( "+ str(data['Volume_change_1d'].values[0]).strip("-")+" %)"))
    
    col1.metric(label ="High",value=round(latest_High,2), delta='') 
    col2.metric(label ="Low",value=round(latest_Low,2), delta='') 
    
    col3.metric(label ="Normal Dividend",value=dividend_split_data['Normal dividend'], delta = "Financial year FY23")
    col4.metric(label ="Stock Split",value=dividend_split_data['split ratio'], delta = dividend_split_data["split date"])
    col3.metric(label ="52 Week High",value=round(week52_data['52 Week High'],2), delta='') 
    col4.metric(label ="52 Week Low",value=round(week52_data['52 Week Low'],2), delta='') 
    
    st.divider()
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
    col1, col2, col3, col4 = st.columns([2,2,3,3])
         
    sma_filter = col3.multiselect('Select SMAs to show',ma_durations)
    ema_filter = col4.multiselect('Select EMAs to show',ma_durations)   
    filter_interval = col1.selectbox('Select a Time Interval', ["6 Months","1 Day", "5 Days", "1 Month","3 Months",  "1 Year", "2 Years"])
    plot_filter = col2.selectbox('Select Bottom Plot',['Volume','MACD','ADX','RSI'])
       
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
    #creating filtering names for sma
    for i in sma_filter:
        selected_mas.append("sma_"+str(i)+"_trace")
            
    #creating filtering names for ema   :                         
    for i in ema_filter:
        selected_mas.append("ema_"+str(i)+"_trace")
    
    fig = generate_charts(historical_sample,selected_mas,holiday_list, plot_filter)
        
    st.plotly_chart(fig,theme="streamlit", use_container_height=True, use_container_width=True, height=1000)
    
def generate_financials(symbol):
    src = pd.read_csv("backend_data/company_financials/"+symbol+"_financials.csv")
    income_stmt = src[src['sheet']=='Income Statement'].drop("Unnamed: 0",axis=1).set_index("item").T.reset_index().rename(columns={"index" : "year"})
    balance_sheet = src[src['sheet']=='Balance Sheet'].drop("Unnamed: 0",axis=1).set_index("item").T.reset_index().rename(columns={"index" : "year"})
    cash_flow = src[src['sheet']=='Cash Flow'].drop("Unnamed: 0",axis=1).set_index("item").T.reset_index().rename(columns={"index" : "year"})
    
    income_stmt['reporting_date'] = income_stmt['year'].astype('str').str.strip(" 00:00:00")
    income_stmt_to_display = income_stmt.set_index('reporting_date').drop('year',axis=1).T.drop('sheet',axis=1)

    balance_sheet['reporting_date'] = balance_sheet['year'].astype('str').str.strip(" 00:00:00")
    balance_sheet_to_display = balance_sheet.set_index('reporting_date').drop('year',axis=1).T.drop('sheet',axis=1)

    cash_flow['reporting_date'] = cash_flow['year'].astype('str').str.strip(" 00:00:00")
    cash_flow_to_display = cash_flow.set_index('reporting_date').drop('year',axis=1).T.drop('sheet',axis=1)
    
    with st.expander("Income Statement"):
        st.dataframe(income_stmt_to_display)
        
    with st.expander("Balance Sheet"):
        st.dataframe(balance_sheet_to_display)
        
    with st.expander("Cash Flow"):
        st.dataframe(cash_flow_to_display)  

def show_all(data,symbol):

    df = data[data["Symbol"]==symbol.upper()].reset_index().drop(['index'],axis=1)
    company_name = df["Name"].values[0]
    company_symbol = df["Symbol"].values[0].split(".NS")[0]
    updated_date = df['Latest created_on'].values[0].split(" ")[0]
    updated_time = df['Latest created_on'].values[0].split(" ")[1][:5]

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
        #try:
        generate_stock(df, company_symbol)
        #except:
            #st.warning("Stock Ticker is invalid or company is no longer be traded")
            
    with tab3:
        #try:
        generate_financials(company_symbol)
        #except:
            #st.warning("Stock Ticker is invalid or company is no longer be traded")
            
def main():

    data = load_data("backend_data/database.csv")
    #st.experimental_get_query_params()['symbol']
    
    symbol = st.sidebar.text_input("Enter stock Ticker")
    if symbol:
        if symbol.upper()[-3:] == ".NS":
            symbol = symbol.upper()
        else:
            symbol = symbol.upper()+".NS"
        show_all(data,symbol)
        st.experimental_set_query_params(symbol=symbol[:-3])

    elif st.experimental_get_query_params() != {}:
        
        show_all(data,st.experimental_get_query_params()['symbol'][0].upper()+".NS")
        st.experimental_set_query_params(symbol=st.experimental_get_query_params()['symbol'][0].upper())
        
    else:
        st.write("Enter the stock symbol/ticker in the text box in the sidebar")
    #try:
    #show_all(data)
    #except:
    #    st.warning("Stock Ticker is invalid or company is no longer be traded")

if __name__ == "__main__":

    main() 
