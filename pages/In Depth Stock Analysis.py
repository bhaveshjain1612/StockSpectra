import streamlit as st
import pandas as pd
import time
import subprocess
import sys
from streamlit_autorefresh import st_autorefresh
import numpy as np
import plotly.graph_objects as go
import pytz
from utils import *

st.set_page_config(page_title="Detailed Analysis", layout = "wide")

def load_data(file_path):
    return pd.read_csv(file_path)

#Generate FirmoGraphic Data
def generate_firmo(data):
    col1, col2 = st.columns([4,2])
    
    col1.header(data.Sector.values[0])
    col1.write("Sector")
    
    col2.header(data.Industry.values[0])
    col2.write("Industry")
    
    st.subheader("About the company")
    
    col1, col2 = st.columns([4,2])
    col1.write(data.Description.values[0])
    col2.plotly_chart(holding_chart(data),use_container_width=True,height=200)

#Generate Stock Based Insights
def generate_stock(data):
    hist_file = "backend_data/historical/"+data.Symbol.values[0].replace(".","_")+".csv"
    hist = load_data(hist_file)
    #st.write(hist_file)
    
    ma_priod_list = [5,10,15,20,25,30,40,50,75,100,150,200]
    historical_with_ma = generate_ma_signals(hist,ma_priod_list)
    historical_with_ma['date_only'] = pd.to_datetime(historical_with_ma['date_only'], format='%Y-%m-%d').astype('datetime64[ns]')
    
    week52_data = week_52(historical_with_ma)
    dividend_split_data = dividends_splits(historical_with_ma)
    
    #intial KPIS
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    latest_close= historical_with_ma[historical_with_ma['Trading Day']==1].Close.values[0]
    second_latest_close= historical_with_ma[historical_with_ma['Trading Day']==2].Close.values[0]
    latest_volume= historical_with_ma[historical_with_ma['Trading Day']==1].Volume.values[0]
    second_latest_volume= historical_with_ma[historical_with_ma['Trading Day']==2].Volume.values[0]
    latest_High= historical_with_ma[historical_with_ma['Trading Day']==1].High.values[0]
    latest_Low= historical_with_ma[historical_with_ma['Trading Day']==1].Low.values[0]   
    
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
    st.subheader("Key Takeaways")
    stksummary = stock_summary(data,historical_with_ma,week52_data,dividend_split_data)
    if len(stksummary)>1:
        col1, col2, col3 = st.columns(3)
        
        
        if len(stksummary)%2 ==1:
            for m in list(stksummary.items())[:(len(stksummary)//2)+1]:
                col2.write(m[1]['Display'])
            #col3.subheader("_")
            for m in list(stksummary.items())[(len(stksummary)//2)+1:]:
                col3.write(m[1]['Display'])
        else:
            for m in list(stksummary.items())[:(len(stksummary)//2)]:
                col2.write(m[1]['Display'])
            for m in list(stksummary.items())[(len(stksummary)//2):]:
                col3.write(m[1]['Display'])
    else:
        col1, col2 = st.columns(2)
        col2.subheader("Key Takeaways")
        for m in list(stksummary.items()):
            col2.write(m[1]['Display'])
    
    col1.header(data.Outlook.values[0].upper())
    col1.subheader(data.Risk.values[0]+" Risk / "+data.Risk.values[0]+" Reward")
    
    st.divider()
    
    st.subheader("Stock Price and technical charts")
    ##Charts
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
    
    #ma duratiosn for plotting
    ma_durations = [5,10,15,20,25,30,40,50,75,100,150,200]
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2])
         
    sma_filter = col3.multiselect('Select SMAs to show',ma_durations)
    ema_filter = col4.multiselect('Select EMAs to show',ma_durations)  
    bollinger_filter = col5.multiselect('Select Bollinger Bands to show',["1 Standard Deviation","2 Standard Deviations"])
    filter_interval = col1.selectbox('Select a Time Interval', ["6 Months","1 Day", "5 Days", "1 Month","3 Months",  "1 Year", "2 Years"])
    plot_filter = col2.selectbox('Select Bottom Plot',['Volume','MACD','ADX','RSI'])
    
    #selecting right MA
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
    
    fig = generate_charts(historical_sample,selected_mas,bollinger_filter,holiday_list, plot_filter)
        
    st.plotly_chart(fig,theme="streamlit", use_container_height=True, use_container_width=True, height=1000)
    
def generate_financials(data):
    def simplify(x):    
        if x >1000000000 or x <-1000000000:
            y = str(round(x/10000000))+" Cr"    
        elif x > 10000000 or x < -10000000:
            y = str(round(x/10000000,2))+" Cr"
        elif x > 100000 or x <-100000:
            y = str(round(x/100000,2))+" L"
        else:
            y = x
        return y
    
    fin_file = "backend_data/company_financials/"+data.Symbol.values[0].replace(".","_")+".csv"
    fin = load_data(fin_file)
    
    #st.dataframe(fin)
    
    kpis = calc_KPIs(fin.set_index('Unnamed: 0').T.reset_index(),'normal')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Net Income",simplify(kpis['Net Income']['current']), simplify(kpis['Net Income']['delta']))
    col2.metric("Debt",simplify(kpis['Debt']['current']), simplify(kpis['Debt']['delta']))
    col3.metric("Free Cash Flow",simplify(kpis['Free Cash Flow']['current']), simplify(kpis['Free Cash Flow']['delta']))
    col4.metric("Basic EPS",round(kpis['Basic EPS']['current'],2), round(kpis['Basic EPS']['delta'],2))
    col5.metric("Net Profit Margin",round(kpis['Net Profit Margin']['current'],2), round(kpis['Net Profit Margin']['delta'],2))
    st.divider()
    col1.metric("ROA",round(kpis['ROA']['current'],2), round(kpis['ROA']['delta'],2))
    col2.metric("ROE",round(kpis['ROE']['current'],2), round(kpis['ROE']['delta'],2))
    col3.metric("ROCE",round(kpis['ROCE']['current'],2), round(kpis['ROCE']['delta'],2))
    col4.metric("Current Ratio",round(kpis['Current Ratio']['current'],2), round(kpis['Current Ratio']['delta'],2))
    col5.metric("DE Ratio",round(kpis['DE Ratio']['current'],2), round(kpis['DE Ratio']['delta'],2))
    
    mapping = load_data("backend_data/financials_mapping.csv")
    #st.dataframe(fin)
    fin = pd.merge(fin.set_index('Unnamed: 0'), mapping, left_on='Unnamed: 0', right_on='index', how='inner').rename(columns={'index': 'items'})
    
    s = indicator_summary(kpis)
    
    col1, col2 = st.columns(2)
    col1.subheader('Pros')
    col2.subheader('Cons')
    pro,con= [],[]
    for i in s:
        if s[i]['Tag']=='Desirable' :
            pro.append(s[i]['To display'][0])
            col1.write(s[i]['To display'][0])
        elif s[i]['Tag']=='Non-Desirable' :
            con.append(s[i]['To display'][0])
            col2.write(s[i]['To display'][0])

    
    #display
    with st.expander("Income Statement"):
        st.dataframe(fin[fin['sheet']=='i'].drop(['Unnamed: 0','sheet'],axis=1))
        
    with st.expander("Balance Sheet"):
        st.dataframe(fin[fin['sheet']=='b'].drop(['Unnamed: 0','sheet'],axis=1))
        
    with st.expander("Cash Flow"):
        st.dataframe(fin[fin['sheet']=='c'].drop(['Unnamed: 0','sheet'],axis=1)) 
    
#Load all insights
def load_insights(data,input_symbol):
    ex_list = data['Exchange'].tolist()
    
    st.title(data.Name.values[0])  
    
    ex_sel = st.sidebar.radio("Exchange: ",ex_list)
    
    if ex_sel=="NSE":
        symbol = input_symbol.upper()+".NS"
        data = data[data['Symbol']==symbol].reset_index() 
    else:
        symbol = input_symbol.upper()+".BO"
        data = data[data['Symbol']==symbol].reset_index()
        
    st.subheader(data.Exchange.values[0]+" : " +data.Symbol.values[0][:-3])   
        
    tab1, tab2, tab3 = st.tabs(["Company Details", "Stock", "Financial"])
    
    with tab1:
        generate_firmo(data)
    with tab2:
        generate_stock(data)
    with tab3:
        generate_financials(data)
    #st.dataframe(data)

def main():
    data = load_data("backend_data/database.csv")
    data = allot_tags(data)
    
    input_symbol = st.sidebar.text_input("Enter stock Ticker")       
    
    if input_symbol:
        st.experimental_set_query_params()
        data = data[(data['Symbol'] == input_symbol.upper()+".NS") | (data['Symbol'] == input_symbol.upper()+".BO")].reset_index().drop('index',axis=1)        
        load_insights(data,input_symbol)
        
    elif st.experimental_get_query_params() != {}:
        in_s = st.experimental_get_query_params()['symbol'][0][:-3]
        data = data[(data['Symbol']==in_s.upper()+".NS")|(data['Symbol']==in_s.upper()+".BO")].reset_index().drop('index',axis=1)
        if st.experimental_get_query_params()['symbol'][0][-3:]=="_BO":
            data = data.sort_values(by="Exchange")
        else:
            data = data.sort_values(by="Exchange", ascending = False)
          
        #st.dataframe(data)
        load_insights(data,in_s)
    

if __name__ == "__main__":

    main() 
