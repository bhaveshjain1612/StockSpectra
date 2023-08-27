import streamlit as st
import pandas as pd
import time
import sys
from streamlit_autorefresh import st_autorefresh
import numpy as np
from utils import *

st.set_page_config(page_title="StockSpectra - Compare Stocks",layout = "wide")

def get_names_from_symbols(symbols, df):
    # Create a dictionary from the dataframe for faster lookup
    symbol_to_name = dict(zip(df['Symbol'], df['Name']))
    
    # Get the names corresponding to the symbols
    names = [symbol_to_name.get(symbol, '') for symbol in symbols]
    
    return names

def load_data(file_path):
    return pd.read_csv(file_path)

#Firmographics
def basics(df,lst):
    df_new = df[df['Name'].isin(lst)][['Name',
                          'Sector',
                          'Industry',
                          'Risk 1-2Months',
                          'Outlook 1-2Months',
                          'Risk >1Year',
                          'Outlook >1Year',
                          'finrank',
                         'Cap',
                         'Dividend Yield']].rename(columns={'finrank':'YoY Financial Strength'}).round(2).set_index('Name').T
    return df_new
#Top Level Stock Data
def stock(df,lst):
    df_new = df[df['Name'].isin(lst)][['Name',
                          'Latest Open',
                         'Latest High',
                         'Latest Low',
                         'Latest Close',
                         'Latest Volume',
                         'Close_change_1d',
                         'Close_change_5d',
                         'Close_change_1m',
                         'Close_change_3m',
                         'Close_change_1y',]].rename(columns={'Latest Open':'Open',
                                                             'Latest High':'High',
                                                             'Latest Low':'Low',
                                                             'Latest Close':'Close',
                                                             'Close_change_1d':'1 Day Return (%)',
                                                             'Close_change_5d':'5 Day Return (%)',
                                                             'Close_change_1m':'1 Month Return (%)',
                                                             'Close_change_3m':'3 Month Return (%)',
                                                             'Close_change_1y':'1 Year Return (%)',}).round(2).set_index('Name').T
    return df_new

# YoY financials
def financials(df,lst):
    x = pd.DataFrame()
    for i in lst:
        x = pd.concat([x,get_financials_single(i,df)],axis=0)
    return x.T

def technical(df,lst):
    x = pd.DataFrame()
    for i in lst:
        x = pd.concat([x,get_technicals_single(i,df)],axis=0)
    return x.T


def main():
    
    st.title("StockSpectra - Compare Stocks")
    st.divider()
    
    data = load_data("backend_data/database.csv")
    #getting stocks from url
    lst= []
    if st.experimental_get_query_params() != {} and 'symbols' in st.experimental_get_query_params().keys():
        lst = st.experimental_get_query_params()['symbols'][0].replace("_",".")
        lst = lst.split(",")
        n = len(lst)
    elif st.experimental_get_query_params() != {} and 'symbol' in st.experimental_get_query_params().keys():
        lst = st.experimental_get_query_params()['symbol'][0].replace("_",".")
        lst = [lst]
        n = len(lst)
        st.experimental_set_query_params()
        
    starting_names = get_names_from_symbols(lst,data)
        
    #selecting exchange 
    col1,col2 = st.columns([1,4])
    if lst !=[]:
        if lst[0][-2:] == 'BO':
            exchange =  col1.selectbox("Select Exchange:",('BSE','NSE'))
        else:
            exchange =  col1.selectbox("Select Exchange:",('NSE','BSE'))
    else:
        exchange =  col1.selectbox("Select Exchange:",('NSE','BSE'))
    data = data[data['Exchange']==exchange]  
    #selecting stocks
    stocks = col2.multiselect('Stocks to Compare', data.sort_values(by='Name').Name,starting_names)
    #st.write(get_names_from_symbols(lst,data))
    
    with st.expander('Firmographics', expanded=True):
        st.dataframe(basics(data,stocks),use_container_width = True)
    
    with st.expander('Stock', expanded=True):
        st.dataframe(stock(data,stocks),use_container_width = True)
        
    with st.expander('YoY Financials', expanded=False):
        st.dataframe(financials(data,stocks),use_container_width = True)
        
    with st.expander('Stock Technicals', expanded=False):
        st.dataframe(technical(data,stocks),use_container_width = True)
        
    #st.experimental_set_query_params()
    
   
if __name__ == "__main__":

    main() 
