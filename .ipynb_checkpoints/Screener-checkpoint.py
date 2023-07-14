#!pip install yfinance

#importing libraries
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
from screener_utils import *
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode


st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)

# Load CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

def colour_coded_table(x):
        positive= ['Buy','Bullish','Low Risk','Strong Buy']
        negative = ['Sell','Bearish','High Risk','Strong Sell']
        neutral = ['Hold','Wait','Moderate Risk','Weak Uptrend','Weak Downtrend']
        unknown = ['Unknown','No Trend']

        def color_category(val):
            if any([x == str(val) for x in positive]):
                color="green" 
            elif any([x == str(val) for x in negative]):
                color = 'red'
            elif any([x == str(val) for x in neutral]):
                color = 'yellow'
            elif any([x == str(val) for x in unknown]):
                color = 'gray'
            else:
                color = ''
            return 'color: %s' % color

        st.dataframe(x.style.applymap(color_category))

# Create dashboard
def create_dashboard(data):
    st.title("Screener")
    updated_date = data['Latest created_on'].values[0].split(" ")[0]
    updated_time = data['Latest created_on'].values[0].split(" ")[1][:5]
    st.write(' '.join(["Data updated on:",updated_date,updated_time]))

    to_process = ["Name",
                "Symbol",
                "Sector",
                "Industry",
                "Dividend Rate",
                "Latest Close",
                "Close_change_1d",
                "Close_change_5d",
                "Close_change_1m",
                "Close_change_3m",
                "Close_change_6m",
                "Close_change_1y",
                "Volume_change_1d",
                "Volume_change_5d",
                "Volume_change_1m",
                "Volume_change_3m",
                "Volume_change_6m",
                "Volume_change_1y",
                 "Beta",
                 "Latest macd",
                 "Latest macd_ema_9",
                  "macd_change_5d"]
    
    outdata = data[to_process]
    #sidebar filters
    with st.sidebar:
        #filter based on name
        try:
            filter_name = st.sidebar.text_input("Enter company name")
            if filter_name:
                df_temp = pd.DataFrame()
                substring = filter_name.lower()
                for record in outdata['Name']:
                    if substring.lower() in record.lower():
                        df_temp = pd.concat([df_temp,outdata[outdata['Name'] == record]],axis=0)
                if df_temp.empty:
                    #st.warning("No Companies found matching "+filter_name)
                    df_temp = outdata.drop(outdata.index, inplace=True)
            else:
                df_temp = outdata


            df_temp = df_temp
            #Filter for intervals
            filter_interval = st.selectbox('Select a Time Interval for % changes', ["1 Day", "5 Days", "1 Month","3 Months", "6 Months", "1 Year"])
            if filter_interval: 
                if filter_interval == "1 Day":
                    df_temp = df_temp.drop(["Close_change_5d","Close_change_1m","Close_change_3m","Close_change_6m","Close_change_1y"],axis=1)
                    df_temp = df_temp.drop(["Volume_change_5d","Volume_change_1m","Volume_change_3m","Volume_change_6m","Volume_change_1y"],axis=1)
                    df_temp.rename(columns = {'Close_change_1d':'Price Change (%)','Volume_change_1d':'Volume Change (%)'}, inplace = True)
                elif filter_interval == "5 Days":
                    df_temp = df_temp.drop(["Close_change_1d","Close_change_1m","Close_change_3m","Close_change_6m","Close_change_1y"],axis=1)
                    df_temp = df_temp.drop(["Volume_change_1d","Volume_change_1m","Volume_change_3m","Volume_change_6m","Volume_change_1y"],axis=1)
                    df_temp.rename(columns = {'Close_change_5d':'Price Change (%)','Volume_change_5d':'Volume Change (%)'}, inplace = True)
                elif filter_interval == "1 Month":
                    df_temp = df_temp.drop(["Close_change_1d","Close_change_5d","Close_change_3m","Close_change_6m","Close_change_1y"],axis=1)
                    df_temp = df_temp.drop(["Volume_change_1d","Volume_change_5d","Volume_change_3m","Volume_change_6m","Volume_change_1y"],axis=1)
                    df_temp.rename(columns = {'Close_change_1m':'Price Change (%)','Volume_change_1m':'Volume Change (%)'}, inplace = True)
                elif filter_interval == "3 Months":
                    df_temp = df_temp.drop(["Close_change_1d","Close_change_5d","Close_change_1m","Close_change_6m","Close_change_1y"],axis=1)
                    df_temp = df_temp.drop(["Volume_change_1d","Volume_change_5d","Volume_change_1m","Volume_change_6m","Volume_change_1y"],axis=1)
                    df_temp.rename(columns = {'Close_change_3m':'Price Change (%)','Volume_change_3m':'Volume Change (%)'}, inplace = True)
                elif filter_interval == "6 Months":
                    df_temp = df_temp.drop(["Close_change_1d","Close_change_5d","Close_change_1m","Close_change_3m","Close_change_1y"],axis=1)
                    df_temp = df_temp.drop(["Volume_change_1d","Volume_change_5d","Volume_change_1m","Volume_change_3m","Volume_change_1y"],axis=1)
                    df_temp.rename(columns = {'Close_change_6m':'Price Change (%)','Volume_change_6m':'Volume Change (%)'}, inplace = True)
                elif filter_interval == "1 Year":
                    df_temp = df_temp.drop(["Close_change_1d","Close_change_5d","Close_change_1m","Close_change_3m","Close_change_6m"],axis=1)
                    df_temp = df_temp.drop(["Volume_change_1d","Volume_change_5d","Volume_change_1m","Volume_change_3m","Volume_change_6m"],axis=1)
                    df_temp.rename(columns = {'Close_change_1y':'Price Change (%)','Volume_change_1y':'Volume Change (%)'}, inplace = True)

            #make decsions on markettrend
            df_temp['Trend'] = df_temp.apply(determine_market_trend, axis=1)

            #vaoltilty based on Beta value
            df_temp['Risk'] = df_temp['Beta'].apply(determine_risk_level)

            # buy sell reccomendation based on ema
            df_temp['Recommendation'] = df_temp.apply(determine_recommendation, axis=1)

            #FIlTERS
            #Sector and Industry
            sector_filter = st.multiselect('Select Sector',np.insert(df_temp['Sector'].unique(),0,"All"),['All'])
            if sector_filter:
                if sector_filter == ['All']:
                    df_temp_filter_sector = df_temp
                else:
                    df_temp_filter_sector = df_temp.query('Sector in @sector_filter')
            else:
                df_temp_filter_sector = df_temp


            industry_filter = st.multiselect('Select Industry',np.insert(df_temp_filter_sector['Industry'].unique(),0,"All"),['All'])
            if industry_filter:
                if industry_filter == ['All']:
                    df_temp_filter_industry = df_temp_filter_sector
                else:
                    df_temp_filter_industry = df_temp_filter_sector.query('Industry in @industry_filter') 
            else:
                df_temp_filter_industry = df_temp_filter_sector

            #recxcomendation based filtering
            recommend_filter = st.multiselect('Select Recommendation',np.insert(df_temp['Recommendation'].unique(),0,"All"),['All'])
            if recommend_filter:
                if recommend_filter == ['All']:
                    df_temp_filter_recommend = df_temp
                else:
                    df_temp_filter_recommend = df_temp.query('Recommendation in @recommend_filter') 
            else:
                df_temp_filter_recommend = df_temp

            #market trend based filtering
            trend_filter = st.multiselect('Select Trend',np.insert(df_temp['Trend'].unique(),0,"All"),['All'])
            if trend_filter:
                if trend_filter == ['All']:
                    df_temp_filter_trend = df_temp
                else:
                    df_temp_filter_trend = df_temp.query('Trend in @trend_filter') 
            else:
                df_temp_filter_trend = df_temp

            #risk factor based filtering
            risk_filter = st.multiselect('Select Risk Factor',np.insert(df_temp['Risk'].unique(),0,"All"),['All'])
            if risk_filter:
                if risk_filter == ['All']:
                    df_temp_filter_risk = df_temp
                else:
                    df_temp_filter_risk = df_temp.query('Risk in @risk_filter') 
            else:
                df_temp_filter_risk = df_temp


            merged_df = df_temp_filter_recommend.loc[df_temp_filter_recommend['Symbol'].isin(df_temp_filter_industry['Symbol']),:]
            merged_df = merged_df.loc[merged_df['Symbol'].isin(df_temp_filter_trend['Symbol']),:]
            merged_df = merged_df.loc[merged_df['Symbol'].isin(df_temp_filter_risk['Symbol']),:]

        except:
            #st.error("No companies found for the selection. Kindly refresh or enter a valid company name.")
            merged_df = pd.DataFrame()
        
    to_display = ["Name",
                "Symbol",
                "Sector",
                "Industry",
                "Dividend Rate",
                "Latest Close",
                  "Price Change (%)",
                  "Volume Change (%)",
                 "Trend",
                 "Risk",
                 "Recommendation"]

    if merged_df.empty:
        st.error("No companies found for the selection")
        st.empty()
    else:
        st.write(str(merged_df[to_display].shape[0])+" Companies found")
        st.dataframe(merged_df[to_display])
        #colour_coded_table(merged_df[to_display])


# Main function
def main():     
    data = load_data("backend_data/database.csv")
    create_dashboard(data)
    

if __name__ == "__main__":
    
    main()
#    '''
    while True:
        now = pd.Timestamp.now().time()
        
        if now.hour == 20 and now.minute == 00 and now.second == 00:
            subprocess.run([f"{sys.executable}", "backend_data/collective_backend.py"])
            data = load_data("backend_data/database.csv")
            #subprocess.terminate()
            st.experimental_rerun()
        time.sleep(1)
#        '''
