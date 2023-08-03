# Importing libraries
import yfinance as yf  # Library for fetching stock data
from datetime import date, datetime  # Libraries for date-related operations
import numpy as np  # Library for numerical computations
import time  # Library for time-related operations
import tqdm  # Progress bar library
from tqdm import tqdm  # tqdm submodule for progress bar
import streamlit as st  # Web application framework for creating dashboards
import sys  # Library for system-related functionalities
from streamlit_autorefresh import st_autorefresh  # Auto-refresh functionality for Streamlit
from utils import *  # Custom utility functions
import pandas as pd  # Library for data handling and manipulation

# Set Streamlit page configuration
st.set_page_config(
    page_title="Dashboard",  # Title of the page
    layout="wide"  # Wide layout to maximize dashboard space
)

# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to add clickable links to the DataFrame
def add_links(df):
    # Function to generate the URL for in-depth stock analysis
    def add_ind_depth_url(Symbol):
        return [f'http://localhost:8501/In_Depth_Stock_Analysis/?symbol={t.replace(".","_")}' for t in Symbol]

    # Function to convert URL to clickable link
    def make_clickable(url, text):
        return f'<a target="_self" href="{url}">{text}</a>'

    # Add 'Analysis' column with clickable links to DataFrame
    df['Analysis'] = add_ind_depth_url(df.Symbol)
    df['Analysis'] = df['Analysis'].apply(make_clickable, text='See in Depth')
        
    if df.empty:
        st.error("No companies found for the selection")
        st.empty()
    else:
        st.write(str(df.shape[0]) + " Companies found")
        st.write(df.to_html(escape=False), unsafe_allow_html=True)

# Function for collective filtering and displaying of data
def collective(df):
    # Split the dashboard into four columns
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # Filter by stock name
    name = col1.text_input('Enter name')
    
    # Filter by stock exchange (NSE or BSE)
    exchange = col2.selectbox("Exchange", ("NSE", "BSE"))
    if exchange == "NSE":
        Lexc = df[df['Exchange'] == 'NSE'].sort_values(by=['stkscore', 'finscore'], ascending=False)
    else:
        Lexc = df[df['Exchange'] == 'BSE'].sort_values(by=['stkscore', 'finscore'], ascending=False)
    
    # Filter by stock rank (outlook)
    stkrank_filter = col3.multiselect('Select Outlook', np.insert(df['stkrank'].unique(), 0, "All"), ['All'])
    if stkrank_filter:
        if stkrank_filter == ['All']:
            L2 = df
        else:
            L2 = df.query('stkrank in @stkrank_filter') 
    else:
        L2 = df
    
    # Filter by financial rank
    finrank_filter = col4.multiselect('Select Financial', np.insert(df['finrank'].unique(), 0, "All"), ['All'])
    if finrank_filter:
        if finrank_filter == ['All']:
            L3 = df
        else:
            L3 = df.query('finrank in @finrank_filter') 
    else:
        L3 = df
    
    # Split the dashboard into three columns
    col1, col2, col3 = st.columns([2, 3, 3])
    
    # Filter by view option (Top Picks, All, or Potential Breakout)
    option = col1.selectbox("View", ("Top Picks", "All", "Potential Breakout"))
    if option == "Top Picks":
        # Filtering stocks based on stock rank and financial rank
        Ltop = df[(df['finrank'] == 'strong') & (df['stkrank'] == 'positive')].sort_values(by=['stkscore', 'finscore'], ascending=False)
    elif option == "Potential Breakout":
        # Filtering stocks based on potential breakout list from a CSV file
        Ltop = df[df['Symbol'].isin(pd.read_csv("backend_data/breakout.csv")['Unnamed: 0'])].sort_values(by=['stkscore', 'finscore'], ascending=False)
    else:
        Ltop = df
    
    # Filter by sector
    sector_filter = col2.multiselect('Select Sector', np.insert(df['Sector'].unique(), 0, "All"), ['All'])
    if sector_filter:
        if sector_filter == ['All']:
            df_filter_sector = df
        else:
            df_filter_sector = df.query('Sector in @sector_filter')
    else:
        df_filter_sector = df
    
    # Filter by industry
    industry_filter = col3.multiselect('Select Industry', np.insert(df_filter_sector['Industry'].unique(), 0, "All"), ['All'])
    if industry_filter:
        if industry_filter == ['All']:
            L1 = df_filter_sector
        else:
            L1 = df_filter_sector.query('Industry in @industry_filter') 
    else:
        L1 = df_filter_sector
        
    # Apply filters and obtain the final DataFrame
    if name:
        L0 = df[df['Name'].str.contains(name, case=False) | df['Symbol'].str.contains(name, case=False)]
    else:
        L0 = df
    
    col1, col2, col3 = st.columns([2, 3, 3])
    df_temp=df
    filter_interval = col1.selectbox('Select a Time Interval for % changes', ["3 Months","1 Day", "5 Days", "1 Month", "6 Months", "1 Year"])
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
            
    sort_by = col2.selectbox('Order by ', ("Name","Latest Close","Price Change (%)"))    
    sort_type = col3.selectbox('Order method', ("None","Ascending","Descending"))

                    
    final = df_temp[df_temp['Symbol'].isin(L0.Symbol)]
    final = final[final['Symbol'].isin(L1.Symbol)]
    final = final[final['Symbol'].isin(L2.Symbol)]
    final = final[final['Symbol'].isin(L3.Symbol)]
    final = final[final['Symbol'].isin(Ltop.Symbol)]
    final = final[final['Symbol'].isin(Lexc.Symbol)]
    
    # Display the final DataFrame with links
    final = final[['Name', 'Symbol', 'Sector', 'Industry', 'Latest Close','Price Change (%)', 'stkrank', 'finrank']].rename(columns={'finrank': 'Company Financials', 'stkrank': 'Outlook'})
    final = final.drop_duplicates(subset='Name').set_index('Name')
    
    if sort_type != "None":
        final = final.sort_values(by=sort_by,ascending=(sort_type=="Ascending"))
    
    if final.shape[0] > 10:
        final = final.head(st.slider('Companies to Display', 0, final.shape[0], 10))
 
    with st.container():
        add_links(final)

def main():
    # Load data from the CSV file and preprocess
    df = load_data("backend_data/database.csv")
    df['finscore'] = df.apply(financial_scores, axis=1)
    df = finrank(df)
    df['stkscore'] = df.apply(stock_scores, axis=1)
    df = stkrank(df)
    
    # Set up the Streamlit dashboard
    st.header("Screening Page")
    st.write("Updated On: " + df['Latest date_only'].values[0])
    
    # Display the filtered data using the collective function
    collective(df)

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Set Streamlit query parameters for refreshing the dashboard
    st.experimental_set_query_params()
    
    # Disclaimer text
    disclaimer_text = '''This product is intended solely for educational purposes and should not be utilized as a financial tool. The creator of this product makes no guarantees or warranties about the accuracy or completeness of the information provided. Any actions taken based on this product's content are at your own risk, and the creator shall not be liable for any damages or losses, whether direct or indirect. Financial matters involve inherent risks and complexities, requiring professional advice. Before making any financial decisions, it is essential to consult with a qualified financial advisor. By using this product, you agree not to hold the creator responsible for any outcomes resulting from its educational content. Remember, this product does not replace professional financial advice, and its use is purely for educational and informational purposes. Proceed with this product only if you understand and accept this disclaimer. If you disagree with these terms, do not use this product for financial purposes and seek alternative financial education and guidance.'''
    
    # Initialize hide state
    if 'hide' not in st.session_state:
        st.session_state.hide = True

    # Function to show/hide the disclaimer button
    def show_hide():
        st.session_state.hide = not st.session_state.hide

    # Display the disclaimer button
    if st.session_state.hide:
        secret = st.container()
        with secret:
            st.header('''**Disclaimer: Educational Purpose Only**''')
            st.write(disclaimer_text)
            st.button('Agree and Proceed', on_click=show_hide)
    else:
        # Show the main dashboard
        main()