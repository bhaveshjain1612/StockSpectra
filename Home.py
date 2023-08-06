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
        return [f'https://stock-recommendation.streamlit.app/In_Depth_Stock_Analysis/?symbol={t.replace(".","_")}' for t in Symbol]

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
    db=df
    
    #_________________ALL COLUMN 1 FILTERS_____________________________
    #Create Filter by investment strategy
    strategy =  col1.selectbox("Strategy", ("None","1-2 Months", "3-6 Months", "> 1 Year","Potential Breakout"))

    # Filter by stock name
    name = col1.text_input('Enter name')
        
    # Filter by view option (Top Picks, All, or Potential Breakout)
    dividend = col1.selectbox("Dividend", strategy_allotting(strategy)[3])
        
    #_________________ALL COLUMN 2 FILTERS_____________________________    
    #Filter based on Risk
    risk_filter = col2.multiselect('Risk/Reward Preference',np.insert(df['Risk'].unique(), 0, "All"), strategy_allotting(strategy)[0])
    if 'All' in risk_filter or risk_filter ==[]:
        risk_filter = df['Risk'].unique()    
        
    # Filter by stock exchange (NSE or BSE)
    exchange = col2.selectbox("Exchange", ("All","NSE", "BSE"))
    if exchange == 'All' :
        exchange = df['Exchange'].unique()
    else:
        exchange = [exchange]
        
    #filtering based on interval    
    filter_interval = col2.selectbox('Select a Time Interval for % changes', ["1 Day", "5 Days", "1 Month", "3 Months","6 Months", "1 Year"])
    
    #_________________ALL COLUMN 3 FILTERS_____________________________    
    # Filter by stock rank (outlook)
    stkrank_filter = col3.multiselect('Outlook Preference',np.insert(df['Outlook'].unique(), 0, "All"), strategy_allotting(strategy)[1])
    if 'All' in stkrank_filter or stkrank_filter ==[]:
        stkrank_filter = df['Outlook'].unique()    
        
    #Filter by sector
    sector_filter = col3.multiselect('Select Sector', np.insert(df['Sector'].unique(), 0, "All"), ['All'])
    if 'All' in sector_filter or sector_filter ==[]:
        sector_filter = df['Sector'].unique()
        
    #Select column to sort by
    sort_by = col3.selectbox('Order by ', ("Name","Latest Close","Dividend Yield","Change (%)"))
    
    #_________________ALL COLUMN 4 FILTERS_____________________________    
    # Filter by financial rank
    finrank_filter = col4.multiselect('Company YoY financials', np.insert(df['finrank'].unique(), 0, "All"), strategy_allotting(strategy)[2])     
    if 'All' in finrank_filter or finrank_filter ==[]:
        finrank_filter = df['finrank'].unique()
        
    # Filter by industry
    industry_filter = col4.multiselect('Select Industry', np.insert(df[df['Sector'].isin(sector_filter)]['Industry'].unique(), 0, "All"), ['All'])
    if 'All' in industry_filter or industry_filter ==[]:
        industry_filter = df[df['Sector'].isin(sector_filter)]['Industry'].unique()     
        
    # Sorting Method 
    sort_type = col4.selectbox('Order method', ("None","Ascending","Descending"))
    
    #_________________Executing Filters and showing df_____________________________ 
    
    #filtering based on possible breakout
    if strategy == 'Potential Breakout':
        final=df[df['Symbol'].isin(load_data("backend_data/breakout.csv")['Unnamed: 0'])]
    else:
        final=df
    #filtering based on name
    if name:
        final = final[final['Name'].str.contains(name, case=False) | df['Symbol'].str.contains(name, case=False)]
        
    #dividend filter:
    final = final.loc[lambda x: (x['Dividend Yield'] > 0) if dividend == "Yes" else ((x['Dividend Yield'] == 0) if dividend == "No" else x['Dividend Yield']!='Not Found')]
        
    # Applying exchange_filters
    final=final[final['Exchange'].isin(exchange)]
    # Applying sector_filters
    final=final[final['Sector'].isin(sector_filter)]
    # Applying industry_filters
    final=final[final['Industry'].isin(industry_filter)]
    # Applying risk_filters
    final=final[final['Risk'].isin(risk_filter)]
    # Applying outlook_filters
    final=final[final['Outlook'].isin(stkrank_filter)]
    # Applying financials_filters
    final=final[final['finrank'].isin(finrank_filter)]
    #applying time interval filter
    final.rename(columns = {"Close_change_"+filter_interval[:3].lower().replace(" ",""):'Change (%)'}, inplace = True)

    # Display the final DataFrame with links
    final = final[['Name', 'Symbol','Exchange', 'Sector', 'Industry', 'Latest Close','Change (%)','Outlook','Risk', 'finrank','Dividend Yield']].rename(columns={'finrank': 'Company Financials', 'stkrank': 'Outlook'})
    final = final.drop_duplicates(subset='Name').set_index('Name')
    
    #Sorting Functions
    if sort_type != "None":
        final = final.sort_values(by=sort_by,ascending=(sort_type=="Ascending"))
    
    if final.shape[0] > 10:
        final = final.head(st.slider('Companies to Display', 0, final.shape[0], 10))
 
    with st.container():
        add_links(final)
    
def main():
    # Load data from the CSV file and preprocess
    df = load_data("backend_data/database.csv")

    df = allot_tags(df)
    df['Annual Dividend'] = df['Dividend Rate'].replace('Not Found', '0').astype(float)
    df['Dividend Yield'] = df['Annual Dividend']*100/df['Latest Close']
    
    # Set up the Streamlit dashboard
    st.header("Find Stocks")
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