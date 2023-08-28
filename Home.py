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
    page_title="StockSpectra - Home",  # Title of the page
    layout="wide"  # Wide layout to maximize dashboard space
)

# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to add clickable links to the DataFrame
def add_links(df):
    # Function to generate the URL for in-depth stock analysis
    def add_ind_depth_url(Symbol):
        return [f'https://stockspectra.streamlit.app/Detailed_Analysis/?symbol={t.replace(".","_")}' for t in Symbol]

    # Function to convert URL to clickable link
    def make_clickable(url, text):
        return f'<a target="_self" href="{url}">{text}</a>'
        #return url

    # Add 'Analysis' column with clickable links to DataFrame
    df['Analysis'] = add_ind_depth_url(df.Symbol)
    df['Analysis'] = df['Analysis'].apply(make_clickable, text='See in Depth')
        
    if df.empty:
        st.error("No companies found for the selection")
        st.empty()
    else:
        st.write(str(df.shape[0]) + " Companies found")
        st.write(df.to_html(escape=False), unsafe_allow_html=True)
        #st.dataframe(df)

# Function for collective filtering and displaying of data
def collective(df):
    # Split the dashboard into four columns
    st.info('''Sse the plethora of filters here to pick the stocks that match your exact needs.''', icon="ℹ️")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    db=df
    
    #_________________ALL COLUMN 1 FILTERS_____________________________
    #Create Filter by investment strategy
    strategy_selected =  col1.selectbox("Investment Duration", ("1-2 Months","> 1 Year"))
    #filtering based on possible breakout
    strategy=strategy_selected
    final=df

    # Filter by stock name
    name = col1.text_input('Enter name')
        
    # Filter by Dividend
    dividend = col1.selectbox("Dividend", strategy_allotting(strategy)[3])
        
    #_________________ALL COLUMN 2 FILTERS_____________________________    
    #Filter based on Risk
    risk_filter = col2.multiselect('Risk/Reward Preference',np.insert(df["Risk "+strategy.replace(" ","")].unique(), 0, "All"), strategy_allotting(strategy)[0])
    if 'All' in risk_filter or risk_filter ==[]:
        risk_filter = df["Risk "+strategy.replace(" ","")].unique()    
        
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
    stkrank_filter = col3.multiselect('Outlook Preference',np.insert(df["Outlook "+strategy.replace(" ","")].unique(), 0, "All"), strategy_allotting(strategy)[1])
    if 'All' in stkrank_filter or stkrank_filter ==[]:
        stkrank_filter = df["Outlook "+strategy.replace(" ","")].unique()    
        
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
    
    #filtering based on name
    if name:
        final = final[final['Name'].str.contains(name, case=False) | final['Symbol'].str.contains(name, case=False)]
        
    #dividend filter:
    final = final.loc[lambda x: (x['Dividend Yield'] > 0) if dividend == "Yes" else ((x['Dividend Yield'] == 0) if dividend == "No" else x['Dividend Yield']!='Not Found')]
        
    # Applying exchange_filters
    final=final[final['Exchange'].isin(exchange)]
    # Applying sector_filters
    final=final[final['Sector'].isin(sector_filter)]
    # Applying industry_filters
    final=final[final['Industry'].isin(industry_filter)]
    # Applying risk_filters
    final=final[final["Risk "+strategy.replace(" ","")].isin(risk_filter)]
    # Applying outlook_filters
    final=final[final["Outlook "+strategy.replace(" ","")].isin(stkrank_filter)]
    # Applying financials_filters
    final=final[final['finrank'].isin(finrank_filter)]
    #applying time interval filter
    final.rename(columns = {"Close_change_"+filter_interval[:3].lower().replace(" ",""):'Change (%)',
                           "Risk "+strategy.replace(" ",""): "Risk",
                           "Outlook "+strategy.replace(" ",""): "Outlook"}, inplace = True)

    # Display the final DataFrame with links
    final = final[['Name', 'Symbol','Exchange', 'Sector', 'Industry', 'Latest Close','Change (%)','Outlook','Risk', 'finrank','Dividend Yield']].rename(columns={'finrank': 'Company Financials'})
    final = final.drop_duplicates(subset='Name').set_index('Name')
    # Beautifying some columns
    final = final.round(2)
    final['Company Financials'] = final['Company Financials'].str.title()
    final['Outlook'] = final['Outlook'].str.title()
    #Sorting Functions
    if sort_type != "None":
        final = final.sort_values(by=sort_by,ascending=(sort_type=="Ascending"))
    
    if final.shape[0] > 10:
        final = final.head(st.slider('Companies to Display', 0, final.shape[0], 10))
 
    with st.container():
        add_links(final)
        
#function for top picks
def top_picks(df):
    strategies = [
        "Short-Term High Gains Strategy",
        "Stable Growth Strategy",
        "High Dividend Yield Strategy",
        "Defensive Strategy",
        "Aggressive Growth Strategy",
        "Conservative Income Strategy",
        "Turnaround Play",
        "Balanced Portfolio Strategy",
        "Value Play",
        "Momentum Chaser"
    ]
    
    #creating sols in dahsboard
    col1, col2, col3, col4 = st.columns([2,1,1,1])
    # Create a selectbox in Streamlit
    selected_strategy = col1.selectbox('Choose a Stock-Picking Strategy:', strategies)
    
    exchange_p = col2.selectbox("Exchange", ("NSE", "BSE"), key = 'tp_exchange')
    tp = df[df['Exchange']==exchange_p]
    
    #Select column to sort by
    sort_by = col3.selectbox('Order by' , ("Name","Latest Close","Dividend Yield","Change (%)"),key='tp_sortby')
    # Sorting Method 
    sort_type = col4.selectbox('Order method', ("None","Ascending","Descending") ,key='tp_sorttype')
    #display strategy info
    st.info(top_pick_strategy(selected_strategy)['info'], icon="ℹ️")
    '''
    #selecting companies with high short volatility, mid to low in mid ter and low in long term
    tp = tp[tp['finrank']=='strong']
    tp = tp[(tp['Outlook 1-2Months']=='positive') | (tp['Outlook 1-2Months']=='very positive')]
    tp = tp[(tp['Outlook >1Year']=='positive') | (tp['Outlook >1Year']=='very positive')]
    tp = tp[(tp['Risk 1-2Months']=='Mid') | (tp['Risk 1-2Months']=='High')]
    tp = tp[ (tp['Risk >1Year']=='Mid')]
    tp = tp[tp['Dividend Yield']>0]
    '''
    tp = top_pick_strategy(selected_strategy)['df']
    
    tp = tp[['Name','Symbol','Sector','Industry','Latest Close','Close_change_1d','Dividend Yield']].rename(columns={'Close_change_1d':'Change (%)'}).drop_duplicates(subset=['Name']).set_index('Name').round(2)
    
    if sort_type != "None":
        tp = tp.sort_values(by=sort_by,ascending=(sort_type=="Ascending"))
    
    if tp.shape[0] > 10:
        tp = tp.head(st.slider('Companies to Display', 0, tp.shape[0], 10))
 
    with st.container():
        add_links(tp)
        
#function for top gainers
def top_price_changes(df):
    st.info('''These stocks have moved the most in the previous trading session.''', icon="ℹ️")
    
    # Filter by stock exchange (NSE or BSE)
    col1,col2 = st.columns(2)
    
    chng_type = col1.selectbox("Price change type", ("Top Gainers", "Top Losers"))
    
    exchange_g = col2.selectbox("Exchange", ("NSE", "BSE"), key = 'tpc_exchange')
    tg = df[df['Exchange']==exchange_g]
    
    tg = tg[['Name','Symbol','Sector','Industry','Latest Close','Close_change_1d','Dividend Yield']].rename(columns={'Close_change_1d':'Change (%)'}).drop_duplicates(subset=['Name']).set_index('Name').round(2)
    
    tg = tg.sort_values(by='Change (%)',ascending=(chng_type=="Top Losers")).head(10)
 
    with st.container():
        add_links(tg)
    

#function for potentiual breakouts
def potential_breakout(df):
    
    st.info('''These are some stocks that have the possibility of shifting from a bearish trend to a strong bullish trend. However, This change is always uncertain. Please proceed with caution''', icon="ℹ️")
    
    #creating sols in dahsboard
    col1, col2, col3,col4 = st.columns(4)
    
    exchange_p = col1.selectbox("Exchange", ("NSE", "BSE"), key = 'pb_exchange')
    
    pb = df[df['Exchange']==exchange_p]
    
    pb=pb[pb['Symbol'].isin(load_data("backend_data/breakout.csv")['Symbol'])]
    
    compare_link_symbols = f'http://stockspectra.streamlit.app/Compare_Stocks/?symbols={",".join(list(pb["Symbol"])).replace(".","_")}'
    col4.header("[Compare Stocks]("+compare_link_symbols+")")

    #Select column to sort by
    sort_by = col2.selectbox('Order by' , ("Name","Latest Close","Dividend Yield","Change (%)"),key='pb_sortby')
    # Sorting Method 
    sort_type = col3.selectbox('Order method', ("None","Ascending","Descending") ,key='pb_sorttype')
    
    pb = pb[['Name','Symbol','Sector','Industry','Latest Close','Close_change_1d','Dividend Yield']].rename(columns={'Close_change_1d':'Change (%)'}).drop_duplicates(subset=['Name']).set_index('Name').round(2)
    
    if sort_type != "None":
        pb = pb.sort_values(by=sort_by,ascending=(sort_type=="Ascending"))
    
    if pb.shape[0] > 10:
        pb = pb.head(st.slider('Companies to Display', 0, pb.shape[0], 10))
 
    with st.container():
        add_links(pb)

#function for sector overview
def sector_overview(df):
    
    col1,col2,col3 = st.columns(3)
    
    universe = col1.selectbox("See Overview for",('Industry','Sector'))
    
    indview = industryview(df,universe)
    
    indsortby = col2.selectbox("Sort by",('Median 1 Day Change (%)', 'Median 5 Day Change (%)',
                                       'Median 1 Month Change (%)', 'Median 3 Month Change (%)',
                                       'Median 6 Month Change (%)', 'Median 1 Year Change (%)',
                                       'Number of Companies', 'Number of LargeCap', 'Number of MidCap',
                                       'Number of SmallCap'))
    indsortmethod = col3.selectbox("Sort Method",('Descending','Ascending'))
    
    st.dataframe(indview.sort_values(by=indsortby,ascending = (indsortmethod=='Ascending')))

# Main Function
def main():
    # Load data from the CSV file and preprocess
    df = load_data("backend_data/database.csv")

    # Set up the Streamlit dashboard
    st.title("StockSpectra")
    st.write("Updated On: " + df['Latest date_only'].values[0])
    
    # Display the filtered data using the collective function
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top Picks", "Top Price Changes","Potential Breakouts","Sector-Industry Overview","Screener"])

    with tab5:
        collective(df)
        
    with tab1:
        top_picks(df)
        
    with tab2:
        top_price_changes(df)
        
    with tab3:
        potential_breakout(df)
        
    with tab4:
        sector_overview(df)

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Set Streamlit query parameters for refreshing the dashboard
    st.experimental_set_query_params()
    
    # Disclaimer text
    disclaimer_text1 = '''This product is intended solely for educational purposes and should not be utilized as a financial tool. The creator of this product makes no guarantees or warranties about the accuracy or completeness of the information provided. Any actions taken based on this product's content are at your own risk, and the creator shall not be liable for any damages or losses, whether direct or indirect. Financial matters involve inherent risks and complexities, requiring professional advice.'''
    disclaimer_text2 = '''Before making any financial decisions, it is essential to consult with a qualified financial advisor. By using this product, you agree not to hold the creator responsible for any outcomes resulting from its educational content. Remember, this product does not replace professional financial advice, and its use is purely for educational and informational purposes. Proceed with this product only if you understand and accept this disclaimer. If you disagree with these terms, do not use this product for financial purposes and seek alternative financial education and guidance.'''
    
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
            st.title("StockSpectra")
            st.header('''**Disclaimer: Educational Purpose Only**''')
            st.write(disclaimer_text1)
            st.write(disclaimer_text2)
            
            st.button('Agree and Proceed', on_click=show_hide)
    else:
        # Show the main dashboard
        main()