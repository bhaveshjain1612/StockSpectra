#importing libraries
import yfinance as yf
from datetime import date
import datetime
import numpy as np
import time
import tqdm
from tqdm import tqdm
from datetime import datetime
import streamlit as st
import sys
from streamlit_autorefresh import st_autorefresh
from utils import *
import pandas as pd

st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)

# Load CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

def add_links(df):
    def add_ind_depth_url(Symbol):
        return [f'http://localhost:8501/In_Depth_Stock_Analysis/?symbol={t.replace(".","_")}' for t in Symbol]

    def make_clickable(url, text):
        return f'<a target="_self" href="{url}">{text}</a>'

    # show data
    df['Analysis'] = add_ind_depth_url(df.Symbol)
    df['Analysis'] = df['Analysis'].apply(make_clickable, text='See in Depth')
        
    if df.empty:
        st.error("No companies found for the selection")
        st.empty()
    else:
        st.write(str(df.shape[0])+" Companies found")
        st.write(df.to_html(escape = False), unsafe_allow_html = True)

def top_picks(df):
    st.header("Top Picks for the day")
    st.write("Updated On: "+df['Latest date_only'].values[0])
    
    #filtering stocks
    df = df[df['finrank']=='strong']
    df = df[df['stkrank']=='positive']
    
    #displaying df ordered by top stock score and top financial score
    df = df.sort_values(by=['stkscore','finscore'],ascending=False)
    df = df.drop_duplicates(subset='Name')
    df = df[['Name','Symbol','Sector','Industry','Latest Close','finrank']].rename(columns={'finrank': 'Company Financials'}).set_index('Name')
    
    pd.set_option('display.max_colwidth', 1)
    
    if df.shape[0]>10:
        df = df.head(st.slider('Companies to Display', 0, df.shape[0],10 ))

    add_links(df)
        
def search(df):
    
    col1,col2,col3,col4,col5 = st.columns([1,2,2,1,1])
    
    #name
    name = col1.text_input('Enter name')
    
    #Sector
    sector_filter = col2.multiselect('Select Sector',np.insert(df['Sector'].unique(),0,"All"),['All'])
    if sector_filter:
        if sector_filter == ['All']:
            df_filter_sector = df
        else:
            df_filter_sector = df.query('Sector in @sector_filter')
    else:
        df_filter_sector = df
    
    #industry filter
    industry_filter = col3.multiselect('Select Industry',np.insert(df_filter_sector['Industry'].unique(),0,"All"),['All'])
    if industry_filter:
        if industry_filter == ['All']:
            L1 = df_filter_sector
        else:
            L1 = df_filter_sector.query('Industry in @industry_filter') 
    else:
        L1 = df_filter_sector
    
    #stockrank filter
    stkrank_filter = col4.multiselect('Select Outlook',np.insert(df_filter_sector['stkrank'].unique(),0,"All"),['All'])
    if stkrank_filter:
        if stkrank_filter == ['All']:
            L2 = df
        else:
            L2 = df.query('stkrank in @stkrank_filter') 
    else:
        L2 = df
    
    #finrank filter
    finrank_filter = col5.multiselect('Select Financial',np.insert(df_filter_sector['finrank'].unique(),0,"All"),['All'])
    if finrank_filter:
        if finrank_filter == ['All']:
            L3 = df
        else:
            L3 = df.query('finrank in @finrank_filter') 
    else:
        L3 = df
        
    if name:
        L0 = df[df['Name'].str.contains(name,case=False)|df['Symbol'].str.contains(name,case=False)]
    else:
        L0 = df
        
    final = L1[L1['Symbol'].isin(L0.Symbol)]
    final = final[final['Symbol'].isin(L2.Symbol)]
    final = final[final['Symbol'].isin(L3.Symbol)]
    
    final = final[['Name','Symbol','Sector','Industry','Latest Close','stkrank','finrank']].rename(columns={'finrank': 'Company Financials','stkrank':'Outlook'})
    final = final.drop_duplicates(subset='Name').set_index('Name')
    
    if final.shape[0]>10:
        final = final.head(st.slider('Companies to Display', 0, final.shape[0],10 ))
 
    add_links(final)

def main():
    df = load_data("backend_data/database.csv")
    
    tab1, tab2 = st.tabs(["Top Picks", "Search",])
    
    #add scores and ranks
    df['finscore'] = df.apply(financial_scores,axis=1)
    df = finrank(df)
    df['stkscore'] = df.apply(stock_scores,axis=1)
    df = stkrank(df)
    
    with tab1:
        top_picks(df)
    with tab2:
        search(df)

    
if __name__ == "__main__":
    
    st.experimental_set_query_params()
    
    lt  = '''This product is intended solely for educational purposes and should not be utilized as a financial tool. The creator of this product makes no guarantees or warranties about the accuracy or completeness of the information provided. Any actions taken based on this product's content are at your oltwn risk, and the creator shall not be liable for any damages or losses, whether direct or indirect. Financial matters involve inherent risks and complexities, requiring professional advice. Before making any financial decisions, it is essential to consult with a qualified financial advisor.By using this product, you agree not to hold the creator responsible for any outcomes resulting from its educational content. Remember, this product does not replace professional financial advice, and its use is purely for educational and informational purposes. Proceed with this product only if you understand and accept this disclaimer. If you disagree with these terms, do not use this product for financial purposes and seek alternative financial education and guidance.'''

    if 'hide' not in st.session_state:
        st.session_state.hide = True

    def show_hide():
        st.session_state.hide = not st.session_state.hide

    if st.session_state.hide:
        secret = st.container()
        with secret:
            st.header('''**Disclaimer: Educational Purpose Only**''')
            st.write(lt)
            st.button('Agree and Proceed', on_click=show_hide)
    else:
        main()