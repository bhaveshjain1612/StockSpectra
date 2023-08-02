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


def main():
    df = load_data("backend_data/database.csv")
    
    st.title("Home")
    st.header("Top Picks for the day:")
    st.subheader("Updated On: "+df['Latest date_only'].values[0])
    
    #add scores and ranks
    df['finscore'] = df.apply(financial_scores,axis=1)
    df = finrank(df)
    df['stkscore'] = df.apply(stock_scores,axis=1)
    
    #filtering stocks
    df = df[df['finrank']=='strong']
    df = df[df['stkscore']>df.stkscore.describe()['75%']]
    
    #displaying df ordered by top stock score and top financial score
    df = df.sort_values(by=['stkscore','finscore'],ascending=False)
    df = df.drop_duplicates(subset='Name')
    df = df[['Name','Symbol','Sector','Industry','Latest Close','finrank']].rename(columns={'finrank': 'Company Financials'}).set_index('Name')
    
    pd.set_option('display.max_colwidth', 1)

    def add_ind_depth_url(Symbol):
        return [f'https://stock-recommendation.streamlit.app/In_Depth_Stock_Analysis?symbol={t.replace(".","_")}' for t in Symbol]

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
        to_show = st.slider('Companies to Display', 0, df.shape[0], 10)
        st.write(df.head(to_show).to_html(escape = False), unsafe_allow_html = True)

    
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
