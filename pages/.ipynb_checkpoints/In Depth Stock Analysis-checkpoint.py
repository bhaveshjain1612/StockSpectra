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

st.set_page_config(layout = "wide")

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

#Generate FirmoGraphic Data
def generate_firmo(data):
    
    col1, col2, col3 = st.columns(3)
    
    col1.header(data.Sector.values[0])
    col1.write("Sector")
    
    col2.header(data.Industry.values[0])
    col2.write("Industry")
    
    col3.header(data.Cap.values[0])
    col3.write("Market Size")
    
    st.subheader("About the company")

    col1, col2 = st.columns([4,2])
    col1.write(data.Description.values[0])
    col2.plotly_chart(holding_chart(data),use_container_width=True,height=200)
    
    st.divider()
    col1,col2 = st.columns([4,1])
    col1.subheader('Related Companies')
    
    #comparision url
    related  = related_companies(data.Symbol.values[0], load_data('backend_data/database.csv'), 5).reset_index()[['Name','Symbol']]
    
    relatedurl = f'https://stock-recommendation.streamlit.app//Compare_Stocks/?symbols={",".join(np.append(related.Symbol.values,data.Symbol.values[0])).replace(".","_")}'
    
    col2.subheader("[Compare Stocks]("+relatedurl+")")
    
    col1,col2,col3,col4,col5 = st.columns(5)
    
    n=0
    for j in [col1,col2,col3,col4,col5]:
        j.write(related.Name.values[n])
        url = f'https://stock-recommendation.streamlit.app/In_Depth_Stock_Analysis/?symbol={related.Symbol.values[n].replace(".","_")}'
        j.write("[In Depth Analysis]("+url+")")
        n+=1
        
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
    col1,col2,col3 = st.columns([1,1,2])
    col1.subheader("Key Takeaways")
    outdur = col2.selectbox("Outlook Duration",("1-2 Months",">1 Year"))
    
    stksummary = stock_summary(data,historical_with_ma,week52_data,dividend_split_data)
    if len(stksummary)>1:
        col1, col2, col3 = st.columns([2,2,2])
        
        if len(stksummary)%2 ==1:
            for m in list(stksummary.items())[:(len(stksummary)//2)+1]:
                col2.write(m[1]['Display'])
            for m in list(stksummary.items())[(len(stksummary)//2)+1:]:
                col3.write(m[1]['Display'])
        else:
            for m in list(stksummary.items())[:(len(stksummary)//2)]:
                col2.write(m[1]['Display'])
            for m in list(stksummary.items())[(len(stksummary)//2):]:
                col3.write(m[1]['Display'])
    else:
        col1, col2 = st.columns(2)
        for m in list(stksummary.items()):
            col2.write(m[1]['Display'])
            
    col1.header(data['Outlook '+outdur.replace(" ","")].values[0].title()+" - "+data['Risk '+outdur.replace(" ","")].values[0].title()+" Risk")
    
    #Technical Inidcators Latest Values
    with st.expander("Technical Indicators (Latest Values)"):
        col1, col2, col3, col4 = st.columns(4)

        col1.write("**Average Directional Index (14)**")
        col1.write(str(round(data['Latest ADX'].values[0],2))+" - "+indicator_tags(data, historical_with_ma)['adx'])
        
        col2.write("**Relative Strength Index (14)**")
        col2.write(str(round(data['Latest rsi'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['rsi'])
        
        col3.write("**Commodity Channel Index (10)**")
        col3.write(str(round(data['Latest CCI_10'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['cci_10'])
        
        col4.write("**Commodity Channel Index (40)**")
        col4.write(str(round(data['Latest CCI_40'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['cci_40'])
        
        col2.write("**Money Flow Index (14)**")
        col2.write(str(round(data['Latest MFI_14'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['mfi_14'])
        
        col1.write("**Moving Average Convergence Divergence**")
        col1.write(str(round(data['Latest macd'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['macd'])
        
        col3.write("**Volume Price Trend**")
        col3.write(str(round(data['Latest VPT'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['VPT'])
        
        col4.write("**Williamson%R (14)**")
        col4.write(str(round(data['Latest %R'].values[0],2))+ " - " +indicator_tags(data, historical_with_ma)['%R'])
     
        #Moving Averages
        st.divider()
        matype = st.selectbox('Moving Average Type',('SMA','EMA'))
        col1,col2 = st.columns([5,3])
        col1.write('Values')
        col2.write('Crossovers')
        c1,c2,c3,c4,c5 = col1.columns(5)
        
        c1.write('**5 Days**')
        c1.write(str(round(historical_with_ma[matype.lower()+"_5"].values[0],2)))
        c1.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_5"])
                 
        c2.write('**10 Days**')
        c2.write(str(round(historical_with_ma[matype.lower()+"_10"].values[0],2)))
        c2.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_10"])
                 
        c3.write('**20 Days**')
        c3.write(str(round(historical_with_ma[matype.lower()+"_20"].values[0],2)))
        c3.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_20"])
                 
        c4.write('**50 Days**')
        c4.write(str(round(historical_with_ma[matype.lower()+"_50"].values[0],2)))
        c4.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_50"])
                 
        c5.write('**100 Days**')
        c5.write(str(round(historical_with_ma[matype.lower()+"_100"].values[0],2)))
        c5.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_100"])
                 
        c1,c2,c3 = col2.columns(3)
        c1.write('**Short**')
        c1.write('5 & 20 Days')
        c1.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_shortcross"])
        
        c2.write('**Median**')
        c2.write('20 & 50 Days')
        c2.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_midcross"])
        
        c3.write('**Long**')
        c3.write('50 & 100 Days')
        c3.write(indicator_tags(data, historical_with_ma)[matype.lower()+"_longcross"])

        #Pivot levels
        st.divider()   
        st.write("Expected pivot points for next day:")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.write("R3")
        col1.write(str(round(data['Latest R3'].values[0],2)))
        col2.write("R2")
        col2.write(str(round(data['Latest R2'].values[0],2)))
        col3.write("R1")
        col3.write(str(round(data['Latest R1'].values[0],2)))
        col4.write("PP")
        col4.write(str(round(data['Latest PP'].values[0],2)))
        col5.write("S1")
        col5.write(str(round(data['Latest S1'].values[0],2)))
        col6.write("S2")
        col6.write(str(round(data['Latest S2'].values[0],2)))
        col7.write("S3")
        col7.write(str(round(data['Latest S3'].values[0],2)))
    
    ##Charts
    st.divider()
    st.subheader("Stock Price and technical charts")

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
    ]
    #ma duratiosn for plotting
    col1, col2 = st.columns([2,8])
    
    ma_names = []
    for i in [5,10,15,20,25,30,40,50,75,100,150,200]:
        ma_names.append('SMA ('+str(i)+')')
        ma_names.append('EMA ('+str(i)+')') 
    ta_list = ma_names+["Volume","ADX (14)", "RSI (14)", "CCI (10)", "CCI (40)","OBV", "VPT", "CMF", 'Williamson%R (14)', 'MACD', 'VWAP', 'MFI (14)',"Bollinger (1 STD)", "Bollinger (2 STD)"]

    filter_interval = col1.selectbox('Select a Time Interval', ["6 Months","1 Day", "5 Days", "1 Month","3 Months",  "1 Year", "2 Years"])
    plot_filter = col2.multiselect('Select Technical Indicators',ta_list, "Volume")
    
    fig = generate_charts(historical_with_ma,filter_interval,plot_filter, holiday_list)
        
    st.plotly_chart(fig,theme="streamlit", use_container_height=True, use_container_width=True, height=1000)
    
    #technical inidcator values 
    
    
# Generate financial details    
def generate_financials(data):
    
    fin_file = "backend_data/company_financials/"+data.Symbol.values[0].replace(".","_")+".csv"
    fin = load_data(fin_file)
    
    #st.dataframe(fin)
    
    kpis = calc_KPIs(fin.set_index('Unnamed: 0').T.reset_index(),'normal')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Net Income",simplify(kpis['Net Income']['current']), simplify(kpis['Net Income']['delta']))
    col2.metric("Debt",simplify(kpis['Debt']['current']), simplify(kpis['Debt']['delta']))
    col3.metric("Free Cash Flow",simplify(kpis['Free Cash Flow']['current']), simplify(kpis['Free Cash Flow']['delta']))
    col4.metric("Basic EPS",kpis['Basic EPS']['current'], kpis['Basic EPS']['delta'])
    col5.metric("Net Profit Margin",kpis['Net Profit Margin']['current'], kpis['Net Profit Margin']['delta'])
    st.divider()
    col1.metric("ROA",kpis['ROA']['current'], kpis['ROA']['delta'])
    col2.metric("ROE",kpis['ROE']['current'], kpis['ROE']['delta'])
    col3.metric("ROCE",kpis['ROCE']['current'], kpis['ROCE']['delta'])
    col4.metric("Current Ratio",kpis['Current Ratio']['current'], kpis['Current Ratio']['delta'])
    col5.metric("DE Ratio",kpis['DE Ratio']['current'], kpis['DE Ratio']['delta'])
    
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
        
#get newws for stocks      
def generate_news(name):
    #try:
    news = pd.read_csv("backend_data/news_articles/"+name.replace(" ","_")+".csv").reset_index()
    #st.dataframe(news)
    
    if news.empty:
        st.write("no News articles about the company in past 14 days")
    else:
        st.header("Most Recent news articles (Last 14 Days)")
        
    for i in news.index:
        st.subheader(news.title.values[i])
        col1, col2, col3 = st.columns([1,1,5])
        col1.write(news.source.values[i])
        col2.write(news.date.values[i])
        link  = "https//:"+news.link.values[i]  
        #st.write(news.summary[i])
        st.write("[Read More....]("+link+")")
        st.divider()
    #except:
    #    st.write("no News articles about the company in past 14 days")
    
#Load all insights
def load_insights(data,input_symbol):
    ex_list = data['Exchange'].tolist()
    
    st.header(data.Name.values[0])  
    
    ex_sel = st.sidebar.radio("Exchange: ",ex_list)
    
    if ex_sel=="NSE":
        symbol = input_symbol.upper()+".NS"
        data = data[data['Symbol']==symbol].reset_index() 
    else:
        symbol = input_symbol.upper()+".BO"
        data = data[data['Symbol']==symbol].reset_index()
        
    st.subheader(data.Exchange.values[0]+" : " +data.Symbol.values[0][:-3])   
        
    tab1, tab2, tab3, tab4 = st.tabs(["Company Details", "Technical Analysis", "Financials", "News"])
    
    with tab1:
        generate_firmo(data)
    with tab2:
        generate_stock(data)
    with tab3:
        generate_financials(data)
    with tab4:
        generate_news(data.Name.values[0])
    #st.dataframe(data)

def main():
    data = load_data("backend_data/database.csv")
    #data = allot_tags(data)
    st.title('StockSpectra - In Depth Analysis')  
    
    input_symbol = st.sidebar.text_input("Enter stock ticker/name")       
    
    if input_symbol:
        st.experimental_set_query_params()
        data_company = data[(data['Symbol'] == input_symbol.upper()+".NS") | (data['Symbol'] == input_symbol.upper()+".BO")].reset_index().drop('index',axis=1)    
        #load_insights(data,input_symbol)
        if data_company.shape[0] == 0:
            found = data[data['Name'].str.contains(input_symbol, case=False) | data['Symbol'].str.contains(input_symbol, case=False)][['Name','Symbol','Exchange','Sector','Industry','Latest Close']].set_index('Name')
            st.subheader("Results of search:")
            add_links(found)
        else:
            load_insights(data_company,input_symbol)
  
    elif st.experimental_get_query_params() != {}:
        try:
            in_s = st.experimental_get_query_params()['symbol'][0][:-3]
            data = data[(data['Symbol']==in_s.upper()+".NS")|(data['Symbol']==in_s.upper()+".BO")].reset_index().drop('index',axis=1)
            if st.experimental_get_query_params()['symbol'][0][-3:]=="_BO":
                data = data.sort_values(by="Exchange")
            else:
                data = data.sort_values(by="Exchange", ascending = False)
            
            load_insights(data,in_s)
            
        except:
            st.experimental_set_query_params()
            
    if st.experimental_get_query_params() == {} and input_symbol==False:
        st.info('Enter The Stock Name/Symbol in the sidebar', icon="ℹ️")
        #st.dataframe(data)
    

if __name__ == "__main__":

    main() 
