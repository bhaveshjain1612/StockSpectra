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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

######################################################################################################################################
# Home Page Fuctions
######################################################################################################################################

#collective code to allot financial scores, outlooks and all other scores in one go
#generate industry view table
def industryview(df,universe): 

    d1 = df.groupby(universe)['Close_change_1d'].agg(['median']).rename(columns={'median':'Median 1 Day Change (%)'})
    d5 = df.groupby(universe)['Close_change_5d'].agg(['median']).rename(columns={'median':'Median 5 Day Change (%)'})
    m1 = df.groupby(universe)['Close_change_1m'].agg(['median']).rename(columns={'median':'Median 1 Month Change (%)'})
    m3 = df.groupby(universe)['Close_change_3m'].agg(['median']).rename(columns={'median':'Median 3 Month Change (%)'})
    m6 = df.groupby(universe)['Close_change_6m'].agg(['median']).rename(columns={'median':'Median 6 Month Change (%)'})
    y1 = df.groupby(universe)['Close_change_1y'].agg(['median']).rename(columns={'median':'Median 1 Year Change (%)'})
    co = df.groupby(universe)['Close_change_1y'].agg(['count']).rename(columns={'count':'Number of Companies'})
    cc = df.pivot_table(index=universe, columns='Cap', aggfunc='size', fill_value=0)
    df = pd.concat([d1,d5,m1,m3,m6,y1,co,cc],axis=1).rename(columns={'Large-cap':'Number of LargeCap','Mid-cap':'Number of MidCap','Small-cap':'Number of SmallCap'})

    return df

def top_pick_strategy(strategy, df):
    result = {}
    
    if strategy == "Short-Term High Gains Strategy" :
        result['info'] = "This strategy targets stocks that are expected to provide significant returns in the short term. While there's a high risk associated with these stocks, the very positive short-term outlook suggests potential for quick gains."
        result['df'] = df[(df['Risk 1-2Months']=='High')
                         & (df['Outlook 1-2Months']=='very positive')
                         & (df['finrank']=='strong')]
        
    elif strategy == "Stable Growth Strategy" :
        result['info'] = "This strategy focuses on stocks that have a stable growth trajectory. The low long-term risk and positive outlook over a year suggest that these stocks are likely to grow steadily over time."
        result['df'] = df[((df['Risk 1-2Months']=='Low') | (df['Risk 1-2Months']=='Mid'))
                         & ((df['Outlook 1-2Months']=='positive') | (df['Outlook 1-2Months']=='neutral'))
                         & (df['Risk >1Year']=='Low')
                         & (df['Outlook >1Year']=='very positive')
                         & (df['finrank']=='strong')]
        
    elif strategy == "High Dividend Yield Strategy" :
        result['info'] = "This strategy targets stocks that pay dividends. The focus is on companies with a strong or mid-level financial year-over-year strength, indicating they have the capacity to continue paying dividends."
        result['df'] = df[((df['Risk >1Year']=='Low') | (df['Risk >1Year']=='Mid'))
                         & ((df['Outlook >1Year']=='very positive') | (df['Outlook >1Year']=='positive') | (df['Outlook >1Year']=='neutral'))
                         & ((df['finrank']=='strong') | (df['finrank']=='mid'))
                         & (df['Dividend Yield'] > 5)]
        
    elif strategy == "Defensive Strategy" :
        result['info'] = "This strategy is for investors who want to minimize risk. The focus is on stocks with low risk in both the short and long term, a neutral or positive outlook, and a strong financial track record. The inclusion of dividends provides an additional source of income."
        result['df'] = df[(df['Risk 1-2Months']=='Low')
                         & ((df['Outlook 1-2Months']=='positive') | (df['Outlook 1-2Months']=='neutral'))
                         & (df['Risk >1Year']=='Low')
                         & ((df['Outlook >1Year']=='positive') | (df['Outlook >1Year']=='neutral'))
                         & (df['finrank']=='strong')
                         & (df['Dividend Yield'] > 0)]
        
    elif strategy == "Aggressive Growth Strategy" :
        result['info'] = "This strategy is for investors with a high risk tolerance, targeting stocks that have the potential for significant growth. While there's a higher level of risk, the very positive outlook suggests substantial potential returns."
        result['df'] = df[(df['Risk 1-2Months']=='High')
                         & (df['Outlook 1-2Months']=='very positive') 
                         & ((df['Risk >1Year']=='High') | (df['Risk >1Year']=='Mid'))
                         & (df['Outlook >1Year']=='very positive') 
                         & ((df['finrank']=='mid') | (df['finrank']=='weak'))
                         & (df['Dividend Yield'] == 0)]
        
    elif strategy == "Conservative Income Strategy" :
        result['info'] = "This strategy is for investors seeking a steady income with minimal risk. Stocks selected under this strategy are expected to have a consistent dividend payout and exhibit low volatility."
        result['df'] = df[(df['Risk 1-2Months']=='Low')
                         & ((df['Outlook 1-2Months']=='positive') | (df['Outlook 1-2Months']=='neutral'))
                         & (df['Risk >1Year']=='Low') 
                         & ((df['Outlook >1Year']=='positive') | (df['Outlook >1Year']=='neutral') )
                         & (df['finrank']=='strong')
                         & (df['Dividend Yield'] > 0)]
        
    elif strategy == "Turnaround Play" :
        result['info'] = "This strategy targets stocks that are currently underperforming but are expected to rebound in the long term. It's a speculative play, betting on the company's potential to turn its fortunes around."
        result['df'] = df[(df['Risk 1-2Months']=='High')
                         & (df['Outlook 1-2Months']=='negative') 
                         & (df['Risk >1Year']=='Mid') 
                         & ((df['Outlook >1Year']=='very positive') | (df['Outlook >1Year']=='positive') )
                         & ((df['finrank']=='weak') | (df['finrank']=='mid'))]
        
    elif strategy == "Balanced Portfolio Strategy" :
        result['info'] = "This strategy aims to maintain a balanced portfolio with a mix of growth and value stocks. It's suitable for investors seeking moderate growth with controlled risk."
        result['df'] = df[(df['Risk 1-2Months']=='Mid')
                         & ((df['Outlook 1-2Months']=='positive'))
                         & (df['Risk >1Year']=='Mid') 
                         & ((df['Outlook >1Year']=='positive'))
                         & (df['finrank']!='weak')]
        
    elif strategy == "Value Play" :
        result['info'] = "This strategy targets undervalued stocks that are expected to appreciate over time. The focus is on companies that are currently overlooked by the market but have strong fundamentals"
        result['df'] = df[(df['Risk 1-2Months']=='Low')
                         & ((df['Outlook 1-2Months']=='positive') | (df['Outlook 1-2Months']=='neutral'))
                         & (df['Risk >1Year']=='Low') 
                         & ((df['Outlook >1Year']=='positive') | (df['Outlook >1Year']=='neutral') )
                         & (df['finrank']=='strong')
                         & (df['Dividend Yield'] > 0)]
        
    elif strategy == "Momentum Chaser" :
        result['info'] = "This strategy is for investors looking to capitalize on current market trends. It targets stocks that have shown strong recent performance and are expected to continue their upward trajectory."
        result['df'] = df[((df['Risk 1-2Months']=='High') | (df['Risk 1-2Months']=='Mid'))
                         & ((df['Outlook 1-2Months']=='very positive'))
                         & (df['Risk >1Year']=='High') 
                         & ((df['Outlook >1Year']=='very positive'))
                         & ((df['finrank']=='mid') | (df['finrank']=='strong')) ]
                          
        
    else:
        result['info'] = 'Stocks with strong financials, positive outlooks for short and long rangfe with decreasing risk over time. They pay regular dividends'
        result['df'] = df[(df['finrank']=='strong')
                        & ((df['Outlook 1-2Months']=='positive') | (df['Outlook 1-2Months']=='very positive'))
                        & ((df['Outlook >1Year']=='positive') | (df['Outlook >1Year']=='very positive'))
                        & ((df['Risk 1-2Months']=='Mid') | (df['Risk 1-2Months']=='High'))
                        & (df['Risk >1Year']=='Mid')
                        & (df['Dividend Yield']>0)]
    
    return result

######################################################################################################################################
# In Depth Functions
######################################################################################################################################

#Related Companies
def related_companies(symbol, df, n):
    # Extract base company details
    base_company = df[df['Symbol']==symbol]
    base_industry = base_company['Industry'].iloc[0]
    base_sector = base_company['Sector'].iloc[0]
    base_cap = base_company['Cap'].iloc[0]
    base_price = base_company['Latest Close'].iloc[0]

    df = df[df['Exchange']==base_company['Exchange'].values[0]]
    df = df[df['Symbol']!=symbol]
    # Helper function to calculate priority score
    def priority_score(row):
        score = 9
        if row['Industry'] == base_industry:
            if row['Cap'] == base_cap:
                if abs(row['Latest Close'] - base_price) <= 0.20 * base_price:
                    score = 1
                elif abs(row['Latest Close'] - base_price) <= 0.50 * base_price:
                    score = 2
                else:
                    score = 3
            else:
                score = 7
        elif row['Sector'] == base_sector:
            if row['Cap'] == base_cap:
                if abs(row['Latest Close'] - base_price) <= 0.20 * base_price:
                    score = 4
                elif abs(row['Latest Close'] - base_price) <= 0.50 * base_price:
                    score = 5
                else:
                    score = 6
            else:
                score = 8
        return score

    # Apply the priority score function to the dataframe
    df['priority'] = df.apply(priority_score, axis=1)

    # Sort by priority and get the top n companies
    closest_n = df.sort_values(by='priority').head(n)

    # Drop the priority column for the final output
    closest_n.drop(columns=['priority'])

    return closest_n



#pie chart for holding composition
def holding_chart(df):
    insider = df['% of Shares Held by All Insider'].values[0]
    institutions = df['% of Shares Held by Institutions'].values[0]
    public = 100-insider-institutions
    

    labels = ["Public", "Promoter", "Institutions"]
    values = [public,insider,institutions]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig.update_layout(title_text="Holding Pattern")
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),)

    return fig

#calculate % cahnge b/w two ends of user defined interval and qty
def calc_change(data,series,duration):
    #duartion is atuple i.e 1 month is (1, 'M'), 1 week is (1, 'W') etc.
    if duration[1]!="D":
        end = data['date_only'][0]
        start = end-np.timedelta64(duration[0], duration[1])
        value_start = data[data['date_only']>=start][series].values[-1]
        value_end = data[data['date_only']>=end][series].values[-1]
    else:
        value_end = data[data["Trading Day"]==1][series].values[0]
        value_start = data[data["Trading Day"]==1+duration[0]][series].values[0]
    #return (value_start,value_end)
    return round(((value_end - value_start) / value_start) * 100,2)

#calculate EMAs for a user specified interval
def calc_ema(df, interval):
    df = df.sort_values('Trading Day', ascending=False)
    ema = df['Close'].ewm(span=interval, adjust=False).mean()
    col_name = "ema_"+str(interval)
    df[col_name] = ema
    df = df.sort_values('Trading Day', ascending=True)
    return df

#calculate SMA
def calc_sma(df, interval):
    df = df.sort_values('Trading Day', ascending=False)
    sma = df['Close'].rolling(interval).mean()
    col_name = "sma_"+str(interval)
    df[col_name] = sma
    df = df.sort_values('Trading Day', ascending=True)
    return df

#generate all neccessary ma signals
def generate_ma_signals(df, list_period):
    for i in list_period:
        df = calc_sma(calc_ema(df, i),i)
    return df

#52 week data
def week_52(data):
    end = data['date_only'][0]
    start = end-np.timedelta64(52, "W")
    high = data[data['date_only']>=start]['High'].max()
    low = data[data['date_only']>=start]['Low'].min()
    high_perc = (high - data.Close.values[0])*100 / high
    low_perc = (low - data.Close.values[0])*100 / -low
    return {'52 Week High':high, '52 Week Low':low,'Diff between high':high_perc, 'Diff between low':low_perc}

#get info about splits and ividends
def dividends_splits(data):
    fy23 = data[data.date_only > pd.to_datetime("2022-03-31", format='%Y-%m-%d')]
    fy23 = fy23[fy23.date_only < pd.to_datetime("2023-04-01", format='%Y-%m-%d')]
    if data[data['Stock Splits']>0].reset_index().empty:
        split_date = 'in past 2 years'
        split = "No Stock split"
    else:
        date_object = datetime.strptime(str(data[data['Stock Splits']>0].reset_index()['date_only'][0]), '%Y-%m-%d %H:%M:%S')
        split_date = "Most Recent Split date: "+str(date_object.strftime('%d-%m-%y'))
        split = str(data[data['Stock Splits']>0].reset_index()['Stock Splits'][0])+":1"
    normal_dividend = fy23.Dividends.sum()
    if normal_dividend==0:
        normal_dividend = "No Dividend"   
    else:
        normal_dividend = "Rs. "+str(round(normal_dividend,2))
    return {"Normal dividend":normal_dividend,"split ratio":split, "split date":split_date}

#generate primary plot
def generate_charts(df, interval, items, holiday_list):
    
    #allotting which indicator goes with price and hwich oes below
    upper_plots, lower_plots = [],[]
    for item in items:
        if item in ["Volume","ADX (14)", "RSI (14)", "CCI (10)", "CCI (40)", "OBV", "VPT", "CMF", 'Williamson%R (14)', 'MACD']:
            lower_plots.append(item)
        elif "SMA (" in item or "EMA (" in item or "Bollinger" in item or item in ['VWAP', 'MFI (14)']:
            upper_plots.append(item)
    
    if len(lower_plots) > 0:
        rheights = [0.6] + [0.4/len(lower_plots) for _ in lower_plots]
    else:
        rheights = [1]
    
    # picking the inteerval
    df['date_only'] = pd.to_datetime(df['date_only'], format='%Y-%m-%d').astype('datetime64[ns]')
    df = df[df['date_only'] >df['date_only'][0]-np.timedelta64(interval.replace(" ","")[0], interval.replace(" ","")[1])]
    
    # Create subplots
    fig = make_subplots(rows=len(lower_plots)+1, cols=1, shared_xaxes=True,  vertical_spacing=0.01,  row_heights=rheights,)

    # Add candlestick trace
    fig.add_trace(go.Candlestick(x=df['Date'],  open=df['Open'],  high=df['High'],  low=df['Low'],  close=df['Close'],  name='Price'), row=1, col=1)
    # Add MACD traces
    if 'MACD' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['macd'], mode='lines', name='MACD'), row=lower_plots.index('MACD')+2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['macd_signal'], mode='lines', name='Signal Line', line=dict(dash='dash')), row=lower_plots.index('MACD')+2, col=1)
        fig.add_shape(
        go.layout.Shape(
                type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=0,  y1=0, line=dict(color="Red"),
            ),row=lower_plots.index('MACD')+2, col=1)

    # Add volume trace
    if 'Volume' in lower_plots:
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=lower_plots.index('Volume')+2, col=1)
        
    # Add short CCI trace
    if 'CCI (10)' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI_10'], name='CCI (10)', mode='lines'), row=lower_plots.index('CCI (10)')+2, col=1)
        fig.add_shape(
        go.layout.Shape( type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=100,  y1=100, line=dict(dash='dot'),
            ),row=lower_plots.index('CCI (10)')+2, col=1)
        fig.add_shape(
        go.layout.Shape( type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=-100,  y1=-100, line=dict(dash='dot'),
            ),row=lower_plots.index('CCI (10)')+2, col=1)
        
    # Add long CCI trace
    if 'CCI (40)' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI_40'], name='CCI (40)', mode='lines'), row=lower_plots.index('CCI (40)')+2, col=1)
        fig.add_shape(
        go.layout.Shape( type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=100,  y1=100, line=dict(dash='dot'),
            ),row=lower_plots.index('CCI (40)')+2, col=1)
        fig.add_shape(
        go.layout.Shape( type="line", x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=-100,  y1=-100, line=dict(dash='dot'),
            ),row=lower_plots.index('CCI (40)')+2, col=1)
    
    # RSI trace
    if 'RSI (14)' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['rsi'], name='RSI (14)', mode='lines'), row=lower_plots.index('RSI (14)')+2, col=1)
    
    # ADX trace
    if 'ADX (14)' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], name='ADX (14)', mode='lines'), row=lower_plots.index('ADX (14)')+2, col=1)
        
    #  VPT traces
    if 'VPT' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['VPT'], mode='lines', name='VPT'), row=lower_plots.index('VPT')+2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['VPT_signal'], mode='lines', name='VPT Signal Line', line=dict(dash='dash')), row=lower_plots.index('VPT')+2, col=1)
        
     # OBV trace
    if 'OBV' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['OBV'], name='OBV', mode='lines'), row=lower_plots.index('OBV')+2, col=1)
        
      # CMF trace
    if 'CMF' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['CMF'], name='CMF', mode='lines'), row=lower_plots.index('CMF')+2, col=1)
        
      # Willamson R trace
    if 'Williamson%R (14)' in lower_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['%R'], name='Williamson%R(14)', mode='lines'), row=lower_plots.index('Williamson%R (14)')+2, col=1)
        
    # VWAP trace
    if 'VWAP' in upper_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['VWAP'], name='VWAP', mode='lines'), row=1, col=1)
        
    # VWAP trace
    if 'MFI (14)' in upper_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MFI_14'], name='MFI (14)', mode='lines'), row=1, col=1)
        
    # Bollinger 1 STD trace
    if 'Bollinger (1 STD)' in upper_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RollingMean'], name='Rolling Mean', mode='lines', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['UpperBand1'], name='Upper Band (1STD)', mode='lines', line=dict(dash='dot', color= 'gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['LowerBand1'], name='Lower Band (1STD)', mode='lines', line=dict(dash='dot', color= 'gray')), row=1, col=1)
    
    # Bollinger 2 STD Trace
    if 'Bollinger (2 STD)' in upper_plots:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RollingMean'], name='Rolling Mean', mode='lines', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['UpperBand2'], name='Upper Band (2STD)', mode='lines', line=dict(dash='dot', color= 'yellow')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['LowerBand2'], name='Lower Band (2STD)', mode='lines', line=dict(dash='dot', color= 'yellow')), row=1, col=1)
        
        #Dividend TRaces
    div = df[df['Dividends']!=0][['Date','Close']].reset_index()
    rng = (df.Close.max() - df.Close.min())/10
    for i in div.index:
        fig.add_shape(
        go.layout.Shape( type="line", x0=div['Date'].values[i], x1=div['Date'].values[i], y0=div['Close'].values[i]-rng,  y1=div['Close'].values[i]+rng, line=dict(dash='dot'),
                ),row=1, col=1)
        fig.add_annotation(x=div['Date'].values[i] ,y=div['Close'].values[i]+rng, text="Dividend")
        
    # moving averages
    for i in upper_plots:
        if 'SMA (' in i or 'EMA (' in i:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[i.lower().replace('(','').replace(')','').replace(' ','_')], name=i, mode='lines',), row=1, col=1)
            
    #Update axes
    fig.update_yaxes(showgrid=True, minor=dict(showgrid=False),showline=True, linewidth=2)

    for i in ['Price']+lower_plots:
        fig.update_yaxes(title_text=i, row=(['Price']+lower_plots).index(i)+1, col=1)
        
    fig.update_xaxes(
            rangeslider_visible=False,showgrid=True,showline=True, linewidth=2,
            rangebreaks=[dict(values=holiday_list),dict(bounds=["sat", "mon"])])  # hide weekends, eg. hide sat to before mon

    fig.update_layout(autosize=False,width=1200,height=800,template="plotly_dark")
    
    return fig


#Generate technical indicator tags
def indicator_tags(data, hist):
    tags = {}
    #RSi
    if data['Latest rsi'].values[0] < 30:
        tags['rsi'] = 'Oversold'
    elif data['Latest rsi'].values[0] < 50:
        tags['rsi'] = 'Bearish'
    elif data['Latest rsi'].values[0] < 60:
        tags['rsi'] = 'Neutral'
    elif data['Latest rsi'].values[0] < 80:
        tags['rsi'] = 'Bullish'
    else:
        tags['rsi'] = 'Overbought'
    #ADX
    if data['Latest ADX'].values[0] > 30:
        tags['adx'] = 'Strong Trend'
    else:
        tags['adx'] = 'Weak Trend'
    #cci 10
    if data['Latest CCI_10'].values[0] < -150:
        tags['cci_10'] = 'Oversold'
    elif data['Latest CCI_10'].values[0] < -50:
        tags['cci_10'] = 'Bearish'
    elif data['Latest CCI_10'].values[0] < 50:
        tags['cci_10'] = 'Neutral'
    elif data['Latest CCI_10'].values[0] < 150:
        tags['cci_10'] = 'Bullish'
    else:
        tags['cci_10'] = 'Overbought'
    #cci 40
    if data['Latest CCI_40'].values[0] < -150:
        tags['cci_40'] = 'Oversold'
    elif data['Latest CCI_40'].values[0] < -50:
        tags['cci_40'] = 'Bearish'
    elif data['Latest CCI_40'].values[0] < 50:
        tags['cci_40'] = 'Neutral'
    elif data['Latest CCI_40'].values[0] < 150:
        tags['cci_40'] = 'Bullish'
    else:
        tags['cci_40'] = 'Overbought'
    #MFI
    if data['Latest MFI_14'].values[0] < 20:
        tags['mfi_14'] = 'Oversold'
    elif data['Latest MFI_14'].values[0] < 80:
        tags['mfi_14'] = 'Neutral'
    else:
        tags['mfi_14'] = 'Overbought'
    #MACD
    if data['Latest macd'].values[0] >0 and data['Latest macd'].values[0] > data['Latest macd_signal'].values[0]:
        tags['macd'] = 'Bullish'
    elif data['Latest macd'].values[0] < 0 and  data['Latest macd_signal'].values[0] > data['Latest macd'].values[0]:
        tags['macd'] = 'Bearish'
    else:
        tags['macd'] = 'Neutral'
    #MACD
    if data['Latest VPT'].values[0] > data['Latest VPT_signal'].values[0]:
        tags['VPT'] = 'Bullish'
    elif data['Latest VPT_signal'].values[0] > data['Latest VPT'].values[0]:
        tags['VPT'] = 'Bearish'
    else:
        tags['VPT'] = 'Neutral'
    #Willamson %R
    if data['Latest %R'].values[0] < -80:
        tags['%R'] = 'Oversold'
    elif data['Latest %R'].values[0] < -60:
        tags['%R'] = 'Bearish'
    elif data['Latest %R'].values[0] < -40:
        tags['%R'] = 'Neutral'
    elif data['Latest %R'].values[0] < -20:
        tags['%R'] = 'Bullish'
    else:
        tags['%R'] = 'Overbought'

    #Moving averages
    for i in ['sma_','ema_']:
        for j in [5,10,20,50,100]:
            if hist.Close.values[0] > hist[i+str(j)].values[0]:
                tags[i+str(j)] = 'Bullish'
            else:
                tags[i+str(j)] = 'Bearish'
                
    #crossovers
    for i in ['ema_','sma_']:
        if hist[i+'5'].values[0] > hist[i+'20'].values[0]:
            tags[i+'shortcross'] = 'Bullish'
        else:
            tags[i+'shortcross'] = 'Bearish'
        if hist[i+'20'].values[0] > hist[i+'50'].values[0]:
            tags[i+'midcross'] = 'Bullish'
        else:
            tags[i+'midcross'] = 'Bearish'
        if hist[i+'50'].values[0] > hist[i+'100'].values[0]:
            tags[i+'longcross'] = 'Bullish'
        else:
            tags[i+'longcross'] = 'Bearish'
            
    return tags

#Calculate Financial KPIs
def calc_KPIs(financials,mode):
    #financials = financials.T
    kpis = {}
    #ROE
    kpis['ROE'] = {'desc' : 'Efficiency of Equity utilisation'}
    try:
        if financials['Stockholders Equity'][0] and financials['Net Income'][0]:
            kpis['ROE']['current'] = round(financials['Stockholders Equity'][0]/financials['Net Income'][0],2)
        else:
            kpis['ROE']['current'] = None
        if financials['Stockholders Equity'][1] and financials['Net Income'][1]:
            kpis['ROE']['previous'] = round(financials['Stockholders Equity'][1]/financials['Net Income'][1],2)
        else:
            kpis['ROE']['previous'] = None
        if kpis['ROE']['previous'] and kpis['ROE']['current']:
            kpis['ROE']['delta'] = round(kpis['ROE']['current'] - kpis['ROE']['previous'],2)
        else:
            kpis['ROE']['delta'] = None
    except:
        kpis['ROE']['delta'] = None
        kpis['ROE']['current'] = None
        kpis['ROE']['previous'] = None
        
    #ROA
    kpis['ROA'] = {'desc' : 'Efficiency of Assets utilisation'}
    try:
        if financials['Total Assets'][0] and financials['Net Income'][0]:
            kpis['ROA']['current'] = round(financials['Net Income'][0]/financials['Total Assets'][0],2)
        else:
            kpis['ROA']['current'] = None
        if financials['Total Assets'][1] and financials['Net Income'][1]:
            kpis['ROA']['previous'] = round(financials['Net Income'][1]/financials['Total Assets'][1],2)
        else:
            kpis['ROA']['previous'] = None
        if kpis['ROA']['previous'] and kpis['ROA']['current']:
            kpis['ROA']['delta'] = round(kpis['ROA']['current'] - kpis['ROA']['previous'],2)
        else:
            kpis['ROA']['delta'] = None
    except:
        kpis['ROA']['delta'] = None
        kpis['ROA']['current'] = None
        kpis['ROA']['previous'] = None
        
     #Current Ratio
    kpis['Current Ratio'] = {'desc' : 'Ability to pay short term liabilities'}
    try:
        if financials['Current Assets'][0] and financials['Current Liabilities'][0]:
            kpis['Current Ratio']['current'] = round(financials['Current Assets'][0]/financials['Current Liabilities'][0],2)
        else:
            kpis['Current Ratio']['current'] = None
        if financials['Current Assets'][1] and financials['Current Liabilities'][1]:
            kpis['Current Ratio']['previous'] = round(financials['Current Assets'][1]/financials['Current Liabilities'][1],2)
        else:
            kpis['Current Ratio']['previous'] = None
        if kpis['Current Ratio']['previous'] and kpis['Current Ratio']['current']:
            kpis['Current Ratio']['delta'] = round(kpis['Current Ratio']['current'] - kpis['Current Ratio']['previous'],2)
        else:
            kpis['Current Ratio']['delta'] = None
    except:
        kpis['Current Ratio']['delta'] = None
        kpis['Current Ratio']['current'] = None
        kpis['Current Ratio']['previous'] = None
        
    #Gross Margin
    kpis['Net Profit Margin'] = {'desc' : 'Profitability of a company'}
    try:
        if financials['Total Revenue'][0] and financials['Net Income'][0]:
            kpis['Net Profit Margin']['current'] = round(financials['Net Income'][0]/financials['Total Revenue'][0],2)
        else:
            kpis['Net Profit Margin']['current'] = None
        if financials['Total Revenue'][1] and financials['Net Income'][1]:
            kpis['Net Profit Margin']['previous'] = round(financials['Net Income'][1]/financials['Total Revenue'][1],2)
        else:
            kpis['Net Profit Margin']['previous'] = None
        if kpis['Net Profit Margin']['previous'] and kpis['Net Profit Margin']['current']:
            kpis['Net Profit Margin']['delta'] = round(kpis['Net Profit Margin']['current'] - kpis['Net Profit Margin']['previous'],2)
        else:
            kpis['Net Profit Margin']['delta'] = None
    except:
        kpis['Net Profit Margin']['delta'] = None
        kpis['Net Profit Margin']['current'] = None
        kpis['Net Profit Margin']['previous'] = None
        
    #Debt to equity ratio
    kpis['DE Ratio'] = {'desc' : 'Total debt comapred to equity'}
    try:
        if financials['Stockholders Equity'][0] and financials['Total Debt'][0]:
            kpis['DE Ratio']['current'] = round(financials['Total Debt'][0]/financials['Stockholders Equity'][0],2)
        else:
            kpis['DE Ratio']['current'] = None
        if financials['Stockholders Equity'][1] and financials['Total Debt'][1]:
            kpis['DE Ratio']['previous'] = round(financials['Total Debt'][1]/financials['Stockholders Equity'][1],2)
        else:
            kpis['DE Ratio']['previous'] = None
        if kpis['DE Ratio']['previous'] and kpis['DE Ratio']['current']:
            kpis['DE Ratio']['delta'] = round(kpis['DE Ratio']['current'] - kpis['DE Ratio']['previous'],2)
        else:
            kpis['DE Ratio']['delta'] = None
    except:
        kpis['DE Ratio']['delta'] = None
        kpis['DE Ratio']['current'] = None
        kpis['DE Ratio']['previous'] = None
        
    #Net Income
    kpis['Net Income'] = {'desc' : 'Net Income of the company'}
    try:
        if financials['Net Income'][0]:
            kpis['Net Income']['current'] = financials['Net Income'][0]
        else:
            kpis['Net Income']['current'] = None
        if financials['Net Income'][1]:
            kpis['Net Income']['previous'] = financials['Net Income'][1]
        else:
            kpis['Net Income']['previous'] = None
        if kpis['Net Income']['previous'] and kpis['Net Income']['current']:
            kpis['Net Income']['delta'] = round(kpis['Net Income']['current'] - kpis['Net Income']['previous'],2)
        else:
            kpis['Net Income']['delta'] = None
    except:
        kpis['Net Income']['delta'] = None
        kpis['Net Income']['current'] = None
        kpis['Net Income']['previous'] = None
        
    #Free Cash Flow
    kpis['Free Cash Flow'] = {'desc' : 'In Hand cash flow'}
    try:
        if financials['Free Cash Flow'][0]:
            kpis['Free Cash Flow']['current'] = financials['Free Cash Flow'][0]
        else:
            kpis['Free Cash Flow']['current'] = None
        if financials['Free Cash Flow'][1]:
            kpis['Free Cash Flow']['previous'] = financials['Free Cash Flow'][1]
        else:
            kpis['Free Cash Flow']['previous'] = None
        if kpis['Free Cash Flow']['previous'] and kpis['Free Cash Flow']['current']:
            kpis['Free Cash Flow']['delta'] = round(kpis['Free Cash Flow']['current'] - kpis['Free Cash Flow']['previous'],2)
        else:
            kpis['Free Cash Flow']['delta'] = None
    except:
        kpis['Free Cash Flow']['delta'] = None
        kpis['Free Cash Flow']['current'] = None
        kpis['Free Cash Flow']['previous'] = None
        
    #Total Debt
    kpis['Debt'] = {'desc' : 'Total debt of the company'}
    try:
        if financials['Total Debt'][0]:
            kpis['Debt']['current'] = financials['Total Debt'][0]
        else:
            kpis['Debt']['current'] = None
        if financials['Total Debt'][1]:
            kpis['Debt']['previous'] = financials['Total Debt'][1]
        else:
            kpis['Debt']['previous'] = None
        if kpis['Debt']['previous'] and kpis['Debt']['current']:
            kpis['Debt']['delta'] = round(kpis['Debt']['current'] - kpis['Debt']['previous'],2)
        else:
            kpis['Debt']['delta'] = None
    except:
        kpis['Debt']['delta'] = None
        kpis['Debt']['current'] = None
        kpis['Debt']['previous'] = None
        
    #Basic EPS
    kpis['Basic EPS'] = {'desc' : 'Earnings of the company per share'}
    try:
        if financials['Net Income'][0]:
            kpis['Basic EPS']['current'] = financials['Basic EPS'][0]
        else:
            kpis['Net Income']['current'] = None
        if financials['Net Income'][1]:
            kpis['Basic EPS']['previous'] = financials['Basic EPS'][1]
        else:
            kpis['Basic EPS']['previous'] = None
        if kpis['Basic EPS']['previous'] and kpis['Basic EPS']['current']:
            kpis['Basic EPS']['delta'] = round(kpis['Basic EPS']['current'] - kpis['Basic EPS']['previous'],2)
        else:
            kpis['Basic EPS']['delta'] = None
    except:
        kpis['Basic EPS']['delta'] = None
        kpis['Basic EPS']['current'] = None
        kpis['Basic EPS']['previous'] = None
        
    #ROCE
    kpis['ROCE'] = {'desc':"Utilization of capital employed"}
    try:
        if financials['EBIT'][0] and financials['Total Assets'][0] and financials['Current Liabilities'][0]:
            kpis['ROCE']['current'] = round(financials['EBIT'][0] / (financials['Total Assets'][0]-financials['Current Liabilities'][0]),2)
        else:
            kpis['ROCE']['current'] = None
        if financials['EBIT'][1] and financials['Total Assets'][1] and financials['Current Liabilities'][1]:
            kpis['ROCE']['previous'] = round(financials['EBIT'][1] / (financials['Total Assets'][1]-financials['Current Liabilities'][1]),2)
        else:
            kpis['ROCE']['previous'] = None
        if kpis['ROCE']['previous'] and kpis['ROCE']['current']:
            kpis['ROCE']['delta'] = round(kpis['ROCE']['current'] - kpis['ROCE']['previous'],2)
        else:
            kpis['ROCE']['delta'] = None
    except:
        kpis['ROCE']['delta'] = None
        kpis['ROCE']['current'] = None
        kpis['ROCE']['previous'] = None
        
    if mode == "delta":
        return pd.DataFrame(kpis).loc['delta'].reset_index().set_index('index').T
    else:
        return kpis
    
#Simp[lyfying parametres
def simplify(x):
    try:
        if x >1000000000 or x <-1000000000:
            y = str(round(x/10000000))+" Cr"    
        elif x > 10000000 or x < -10000000:
            y = str(round(x/10000000,2))+" Cr"
        elif x > 100000 or x <-100000:
            y = str(round(x/100000,2))+" L"
        else:
            y = x
        return y
    except:
        return None
    
#financial explanantion
def indicator_summary(x):

    # Initialize an empty summary dictionary
    summary = {}

    # Net Income
    summary['Net Income'] = {
        'Explanation': "Net Income is the company's total revenue minus expenses.",
        'Positive Change': [
            "Improved profitability.",
            "Higher profits may lead to increased shareholder returns and potential growth opportunities."
        ],
        'Negative Change': [
            "Financial challenges or declining performance.",
            "Lower profits can impact the company's ability to invest and grow."
        ],
        'Change': x['Net Income']['delta'],
        'type':1
    }

    # ROA (Return on Assets)
    summary['ROA'] = {
        'Explanation': "ROA measures how efficiently a company uses its assets to generate profit.",
        'Positive Change': [
            "Better asset utilization and potential profitability growth.",
            "Higher ROA is generally desirable as it shows efficient use of resources."
        ],
        'Negative Change': [
            "Inefficiency in using assets to generate profit.",
            "Lower ROA could imply potential risks to the company's financial health."
        ],
        'Change': x['ROA']['delta'],
        'type':1
    }

    # Debt
    summary['Debt'] = {
        'Explanation': "Debt represents the company's total liabilities.",
        'Positive Change': [
            "Lower interest expenses and improved financial stability.",
            "A reduced debt burden may improve the company's creditworthiness."
        ],
        'Negative Change': [
            "Higher interest expenses and potential difficulties in meeting obligations.",
            "Higher debt levels can negatively impact the company's credit rating."
        ],
        'Change': x['Debt']['delta'],
        'type':-1
    }

    # ROE (Return on Equity)
    summary['ROE'] = {
        'Explanation': "ROE measures the company's profitability relative to shareholders' equity.",
        'Positive Change': [
            "Higher utilization of shareholders capital.",
            "Higher ROE is generally desirable as it shows effective use of equity capital."
        ],
        'Negative Change': [
            "Declining utilization of shareholders capital.",
            "Lower ROE could indicate less effective utilization of equity investments."
        ],
        'Change': x['ROE']['delta'],
        'type':1
    }

    # Free Cash Flow
    summary['Free Cash Flow'] = {
        'Explanation': "Free Cash Flow represents the cash generated by the company's core operations.",
        'Positive Change': [
            "Improved financial health and potential for investments.",
            "Higher free cash flow allows the company to fund expansions and pay dividends."
        ],
        'Negative Change': [
            "Financial constraints and reduced investment opportunities.",
            "Lower free cash flow might impact the company's ability to meet financial obligations."
        ],
        'Change': x['Free Cash Flow']['delta'],
        'type':1
    }

    # ROCE (Return on Capital Employed)
    summary['ROCE'] = {
        'Explanation': "ROCE measures the efficiency of capital investments in generating profits.",
        'Positive Change': [
            "Better returns on capital investments.",
            "Higher ROCE is generally desirable as it shows efficient capital utilization."
        ],
        'Negative Change': [
            "Lower profitability from capital investments.",
            "Lower ROCE could indicate inefficiency in allocating capital resources."
        ],
        'Change': x['ROCE']['delta'],
        'type':1
    }

    # Basic EPS (Earnings Per Share)
    summary['Basic EPS'] = {
        'Explanation': "Basic EPS measures the company's earnings available to common shareholders per outstanding share.",
        'Positive Change': [
            "Higher profitability and shareholder returns.",
            "Higher EPS is generally desirable as it shows higher earnings for each share."
        ],
        'Negative Change': [
            "Declining profitability and lower shareholder returns.",
            "Lower EPS could impact investor confidence and share value."
        ],
        'Change': x['Basic EPS']['delta'],
        'type':1
    }

    # Current Ratio
    summary['Current Ratio'] = {
        'Explanation': "Current Ratio measures the company's ability to pay its short-term liabilities.",
        'Positive Change': [
            "Improved liquidity and better short-term solvency.",
            "Higher Current Ratio suggests the company can easily meet its short-term obligations."
        ],
        'Negative Change': [
            "Liquidity challenges and potential difficulties in paying debts.",
            "Lower Current Ratio could indicate increased financial risk and lower creditworthiness."
        ],
        'Change': x['Current Ratio']['delta'],
        'type':1
    }

    # Net Profit Margin
    summary['Net Profit Margin'] = {
        'Explanation': "Net Profit Margin measures the company's profitability as a percentage of revenue.",
        'Positive Change': [
            "Higher profitability on each rupee of sales.",
            "Higher Net Profit Margin is generally desirable as it shows improved efficiency."
        ],
        'Negative Change': [
            "Reduced profitability on sales.",
            "Lower Net Profit Margin could indicate higher costs or lower sales revenue."
        ],
        'Change': x['Net Profit Margin']['delta'],
        'type':1
    }

    # DE Ratio (Debt-to-Equity Ratio)
    summary['DE Ratio'] = {
        'Explanation': "DE Ratio measures the company's financial leverage, comparing debt to equity.",
        'Positive Change': [
            "Reduced financial risk and improved solvency.",
            "Lower DE Ratio suggests the company relies less on debt for financing."
        ],
        'Negative Change': [
            "Higher financial risk and potential difficulties in repaying debt.",
            "Higher DE Ratio could indicate increased dependence on debt financing."
        ],
        'Change': x['DE Ratio']['delta'],
        'type':-1
    }

    # Tag desirable or non-desirable change for each indicator
    for indicator in summary:
        try:
            if summary[indicator]['Change']*summary[indicator]['type'] > 0:
                summary[indicator]['Tag'] = 'Desirable'
                summary[indicator]['To display'] = summary[indicator]['Positive Change']
            elif summary[indicator]['Change']*summary[indicator]['type'] < 0:
                summary[indicator]['Tag'] = 'Non-Desirable'
                summary[indicator]['To display'] = summary[indicator]['Negative Change']
            else:
                summary[indicator]['Tag'] = 'NA'
        except:
            summary[indicator]['Tag'] = 'NA'

    return summary

#in depth stock summary
def stock_summary(stock_data,historical,week_52,dividend_split):
    summary = {}
    
    #yearly return   
    summary['Yearly Returns'] = {}
    if stock_data.Close_change_1y.values[0] > 100:
        summary['Yearly Returns']['Display'] = "Multibagger Returns of "+str(round(stock_data.Close_change_1y.values[0],2)) +"% in 1 past Year"
        summary['Yearly Returns']['Type'] = 1
    elif stock_data.Close_change_1y.values[0] < -100:
        summary['Yearly Returns']['Display'] = "Fell by "+str(round(stock_data.Close_change_1y.values[0],2)*-1) +"% in 1 past Year"
        summary['Yearly Returns']['Type'] = -1
    else:
        summary.pop("Yearly Returns")

    #52 week proximity   
    summary['52 Week Proximity'] = {}
    if stock_data['Latest Close'].values[0] - week_52['52 Week Low'] > -1*(stock_data['Latest Close'].values[0] - week_52['52 Week High']):
        summary['52 Week Proximity']['Display'] = "Closer to 52 Week High"
        summary['52 Week Proximity']['Type'] = 1
    elif stock_data['Latest Close'].values[0] - week_52['52 Week Low'] < -1*(stock_data['Latest Close'].values[0] - week_52['52 Week High']):
        summary['52 Week Proximity']['Display'] = "Closer to 52 Week Low"
        summary['52 Week Proximity']['Type'] = -1
    else:
        summary.pop("52 Week Proximity")

    #RSI
    summary['RSI'] = {}
    if stock_data['Latest rsi'].values[0] < 30:
        summary['RSI']['Display'] = "Stock appears oversold and may rise soon"
        summary['RSI']['Type'] = 1
    elif stock_data['Latest rsi'].values[0] > 80:
        summary['RSI']['Display'] = "Stock appears overbought and may fall soon"
        summary['RSI']['Type'] = -1
    elif stock_data['Latest rsi'].values[0] > 60:
        summary['RSI']['Display'] = "RSI indicates Good Price strength"
        summary['RSI']['Type'] = -1
    else:
        summary.pop("RSI")

    #ADX
    summary['ADX'] = {}
    if stock_data['Latest ADX'].values[0] > 25 and stock_data['Close_change_10d'].values[0] > 0:
        summary['ADX']['Display'] = "Stock having strong uptrend"
        summary['ADX']['Type'] = 1
    elif stock_data['Latest ADX'].values[0] > 25 and stock_data['Close_change_10d'].values[0] < 0:
        summary['ADX']['Display'] = "Stock having strong downtrend"
        summary['ADX']['Type'] = -1
    elif stock_data['Latest ADX'].values[0] <= 25:
        summary['ADX']['Display'] = "Stock having weak trend"
        summary['ADX']['Type'] = -1
    else:
        summary.pop("ADX")

    #Dividend
    summary['Dividend'] = {}
    if dividend_split['Normal dividend'] != "No Dividend":
        summary['Dividend']['Display'] = "Regular Dividend Payout"
        summary['Dividend']['Type'] = 1
    elif dividend_split['Normal dividend'] == "No Dividend":
        summary['Dividend']['Display'] = "Stock doesnt pay dividend"
        summary['Dividend']['Type'] = -1
    else:
        summary.pop("Dividend")

    #PE Ratio
    summary['PE Ratio'] = {}
    if stock_data['P/E ratio'].values[0] > stock_data['Industry Median P/E Ratio'].values[0]:
        summary['PE Ratio']['Display'] = "PE Ratio above Industry Median"
        summary['PE Ratio']['Type'] = 1
    elif stock_data['P/E ratio'].values[0] < stock_data['Industry Median P/E Ratio'].values[0]:
        summary['PE Ratio']['Display'] = "PE Ratio below Industry Median"
        summary['PE Ratio']['Type'] = -1
    else:
        summary.pop("PE Ratio")

    #Momentum
    summary['Momentum'] = {}
    if stock_data['Latest Close'].values[0] > historical['sma_5'].values[0] and historical['sma_5'].values[0] > historical['sma_25'].values[0]:
        summary['Momentum']['Display'] = "Bullish Momentum"
        summary['Momentum']['Type'] = 1
    elif stock_data['Latest Close'].values[0] < historical['sma_5'].values[0] and historical['sma_5'].values[0] < historical['sma_25'].values[0]:
        summary['Momentum']['Display'] = "Bearish Momentum"
        summary['Momentum']['Type'] = -1
    else:
        summary['Momentum']['Display'] = "Mid Range Momentum"
        summary['Momentum']['Type'] = 1
    
    # Pivot Points
    summary['Pivot points'] = {}
    if stock_data['Latest Close'].values[0] < stock_data['Latest S3_previousday'].values[0]:
        summary['Pivot points']['Display'] = "Price fell below S3 levels"
        summary['Pivot points']['Type'] = - 1
    elif stock_data['Latest Close'].values[0] < stock_data['Latest S2_previousday'].values[0]:
        summary['Pivot points']['Display'] = "Price fell below S2 levels"
        summary['Pivot points']['Type'] = - 1
    elif stock_data['Latest Close'].values[0] < stock_data['Latest S1_previousday'].values[0]:
        summary['Pivot points']['Display'] = "Price fell below S1 levels"
        summary['Pivot points']['Type'] = - 1
    elif stock_data['Latest Close'].values[0] > stock_data['Latest R3_previousday'].values[0]:
        summary['Pivot points']['Display'] = "Price rose above R3 levels"
        summary['Pivot points']['Type'] =  1
    elif stock_data['Latest Close'].values[0] > stock_data['Latest R2_previousday'].values[0]:
        summary['Pivot points']['Display'] = "Price rose above R2 levels"
        summary['Pivot points']['Type'] =  1
    elif stock_data['Latest Close'].values[0] > stock_data['Latest R1_previousday'].values[0]:
        summary['Pivot points']['Display'] = "Price rose above R1 levels"
        summary['Pivot points']['Type'] =  1
    else:
        summary.pop('Pivot points')
        
    return summary

######################################################################################################################################
# Comparision FUnctions
######################################################################################################################################

#single company financials
def get_financials_single(Name,db): 
    symbol = db[db['Name']==Name].Symbol.values[0]
    
    fin_file = "backend_data/company_financials/"+symbol.replace(".","_")+".csv"
    fin = pd.read_csv(fin_file)

    #st.dataframe(fin)

    kpis = calc_KPIs(fin.set_index('Unnamed: 0').T.reset_index(),'normal')

    df ={
    'Net Income': simplify(kpis['Net Income']['current'])
    ,'Net Income Change': simplify(kpis['Net Income']['delta'])
    ,'Debt': simplify(kpis['Debt']['current'])
    ,'Debt Change': simplify(kpis['Debt']['delta'])
    ,'Free Cash Flow': simplify(kpis['Free Cash Flow']['current'])
    ,'Free Cash Flow Change': simplify(kpis['Free Cash Flow']['delta'])
    ,'Basic EPS': kpis['Basic EPS']['current']
    ,'Basic EPS Change': kpis['Basic EPS']['delta']
    ,'Net Profit Margin': kpis['Net Profit Margin']['current']
    ,'Net Profit Margin Change': kpis['Net Profit Margin']['delta']
    ,'ROA': kpis['ROA']['current']
    ,'ROA Change': kpis['ROA']['delta']
    ,'ROE': kpis['ROE']['current']
    ,'ROE Change': kpis['ROE']['delta']
    ,'ROCE': kpis['ROCE']['current']
    ,'ROCE Change': kpis['ROCE']['delta']
    ,'Current Ratio': kpis['Current Ratio']['current']
    ,'Current Ratio Change': kpis['Current Ratio']['delta']
    ,'DE Ratio': kpis['DE Ratio']['current']
    ,'DE Ratio Change': kpis['DE Ratio']['delta']}

    return pd.DataFrame(df,index=[Name])

#Single company technicals
def get_technicals_single(Name,data): 
    
    data = data[data["Name"]==Name]
    
    def trend(a, b):
        return "Bullish" if a > b else "Bearish"

    df ={
    "ADX":(round(data['Latest ADX'].values[0],2)),
    "RSI (14)":(round(data['Latest rsi'].values[0],2)),
    "CCI (10)":(round(data['Latest CCI_10'].values[0],2)),
    "CCI (40)":(round(data['Latest CCI_40'].values[0],2)),
    "MFI (14)":(round(data['Latest MFI_14'].values[0],2)),
    "MACD" : (round(data['Latest macd'].values[0],2)),
    "VWAP" : (round(data['Latest VWAP'].values[0],2)),
    "Willaimson %R": (round(data['Latest %R'].values[0],2)),
    "Price v. SMA (5)" : trend(data['Latest Close'].values[0], data['Latest sma_5'].values[0]),
    "Price v. SMA (20)" : trend(data['Latest Close'].values[0], data['Latest sma_20'].values[0]),
    "Price v. SMA (50)" : trend(data['Latest Close'].values[0], data['Latest sma_50'].values[0]),
    "Price v. SMA (100)" : trend(data['Latest Close'].values[0], data['Latest sma_100'].values[0]),
    "SMA(5) v. SMA (20)" : trend(data['Latest sma_5'].values[0], data['Latest sma_20'].values[0]),
    "SMA(20) v. SMA (50)" : trend(data['Latest sma_20'].values[0], data['Latest sma_50'].values[0]),
    "SMA(50) v. SMA (100)" : trend(data['Latest sma_50'].values[0], data['Latest sma_100'].values[0]),   
    }

    return pd.DataFrame(df,index=[Name])

######################################################################################################################################
# Glossary Functions
######################################################################################################################################

def get_indicatordetails():
    # Define the indicators and their details
    indicators = {
        "CCI": {
            "full_name": "Commodity Channel Index (CCI)",
            "description": "The Commodity Channel Index (CCI) is a tool that helps investors understand the momentum of an investment. Think of it as a thermometer for stocks or other investments. It tells you if an investment might be 'too hot' (overbought) or 'too cold' (oversold).",
            "how_it_works": "The CCI measures how the current price of an investment compares to its average price over a certain period. If the CCI is above zero, it means the price is higher than its average. If it's below zero, the price is lower than its average.",
            "usefulness": "The CCI can give hints about potential new trends. For example, if the CCI moves from a low value to above 100, it might mean that the price is starting a new upward trend. On the other hand, if the CCI drops below -100, a downward trend might be starting. This can help investors decide when to buy or sell.",
            "calculation": r'''
    \begin{align*}
    \text{CCI} & = \frac{\text{Typical Price} - \text{20-period SMA of TP}}{\text{Mean Deviation} \times 0.015} \\
    \text{Where:} \\
    \text{Typical Price (TP)} & = \frac{\text{High} + \text{Low} + \text{Close}}{3}
    \end{align*}
    '''
        },
        "VPT": {
            "full_name": "Volume Price Trend (VPT)",
            "description": "Volume Price Trend (VPT) is a technical analysis indicator that combines price and volume data. It helps in determining the strength of price movements.",
            "how_it_works": "VPT is similar to the On-Balance Volume (OBV) but incorporates price changes. It adds or subtracts a multiple of the percentage change in share price trend and current volume, depending upon the movement of the price.",
            "usefulness": "VPT can provide insights into the strength of a price trend, help in confirming price trends, and indicate potential reversals when there's a divergence between VPT and price.",
            "calculation": r'''
    \begin{align*}
    \text{VPT} & = \text{Previous VPT} + \text{Volume} \times (\text{Close}_{\text{today}} - \text{Close}_{\text{yesterday}})
    \end{align*}
    '''
        },
        "ADX": {
            "full_name": "Average Directional Index (ADX)",
            "description": "The Average Directional Index (ADX) is a technical indicator that measures the strength of a trend. It doesn't indicate the direction of the trend, just its strength.",
            "how_it_works": "ADX ranges between 0 to 100. Generally, ADX readings below 20 indicate a weak trend or a non-trending market, while readings above 20 indicate a strong trend.",
            "usefulness": "ADX can help traders identify the strongest and most profitable trends, provide insights into whether a trend is strengthening or weakening, and assist in filtering out price consolidations.",
            "calculation": r"ADX = \frac{ \text{Moving Average of DX} }{ \text{Period} } \text{ where DX is the difference between +DI and -DI}"
        },
            "RSI": {
            "full_name": "Relative Strength Index (RSI)",
            "description": "RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions in a traded security.",
            "how_it_works": "RSI compares the magnitude of recent gains to recent losses to determine overbought or oversold conditions.",
            "usefulness": "RSI can help traders identify potential buy or sell opportunities, recognize potential price reversals, and gauge the strength of a trend.",
            "calculation": r"RSI = 100 - \frac{100}{1 + RS} \text{ where } RS \text{ is the average of } n \text{ days' up closes divided by the average of } n \text{ days' down closes.}"
        },
        "EMA": {
            "full_name": "Exponential Moving Average (EMA)",
            "description": "EMA is a type of moving average that gives more weight to recent prices, making it more responsive to new information.",
            "how_it_works": "EMA reacts faster to price changes compared to a Simple Moving Average (SMA).",
            "usefulness": "EMA can help traders identify trend direction, determine support and resistance levels, and recognize potential entry and exit points.",
            "calculation": r"EMA = (Close - Previous EMA) \times (2 / (Selected Time Period + 1)) + Previous EMA"
        },
        "SMA": {
            "full_name": "Simple Moving Average (SMA)",
            "description": "SMA is an arithmetic moving average calculated by adding recent closing prices and then dividing that by the number of time periods.",
            "how_it_works": "SMA provides a smoothed line that tracks the price over a given period.",
            "usefulness": "SMA can help traders identify trend direction, recognize potential price reversals, and determine support and resistance levels.",
            "calculation": r"SMA = \frac{Sum \ of \ Closing \ Prices}{Number \ of \ Periods}"
        },
        "VWAP": {
            "full_name": "Volume Weighted Average Price (VWAP)",
            "description": "VWAP is the average price a security has traded at throughout the day, based on both volume and price.",
            "how_it_works": "VWAP provides a benchmark that gives an idea of the average price at which investors have bought a security over a given time frame.",
            "usefulness": "VWAP can help traders determine the market direction, recognize fair value for a security, and identify potential buy or sell signals.",
            "calculation": r"VWAP = \frac{\sum (Price \times Volume)}{\sum Volume}"
        },
        "OBV": {
            "full_name": "On-Balance Volume (OBV)",
            "description": "OBV is a momentum indicator that uses volume flow to predict changes in stock price.",
            "how_it_works": "OBV measures buying and selling pressure by adding volume on up days and subtracting volume on down days.",
            "usefulness": "OBV can help traders identify potential price reversals, confirm price trends, and recognize accumulation or distribution phases.",
            "calculation": r'''
    \begin{align*}
    \text{If } \text{Close}_{\text{today}} > \text{Close}_{\text{yesterday}} & : \\
    \text{OBV}_{\text{today}} & = \text{OBV}_{\text{yesterday}} + \text{Volume}_{\text{today}} \\
    \text{If } \text{Close}_{\text{today}} < \text{Close}_{\text{yesterday}} & : \\
    \text{OBV}_{\text{today}} & = \text{OBV}_{\text{yesterday}} - \text{Volume}_{\text{today}}
    \end{align*}
    '''
        },
        "Williams %R": {
            "full_name": "Williams %R",
            "description": "Williams %R, also known as the Williams Percent Range, is a type of momentum indicator that moves between 0 and -100 and measures overbought and oversold levels.",
            "how_it_works": "The Williams %R oscillates between 0 to -100. Readings from 0 to -20 are considered overbought, and readings from -80 to -100 are considered oversold.",
            "usefulness": "Williams %R can help traders identify potential price reversals, recognize overbought or oversold conditions, and confirm momentum shifts.",
            "calculation": r"Williams \ \%R = \frac{Highest \ High \ - \ Close}{Highest \ High \ - \ Lowest \ Low} \times -100"
        },
            "MACD": {
            "full_name": "Moving Average Convergence Divergence (MACD)",
            "description": "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.",
            "how_it_works": "MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of the MACD, called the 'signal line,' is then plotted on top of the MACD line.",
            "usefulness": "MACD can help traders identify potential buy or sell opportunities around crossovers of the MACD line and the signal line, recognize potential overbought or oversold conditions, and confirm the strength of a trend.",
            "calculation": r"MACD = 12-Period \ EMA - 26-Period \ EMA \ and \ Signal \ Line = 9-Period \ EMA \ of \ MACD"
        },
        "Bollinger Bands": {
            "full_name": "Bollinger Bands",
            "description": "Bollinger Bands consist of a middle band being an N-period simple moving average (SMA), an upper band at K times an N-period standard deviation above the middle band, and a lower band at K times an N-period standard deviation below the middle band.",
            "how_it_works": "Bollinger Bands are able to adapt to volatility in the price of a stock. A band squeeze denotes a period of low volatility and is considered by traders to be a potential indicator of future increased volatility.",
            "usefulness": "Bollinger Bands can help traders identify periods of high or low volatility, recognize potential buy or sell opportunities, and determine overbought or oversold conditions.",
            "calculation": r"Middle \ Band = 20-day \ SMA, \ Upper \ Band = 20-day \ SMA + (20-day \ standard \ deviation \times 2), \ Lower \ Band = 20-day \ SMA - (20-day \ standard \ deviation \times 2)"
        },
        "Standard Deviation": {
            "full_name": "Standard Deviation",
            "description": "Standard Deviation is a statistical measure of volatility. It represents how spread out the numbers are in a data set.",
            "how_it_works": "In finance, standard deviation is used to measure price volatility and can help gauge the risk associated with a particular investment.",
            "usefulness": "Standard Deviation can help traders and investors understand the volatility of an investment, gauge the risk associated with a particular security or portfolio, and determine the dispersion of returns.",
            "calculation": r"Standard \ Deviation = \sqrt{\frac{\sum (X - \text{Mean})^2}{N}}"
        },
        "Net Income": {
            "full_name": "Net Income",
            "description": "Net Income represents a company's total earnings or profit.",
            "how_it_works": "It's calculated by subtracting total expenses from total revenues. It provides a clear picture of the overall profitability of a company over a specific period of time.",
            "usefulness": "Net Income is a key metric to assess a company's profitability and is often used by investors to compare the profitability of companies within the same industry.",
            "calculation": r"\text{Net Income} = \text{Total Revenues} - \text{Total Expenses}"
        },
            "ROA": {
            "full_name": "Return on Assets (ROA)",
            "description": "ROA is a measure of how effectively a company's assets are being used to generate profits.",
            "how_it_works": "It's calculated by dividing net income by total assets. This ratio gives an idea of how efficiently the company is converting its investment in assets into net income.",
            "usefulness": "ROA is useful for comparing the profitability of companies in the same industry and for understanding if a company is generating enough profit from its assets.",
            "calculation": r"\text{ROA} = \frac{\text{Net Income}}{\text{Total Assets}}"
        },

        "Debt": {
            "full_name": "Debt",
            "description": "Debt refers to the amount of money borrowed by a company and due for repayment.",
            "how_it_works": "Companies can raise capital either through equity (like issuing shares) or through debt (like taking loans).",
            "usefulness": "Analyzing a company's debt levels helps investors understand its financial health and its ability to meet its financial obligations."
        },
            "ROE": {
            "full_name": "Return on Equity (ROE)",
            "description": "ROE measures a company's profitability by revealing how much profit a company generates with the money shareholders have invested.",
            "how_it_works": "It's calculated by dividing net income by shareholder's equity. This ratio indicates how well the company is generating earnings from its equity investments.",
            "usefulness": "ROE is useful for comparing the profitability of companies in the same sector and understanding the efficiency of generating profits from shareholders' equity.",
            "calculation": r"\text{ROE} = \frac{\text{Net Income}}{\text{Shareholder's Equity}}"
        },

        "Free Cash Flow": {
            "full_name": "Free Cash Flow",
            "description": "Free Cash Flow (FCF) represents the cash a company generates after accounting for cash outflows to support operations and maintain its capital assets.",
            "how_it_works": "It's the cash produced by the company's normal business operations after deducting capital expenditures.",
            "usefulness": "FCF is a key indicator of a company's financial flexibility and its ability to generate cash. It's often used by investors to assess the quality of a company's earnings.",
            "calculation": r"\text{FCF} = \text{Operating Cash Flow} - \text{Capital Expenditures}"
        },

        "ROCE": {
            "full_name": "Return on Capital Employed (ROCE)",
            "description": "ROCE is a financial metric that determines how efficiently a company is generating profits from its capital.",
            "how_it_works": "It's calculated by dividing Earnings Before Interest and Tax (EBIT) by capital employed. It gives an idea of how well the company is using its capital to generate profits.",
            "usefulness": "ROCE is useful for comparing the profitability and efficiency of companies in the same sector.",
            "calculation": r"\text{ROCE} = \frac{\text{Earnings Before Interest and Tax (EBIT)}}{\text{Capital Employed}}"
        },

        "Basic EPS": {
            "full_name": "Basic Earnings Per Share (EPS)",
            "description": "EPS measures the amount of net income earned per share of stock outstanding.",
            "how_it_works": "It's calculated by dividing the net income by the average number of shares outstanding during a period.",
            "usefulness": "EPS is a key metric used by investors to assess a company's profitability on a per-share basis.",
            "calculation": r"\text{EPS} = \frac{\text{Net Income} - \text{Dividends on Preferred Stock}}{\text{Average Outstanding Shares}}"
        },

        "Current Ratio": {
            "full_name": "Current Ratio",
            "description": "The current ratio is a liquidity ratio that measures a company's ability to cover its short-term obligations with its short-term assets.",
            "how_it_works": "It's calculated by dividing current assets by current liabilities. A ratio above 1 indicates that the company has more assets than liabilities.",
            "usefulness": "The current ratio helps investors assess a company's short-term financial health and its ability to pay off its short-term liabilities with its short-term assets.",
            "calculation": r"\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}"
        },

        "Net Profit Margin": {
            "full_name": "Net Profit Margin",
            "description": "Net Profit Margin is a profitability ratio that shows how much of each dollar of revenues is kept as net profit.",
            "how_it_works": "It's calculated by dividing net profit by total revenue and then multiplying by 100 to get a percentage.",
            "usefulness": "The net profit margin helps investors assess how effectively a company is converting its revenues into actual profit.",
            "calculation": r"\text{Net Profit Margin} = \frac{\text{Net Profit}}{\text{Total Revenue}} \times 100\%"
        },

        "DE Ratio": {
            "full_name": "Debt-to-Equity (DE) Ratio",
            "description": "The DE ratio is a measure of a company's financial leverage, indicating the proportion of equity and debt a company is using to finance its assets.",
            "how_it_works": "It's calculated by dividing total liabilities by shareholder's equity. A high DE ratio indicates that a company may have too much debt.",
            "usefulness": "The DE ratio helps investors understand the risk associated with a company's debt levels.",
            "calculation": r"\text{DE Ratio} = \frac{\text{Total Liabilities}}{\text{Shareholder's Equity}}"
        },
        "MFI" : {
            "full_name": "Money Flow Index (MFI)",
            "description": "The Money Flow Index (MFI) is a momentum indicator that measures the inflow and outflow of money into an asset over a specific period of time.",
            "how_it_works": "MFI takes both price and volume into consideration. A value above 80 is generally considered overbought, while a value below 20 is considered oversold.",
            "usefulness": "MFI can be used to identify potential price reversals and validate price movements. It's also useful for spotting divergences between price and volume momentum.",
            "calculation": r"\text{MFI} = 100 - \left( \frac{100}{1 + \text{Money Ratio}} \right) \text{ where Money Ratio} = \frac{\text{Positive Money Flow}}{\text{Negative Money Flow}}"
        },
        "Pivot Points": {
            "full_name": "Pivot Points",
            "description": "Pivot Points are horizontal support and resistance levels used to determine potential price movements.",
            "how_it_works": "Pivot Points are used as predictive indicators. If the market opens above the pivot point, then the bias for the day is bullish, and if it opens below the pivot point, the bias is bearish.",
            "usefulness": "They provide traders with levels to place stop losses, take profits, or identify entry points.",
            "calculation": r"""
            \begin{align*}
            \text{Pivot Point (PP)} & : \frac{\text{High} + \text{Low} + \text{Close}}{3} \\
            \text{Resistance 1 (R1)} & : 2 \times \text{PP} - \text{Low} \\
            \text{Resistance 2 (R2)} & : \text{PP} + \text{High} - \text{Low} \\
            \text{Resistance 3 (R3)} & : \text{High} + 2(\text{PP} - \text{Low}) \\
            \text{Support 1 (S1)} & : 2 \times \text{PP} - \text{High} \\
            \text{Support 2 (S2)} & : \text{PP} - \text{High} + \text{Low} \\
            \text{Support 3 (S3)} & : \text{Low} - 2(\text{High} - \text{PP})
            \end{align*}
            """
        }
        # ... add the rest of the indicators similarly
    }
    
    return indicators