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
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def retrieve_api(symbol):
    if symbol.upper()[-3:]==".NS":
        symbol = symbol.upper()
    else:
        symbol = symbol.upper()+".NS"
    return(yf.Ticker(symbol))

def get_hist_data(result):
    date_today = date.today()
    historical = result.history(end=date_today, start="2020-01-01", period = "1d")
    '''
    creating a new column of date and trading day compared to today i.e. lets say today is 5 JUly i.e. day 1, then 4 July will be 2 and so on. 
    We will also drop days on which no volume was traded as they represnt market holidays.
    As the columns high and low are not needed, they will be dropped as well.
    '''
    #removing days with zero trading
    historical = historical[historical['Volume']!=0]
    
    #sortingthe df and adding trading day column
    historical = historical.sort_values('Date', ascending=False).reset_index().reset_index()
    historical = historical.rename(columns={'index': 'Trading Day'})
    historical['Trading Day'] = historical['Trading Day']+1
    
    #creating a date only column
    date_only = []
    for i in historical['Date']:
        x = str(pd.to_datetime(i).year).zfill(2)+"-"+str(pd.to_datetime(i).month).zfill(2)+"-"+str(pd.to_datetime(i).day).zfill(2)
        date_only.append(np.datetime64(x))
    historical['date_only'] = date_only
    
    #Keeping only values in the last 2 years
    historical_sample = historical[historical['date_only'] >historical['date_only'][0]-np.timedelta64(2, "Y")]
    
    #This code is udes for filtering based on dusrations
    #get_hist_data(m)[get_hist_data(m)['date_only'] > get_hist_data(m)['date_only'][0]-np.timedelta64(1, "M")]
    return(historical_sample)

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

#generate graph based on df and inputs
def generate_charts(historical_sample, selected_ma, holiday_list):
    candlesticks = go.Candlestick(
                        x=historical_sample['date_only'],
                        open=historical_sample['Open'],
                        high=historical_sample['High'],
                        low=historical_sample['Low'],
                        close=historical_sample['Close'],
                        showlegend=False,
                        name= 'Price'
                    )

    volume_bars = go.Bar(
                    x=historical_sample['date_only'],
                    y=historical_sample['Volume'],
                    name = 'Volume',
                    showlegend=False,
                    marker={
                        "color": "rgba(128,128,128,0.5)",
                    }
                )
    ma_traces = {
        'ema_5_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_5'], mode='lines', name = 'EMA 5 days'),
        'sma_5_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_5'], mode='lines', name = 'SMA 5 days'),
        'ema_10_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_10'], mode='lines', name = 'EMA 10 days'),
        'sma_10_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_10'], mode='lines', name = 'SMA 10 days'),
        'ema_15_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_15'], mode='lines', name = 'EMA 15 days'),
        'sma_15_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_15'], mode='lines', name = 'SMA 15 days'),
        'ema_20_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_20'], mode='lines', name = 'EMA 20 days'),
        'sma_20_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_20'], mode='lines', name = 'SMA 20 days'),
        'ema_25_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_25'], mode='lines', name = 'EMA 25 days'),
        'sma_25_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_25'], mode='lines', name = 'SMA 25 days'),
        'ema_30_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_30'], mode='lines', name = 'EMA 30 days'),
        'sma_30_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_30'], mode='lines', name = 'SMA 30 days'),
        'ema_40_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_40'], mode='lines', name = 'EMA 40 days'),
        'sma_40_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_40'], mode='lines', name = 'SMA 40 days'),
        'ema_50_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_50'], mode='lines', name = 'EMA 50 days'),
        'sma_50_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_50'], mode='lines', name = 'SMA 50 days'),
        'ema_75_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_75'], mode='lines', name = 'EMA 75 days'),
        'sma_75_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_75'], mode='lines', name = 'SMA 75 days'),
        'ema_100_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_100'], mode='lines', name = 'EMA 100 days'),
        'sma_100_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_100'], mode='lines', name = 'SMA 100 days'),
        'ema_150_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_150'], mode='lines', name = 'EMA 150 days'),
        'sma_150_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_150'], mode='lines', name = 'SMA 150 days'),
        'ema_200_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['ema_200'], mode='lines', name = 'EMA 200 days'),
        'sma_200_trace':go.Scatter(x=historical_sample['date_only'], y=historical_sample['sma_200'], mode='lines', name = 'SMA 200 days')}

    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.03, subplot_titles=('Price', 'Volume'), 
                   row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(candlesticks,row=1, col=1)

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(volume_bars, row=2, col=1)

    for i in selected_ma:    
        fig.add_trace(ma_traces[i],row=1, col=1)

    # Do not show OHLC's rangeslider plot 
    #fig.update_yaxes(title="Price", showgrid=True, minor=dict(showgrid=True))
    fig.update_yaxes(title="Volume", showgrid=False, minor=dict(showgrid=False))
    fig.update_xaxes(
            rangeslider_visible=False,showgrid=True,
            rangebreaks=[
                 dict(values=holiday_list),
                dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            ]
        )
    
    #fig.update_xaxes(major=dict(showgrid=True))
    #fig.update_yaxes(major=dict(showgrid=True))
    
    return fig

def ohlc(hist,df):
    fig = go.Figure(data=go.Ohlc(x=['Day','52 Week'],
                    open=[hist[hist['Trading Day']==1].Open.values[0],hist[hist['Trading Day']==1].Open.values[0]],
                    high=[hist[hist['Trading Day']==1].High.values[0],df['52 Week High'].values[0]],
                    low=[hist[hist['Trading Day']==1].Low.values[0],df['52 Week Low'].values[0]],
                    close=[hist[hist['Trading Day']==1].Close.values[0],hist[hist['Trading Day']==1].Close.values[0]]))
    fig.update_xaxes(
            rangeslider_visible=False)
    fig.update_traces(tickwidth=0.5)
    return fig
