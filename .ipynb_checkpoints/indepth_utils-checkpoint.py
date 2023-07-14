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

def generate_charts(historical_sample, selected_ma, holiday_list, to_show):
    candlesticks = go.Candlestick(
                        x=historical_sample['date_only'],
                        open=historical_sample['Open'],
                        high=historical_sample['High'],
                        low=historical_sample['Low'],
                        close=historical_sample['Close'],
                        showlegend=False,
                        name= 'Price',
                        legendgroup = '1'
                    )

    volume_bars = go.Bar(
                    x=historical_sample['date_only'],
                    y=historical_sample['Volume'],
                    name = 'Volume',
                    showlegend=False,
                    legendgroup = '2',
                    marker={
                        "color": "rgba(128,128,128,0.5)"})
    
    macd_line = go.Scatter(x=historical_sample['date_only'],
                           y=historical_sample["macd"], 
                           mode='lines', 
                           legendgroup = '2',
                           name = "MACD")
    
    macd_signal_line = go.Scatter(x=historical_sample['date_only'],
                           y=historical_sample["macd_ema_9"], 
                           mode='lines', 
                            legendgroup = '2',
                           name = "Signal")

    ma_traces = {}
    for i in [5,10,15,20,25,30,40,50,75,100,150,200]:
        for j in ['ema','sma']:
            name = j+"_"+str(i)+"_trace"
            plot = go.Scatter(x=historical_sample['date_only'], y=historical_sample[j+"_"+str(i)], mode='lines', name = j.upper()+" "+str(i)+" days",legendgroup = '1')
            ma_traces[name] = plot
    
    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.05,  
                   row_width=[0.3, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(candlesticks,row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    legend_gap = 440
    for i in selected_ma:    
        fig.add_trace(ma_traces[i],row=1, col=1)
        legend_gap=legend_gap-20
        
    if to_show.upper() =="VOLUME":
        # Bar trace for volumes on 2nd row without legend
        fig.add_trace(volume_bars, row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        #adding mACD
        fig.add_trace(macd_line, row=2, col=1)
        fig.add_trace(macd_signal_line, row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
    fig.update_yaxes(showgrid=True, minor=dict(showgrid=False),showline=True, linewidth=2)
    fig.update_xaxes(
            rangeslider_visible=False,showgrid=True,showline=True, linewidth=2,
            rangebreaks=[dict(values=holiday_list),dict(bounds=["sat", "mon"])])  # hide weekends, eg. hide sat to before mon

    fig.update_layout(autosize=False,width=1200,height=800,legend_tracegroupgap = legend_gap,template="plotly_white")
    if legend_gap==440:
        fig.update_layout(legend={"yanchor":"top","y": 0.3})
        
    return fig

