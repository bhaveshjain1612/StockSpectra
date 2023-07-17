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

#52 week data
def week_52(data):
    end = data['date_only'][0]
    start = end-np.timedelta64(52, "W")
    high = data[data['date_only']>=start]['High'].max()
    low = data[data['date_only']>=start]['Low'].min()
    high_perc = (high - data.Close.values[0])*100 / high
    low_perc = (low - data.Close.values[0])*100 / -low
    return {'52 Week High':high, '52 Week Low':low,'Diff between high':high_perc, 'Diff between low':low_perc}

#generate ADX signals
def ADX(data: pd.DataFrame, period: int):
    """
    Computes the ADX indicator.
    """
    
    df = data.copy()
    df = df.sort_values('Trading Day', ascending=False)
    
    alpha = 1/period

    # TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']
    
    df = df.sort_values('Trading Day', ascending=True)

    return df

#generate charts
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
    
    ADX_line = go.Scatter(x=historical_sample['date_only'],
                           y=historical_sample["ADX"], 
                           mode='lines', 
                            legendgroup = '2',
                           name = "ADX")
    
    RSI_line = go.Scatter(x=historical_sample['date_only'],
                           y=historical_sample["rsi"], 
                           mode='lines', 
                            legendgroup = '2',
                           name = "RSI")

    ma_traces = {}
    for i in [5,10,15,20,25,30,40,50,75,100,150,200]:
        for j in ['ema','sma']:
            name = j+"_"+str(i)+"_trace"
            plot = go.Scatter(x=historical_sample['date_only'], y=historical_sample[j+"_"+str(i)], mode='lines', name = j.upper()+" "+str(i)+" days",legendgroup = '1')
            ma_traces[name] = plot
    
    # Create subplots and mention plot grid size
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.1,  
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
    elif to_show.upper() =="ADX":
        # Bar trace for volumes on 2nd row without legend
        fig.add_trace(ADX_line, row=2, col=1)
        fig.update_yaxes(title_text="ADX", row=2, col=1)
    elif to_show.upper() =="RSI":
        # Bar trace for volumes on 2nd row without legend
        fig.add_trace(RSI_line, row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
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

#RSI
def RSI(df,window_length):
    diff = []
    for i in range(df.shape[0]-1):
        diff.append(df.Close[i] - df.Close[i+1])
    diff.append(np.nan)
    df['diff'] = diff
    df = df.sort_values('Trading Day', ascending=False)
    df['gain'] = df['diff'].clip(lower=0).round(2)
    df['loss'] = df['diff'].clip(upper=0).abs().round(2)
    df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
    df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
    #avg_gain
    for i, row in enumerate(df['avg_gain'].iloc[window_length+1:]):
        df['avg_gain'].iloc[i + window_length + 1] =\
            (df['avg_gain'].iloc[i + window_length] *
             (window_length - 1) +
             df['gain'].iloc[i + window_length + 1])/ window_length
    #abg_loss
    for i, row in enumerate(df['avg_loss'].iloc[window_length+1:]):
        df['avg_loss'].iloc[i + window_length + 1] =\
            (df['avg_loss'].iloc[i + window_length] *
             (window_length - 1) +
             df['loss'].iloc[i + window_length + 1])/ window_length
    df['rs'] = df['avg_gain'] / df['avg_loss']
    # Calculate RSI
    df['rsi'] = 100 - (100 / (1.0 + df['rs']))
    df = df.drop(['gain','loss','rs','avg_gain','avg_loss','diff'],axis=1)
    df = df.sort_values('Trading Day', ascending=True)
    return df

####################################
### FInancials
####################################


def calc_KPIs(income_stmt,balance_sheet,cash_flow):
    
    income_stmt = income_stmt[income_stmt['year']!='sheet']
    balance_sheet = balance_sheet[balance_sheet['year']!='sheet']
    cash_flow = cash_flow[cash_flow['year']!='sheet']
        
    kpis = {}
    #ROE
    kpis['ROE'] = {'desc' : 'Efficiency of Equity utilisation'}
    if balance_sheet['Stockholders Equity'][0] and income_stmt['Net Income'][0]:
        kpis['ROE']['current'] = balance_sheet['Stockholders Equity'][0]/income_stmt['Net Income'][0]
    else:
        kpis['ROE']['current'] = None
    if balance_sheet['Stockholders Equity'][1] and income_stmt['Net Income'][1]:
        kpis['ROE']['previous'] = balance_sheet['Stockholders Equity'][1]/income_stmt['Net Income'][1]
    else:
        kpis['ROE']['previous'] = None
    if kpis['ROE']['previous'] and kpis['ROE']['current']:
        kpis['ROE']['delta'] = kpis['ROE']['current'] - kpis['ROE']['previous']
    else:
        kpis['ROE']['delta'] = None
        
    #ROA
    kpis['ROA'] = {'desc' : 'Efficiency of Assets utilisation'}
    if balance_sheet['Total Assets'][0] and income_stmt['Net Income'][0]:
        kpis['ROA']['current'] = income_stmt['Net Income'][0]/balance_sheet['Total Assets'][0]
    else:
        kpis['ROA']['current'] = None
    if balance_sheet['Total Assets'][1] and income_stmt['Net Income'][1]:
        kpis['ROA']['previous'] = income_stmt['Net Income'][1]/balance_sheet['Total Assets'][1]
    else:
        kpis['ROA']['previous'] = None
    if kpis['ROA']['previous'] and kpis['ROA']['current']:
        kpis['ROA']['delta'] = kpis['ROA']['current'] - kpis['ROA']['previous']
    else:
        kpis['ROA']['delta'] = None
        
    #Current Ratio
    kpis['Current Ratio'] = {'desc' : 'Ability to pay short term liabilities'}
    if balance_sheet['Current Assets'][0] and balance_sheet['Current Liabilities'][0]:
        kpis['Current Ratio']['current'] = balance_sheet['Current Assets'][0]/balance_sheet['Current Liabilities'][0]
    else:
        kpis['Current Ratio']['current'] = None
    if balance_sheet['Current Assets'][1] and balance_sheet['Current Liabilities'][1]:
        kpis['Current Ratio']['previous'] = balance_sheet['Current Assets'][1]/balance_sheet['Current Liabilities'][1]
    else:
        kpis['Current Ratio']['previous'] = None
    if kpis['Current Ratio']['previous'] and kpis['Current Ratio']['current']:
        kpis['Current Ratio']['delta'] = kpis['Current Ratio']['current'] - kpis['Current Ratio']['previous']
    else:
        kpis['Current Ratio']['delta'] = None
        
    #Gross Margin
    kpis['Net Profit Margin'] = {'desc' : 'Profitability of a company'}
    if income_stmt['Total Revenue'][0] and income_stmt['Net Income'][0]:
        kpis['Net Profit Margin']['current'] = income_stmt['Net Income'][0]/income_stmt['Total Revenue'][0]
    else:
        kpis['Net Profit Margin']['current'] = None
    if income_stmt['Total Revenue'][1] and income_stmt['Net Income'][1]:
        kpis['Net Profit Margin']['previous'] = income_stmt['Net Income'][1]/income_stmt['Total Revenue'][1]
    else:
        kpis['Net Profit Margin']['previous'] = None
    if kpis['Net Profit Margin']['previous'] and kpis['Net Profit Margin']['current']:
        kpis['Net Profit Margin']['delta'] = kpis['Net Profit Margin']['current'] - kpis['Net Profit Margin']['previous']
    else:
        kpis['Net Profit Margin']['delta'] = None
        
    #Debt to equity ratio
    kpis['DE Ratio'] = {'desc' : 'Total debt comapred to equity'}
    if balance_sheet['Stockholders Equity'][0] and balance_sheet['Total Debt'][0]:
        kpis['DE Ratio']['current'] = balance_sheet['Total Debt'][0]/balance_sheet['Stockholders Equity'][0]
    else:
        kpis['DE Ratio']['current'] = None
    if balance_sheet['Stockholders Equity'][1] and balance_sheet['Total Debt'][1]:
        kpis['DE Ratio']['previous'] = balance_sheet['Total Debt'][1]/balance_sheet['Stockholders Equity'][1]
    else:
        kpis['DE Ratio']['previous'] = None
    if kpis['DE Ratio']['previous'] and kpis['DE Ratio']['current']:
        kpis['DE Ratio']['delta'] = kpis['DE Ratio']['current'] - kpis['DE Ratio']['previous']
    else:
        kpis['DE Ratio']['delta'] = None
        
    #Net Income
    kpis['Net Income'] = {'desc' : 'Net Income of the company'}
    if income_stmt['Net Income'][0]:
        kpis['Net Income']['current'] = income_stmt['Net Income'][0]
    else:
        kpis['Net Income']['current'] = None
    if income_stmt['Net Income'][1]:
        kpis['Net Income']['previous'] = income_stmt['Net Income'][1]
    else:
        kpis['Net Income']['previous'] = None
    if kpis['Net Income']['previous'] and kpis['Net Income']['current']:
        kpis['Net Income']['delta'] = kpis['Net Income']['current'] - kpis['Net Income']['previous']
    else:
        kpis['Net Income']['delta'] = None
        
    #Free Cash Flow
    kpis['Free Cash Flow'] = {'desc' : 'In Hand cash flow'}
    if cash_flow['Free Cash Flow'][0]:
        kpis['Free Cash Flow']['current'] = cash_flow['Free Cash Flow'][0]
    else:
        kpis['Free Cash Flow']['current'] = None
    if cash_flow['Free Cash Flow'][1]:
        kpis['Free Cash Flow']['previous'] = cash_flow['Free Cash Flow'][1]
    else:
        kpis['Free Cash Flow']['previous'] = None
    if kpis['Free Cash Flow']['previous'] and kpis['Free Cash Flow']['current']:
        kpis['Free Cash Flow']['delta'] = kpis['Free Cash Flow']['current'] - kpis['Free Cash Flow']['previous']
    else:
        kpis['Free Cash Flow']['delta'] = None
        
    #Total Debt
    kpis['Debt'] = {'desc' : 'Total debt of the company'}
    if balance_sheet['Total Debt'][0]:
        kpis['Debt']['current'] = balance_sheet['Total Debt'][0]
    else:
        kpis['Debt']['current'] = None
    if balance_sheet['Total Debt'][1]:
        kpis['Debt']['previous'] = balance_sheet['Total Debt'][1]
    else:
        kpis['Debt']['previous'] = None
    if kpis['Debt']['previous'] and kpis['ROE']['current']:
        kpis['Debt']['delta'] = kpis['Debt']['current'] - kpis['Debt']['previous']
    else:
        kpis['Debt']['delta'] = None
        
    #Basic EPS
    kpis['Basic EPS'] = {'desc' : 'Earnings of the company per share'}
    if income_stmt['Net Income'][0]:
        kpis['Basic EPS']['current'] = income_stmt['Basic EPS'][0]
    else:
        kpis['Net Income']['current'] = None
    if income_stmt['Net Income'][1]:
        kpis['Basic EPS']['previous'] = income_stmt['Basic EPS'][1]
    else:
        kpis['Basic EPS']['previous'] = None
    if kpis['Basic EPS']['previous'] and kpis['Basic EPS']['current']:
        kpis['Basic EPS']['delta'] = kpis['Basic EPS']['current'] - kpis['Basic EPS']['previous']
    else:
        kpis['Basic EPS']['delta'] = None
        
    #ROCE
    kpis['ROCE'] = {'desc':"Utilization of capital employed"}
    if income_stmt['EBIT'][0] and balance_sheet['Total Assets'][0] and balance_sheet['Current Liabilities'][0]:
        kpis['ROCE']['current'] = income_stmt['EBIT'][0] / (balance_sheet['Total Assets'][0]-balance_sheet['Current Liabilities'][0])
    else:
        kpis['ROCE']['current'] = None
    if income_stmt['EBIT'][1] and balance_sheet['Total Assets'][1] and balance_sheet['Current Liabilities'][1]:
        kpis['ROCE']['previous'] = income_stmt['EBIT'][1] / (balance_sheet['Total Assets'][1]-balance_sheet['Current Liabilities'][1])
    else:
        kpis['ROCE']['previous'] = None
    if kpis['ROCE']['previous'] and kpis['ROCE']['current']:
        kpis['ROCE']['delta'] = kpis['ROCE']['current'] - kpis['ROCE']['previous']
    else:
        kpis['ROCE']['delta'] = None
        
        
    return kpis