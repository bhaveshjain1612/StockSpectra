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

#sort based on filter


#allot scores to comapnies based on financials
def financial_scores(row):
    def fin_test(df,param,score,type):
        try:
            if df[param]> 0:
                score+=type
            else:
                score-=type
        except:
            score += 0
        return score
    
    score = 0
    
    score  = fin_test(row,"ROE",score,1)
    score  = fin_test(row,"ROA",score,1)
    score  = fin_test(row,"Current Ratio",score,1)
    score  = fin_test(row,"Net Profit Margin",score,1)
    score  = fin_test(row,"Net Income",score,1)
    score  = fin_test(row,"Free Cash Flow",score,1)
    score  = fin_test(row,"ROCE",score,1)
    score  = fin_test(row,"Basic EPS",score,1)
    score  = fin_test(row,"P/E ratio",score,1)
    score  = fin_test(row,"DE Ratio",score,-1)
    score  = fin_test(row,"Debt",score,-1)
    
    return score

#allot rank based on scores and overall perfomrmance of all companies
def finrank(df):
    u = df.finscore.describe()['75%']
    l = df.finscore.describe()['25%']
    
    r = []
    for i in df.index:
        if df.finscore.values[i] <= l:
            r.append('weak')
        elif df.finscore.values[i] > l and df.finscore.values[i] <= u:
            r.append('mid')
        else:
            r.append('strong')
    
    df['finrank'] = r
    
    return df

#allot rank based on scores and overall perfomrmance of all companies
def stkrank(df):
    u = df.stkscore.describe()['75%']
    l = df.stkscore.describe()['25%']
    
    r = []
    for i in df.index:
        if df.stkscore.values[i] <= l:
            r.append('negative')
        elif df.stkscore.values[i] > l and df.stkscore.values[i] <= u:
            r.append('neutral')
        else:
            r.append('positive')
    
    df['stkrank'] = r
    
    return df

#allot scores based on stock performance
def stock_scores(row):
    score = 0
    
    #indicator 1 - macd > signal and macd moving upwards
    try:
        if row['Latest macd'] > row['Latest macd_ema_9'] and row['macd_change_5d']>0:
            score+=1
        else:
            score-=1
    except:
        score+=0
        
    #indicator 2 - rsi should be below 70, shouuld not be increasing if more than 60, should not be decreasing if less than 30
    try:
        if row['Latest rsi']>70 or (row['Latest rsi']>60 and row['rsi_change_5d']>0) or (row['Latest rsi']<30 and row['rsi_change_5d']<0):
            score-=2
        else:
            score+=2
    except:
        score+=0
        
    #indicator 3 - increasing price along with strong trend(ADX)
    try:
        if row['Latest ADX']>25 and row['Close_change_5d']>0 and row['ADX_change_5d']>0:
            score+=1
        else:
            score-=1
    except:
        score+=0  
        
    #indicator 4 - price above ema and ema anove sma
    try:
        if row['Latest Close']>row['Latest ema_5d'] and row['Latest sma_20d']<row['Latest ema_5d']:
            score+=1
        else:
            score-=1
    except:
        score+=0  
        
    #indicator 5 - bollinger bands
    try:
        if row['Latest Close']>row['Latest UpperBand1'] and  row['Latest Close']<row['Latest UpperBand2']:
            score+=1
        else:
            score-=1
    except:
        score+=0  
        
    return score

######################################################################################################################################
# In Depth Functions
######################################################################################################################################

#pie chart for holding composition
def holding_chart(df):
    insider = df['% of Shares Held by All Insider'].values[0]
    institutions = df['% of Shares Held by Institutions'].values[0]
    public = 100-insider-institutions
    

    labels = ["Public", "Insider", "Institutions"]
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
def generate_charts(historical_sample, selected_ma, bollinger_filter, holiday_list, to_show):
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
    
    bollinger_u_2 = go.Scatter(x = historical_sample['date_only'],
                         y = historical_sample['UpperBand2'],
                         name = 'Upper Bollinger (2 STD)',
                               line_color = 'yellow',
                               line = {'dash': 'dash'},
                         opacity = 0.5)
    
    bollinger_l_2 = go.Scatter(x = historical_sample['date_only'],
                         y = historical_sample['LowerBand2'],
                         name = 'Lower Bollinger (2 STD)',
                               line_color = 'yellow',
                               line = {'dash': 'dash'},
                         opacity = 0.5)
    
    bollinger_u_1 = go.Scatter(x = historical_sample['date_only'],
                         y = historical_sample['UpperBand1'],
                         line = {'dash': 'dot'},
                         name = 'Upper Bollinger (1 STD)',
                               line_color='gray',
                         opacity = 0.5)
    
    bollinger_l_1 = go.Scatter(x = historical_sample['date_only'],
                         y = historical_sample['LowerBand1'],
                         line = {'dash': 'dot'},
                         name = 'Lower Bollinger (1 STD)',
                               line_color='gray',
                         opacity = 0.5)
    
    rolling_mean = go.Scatter(x = historical_sample['date_only'],
                         y = historical_sample['RollingMean'],
                         name = 'Rolling Mean',
                         opacity = 0.5)

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
    
    if bollinger_filter==["1 Standard Deviation","2 Standard Deviations"] or bollinger_filter==["2 Standard Deviations","1 Standard Deviation"]:
        fig.add_trace(bollinger_u_2,row=1, col=1)
        fig.add_trace(bollinger_l_2,row=1, col=1)
        fig.add_trace(bollinger_u_1,row=1, col=1)
        fig.add_trace(bollinger_l_1,row=1, col=1)
        fig.add_trace(rolling_mean,row=1, col=1)
    elif bollinger_filter==["1 Standard Deviation"]:
        fig.add_trace(bollinger_u_1,row=1, col=1)
        fig.add_trace(bollinger_l_1,row=1, col=1)
        fig.add_trace(rolling_mean,row=1, col=1)        
    elif bollinger_filter==["2 Standard Deviations"]:
        fig.add_trace(bollinger_u_2,row=1, col=1)
        fig.add_trace(bollinger_l_2,row=1, col=1)
        fig.add_trace(rolling_mean,row=1, col=1)
    

    for i in selected_ma:    
        fig.add_trace(ma_traces[i],row=1, col=1)
        
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

    fig.update_layout(autosize=False,width=1200,height=800,template="plotly_white")

    fig.update_layout(legend={"yanchor":"top","y": 0.9})
    
    return fig

#Calculate Financial KPIs
def calc_KPIs(financials,mode):
    #financials = financials.T
    kpis = {}
    #ROE
    kpis['ROE'] = {'desc' : 'Efficiency of Equity utilisation'}
    try:
        if financials['Stockholders Equity'][0] and financials['Net Income'][0]:
            kpis['ROE']['current'] = financials['Stockholders Equity'][0]/financials['Net Income'][0]
        else:
            kpis['ROE']['current'] = None
        if financials['Stockholders Equity'][1] and financials['Net Income'][1]:
            kpis['ROE']['previous'] = financials['Stockholders Equity'][1]/financials['Net Income'][1]
        else:
            kpis['ROE']['previous'] = None
        if kpis['ROE']['previous'] and kpis['ROE']['current']:
            kpis['ROE']['delta'] = kpis['ROE']['current'] - kpis['ROE']['previous']
        else:
            kpis['ROE']['delta'] = None
    except:
        kpis['ROE']['delta'] = None
        kpis['ROE']['current'] = None
        kpis['ROE']['previous'] = None
        
    #ROA
    kpis['ROA'] = {'desc' : 'Efficiency of Assets utilisation'}
    if financials['Total Assets'][0] and financials['Net Income'][0]:
        kpis['ROA']['current'] = financials['Net Income'][0]/financials['Total Assets'][0]
    else:
        kpis['ROA']['current'] = None
    if financials['Total Assets'][1] and financials['Net Income'][1]:
        kpis['ROA']['previous'] = financials['Net Income'][1]/financials['Total Assets'][1]
    else:
        kpis['ROA']['previous'] = None
    if kpis['ROA']['previous'] and kpis['ROA']['current']:
        kpis['ROA']['delta'] = kpis['ROA']['current'] - kpis['ROA']['previous']
    else:
        kpis['ROA']['delta'] = None
        
     #Current Ratio
    kpis['Current Ratio'] = {'desc' : 'Ability to pay short term liabilities'}
    if financials['Current Assets'][0] and financials['Current Liabilities'][0]:
        kpis['Current Ratio']['current'] = financials['Current Assets'][0]/financials['Current Liabilities'][0]
    else:
        kpis['Current Ratio']['current'] = None
    if financials['Current Assets'][1] and financials['Current Liabilities'][1]:
        kpis['Current Ratio']['previous'] = financials['Current Assets'][1]/financials['Current Liabilities'][1]
    else:
        kpis['Current Ratio']['previous'] = None
    if kpis['Current Ratio']['previous'] and kpis['Current Ratio']['current']:
        kpis['Current Ratio']['delta'] = kpis['Current Ratio']['current'] - kpis['Current Ratio']['previous']
    else:
        kpis['Current Ratio']['delta'] = None
        
    #Gross Margin
    kpis['Net Profit Margin'] = {'desc' : 'Profitability of a company'}
    if financials['Total Revenue'][0] and financials['Net Income'][0]:
        kpis['Net Profit Margin']['current'] = financials['Net Income'][0]/financials['Total Revenue'][0]
    else:
        kpis['Net Profit Margin']['current'] = None
    if financials['Total Revenue'][1] and financials['Net Income'][1]:
        kpis['Net Profit Margin']['previous'] = financials['Net Income'][1]/financials['Total Revenue'][1]
    else:
        kpis['Net Profit Margin']['previous'] = None
    if kpis['Net Profit Margin']['previous'] and kpis['Net Profit Margin']['current']:
        kpis['Net Profit Margin']['delta'] = kpis['Net Profit Margin']['current'] - kpis['Net Profit Margin']['previous']
    else:
        kpis['Net Profit Margin']['delta'] = None
        
    #Debt to equity ratio
    kpis['DE Ratio'] = {'desc' : 'Total debt comapred to equity'}
    try:
        if financials['Stockholders Equity'][0] and financials['Total Debt'][0]:
            kpis['DE Ratio']['current'] = financials['Total Debt'][0]/financials['Stockholders Equity'][0]
        else:
            kpis['DE Ratio']['current'] = None
        if financials['Stockholders Equity'][1] and financials['Total Debt'][1]:
            kpis['DE Ratio']['previous'] = financials['Total Debt'][1]/financials['Stockholders Equity'][1]
        else:
            kpis['DE Ratio']['previous'] = None
        if kpis['DE Ratio']['previous'] and kpis['DE Ratio']['current']:
            kpis['DE Ratio']['delta'] = kpis['DE Ratio']['current'] - kpis['DE Ratio']['previous']
        else:
            kpis['DE Ratio']['delta'] = None
    except:
        kpis['DE Ratio']['delta'] = None
        kpis['DE Ratio']['current'] = None
        kpis['DE Ratio']['previous'] = None
        
    #Net Income
    kpis['Net Income'] = {'desc' : 'Net Income of the company'}
    if financials['Net Income'][0]:
        kpis['Net Income']['current'] = financials['Net Income'][0]
    else:
        kpis['Net Income']['current'] = None
    if financials['Net Income'][1]:
        kpis['Net Income']['previous'] = financials['Net Income'][1]
    else:
        kpis['Net Income']['previous'] = None
    if kpis['Net Income']['previous'] and kpis['Net Income']['current']:
        kpis['Net Income']['delta'] = kpis['Net Income']['current'] - kpis['Net Income']['previous']
    else:
        kpis['Net Income']['delta'] = None
        
    #Free Cash Flow
    kpis['Free Cash Flow'] = {'desc' : 'In Hand cash flow'}
    if financials['Free Cash Flow'][0]:
        kpis['Free Cash Flow']['current'] = financials['Free Cash Flow'][0]
    else:
        kpis['Free Cash Flow']['current'] = None
    if financials['Free Cash Flow'][1]:
        kpis['Free Cash Flow']['previous'] = financials['Free Cash Flow'][1]
    else:
        kpis['Free Cash Flow']['previous'] = None
    if kpis['Free Cash Flow']['previous'] and kpis['Free Cash Flow']['current']:
        kpis['Free Cash Flow']['delta'] = kpis['Free Cash Flow']['current'] - kpis['Free Cash Flow']['previous']
    else:
        kpis['Free Cash Flow']['delta'] = None
        
    #Total Debt
    kpis['Debt'] = {'desc' : 'Total debt of the company'}
    if financials['Total Debt'][0]:
        kpis['Debt']['current'] = financials['Total Debt'][0]
    else:
        kpis['Debt']['current'] = None
    if financials['Total Debt'][1]:
        kpis['Debt']['previous'] = financials['Total Debt'][1]
    else:
        kpis['Debt']['previous'] = None
    if kpis['Debt']['previous'] and kpis['ROE']['current']:
        kpis['Debt']['delta'] = kpis['Debt']['current'] - kpis['Debt']['previous']
    else:
        kpis['Debt']['delta'] = None
        
    #Basic EPS
    kpis['Basic EPS'] = {'desc' : 'Earnings of the company per share'}
    if financials['Net Income'][0]:
        kpis['Basic EPS']['current'] = financials['Basic EPS'][0]
    else:
        kpis['Net Income']['current'] = None
    if financials['Net Income'][1]:
        kpis['Basic EPS']['previous'] = financials['Basic EPS'][1]
    else:
        kpis['Basic EPS']['previous'] = None
    if kpis['Basic EPS']['previous'] and kpis['Basic EPS']['current']:
        kpis['Basic EPS']['delta'] = kpis['Basic EPS']['current'] - kpis['Basic EPS']['previous']
    else:
        kpis['Basic EPS']['delta'] = None
        
    #ROCE
    kpis['ROCE'] = {'desc':"Utilization of capital employed"}
    if financials['EBIT'][0] and financials['Total Assets'][0] and financials['Current Liabilities'][0]:
        kpis['ROCE']['current'] = financials['EBIT'][0] / (financials['Total Assets'][0]-financials['Current Liabilities'][0])
    else:
        kpis['ROCE']['current'] = None
    if financials['EBIT'][1] and financials['Total Assets'][1] and financials['Current Liabilities'][1]:
        kpis['ROCE']['previous'] = financials['EBIT'][1] / (financials['Total Assets'][1]-financials['Current Liabilities'][1])
    else:
        kpis['ROCE']['previous'] = None
    if kpis['ROCE']['previous'] and kpis['ROCE']['current']:
        kpis['ROCE']['delta'] = kpis['ROCE']['current'] - kpis['ROCE']['previous']
    else:
        kpis['ROCE']['delta'] = None
        
    if mode == "delta":
        return pd.DataFrame(kpis).loc['delta'].reset_index().set_index('index').T
    else:
        return kpis
    
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
        if summary[indicator]['Change']*summary[indicator]['type'] > 0:
            summary[indicator]['Tag'] = 'Desirable'
            summary[indicator]['To display'] = summary[indicator]['Positive Change']
        elif summary[indicator]['Change']*summary[indicator]['type'] < 0:
            summary[indicator]['Tag'] = 'Non-Desirable'
            summary[indicator]['To display'] = summary[indicator]['Negative Change']
        else:
            summary[indicator]['Tag'] = 'NA'

    return summary

#in depth stock summary
def stock_summary(stock_data,historical,week_52,dividend_split):
    summary = {}
    #monthly return
    summary['Monthly Returns'] = {}
    if stock_data.Close_change_1m.values[0] > 10:
        summary['Monthly Returns']['Display'] = "Up by "+str(round(stock_data.Close_change_1m.values[0],2)) +"% in 1 past month"
        summary['Monthly Returns']['Type'] = 1
    elif stock_data.Close_change_1m.values[0] < -10:
        summary['Monthly Returns']['Display'] = "Fell by "+str(round(stock_data.Close_change_1m.values[0],2)*-1) +"% in 1 past month"
        summary['Monthly Returns']['Type'] = -1
    else:
        summary.pop("Monthly Returns")

    #yearly return   
    summary['Yearly Returns'] = {}
    if stock_data.Close_change_1y.values[0] > 20:
        summary['Yearly Returns']['Display'] = "Up by "+str(round(stock_data.Close_change_1y.values[0],2)) +"% in 1 past Year"
        summary['Yearly Returns']['Type'] = 1
    elif stock_data.Close_change_1y.values[0] < -20:
        summary['Yearly Returns']['Display'] = "Fell by "+str(round(stock_data.Close_change_1y.values[0],2)*-1) +"% in 1 past Year"
        summary['Yearly Returns']['Type'] = -1
    else:
        summary.pop("Yearly Returns")

    #52 week proximity   
    summary['52 Week Proximity'] = {}
    if stock_data['Latest Close'].values[0] - week_52['52 Week Low'] > -1*(stock_data['Latest Close'].values[0] - week_52['52 Week High']):
        summary['52 Week Proximity']['Display'] = "CLoser to 52 Week High"
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
    elif stock_data['Latest rsi'].values[0] > 70:
        summary['RSI']['Display'] = "Stock appears overbought and may fall soon"
        summary['RSI']['Type'] = -1
    else:
        summary.pop("RSI")

    #ADX
    summary['ADX'] = {}
    if stock_data['Latest ADX'].values[0] > 27 and stock_data['Close_change_10d'].values[0] > 0:
        summary['ADX']['Display'] = "Stock having strong uptrend"
        summary['ADX']['Type'] = 1
    elif stock_data['Latest ADX'].values[0] > 27 and stock_data['Close_change_10d'].values[0] < 0:
        summary['ADX']['Display'] = "Stock having strong downtrend"
        summary['ADX']['Type'] = -1
    elif stock_data['Latest ADX'].values[0] <= 27:
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
        summary['Momentum']['Display'] = "Bullish Moemntum"
        summary['Momentum']['Type'] = 1
    elif stock_data['Latest Close'].values[0] < historical['sma_5'].values[0] and historical['sma_5'].values[0] < historical['sma_25'].values[0]:
        summary['Momentum']['Display'] = "Bearish Momentum"
        summary['Momentum']['Type'] = -1
    else:
        summary['Momentum']['Display'] = "Mid Range Momentum"
        summary['Momentum']['Type'] = 1

    return summary