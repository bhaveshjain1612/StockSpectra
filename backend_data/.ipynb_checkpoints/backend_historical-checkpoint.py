#importing libraries
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

#get api data
def retrieve_api(symbol):
    if symbol.upper()[-3:]==".NS":
        symbol = symbol.upper()
    else:
        symbol = symbol.upper()+".NS"
    return(yf.Ticker(symbol))

#historical data
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

#EMa claculations
def calc_ema(df,series, interval):
    df = df.sort_values('Trading Day', ascending=False)
    ema = df[series].ewm(span=interval, adjust=False).mean()
    col_name = series+"_ema_"+str(interval)
    df[col_name] = ema
    df = df.sort_values('Trading Day', ascending=True)
    return df

#compile for a single company
def compile_single(symbol):
    #compilation test
    #raw layer
    result = retrieve_api(symbol)
    #raw processed
    historic_data =  get_hist_data(result)
    #calculation
    ema_calculated = calc_ema(calc_ema(get_hist_data(result),'Close',12),'Close',26)
    ema_calculated["macd"] = ema_calculated["Close_ema_12"]-ema_calculated["Close_ema_26"]
    ema_calculated = calc_ema(ema_calculated,"macd",9)

    #compilation
    ema_calculated['symbol'] = symbol
    ema_calculated["created_on"] = datetime.now()

    #writing csv
    compilation_name = symbol[:-3]+"_hist.csv"
    compilation_adress = "historical/"
    ema_calculated.to_csv(compilation_adress+compilation_name, index=False)
    return ema_calculated       


#calculate % change b/w two ends of user defined interval and qty
def calc_change(data,series,duration, changetype):
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
    if changetype == "%":
        return round(((value_end - value_start) / value_start) * 100,2)
    else:
        return (value_end - value_start)
    
#calculate most recent vals
def calc_most_recent(data,column_list):
    keys, values = [], []
    for i in column_list:
        values.append(data[i].values[0])
        keys.append("Latest "+i) 
    result_dict = dict(zip(keys, values))
    return result_dict
    
#collecting key parameters
def collect_features(df):
    temp = pd.DataFrame()
       
    changes = {}
    for j in [(1,"D"),(5,"D"),(1,"M"),(3,"M"),(6,"M"),(1,"Y")]:
            for i in ['Close','Volume']:
                col_name = i+"_change_"+str(j[0])+j[1].lower()
                changes[col_name] = calc_change(df,i,j,'%')
            for m in ['macd','macd_ema_9']:
                col_name = m+"_change_"+str(j[0])+j[1].lower()
                changes[col_name] = calc_change(df,m,j,"abs")
    changes = pd.DataFrame(changes,index=[0])
    
    temp_latest = calc_most_recent(df,["date_only","Close","Volume","Close_ema_12","Close_ema_26","macd","macd_ema_9","created_on"])
    
    temp= pd.concat([temp, pd.DataFrame(temp_latest , index=[0])],axis=1)
    temp= pd.concat([temp, changes],axis=1)
    
    temp['symbol'] = df['symbol'].values[0]
        
    return temp

    
#getting input list of signals
db_firmo = pd.read_csv("db_firmo.csv")
symbol_list = db_firmo.Symbol
kpi_df = pd.DataFrame()

#creating historical files for all companies
for i in tqdm(symbol_list):
    x = collect_features(compile_single(i))
    kpi_df = pd.concat([kpi_df,x],axis=0)
kpi_df.to_csv("historical_kpis.csv", index=False)