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

#api retrieval
def retrieve_api(symbol):
    if symbol.upper()[-3:]==".NS":
        symbol = symbol.upper()
    else:
        symbol = symbol.upper()+".NS"
    return(yf.Ticker(symbol))

#company data extraction
def get_company_data(result):
    #creating a function to handle missing values from API whiloe allotting variables
    def assign_variable(result,value): #result is the PAI result, value is the parameter neededfrom API as a string
        try:
            variable = result.info[value] 
        except Exception as e:
            variable = "Not Found"  # Assign  default value in case of an error
        return variable
    #allotting key parmeters
    industry = assign_variable(result,'industry')
    dividendRate = assign_variable(result,'dividendRate')
    sector = assign_variable(result,'sector')
    website = assign_variable(result,'website')
    beta = assign_variable(result,'beta')
    name = assign_variable(result,'longName')
    earningsGrowth = assign_variable(result,'earningsGrowth')
    revenueGrowth = assign_variable(result,'revenueGrowth')
    fistrtradedate = assign_variable(result,'firstTradeDateEpochUtc')
    sharesOutstanding = assign_variable(result,'sharesOutstanding')
    trailingEps = assign_variable(result, 'trailingEps')
    summary = assign_variable(result, 'longBusinessSummary')
    symbol = assign_variable(result, 'symbol')
    previousClose = assign_variable(result, 'previousClose')
    previousVolume = assign_variable(result, 'volume')
    marketCap = assign_variable(result, 'marketCap')
    fiftyTwoWeekLow = assign_variable(result, 'fiftyTwoWeekLow')
    fiftyTwoWeekHigh = assign_variable(result, 'fiftyTwoWeekHigh')
    pegRatio = assign_variable(result, 'pegRatio')
    
    #compiling into adf
    company_parameters = ["Name",
                          "Symbol",
                        "Website",
                        "Description",
                        "Sector",
                        "Industry",
                        "Beta",
                        "Dividend Rate",
                        "Earnings Growth", 
                        "Revenue Growth",
                        "Shares OutStanding",
                        "EPS (trailing)",
                         "Market Cap",
                          "52 Week Low",
                         "52 Week High",
                         "PEG ratio"]                          
    company_values = [name,
                      symbol,
                        website,
                      summary,
                        sector,
                        industry,
                        beta,
                        dividendRate, 
                        earningsGrowth, 
                        revenueGrowth,
                        sharesOutstanding,
                        trailingEps,
                     marketCap,
                     fiftyTwoWeekLow,
                    fiftyTwoWeekHigh,
                     pegRatio]
    params_df =  pd.DataFrame({
        'Parameter': company_parameters,
        "Value": company_values})
     
    return(params_df)

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

#calculate P/E ratio
def calc_peratio(company_params, historical_data):
    eps = company_params[company_params['Parameter']=='EPS (trailing)'].iloc[0,1]
    latest_price = historical_data['Close'][0]
    if eps == "Not Found" or latest_price == "Not Found":
        return({"P/E ratio":"Not Found"})
    else:
        pe_ratio = latest_price/eps
        return({"P/E ratio":round(pe_ratio,4)})

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


#calculate EMAs for a user specified interval
def calc_ema(df,series, interval):
    df = df.sort_values('Trading Day', ascending=False)
    ema = df[series].ewm(span=interval, adjust=False).mean()
    col_name = series+"_ema_"+str(interval)
    df[col_name] = ema
    df = df.sort_values('Trading Day', ascending=True)
    return df

#calculate most recent vals
def calc_most_recent(data,column_list):
    keys, values = [], []
    for i in column_list:
        values.append(data[i].values[0])
        keys.append("Latest "+i) 
    result_dict = dict(zip(keys, values))
    return result_dict

#remove conmpanies yet to be listed
def remove_not_traded_listed(df):
    date_format = "%d-%b-%Y"
    company_traded = []
    for i in df[df.columns[3]]:
        company_traded.append(datetime.strptime(i, date_format).date()>=datetime.now().date())
    df['Company_to_be_removed']=company_traded
    return df[df['Company_to_be_removed']!=True]

#calculate and add industrya and sector median
def calculate_pe_percentiles(df):
    # Calculate the 40th, 50th (median), and 60th percentiles of the PE ratio column
    df = df[df['P/E ratio'] != 'Not Found']   
    df['P/E ratio'] = pd.to_numeric(df['P/E ratio'])   
    percentiles = df.groupby(['Sector','Industry'])['P/E ratio'].quantile([0.4, 0.5, 0.6])
    # Extract the percentile values for each group
    result_df = percentiles.unstack().reset_index()
    # Rename the columns for clarity
    result_df.columns = ['Sector','Industry', 'PE 40th Percentile', 'PE Median', 'PE 60th Percentile']

    return pd.merge(df,result_df, on=['Sector','Industry'])

#compailation of all values for a single symbol
def compile_single(symbol):
    #compilation test
    #raw layer
    result = retrieve_api(symbol)
    #raw processed
    company_data = get_company_data(result)
    historic_data =  get_hist_data(result)
    #calculation
    ema_calculated = calc_ema(calc_ema(get_hist_data(result),'Close',12),'Close',26)
    ema_calculated["macd"] = ema_calculated["Close_ema_12"]-ema_calculated["Close_ema_26"]
    ema_calculated = calc_ema(ema_calculated,"macd",9)
    latest_figs = calc_most_recent(ema_calculated,
                                   ["date_only","Close","Volume","Close_ema_12","Close_ema_26","macd","macd_ema_9"])
    peratio = calc_peratio(company_data, historic_data)

    #compilation
    all_data = company_data.set_index(company_data.columns[0]).T
    all_data = pd.concat([all_data, pd.DataFrame(latest_figs, index=["Value"])],axis=1)
    all_data['P/E ratio'] = peratio["P/E ratio"]
    for j in [(1,"D"),(5,"D"),(1,"M"),(3,"M"),(6,"M"),(1,"Y")]:
        for i in ['Close','Volume']:
            col_name = i+"_change_"+str(j[0])+j[1].lower()
            all_data[col_name] = calc_change(ema_calculated,i,j,'%')
        for m in ['macd','macd_ema_9']:
            col_name = m+"_change_"+str(j[0])+j[1].lower()
            all_data[col_name] = calc_change(ema_calculated,m,j,"abs")

    #Tweaks
    replace_map = {'Consumer Cyclical': 'FMCG', 'Consumer Defensive': 'FMCG'}
    all_data['Sector'] = all_data['Sector'].replace(replace_map)

    #returning
    return all_data

#compile on a list of values
def compile_all(list_of_symbols): 
    df = pd.DataFrame()
    #loop tghrough list of symbols executing compile_single and appending result of each
    for i in tqdm(list_of_symbols):
        try:
            df_s = compile_single(i)
            if df_s.empty:
                continue
            else:
                df = pd.concat([df,df_s],axis=0)
        except:
            continue
            
    #final cleaning (removal of compaines with not found sector  or industry)
    df = df[df['Sector']!='Not Found']
    df = df[df['Industry']!='Not Found']
    #adding sector and insutry median PEs
    df = calculate_pe_percentiles(df)
    
    return df

#comany list from NSE
company_list_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
in_companies = pd.read_csv(company_list_url)
in_companies.to_csv('company_list.csv', index=False)

#compiling
compiled = compile_all(remove_not_traded_listed(in_companies).SYMBOL)
#adding creation timetsamp
compiled["created_on"] = datetime.now()

compiled.to_csv('database.csv', index=False)