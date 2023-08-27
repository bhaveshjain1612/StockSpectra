# Importing required libraries
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import date, datetime, timedelta
import numpy as np
import warnings

# Code where you want to ignore warnings
warnings.filterwarnings("ignore")

# Function to retrieve data from Yahoo Finance API
def retrieve_api(symbol):
    return yf.Ticker(symbol)

# *********************************************************************************************************************
# Firmographics Section
# *********************************************************************************************************************
'''
# Function to extract company data from API result
def get_company_data(result):
    # Creating a function to handle missing values from API while assigning variables
    def assign_variable(result, value):
        try:
            variable = result.info[value]
        except Exception as e:
            variable = "Not Found"  # Assigning a default value in case of an error
        return variable
    
    # Allotting key parameters from the API result
    industry = assign_variable(result, 'industry')
    dividendRate = assign_variable(result, 'dividendRate')
    sector = assign_variable(result, 'sector')
    website = assign_variable(result, 'website')
    beta = assign_variable(result, 'beta')
    name = assign_variable(result, 'longName')
    earningsGrowth = assign_variable(result, 'earningsGrowth')
    revenueGrowth = assign_variable(result, 'revenueGrowth')
    firstTradeDate = assign_variable(result, 'firstTradeDateEpochUtc')
    sharesOutstanding = assign_variable(result, 'sharesOutstanding')
    trailingEps = assign_variable(result, 'trailingEps')
    summary = assign_variable(result, 'longBusinessSummary')
    symbol = assign_variable(result, 'symbol')
    previousClose = assign_variable(result, 'previousClose')
    previousVolume = assign_variable(result, 'volume')
    marketCap = assign_variable(result, 'marketCap')
    fiftyTwoWeekLow = assign_variable(result, 'fiftyTwoWeekLow')
    fiftyTwoWeekHigh = assign_variable(result, 'fiftyTwoWeekHigh')
    pegRatio = assign_variable(result, 'pegRatio')
    
    # Determine the exchange based on the symbol suffix
    if symbol[-3:] == ".NS":
        exchange = "NSE"
    elif symbol[-3:] == ".BO":
        exchange = "BSE"
    
    # Compiling the data into a DataFrame
    company_parameters = ["Name", "Symbol", "Exchange", "Website", "Description", "Sector", "Industry", "Beta",
                          "Dividend Rate", "Earnings Growth", "Revenue Growth", "Shares OutStanding",
                          "EPS (trailing)", "Market Cap"]
    company_values = [name, symbol, exchange, website, summary, sector, industry, beta, dividendRate, earningsGrowth,
                      revenueGrowth, sharesOutstanding, trailingEps, marketCap]
    params_df = pd.DataFrame({'Parameter': company_parameters, "Value": company_values})
     
    return params_df.set_index('Parameter').transpose()

# Function to compile firmographics for a single company
def compile_single_firmo(symbol):
    # Retrieve raw data from API
    result = retrieve_api(symbol)
    # Process raw data to get company data
    company_data = get_company_data(result)
    # Make tweaks to the data, such as replacing sector values
    replace_map = {'Consumer Cyclical': 'FMCG', 'Consumer Defensive': 'FMCG'}
    company_data['Sector'] = company_data['Sector'].replace(replace_map)
    # Return the compiled company data
    return company_data

# Function to compile firmographics for all companies in a list
def compile_all(list_of_symbols): 
    df = pd.DataFrame()
    # Loop through the list of symbols, execute compile_single for each, and append the results to the DataFrame
    for i in tqdm(list_of_symbols):
        try:
            df_s = compile_single_firmo(i)
            if df_s.empty:
                continue
            else:
                df = pd.concat([df, df_s], axis=0)
        except:
            continue
            
    # Final cleaning: Remove companies with 'Not Found' sector or industry
    df = df[df['Sector'] != 'Not Found']
    df = df[df['Industry'] != 'Not Found']
    # Add sector and industry median PEs (not implemented in this code snippet)
    # df = add_sector_industry_pe_median(df)
    
    return df

# Read the company list from a CSV file
in_companies = pd.read_csv('company_list.csv')

# Compile firmographics for all companies in the list
compiled = compile_all(in_companies.SYMBOL)
# Add creation timestamp to the DataFrame
compiled["created_on"] = datetime.now()

# Save the compiled firmographics to a CSV file
compiled.to_csv('db_firmo.csv', index=False)
'''
# *********************************************************************************************************************
#Stock Data
# *********************************************************************************************************************

# Function to retrieve historical data from Yahoo Finance API for a given result
def get_hist_data(result):
    # Get today's date
    date_today = date.today() + timedelta(days=1)
    # Fetch historical data from 2020-01-01 until today for daily intervals
    historical = result.history(end=date_today, start="2020-01-01", period="1d")

    # Remove days with zero trading volume (representing market holidays)
    historical = historical[historical['Volume'] != 0]
    
    # Sort the DataFrame by date in descending order and add a 'Trading Day' column representing the trading day number,
    # i.e., if today is 5 July (day 1), then 4 July will be 2, and so on.
    historical = historical.sort_values('Date', ascending=False).reset_index().reset_index()
    historical = historical.rename(columns={'index': 'Trading Day'})
    historical['Trading Day'] = historical['Trading Day'] + 1
    
    # Create a new column with date only (removing time information)
    date_only = []
    for i in historical['Date']:
        x = str(pd.to_datetime(i).year).zfill(2) + "-" + str(pd.to_datetime(i).month).zfill(2) + "-" + str(pd.to_datetime(i).day).zfill(2)
        date_only.append(np.datetime64(x))
    historical['date_only'] = date_only
    
    # Keep only values from the last 2 years
    historical_sample = historical[historical['date_only'] > historical['date_only'][0] - np.timedelta64(2, "Y")]
    
    return historical_sample

# Function to calculate Exponential Moving Average (EMA) for a given DataFrame, series, and interval
def calc_ema(df, interval):
    df = df.sort_values('Trading Day', ascending=False)
    ema = df['Close'].ewm(span=interval, adjust=False).mean()
    col_name = "ema_" + str(interval)
    df[col_name] = ema
    df = df.sort_values('Trading Day', ascending=True)
    return df

# Function to calc macd
def calc_macd(df):
    df = df.sort_values('Trading Day', ascending=False)

    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=25, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macde_histogram'] = df['macd'] - df['macd_signal']
    df.drop(['ema_12','ema_26'],axis=1, inplace= True)
    
    df = df.sort_values('Trading Day', ascending=True)
    
    return df

# Function to calculate Simple Moving Average (SMA) for a given DataFrame and interval
def calc_sma(df, interval):
    df = df.sort_values('Trading Day', ascending=False)
    sma = df['Close'].rolling(interval).mean()
    col_name = "sma_" + str(interval)
    df[col_name] = sma
    df = df.sort_values('Trading Day', ascending=True)
    return df

# Function to calculate Relative Strength Index (RSI) for a given DataFrame and window_length
def calc_rsi(df, window_length):
    # Calculate differences between closing prices of consecutive days
    diff = []
    for i in range(df.shape[0] - 1):
        diff.append(df.Close[i] - df.Close[i + 1])
    diff.append(np.nan)
    df['diff'] = diff
    df = df.sort_values('Trading Day', ascending=False)

    # Calculate gains and losses based on differences
    df['gain'] = df['diff'].clip(lower=0).round(2)
    df['loss'] = df['diff'].clip(upper=0).abs().round(2)

    # Calculate average gains and losses for the given window length
    df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length + 1]
    df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length + 1]

    # Calculate average gains and losses for the rest of the DataFrame
    for i, row in enumerate(df['avg_gain'].iloc[window_length + 1:]):
        df['avg_gain'].iloc[i + window_length + 1] = \
            (df['avg_gain'].iloc[i + window_length] * (window_length - 1) + df['gain'].iloc[i + window_length + 1]) / window_length

    for i, row in enumerate(df['avg_loss'].iloc[window_length + 1:]):
        df['avg_loss'].iloc[i + window_length + 1] = \
            (df['avg_loss'].iloc[i + window_length] * (window_length - 1) + df['loss'].iloc[i + window_length + 1]) / window_length

    # Calculate Relative Strength (RS) and RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1.0 + df['rs']))
    df = df.drop(['gain', 'loss', 'rs', 'avg_gain', 'avg_loss', 'diff'], axis=1)
    df = df.sort_values('Trading Day', ascending=True)
    return df

# Function to calculate Average Directional Index (ADX) for a given DataFrame and period
def calc_ADX(data: pd.DataFrame, period: int):
    """
    Computes the ADX indicator.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    df = data.copy()
    df = df.sort_values('Trading Day', ascending=False)
    
    # Calculate alpha for the exponential moving average
    alpha = 1 / period

    # Calculate True Range (TR)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # Calculate Average True Range (ATR) using exponential moving average of TR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # Calculate +DM and -DM values
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH'] > 0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L'] > 0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # Calculate smoothed +DM and -DM values using exponential moving average
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()

    # Calculate +DMI and -DMI values as a percentage of ATR
    df['+DMI'] = (df['S+DM'] / df['ATR']) * 100
    df['-DMI'] = (df['S-DM'] / df['ATR']) * 100
    del df['S+DM'], df['S-DM']

    # Calculate Directional Movement Index (DX) and ADX using exponential moving average
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI']) / (df['+DMI'] + df['-DMI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']
    
    df = df.sort_values('Trading Day', ascending=True)

    return df

# Function to calculate Bollinger Bands for a given DataFrame
def bollinger(df):
    df = df.sort_values('Trading Day', ascending=False) 

    # Calculate rolling mean and standard deviation
    window = 20  # Number of periods for rolling calculation
    df['RollingMean'] = df['Close'].rolling(window=window).mean()
    df['RollingStd'] = df['Close'].rolling(window=window).std()

    # Calculate Bollinger Bands
    df['UpperBand1'] = df['RollingMean'] + df['RollingStd']
    df['LowerBand1'] = df['RollingMean'] - df['RollingStd']
    df['UpperBand2'] = df['RollingMean'] + 2 * df['RollingStd']
    df['LowerBand2'] = df['RollingMean'] - 2 * df['RollingStd']

    df = df.sort_values('Trading Day', ascending=True)
    
    return df

#Calculate CCI
def calculate_cci(df, interval):
    df = df.sort_values('Trading Day', ascending=False)
    # Calculate the Typical Price
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate the moving average of the Typical Price
    df['MA_TP'] = df['TP'].rolling(window=interval).mean()
    
    # Calculate the mean deviation
    df['MD'] = df['TP'].rolling(window=interval).apply(lambda x: abs(x - x.mean()).mean())
    
    # Calculate the CCI
    df[f'CCI_{interval}'] = (df['TP'] - df['MA_TP']) / (0.015 * df['MD'])
    
    # Drop the intermediate columns used for calculations
    df.drop(['TP', 'MA_TP', 'MD'], axis=1, inplace=True)
    df = df.sort_values('Trading Day', ascending=True)   
    return df


# VPT
def calculate_vpt(df):
    df = df.sort_values('Trading Day', ascending=False)
    
    df['VPT'] = (df['Close'].pct_change() + 1).cumprod() * df['Volume']
    df['VPT_signal'] = df['VPT'].ewm(span=9, adjust=False).mean()
    
    df = df.sort_values('Trading Day', ascending=True)
    
    return df

# VWAP
def calculate_vwap(df):
    df = df.sort_values('Trading Day', ascending=False)
    
    df['cum_vol'] = df['Volume'].cumsum()
    df['cum_vol_price'] = (df['Close'] * df['Volume']).cumsum()
    df['VWAP'] = df['cum_vol_price'] / df['cum_vol']
    df.drop(['cum_vol', 'cum_vol_price'], axis=1, inplace=True)
    
    df = df.sort_values('Trading Day', ascending=True)
    
    return df

# MFI
def calculate_mfi(df, interval=14):
    df = df.sort_values('Trading Day', ascending=False)
    
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['money_flow'] = df['TP'] * df['Volume']
    
    df['positive_money_flow'] = df['money_flow'].where(df['TP'] > df['TP'].shift(1), 0)
    df['negative_money_flow'] = df['money_flow'].where(df['TP'] < df['TP'].shift(1), 0)
    
    df['positive_money_flow'] = df['positive_money_flow'].rolling(window=interval).sum()
    df['negative_money_flow'] = df['negative_money_flow'].rolling(window=interval).sum()
    
    MFI = 100 - (100 / (1 + df['positive_money_flow'] / df['negative_money_flow']))
    df[f'MFI_{interval}'] = MFI
    
    df.drop(['TP', 'money_flow', 'positive_money_flow', 'negative_money_flow'], axis=1, inplace=True)
    
    df = df.sort_values('Trading Day', ascending=True)
    
    return df

# OBV
def calculate_obv(df):
    df = df.sort_values('Trading Day', ascending=False)
    
    df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
    
    df = df.sort_values('Trading Day', ascending=True)
    
    return df

# Pivot Points with additional levels
def calculate_pivotpoints(df):
    df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = (2 * df['PP']) - df['Low']
    df['S1'] = (2 * df['PP']) - df['High']
    df['R2'] = df['PP'] + (df['High'] - df['Low'])
    df['S2'] = df['PP'] - (df['High'] - df['Low'])
    df['R3'] = df['High'] + 2 * (df['PP'] - df['Low'])
    df['S3'] = df['Low'] - 2 * (df['High'] - df['PP'])
    return df

#calculate williams r
def calculate_williams_r(df, period=14):
    df = df.sort_values('Trading Day', ascending=False)  
    
    df['Highest High'] = df['High'].rolling(window=period).max()
    df['Lowest Low'] = df['Low'].rolling(window=period).min()
    df['%R'] = (-100) * (df['Highest High'] - df['Close']) / (df['Highest High'] - df['Lowest Low'])
    df.drop(['Highest High', 'Lowest Low'], axis=1, inplace=True)
    
    df = df.sort_values('Trading Day', ascending=True)  
    
    return df

# Chaikin Money Flow (CMF)
def calculate_cmf(df, period=20):
    df = df.sort_values('Trading Day', ascending=False)  
    
    df['Money Flow Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['Money Flow Volume'] = df['Money Flow Multiplier'] * df['Volume']
    df['CMF'] = df['Money Flow Volume'].rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    df.drop(['Money Flow Multiplier', 'Money Flow Volume'], axis=1, inplace=True)
    
    df = df.sort_values('Trading Day', ascending=True)
    
    return df

# Function to compile historical data and technical indicators for a single company
def compile_single(symbol):
    # Get historical data for the given symbol
    result = retrieve_api(symbol)
    historic_data = get_hist_data(result)
    
    # Calculate technical indicators
    data = bollinger(historic_data) 
    data = calculate_cci(data, 10)
    data = calculate_cci(data, 40)
    data = calculate_obv(data)
    data = calculate_vpt(data)
    data = calculate_vwap(data)
    data = calculate_mfi(data, 14)
    data = calculate_pivotpoints(data)
    data = calculate_cmf(data)
    data = calculate_williams_r(data)
    data = calc_macd(data)
    data = calc_rsi(data, 14)
    data = calc_ADX(data, 14)
    data = calc_sma(data, 5)
    data = calc_sma(data, 20)
    data = calc_sma(data, 50)
    data = calc_sma(data, 100)
    
    # Add symbol and creation timestamp columns to the DataFrame
    data['symbol'] = symbol
    data["created_on"] = datetime.now()

    # Save the DataFrame to a CSV file
    compilation_name = "historical/" + symbol.replace('.', '_') + ".csv"
    data.to_csv(compilation_name, index=False)
    
    return data

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

def stds(data):
    durations = [(1, "M"), (3, "M"), (6, "M"), (1, "Y")]
    end = data['date_only'][0]
    
    std_values = [
        data[data['date_only'] >= (end - np.timedelta64(duration[0], duration[1]))]['Close'].std()
        for duration in durations
    ]
    
    columns = ['std_' + str(duration[0]) + duration[1] for duration in durations]
    volatility = pd.DataFrame([std_values], columns=columns)
    
    return volatility

def calc_most_recent(data, column_list):
    return {"Latest " + col: data[col].values[0] for col in column_list}

# Function to collect key parameters and generate features
def collect_features(df):
    temp = pd.DataFrame()
       
    # Calculate percentage change for different durations and technical indicators
    changes = {}
    for j in [(1, "D"), (5, "D"), (10, "D"), (1, "M"), (3, "M"), (6, "M"), (1, "Y")]:
        for i in ['Close', 'Volume']:
            col_name = i + "_change_" + str(j[0]) + j[1].lower()
            changes[col_name] = calc_change(df, i, j, '%')
            
    for n in [(1, "D"), (5, "D"), (10, "D"),]:
        for m in ['macd', 'macd_signal','CCI_10', 'CCI_40', 'OBV', 'VPT', 'VPT_signal', 'VWAP', 'MFI_14', 'rsi', 'ADX', 'sma_5', 'sma_20', 'sma_50', 'sma_100' ]:
            col_name = m + "_change_" + str(j[0]) + j[1].lower()
            changes[col_name] = calc_change(df, m, n, "abs")
    changes = pd.DataFrame(changes, index=[0])
    
    # Calculate most recent values for specific columns
    temp_latest = calc_most_recent(df, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
       'Dividends', 'Stock Splits', 'date_only', 'RollingMean', 'RollingStd',
       'UpperBand1', 'LowerBand1', 'UpperBand2', 'LowerBand2', 'CCI_10',
       'CCI_40', 'OBV', 'VPT', 'VPT_signal', 'VWAP', 'MFI_14', 'PP', 'R1',
       'S1', 'R2', 'S2', 'R3', 'S3', 'CMF', '%R', 'macd', 'macd_signal',
       'macde_histogram', 'rsi', 'ADX', 'sma_5', 'sma_20', 'sma_50', 'sma_100',
       'symbol', 'created_on'])
    
    temp_latest_sr = calc_most_recent(df[df['Trading Day']==2].rename(columns= {'PP': 'PP_previousday', 'R1': 'R1_previousday', 'S1': 'S1_previousday', 
                                              'R2': 'R2_previousday', 'S2': 'S2_previousday', 
                                              'R3': 'R3_previousday', 'S3': 'S3_previousday',}), 
                                   ['PP_previousday','R1_previousday','S1_previousday',
                                    'R2_previousday','S2_previousday',
                                    'R3_previousday','S3_previousday'])
    
    temp = pd.concat([temp, pd.DataFrame(temp_latest, index=[0])], axis=1)
    temp = pd.concat([temp, pd.DataFrame(temp_latest_sr, index=[0])], axis=1)
    temp = pd.concat([temp, changes], axis=1)
    temp = pd.concat([temp, stds(df)], axis=1)
    
    temp['symbol'] = df['symbol'].values[0]
        
    return temp

# Read the company list from the 'db_firmo.csv' file
db_firmo = pd.read_csv("db_firmo.csv")
symbol_list = db_firmo.Symbol
kpi_df = pd.DataFrame()

# Loop through the list of symbols, compile historical data and technical indicators, and generate features for each company
for i in tqdm(symbol_list):
    try:
        x = collect_features(compile_single(i))
        kpi_df = pd.concat([kpi_df, x], axis=0)
    except:
        continue

# Save the compiled features to a CSV file named 'historical_kpis.csv'
kpi_df.to_csv("historical_kpis.csv", index=False)
print("Stock Historicals updated")

# *********************************************************************************************************************
#Company Financials
# *********************************************************************************************************************
'''
# Function to calculate financial Key Performance Indicators (KPIs)
def calc_KPIs(financials, mode):
    financials = financials.T
    kpis = {}
    # ROE - Return on Equity
    try:
        kpis['ROE'] = {'desc': 'Efficiency of Equity utilization'}
        if financials['Stockholders Equity'][0] and financials['Net Income'][0]:
            kpis['ROE']['current'] = financials['Stockholders Equity'][0] / financials['Net Income'][0]
        else:
            kpis['ROE']['current'] = None
        if financials['Stockholders Equity'][1] and financials['Net Income'][1]:
            kpis['ROE']['previous'] = financials['Stockholders Equity'][1] / financials['Net Income'][1]
        else:
            kpis['ROE']['previous'] = None
        if kpis['ROE']['previous'] and kpis['ROE']['current']:
            kpis['ROE']['delta'] = kpis['ROE']['current'] - kpis['ROE']['previous']
        else:
            kpis['ROE']['delta'] = None
    except:
        kpis['ROE'] = {'delta': None, 'current': None, 'previous': None}

    # ROA - Return on Assets
    try:
        kpis['ROA'] = {'desc': 'Efficiency of Assets utilization'}
        if financials['Total Assets'][0] and financials['Net Income'][0]:
            kpis['ROA']['current'] = financials['Net Income'][0] / financials['Total Assets'][0]
        else:
            kpis['ROA']['current'] = None
        if financials['Total Assets'][1] and financials['Net Income'][1]:
            kpis['ROA']['previous'] = financials['Net Income'][1] / financials['Total Assets'][1]
        else:
            kpis['ROA']['previous'] = None
        if kpis['ROA']['previous'] and kpis['ROA']['current']:
            kpis['ROA']['delta'] = kpis['ROA']['current'] - kpis['ROA']['previous']
        else:
            kpis['ROA']['delta'] = None
    except:
        kpis['ROA'] = {'delta': None, 'current': None, 'previous': None}

    # Current Ratio - Ability to pay short-term liabilities
    try:
        kpis['Current Ratio'] = {'desc': 'Ability to pay short-term liabilities'}
        if financials['Current Assets'][0] and financials['Current Liabilities'][0]:
            kpis['Current Ratio']['current'] = financials['Current Assets'][0] / financials['Current Liabilities'][0]
        else:
            kpis['Current Ratio']['current'] = None
        if financials['Current Assets'][1] and financials['Current Liabilities'][1]:
            kpis['Current Ratio']['previous'] = financials['Current Assets'][1] / financials['Current Liabilities'][1]
        else:
            kpis['Current Ratio']['previous'] = None
        if kpis['Current Ratio']['previous'] and kpis['Current Ratio']['current']:
            kpis['Current Ratio']['delta'] = kpis['Current Ratio']['current'] - kpis['Current Ratio']['previous']
        else:
            kpis['Current Ratio']['delta'] = None
    except:
        kpis['Current Ratio'] = {'delta': None, 'current': None, 'previous': None}

    # Gross Margin - Profitability of a company
    try:
        kpis['Net Profit Margin'] = {'desc': 'Profitability of a company'}
        if financials['Total Revenue'][0] and financials['Net Income'][0]:
            kpis['Net Profit Margin']['current'] = financials['Net Income'][0] / financials['Total Revenue'][0]
        else:
            kpis['Net Profit Margin']['current'] = None
        if financials['Total Revenue'][1] and financials['Net Income'][1]:
            kpis['Net Profit Margin']['previous'] = financials['Net Income'][1] / financials['Total Revenue'][1]
        else:
            kpis['Net Profit Margin']['previous'] = None
        if kpis['Net Profit Margin']['previous'] and kpis['Net Profit Margin']['current']:
            kpis['Net Profit Margin']['delta'] = kpis['Net Profit Margin']['current'] - kpis['Net Profit Margin']['previous']
        else:
            kpis['Net Profit Margin']['delta'] = None
    except:
        kpis['Net Profit Margin'] = {'delta': None, 'current': None, 'previous': None}

    # DE Ratio - Debt to Equity Ratio
    try:
        kpis['DE Ratio'] = {'desc': 'Total debt compared to equity'}
        if financials['Stockholders Equity'][0] and financials['Total Debt'][0]:
            kpis['DE Ratio']['current'] = financials['Total Debt'][0] / financials['Stockholders Equity'][0]
        else:
            kpis['DE Ratio']['current'] = None
        if financials['Stockholders Equity'][1] and financials['Total Debt'][1]:
            kpis['DE Ratio']['previous'] = financials['Total Debt'][1] / financials['Stockholders Equity'][1]
        else:
            kpis['DE Ratio']['previous'] = None
        if kpis['DE Ratio']['previous'] and kpis['DE Ratio']['current']:
            kpis['DE Ratio']['delta'] = kpis['DE Ratio']['current'] - kpis['DE Ratio']['previous']
        else:
            kpis['DE Ratio']['delta'] = None
    except:
        kpis['DE Ratio'] = {'delta': None, 'current': None, 'previous': None}

    # Net Income
    try:
        kpis['Net Income'] = {'desc': 'Net Income of the company'}
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
    except:
        kpis['Net Income'] = {'delta': None, 'current': None, 'previous': None}

    # Free Cash Flow
    try:
        kpis['Free Cash Flow'] = {'desc': 'In Hand cash flow'}
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
    except:
        kpis['Free Cash Flow'] = {'delta': None, 'current': None, 'previous': None}

    # Total Debt
    try:
        kpis['Debt'] = {'desc': 'Total debt of the company'}
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
    except:
        kpis['Debt'] = {'delta': None, 'current': None, 'previous': None}

    # Basic EPS - Earnings of the company per share
    try:
        kpis['Basic EPS'] = {'desc': 'Earnings of the company per share'}
        if financials['Net Income'][0]:
            kpis['Basic EPS']['current'] = financials['Basic EPS'][0]
        else:
            kpis['Basic EPS']['current'] = None
        if financials['Net Income'][1]:
            kpis['Basic EPS']['previous'] = financials['Basic EPS'][1]
        else:
            kpis['Basic EPS']['previous'] = None
        if kpis['Basic EPS']['previous'] and kpis['Basic EPS']['current']:
            kpis['Basic EPS']['delta'] = kpis['Basic EPS']['current'] - kpis['Basic EPS']['previous']
        else:
            kpis['Basic EPS']['delta'] = None
    except:
        kpis['Basic EPS'] = {'delta': None, 'current': None, 'previous': None}

    # ROCE - Return on Capital Employed
    try:
        kpis['ROCE'] = {'desc': "Utilization of capital employed"}
        if financials['EBIT'][0] and financials['Total Assets'][0] and financials['Current Liabilities'][0]:
            kpis['ROCE']['current'] = financials['EBIT'][0] / (financials['Total Assets'][0] - financials['Current Liabilities'][0])
        else:
            kpis['ROCE']['current'] = None
        if financials['EBIT'][1] and financials['Total Assets'][1] and financials['Current Liabilities'][1]:
            kpis['ROCE']['previous'] = financials['EBIT'][1] / (financials['Total Assets'][1] - financials['Current Liabilities'][1])
        else:
            kpis['ROCE']['previous'] = None
        if kpis['ROCE']['previous'] and kpis['ROCE']['current']:
            kpis['ROCE']['delta'] = kpis['ROCE']['current'] - kpis['ROCE']['previous']
        else:
            kpis['ROCE']['delta'] = None
    except:
        kpis['ROCE'] = {'delta': None, 'current': None, 'previous': None}

    if mode == "delta":
        return pd.DataFrame(kpis).loc['delta'].reset_index().set_index('index').T
    else:
        return kpis


# Function to extract holdings data
def holding(result):
    hkpi = {
        'index': result.major_holders[1],
        'delta': result.major_holders[0].str.replace("%", "").astype(float)
    }
    hkpi = pd.DataFrame(hkpi).set_index('index').T
    return hkpi


# Function to calculate financial deltas for one company
def single_financials(symbol):
    result = retrieve_api(symbol)
    i = result.income_stmt
    b = result.balance_sheet
    c = result.cashflow
    fin = pd.concat([i, b, c], axis=0)

    fin.to_csv("company_financials/" + symbol.replace(".", "_") + ".csv")

    fkpi = calc_KPIs(fin, "delta")
    hkpi = holding(result)

    finkpi = pd.concat([fkpi, hkpi], axis=1)
    finkpi['symbol'] = symbol

    return finkpi


# Main process to create financial files for all companies
db_firmo = pd.read_csv("db_firmo.csv")
symbol_list = db_firmo.Symbol
fin_kpi_df = pd.DataFrame()

# Creating financial files for all companies
for i in tqdm(symbol_list):
    try:
        x = single_financials(i)
        fin_kpi_df = pd.concat([fin_kpi_df, x], axis=0)
    except:
        continue

fin_kpi_df.to_csv("financial_kpis.csv", index=False)
'''
# *********************************************************************************************************************
#Merging
# *********************************************************************************************************************

#adding median pe values
def add_sector_industry_pe_median(collective):
    industry_up_pe_ratios =  pd.DataFrame(collective[collective["P/E ratio"]!="Not Found"].groupby("Industry")["P/E ratio"].quantile(0.60)).reset_index().rename(columns={"P/E ratio":"Industry upper P/E Ratio"})
    industry_low_pe_ratios =  pd.DataFrame(collective[collective["P/E ratio"]!="Not Found"].groupby("Industry")["P/E ratio"].quantile(0.40)).reset_index().rename(columns={"P/E ratio":"Industry lower P/E Ratio"})
    industry_median_pe_ratios =  pd.DataFrame(collective[collective["P/E ratio"]!="Not Found"].groupby("Industry")["P/E ratio"].median()).reset_index().rename(columns={"P/E ratio":"Industry Median P/E Ratio"})
    
    sector_up_pe_ratios =  pd.DataFrame(collective[collective["P/E ratio"]!="Not Found"].groupby("Sector")["P/E ratio"].quantile(0.40)).reset_index().rename(columns={"P/E ratio":"Sector lower P/E Ratio"})   
    sector_low_pe_ratios =  pd.DataFrame(collective[collective["P/E ratio"]!="Not Found"].groupby("Sector")["P/E ratio"].quantile(0.60)).reset_index().rename(columns={"P/E ratio":"Sector upper P/E Ratio"})   
    sector_median_pe_ratios =  pd.DataFrame(collective[collective["P/E ratio"]!="Not Found"].groupby("Sector")["P/E ratio"].median()).reset_index().rename(columns={"P/E ratio":"Sector Median P/E Ratio"})   
    
    new_collective = pd.merge(pd.merge(collective, sector_median_pe_ratios, on='Sector'), industry_median_pe_ratios, on ="Industry")
    new_collective = pd.merge(pd.merge(new_collective, sector_up_pe_ratios, on='Sector'), sector_low_pe_ratios, on ="Sector")
    new_collective = pd.merge(pd.merge(new_collective, industry_up_pe_ratios, on='Industry'), industry_low_pe_ratios, on ="Industry")
        
    return new_collective

#collecting all data together
firmo = pd.read_csv("db_firmo.csv")
financial = pd.read_csv("financial_kpis.csv")
historic = pd.read_csv("historical_kpis.csv")

database = pd.merge(firmo, financial, left_on='Symbol', right_on='symbol', how='inner')
database = pd.merge(database, historic, left_on='Symbol', right_on='symbol', how='inner')
database = database.drop(['symbol_x','symbol_y'],axis=1)

#calculating PE ratios
database = database[database['EPS (trailing)']!="Not Found"]
database['EPS (trailing)'] = database['EPS (trailing)'].astype('float')
database['P/E ratio']=database['Latest Close']/database['EPS (trailing)']
database = add_sector_industry_pe_median(database)

#calculating volatility
database['volatility_1M'] = database['std_1M']/database['Latest Close']
database['volatility_3M'] = database['std_3M']/database['Latest Close']
database['volatility_6M'] = database['std_6M']/database['Latest Close']
database['volatility_1Y'] = database['std_1Y']/database['Latest Close']

#allotting all necessary tags
#finaical score:
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

#allotting a score based on stock performance
def allot_outlook(df):

    # Weights for each parameter
    short_term_weights = {
        'rsi': 3,
        'macd': 2,
        'adx': 1,
        'bollinger': 3,
        'pe_ratio': 1,
        'moving_averages': 2,
        'financials_yoy': 1,
        'cci_10': 3,
        'cci_40': 1,
        'vpt': 2,
        'vwap': 3,
        'mfi_14': 3
    }


    for index, row in df.iterrows():
        
        short_term_score = 0
        long_term_score = 0
        
        #Short term logic
        # RSI logic
        if row['Latest rsi'] > 60 and row['Latest rsi'] < 80: # bullish
            short_term_score += short_term_weights['rsi']
        elif row['Latest rsi'] < 30 or row['Latest rsi'] > 80: # overbought
            short_term_score -= short_term_weights['rsi']

        # ADX logic
        if row['Latest ADX'] > 30:
            short_term_score += short_term_weights['adx'] # strong trend
        else:
            short_term_score -= short_term_weights['adx'] # weak trend

        # MACD logic
        if row['Latest macd'] > 0:
            short_term_score += short_term_weights['macd']

        # Bollinger Bands logic
        price_position = (row['Latest Close'] - row['Latest RollingMean']) / (row['Latest UpperBand2'] - row['Latest LowerBand2'])
        if price_position < -0.5:
            short_term_score += short_term_weights['bollinger']
        elif price_position > 0.5:
            short_term_score -= short_term_weights['bollinger']

        # Price change logic
        if row['Close_change_1d'] > 0:
            short_term_score += 1
            
        if row['Close_change_1m'] > 1:
            long_term_score += 1
        else:
            long_term_score -= 1
            
        if row['Close_change_6m'] > 5:
            long_term_score += 1
        else:
            long_term_score -= 1
            
        if row['Close_change_1y'] > 10:
            long_term_score += 1
        else:
            long_term_score -= 1
            
            
        # if DIvidend
        if row['Dividend Rate'] != 'Not Found':
            long_term_score -= 1
        else:
            long_term_score += 1
        
        # P/E Ratio logic
        if row['P/E ratio'] < row['Industry Median P/E Ratio']:
            long_term_score -= 1
        else:
            long_term_score += 1
        
        # Moving Averages logic
        if row['Latest sma_50'] > row['Latest sma_100']:
            long_term_score += 1
        else:
            long_term_score -= 1
        
        # CCI logic for short term
        if row['Latest CCI_10'] > 100:
            short_term_score += short_term_weights['cci_10']
        elif row['Latest CCI_10'] < -100:
            short_term_score -= short_term_weights['cci_10']
            
         # CCI logic for long term
        if row['Latest CCI_40'] > 100:
            short_term_score += short_term_weights['cci_40']
        elif row['Latest CCI_40'] < -100:
            short_term_score -= short_term_weights['cci_40']
        
        # VPT logic for short term
        if row['Latest VPT'] -  row['Latest VPT_signal'] > 0:
            short_term_score += short_term_weights['vpt']
        else:
            short_term_score -= short_term_weights['vpt']
            
        # MFI logic for short term
        if row['Latest MFI_14'] > 80 and row['Latest MFI_14'] < 20: # overbought
            short_term_score -= short_term_weights['mfi_14']
        elif row['Latest MFI_14'] > 65 and row['Latest MFI_14'] < 80: # bullish
            short_term_score += short_term_weights['mfi_14']
        
        # Financials YoY score logic
        short_term_score += row['finscore']*0.5 
        long_term_score += row['finscore'] 
        
        # Translating the scores into outlooks
        df.at[index, 'short_term_score'] = short_term_score 
        df.at[index, 'long_term_score'] = long_term_score 
        
        def outlook_category(df,col,outname):
            
            scr = df[col]

            def assign_tier(score):
                if score > scr.max()/2:
                    return 'very positive'
                elif score > scr.max()/4:
                    return 'positive'
                elif score > 0:
                    return 'neutral'
                else :
                    return 'negative'
                
            df[outname] = df[col].apply(assign_tier)

            return df

    df = outlook_category(df,'short_term_score','Outlook 1-2Months')
    df = outlook_category(df,'long_term_score','Outlook >1Year')
    return df

def allot_risk(df,col,outname):
    u = 0.6
    l = 0.4
    #out_df =pd.DataFrame()

    percentiles = df[col].quantile([u,l])
    
    def assign_tier(score):
        if score >= percentiles[u] :
            return 'High'
        if score >= percentiles[l] :
            return 'Mid'
        else:
            return 'Low'

    df[outname] = df[col].apply(assign_tier)
       
    return df

#finanical strength ranking
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

#allcoate risk based on standard deviation
def allot_risk(df,col,outname):
    u = 0.6
    l = 0.4
    #out_df =pd.DataFrame()

    percentiles = df[col].quantile([u,l])
    
    def assign_tier(score):
        if score >= percentiles[u] :
            return 'High'
        if score >= percentiles[l] :
            return 'Mid'
        else:
            return 'Low'

    df[outname] = df[col].apply(assign_tier)
       
    return df

#funaction to allot all tags
def allot_tags(df):
    df['finscore'] = df.apply(financial_scores,axis=1)
    df = allot_risk(df,'volatility_1M','Risk 1-2Months')
    df = allot_risk(df,'volatility_6M','Risk 5-6Months')
    df = allot_risk(df,'volatility_1Y','Risk >1Year')
    df = finrank(df)
    df = allot_outlook(df)
    return df

#allot tags and dividend yield cols
database = allot_tags(database)
database['Annual Dividend'] = database['Dividend Rate'].replace('Not Found', '0').astype(float)
database['Dividend Yield'] = database['Annual Dividend']*100/database['Latest Close']

# Allot market cap catergorization
def categorize_cap(value):
        if value >= 20000000000:  # 20,000 crores in rupees
            return 'Large-cap'
        elif 5000000000 <= value < 20000000000:  # Between 5,000 crores and 20,000 crores in rupees
            return 'Mid-cap'
        else:
            return 'Small-cap'

database = database[database['Market Cap'] != 'Not Found']
database['Market Cap'] = database['Market Cap'].astype('int64')
database['Cap'] = database['Market Cap'].apply(categorize_cap)

#exporting the final db
database["Latest created_on"] = datetime.now()
database.to_csv('database.csv', index=False,mode='w')
print("Data Updated")

####################################################################################################################################################
#Breakouts
####################################################################################################################################################
#calculate breakouts
def find_breakouts():
    def breakout_conditions(data):
            return (
                (data['Close'] >= data['UpperBand1']) & 
                (data['Close'].shift(-1) >= data['Open'].shift(-1)) &
                (data['Close'] >= data['Close'].shift(-1)) & 
                (data['Close'].shift(-1) >= data['UpperBand1'].shift(-1)) &
                (data['Close'] >= data['Open']) &  
                (data['rsi'] <= 70) &
                ((data['CCI_10'] - data['CCI_40']> 80) | ((data['CCI_40'] - data['CCI_10'] > 0) & (data['CCI_40'] - data['CCI_10'] < 50))) & #(data['CCI_10'] < 200) &
                (data['CCI_40'] > 100) & (data['CCI_10'] > 100) &
                (data['MFI_14'] < 85) & (data['MFI_14'] > 40)  
                & (data['VPT'] >= data['VPT'].shift(-1))
            )

    sym,brk = [],[]
    for i in pd.read_csv("database.csv").Symbol:
        x = pd.read_csv('historical/'+i.replace(".","_")+".csv")
        try:
            b = breakout_conditions(x)[0]
            sym.append(i)
            brk.append(b)
        except:
            continue

    df1 = pd.DataFrame()
    df1['Symbol'] =  sym
    df1['Breakout'] =  brk    
    df1 = df1[df1['Breakout']==True]
    
    return df1

find_breakouts().to_csv('breakout.csv')
print('Breakouts updated')
