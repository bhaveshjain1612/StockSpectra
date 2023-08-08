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
def calc_ema(df, series, interval):
    df = df.sort_values('Trading Day', ascending=False)
    ema = df[series].ewm(span=interval, adjust=False).mean()
    col_name = series + "_ema_" + str(interval)
    df[col_name] = ema
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

# Function to compile historical data and technical indicators for a single company
def compile_single(symbol):
    # Get historical data for the given symbol
    result = retrieve_api(symbol)
    historic_data = get_hist_data(result)
    
    # Calculate technical indicators: EMA, SMA, RSI, ADX, and Bollinger Bands
    ema_calculated = calc_ema(calc_sma(calc_ema(calc_ADX(calc_rsi(calc_ema(calc_ema(get_hist_data(result), 'Close', 12), 'Close', 26), 14), 14), 'Close', 20), 20), 'Close', 5)
    ema_calculated["macd"] = ema_calculated["Close_ema_12"] - ema_calculated["Close_ema_26"]
    ema_calculated = calc_ema(ema_calculated, "macd", 9)
    ema_calculated = bollinger(ema_calculated)

    # Add symbol and creation timestamp columns to the DataFrame
    ema_calculated['symbol'] = symbol
    ema_calculated["created_on"] = datetime.now()

    # Save the DataFrame to a CSV file
    compilation_name = "historical/" + symbol.replace('.', '_') + ".csv"
    ema_calculated.to_csv(compilation_name, index=False)
    
    return ema_calculated

# Function to calculate percentage change between two ends of a user-defined interval and quantity
def calc_change(data, series, duration, changetype):
    # If duration is not in days (e.g., 1 month is (1, 'M')), calculate the starting and ending values accordingly
    if duration[1] != "D":
        end = data['date_only'][0]
        start = end - np.timedelta64(duration[0], duration[1])
        value_start = data[data['date_only'] >= start][series].values[-1]
        value_end = data[data['date_only'] >= end][series].values[-1]
    else:
        value_end = data[data["Trading Day"] == 1][series].values[0]
        value_start = data[data["Trading Day"] == 1 + duration[0]][series].values[0]
   
    # Calculate percentage change or absolute change based on changetype
    if changetype == "%":
        return round(((value_end - value_start) / value_start) * 100, 2)
    else:
        return (value_end - value_start)
    
# Function to calculate 3 month, 1 uyear, 1month standard deviation
def stds(data):
    volatility = pd.DataFrame()
    # If duration is not in days (e.g., 1 month is (1, 'M')), calculate the starting and ending values accordingly
    for duration in [(1,"M"),(3,"M"),(6,"M"),(1,"Y")]:
        end = data['date_only'][0]
        start = end - np.timedelta64(duration[0], duration[1])
        volatility['std_'+str(duration[0])+ duration[1]] = [data[data['date_only'] >= start]['Close'].std()]
    #print(volatility)
    return volatility

# Function to calculate the most recent values for given columns
def calc_most_recent(data, column_list):
    keys, values = [], []
    for i in column_list:
        values.append(data[i].values[0])
        keys.append("Latest " + i) 
    result_dict = dict(zip(keys, values))
    return result_dict

# Function to collect key parameters and generate features
def collect_features(df):
    temp = pd.DataFrame()
       
    # Calculate percentage change for different durations and technical indicators
    changes = {}
    for j in [(1, "D"), (5, "D"), (10, "D"), (1, "M"), (3, "M"), (6, "M"), (1, "Y")]:
        for i in ['Close', 'Volume']:
            col_name = i + "_change_" + str(j[0]) + j[1].lower()
            changes[col_name] = calc_change(df, i, j, '%')
        for m in ['macd', 'macd_ema_9', 'rsi', 'ADX']:
            col_name = m + "_change_" + str(j[0]) + j[1].lower()
            changes[col_name] = calc_change(df, m, j, "abs")
    changes = pd.DataFrame(changes, index=[0])
    
    # Calculate most recent values for specific columns
    temp_latest = calc_most_recent(df, ["date_only", "Close", "Volume", "Close_ema_12", "Close_ema_26", "macd", "macd_ema_9",
                                        "rsi", "ADX", 'Close_ema_20', 'sma_20', 'Close_ema_5', 'RollingMean', 'RollingStd',
                                        'UpperBand1', 'LowerBand1', 'UpperBand2', 'LowerBand2', 'symbol', "created_on"])
    
    temp = pd.concat([temp, pd.DataFrame(temp_latest, index=[0])], axis=1)
    temp = pd.concat([temp, changes], axis=1)
    temp = pd.concat([temp, stds(df)], axis=1)
    
    temp['symbol'] = df['symbol'].values[0]
        
    return temp

# Read the company list from the 'db_firmo.csv' file
db_firmo = pd.read_csv("db_firmo.csv")
symbol_list = db_firmo.Symbol#.head(10)
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

#exporting the final db
database["Latest created_on"] = datetime.now()
database.to_csv('database.csv', index=False,mode='w')
print("Data Updated")

####################################################################################################################################################
#Breakouts
####################################################################################################################################################
#calculate bollinger breaouts

def check_bollinger_flag(df):
    bollinger_band = df.UpperBand2
    closing_prices = df.Close
    x = {}
    i=0
        # Check the conditions for the past 3 sessions
    if (closing_prices[i] > bollinger_band[i]) and (closing_prices[i] >= df.Open.values[i]) and (closing_prices[i+1] > bollinger_band[i+1]) and (closing_prices[i+2] < bollinger_band[i+2]) and df.rsi.values[0]<70 and closing_prices[i+1]<closing_prices[i]:
        flag=True
    else:
        flag=False
    
    m=0
    for i in range(1,20):
        if (closing_prices[i] > bollinger_band[i]) and (closing_prices[i] >= df.Open.values[i]) and (closing_prices[i+1] > bollinger_band[i+1]) and (closing_prices[i+2] < bollinger_band[i+2]) and df.rsi.values[0]<70 and closing_prices[i+1]<closing_prices[i]:
            m+=1
        else:
            m+=0
            
    if flag  and m==0:
        x['Breakout']=True,
    else:
        x['Breakout']=False
            
    return(pd.DataFrame(x,index=[df.symbol.values[0]]))

database= pd.read_csv("database.csv")
final = pd.DataFrame()
for i in database.Symbol:
    try:
        df = pd.read_csv("historical/"+i.replace(".","_")+".csv")
        x = check_bollinger_flag(df)
        final = pd.concat([final,x],axis=0)
    except:
        continue
final[final['Breakout']==True].to_csv("breakout.csv")
print('Breakouts updated')