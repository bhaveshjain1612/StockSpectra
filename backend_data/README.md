# Backend Data Flow and Files

This README is an insight into how the data is pulled and worked on to ensure the app stays running and updated.

## Folder Structure
The project is organized into the following folder structure:
```
project_root/
├── pages
├── **backend_data**
|   ├── collective_backend.py
│   ├── historical
│   │   ├── company historical files
│   ├── financial
│   │   ├── company financial files
│   ├── db_firmo.csv
│   ├── historical_kpis.csv
│   ├── financial_kpis.csv
│   ├── company_list.csv
│   ├── financials_mapping.csv
│   ├── breakout.csv
│   ├── database.csv
├── requirements.txt
├── utils.py
└── Home.py
```
### About some key files:
- **company_list.csv**: A CSV file containing a list of company symbols.
- **historical/**: Folder containing historical data and technical indicators for each company.
  - *COMPANY_SYMBOL.csv*: CSV files for each company, containing historical data and calculated technical indicators.
   
- company_financials/: Folder containing financial data and KPIs for each company.
  - *COMPANY_SYMBOL.csv*: CSV files for each company, containing financial data and calculated KPIs.
   
- **db_firmo.csv**: CSV file containing firmographics data for all companies.
- **historical_kpis.csv**: CSV file containing historical data and technical indicators along with calculated KPIs.
- **financial_kpis.csv**: CSV file containing financial KPIs for all companies.
- **financials_mapping.csv**: CSV file containing a mapping of which financial parameter belongs to which sheet.
- **breakout.csv**: CSV file containing a list of potential breakouts.

## Process Flow
1. Importing Required Libraries: Import necessary libraries, including pandas, yfinance, tqdm, datetime, numpy, and warnings. Warnings are filtered to be ignored.
2. Firmographics Section:
 - Define a function *retrieve_api* to retrieve data from Yahoo Finance API.
 - Define a function *get_company_data* to extract company data from the API result.
 - Define a function *compile_single_firmo* to compile firmographics for a single company using the retrieved data.
 - Define a function *compile_all* to compile firmographics for all companies in a list.
 - Read the company list from **company_list.csv**.
 - Compile firmographics for all companies and save the compiled data to **db_firmo.csv**.
3. Stock Data Section:
 - Define a function get_hist_data to retrieve historical stock data from Yahoo Finance API.
 - Define functions to calculate Exponential Moving Average (EMA), Simple Moving Average (SMA), Relative Strength Index (RSI), Average Directional Index (ADX), and Bollinger Bands for given data.
 - Define a function compile_single to compile historical data and technical indicators for a single company.
 - Define functions to calculate percentage change between two ends of a user-defined interval and quantity.
 - Define a function to calculate the most recent values for given columns.
 - Define a function collect_features to collect key parameters and generate features.
 - Loop through the list of symbols, compile historical data and technical indicators, and generate features for each company. Save the compiled features to **historical_kpis.csv**.
4. Company Financials Section:
 - Define functions to calculate financial KPIs and extract holdings data.
 - Define a function to calculate financial deltas for a single company.
 - Loop through the list of symbols, create financial files for all companies, and save the financial KPIs to **financial_kpis.csv**.
5. Merging Section:
 - Define a function to add sector and industry median P/E ratios to the dataset.
 - Merge the calculated financial KPIs and firmographics data with the historical and technical indicators data to create a comprehensive dataset.
 - Calculate stocks that are potential breakouts and save them as **breakout.csv**

## Automation

A GitHub workflow is set to run automatically on weekdays at 6 PM IST (12 30 UTC). This updates and pushes the updated files to the repository ensuring latest data is fed always
