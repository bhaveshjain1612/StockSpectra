# StockProject 
![Static Badge](https://img.shields.io/badge/Status-Online-green)

👋 Greetings,

I'm thrilled to introduce the **Stock Insights and Analysis App** – a project born from my drive as a college student to simplify stock screening. What started as a personal endeavor soon evolved into a showcase of my *analytics, business, UI, and programming skills*. This application is designed to help you make informed investment decisions by providing valuable insights, in-depth analysis, and real-time stock trends. Whether you are a seasoned investor or just starting, this app offers a range of features to assist you in selecting potential stocks and understanding their performance. Please review the [**Disclaimer**](#disclaimer) before using this application.

The app, hosted on **Streamlit** is a result of relentless efforts to merge technology and finance seamlessly. With algorithms powered by NSE and BSE data from Yahoo Finance, it identifies potential stocks and offers in-depth analysis. From real-time trend analysis to comprehensive financial insights, each feature is meticulously crafted to empower users in their investment decisions.

But this journey is far from over. I'm eager to receive your feedback and suggestions. Your insights can refine and elevate this project, making it a valuable resource for the wider community. Please share your thoughts as we continue to shape this app together.

Link to the app: https://stock-recommendation.streamlit.app/

## About the App

The Stock Insights and Analysis App embodies our commitment to enhancing financial literacy and facilitating well-informed investment choices. With its intuitive interface and robust capabilities, this application equips users with a range of tools to streamline their stock market research and decision-making process.

### Notable Features
- **Technical Trends and Indicators**: Access real-time technical trends and indicators, empowering you to uncover potential stock opportunities.
- **Thorough Stock Analysis**: Gain deep insights into individual stocks, unraveling crucial performance metrics, sector analyses, and financial indicators.
- **Financial Statements**: Access annual financial statements to gain an in-depth understanding of a company's financial health and performance.
- **Breakout Opportunities**: Pinpoint potential breakout stocks through meticulous analysis and trend identification.

### Data Integrity and Updates
The app sources data from NSE and BSE, ensuring that you're equipped with up-to-date stock information. Daily data updates at 7:30 PM IST guarantee access to the latest market trends, while annual financial statement updates offer a holistic view of a company's financial standing.\

To see the detailed data flow process [click here](https://github.com/bhaveshjain1612/StockProject/edit/main/backend_data/README.md)

#### App Structure:

For detailed usage instructions, [click here.](https://stock-recommendation.streamlit.app/Instructions)\
The data is stored in the *backend_data* folder and the fucntions for all pages are in the *utils.py* file\
Below is an overview of the app structure:

#### Page 1. Stock Screening

The "Stock Screening" section allows you to filter and view stocks based on various criteria, making it easy to find stocks that meet your specific preferences. This is the home page of the app.

**Set Filters:**
- **Strategy**: Allows you to shuffle between 4 key strategies - > 1 year, 3 - 6 months, 1-2 months and potential breakouts. Only one can be selected in a go.
All other parameters are varied basis the strategy selected. If left to *None*, all parameters will stay default.
- **Risk/ Reward** : Is to select the risk factor associated with the company.Selecting All allows you to view all 3. More than one can be selected.
    - *High* : Companies in this group tend tend to have high variation between returns. They may give very high, or negative returns
    - *Mid* : Companies in this group tend tend to have not very high or very low variation between returns.
    - *Low* : Companies in this group tend tend to have low variation between returns, i.e. while returns may be less, so is the risk compared to High rtanked companies. 
- **Outlook Filter** : Select cpomapines based on the technical analysis outllok of stock data.Selecting All allows you to view all 3. More than one can be selected.
    - *Positive* : Companies in this group tend are more likley to have positive returns in the future compared to mrket movement.
    - *Neutral* : Companies in this group tend are more likley to have at par returns in the future compared to market.
    - *Negative* : Companies in this group tend are more likley to have sub par returns in the future. 
- **Financial Strength**: Similar to stock rank, you can filter stocks based on their financial rank. These strengths represent the analysis of latest Year on Year company financials.
Selecting All allows you to view all 3. More than one can be selected.
    - *Strong* : Good YoY Financials compared to others 
    - *Mid* : Near median FInancial strength of YoY parameters
    - *Weak* : Sub par recent YoY financials compared to other stocks
**Sort the companies:**
- Use the two dropdowns in 3rd row to sort them by Name, Price, Dividend Yield or Change for the selected interval.

**Display Companies:**
- Adjust the number of companies you want to display using the slider.
- The displayed table includes a column labeled "Analysis" with clickable links. Clicking on "See in Depth" will take you to a detailed analysis page for the respective stock.

#### Page 2. In-Depth Stock Analysis

The "In-Depth Stock Analysis" section allows you to explore detailed insights for individual stocks.

- **Enter Stock Ticker**: Input the stock ticker in the sidebar using the text input box, and press Enter. The app will display insights for the specified company.
- **Share Insights via URL**: The app supports loading insights based on the stock ticker passed through the URL query parameters. Share specific company insights with others by providing the stock ticker in the URL.

Tabs for Comprehensive Analysis

- **Company Details Tab**: Displays firmographic data about the company, including its sector, industry, and a brief description. A holding chart provides graphical representation related to the company's holdings.
- **Stock Tab**: Provides key performance indicators (KPIs) such as close price, volume, high, low, normal dividend, stock split, 52-week high, and 52-week low. Interactive charts allow customization options like different Simple Moving Averages (SMAs) and Exponential Moving Averages (EMAs), Bollinger Bands, time intervals, and additional indicators.
- **Financial Tab**: Provides financial metrics such as Net Income, Debt, Free Cash Flow, Basic EPS, Net Profit Margin, ROA, ROE, ROCE, Current Ratio, and DE Ratio. Desirable and non-desirable indicators are listed, and detailed financial data is accessible for the company.

#### Page 3. Instructions

The "Instructions" section provides an overview of the Financial Dashboard app and its key features:

- **Key Features**: Discover how the app helps you with technical trend identification, in-depth stock analysis, access to financial statements, and short-term trading opportunities.
- **Data Sources and Updates**: Learn about the data sources from the National Stock Exchange (NSE) and Bombay Stock Exchange (BSE) and the scheduled daily data update at 7:30 PM IST.
- **How to Use**: Understand how to navigate through the app and utilize it the best as per your needs.
- **GLossary**: A glossary with formulae of some of the frequently mentioned technical terms.


<h2 id="disclaimer">Disclaimer</h2>

This app is intended for educational purposes only. The information provided within this app is not intended to serve as financial advice or recommendations for stock trading or investment decisions. The stock market involves inherent risks, and users should exercise caution and conduct their own research before making any investment decisions.

The insights and analysis provided by this app are based on historical data and technical trends, which may not accurately predict future stock performance. The app creators and developers do not guarantee the accuracy, completeness, or timeliness of the information provided. Users are solely responsible for any trading or investment decisions made based on the information obtained from this app.

Investing in the stock market carries risks, and it is important to consult with a qualified financial advisor before making any investment decisions. The creators of this app shall not be held liable for any losses or damages resulting from the use of this app or its contents.

Remember that stock trading and investments are subject to market fluctuations and individual risk tolerance. Always do your own due diligence and seek professional advice when needed.


---

Happy exploring and learning about the stock market with the Financial Dashboard app!

---
