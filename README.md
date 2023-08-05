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
Below is an overview of the app structure:

#### Page 1. Stock Screening

The "Stock Screening" section allows you to filter and view stocks based on various criteria, making it easy to find stocks that meet your specific preferences. This is the home page of the app.

- **Filter by Stock Name**: Enter a partial or full name of the stock you are interested in, and the dashboard will filter the relevant results accordingly.
- **Filter by Stock Exchange**: Select the desired stock exchange (NSE or BSE) from the dropdown menu. The dashboard will display stocks from the selected exchange.
- **Filter by Stock Rank (Outlook)**: Choose one or more stock ranks (e.g., positive, negative, neutral) to filter stocks based on their outlook. The dashboard will update with the selected stocks.
- **Filter by Financial Rank**: Similar to stock rank, you can filter stocks based on their financial rank (e.g., strong, weak). Choose one or more financial ranks to apply the filter.
- Filter by **Stock Risk**: Choose one or more stock ranks to filter stocks based on their possible risk factor as analysed. The dashboard will update with the selected stocks.
    - **High** : These are high risk/high reward stocks.
    - **Low** : These are low risk/low reward stocks
    - **Mid** : These are moderate risk/moderate reward stocks
- Choose from three viewing options:
    - **Top Picks**: Displays the top-rated stocks with strong outlook, low - medium risk and financials.
    - **All**: Shows all stocks based on applied filters.
    - **Potential Breakout**: Lists stocks that are potential breakout candidates. Most of these are high risk companies so proceed with caution.
- **Filter by Sector and Industry**: Further refine your search by selecting specific sectors and industries. Use the multiselect dropdowns to include or exclude particular sectors/industries.
- **Display Companies**: Adjust the number of companies you want to display using the slider. You can view a maximum of 10 companies at once.

#### Page 2. In-Depth Stock Analysis

The "In-Depth Stock Analysis" section allows you to explore detailed insights for individual stocks.

- **Enter Stock Ticker**: Input the stock ticker in the sidebar using the text input box, and press Enter. The app will display insights for the specified company.
- **Share Insights via URL**: The app supports loading insights based on the stock ticker passed through the URL query parameters. Share specific company insights with others by providing the stock ticker in the URL.

Tabs for Comprehensive Analysis

- **Firmo Tab**: Displays firmographic data about the company, including its sector, industry, and a brief description. A holding chart provides graphical representation related to the company's holdings.
- **Stock Tab**: Provides key performance indicators (KPIs) such as close price, volume, high, low, normal dividend, stock split, 52-week high, and 52-week low. Interactive charts allow customization options like different Simple Moving Averages (SMAs) and Exponential Moving Averages (EMAs), Bollinger Bands, time intervals, and additional indicators.
- **Financial Tab**: Provides financial metrics such as Net Income, Debt, Free Cash Flow, Basic EPS, Net Profit Margin, ROA, ROE, ROCE, Current Ratio, and DE Ratio. Desirable and non-desirable indicators are listed, and detailed financial data is accessible for the company.

#### Page 3. Instructions

The "About" section provides an overview of the Financial Dashboard app and its key features:

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
