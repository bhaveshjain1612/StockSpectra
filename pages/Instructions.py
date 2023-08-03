import streamlit as st
import sys
from streamlit_autorefresh import st_autorefresh
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)

# ... (rest of the code)

# Define the "How To Use" page
def home():
    # Stock Screening
    st.header("Stock Screening")

    # Introduction
    introduction_text = '''

    This application is designed to help you quickly filter through a vast database of stocks, making it easy to find top picks, explore potential breakout candidates, and access in-depth stock analysis for making informed investment decisions. Please review the disclaimer before using this application for educational purposes.

    ---

    '''
    
    st.markdown(introduction_text, unsafe_allow_html=True)
    st.subheader('How to Use')
    # How to Use
    how_to_use_text = '''

    1. **Set Filters:**
       - Filter by **Stock Name**: Enter a partial or full name of the stock you are interested in, and the dashboard will filter the relevant results accordingly.
       - Filter by **Stock Exchange**: Select the desired stock exchange (NSE or BSE) from the dropdown menu. The dashboard will display stocks from the selected exchange.
       - Filter by **Stock Rank (Outlook)**: Choose one or more stock ranks (e.g., positive, negative, neutral) to filter stocks based on their outlook. The dashboard will update with the selected stocks.
       - Filter by **Financial Rank**: Similar to stock rank, you can filter stocks based on their financial rank (e.g., strong, weak). Choose one or more financial ranks to apply the filter.

    2. **View Options:**
       - Choose from three viewing options:
         - **Top Picks**: Displays the top-rated stocks with strong outlook and financials.
         - **All**: Shows all stocks based on applied filters.
         - **Potential Breakout**: Lists stocks that are potential breakout candidates .

    3. **Filter by Sector and Industry:**
       - Further refine your search by selecting specific sectors and industries. Use the multiselect dropdowns to include or exclude particular sectors/industries.

    4. **Display Companies:**
       - Adjust the number of companies you want to display using the slider.

    5. **Clickable Links:**
       - The displayed table includes a column labeled "Analysis" with clickable links. Clicking on "See in Depth" will take you to a detailed analysis page for the respective stock.

    6. **Auto-Refresh:**
       - The dashboard automatically refreshes to display updated data periodically. No need to manually refresh the page.

    '''

    st.markdown(how_to_use_text, unsafe_allow_html=True)


    
def indepth():
    
        # Stock Analysis
    st.header("Stock Analysis")
    st.markdown("On the main page, you can input the stock ticker in the sidebar using the text input box.")
    st.markdown("After entering the stock ticker and pressing Enter, the app will display the insights for the specified company.")
    st.markdown("The app also supports loading insights based on the stock ticker passed through the URL query parameters. This allows you to share specific company insights with others by providing the stock ticker in the URL.")

    with st.expander("Firmo Tab",expanded=True):
        st.markdown(
            """
            - The "Firmo" tab displays firmographic data about the company.
            - It shows the sector and industry to which the company belongs.
            - The "About the company" section provides a brief description of the company.
            - The "Holding Chart" is displayed below the description, showing some graphical representation related to the company's holdings.
            """
        )

    with st.expander("Stock Tab",expanded=True):
        st.markdown(
            """
            - The "Stock" tab provides insights related to the company's stock.
            - It displays key performance indicators (KPIs) such as close price, volume, high, low, normal dividend, stock split, 52-week high, and 52-week low.
            - The KPIs include the current values and their deltas compared to the previous day's data.
            - Below the KPIs, there are interactive charts with several options to customize the view:
                - You can select different Simple Moving Averages (SMAs) and Exponential Moving Averages (EMAs) to plot on the chart.
                - You can choose to display Bollinger Bands with 1 or 2 standard deviations.
                - You can select a time interval from options like 1 Day, 5 Days, 1 Month, etc., to zoom in on specific periods.
                - You can also choose to plot additional indicators like Volume, MACD, ADX, or RSI on the bottom part of the chart.
            - The chart is updated based on the selected options.
            """
        )

    with st.expander("Financial Tab",expanded=True):
        st.markdown(
            """
            - The "Financial" tab provides financial insights about the company.
            - It displays key financial metrics such as Net Income, Debt, Free Cash Flow, Basic EPS, Net Profit Margin, Return on Assets (ROA), Return on Equity (ROE), Return on Capital Employed (ROCE), Current Ratio, and Debt to Equity (DE) Ratio.
            - The metrics include the current values and their deltas compared to the previous period.
            - Below the metrics, there are two columns, "Pros" and "Cons," displaying the desirable and non-desirable indicators based on their tags.
            - You can expand the "Income Statement," "Balance Sheet," and "Cash Flow" sections to view the detailed financial data for the company.
            """
        )
        
def about():
    st.header("About")
    
    st.markdown("Welcome to the Stock Insights and Analysis App!\n"
                "This app is designed to help you make informed decisions in the stock market "
                "by providing valuable insights and in-depth analysis. Whether you are a seasoned "
                "investor or just starting, this app offers a range of features to assist you "
                "in selecting potential stocks and understanding their performance.")
    
    st.subheader("Key Features:")
    st.markdown("1. **Select Stocks Based on Technical Trends:** Get access to real-time technical trends "
                "and indicators to identify potential stock opportunities. The app uses the Yahoo Finance "
                "library to fetch data from NSE and BSE.")
    st.markdown("2. **In-depth Stock Analysis:** Dive deep into individual stocks to gain valuable insights "
                "about the company's performance, sector, and industry. Analyze historical data and key financial "
                "indicators to make informed investment decisions.")
    st.markdown("3. **Financial Statements:** Access annual financial statements to understand the financial "
                "health and performance of the company. The financial data is updated annually.")
    st.markdown("4. **Short-term Trading Opportunities:** Discover potential short-term trading opportunities "
                "and breakout stocks through comprehensive analysis and trend identification.")
    
    st.subheader("Data Sources and Updates:")
    st.markdown("The app utilizes data from the National Stock Exchange (NSE) and Bombay Stock Exchange (BSE) "
                "to provide up-to-date stock information. The daily data update is scheduled at 7:30 PM IST, "
                "ensuring you have access to the latest market trends and insights.")
    st.markdown("The financial statements provided in the app are updated annually, giving you a comprehensive "
                "view of the company's financial position.")
    
    st.subheader("Disclaimer:")
    st.markdown("This app is intended for educational purposes only. The information provided within this app is "
                "not intended to serve as financial advice or recommendations for stock trading or investment decisions. "
                "The stock market involves inherent risks, and users should exercise caution and conduct their own "
                "research before making any investment decisions.")
    st.markdown("The insights and analysis provided by this app are based on historical data and technical trends, "
                "which may not accurately predict future stock performance. The app creators and developers do not "
                "guarantee the accuracy, completeness, or timeliness of the information provided. Users are solely "
                "responsible for any trading or investment decisions made based on the information obtained from this app.")
    st.markdown("Investing in the stock market carries risks, and it is important to consult with a qualified financial "
                "advisor before making any investment decisions. The creators of this app shall not be held liable for "
                "any losses or damages resulting from the use of this app or its contents.")
    st.markdown("Remember that stock trading and investments are subject to market fluctuations and individual risk tolerance. "
                "Always do your own due diligence and seek professional advice when needed.")
    
    st.info("Happy exploring and learning about the stock market with the Stock Insights and Analysis App!")
    
def glossary():
    # Define the dictionary with topics and their summaries/formulas
    topics = ["ROE", "ROCE", "DE Ratio", "Net Income", "Debt", "ROA", "Current Ratio", "Free Cash Flow", "Net Profit Margin","ADX", "RSI", "Bollinger Bands", "EMA", "SMA", "MACD","Dividend", "Stock Split", "52 Week High", "52 Week Low", "NSE", "BSE", "Volume of Trading", "High", "Low", "Open", "Close"]
    topics.sort()
    # Search bar for topics
    search_query = st.selectbox("Search topics",topics)
    filtered_topics = [topic for topic in topics if search_query.lower() in topic.lower()]

    # Displaying selected topic and summary with formula (if available)
    if len(filtered_topics) > 0:
        for topic in filtered_topics:
            st.subheader(topic)
            if topic == "ROE":
                st.write("Return on Equity (ROE) is a financial ratio that measures a company's profitability by calculating the return generated on shareholders' equity.")
                st.latex(r"\text{ROE} = \frac{\text{Net Income}}{\text{Shareholders' Equity}} \times 100")
            elif topic == "ROCE":
                st.write("Return on Capital Employed (ROCE) is a financial ratio that measures a company's profitability by calculating the return generated on the capital employed in the business.")
                st.latex(r"\text{ROCE} = \frac{\text{EBIT}}{\text{Capital Employed}} \times 100")
            elif topic == "DE Ratio":
                st.write("Debt to Equity (DE) Ratio is a financial ratio that compares a company's total debt to its shareholders' equity.")
                st.latex(r"\text{DE Ratio} = \frac{\text{Total Debt}}{\text{Shareholders' Equity}}")
            elif topic == "Net Income":
                st.write("Net Income represents a company's total earnings or profit after deducting all expenses and taxes.")
                st.latex(r"\text{Net Income} = \text{Total Revenue} - \text{Total Expenses}")
            elif topic == "Debt":
                st.write("Debt refers to the total liabilities of a company, including long-term and short-term obligations.")
                st.latex(r"\text{Debt} = \text{Long-term Debt} + \text{Short-term Debt}")
            elif topic == "ROA":
                st.write("Return on Assets (ROA) is a financial ratio that measures a company's profitability by calculating the return generated on its total assets.")
                st.latex(r"\text{ROA} = \frac{\text{Net Income}}{\text{Total Assets}} \times 100")
            elif topic == "Current Ratio":
                st.write("Current Ratio is a financial ratio that measures a company's ability to pay its short-term obligations.")
                st.latex(r"\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}")
            elif topic == "Free Cash Flow":
                st.write("Free Cash Flow (FCF) is a measure of a company's ability to generate cash from its operations after deducting capital expenditures.")
                st.latex(r"\text{FCF} = \text{Operating Cash Flow} - \text{Capital Expenditures}")
            elif topic == "Net Profit Margin":
                st.write("Net Profit Margin is a financial ratio that measures a company's profitability by calculating the percentage of profit generated from its total revenue.")
                st.latex(r"\text{Net Profit Margin} = \frac{\text{Net Income}}{\text{Total Revenue}} \times 100")
            elif topic == "ADX":
                st.write("Average Directional Index (ADX) is a technical indicator used to measure the strength of a trend.")
                st.latex(r"\text{ADX} = \frac{100}{N} \sum_{i=1}^N \frac{|(+DI_i) - (-DI_i)|}{(+DI_i) + (-DI_i)}")
                st.write("The +DI and -DI are the positive and negative directional movement indicators, and N is the number of periods.")
            elif topic == "RSI":
                st.write("Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.")
                st.latex(r"\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}")
                st.write("The Average Gain is the sum of gains over a specific period, and the Average Loss is the sum of losses over the same period.")
            elif topic == "Bollinger Bands":
                st.write("Bollinger Bands consist of a middle band (SMA) and two outer bands that are standard deviations away from the middle band.")
                st.latex(r"\text{Middle Band (SMA)} = \frac{1}{N} \sum_{i=1}^N \text{Closing Price}_i")
                st.latex(r"\text{Upper Band} = \text{Middle Band} + k \times \text{Standard Deviation}")
                st.latex(r"\text{Lower Band} = \text{Middle Band} - k \times \text{Standard Deviation}")
                st.write("N is the number of periods, and k is the number of standard deviations (typically set to 2).")
            elif topic == "EMA":
                st.write("Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent data points.")
                st.latex(r"\text{EMA} = (1 - \alpha) \times \text{Previous EMA} + \alpha \times \text{Current Price}")
                st.write("The smoothing factor α determines the weight given to the current price, and it is usually calculated as 2 / (N + 1), where N is the number of periods.")
            elif topic == "SMA":
                st.write("Simple Moving Average (SMA) is a type of moving average that calculates the average of a specified number of periods.")
                st.latex(r"\text{SMA} = \frac{1}{N} \sum_{i=1}^N \text{Closing Price}_i")
                st.write("N is the number of periods.")
            elif topic == "MACD":
                st.write("Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that calculates the difference between two EMAs.")
                st.latex(r"\text{MACD Line} = \text{12-period EMA} - \text{26-period EMA}")
                st.latex(r"\text{Signal Line} = \text{9-period EMA of MACD Line}")
                st.latex(r"\text{MACD Histogram} = \text{MACD Line} - \text{Signal Line}")
                st.write("N is the number of periods, and k is the number of standard deviations (typically set to 2).")
            elif topic == "Dividend":
                st.write("Dividend is a distribution of a portion of a company's earnings to its shareholders.")
                st.write("Dividend per Share (DPS) is calculated as follows:")
                st.latex(r"\text{DPS} = \frac{\text{Total Dividend Paid}}{\text{Total Number of Shares}}")
            elif topic == "Stock Split":
                st.write("A stock split is a corporate action that increases the number of shares in a company without changing the overall market value.")
                st.write("A 2-for-1 stock split means that each shareholder will have two shares for every share they previously owned.")
            elif topic in ["52 Week High", "52 Week Low"]:
                st.write(f"{topic} refers to the highest and lowest prices of a stock in the last 52 weeks.")
            elif topic in ["NSE", "BSE"]:
                st.write(f"{topic} refers to the National Stock Exchange and Bombay Stock Exchange, respectively.")
            elif topic == "Volume of Trading":
                st.write("Volume of trading represents the total number of shares traded in a specific period.")
            elif topic in ["High", "Low", "Open", "Close"]:
                st.write(f"{topic} represents the highest, lowest, opening, and closing prices of a stock in a given trading session.")
        else:
            st.write("Summary and formula for this topic are not available.")
    else:
        st.write(f"No topics found for your search in the {selected_topic} category.")
    
def main():
    st.title("How To Use the Financial Dashboard")
    st.write("This page serves as guide on what to expect and how to use this app.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["About", "Find Stocks", "In Depth Stock Analysis", "Glossary"])
    
    with tab3:
        indepth()
    with tab2:
        home()
    with tab1:
        about()
    with tab4:
        glossary()


# Check if the script is being run as the main module
if __name__ == "__main__":
    # Set Streamlit query parameters for refreshing the dashboard
    st.experimental_set_query_params()

    # Display the "How To Use" page
    main()
