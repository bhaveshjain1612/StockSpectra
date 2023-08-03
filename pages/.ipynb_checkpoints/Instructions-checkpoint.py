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
    
def main():
    st.title("How To Use the Financial Dashboard")
    st.write("This page serves as guide on what to expect and how to use this app.")
    
    tab1, tab2, tab3 = st.tabs(["About", "Find Stocks", "In Depth"])
    
    with tab3:
        indepth()
    with tab2:
        home()
    with tab1:
        about()


# Check if the script is being run as the main module
if __name__ == "__main__":
    # Set Streamlit query parameters for refreshing the dashboard
    st.experimental_set_query_params()

    # Display the "How To Use" page
    main()
