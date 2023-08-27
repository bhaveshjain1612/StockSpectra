import streamlit as st

# Define the indicators and their details
indicators = {
    "CCI": {
        "full_name": "Commodity Channel Index (CCI)",
        "description": "The Commodity Channel Index (CCI) is a tool that helps investors understand the momentum of an investment. Think of it as a thermometer for stocks or other investments. It tells you if an investment might be 'too hot' (overbought) or 'too cold' (oversold).",
        "how_it_works": "The CCI measures how the current price of an investment compares to its average price over a certain period. If the CCI is above zero, it means the price is higher than its average. If it's below zero, the price is lower than its average.",
        "usefulness": "The CCI can give hints about potential new trends. For example, if the CCI moves from a low value to above 100, it might mean that the price is starting a new upward trend. On the other hand, if the CCI drops below -100, a downward trend might be starting. This can help investors decide when to buy or sell.",
        "calculation": r'''
\begin{align*}
\text{CCI} & = \frac{\text{Typical Price} - \text{20-period SMA of TP}}{\text{Mean Deviation} \times 0.015} \\
\text{Where:} \\
\text{Typical Price (TP)} & = \frac{\text{High} + \text{Low} + \text{Close}}{3}
\end{align*}
'''
    },
    "VPT": {
        "full_name": "Volume Price Trend (VPT)",
        "description": "Volume Price Trend (VPT) is a technical analysis indicator that combines price and volume data. It helps in determining the strength of price movements.",
        "how_it_works": "VPT is similar to the On-Balance Volume (OBV) but incorporates price changes. It adds or subtracts a multiple of the percentage change in share price trend and current volume, depending upon the movement of the price.",
        "usefulness": "VPT can provide insights into the strength of a price trend, help in confirming price trends, and indicate potential reversals when there's a divergence between VPT and price.",
        "calculation": r'''
\begin{align*}
\text{VPT} & = \text{Previous VPT} + \text{Volume} \times (\text{Close}_{\text{today}} - \text{Close}_{\text{yesterday}})
\end{align*}
'''
    },
    "ADX": {
        "full_name": "Average Directional Index (ADX)",
        "description": "The Average Directional Index (ADX) is a technical indicator that measures the strength of a trend. It doesn't indicate the direction of the trend, just its strength.",
        "how_it_works": "ADX ranges between 0 to 100. Generally, ADX readings below 20 indicate a weak trend or a non-trending market, while readings above 20 indicate a strong trend.",
        "usefulness": "ADX can help traders identify the strongest and most profitable trends, provide insights into whether a trend is strengthening or weakening, and assist in filtering out price consolidations.",
        "calculation": r"ADX = \frac{ \text{Moving Average of DX} }{ \text{Period} } \text{ where DX is the difference between +DI and -DI}"
    },
        "RSI": {
        "full_name": "Relative Strength Index (RSI)",
        "description": "RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions in a traded security.",
        "how_it_works": "RSI compares the magnitude of recent gains to recent losses to determine overbought or oversold conditions.",
        "usefulness": "RSI can help traders identify potential buy or sell opportunities, recognize potential price reversals, and gauge the strength of a trend.",
        "calculation": r"RSI = 100 - \frac{100}{1 + RS} \text{ where } RS \text{ is the average of } n \text{ days' up closes divided by the average of } n \text{ days' down closes.}"
    },
    "EMA": {
        "full_name": "Exponential Moving Average (EMA)",
        "description": "EMA is a type of moving average that gives more weight to recent prices, making it more responsive to new information.",
        "how_it_works": "EMA reacts faster to price changes compared to a Simple Moving Average (SMA).",
        "usefulness": "EMA can help traders identify trend direction, determine support and resistance levels, and recognize potential entry and exit points.",
        "calculation": r"EMA = (Close - Previous EMA) \times (2 / (Selected Time Period + 1)) + Previous EMA"
    },
    "SMA": {
        "full_name": "Simple Moving Average (SMA)",
        "description": "SMA is an arithmetic moving average calculated by adding recent closing prices and then dividing that by the number of time periods.",
        "how_it_works": "SMA provides a smoothed line that tracks the price over a given period.",
        "usefulness": "SMA can help traders identify trend direction, recognize potential price reversals, and determine support and resistance levels.",
        "calculation": r"SMA = \frac{Sum \ of \ Closing \ Prices}{Number \ of \ Periods}"
    },
    "VWAP": {
        "full_name": "Volume Weighted Average Price (VWAP)",
        "description": "VWAP is the average price a security has traded at throughout the day, based on both volume and price.",
        "how_it_works": "VWAP provides a benchmark that gives an idea of the average price at which investors have bought a security over a given time frame.",
        "usefulness": "VWAP can help traders determine the market direction, recognize fair value for a security, and identify potential buy or sell signals.",
        "calculation": r"VWAP = \frac{\sum (Price \times Volume)}{\sum Volume}"
    },
    "OBV": {
        "full_name": "On-Balance Volume (OBV)",
        "description": "OBV is a momentum indicator that uses volume flow to predict changes in stock price.",
        "how_it_works": "OBV measures buying and selling pressure by adding volume on up days and subtracting volume on down days.",
        "usefulness": "OBV can help traders identify potential price reversals, confirm price trends, and recognize accumulation or distribution phases.",
        "calculation": r'''
\begin{align*}
\text{If } \text{Close}_{\text{today}} > \text{Close}_{\text{yesterday}} & : \\
\text{OBV}_{\text{today}} & = \text{OBV}_{\text{yesterday}} + \text{Volume}_{\text{today}} \\
\text{If } \text{Close}_{\text{today}} < \text{Close}_{\text{yesterday}} & : \\
\text{OBV}_{\text{today}} & = \text{OBV}_{\text{yesterday}} - \text{Volume}_{\text{today}}
\end{align*}
'''
    },
    "Williams %R": {
        "full_name": "Williams %R",
        "description": "Williams %R, also known as the Williams Percent Range, is a type of momentum indicator that moves between 0 and -100 and measures overbought and oversold levels.",
        "how_it_works": "The Williams %R oscillates between 0 to -100. Readings from 0 to -20 are considered overbought, and readings from -80 to -100 are considered oversold.",
        "usefulness": "Williams %R can help traders identify potential price reversals, recognize overbought or oversold conditions, and confirm momentum shifts.",
        "calculation": r"Williams \ \%R = \frac{Highest \ High \ - \ Close}{Highest \ High \ - \ Lowest \ Low} \times -100"
    },
        "MACD": {
        "full_name": "Moving Average Convergence Divergence (MACD)",
        "description": "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.",
        "how_it_works": "MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. The result of that calculation is the MACD line. A nine-day EMA of the MACD, called the 'signal line,' is then plotted on top of the MACD line.",
        "usefulness": "MACD can help traders identify potential buy or sell opportunities around crossovers of the MACD line and the signal line, recognize potential overbought or oversold conditions, and confirm the strength of a trend.",
        "calculation": r"MACD = 12-Period \ EMA - 26-Period \ EMA \ and \ Signal \ Line = 9-Period \ EMA \ of \ MACD"
    },
    "Bollinger Bands": {
        "full_name": "Bollinger Bands",
        "description": "Bollinger Bands consist of a middle band being an N-period simple moving average (SMA), an upper band at K times an N-period standard deviation above the middle band, and a lower band at K times an N-period standard deviation below the middle band.",
        "how_it_works": "Bollinger Bands are able to adapt to volatility in the price of a stock. A band squeeze denotes a period of low volatility and is considered by traders to be a potential indicator of future increased volatility.",
        "usefulness": "Bollinger Bands can help traders identify periods of high or low volatility, recognize potential buy or sell opportunities, and determine overbought or oversold conditions.",
        "calculation": r"Middle \ Band = 20-day \ SMA, \ Upper \ Band = 20-day \ SMA + (20-day \ standard \ deviation \times 2), \ Lower \ Band = 20-day \ SMA - (20-day \ standard \ deviation \times 2)"
    },
    "Standard Deviation": {
        "full_name": "Standard Deviation",
        "description": "Standard Deviation is a statistical measure of volatility. It represents how spread out the numbers are in a data set.",
        "how_it_works": "In finance, standard deviation is used to measure price volatility and can help gauge the risk associated with a particular investment.",
        "usefulness": "Standard Deviation can help traders and investors understand the volatility of an investment, gauge the risk associated with a particular security or portfolio, and determine the dispersion of returns.",
        "calculation": r"Standard \ Deviation = \sqrt{\frac{\sum (X - \text{Mean})^2}{N}}"
    },
    "Net Income": {
        "full_name": "Net Income",
        "description": "Net Income represents a company's total earnings or profit.",
        "how_it_works": "It's calculated by subtracting total expenses from total revenues. It provides a clear picture of the overall profitability of a company over a specific period of time.",
        "usefulness": "Net Income is a key metric to assess a company's profitability and is often used by investors to compare the profitability of companies within the same industry.",
        "calculation": r"\text{Net Income} = \text{Total Revenues} - \text{Total Expenses}"
    },
        "ROA": {
        "full_name": "Return on Assets (ROA)",
        "description": "ROA is a measure of how effectively a company's assets are being used to generate profits.",
        "how_it_works": "It's calculated by dividing net income by total assets. This ratio gives an idea of how efficiently the company is converting its investment in assets into net income.",
        "usefulness": "ROA is useful for comparing the profitability of companies in the same industry and for understanding if a company is generating enough profit from its assets.",
        "calculation": r"\text{ROA} = \frac{\text{Net Income}}{\text{Total Assets}}"
    },

    "Debt": {
        "full_name": "Debt",
        "description": "Debt refers to the amount of money borrowed by a company and due for repayment.",
        "how_it_works": "Companies can raise capital either through equity (like issuing shares) or through debt (like taking loans).",
        "usefulness": "Analyzing a company's debt levels helps investors understand its financial health and its ability to meet its financial obligations."
    },
        "ROE": {
        "full_name": "Return on Equity (ROE)",
        "description": "ROE measures a company's profitability by revealing how much profit a company generates with the money shareholders have invested.",
        "how_it_works": "It's calculated by dividing net income by shareholder's equity. This ratio indicates how well the company is generating earnings from its equity investments.",
        "usefulness": "ROE is useful for comparing the profitability of companies in the same sector and understanding the efficiency of generating profits from shareholders' equity.",
        "calculation": r"\text{ROE} = \frac{\text{Net Income}}{\text{Shareholder's Equity}}"
    },

    "Free Cash Flow": {
        "full_name": "Free Cash Flow",
        "description": "Free Cash Flow (FCF) represents the cash a company generates after accounting for cash outflows to support operations and maintain its capital assets.",
        "how_it_works": "It's the cash produced by the company's normal business operations after deducting capital expenditures.",
        "usefulness": "FCF is a key indicator of a company's financial flexibility and its ability to generate cash. It's often used by investors to assess the quality of a company's earnings.",
        "calculation": r"\text{FCF} = \text{Operating Cash Flow} - \text{Capital Expenditures}"
    },

    "ROCE": {
        "full_name": "Return on Capital Employed (ROCE)",
        "description": "ROCE is a financial metric that determines how efficiently a company is generating profits from its capital.",
        "how_it_works": "It's calculated by dividing Earnings Before Interest and Tax (EBIT) by capital employed. It gives an idea of how well the company is using its capital to generate profits.",
        "usefulness": "ROCE is useful for comparing the profitability and efficiency of companies in the same sector.",
        "calculation": r"\text{ROCE} = \frac{\text{Earnings Before Interest and Tax (EBIT)}}{\text{Capital Employed}}"
    },

    "Basic EPS": {
        "full_name": "Basic Earnings Per Share (EPS)",
        "description": "EPS measures the amount of net income earned per share of stock outstanding.",
        "how_it_works": "It's calculated by dividing the net income by the average number of shares outstanding during a period.",
        "usefulness": "EPS is a key metric used by investors to assess a company's profitability on a per-share basis.",
        "calculation": r"\text{EPS} = \frac{\text{Net Income} - \text{Dividends on Preferred Stock}}{\text{Average Outstanding Shares}}"
    },

    "Current Ratio": {
        "full_name": "Current Ratio",
        "description": "The current ratio is a liquidity ratio that measures a company's ability to cover its short-term obligations with its short-term assets.",
        "how_it_works": "It's calculated by dividing current assets by current liabilities. A ratio above 1 indicates that the company has more assets than liabilities.",
        "usefulness": "The current ratio helps investors assess a company's short-term financial health and its ability to pay off its short-term liabilities with its short-term assets.",
        "calculation": r"\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}"
    },

    "Net Profit Margin": {
        "full_name": "Net Profit Margin",
        "description": "Net Profit Margin is a profitability ratio that shows how much of each dollar of revenues is kept as net profit.",
        "how_it_works": "It's calculated by dividing net profit by total revenue and then multiplying by 100 to get a percentage.",
        "usefulness": "The net profit margin helps investors assess how effectively a company is converting its revenues into actual profit.",
        "calculation": r"\text{Net Profit Margin} = \frac{\text{Net Profit}}{\text{Total Revenue}} \times 100\%"
    },

    "DE Ratio": {
        "full_name": "Debt-to-Equity (DE) Ratio",
        "description": "The DE ratio is a measure of a company's financial leverage, indicating the proportion of equity and debt a company is using to finance its assets.",
        "how_it_works": "It's calculated by dividing total liabilities by shareholder's equity. A high DE ratio indicates that a company may have too much debt.",
        "usefulness": "The DE ratio helps investors understand the risk associated with a company's debt levels.",
        "calculation": r"\text{DE Ratio} = \frac{\text{Total Liabilities}}{\text{Shareholder's Equity}}"
    },
    # ... add the rest of the indicators similarly
}
def main():
# Create a select box in the Streamlit app
    selected_indicator = st.selectbox("Select an Indicator", list(indicators.keys()))

    # Display the details of the selected indicator
    st.write("###", indicators[selected_indicator]["full_name"])
    st.write("**What is it?**")
    st.write(indicators[selected_indicator]["description"])
    st.write("**How does it work?**")
    st.write(indicators[selected_indicator]["how_it_works"])
    st.write("**Why is it useful?**")
    st.write(indicators[selected_indicator]["usefulness"])
    try:
        st.write("**How is it calculated?**")
        st.latex(indicators[selected_indicator]["calculation"])
    except:
        st.write("No calculations")
    x = 'https://www.investopedia.com/search?q='+indicators[selected_indicator]["full_name"].replace(" ","+")
    st.write(f"For more details, check out [this Link]( {x} )")

# Run the app
if __name__ == "__main__":
    main()
