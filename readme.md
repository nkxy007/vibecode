# WILLIAM O'NEIL'S CANSLIM Stock Analyzer with LLM and Technical Levels

This project is a Streamlit web application designed to analyze stocks based on William O'Neil's CANSLIM methodology. It leverages the power of Large Language Models (LLMs) like OpenAI's GPT-4o for data research and analysis, combined with technical calculations using `yfinance` for potential trade levels.

**Current Time Context:** Wednesday, April 2, 2025 at 8:13 AM AEDT (Sydney, Australia) - LLM responses may depend on their knowledge cutoff relative to this time.

## Description

The application takes a stock symbol as input and performs the following steps:

1.  **LLM Research:** Uses an LLM (via `client.responses.create`) to research the latest publicly available data relevant to each component of the CANSLIM framework (Current Earnings, Annual Earnings, New factors, Supply/Demand, Leadership, Institutional Sponsorship, Market Direction) and recent price/trend data.
2.  **LLM Formatting & Analysis:** Uses a second LLM call (via `client.chat.completions.create`) to analyze the researched data according to CANSLIM rules and generate a final recommendation (Buy, Sell, Hold, or Uncertain). It also formats the researched trend data.
3.  **Technical Level Calculation (if "Buy"):** If the CANSLIM recommendation is "Buy" and the current price was found during research, the application fetches historical stock data using `yfinance` and calculates potential support, resistance, stop-loss, and take-profit levels based on recent price action.

## Features

* Input a stock symbol for analysis.
* LLM-powered research for CANSLIM data points using potentially up-to-date information (depending on LLM capabilities like Browse).
* LLM-powered analysis and interpretation of researched data against CANSLIM criteria.
* Generates a final Buy/Sell/Hold/Uncertain recommendation based on the CANSLIM analysis.
* Researches and displays recent stock price and 1-month/3-month trends for both the stock and the market (S&P 500).
* Calculates and suggests potential trade levels (Support, Resistance, Stop Loss, Take Profit) using historical data if the CANSLIM recommendation is "Buy".

## Technology Stack

* **Python:** Core programming language.
* **Streamlit:** Framework for building the interactive web application.
* **OpenAI API (GPT-4o):** Used for data research, analysis, and formatting.
* **yfinance:** Library for fetching historical stock market data.
* **Pandas:** Used for data manipulation of historical stock data.
* **NumPy:** Used for numerical calculations.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Set Up OpenAI API Key:**
    * Create a file named `creds.py` in the root directory of the project.
    * Add your OpenAI API key to this file like this:
        ```python
        # creds.py
        OPENAI_KEY = "sk-..." # Replace sk-... with your actual OpenAI API key
        ```
    * **IMPORTANT:** Add `creds.py` to your `.gitignore` file to avoid accidentally committing your secret key to GitHub. Create a `.gitignore` file if you don't have one and add the line `creds.py`.

4.  **Install Dependencies:**
    * Make sure you have a `requirements.txt` file in your repository with the following content:
        ```txt
        # requirements.txt
        streamlit
        openai
        yfinance
        pandas
        numpy
        ```
    * Install the required libraries:
        ```bash
        pip install -r requirements.txt
        ```

## Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run william_app.py
    ```
    *(Replace `william_app.py` if your main application file has a different name)*

2.  **Interact with the App:**
    * Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.
    * Enter a valid stock symbol (e.g., `AAPL`, `NVDA`, `MSFT`).
    * Click the "Analyze [Symbol] with LLM" button.
    * Wait for the LLM research, analysis, and potential technical calculations to complete.
    * Review the CANSLIM breakdown, recommendation, trend analysis, and (if applicable) the calculated trade levels.

## Screenshots

*(Add screenshots here showing the app interface: input field, CANSLIM analysis section, recommendation, trend section, and the potential trade levels section if a "Buy" is recommended)*

* *App Input*
* *CANSLIM Analysis Output*
* *Trend Analysis Output*
* *Potential Trade Levels Output (Example)*

## Important Disclaimers

* **Not Financial Advice:** This application is for educational and demonstration purposes only. The analysis, recommendations, and calculated levels provided are NOT financial advice.
* **Data Accuracy:** The information relies on data researched by an LLM and fetched from `yfinance`. LLMs can make mistakes, hallucinate, or have outdated information. Financial data sources may also have inaccuracies or delays. **Always verify information from multiple reliable sources.**
* **Calculation Basis:** Support, resistance, stop-loss, and take-profit levels are calculated using simple historical price analysis (recent lows/highs). They are suggestions and not guaranteed market levels. Market conditions can change rapidly.
* **API Costs:** This application makes multiple calls to the OpenAI API for each analysis, which will incur costs based on your OpenAI account usage and pricing. Monitor your API usage.
* **Risk:** Trading stocks involves significant risk, including the potential loss of principal. Always conduct your own thorough research and consult with a qualified financial advisor before making any investment decisions.

## Future Enhancements (Ideas)

* Incorporate more sophisticated technical indicators (e.g., Moving Averages, RSI, MACD) into the analysis or level calculation.
* Add interactive charts (e.g., using Plotly) to visualize price history, support/resistance, and trends.
* Allow user configuration of parameters (e.g., lookback periods for S/R, risk/reward ratio).
* Implement more robust error
