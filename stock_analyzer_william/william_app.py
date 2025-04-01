# canslim_app_v7.py
import streamlit as st
import datetime
import openai
import re
import json
import creds
import yfinance as yf # Import yfinance
import pandas as pd    # Import pandas
import numpy as np     # Import numpy for calculations

# --- Initialize OpenAI Client (Best practice: outside functions if possible,
# but keeping inside due to potential Streamlit secrets scoping)

# --- Function to RESEARCH stock data via LLM (responses.create) ---
# (Keep the research_stock_data_llm_chat function from v6 exactly as it was)
def research_stock_data_llm_chat(stock_symbol, analysis_type="CANSLIM"):
    """
    Uses the custom client.responses.create(input=...) call and
    parses the specific JSON output structure provided.
    Relies on the model's potential Browse capability.
    """
    st.info(f"🤖 Asking LLM ({analysis_type}) to research data for {stock_symbol} via Custom Call...")
    try:
        client = openai.OpenAI(api_key=creds.OPENAI_KEY)
    except Exception as e:
        st.error(f"Error initializing OpenAI client for research: {e}")
        return None

    current_date_str = datetime.datetime.now().strftime('%Y-%m-%d') # Use current date

    # --- Define the Research Prompt (Make sure 'N' asks for Current Price) ---
    system_message = f"You are an AI research assistant finding the latest financial data for {stock_symbol.upper()}. Prioritize accuracy and recency. State if data is unavailable."
    if analysis_type == "CANSLIM":
        user_prompt = f"""
        **Task:** Research and gather the most recent, publicly available data points for a CANSLIM analysis of the stock "{stock_symbol.upper()}".
        **Context:** Today is approximately {current_date_str}. Use your Browse capabilities or latest knowledge to find the latest information.
        **Required Data Points:**
        * C - Current Quarterly Earnings: Latest reported Quarterly EPS Growth % Year-over-Year (YoY). Target: >25%.
        * A - Annual Earnings Growth: Estimated or historical 3-5 year annual EPS growth rate. Target: >25%.
        * N - New Product/Management/Highs: Any significant recent news (products, management changes, major contracts)? Is the stock price near its 52-week high? Provide **Current Price** and 52-week high.
        * S - Supply and Demand: Recent daily trading volume compared to average volume. Any notes on share structure (float, outstanding shares)? Look for signs of accumulation/distribution.
        * L - Leader or Laggard: What is the stock's industry? Is it considered a leader in its industry? Provide its Relative Strength (RS) rating or a qualitative description if the exact number isn't found. Target RS > 80-85.
        * I - Institutional Sponsorship: What is the recent trend in institutional ownership (increasing/decreasing)? Are there notable high-quality institutions holding it?
        * M - Market Direction: What is the current trend of the overall market (e.g., S&P 500 or Nasdaq)? (e.g., Confirmed Uptrend, Correction, Sideways).

        **Output Format:**
        Provide the gathered information clearly, labeling each point corresponding to the CANSLIM letter (C, A, N, S, L, I, M). Be factual and state if specific data cannot be found. Do NOT perform the analysis yet, just report the findings. Example:
        C: Latest Quarterly EPS Growth YoY: +30%
        A: 3-Year Annual EPS Growth Rate: +28%
        N: Recent news: Launched new AI chip. Current Price: $150, 52-Week High: $155.
        S: Volume today is 1.5x average volume. Shares Outstanding: 1B. Signs of accumulation.
        L: Industry: Semiconductors. Leader. RS Rating: 92.
        I: Institutional ownership increased by 5% last quarter. Major holders include Fund X, Fund Y.
        M: Market Trend (S&P 500): Confirmed Uptrend.
        """
        model_to_use = "gpt-4o"
        temperature_research = 0.3
    elif analysis_type == "Trend":
        user_prompt = f"""
        **Task:** Research the most recent price and trend data for the stock "{stock_symbol.upper()}" and the general market (S&P 500).
        **Context:** Today is approximately {current_date_str}. Use your Browse capabilities or latest knowledge to find the latest information.
        **Required Data Points:**
        * Latest known stock price for {stock_symbol.upper()}.
        * General price trend for {stock_symbol.upper()} over the past 1 month (approx).
        * General price trend for {stock_symbol.upper()} over the past 3 months (approx).
        * General price trend for the S&P 500 index over the past 1 month (approx).
        * General price trend for the S&P 500 index over the past 3 months (approx).

        **Output Format:**
        Provide the gathered information clearly and concisely. Do NOT add analysis, just report the data/trend observation. Example:
        Stock Price ({stock_symbol.upper()}): $XXX.XX
        Stock Trend (1M): Uptrend
        Stock Trend (3M): Sideways
        Market Trend (1M - S&P 500): Uptrend
        Market Trend (3M - S&P 500): Uptrend
        """
        model_to_use = "gpt-4o"
        temperature_research = 0.2
    else:
        st.error(f"Invalid analysis type for research: {analysis_type}")
        return None

    full_input = f"{system_message}\n\n{user_prompt}"
    try:
        response = client.responses.create(
            model=model_to_use,
            input=full_input,
            tools=[{"type": "web_search_preview"}],
            temperature=temperature_research
        )
        # --- Parsing Logic for the specific responses.create output ---
        researched_content = None
        # ... (Keep the parsing logic from v6 here) ...
        try:
            if response and hasattr(response, 'output') and isinstance(response.output, list) and len(response.output) > 0:
                # Assuming the relevant text is often in the second output item based on previous debugging/structure
                target_output_index = 1 if len(response.output) > 1 else 0
                output_item = response.output[target_output_index]

                if hasattr(output_item, 'content') and isinstance(output_item.content, list) and len(output_item.content) > 0:
                    for content_item in output_item.content:
                        if hasattr(content_item, 'type') and content_item.type == 'output_text':
                            if hasattr(content_item, 'text'):
                                researched_content = content_item.text.strip()
                                break # Found the text

            if researched_content is None:
                 st.warning(f"Could not find 'output_text' in the expected structure for {stock_symbol} ({analysis_type}). Response might be structured differently.")

        except AttributeError as ae:
            st.error(f"AttributeError while parsing response: {ae}. Unexpected response structure.")
            researched_content = None
        except Exception as e:
            st.error(f"Error during response parsing: {e}")
            researched_content = None

        # --- End of Parsing Logic ---

        if researched_content:
            st.success(f"✅ LLM Research successful for {stock_symbol} ({analysis_type}).")
            return researched_content
        else:
            st.error(f"🚨 LLM Research ({analysis_type}) failed to extract content for {stock_symbol}.")
            try:
                st.json(json.dumps(response.dict() if hasattr(response, 'dict') else str(response), indent=2))
            except Exception as dump_error:
                st.text(f"Could not dump response object: {response}, Error: {dump_error}")
            return None

    except openai.APIError as e:
        st.error(f"🚨 OpenAI API Error during LLM Research ({analysis_type}) for {stock_symbol}: {e}")
        return None
    except Exception as e:
        st.error(f"🚨 Unexpected error during LLM Research ({analysis_type}) for {stock_symbol}: {e}")
        return None


# --- Function to FORMAT researched data via LLM Chat Completions ---
# (Keep the format_researched_data_llm_chat function from v6 exactly as it was)
def format_researched_data_llm_chat(stock_symbol, researched_data, analysis_type="CANSLIM"):
    """
    Uses client.chat.completions.create to format the researched data
    into the final analysis output.
    """
    st.info(f"🤖 Asking LLM ({analysis_type}) to format researched data for {stock_symbol} via Chat Completion...")
    try:
        client = openai.OpenAI(api_key=creds.OPENAI_KEY)
    except Exception as e:
        st.error(f"Error initializing OpenAI client for formatting: {e}")
        return None if analysis_type == "Trend" else (None, None)

    # --- Define the Formatting Prompt (same prompts as before) ---
    if analysis_type == "CANSLIM":
        prompt = f"""
        **Task:** Analyze the provided research data for stock "{stock_symbol.upper()}" using William O'Neil's CANSLIM methodology and format the output.
        **Provided Research Data:**
        ```
        {researched_data}
        ```
        **Analysis and Formatting Instructions:**
        1. Go through each letter of CANSLIM (C, A, N, S, L, I, M).
        2. For each letter, analyze the relevant information from the "Provided Research Data".
        3. State the finding and briefly comment on whether it meets the typical CANSLIM criteria.
        4. If data for a specific point was noted as "not found" or is missing, explicitly state that.
        5. Present the analysis clearly, point-by-point for each letter.
        6. Conclude with a final recommendation on a **separate, final line**, formatted *exactly* like this:
           **Recommendation: [Buy/Sell/Hold/Uncertain]**
           Base the recommendation *only* on the analysis of the provided data according to the CANSLIM rules.
        """
        model_to_use = "gpt-4o"
        max_tokens_format = 1000
        temperature_format = 0.5

        try:
            # *** Using client.chat.completions.create for Formatting/Analysis ***
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing pre-researched stock data using CANSLIM and formatting the results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature_format,
                max_tokens=max_tokens_format
            )
            # --- Standard Response Parsing ---
            formatted_content = None
            if response.choices:
                message = response.choices[0].message
                if hasattr(message, 'content'):
                    formatted_content = message.content.strip()

            if not formatted_content:
                st.error(f"🚨 LLM Formatting (CANSLIM) failed for {stock_symbol}.")
                return None, None

            st.success(f"✅ LLM Formatting complete for {stock_symbol} (CANSLIM).")
            # --- Parse Recommendation ---
            recommendation = "Uncertain"
            match = re.search(r"\*\*Recommendation:\s*(Buy|Sell|Hold|Uncertain)\*\*$", formatted_content, re.IGNORECASE | re.MULTILINE)
            if match: recommendation = match.group(1).capitalize()
            # --- Prepare Final Analysis Text ---
            analysis_text = f"**CANSLIM Analysis for {stock_symbol.upper()}** (Data researched & analyzed by LLM: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n"
            analysis_text += formatted_content
            analysis_text += "\n\n***\n*Disclaimer: Analysis based on LLM-researched data. Verify all data. Not financial advice.*"
            return analysis_text, recommendation

        except openai.APIError as e:
            st.error(f"🚨 OpenAI API Error (Formatting CANSLIM): {e}")
            return None, None
        except Exception as e:
            st.error(f"🚨 Unexpected error (Formatting CANSLIM): {e}")
            return None, None

    elif analysis_type == "Trend":
        prompt = f"""
        **Task:** Format the provided research data for stock "{stock_symbol.upper()}" into the specified line-by-line output.
        **Provided Research Data:**
        ```
        {researched_data}
        ```
        **Formatting Instructions:**
        Extract the required information and present it *exactly* in this format:

        Recent Price ({stock_symbol.upper()}): [Price or "Data unavailable"]
        Stock Trend (1M): [Trend or "Data unavailable"]
        Stock Trend (3M): [Trend or "Data unavailable"]
        Market Trend (1M - S&P 500): [Trend or "Data unavailable"]
        Market Trend (3M - S&P 500): [Trend or "Data unavailable"]
        """
        model_to_use = "gpt-4o"
        max_tokens_format = 200
        temperature_format = 0.1

        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You format pre-researched stock trend data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature_format,
                max_tokens=max_tokens_format
            )
            # --- Standard Response Parsing ---
            formatted_content = None
            if response.choices:
                 message = response.choices[0].message
                 if hasattr(message, 'content'):
                      formatted_content = message.content.strip()

            if not formatted_content:
                st.error(f"🚨 LLM Formatting (Trend) failed for {stock_symbol}.")
                return None

            st.success(f"✅ LLM Formatting complete for {stock_symbol} (Trend).")
            trend_analysis_text = formatted_content
            trend_analysis_text += "\n\n*Note: Price/trend data researched via LLM. Verify.*"
            return trend_analysis_text

        except openai.APIError as e:
            st.error(f"🚨 OpenAI API Error (Formatting Trend): {e}")
            return None
        except Exception as e:
            st.error(f"🚨 Unexpected error (Formatting Trend): {e}")
            return None
    else:
        st.error(f"Invalid analysis type for formatting: {analysis_type}")
        return None if analysis_type == "Trend" else (None, None)


# --- NEW Function to Calculate Trade Levels ---
def calculate_trade_levels(stock_symbol, current_price_ref):
    """
    Calculates potential support, resistance, stop-loss, and take-profit levels
    based on historical data using yfinance and pandas.

    Args:
        stock_symbol (str): The stock ticker.
        current_price_ref (float): The current price (from LLM research) for reference.

    Returns:
        dict: A dictionary containing calculated levels, or None if calculation fails.
    """
    st.info(f"⚙️ Calculating potential trade levels for {stock_symbol}...")
    try:
        # Fetch 7 months of data to ensure enough for 6 months analysis + rolling periods
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="7mo", interval="1d") # Daily data
        print("--"*40)
        print(f"price history: \n {hist}")
        print("--"*40)

        if hist.empty or len(hist) < 60: # Need at least ~3 months of data for reasonable S/R
            st.warning(f"Insufficient historical data found for {stock_symbol} (need ~3-6 months). Cannot calculate levels.")
            return None

        # --- Simple Support/Resistance Calculation ---
        # Using rolling minima/maxima over different periods within the last ~6 months (approx 126 trading days)
        hist_6m = hist.tail(126) # Approx 6 months

        # Support Levels (recent lows)
        low_60d = hist_6m['Low'].tail(60).min() # Min low over last ~3 months
        low_180d = hist['Low'].tail(180).min() # Min low over last ~6 months (using full 180 days available)

        # Resistance Levels (recent highs)
        high_60d = hist_6m['High'].tail(60).max() # Max high over last ~3 months
        high_180d = hist['High'].tail(180).max() # Max high over last ~6 months

        # --- Entry Suggestion ---
        # Use the LLM's current price as a reference, suggest entry near support if applicable
        entry_suggestion = f"Current Price Ref: ${current_price_ref:.2f}. Consider entry near Support 1 (${low_60d:.2f}) on pullback, or on confirmation above Resistance 1 (${high_60d:.2f})."

        # --- Stop Loss Calculation ---
        # Place below a key support level. Let's use the 60-day low.
        # Add a small buffer (e.g., 1-2% of the support level)
        stop_loss_price = low_60d * 0.99 # 1% buffer below 60d low
        stop_loss_suggestion = f"Consider Stop Loss below Support 1, e.g., around ${stop_loss_price:.2f}."

        # --- Take Profit Calculation ---
        # Option 1: Target Resistance 1
        tp1_resistance = high_60d
        # Option 2: Risk/Reward Ratio (e.g., 2:1)
        risk_per_share = current_price_ref - stop_loss_price
        if risk_per_share > 0: # Avoid division by zero or negative risk
            tp2_rr = current_price_ref + (2 * risk_per_share) # 2:1 Reward/Risk
            take_profit_suggestion = f"Potential Take Profit Levels: Near Resistance 1 (${tp1_resistance:.2f}), or based on 2:1 R/R (~${tp2_rr:.2f})."
        else:
            take_profit_suggestion = f"Potential Take Profit Level: Near Resistance 1 (${tp1_resistance:.2f}). (R/R calc skipped due to low risk). "


        st.success(f"✅ Trade level calculation complete for {stock_symbol}.")

        return {
            "Entry Suggestion": entry_suggestion,
            "Support 1 (60d Low)": f"${low_60d:.2f}",
            "Support 2 (180d Low)": f"${low_180d:.2f}",
            "Resistance 1 (60d High)": f"${high_60d:.2f}",
            "Resistance 2 (180d High)": f"${high_180d:.2f}",
            "Stop Loss Suggestion": stop_loss_suggestion,
            "Take Profit Suggestion": take_profit_suggestion
        }

    except Exception as e:
        st.error(f"🚨 Error calculating trade levels for {stock_symbol}: {e}")
        # import traceback; st.error(traceback.format_exc()) # Uncomment for detailed debug
        return None

# --- Streamlit App UI ---
st.set_page_config(page_title="CANSLIM Stock Analyzer", layout="wide")

st.title("📈 CANSLIM Stock Analyzer (v7 - Added SL/TP Calculation)")
st.caption("Uses LLM for research/formatting & Calculates potential SL/TP levels if 'Buy'.")
st.warning("Note: Uses LLM API (potentially twice per analysis) & yfinance. Check costs. Data/calculations are estimates. **NOT FINANCIAL ADVICE.**")

# --- Input Section ---
st.header("Stock Input")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, NVDA, MSFT):", "").upper()

# --- Analysis Trigger ---
analyze_button = st.button(f"Analyze {stock_symbol} with LLM" if stock_symbol else "Analyze Stock")

# --- Analysis Output Section ---
if analyze_button and stock_symbol:

    # Shared variables for results
    researched_canslim_data = None
    analysis_result = None
    recommendation_result = None
    current_price = None # Variable to store parsed price

    # --- 1. Research Phase (CANSLIM) ---
    with st.spinner(f"🤖 Researching CANSLIM data for {stock_symbol}..."):
        researched_canslim_data = research_stock_data_llm_chat(stock_symbol, "CANSLIM")

    if researched_canslim_data:
        # --- Attempt to Parse Current Price from Research Data ---
        # Look for "Current Price: $XXX.XX" in the N section findings
        price_pattern = r"Current Price:\s*\$?([\d,]+\.?\d*)"
        price_match = re.search(price_pattern, researched_canslim_data, re.IGNORECASE)
        if price_match:
            try:
                current_price = float(price_match.group(1).replace(',', ''))
                st.info(f"Parsed current price reference: ${current_price:.2f}")
            except ValueError:
                st.warning("Could not parse current price number from research data.")
                current_price = None # Ensure it's None if parsing fails
        else:
            st.warning("Could not find 'Current Price:' pattern in research data (needed for SL/TP calculation).")
            current_price = None


        # --- 2. Formatting/Analysis Phase (CANSLIM) ---
        with st.spinner(f"🤖 Formatting CANSLIM analysis for {stock_symbol}..."):
            analysis_result, recommendation_result = format_researched_data_llm_chat(stock_symbol, researched_canslim_data, "CANSLIM")

        if analysis_result and recommendation_result:
            # Display CANSLIM results
            st.header(f"CANSLIM Analysis for {stock_symbol}")
            st.subheader("CANSLIM Analysis Details (Researched & Formatted by LLM):")
            st.markdown(analysis_result)

            st.subheader("Overall CANSLIM Recommendation (from LLM):")
            # Display recommendation styling
            if recommendation_result == "Buy": st.success(f"**{recommendation_result}**")
            elif recommendation_result == "Sell": st.error(f"**{recommendation_result}**")
            elif recommendation_result == "Hold": st.warning(f"**{recommendation_result}**")
            else: st.info(f"**{recommendation_result}**")

            # --- 3. Calculate & Display Trade Levels (ONLY IF 'Buy' and price found) ---
            if recommendation_result == "Uncertain":
                if current_price is not None:
                    st.markdown("---")
                    st.header("Potential Trade Levels (Calculated)")
                    trade_levels = None
                    with st.spinner(f"⚙️ Calculating potential trade levels for {stock_symbol}..."):
                         trade_levels = calculate_trade_levels(stock_symbol, current_price)

                    if trade_levels:
                        # Display the calculated levels clearly
                        st.markdown(f"**Entry Suggestion:** {trade_levels.get('Entry Suggestion', 'N/A')}")
                        st.markdown(f"**Support 1 (Approx 60d Low):** {trade_levels.get('Support 1 (60d Low)', 'N/A')}")
                        st.markdown(f"**Support 2 (Approx 180d Low):** {trade_levels.get('Support 2 (180d Low)', 'N/A')}")
                        st.markdown(f"**Resistance 1 (Approx 60d High):** {trade_levels.get('Resistance 1 (60d High)', 'N/A')}")
                        st.markdown(f"**Resistance 2 (Approx 180d High):** {trade_levels.get('Resistance 2 (180d High)', 'N/A')}")
                        st.markdown(f"**Stop Loss Suggestion:** {trade_levels.get('Stop Loss Suggestion', 'N/A')}")
                        st.markdown(f"**Take Profit Suggestion:** {trade_levels.get('Take Profit Suggestion', 'N/A')}")
                        st.caption("*Levels calculated based on historical daily data (approx periods). These are suggestions, not guarantees. Market conditions apply.*")
                    else:
                        st.warning("Could not calculate potential trade levels.")
                else:
                    st.warning("Skipping SL/TP calculation because current price could not be determined from LLM research.")


            st.markdown("---") # Separator before Trend section

            # --- 4. Research Phase (Trend) ---
            researched_trend_data = None
            with st.spinner(f"🤖 Researching Price/Trend data for {stock_symbol}..."):
                researched_trend_data = research_stock_data_llm_chat(stock_symbol, "Trend")

            if researched_trend_data:
                # --- 5. Formatting Phase (Trend) ---
                trend_data_text = None
                with st.spinner(f"🤖 Formatting Price/Trend data for {stock_symbol}..."):
                     trend_data_text = format_researched_data_llm_chat(stock_symbol, researched_trend_data, "Trend")

                if trend_data_text:
                    st.header(f"Recent Price & Trend Analysis for {stock_symbol}")
                    st.markdown(trend_data_text) # Displays formatted trend data + note
                else:
                    st.warning("Could not format price and trend data.")
            else:
                st.warning("Could not research price and trend data via LLM.")
        else:
            st.error("Analysis halted because CANSLIM formatting/analysis failed.")
    else:
        st.error("Analysis halted because CANSLIM data research failed.")


elif analyze_button and not stock_symbol:
    st.error("⚠️ Please enter a stock symbol.")

st.markdown("---")
# Get current time in AEDT (Sydney time)
now_aedt = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=11))) # AEDT is UTC+11
current_run_time_aedt = now_aedt.strftime('%Y-%m-%d %H:%M:%S %Z')
st.markdown(f"App run location: Sydney, Australia. App run at: {current_run_time_aedt}. Uses LLM API + yfinance.")