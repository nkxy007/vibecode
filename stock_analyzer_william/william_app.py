# canslim_app_v11.py
import streamlit as st
import datetime
import openai
import re
import json
import creds
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
import pandas_ta as ta
from scipy.signal import find_peaks


# --- Initialize OpenAI Client (Helper Function) ---
def get_openai_client():
    """Initializes and returns the OpenAI client."""
    try:
        client = openai.OpenAI(api_key=creds.OPENAI_KEY)
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None


# --- Image Encoding Function ---
def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"Error encoding image: File not found at {image_path}")
        return None
    except Exception as e:
        st.error(f"Error encoding image {image_path}: {e}")
        return None


# --- Function to RESEARCH stock data via LLM (responses.create) ---
def research_stock_data_llm_chat(stock_symbol, analysis_type="CANSLIM"):
    """Uses client.responses.create(input=...) for research."""
    st.info(
        f"🤖 Asking LLM ({analysis_type}) to research data for {stock_symbol} via Custom Call..."
    )
    client = get_openai_client()
    if not client:
        return None
    # Using provided time context
    current_date_str = "April 2, 2025"  # Based on context
    system_message = f"You are an AI research assistant finding the latest financial data for {stock_symbol.upper()}. Prioritize accuracy and recency up to {current_date_str}. State if data is unavailable."
    # --- Prompts (CANSLIM/Trend - same as baseline) ---
    if analysis_type == "CANSLIM":
        user_prompt = f"""**Task:** Research CANSLIM data for "{stock_symbol.upper()}". **Context:** Today is ~{current_date_str}. **Required:** C(Q EPS% YoY), A(Ann EPS% 3-5yr), N(News? **Current Price**? 52wk High?), S(Vol vs Avg? Accum/Dist? Structure?), L(Industry Leader? RS Rating/Desc?), I(Inst. Own Trend? Quality?), M(Market Trend S&P/Nasdaq?). **Format:** Label points C, A, N... Factual. State if missing. Ex: N: News. Current Price: $150, 52-Week High: $155."""  # Concise
        model_to_use = "gpt-4o"
        temperature_research = 0.3
    elif analysis_type == "Trend":
        user_prompt = f"""**Task:** Research price/trend for "{stock_symbol.upper()}" & S&P 500. **Context:** Today ~{current_date_str}. **Required:** Latest Stock Price? 1M Stock Trend? 3M Stock Trend? 1M Market Trend? 3M Market Trend? **Format:** Line-by-line. Ex: Stock Price ({stock_symbol.upper()}): $XXX.XX"""  # Concise
        model_to_use = "gpt-4o"
        temperature_research = 0.2
        print("--" * 40)
        print(user_prompt)
        print(f"searching for {analysis_type} data")
    else:
        st.error(f"Invalid research type: {analysis_type}")
        return None

    full_input = f"{system_message}\n\n{user_prompt}"
    try:
        response = client.responses.create(
            model=model_to_use,
            input=full_input,
            tools=[{"type": "web_search_preview"}],
            temperature=temperature_research,
        )
        # --- Parsing Logic (Assuming structure based on previous findings) ---
        print("--" * 40)
        print(f"Response from AI: {response}")
        print("--" * 40)
        researched_content = None
        try:
            if (
                response
                and hasattr(response, "output")
                and isinstance(response.output, list)
                and len(response.output) > 0
            ):
                target_idx = 1 if len(response.output) > 1 else 0
                output_item = response.output[target_idx]
                if (
                    hasattr(output_item, "content")
                    and isinstance(output_item.content, list)
                    and output_item.content
                ):
                    for item in output_item.content:
                        if (
                            hasattr(item, "type")
                            and item.type == "output_text"
                            and hasattr(item, "text")
                        ):
                            researched_content = item.text.strip()
                            break
            if researched_content is None:
                st.warning(
                    f"Could not parse 'output_text' for {stock_symbol} ({analysis_type})."
                )
        except Exception as e:
            st.error(f"Parse error: {e}")
            researched_content = None
        # --- End Parsing ---
        if researched_content:
            st.success(f"✅ LLM Research successful ({analysis_type}).")
            return researched_content
        else:
            st.error(f"🚨 LLM Research failed ({analysis_type}).")
            return None
    except openai.APIError as e:
        st.error(f"🚨 API Error (Research {analysis_type}): {e}")
        return None
    except Exception as e:
        st.error(f"🚨 Error (Research {analysis_type}): {e}")
        return None


# --- Function to FORMAT researched data via LLM (chat.completions) ---
def format_researched_data_llm_chat(
    stock_symbol, researched_data, analysis_type="CANSLIM"
):
    """Uses client.chat.completions.create to format/analyze..."""
    st.info(
        f"🤖 Asking LLM ({analysis_type}) to format researched data via Chat Completion..."
    )
    client = get_openai_client()
    if not client:
        return None if analysis_type == "Trend" else (None, None)
    # --- Prompts and Logic (CANSLIM/Trend - same as baseline) ---
    if analysis_type == "CANSLIM":
        prompt = f"""**Task:** Analyze provided data for "{stock_symbol.upper()}" using CANSLIM methodology and format output. **Data:** ``` {researched_data} ``` **Instructions:** 1. Analyze each letter (C,A,N...). 2. Comment vs criteria. 3. State if missing. 4. Present point-by-point. 5. Conclude: **Recommendation: [Buy/Sell/Hold/Uncertain]** (separate line, based only on CANSLIM)."""  # Concise
        model = "gpt-4o"
        max_tok = 1000
        temp = 0.5
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You analyze pre-researched data via CANSLIM.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_tok,
            )
            content = (
                response.choices[0].message.content.strip()
                if response.choices
                else None
            )
            if not content:
                st.error(f"🚨 LLM Format (CANSLIM) failed.")
                return None, None
            st.success(f"✅ LLM Formatting complete (CANSLIM).")
            rec = "Uncertain"
            match = re.search(
                r"\*\*Recommendation:\s*(Buy|Sell|Hold|Uncertain)\*\*$",
                content,
                re.I | re.M,
            )
            if match:
                rec = match.group(1).capitalize()
            text = f"**CANSLIM Analysis for {stock_symbol.upper()}** (LLM Analyzed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n{content}\n\n***\n*Disclaimer...*"  # Concise
            return text, rec
        except Exception as e:
            st.error(f"🚨 Error (Format CANSLIM): {e}")
            return None, None
    elif analysis_type == "Trend":
        prompt = f"""**Task:** Format provided data for "{stock_symbol.upper()}". **Data:** ``` {researched_data} ``` **Format:** Recent Price ({stock_symbol.upper()}): [Price]\nStock Trend (1M): [Trend]\nStock Trend (3M): [Trend]\nMarket Trend (1M - S&P 500): [Trend]\nMarket Trend (3M - S&P 500): [Trend]"""  # Concise
        model = "gpt-4o"
        max_tok = 200
        temp = 0.1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You format trend data."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_tok,
            )
            content = (
                response.choices[0].message.content.strip()
                if response.choices
                else None
            )
            if not content:
                st.error(f"🚨 LLM Format (Trend) failed.")
                return None
            st.success(f"✅ LLM Formatting complete (Trend).")
            text = content + "\n\n*Note: Verify trend data.*"
            return text
        except Exception as e:
            st.error(f"🚨 Error (Format Trend): {e}")
            return None
    else:
        st.error(f"Invalid format type: {analysis_type}")
        return None if analysis_type == "Trend" else (None, None)


# --- Function to Calculate Trade Levels (Accepts hist_data) ---
def calculate_trade_levels(stock_symbol, current_price_ref, hist_data):
    """
    Calculates a comprehensive set of potential trade levels including:
    - Standard Daily Pivot Points (based on previous day HLC)
    - Key Moving Averages (50MA, 200MA)
    - Recent Swing Highs/Lows (based on 6-month lookback using find_peaks)
    - 14-Day Average True Range (ATR)
    - ATR-based Stop Loss suggestion below nearest swing low.
    - Entry/Take Profit suggestions based on swing levels.

    Args:
        stock_symbol (str): The stock ticker symbol.
        hist_data (pd.DataFrame): DataFrame containing historical stock data with
                                  columns 'High', 'Low', 'Close'. Ideally >= 1 year
                                  to ensure 200MA and swing calculations are robust.

    Returns:
        dict: A dictionary containing all calculated levels and suggestions,
              or None if calculation fails due to insufficient data or missing libraries.

    Requires: pandas_ta, scipy
    """
    st.info(f"⚙️ Calculating comprehensive trade levels for {stock_symbol}...")
    calculated_levels = {}

    # --- Dependency Check ---
    try:
        import pandas_ta as ta
        from scipy.signal import find_peaks
    except ImportError:
        st.error(
            "🚨 Libraries Missing: Please install `pandas_ta` and `scipy` to use comprehensive level calculations."
        )
        st.code("pip install pandas_ta scipy")
        return None

    # --- Data Validation ---
    required_cols = ["High", "Low", "Close"]
    if hist_data is None or hist_data.empty:
        st.warning(f"Historical data for {stock_symbol} is empty.")
        return None
    if not all(col in hist_data.columns for col in required_cols):
        st.warning(
            f"Historical data missing required columns (High, Low, Close) for {stock_symbol}."
        )
        return None

    # Use latest close as reference
    current_close = hist_data["Close"].iloc[-1]
    calculated_levels["Current Close"] = f"${current_close:.2f}"

    # --- 1. Pivot Point Calculation ---
    if len(hist_data) >= 2:
        try:
            prev_high = hist_data["High"].iloc[-2]
            prev_low = hist_data["Low"].iloc[-2]
            prev_close = hist_data["Close"].iloc[-2]

            pivot_point = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * pivot_point) - prev_low
            s1 = (2 * pivot_point) - prev_high
            r2 = pivot_point + (prev_high - prev_low)
            s2 = pivot_point - (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot_point - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot_point)

            calculated_levels.update(
                {
                    "Pivot Point (P)": f"${pivot_point:.2f}",
                    "Pivot R1": f"${r1:.2f}",
                    "Pivot S1": f"${s1:.2f}",
                    "Pivot R2": f"${r2:.2f}",
                    "Pivot S2": f"${s2:.2f}",
                    "Pivot R3": f"${r3:.2f}",
                    "Pivot S3": f"${s3:.2f}",
                }
            )
            st.success("✅ Pivots calculated.")
        except IndexError:
            st.warning("Could not calculate Pivots (needs >= 2 days history).")
        except Exception as e:
            st.warning(f"Pivot calculation error: {e}")
    else:
        st.warning("Skipping Pivots (needs >= 2 days history).")

    # --- 2. Moving Average Calculation ---
    if len(hist_data) >= 50:
        ma50 = hist_data["Close"].rolling(window=50).mean().iloc[-1]
        calculated_levels["MA 50-Day"] = f"${ma50:.2f}"
    else:
        calculated_levels["MA 50-Day"] = "N/A (<50 days)"

    if len(hist_data) >= 200:
        ma200 = hist_data["Close"].rolling(window=200).mean().iloc[-1]
        calculated_levels["MA 200-Day"] = f"${ma200:.2f}"
    else:
        calculated_levels["MA 200-Day"] = "N/A (<200 days)"
    st.success("✅ MAs calculated.")

    # --- 3. Swing High/Low & ATR Calculation ---
    if len(hist_data) >= 60:  # Need sufficient data for ATR and swings
        try:
            # Calculate ATR
            hist_data.ta.atr(length=14, append=True)  # Adds 'ATRr_14' column
            latest_atr = (
                hist_data["ATRr_14"].dropna().iloc[-1]
                if not hist_data["ATRr_14"].dropna().empty
                else (current_close * 0.01)
            )  # Fallback ATR
            calculated_levels["ATR (14-Day)"] = f"${latest_atr:.2f}"

            # Calculate Swing Highs/Lows (using ~6 months data)
            hist_6m = hist_data.tail(126)  # Approx 6 months of trading days
            distance_param = 5  # Min separation between peaks
            prominence_factor = 0.2  # Sensitivity factor for peak detection
            # Calculate prominence based on std dev of the 6m window - prevents small insignificant wiggles being peaks
            prominence_high = hist_6m["High"].std() * prominence_factor
            prominence_low = hist_6m["Low"].std() * prominence_factor

            # Find peaks (positive for highs, negative for lows)
            swing_high_indices, _ = find_peaks(
                hist_6m["High"],
                distance=distance_param,
                prominence=prominence_high if prominence_high > 0 else None,
            )
            swing_low_indices, _ = find_peaks(
                -hist_6m["Low"],
                distance=distance_param,
                prominence=prominence_low if prominence_low > 0 else None,
            )  # Note the negative series

            swing_highs = (
                hist_6m["High"].iloc[swing_high_indices].sort_values(ascending=False)
            )
            swing_lows = (
                hist_6m["Low"].iloc[swing_low_indices].sort_values(ascending=True)
            )

            # Identify nearest levels relative to current price
            support_levels = swing_lows[swing_lows < current_close]
            resistance_levels = swing_highs[swing_highs > current_close]

            nearest_support = (
                support_levels.iloc[-1]
                if not support_levels.empty
                else hist_6m["Low"].min()
            )
            nearest_resistance = (
                resistance_levels.iloc[-1]
                if not resistance_levels.empty
                else hist_6m["High"].max()
            )  # Note: resistance levels sorted descending, so [-1] is closest *above*

            next_support = support_levels.iloc[-2] if len(support_levels) > 1 else None
            next_resistance = (
                resistance_levels.iloc[-2] if len(resistance_levels) > 1 else None
            )  # [-2] is next further away above price

            calculated_levels["Swing Low 1 (Nearest)"] = f"${nearest_support:.2f}"
            calculated_levels["Swing Low 2"] = (
                f"${next_support:.2f}" if next_support is not None else "N/A"
            )
            calculated_levels["Swing High 1 (Nearest)"] = f"${nearest_resistance:.2f}"
            calculated_levels["Swing High 2"] = (
                f"${next_resistance:.2f}" if next_resistance is not None else "N/A"
            )

            # ATR Stop Loss Suggestion
            sl_price_atr = nearest_support - (
                1.0 * latest_atr
            )  # Example: 1x ATR below nearest swing low
            sl_suggestion_atr = (
                f"~${sl_price_atr:.2f} (Based on 1*ATR below Nearest Swing Low)"
            )
            calculated_levels["ATR Stop Suggestion"] = sl_suggestion_atr

            # Entry/TP Suggestions (from swing logic)
            entry_suggestion = f"Consider entry near Swing Low 1 (${nearest_support:.2f}) or on a break above Swing High 1 (${nearest_resistance:.2f})."
            calculated_levels["Swing Entry Suggestion"] = entry_suggestion

            risk_atr = current_close - sl_price_atr
            if risk_atr > 0.01:  # Avoid division by zero or tiny risk
                tp_price_rr = current_close + (2 * risk_atr)  # Example 2:1 R:R target
                tp_suggestion = f"Swing Targets: Near Swing High 1 (${nearest_resistance:.2f}), potentially Swing High 2 (${next_resistance:.2f} if valid). Consider 2:1 R:R target (~${tp_price_rr:.2f}) based on ATR stop."
            else:
                tp_suggestion = f"Swing Targets: Near Swing High 1 (${nearest_resistance:.2f}), potentially Swing High 2 (${next_resistance:.2f} if valid). (R:R calc requires valid ATR stop)."
            calculated_levels["Swing Take Profit Suggestion"] = tp_suggestion

            calculated_levels["6-Month Low"] = (
                f"${hist_6m['Low'].min():.2f}"  # Keep for context
            )
            calculated_levels["6-Month High"] = (
                f"${hist_6m['High'].max():.2f}"  # Keep for context
            )

            st.success("✅ Swings/ATR calculated.")

        except Exception as e:
            st.error(f"🚨 Error calculating Swing/ATR levels: {e}")
            calculated_levels["ATR (14-Day)"] = "Error"
            calculated_levels["Swing Levels"] = "Error during calculation"
    else:
        st.warning("Skipping Swing/ATR calculations (needs >= 60 days history).")

    # --- Final Check & Return ---
    if len(calculated_levels) <= 1:  # Only contains current close
        st.error("🚨 Comprehensive level calculation failed to produce results.")
        return None

    st.success(f"✅ Comprehensive trade level calculation complete.")
    return calculated_levels


# --- Function to Generate OHLC Chart (Accepts hist_df) ---
def generate_ohlc_chart(stock_symbol, hist_df, filename="stock_chart.png"):
    """Generates a simple OHLC line chart using matplotlib."""
    st.info(f"📊 Generating chart for {stock_symbol}...")
    try:
        # --- Charting Logic (using hist_df parameter) ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist_df.index, hist_df["Close"], label="Close", color="blue", lw=1.5)
        # Optional MAs using hist_df
        if len(hist_df) > 50:
            hist_df["MA50"] = hist_df["Close"].rolling(50).mean()
            ax.plot(
                hist_df.index,
                hist_df["MA50"],
                label="50MA",
                color="orange",
                ls="--",
                lw=1,
            )
        if len(hist_df) > 200:
            hist_df["MA200"] = hist_df["Close"].rolling(200).mean()
            ax.plot(
                hist_df.index,
                hist_df["MA200"],
                label="200MA",
                color="red",
                ls="--",
                lw=1,
            )
        ax.set_title(f"{stock_symbol} Price Chart (Recent Period)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename, format="png", bbox_inches="tight")
        plt.close(fig)
        return filename
    except Exception as e:
        st.error(f"🚨 Error generating chart: {e}")
        return None


# --- Function for IMAGE-BASED AI Chart Analysis (using responses.create) ---
def get_llm_image_chart_analysis(stock_symbol, image_path):
    """Uses LLM (responses.create) to analyze chart image."""
    st.info(f"🤖 Asking LLM to analyze chart image for {stock_symbol}...")
    client = get_openai_client()
    if not client:
        return None

    base64_image = encode_image(image_path)
    if not base64_image:
        st.error("Failed to encode chart image for AI analysis.")
        return None

    # Prompt focuses analysis on the image provided
    chart_analysis_prompt = f"""
    **Task:** Analyze the provided OHLC chart image for the stock "{stock_symbol.upper()}".
    **Analysis Instructions:**
    1. Identify potential **support zones** (price levels where buying pressure might appear).
    2. Identify potential **resistance zones** (price levels where selling pressure might appear).
    3. Describe any notable **chart patterns or trends** visible (e.g., uptrend, channel, consolidation).
    4. Suggest potential **entry or exit areas** based purely on the visual chart analysis. Be specific with approximate price levels if possible from the chart.
    **Output Format:** Use clear headings or bullet points for Support, Resistance, Patterns/Trends, and Potential Entry/Exit Areas. Be concise.
    """

    try:
        # Using the specific client.responses.create structure for image input
        response = client.responses.create(
            model="gpt-4o",  # Ensure this model supports image input
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": chart_analysis_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
            temperature=0.4,
        )

        # --- Assume response structure similar to research call ---
        ai_analysis_content = None
        try:
            if (
                response
                and hasattr(response, "output")
                and isinstance(response.output, list)
                and len(response.output) > 0
            ):
                target_idx = 1 if len(response.output) > 1 else 0
                output_item = response.output[target_idx]
                if (
                    hasattr(output_item, "content")
                    and isinstance(output_item.content, list)
                    and output_item.content
                ):
                    for item in output_item.content:
                        if (
                            hasattr(item, "type")
                            and item.type == "output_text"
                            and hasattr(item, "text")
                        ):
                            ai_analysis_content = item.text.strip()
                            break
            if ai_analysis_content is None:
                st.warning(f"Could not parse LLM image analysis response.")
        except Exception as e:
            st.error(f"Parse error (Image Analysis): {e}")
            ai_analysis_content = None
        # --- End Parsing ---

        if ai_analysis_content:
            st.success(f"✅ LLM Chart Image Analysis complete.")
            return ai_analysis_content
        else:
            st.error(f"🚨 LLM Chart Image Analysis failed.")
            return None

    except openai.APIError as e:
        st.error(f"🚨 OpenAI API Error (Image Analysis): {e}")
        return None
    except Exception as e:
        st.error(f"🚨 Unexpected error (Image Analysis): {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="CANSLIM Stock Analyzer", layout="wide")

st.title("📈 CANSLIM Stock Analyzer (v11 - Simple Levels + AI Chart Analysis)")
st.caption(
    "Uses LLM for research/analysis, simple SL/TP calc, shows chart, & gets AI image analysis if 'Buy'."
)
st.warning(
    "Note: Uses LLM API (incl. Image Input) + yfinance. Check costs (Image input can be more expensive). **NOT FINANCIAL ADVICE.** Requires `matplotlib`."
)

# --- Input Section ---
st.header("Stock Input")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, NVDA, MSFT):", "").upper()

# --- Analysis Trigger ---
analyze_button = st.button(
    f"Analyze {stock_symbol} with LLM" if stock_symbol else "Analyze Stock"
)

# --- Analysis Output Section ---
if analyze_button and stock_symbol:

    # Initialize variables
    researched_canslim_data, analysis_result, recommendation_result, current_price = (
        None,
        None,
        None,
        None,
    )
    (
        hist_data,
        simple_trade_levels,
        chart_filename,
        researched_trend_data,
        trend_data_text,
        ai_chart_analysis_text,
    ) = (None, None, None, None, None, None)
    CHART_FILE = f"{stock_symbol}_chart.png"  # Define chart filename

    # --- 1. Research CANSLIM ---
    with st.spinner(f"🤖 Researching CANSLIM data..."):
        researched_canslim_data = research_stock_data_llm_chat(stock_symbol, "CANSLIM")

    if researched_canslim_data:
        # --- Parse Price ---
        match = re.search(
            r"Current Price:\s*\$?([\d,]+\.?\d*)", researched_canslim_data, re.I
        )
        if match:
            try:
                current_price = float(match.group(1).replace(",", ""))
                st.sidebar.info(f"Parsed Price: ${current_price:.2f}")
            except ValueError:
                st.sidebar.warning("Invalid price.")
        else:
            st.sidebar.warning("Price not in web research.")

        # --- 2. Format CANSLIM & Get Recommendation ---
        with st.spinner(f"🤖 Formatting CANSLIM analysis..."):
            analysis_result, recommendation_result = format_researched_data_llm_chat(
                stock_symbol, researched_canslim_data, "CANSLIM"
            )

        if analysis_result and recommendation_result:
            # Display CANSLIM
            st.header(f"CANSLIM Analysis: {stock_symbol}")
            st.markdown(analysis_result)
            st.subheader("Recommendation:")
            if recommendation_result == "Buy":
                st.success(f"**{recommendation_result}**")
            elif recommendation_result == "Sell":
                st.error(f"**{recommendation_result}**")
            elif recommendation_result == "Hold":
                st.warning(f"**{recommendation_result}**")
            else:
                st.info(f"**{recommendation_result}**")

            # --- 3. Conditional Block for "Buy" ---
            if recommendation_result in ["Buy", "Uncertain"]:
                # if current_price is not None:
                st.markdown("---")
                # Fetch History ONCE
                with st.spinner(f"Fetching history for {stock_symbol}..."):
                    try:
                        ticker = yf.Ticker(stock_symbol)
                        print(f"Searching data on Ticker: {ticker}")
                        # Fetch enough data for calculations and chart context (e.g., 1 year)
                        hist_data = ticker.history(period="8mo", interval="1d")
                        print("--" * 30)
                        print(hist_data)
                        print("--" * 30)
                        if hist_data.empty:
                            st.warning(f"No history found.")
                            hist_data = None
                    except Exception as e:
                        st.error(f"History fetch error: {e}")
                        hist_data = None

                    if hist_data is not None:
                        current_price = (
                            hist_data["Close"].iloc[-1]
                            if "Close" in hist_data.columns
                            else 0
                        )
                        print(f"Current Price from history: {current_price}")
                        st.header("Trade Analysis & Chart")
                        cols = st.columns([1, 1.2])  # Adjust column widths if needed

                        with cols[0]:  # Left column for levels & AI text
                            # Calculate Simple Levels (passing hist_data)
                            with st.spinner(f"⚙️ Calculating simple levels..."):
                                simple_trade_levels = calculate_trade_levels(
                                    stock_symbol, current_price, hist_data
                                )  # Pass hist_data
                            if simple_trade_levels:
                                st.subheader("Code-Calculated Levels (Simple)")
                                for key, val in simple_trade_levels.items():
                                    st.markdown(f"**{key}:** {val}")
                                st.caption("*Based on historical Min/Max.*")
                            else:
                                st.warning("Simple level calculation failed.")

                            st.markdown("---")

                            # Generate Chart (needed for path)
                            with st.spinner(f"📊 Generating chart..."):
                                chart_filename = generate_ohlc_chart(
                                    stock_symbol, hist_data, filename=CHART_FILE
                                )  # Pass hist_data

                            # Get AI Image Analysis (if chart generated)
                            if chart_filename:
                                with st.spinner(
                                    "🤖 Requesting AI chart image analysis..."
                                ):
                                    ai_chart_analysis_text = (
                                        get_llm_image_chart_analysis(
                                            stock_symbol, chart_filename
                                        )
                                    )  # Use new function

                                if ai_chart_analysis_text:
                                    st.subheader("AI Chart Analysis (Image-based)")
                                    st.markdown(ai_chart_analysis_text)
                                    st.caption(
                                        "*AI interpretation of the chart image. Opinion, not advice.*"
                                    )
                                elif chart_filename:  # If chart generated but AI failed
                                    st.warning("Could not get AI chart analysis.")
                            else:
                                st.warning(
                                    "Chart generation failed, skipping AI image analysis."
                                )

                        with cols[1]:  # Right column for the chart
                            if chart_filename:
                                st.image(
                                    chart_filename,
                                    caption=f"{stock_symbol} Price Chart",
                                    use_column_width=True,
                                )
                            # Clean up the generated chart file after displaying
                            if chart_filename and os.path.exists(chart_filename):
                                try:
                                    os.remove(chart_filename)
                                except Exception as e:
                                    st.warning(f"Could not remove temp chart file: {e}")

                    else:
                        st.warning(
                            "Skipping 'Buy' analysis additions: History fetch failed."
                        )

            # --- 4. Trend Analysis ---
            st.markdown("---")
            with st.spinner(f"🤖 Researching/Formatting Trend data..."):
                researched_trend_data = research_stock_data_llm_chat(
                    stock_symbol, "Trend"
                )
                if researched_trend_data:
                    trend_data_text = format_researched_data_llm_chat(
                        stock_symbol, researched_trend_data, "Trend"
                    )
            if trend_data_text:
                st.header(f"Recent Price & Trend")
                st.markdown(trend_data_text)
            else:
                st.warning("Trend data failed.")

        else:
            st.error("Halted: CANSLIM formatting failed.")
    else:
        st.error("Halted: CANSLIM research failed.")

elif analyze_button and not stock_symbol:
    st.error("⚠️ Please enter a stock symbol.")

st.markdown("---")
# Use appropriate timezone and format based on context
# i.e Context: Wednesday, April 2, 2025 at 9:56:29 PM AEDT (Sydney)
try:
    aedt = datetime.timezone(datetime.timedelta(hours=11))
    # AEDT is UTC+11
    # Hardcode based on context time for consistency in this snapshot
    # now_aedt = datetime.datetime.now(aedt)
    now = datetime.datetime.now(aedt)
    current_run_time_aedt = now.strftime("%B %-d, %Y %-I:%M %p %Z")
except Exception:
    current_run_time_aedt = (
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (Local Time)"
    )
st.markdown(
    f"Location Context: Beverly Hills, NSW, Australia. Analysis Approx Time: {current_run_time_aedt}. Uses LLM API + yfinance."
)
