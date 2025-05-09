import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px
import yfinance as yf
import re
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta, gamma as bs_gamma, vega as bs_vega
from math import log, sqrt
from scipy.stats import norm

if "saved_ticker" not in st.session_state:
    st.session_state.saved_ticker = ""
if "strike_range" not in st.session_state:
    st.session_state.strike_range = 20
if "last_expiry_date" not in st.session_state:
    st.session_state.last_expiry_date = None


st.set_page_config(
    page_title="Volume Analysis",
    page_icon="📊",
    layout="wide"
)
def extract_expiry_from_contract(contract_symbol):
    pattern = r'[A-Z]+W?(?P<date>\d{6}|\d{8})[CP]\d+'
    match = re.search(pattern, contract_symbol)
    if match:
        date_str = match.group("date")
        try:
            if len(date_str) == 6:
                return datetime.strptime(date_str, "%y%m%d").date()
            else:
                return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            return None
    return None

@st.cache_data(ttl=10)
def fetch_options_for_date(ticker, date):
    #print(f"Fetching options for {ticker} / {date}")
    stock = yf.Ticker(ticker)
    try:
        chain = stock.option_chain(date)
        #print(f"Options fetched for {ticker} / {date}")
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        #print(f"Options copied for {ticker} / {date} + {calls.shape[0]} calls, {puts.shape[0]} puts")
        calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
        #print(calls)
        puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
        return calls, puts
    except Exception as e:
        st.error(f"Error fetching options for {ticker} / {date}: {e}")
        return pd.DataFrame(), pd.DataFrame()
@st.cache_data(ttl=1)
def get_current_price(ticker):
    formatted_ticker = ticker.replace('%5E', '^')
    if formatted_ticker in ['^SPX', '^VIX', '^NDX'] or ticker in ['%5ESPX', '%5ENDX']:
        symbol = formatted_ticker.replace('^', '')
        live_price = OGamma_api.get_live_price(symbol)
        if live_price is not None:
            return round(float(live_price), 2)
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice")
        if price is None:
            price = stock.fast_info.get("lastPrice")
        return float(price) if price is not None else None
    except Exception as e:
        st.error(f"Error fetching price from Yahoo Finance: {e}")
        return None
def calculate_greeks(flag, S, K, t, sigma):
    try:
        t = max(t, 1/1440)
        d1 = (log(S/K) + 0.5*sigma**2*t) / (sigma*sqrt(t))
        d2 = d1 - sigma*sqrt(t)
        delta_val = bs_delta(flag, S, K, t, 0, sigma)
        gamma_val = bs_gamma(flag, S, K, t, 0, sigma)
        vanna_val = -norm.pdf(d1) * (d2/sigma)
        return delta_val, gamma_val, vanna_val
    except:
        return None, None, None

def calculate_charm(flag, S, K, t, sigma):
    try:
        t = max(t, 1/1440)
        d1 = (log(S/K) + 0.5*sigma**2*t) / (sigma*sqrt(t))
        d2 = d1 - sigma*sqrt(t)
        norm_d1 = norm.pdf(d1)
        raw_charm = -norm_d1*(2*0.5*sigma**2*t - d2*sigma*sqrt(t))/(2*t*sigma*sqrt(t))
        if flag == 'p':
            return -raw_charm
        return raw_charm
    except:
        return None

def calculate_speed(flag, S, K, t, sigma):
    try:
        t = max(t, 1/1440)
        d1 = (log(S/K) + 0.5*sigma**2*t) / (sigma*sqrt(t))
        gamma_val = bs_gamma(flag, S, K, t, 0, sigma)
        speed_val = -gamma_val*((d1/(sigma*sqrt(t)))+1)/S
        return speed_val
    except:
        return None

def calculate_vomma(flag, S, K, t, sigma):
    try:
        t = max(t, 1/1440)
        d1 = (log(S/K) + 0.5*sigma**2*t) / (sigma*sqrt(t))
        d2 = d1 - sigma*sqrt(t)
        vega_val = bs_vega(flag, S, K, t, 0, sigma)
        vomma_val = vega_val*(d1*d2/sigma)
        return vomma_val
    except:
        return None
    

def TRACE_compute_greeks_and_charts(ticker, expiry_date_str):
    print(ticker, expiry_date_str)
    if not expiry_date_str:
        return None, None, None, None, None, None
    
    # Fetch options data
    calls, puts = fetch_options_for_date(ticker, expiry_date_str)
    
    # Check if we have data
    if calls.empty and puts.empty:
        return None, None, None, None, None, None
    
    # Parse dates
    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    today = datetime.today().date()
    t_days = (selected_expiry - today).days
    
    # Validate trading day
    # Get current price
    S = get_current_price(ticker)
    if not S:
        st.error("Could not fetch underlying price.")
        return None, None, None, None, None, None
    
    # Calculate time to expiry
    t = max(t_days, 1) / 365.0  # Ensure t is at least 1 day to avoid calculation errors
    
    # Make copies of the DataFrames to avoid modifying the originals
    calls = calls.copy()
    puts = puts.copy()
    
    # Add estimated implied volatility if missing
    if 'impliedVolatility' not in calls.columns or calls['impliedVolatility'].isna().all():
        print("Warning: No implied volatility data for calls. Using estimated values.")
        calls['impliedVolatility'] = calls.apply(
            lambda row: 0.3 + 0.2 * abs(row['strike']/S - 1),  # Simple smile approximation
            axis=1
        )
    
    if 'impliedVolatility' not in puts.columns or puts['impliedVolatility'].isna().all():
        print("Warning: No implied volatility data for puts. Using estimated values.")
        puts['impliedVolatility'] = puts.apply(
            lambda row: 0.3 + 0.2 * abs(row['strike']/S - 1),  # Simple smile approximation
            axis=1
        )
    
    # Define function to compute row-level Greeks
    def compute_row_greeks(row, flag):
        sigma = row.get("impliedVolatility", None)
        
        # If sigma is missing or invalid, use an estimate based on moneyness
        if sigma is None or sigma <= 0:
            moneyness = row["strike"] / S
            # Simple volatility smile approximation
            sigma = 0.3 + 0.2 * abs(moneyness - 1)
            print(f"Using estimated IV of {sigma:.2f} for {flag} option at strike {row['strike']}")
        
        # Calculate Greeks with the valid sigma
        try:
            delta_val, gamma_val, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma)
            charm_val = calculate_charm(flag, S, row["strike"], t, sigma)
            speed_val = calculate_speed(flag, S, row["strike"], t, sigma)
            vomma_val = calculate_vomma(flag, S, row["strike"], t, sigma)
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            # Fallback to approximate values
            if flag == 'c':
                # For calls
                moneyness = row["strike"] / S
                delta_val = max(0.01, min(0.99, 0.5 + (1 - moneyness) * 2))
                gamma_val = max(0.01, 0.1 * (1 - min(abs(moneyness - 1), 0.5) * 2))
            else:
                # For puts
                moneyness = S / row["strike"]
                delta_val = min(-0.01, max(-0.99, -0.5 - (1 - moneyness) * 2))
                gamma_val = max(0.01, 0.1 * (1 - min(abs(moneyness - 1), 0.5) * 2))
            
            vanna_val = gamma_val * 0.1
            charm_val = delta_val * -0.01
            speed_val = gamma_val * -0.01
            vomma_val = gamma_val * 10
            
        return {
            "calc_delta": delta_val, 
            "calc_gamma": gamma_val, 
            "calc_vanna": vanna_val,
            "calc_charm": charm_val, 
            "calc_speed": speed_val, 
            "calc_vomma": vomma_val,
            "calc_theta": -0.01 * row.get("lastPrice", S*0.05) * (1/max(t_days, 1))  # Simple theta approximation
        }
    
    # Calculate Greeks for calls
    calls_greeks = calls.apply(lambda r: pd.Series(compute_row_greeks(r, 'c')), axis=1)
    calls = pd.concat([calls, calls_greeks], axis=1)
    
    # Calculate Greeks for puts
    puts_greeks = puts.apply(lambda r: pd.Series(compute_row_greeks(r, 'p')), axis=1)
    puts = pd.concat([puts, puts_greeks], axis=1)
    
    # Ensure openInterest exists (needed for exposure calculations)
    if 'openInterest' not in calls.columns or calls['openInterest'].isna().all():
        print("Warning: No open interest data for calls. Using placeholder values.")
        calls['openInterest'] = 100  # Placeholder value
    
    if 'openInterest' not in puts.columns or puts['openInterest'].isna().all():
        print("Warning: No open interest data for puts. Using placeholder values.")
        puts['openInterest'] = 100  # Placeholder value
    
    # Compute exposures - with safety checks to prevent NaN propagation
    # For calls
    calls['GEX'] = calls.apply(
        lambda r: r.get('calc_gamma', 0.01) * (S**2) * r.get('openInterest', 100) * 100,
        axis=1
    )
    calls["VEX"] = calls.apply(
        lambda r: r.get('calc_vanna', 0.001) * r.get('openInterest', 100) * 100,
        axis=1
    )
    calls["DEX"] = calls.apply(
        lambda r: r.get('calc_delta', 0.5) * r.get('openInterest', 100) * 100,
        axis=1
    )
    calls["Charm"] = calls.apply(
        lambda r: r.get('calc_charm', -0.001) * r.get('openInterest', 100) * 100,
        axis=1
    )
    calls["Speed"] = calls.apply(
        lambda r: r.get('calc_speed', -0.0001) * r.get('openInterest', 100) * 100,
        axis=1
    )
    calls["Vomma"] = calls.apply(
        lambda r: r.get('calc_vomma', 0.01) * r.get('openInterest', 100) * 100,
        axis=1
    )
    
    # For puts
    puts['GEX'] = puts.apply(
        lambda r: -1 * r.get('calc_gamma', 0.01) * (S**2) * r.get('openInterest', 100) * 100,
        axis=1
    )
    puts["VEX"] = puts.apply(
        lambda r: r.get('calc_vanna', 0.001) * r.get('openInterest', 100) * 100,
        axis=1
    )
    puts["DEX"] = puts.apply(
        lambda r: r.get('calc_delta', -0.5) * r.get('openInterest', 100) * 100,
        axis=1
    )
    puts["Charm"] = puts.apply(
        lambda r: r.get('calc_charm', 0.001) * r.get('openInterest', 100) * 100,
        axis=1
    )
    puts["Speed"] = puts.apply(
        lambda r: r.get('calc_speed', -0.0001) * r.get('openInterest', 100) * 100,
        axis=1
    )
    puts["Vomma"] = puts.apply(
        lambda r: r.get('calc_vomma', 0.01) * r.get('openInterest', 100) * 100,
        axis=1
    )
    
    # Instead of dropping rows with NA values, fill them with approximations
    # This prevents empty DataFrames
    for df in [calls, puts]:
        cols_to_check = ["calc_gamma", "calc_vanna", "calc_delta", "GEX", "VEX", "DEX"]
        for col in cols_to_check:
            if col in df.columns and df[col].isna().any():
                # Use a small non-zero value as placeholder
                placeholder = 0.001 if col in ["calc_gamma", "calc_vanna", "calc_delta"] else 10
                df[col].fillna(placeholder, inplace=True)
    
    print(f"Processed {len(calls)} call options and {len(puts)} put options")
    return calls, puts, S, t, selected_expiry, today

    
def filter_to_n_strikes(df, n, current_price):
    """
    Filter dataframe to n strikes closest to the current price
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing 'strike' column
    n : int
        Number of strikes to keep
    current_price : float
        Current price to center around
        
    Returns:
    --------
    pandas DataFrame
        Filtered dataframe
    """
    if 'strike' not in df.columns or df.empty:
        return df
        
    # Sort by distance to current price
    df['distance'] = abs(df['strike'] - current_price)
    df = df.sort_values('distance')
    
    # Take n closest strikes
    result = df.head(n).drop('distance', axis=1)
    
    # Sort by strike price for display
    return result.sort_values('strike')

def load_options_data(ticker, expiry_date_str):
    """Load options data efficiently using session state and caching."""
    # Use a different key for tracking than the widget key
    
    calls, puts, S, t, selected_expiry, today = TRACE_compute_greeks_and_charts(ticker, expiry_date_str)
    
    return calls, puts, S, t, selected_expiry, today

    
def create_options_heatmap(calls, puts, current_price, ticker, expiry_date):
    """
    Create a heatmap visualization for options volume data.
    
    Parameters:
    -----------
    calls : list of dict
        List of call options with 'strike' and 'volume' keys
    puts : list of dict
        List of put options with 'strike' and 'volume' keys
    current_price : float
        Current price of the underlying asset
    ticker : str
        Ticker symbol of the underlying asset
    expiry_date : str
        Expiration date of the options
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The heatmap figure object
    """
    # Process calls and puts into dataframes
    
    try:
        expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_date - datetime.now()).days
        if days_to_expiry < 0:
            days_to_expiry = 0
    except:
        st.warning(f"Could not parse expiry date: {expiry_date}")
        days_to_expiry = 30  # Default fallback
        
    
        # Calculate delta-adjusted OI
    call_daoi_by_strike, put_daoi_by_strike, total_call_daoi, total_put_daoi = calculate_delta_adjusted_oi(calls, puts)
    
    call_put_daoi_ratio = total_call_daoi / total_put_daoi if total_put_daoi > 0 else float('inf')
    
    if call_put_daoi_ratio > 1.5:
        sentiment = "BULLISH"
        sentiment_color = "green"
    elif call_put_daoi_ratio < 0.67:
        sentiment = "BEARISH"
        sentiment_color = "red"
    else:
        sentiment = "NEUTRAL"
        sentiment_color = "orange"
        
    
    
    calls_df = pd.DataFrame(calls) if len(calls) > 0 else pd.DataFrame(columns=['strike', 'volume'])
    calls_df['option_type'] = 'call'
    
    puts_df = pd.DataFrame(puts) if len(puts) > 0 else pd.DataFrame(columns=['strike', 'volume'])
    puts_df['option_type'] = 'put'
    
    # Combine both datasets
    combined_df = pd.concat([calls_df, puts_df])
    
    # Make sure we have data to work with
# Make sure we have data to work with
    if combined_df.empty:
        return px.imshow(pd.DataFrame([[0]]), title="No data available"), None
    
    # Filter to strikes closest to current price
    combined_df = filter_to_n_strikes(combined_df, 50, current_price)
    
    # Get unique sorted strike prices
    strike_range = sorted(combined_df['strike'].unique())
    
    # Check if strikes have irregular spacing
    if len(strike_range) > 1:
        # Calculate differences between consecutive strikes
        diffs = [strike_range[i+1] - strike_range[i] for i in range(len(strike_range)-1)]
        
        # If spacing is irregular, create a standardized spacing
        if len(set(diffs)) > 1:
            min_diff = min(diffs)
            # Create evenly spaced strikes
            min_strike = min(strike_range)
            max_strike = max(strike_range)
            standardized_strikes = []
            current_strike = min_strike
            while current_strike <= max_strike:
                standardized_strikes.append(current_strike)
                current_strike += min_diff
            
            # Reindex combined_df to match standardized strikes
            # This ensures all gaps are filled with zeros
            combined_df_reindexed = pd.DataFrame(index=standardized_strikes)
            for idx, row in combined_df.iterrows():
                closest_strike = min(standardized_strikes, key=lambda x: abs(x - row['strike']))
                if abs(closest_strike - row['strike']) < min_diff / 2:  # Only if close enough
                    temp_df = pd.DataFrame({'strike': [closest_strike], 
                                          'volume': [row['volume']], 
                                          'option_type': [row['option_type']]})
                    combined_df_reindexed = pd.concat([combined_df_reindexed, temp_df])
            
            combined_df = combined_df_reindexed.reset_index(drop=True)
            strike_range = standardized_strikes
    
    # Categorize options
    def categorize_option(strike, price, option_type):
        threshold = 0.001  # 5% threshold for ATM
        if option_type == 'call':
            if strike < price * (1 - threshold):
                return 'Call ITM'
            elif strike <= price * (1 + threshold):
                return 'Call ATM'
            else:
                return 'Call OTM'
        else:  # put
            if strike > price * (1 + threshold):
                return 'Put ITM'
            elif strike >= price * (1 - threshold):
                return 'Put ATM'
            else:
                return 'Put OTM'
    
    # Add category column
    combined_df['category'] = combined_df.apply(
        lambda row: categorize_option(row['strike'], current_price, row['option_type']), 
        axis=1
    )
    
    # Prepare data for heatmap
    categories = ['Call ITM', 'Call ATM', 'Call OTM', 'Put OTM', 'Put ATM', 'Put ITM']
    
    # Create a properly structured dataframe for the heatmap
    heatmap_data = np.zeros((len(categories), len(strike_range)))
    
    # Fill in the matrix with volume data
    for i, category in enumerate(categories):
        category_data = combined_df[combined_df['category'] == category]
        for _, row in category_data.iterrows():
            if row['strike'] in strike_range:
                j = strike_range.index(row['strike'])
                heatmap_data[i, j] = row['volume']
    
    # Convert to DataFrame with proper indices
    pivot_df = pd.DataFrame(heatmap_data, index=categories, columns=strike_range)
    
    # Analyze the data for interpretation
    total_call_volume = pivot_df.loc[categories[:3]].sum().sum()
    total_put_volume = pivot_df.loc[categories[3:]].sum().sum()
    
    # Find max volumes and their positions
    max_call_strike = pivot_df.loc[categories[:3]].max(axis=0).idxmax()
    max_call_volume = pivot_df.loc[categories[:3]].max(axis=0).max()
    max_put_strike = pivot_df.loc[categories[3:]].max(axis=0).idxmax()
    max_put_volume = pivot_df.loc[categories[3:]].max(axis=0).max()
    
    # Determine sentiment with colors and icons
    if total_call_volume > total_put_volume * 1.2:
        sentiment = "BULLISH 🐂"
        sentiment_color = "green"
    elif total_put_volume > total_call_volume * 1.2:
        sentiment = "BEARISH 🐻"
        sentiment_color = "red"
    else:
        sentiment = "MIXED 📊"
        sentiment_color = "orange"
    
    # Calculate expected trading range
    weighted_strikes = []
    weights = []
    for strike in strike_range:
        col_volume = pivot_df[strike].sum()
        if col_volume > 0:
            weighted_strikes.append(strike)
            weights.append(col_volume)
    
    if weighted_strikes:
        avg_strike = np.average(weighted_strikes, weights=weights)
        strike_std = np.sqrt(np.average((np.array(weighted_strikes) - avg_strike)**2, weights=weights))
        lower_range = max(min(strike_range), avg_strike - strike_std)
        upper_range = min(max(strike_range), avg_strike + strike_std)
    else:
        lower_range = min(strike_range)
        upper_range = max(strike_range)
    
    # Calculate dynamic colors for expected range
    range_size = upper_range - lower_range
    if range_size <= 15:
        range_color = "darkgreen"  # Tight range suggests low volatility
    elif range_size <= 30:
        range_color = "green"
    elif range_size <= 50:
        range_color = "orange"  # Medium volatility
    else:
        range_color = "red"  # Wide range suggests high volatility
    
    # Create the heatmap with improved formatting
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Strike Price", y="Option Category", color="Volume"),
        x=strike_range,
        y=categories,
        color_continuous_scale="balance",
        title=f"{ticker} Options Volume by Strike Price (Expiry: {expiry_date})"
    )
    
    # Add individual annotations for each metric in a row layout
    x_positions = [0.05, 0.25, 0.45, 0.65, 0.85]
    y_position = 1.1
    
    fig.add_annotation(
        x=0.5,
        y=1.25,
        xref="paper",
        yref="paper",
        text=f"Delta-Adjusted OI Ratio: {call_put_daoi_ratio:.2f} ({sentiment})",
        showarrow=False,
        font=dict(size=12, color=sentiment_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=sentiment_color,
        borderwidth=1,
        borderpad=4,
        align="center"
    )
    # Market Sentiment
    fig.add_annotation(
        x=x_positions[0],
        y=y_position,
        xref="paper",
        yref="paper",
        text=f"<b>{sentiment}</b>",
        showarrow=False,
        font=dict(size=14, color=sentiment_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=sentiment_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Expected Range
    fig.add_annotation(
        x=x_positions[1],
        y=y_position,
        xref="paper",
        yref="paper",
        text=f"📍 <b>Range: ${lower_range:.0f}-${upper_range:.0f}</b>",
        showarrow=False,
        font=dict(size=12,color=range_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=range_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Max Call Volume with dynamic color
    call_ratio = max_call_volume / max(total_call_volume / len(strike_range), 1)  # Relative to average
    if call_ratio > 2:
        call_color = "darkgreen"  # Unusually high concentration
    elif call_ratio > 1.5:
        call_color = "green"
    else:
        call_color = "mediumseagreen"  # Normal concentration
    
    fig.add_annotation(
        x=x_positions[2],
        y=y_position,
        xref="paper",
        yref="paper",
        text=f"📈 Call Max: <b>{int(max_call_volume)}</b> @ ${max_call_strike:.0f}",
        showarrow=False,
        font=dict(size=12, color=call_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=call_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Max Put Volume with dynamic color
    put_ratio = max_put_volume / max(total_put_volume / len(strike_range), 1)  # Relative to average
    if put_ratio > 2:
        put_color = "darkred"  # Unusually high concentration
    elif put_ratio > 1.5:
        put_color = "red"
    else:
        put_color = "salmon"  # Normal concentration
    
    fig.add_annotation(
        x=x_positions[3],
        y=y_position,
        xref="paper",
        yref="paper",
        text=f"📉 Put Max: <b>{int(max_put_volume)}</b> @ ${max_put_strike:.0f}",
        showarrow=False,
        font=dict(size=12,color=put_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=put_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Call/Put Ratio with dynamic color based on value
    call_put_ratio = total_call_volume/total_put_volume
    
    # Determine color based on ratio
    if call_put_ratio > 1.5:
        ratio_color = "green"  # Very bullish
    elif call_put_ratio > 1.1:
        ratio_color = "lightgreen"  # Slightly bullish
    elif call_put_ratio >= 0.9:
        ratio_color = "orange"  # Neutral/Mixed
    elif call_put_ratio > 0.7:
        ratio_color = "salmon"  # Slightly bearish
    else:
        ratio_color = "red"  # Very bearish
    
    fig.add_annotation(
        x=x_positions[4],
        y=y_position,
        xref="paper",
        yref="paper",
        text=f"⚖️ C/P Ratio: <b>{call_put_ratio:.2f}</b>",
        showarrow=False,
        font=dict(size=12,color="black"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=ratio_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Add text annotations with the volume values
    for i, category in enumerate(categories):
        for j, strike in enumerate(strike_range):
            vol = pivot_df.iloc[i, j]
            if vol > 0:  # Only show non-zero values
                fig.add_annotation(
                    x=strike, y=category,
                    text=str(int(vol)),
                    showarrow=False,
                    font=dict(color="white" if vol > pivot_df.values.max()/2 else "black")
                )
    
    # Add current price line
    fig.add_vline(x=current_price, line_width=2, line_dash="dash", line_color="red",
                annotation_text="Current Price", annotation_position="top right")

    # Add expected range lines  
    fig.add_vline(x=lower_range, line_width=2, line_dash="dash", line_color=range_color,
                annotation_text=f"Lower: ${lower_range:.0f}", annotation_position="top right")
    fig.add_vline(x=upper_range, line_width=2, line_dash="dash", line_color=range_color,
                annotation_text=f"Upper: ${upper_range:.0f}", annotation_position="top right")

    # Add a shaded region for the expected range
    fig.add_vrect(x0=lower_range, x1=upper_range, 
                fillcolor=range_color, opacity=0.1, line_width=0)
    
    
    # Update the layout with improved formatting
    fig.update_layout(
        height=600,
        width=1200,
        xaxis_title="Strike Price",
        yaxis_title="Option Category",
        margin=dict(t=150),  # Increased margin for interpretation boxes
        title=dict(
            text=f"{ticker} Options Volume by Strike Price (Expiry: {expiry_date})",
            font=dict(size=20),
            x=0.5,
            xanchor='center',
            y=0.95  # Moved title slightly down to avoid overlap
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=strike_range,
            ticktext=[f"{s:.1f}" for s in strike_range],
            tickangle=45
        ),
        coloraxis_colorbar=dict(
            title="Volume",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside"
        ),
        showlegend=False
    )
    
    # Ensure the heatmap cells are properly sized with consistent spacing
    fig.update_traces(
        xgap=1,  # Consistent gaps between cells
        ygap=1
    )
    
        # Prepare data for the meter chart
    category_totals = pivot_df.sum(axis=1)
    
    meter_data = {
        'strike_range': strike_range,
        'categories': categories,
        'category_totals': category_totals,
        'current_price': current_price,
        'ticker': ticker,
        'expiry_date': expiry_date,
        'call_categories': categories[:3],  # First 3 are call categories
        'put_categories': categories[3:],   # Last 3 are put categories
        'total_call_volume': total_call_volume,
        'total_put_volume': total_put_volume,
        'lower_range': lower_range,
        'upper_range': upper_range,
        'range_color': range_color
    }
    
    return fig, meter_data

def create_options_volume_meter(meter_data):
    """
    Create an enhanced horizontal meter chart showing options volume distribution by moneyness
    with additional directional bias indicators.
    
    Parameters:
    -----------
    meter_data : dict
        Dictionary containing processed data from the heatmap function
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The enhanced horizontal meter chart figure object
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Extract data from meter_data
    categories = meter_data['categories']
    category_totals = meter_data['category_totals']
    current_price = meter_data['current_price']
    ticker = meter_data['ticker']
    expiry_date = meter_data['expiry_date']
    call_categories = meter_data['call_categories']
    put_categories = meter_data['put_categories']
    total_call_volume = meter_data['total_call_volume'] 
    total_put_volume = meter_data['total_put_volume']
    lower_range = meter_data['lower_range']
    upper_range = meter_data['upper_range']
    range_color = meter_data['range_color']
    
    # Calculate total volume
    total_volume = total_call_volume + total_put_volume
    
    # Define colors for each category with more pronounced contrast
    category_colors = {
        'Call ITM': '#006400',      # Dark green
        'Call ATM': '#00AA00',      # Medium green
        'Call OTM': '#90EE90',      # Light green
        'Put OTM': '#FFC0CB',       # Light red
        'Put ATM': '#CD5C5C',       # Medium red
        'Put ITM': '#8B0000'        # Dark red
    }
    
    # Prepare data for stacked horizontal bar chart (meter)
    fig = go.Figure()
    
    # First add call categories from ITM to OTM
    for category in call_categories:
        if category_totals[category] > 0:
            percent = category_totals[category] / total_volume * 100
            fig.add_trace(
                go.Bar(
                    name=category,
                    y=['Volume Distribution'],
                    x=[category_totals[category]],
                    orientation='h',
                    marker_color=category_colors[category],
                    text=f"{category}: {percent:.1f}%",
                    hoverinfo='text',
                    textposition='inside' if percent > 5 else 'none'
                )
            )
    
    # Then add put categories from OTM to ITM
    for category in put_categories:
        if category_totals[category] > 0:
            percent = category_totals[category] / total_volume * 100
            fig.add_trace(
                go.Bar(
                    name=category,
                    y=['Volume Distribution'],
                    x=[category_totals[category]],
                    orientation='h',
                    marker_color=category_colors[category],
                    text=f"{category}: {percent:.1f}%",
                    hoverinfo='text',
                    textposition='inside' if percent > 5 else 'none'
                )
            )
    
    # Calculate detailed percentages for directional analysis
    call_itm_percent = category_totals.get('Call ITM', 0) / total_volume * 100
    call_atm_percent = category_totals.get('Call ATM', 0) / total_volume * 100
    call_otm_percent = category_totals.get('Call OTM', 0) / total_volume * 100
    
    put_itm_percent = category_totals.get('Put ITM', 0) / total_volume * 100
    put_atm_percent = category_totals.get('Put ATM', 0) / total_volume * 100
    put_otm_percent = category_totals.get('Put OTM', 0) / total_volume * 100
    
    # Calculate directional indicators
    bullish_bias = (call_otm_percent + call_atm_percent * 0.5) 
    bearish_bias = (put_otm_percent + put_atm_percent * 0.5)
    
    # Enhanced directional bias analysis
    call_put_ratio = total_call_volume/total_put_volume if total_put_volume > 0 else float('inf')
    
    # Calculate volume concentration in OTM options (speculative bias)
    otm_concentration = (call_otm_percent + put_otm_percent) / (call_itm_percent + put_itm_percent + 0.001)
    
    # Calculate skew (where is the volume concentrated)
    call_skew = call_otm_percent / (call_itm_percent + 0.001)  # Higher means calls concentrated OTM (bullish)
    put_skew = put_otm_percent / (put_itm_percent + 0.001)    # Higher means puts concentrated OTM (less bearish)
    
    # Calculate directional conviction score (higher = stronger conviction)
    if bullish_bias > bearish_bias:
        directional_score = (bullish_bias - bearish_bias) * (call_skew / max(1.0, put_skew))
        directional_bias = "BULLISH"
        score_color = "green"
    else:
        directional_score = (bearish_bias - bullish_bias) * (put_skew / max(1.0, call_skew))
        directional_bias = "BEARISH"
        score_color = "red"
    
    # Scale score to 0-100 range
    directional_score = min(100, directional_score * 2)
    
    # Determine sentiment intensity 
    if directional_score > 75:
        intensity = "STRONG"
    elif directional_score > 50:
        intensity = "MODERATE"
    elif directional_score > 25:
        intensity = "MILD"
    else:
        intensity = "NEUTRAL"
        directional_bias = "MIXED"
        score_color = "orange"
    
    # Calculate hedging ratio (ITM puts / total puts) - higher means more protective puts
    hedging_ratio = category_totals.get('Put ITM', 0) / (total_put_volume + 0.001) * 100
    
    # Calculate speculative ratio (OTM calls / total calls) - higher means more leveraged upside bets
    speculative_ratio = category_totals.get('Call OTM', 0) / (total_call_volume + 0.001) * 100
    
    # Determine market view
    if hedging_ratio > 50 and speculative_ratio < 30:
        market_view = "PROTECTIVE (Hedging against downside)"
        market_color = "orange"
    elif hedging_ratio < 30 and speculative_ratio > 50:
        market_view = "SPECULATIVE (Betting on upside)"
        market_color = "blue"
    elif hedging_ratio > 40 and speculative_ratio > 40:
        market_view = "VOLATILE (Positioning for big move either direction)"
        market_color = "purple"
    else:
        market_view = "BALANCED (No strong directional conviction)"
        market_color = "gray"
    
    # Calculate call premium total (higher premium = more capital deployed in calls)
    call_premium_factor = (
        category_totals.get('Call ITM', 0) * 3 + 
        category_totals.get('Call ATM', 0) * 2 + 
        category_totals.get('Call OTM', 0) * 1
    )
    
    # Calculate put premium total
    put_premium_factor = (
        category_totals.get('Put ITM', 0) * 3 + 
        category_totals.get('Put ATM', 0) * 2 + 
        category_totals.get('Put OTM', 0) * 1
    )
    
    # Calculate premium ratio (estimate of capital allocation)
    premium_ratio = call_premium_factor / (put_premium_factor + 0.001)
    
    # Determine capital bias based on premium ratio
    if premium_ratio > 2:
        capital_bias = "STRONG CALL PREMIUM (Heavy capital in calls)"
        capital_color = "darkgreen"
    elif premium_ratio > 1.3:
        capital_bias = "MODERATE CALL PREMIUM"
        capital_color = "green"
    elif premium_ratio > 0.8:
        capital_bias = "BALANCED PREMIUM"
        capital_color = "gray"
    elif premium_ratio > 0.5:
        capital_bias = "MODERATE PUT PREMIUM"
        capital_color = "darkred"
    else:
        capital_bias = "STRONG PUT PREMIUM (Heavy capital in puts)"
        capital_color = "red"
    
    # Add directional gauge (arrow and score)
    if directional_bias == "BULLISH":
        arrow = "▲"  # Up arrow for bullish
    elif directional_bias == "BEARISH":
        arrow = "▼"  # Down arrow for bearish
    else:
        arrow = "◆"  # Diamond for neutral/mixed
    
    # Add summary annotations
    fig.add_annotation(
        x=0.02,
        y=1.6,
        xref="paper",
        yref="paper",
        text=f"<b>Directional Bias:</b> {arrow} {intensity} {directional_bias} ({directional_score:.0f}/100)",
        showarrow=False,
        font=dict(size=16, color=score_color),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=score_color,
        borderwidth=2,
        borderpad=6
    )
    
    # Add call/put ratio
    fig.add_annotation(
        x=0.98,
        y=1.6,
        xref="paper",
        yref="paper",
        text=f"<b>Call/Put Ratio:</b> {call_put_ratio:.2f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="right",
        bgcolor="rgba(255,255,255,0.9)",
        borderwidth=1,
        borderpad=4
    )
    
    # Add market view annotation
    fig.add_annotation(
        x=0.02,
        y=1.3,
        xref="paper",
        yref="paper",
        text=f"<b>Market View:</b> {market_view}",
        showarrow=False,
        font=dict(size=14, color=market_color),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=market_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Add capital allocation bias
    fig.add_annotation(
        x=0.98,
        y=1.3,
        xref="paper",
        yref="paper",
        text=f"<b>Capital Bias:</b> {capital_bias}",
        showarrow=False,
        font=dict(size=14, color=capital_color),
        align="right",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=capital_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Add skew indicators
    fig.add_annotation(
        x=0.02,
        y=1.0,
        xref="paper",
        yref="paper",
        text=f"<b>Call Skew:</b> {call_skew:.2f} (>1 = OTM bias)",
        showarrow=False,
        font=dict(size=12, color="green"),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        borderwidth=1,
        borderpad=4
    )
    
    fig.add_annotation(
        x=0.98,
        y=1.0,
        xref="paper",
        yref="paper",
        text=f"<b>Put Skew:</b> {put_skew:.2f} (>1 = OTM bias)",
        showarrow=False,
        font=dict(size=12, color="red"),
        align="right",
        bgcolor="rgba(255,255,255,0.9)",
        borderwidth=1,
        borderpad=4
    )
    
    # Add key percentages (calls)
    call_bullet_color = "darkgreen" if call_itm_percent > call_otm_percent else "green"
    fig.add_annotation(
        x=0.25,
        y=-0.45,
        xref="paper",
        yref="paper",
        text=f"<b>Calls:</b> ITM {call_itm_percent:.1f}% • ATM {call_atm_percent:.1f}% • OTM {call_otm_percent:.1f}%",
        showarrow=False,
        font=dict(size=12, color=call_bullet_color),
        align="center",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=call_bullet_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Add key percentages (puts)
    put_bullet_color = "darkred" if put_itm_percent > put_otm_percent else "red"
    fig.add_annotation(
        x=0.75,
        y=-0.45,
        xref="paper",
        yref="paper",
        text=f"<b>Puts:</b> OTM {put_otm_percent:.1f}% • ATM {put_atm_percent:.1f}% • ITM {put_itm_percent:.1f}%",
        showarrow=False,
        font=dict(size=12, color=put_bullet_color),
        align="center",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=put_bullet_color,
        borderwidth=1,
        borderpad=4
    )
    
    # Add a directional strength meter (horizontal gauge)
    gauge_x = np.linspace(0, 100, 100)
    gauge_y = np.zeros(100)
    
    midpoint = 50
    if directional_bias == "BULLISH":
        indicator_pos = midpoint + directional_score/2
    elif directional_bias == "BEARISH":
        indicator_pos = midpoint - directional_score/2
    else:
        indicator_pos = midpoint
    
    # Create the gauge axis
    fig.add_trace(
        go.Scatter(
            x=gauge_x,
            y=gauge_y,
            mode='lines',
            line=dict(
                color='lightgray',
                width=5,
            ),
            showlegend=False,
            hoverinfo='none',
            yaxis='y2'
        )
    )
    
    # Add colored segments to the gauge
    fig.add_trace(
        go.Scatter(
            x=gauge_x[:midpoint],
            y=gauge_y[:midpoint],
            mode='lines',
            line=dict(
                color='red',
                width=5,
            ),
            showlegend=False,
            hoverinfo='none',
            yaxis='y2'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=gauge_x[midpoint:],
            y=gauge_y[midpoint:],
            mode='lines',
            line=dict(
                color='green',
                width=5,
            ),
            showlegend=False,
            hoverinfo='none',
            yaxis='y2'
        )
    )
    
    # Add position indicator on the gauge
    fig.add_trace(
        go.Scatter(
            x=[indicator_pos],
            y=[0],
            mode='markers',
            marker=dict(
                color='black',
                size=12,
                symbol='diamond',
            ),
            showlegend=False,
            hoverinfo='none',
            yaxis='y2'
        )
    )
    
    # Add gauge labels
    fig.add_annotation(
        x=0,
        y=0.5,
        xref="paper",
        yref="y2",
        text="Bearish",
        showarrow=False,
        font=dict(size=10, color="red"),
        yshift=15
    )
    
    fig.add_annotation(
        x=1,
        y=0.5,
        xref="paper",
        yref="y2",
        text="Bullish",
        showarrow=False,
        font=dict(size=10, color="green"),
        yshift=15
    )
    
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="y2",
        text="Neutral",
        showarrow=False,
        font=dict(size=10, color="gray"),
        yshift=15
    )
    
    # Update the layout
    fig.update_layout(
        height=420,  # Increased height to accommodate directional gauge
        width=1200,  # Same width as the heatmap
        title=dict(
            text=f"{ticker} Options Volume Distribution & Directional Bias (Expiry: {expiry_date})",
            font=dict(size=16),
            x=0.5,
            xanchor='center',
            y=0.97
        ),
        barmode='stack',  # Stack the bar segments
        uniformtext=dict(minsize=10, mode='hide'),  # Text size in bars
        yaxis=dict(
            title=None,
            domain=[0.2, 0.85],  # Adjust domain to make room for gauge
            showticklabels=False
        ),
        yaxis2=dict(
            title=None,
            domain=[0, 0.1],  # Position of gauge
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        xaxis=dict(
            title="Volume",
            domain=[0, 1]
        ),
        xaxis2=dict(
            title="Directional Bias",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(t=120, b=150),  # Increased margins for annotations
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.87,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def calculate_delta_adjusted_oi(calls_df, puts_df):
    """
    Calculate delta-adjusted open interest for calls and puts using pandas DataFrames.
    
    Parameters:
    -----------
    calls_df : pandas.DataFrame
        DataFrame containing call options data
    puts_df : pandas.DataFrame
        DataFrame containing put options data
        
    Returns:
    --------
    tuple
        (call_daoi_by_strike, put_daoi_by_strike, total_call_daoi, total_put_daoi)
    """
    import pandas as pd
    import numpy as np
    
    # Initialize dictionaries to store results
    call_daoi_by_strike = {}
    put_daoi_by_strike = {}
    
    # Calculate delta-adjusted OI for calls
    if not calls_df.empty and 'openInterest' in calls_df.columns and 'calc_delta' in calls_df.columns:
        # Filter for valid data
        valid_calls = calls_df[(calls_df['openInterest'] > 0) & (~calls_df['calc_delta'].isna())]
        
        # Calculate delta-adjusted OI
        valid_calls['daoi'] = valid_calls['openInterest'] * valid_calls['calc_delta'].abs()
        
        # Group by strike and sum
        if not valid_calls.empty:
            call_daoi = valid_calls.groupby('strike')['daoi'].sum()
            call_daoi_by_strike = call_daoi.to_dict()
    
    # Calculate delta-adjusted OI for puts
    if not puts_df.empty and 'openInterest' in puts_df.columns and 'calc_delta' in puts_df.columns:
        # Filter for valid data
        valid_puts = puts_df[(puts_df['openInterest'] > 0) & (~puts_df['calc_delta'].isna())]
        
        # Calculate delta-adjusted OI
        valid_puts['daoi'] = valid_puts['openInterest'] * valid_puts['calc_delta'].abs()
        
        # Group by strike and sum
        if not valid_puts.empty:
            put_daoi = valid_puts.groupby('strike')['daoi'].sum()
            put_daoi_by_strike = put_daoi.to_dict()
    
    # Calculate totals
    total_call_daoi = sum(call_daoi_by_strike.values())
    total_put_daoi = sum(put_daoi_by_strike.values())
    
    return call_daoi_by_strike, put_daoi_by_strike, total_call_daoi, total_put_daoi

def find_option_walls(call_daoi_by_strike, put_daoi_by_strike, percentile_threshold=90):
    """
    Identify option walls based on delta-adjusted open interest.
    
    Parameters:
    -----------
    call_daoi_by_strike : dict
        Dictionary with strike prices as keys and delta-adjusted OI as values for calls
    put_daoi_by_strike : dict
        Dictionary with strike prices as keys and delta-adjusted OI as values for puts
    percentile_threshold : int
        Percentile threshold to identify significant walls (default: 90)
        
    Returns:
    --------
    tuple
        (call_walls, put_walls)
    """
    import numpy as np
    
    call_walls = {}
    put_walls = {}
    
    # Identify call walls (significant concentrations)
    if call_daoi_by_strike:
        call_values = list(call_daoi_by_strike.values())
        if len(call_values) > 1:  # Need at least 2 values for percentile calculation
            call_threshold = np.percentile(call_values, percentile_threshold)
            call_walls = {k: v for k, v in call_daoi_by_strike.items() if v >= call_threshold}
        else:
            # If only one value, it's a wall by default
            call_walls = call_daoi_by_strike.copy()
    
    # Identify put walls (significant concentrations)
    if put_daoi_by_strike:
        put_values = list(put_daoi_by_strike.values())
        if len(put_values) > 1:  # Need at least 2 values for percentile calculation
            put_threshold = np.percentile(put_values, percentile_threshold)
            put_walls = {k: v for k, v in put_daoi_by_strike.items() if v >= put_threshold}
        else:
            # If only one value, it's a wall by default
            put_walls = put_daoi_by_strike.copy()
    
    return call_walls, put_walls

def create_delta_adjusted_oi_chart(calls_df, puts_df, current_price, ticker, expiry_date):
    """
    Create a chart visualizing delta-adjusted open interest for calls and puts.
    
    Parameters:
    -----------
    calls_df : pandas.DataFrame
        DataFrame containing call options data
    puts_df : pandas.DataFrame
        DataFrame containing put options data
    current_price : float
        Current price of the underlying asset
    ticker : str
        Ticker symbol of the underlying asset
    expiry_date : str
        Expiration date of the options
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The chart figure object
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Calculate delta-adjusted OI
    call_daoi_by_strike, put_daoi_by_strike, total_call_daoi, total_put_daoi = calculate_delta_adjusted_oi(calls_df, puts_df)
    
    # Find option walls
    call_walls, put_walls = find_option_walls(call_daoi_by_strike, put_daoi_by_strike)
    
    # Prepare data for visualization
    strikes = sorted(set(list(call_daoi_by_strike.keys()) + list(put_daoi_by_strike.keys())))
    
    # If no strikes were found, return an empty figure with a message
    if not strikes:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No valid delta-adjusted open interest data available",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=500, width=1200)
        return fig
    
    call_daoi_values = [call_daoi_by_strike.get(strike, 0) for strike in strikes]
    put_daoi_values = [put_daoi_by_strike.get(strike, 0) * -1 for strike in strikes]  # Invert for visualization
    
    # Create the figure
    fig = go.Figure()
    
    # Add call delta-adjusted OI bars
    fig.add_trace(go.Bar(
        name='Call Delta-Adjusted OI',
        x=strikes,
        y=call_daoi_values,
        marker_color='#9cec5b',
        opacity=0.7
    ))
    
    # Add put delta-adjusted OI bars
    fig.add_trace(go.Bar(
        name='Put Delta-Adjusted OI',
        x=strikes,
        y=put_daoi_values,
        marker_color="#c71585",
        opacity=0.7
    ))
    
    # Add current price line
    fig.add_vline(
        x=current_price,
        line_width=2,
        line_dash="dash",
        line_color="#511cf6",
        annotation_text=f"Current Price ${current_price:.2f}",
        annotation_font_color="orange",
        annotation_y=0.95,  # Position slightly below the top (1.0 = top)
        annotation_yref="paper",
        annotation_position="top right"
    )
    

    # Add interpretation annotations
    call_put_daoi_ratio = total_call_daoi / total_put_daoi if total_put_daoi > 0 else float('inf')
    
    if call_put_daoi_ratio > 1.5:
        sentiment = "BULLISH"
        sentiment_color = "green"
    elif call_put_daoi_ratio < 0.67:
        sentiment = "BEARISH"
        sentiment_color = "red"
    else:
        sentiment = "NEUTRAL"
        sentiment_color = "orange"
    
    # Add ratio and sentiment annotation
    fig.add_annotation(
        x=0.5,
        y=1.15,
        xref="paper",
        yref="paper",
        text=f"Delta-Adjusted OI Ratio (Calls/Puts): {call_put_daoi_ratio:.2f} - {sentiment} BIAS",
        showarrow=False,
        font=dict(size=14, color=sentiment_color),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=sentiment_color,
        borderwidth=2,
        borderpad=6,
        align="center"
    )
    

    # Add interpretation guidance
    interpretation_text = ""
    interpretation_color = "black"
    if call_walls and put_walls:
        highest_call_wall = max(call_walls, key=lambda x: call_walls[x])
        highest_put_wall = max(put_walls, key=lambda x: put_walls[x])
        
        if highest_call_wall > current_price and highest_put_wall < current_price:
            interpretation_text = f"↔️ Price likely range-bound between ${highest_put_wall:.2f} and ${highest_call_wall:.2f}"
            interpretation_color = "blue"
        elif highest_call_wall > current_price:
            interpretation_text = f"🔴 Resistance at ${highest_call_wall:.2f} may cap upside moves"
            interpretation_color = "red"
        elif highest_put_wall < current_price:
            interpretation_text = f"🟢 Support at ${highest_put_wall:.2f} may limit downside moves"
            interpretation_color = "green"
    
    if interpretation_text:
        fig.add_annotation(
            x=0.02,
            y=0.9,
            xref="paper",
            yref="paper",
            text=interpretation_text,
            showarrow=False,
            font=dict(size=12,color=interpretation_color),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=interpretation_color,
            borderwidth=2,
            borderpad=6,
            align="center"
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Delta-Adjusted Open Interest (Expiry: {expiry_date})",
        xaxis_title="Strike Price",
        yaxis_title="Delta-Adjusted Open Interest",
        barmode='overlay',
        height=700,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=150)  # Increased top margin for annotations
    )
    
    return fig
   
  
     
def main_dashboard():

    
    # Dashboard input controls
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        user_ticker = st.text_input(
            "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):",
            value=st.session_state.get("saved_ticker", "SPY"),
            key="dashboard_ticker"
        )
    
    with col3:
        strike_default = st.session_state.get("strike_range", 20)
        if isinstance(strike_default, tuple):
            strike_default = int(strike_default[0])
        strike_val = st.slider(
            "Strike Range (±)",
            min_value=1,
            max_value=2000,
            value=strike_default,
            step=1
        )
        st.session_state.strike_range = strike_val
            
    ticker = user_ticker
    if ticker and ticker != st.session_state.get("saved_ticker"):
        st.session_state.saved_ticker = ticker
        
        # Add this: update last_ticker and force refresh all tabs that depend on ticker
        if ticker != st.session_state.get("last_ticker"):
            st.session_state.last_ticker = ticker
            
    
    if not ticker:
        st.warning("Please enter a valid ticker.")
        return

    # Get available options dates
    stock = yf.Ticker(ticker)
    available_dates = stock.options
    
    
    
    
    
    with col2:
        expiry_date_str = st.selectbox(
            "Select an Exp. Date:",
            options=available_dates,
            key="dashboard_expiry_main"
        ) if available_dates else None
    if expiry_date_str is not None and expiry_date_str != st.session_state.get("last_expiry_date"):
        st.session_state.last_expiry_date = expiry_date_str
    
    if not available_dates or not expiry_date_str:
        st.warning("No options data available or no expiration date selected.")
        return
    
    
    # Load data efficiently with caching
    calls, puts, S, t, selected_expiry, today = load_options_data(ticker, expiry_date_str)
    current_price = S
    expiry_date_str = selected_expiry.strftime("%Y-%m-%d")



    
    
    result = create_options_heatmap(calls, puts, current_price, ticker, expiry_date_str)
    if isinstance(result, tuple):
        fig_heat, meter_data = result
    else:
        fig_heat = result
        meter_data = None
            
    #fig_heat, meter_data, *extra_values = create_options_heatmap(calls, puts, current_price, ticker, expiry_date_str)
    st.plotly_chart(fig_heat,use_container_width=True)
    if meter_data:
        meter_fig = create_options_volume_meter(meter_data)
        st.plotly_chart(meter_fig, use_container_width=True)    
    
    calls_filtered = filter_to_n_strikes(calls, 100, S)
    puts_filtered = filter_to_n_strikes(puts, 100, S)
    
    fig_daoi = create_delta_adjusted_oi_chart(calls_filtered, puts_filtered, current_price, ticker, expiry_date_str)
    st.plotly_chart(fig_daoi, use_container_width=True)
    

    
if __name__ == "__main__":
    main_dashboard()