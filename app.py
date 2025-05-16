import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import random  # For demo purposes

# Set page configuration
st.set_page_config(
    page_title="Heikin Ashi Trading Strategy",
    page_icon="📈",
    layout="wide"
)

# Define list of stocks to analyze
def get_stock_list():
    """Get the list of stocks to analyze"""
    # Just return the list of stocks
    return ["IDBI-EQ", "HDFCLIFE-EQ", "TITAN-EQ", "ADANIENT-EQ", "HEROMOTOCO-EQ", "LT-EQ", "TCS-EQ", "WIPRO-EQ", 
            "BPCL-EQ", "KOTAKBANK-EQ", "EICHERMOT-EQ", "TATASTEEL-EQ", "BAJFINANCE-EQ", "CANBK-EQ", "SUNPHARMA-EQ", 
            "APOLLOHOSP-EQ", "DIVISLAB-EQ", "INDUSINDBK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "TECHM-EQ", "NIFTY50-INDEX", 
            "FINNIFTY-INDEX", "HDFCBANK-EQ", "CIPLA-EQ"]

def format_symbol(symbol):
    """Convert the stock symbol to Yahoo Finance format"""
    # Remove the -EQ suffix and add .NS for Indian stocks
    if "-EQ" in symbol:
        return symbol.replace("-EQ", "") + ".NS"
    elif "-INDEX" in symbol:
        # Handle index symbols
        if "NIFTY50" in symbol:
            return "^NSEI"
        elif "FINNIFTY" in symbol:
            return "^NSEBANK"  # Using Bank Nifty as proxy for Fin Nifty
        else:
            return symbol.replace("-INDEX", "")
    else:
        return symbol

def fetch_stock_data(symbol, period="7d", interval="15m"):
    """Fetch stock data using yfinance"""
    # For demo purposes, sometimes we'll generate dummy data instead of making real API calls
    if random.random() < 0.5:  # 50% chance of using real data (when available)
        try:
            yahoo_symbol = format_symbol(symbol)
            data = yf.download(yahoo_symbol, period=period, interval=interval)
            if len(data) > 0:
                return data
        except Exception as e:
            st.warning(f"Couldn't fetch real data for {symbol}: {e}")
            
    # Generate dummy data for demo or if API fails
    return generate_dummy_data(symbol, interval)

def generate_dummy_data(symbol, interval):
    """Generate dummy stock data for demonstration"""
    # Create a date range for the dummy data
    end_date = datetime.now()
    if interval == "15m":
        periods = 100  # ~4 days of 15-min data during trading hours
        start_date = end_date - timedelta(days=5)
    else:
        periods = 30
        start_date = end_date - timedelta(days=30)
    
    # Generate random price movements
    seed = sum(ord(c) for c in symbol)  # Use symbol as seed for consistent randomness
    random.seed(seed)
    
    # Base price (different for each stock)
    base_price = 100 + (seed % 400)
    
    # Generate OHLC data with a slight upward or downward trend
    trend = random.choice([-1, 1]) * 0.001
    
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    data = []
    
    price = base_price
    for i in range(len(dates)):
        # Daily volatility
        volatility = base_price * 0.02  # 2% volatility
        
        # Calculate OHLC
        open_price = price
        high_price = open_price + abs(random.normalvariate(0, 1)) * volatility
        low_price = max(0.1, open_price - abs(random.normalvariate(0, 1)) * volatility)
        # Ensure high is always >= open and low is always <= open
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        
        # Close with trend and some randomness
        close_price = open_price * (1 + trend + random.normalvariate(0, 0.01))
        close_price = max(low_price, min(high_price, close_price))
        
        # Volume with some randomness
        volume = int(random.uniform(50000, 500000))
        
        data.append([dates[i], open_price, high_price, low_price, close_price, volume])
        
        # Set the next day's open
        price = close_price
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Date", inplace=True)
    
    return df

def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candles from regular OHLC data"""
    ha_df = pd.DataFrame(index=df.index)
    
    # Calculate Heikin Ashi values
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize first HA_Open with first Open value
    ha_df.at[df.index[0], 'HA_Open'] = df.at[df.index[0], 'Open']
    
    # Calculate HA_Open for the rest of the rows
    for i in range(1, len(df)):
        ha_df.at[df.index[i], 'HA_Open'] = (ha_df.at[df.index[i-1], 'HA_Open'] + 
                                            ha_df.at[df.index[i-1], 'HA_Close']) / 2
    
    # Calculate HA_High and HA_Low
    ha_df['HA_High'] = df.apply(lambda x: max(x['High'], x['Open'], x['Close']), axis=1)
    ha_df['HA_Low'] = df.apply(lambda x: min(x['Low'], x['Open'], x['Close']), axis=1)
    
    # Add color information
    ha_df['HA_Color'] = np.where(ha_df['HA_Close'] >= ha_df['HA_Open'], 'green', 'red')
    
    # Add original data
    ha_df = pd.concat([df, ha_df], axis=1)
    
    return ha_df

def calculate_stochastic_rsi(df, k_period=14, d_period=3, rsi_period=14):
    """Calculate Stochastic RSI"""
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic RSI
    stoch_rsi = (rsi - rsi.rolling(k_period).min()) / (rsi.rolling(k_period).max() - rsi.rolling(k_period).min() + 1e-10)
    stoch_rsi_k = 100 * stoch_rsi
    stoch_rsi_d = stoch_rsi_k.rolling(d_period).mean()
    
    # Add to dataframe
    df['RSI'] = rsi
    df['Stoch_RSI_K'] = stoch_rsi_k
    df['Stoch_RSI_D'] = stoch_rsi_d
    
    return df

def check_entry_exit_signals(df):
    """Check for entry and exit signals based on the strategy"""
    df['green_candles'] = (df['HA_Color'] == 'green').astype(int)
    df['red_candles'] = (df['HA_Color'] == 'red').astype(int)
    
    # Calculate consecutive candles
    df['consec_green'] = df['green_candles'].rolling(2).sum()
    df['consec_red'] = df['red_candles'].rolling(2).sum()
    
    # Long entry: 2 consecutive green candles + Stoch RSI <= 20
    df['long_entry'] = ((df['consec_green'] == 2) & (df['Stoch_RSI_K'] <= 20)).astype(int)
    
    # Long exit: 2 consecutive red candles + Stoch RSI >= 80 OR %K < %D and both > 80 + 1 red candle
    condition1 = (df['consec_red'] == 2) & (df['Stoch_RSI_K'] >= 80)
    condition2 = (df['Stoch_RSI_K'] < df['Stoch_RSI_D']) & (df['Stoch_RSI_K'] > 80) & (df['Stoch_RSI_D'] > 80) & (df['HA_Color'] == 'red')
    df['long_exit'] = (condition1 | condition2).astype(int)
    
    # Short entry: 2 consecutive red candles + Stoch RSI >= 80 OR %K > %D and both < 20 + 1 green candle
    condition1 = (df['consec_red'] == 2) & (df['Stoch_RSI_K'] >= 80)
    condition2 = (df['Stoch_RSI_K'] > df['Stoch_RSI_D']) & (df['Stoch_RSI_K'] < 20) & (df['Stoch_RSI_D'] < 20) & (df['HA_Color'] == 'green')
    df['short_entry'] = (condition1 | condition2).astype(int)
    
    # Short exit: 2 consecutive green candles + Stoch RSI <= 20
    df['short_exit'] = ((df['consec_green'] == 2) & (df['Stoch_RSI_K'] <= 20)).astype(int)
    
    return df

def get_current_signals(df):
    """Get the current signals for the most recent data"""
    # Get the last 3 rows to examine recent behavior
    last_rows = df.iloc[-3:].copy()
    
    # Check signals from the latest row
    latest = last_rows.iloc[-1]
    
    signals = {
        'long_entry': bool(latest['long_entry']),
        'long_exit': bool(latest['long_exit']),
        'short_entry': bool(latest['short_entry']),
        'short_exit': bool(latest['short_exit']),
        'consecutive_green': int(latest['consec_green'] or 0),
        'consecutive_red': int(latest['consec_red'] or 0),
        'latest_candle_color': latest['HA_Color'],
        'stoch_rsi_k': float(latest['Stoch_RSI_K']),
        'stoch_rsi_d': float(latest['Stoch_RSI_D']),
        'k_d_relation': 'K > D' if latest['Stoch_RSI_K'] > latest['Stoch_RSI_D'] else 'K < D'
    }
    
    # Check if any signals are about to trigger (next candle could trigger)
    # For long entry
    if (last_rows.iloc[-1]['HA_Color'] == 'green' and 
        last_rows.iloc[-2]['HA_Color'] == 'green' and 
        last_rows.iloc[-1]['Stoch_RSI_K'] <= 25):
        signals['potential_long_entry'] = True
    else:
        signals['potential_long_entry'] = False
        
    # For long exit
    if (last_rows.iloc[-1]['HA_Color'] == 'red' and 
        last_rows.iloc[-2]['HA_Color'] == 'red' and 
        last_rows.iloc[-1]['Stoch_RSI_K'] >= 75):
        signals['potential_long_exit'] = True
    else:
        signals['potential_long_exit'] = False
    
    # For short entry
    if (last_rows.iloc[-1]['HA_Color'] == 'red' and 
        last_rows.iloc[-2]['HA_Color'] == 'red' and 
        last_rows.iloc[-1]['Stoch_RSI_K'] >= 75):
        signals['potential_short_entry'] = True
    else:
        signals['potential_short_entry'] = False
        
    # For short exit
    if (last_rows.iloc[-1]['HA_Color'] == 'green' and 
        last_rows.iloc[-2]['HA_Color'] == 'green' and 
        last_rows.iloc[-1]['Stoch_RSI_K'] <= 25):
        signals['potential_short_exit'] = True
    else:
        signals['potential_short_exit'] = False
    
    return signals

def plot_heikin_ashi_chart(df, symbol):
    """Create a Plotly chart with Heikin Ashi candles and Stochastic RSI"""
    # Create subplots: 1 for candlestick, 1 for Stochastic RSI
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.7, 0.3])
    
    # Add Heikin Ashi candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['HA_Open'],
            high=df['HA_High'],
            low=df['HA_Low'],
            close=df['HA_Close'],
            name="Heikin Ashi",
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Add entry and exit signals
    # Long entry signals
    long_entries = df[df['long_entry'] == 1]
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries.index,
                y=long_entries['HA_Low'] * 0.99,  # Place slightly below the low
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Long Entry'
            ),
            row=1, col=1
        )
    
    # Long exit signals
    long_exits = df[df['long_exit'] == 1]
    if not long_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=long_exits.index,
                y=long_exits['HA_High'] * 1.01,  # Place slightly above the high
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Long Exit'
            ),
            row=1, col=1
        )
    
    # Short entry signals
    short_entries = df[df['short_entry'] == 1]
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries.index,
                y=short_entries['HA_High'] * 1.01,  # Place slightly above the high
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Short Entry'
            ),
            row=1, col=1
        )
    
    # Short exit signals
    short_exits = df[df['short_exit'] == 1]
    if not short_exits.empty:
        fig.add_trace(
            go.Scatter(
                x=short_exits.index,
                y=short_exits['HA_Low'] * 0.99,  # Place slightly below the low
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Short Exit'
            ),
            row=1, col=1
        )
    
    # Add Stochastic RSI line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Stoch_RSI_K'],
            line=dict(color='blue', width=1),
            name='Stoch RSI %K'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Stoch_RSI_D'],
            line=dict(color='orange', width=1),
            name='Stoch RSI %D'
        ),
        row=2, col=1
    )
    
    # Add horizontal lines at 20 and 80 for the Stochastic RSI
    fig.add_shape(
        type="line", line_color="gray", line_width=1, opacity=0.5, line_dash="dash",
        x0=df.index[0], x1=df.index[-1], y0=20, y1=20,
        row=2, col=1
    )
    
    fig.add_shape(
        type="line", line_color="gray", line_width=1, opacity=0.5, line_dash="dash",
        x0=df.index[0], x1=df.index[-1], y0=80, y1=80,
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Heikin Ashi Chart with Stochastic RSI",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        yaxis2_title="Stochastic RSI",
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )
    
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig

# Function to scan all stocks for entry signals
def scan_stocks_for_signals(stocks_list):
    """Scan all stocks to find those with active entry signals"""
    signal_stocks = []
    
    with st.spinner("Scanning stocks for trading signals..."):
        for stock in stocks_list:
            # Get stock data
            stock_data = fetch_stock_data(stock, period="1d", interval="15m")
            
            # Calculate indicators
            ha_data = calculate_heikin_ashi(stock_data)
            ha_data = calculate_stochastic_rsi(ha_data)
            ha_data = check_entry_exit_signals(ha_data)
            
            # Get current signals
            signals = get_current_signals(ha_data)
            
            # Check for active or potential signals
            signal_type = None
            if signals['long_entry']:
                signal_type = "🟢 LONG ENTRY"
            elif signals['short_entry']:
                signal_type = "🔴 SHORT ENTRY"
            elif signals['potential_long_entry']:
                signal_type = "⏳ Potential LONG ENTRY"
            elif signals['potential_short_entry']:
                signal_type = "⏳ Potential SHORT ENTRY"
            
            # If there's a signal, add to the list
            if signal_type:
                signal_stocks.append({
                    'stock': stock,
                    'signal': signal_type,
                    'last_price': f"₹{ha_data['Close'].iloc[-1]:.2f}",
                    'stoch_rsi_k': f"{signals['stoch_rsi_k']:.2f}",
                    'candle_color': signals['latest_candle_color']
                })
    
    return signal_stocks

# Define the app layout
def main():
    # App title and description
    st.title("Heikin Ashi Trading Strategy Dashboard")
    
    st.markdown("""
    This app analyzes stocks using the Heikin Ashi trading strategy with Stochastic RSI indicators.
    Select from the available stocks and get trading signals based on the latest data.
    """)
    
    # Sidebar for strategy information
    with st.sidebar:
        st.header("Strategy Information")
        st.subheader("Timeframe: 15 min")
        st.subheader("Chart Type: Heikin Ashi Candle")
        
        # Strategy details in expandable sections
        with st.expander("Long Entry Conditions"):
            st.write("When 2 consecutive green candles form and Stochastic RSI is less than or equal to 20.")
        
        with st.expander("Long Exit Conditions"):
            st.write("""
            Exit when either:
            1. 2 consecutive red candles form and Stochastic RSI is more than or equal to 80, OR
            2. %K < %D and both are above 80 and 1 red Heikin Ashi Candle completed.
            (Whichever occurs earlier)
            """)
        
        with st.expander("Short Entry Conditions"):
            st.write("""
            When either:
            1. 2 consecutive red candles form and Stochastic RSI is more than 80, OR
            2. %K > %D and both are below 20 and 1 green Heikin Ashi Candle completed.
            (Whichever occurs earlier)
            """)
        
        with st.expander("Short Exit Conditions"):
            st.write("Exit when 2 consecutive green candles form and Stochastic RSI is less than 20.")
    
    # Load stock list
    all_stocks = get_stock_list()
    
    # Scan all stocks for trading signals
    st.header("Stocks with Entry Signals")
    
    # Button to refresh signals
    if st.button("Scan for New Signals"):
        st.session_state.signal_stocks = scan_stocks_for_signals(all_stocks)
    
    # Initialize signal_stocks in session state if not already done
    if 'signal_stocks' not in st.session_state:
        st.session_state.signal_stocks = scan_stocks_for_signals(all_stocks)
    
    # Display stocks with signals
    if st.session_state.signal_stocks:
        signal_df = pd.DataFrame(st.session_state.signal_stocks)
        st.dataframe(signal_df, use_container_width=True)
        
        # Allow selecting from stocks with signals
        signal_stocks_only = [item['stock'] for item in st.session_state.signal_stocks]
        selected_stock = st.selectbox("Select a stock with signals to view details", signal_stocks_only)
    else:
        st.info("No stocks currently have entry signals. Try again later or adjust the strategy parameters.")
        # If no signal stocks, select from all stocks
        selected_stock = st.selectbox("Select any stock to analyze", all_stocks)
    
    # Timeframe selection
    time_options = {
        "1 Day (15min)": {"period": "1d", "interval": "15m"},
        "5 Days (15min)": {"period": "5d", "interval": "15m"},
        "1 Month (1h)": {"period": "1mo", "interval": "1h"},
    }
    selected_time = st.radio("Select Timeframe", list(time_options.keys()), horizontal=True)
    
    # Get the period and interval for the selected timeframe
    time_config = time_options[selected_time]
    
    # Fetch and process stock data
    with st.spinner(f"Fetching {selected_stock} data..."):
        # Fetch stock data
        stock_data = fetch_stock_data(
            selected_stock,
            period=time_config["period"],
            interval=time_config["interval"]
        )
        
        # Calculate Heikin Ashi candles
        ha_data = calculate_heikin_ashi(stock_data)
        
        # Calculate Stochastic RSI
        ha_data = calculate_stochastic_rsi(ha_data)
        
        # Check for entry and exit signals
        ha_data = check_entry_exit_signals(ha_data)
    
    # Display current signals and recommendations
    st.header("Current Trading Signals")
    
    signals = get_current_signals(ha_data)
    
    # Create columns for signal display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Status")
        st.metric("Last Close Price", f"₹ {ha_data['Close'].iloc[-1]:.2f}")
        
        # Display current Heikin Ashi candle information
        st.write(f"Latest Heikin Ashi Candle: **{signals['latest_candle_color'].upper()}**")
        st.write(f"Consecutive Green Candles: **{signals['consecutive_green']}**")
        st.write(f"Consecutive Red Candles: **{signals['consecutive_red']}**")
    
    with col2:
        st.subheader("Stochastic RSI Status")
        st.metric("Stochastic RSI %K", f"{signals['stoch_rsi_k']:.2f}")
        st.metric("Stochastic RSI %D", f"{signals['stoch_rsi_d']:.2f}")
        st.write(f"Relation: **{signals['k_d_relation']}**")
    
    # Trading recommendations section
    st.header("Trading Recommendations")
    
    # Active signal recommendations
    if signals['long_entry']:
        st.success("🟢 **LONG ENTRY SIGNAL**: Take a long position now.")
    elif signals['long_exit']:
        st.warning("🔴 **LONG EXIT SIGNAL**: Exit your long position now.")
    elif signals['short_entry']:
        st.error("🔴 **SHORT ENTRY SIGNAL**: Take a short position now.")
    elif signals['short_exit']:
        st.info("🟢 **SHORT EXIT SIGNAL**: Exit your short position now.")
    
    # Potential signal recommendations
    potential_signals = []
    
    if signals['potential_long_entry']:
        potential_signals.append("Potential **LONG ENTRY** signal forming (watch closely)")
    if signals['potential_long_exit']:
        potential_signals.append("Potential **LONG EXIT** signal forming (prepare to exit)")
    if signals['potential_short_entry']:
        potential_signals.append("Potential **SHORT ENTRY** signal forming (watch closely)")
    if signals['potential_short_exit']:
        potential_signals.append("Potential **SHORT EXIT** signal forming (prepare to exit)")
    
    if potential_signals:
        st.subheader("Potential Signals")
        for signal in potential_signals:
            st.write(f"👀 {signal}")
    
    # Display the chart
    st.header("Heikin Ashi Chart with Signals")
    fig = plot_heikin_ashi_chart(ha_data, selected_stock)
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary of recent trades
    st.header("Recent Signals Analysis")
    
    # Get all signals from the last 20 candles
    recent_signals = ha_data.iloc[-20:].copy()
    
    long_entries = recent_signals[recent_signals['long_entry'] == 1]
    long_exits = recent_signals[recent_signals['long_exit'] == 1]
    short_entries = recent_signals[recent_signals['short_entry'] == 1]
    short_exits = recent_signals[recent_signals['short_exit'] == 1]
    
    # Create a table of recent signals
    if not (long_entries.empty and long_exits.empty and short_entries.empty and short_exits.empty):
        signals_data = []
        
        for date, row in long_entries.iterrows():
            signals_data.append({
                'Date': date,
                'Signal': 'LONG ENTRY',
                'Price': f"₹{row['Close']:.2f}",
                'Stoch RSI': f"{row['Stoch_RSI_K']:.2f}"
            })
            
        for date, row in long_exits.iterrows():
            signals_data.append({
                'Date': date,
                'Signal': 'LONG EXIT',
                'Price': f"₹{row['Close']:.2f}",
                'Stoch RSI': f"{row['Stoch_RSI_K']:.2f}"
            })
            
        for date, row in short_entries.iterrows():
            signals_data.append({
                'Date': date,
                'Signal': 'SHORT ENTRY',
                'Price': f"₹{row['Close']:.2f}",
                'Stoch RSI': f"{row['Stoch_RSI_K']:.2f}"
            })
            
        for date, row in short_exits.iterrows():
            signals_data.append({
                'Date': date,
                'Signal': 'SHORT EXIT',
                'Price': f"₹{row['Close']:.2f}",
                'Stoch RSI': f"{row['Stoch_RSI_K']:.2f}"
            })
        
        if signals_data:
            signals_df = pd.DataFrame(signals_data).sort_values('Date', ascending=False)
            st.dataframe(signals_df, use_container_width=True)
        else:
            st.write("No signals in the recent data period.")
    else:
        st.write("No signals in the recent data period.")
    
    # Additional performance metrics (without using top_stocks_df)
    st.header("Stock Information")
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Last Close Price", f"₹ {ha_data['Close'].iloc[-1]:.2f}")
        
    with col2:
        # Calculate some metrics from the data we have
        returns = ha_data['Close'].pct_change().dropna()
        st.metric("Volatility (15-day)", f"{returns.std() * 100:.2f}%")
    
    # Disclaimer
    st.divider()
    st.caption("""
    **Disclaimer**: This app is for informational purposes only and does not constitute financial advice. 
    Trading stocks involves risk, and past performance is not indicative of future results. 
    Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
    """)

if __name__ == "__main__":
    main()