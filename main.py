import ccxt
import schedule
import time
import requests
import pandas as pd
import ta
import numpy as np
from typing import List, Optional

# Replace these with your actual API Keys
binance_api_key = 'Q7EmDaDslId3SjDP8xpInWY9pqidecf58vScj5PxmNywtxbCW8JUZEJYUOW6TPhX'
binance_api_secret = 'WWRPXAOWwc05hDQJ2k4ydQ8qOELOInWen6R8RdeGRp20Ta0NVCucc7zN2RGKWJet'

# Binance exchange instance
binance_exchange = ccxt.binance({
    'Q7EmDaDslId3SjDP8xpInWY9pqidecf58vScj5PxmNywtxbCW8JUZEJYUOW6TPhX': binance_api_key,
    'WWRPXAOWwc05hDQJ2k4ydQ8qOELOInWen6R8RdeGRp20Ta0NVCucc7zN2RGKWJet': binance_api_secret,
})

# Telegram bot token and channel ID
telegram_bot_token = '6753309076:AAFohHvtTFGIJ5Yt3j7e0Sj_ltS5hHyEv_E'
telegram_channel_id = '-1002068442484'  # Replace with your actual channel username

def send_telegram_message(message):
    try:
        telegram_url = f'https://api.telegram.org/bot{telegram_bot_token}/sendMessage'
        params = {'chat_id': telegram_channel_id, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(telegram_url, params=params)
        response.raise_for_status()
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# Add Support and Resistance Logic with Volume Analysis
def add_support_resistance_with_volume(df, timeframe):
    """
    Calculate professional support and resistance levels dynamically with volume analysis.
    :param df: DataFrame with OHLCV data.
    :param timeframe: Timeframe of the data (e.g., '15m', '1h', '4h', '1d').
    :return: DataFrame with support/resistance status and volume profiles.
    """
    # Dynamic window size based on timeframe
    if timeframe == '15m':
        window = 3
    elif timeframe == '1h':
        window = 5
    elif timeframe == '4h':
        window = 20
    elif timeframe == '1d':
        window = 50
    else:
        window = 10  # Default

    df['support'] = np.nan
    df['resistance'] = np.nan
    df['support_resistance'] = "Natural"  # Default state

    # Detect Pivot Points
    df['pivot_high'] = df['high'][
        (df['high'].shift(window) < df['high']) &
        (df['high'].shift(-window) < df['high']) &
        (df['high'] == df['high'].rolling(2 * window + 1, center=True).max())
    ]
    
    df['pivot_low'] = df['low'][
        (df['low'].shift(window) > df['low']) &
        (df['low'].shift(-window) > df['low']) &
        (df['low'] == df['low'].rolling(2 * window + 1, center=True).min())
    ]

    # Track Last Valid Pivots
    df['last_pivot_high'] = df['pivot_high'].ffill()
    df['last_pivot_low'] = df['pivot_low'].ffill()

    # Volume Profile Calculation
    volume_at_price = df.groupby('close')['volume'].sum()  # Aggregate volume at each price level
    volume_profile = volume_at_price.sort_values(ascending=False)  # Sort by highest volume
    top_volumes = volume_profile.head(5)  # Top 5 price levels with highest volume (POC)

    # Assign Volume-Based Support and Resistance
    df['volume_support'] = top_volumes.index.min()  # Lowest high-volume level
    df['volume_resistance'] = top_volumes.index.max()  # Highest high-volume level

    # Calculate Support and Resistance Levels
    for i in range(len(df)):
        if i < window:
            continue  # Skip early candles where pivots can't be detected
            
        c_close = df['close'].iloc[i]
        c_high = df['high'].iloc[i]
        c_low = df['low'].iloc[i]
        
        last_ph = df['last_pivot_high'].iloc[i]
        last_pl = df['last_pivot_low'].iloc[i]
        vol_support = df['volume_support'].iloc[i]
        vol_resistance = df['volume_resistance'].iloc[i]

        # Update Support and Resistance Levels
        if pd.notna(last_ph):
            df.at[df.index[i], 'resistance'] = last_ph
        if pd.notna(last_pl):
            df.at[df.index[i], 'support'] = last_pl

        # Detect Reversals and Breakouts
        if c_close > last_ph or c_close > vol_resistance:
            df.at[df.index[i], 'support_resistance'] = "/Strong upward breakout"
        elif c_close < last_pl or c_close < vol_support:
            df.at[df.index[i], 'support_resistance'] = "/Strong downward breakout"
        elif (c_high >= last_ph or c_high >= vol_resistance) and (c_close <= last_ph or c_close <= vol_resistance):
            df.at[df.index[i], 'support_resistance'] = "/Reversal from resistance"
        elif (c_low <= last_pl or c_low <= vol_support) and (c_close >= last_pl or c_close >= vol_support):
            df.at[df.index[i], 'support_resistance'] = "/Reversal from support"
        else:
            df.at[df.index[i], 'support_resistance'] = "Natural"

    # Cleanup Temporary Columns
    df.drop(columns=['pivot_high', 'pivot_low', 'last_pivot_high', 'last_pivot_low'], inplace=True)
    return df

# Calculate Indicators
def calculate_indicators(df):
    """
    Calculate technical indicators for the given DataFrame.
    :param df: Input DataFrame containing OHLCV data.
    :return: DataFrame with calculated indicators.
    """
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd_line'] = macd.macd()
    df['signal_line'] = macd.macd_signal()
    df['histogram'] = df['macd_line'] - df['signal_line']

    # OBV
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # EMA 10, EMA 50, SMA 9, SMA 20, SMA 50, SMA 100
    df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['ema55'] = ta.trend.ema_indicator(df['close'], window=55)
    df['sma100'] = ta.trend.sma_indicator(df['close'], window=100)

    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # OBV Levels (Fixed High and Low Thresholds)
    obv_mean = df['obv'].mean()
    obv_std = df['obv'].std()
    df['obv_level'] = np.select(
        [
            df['obv'] > obv_mean + 2 * obv_std,  # Overbought Zone (High Level)
            df['obv'] < obv_mean - 2 * obv_std   # Oversold Zone (Low Level)
        ],
        ["High", "Low"],  # Labels for Overbought and Oversold Zones
        default="🔘"  # Default for Neutral Zone
    )

    # ADX (New Addition)
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['adx_plus_di'] = adx_indicator.adx_pos()
    df['adx_minus_di'] = adx_indicator.adx_neg()

    # Market State Based on SMA Conditions
    df['market_state'] = np.select(
        [
            (df['close'] > df['ema9']) & 
            (df['ema9'] > df['ema20']) & 
            (df['ema20'] > df['ema55']) & 
            (df['ema55'] > df['sma100']),  # Bullish Condition (🟢)
            (df['close'] < df['ema9']) & 
            (df['ema9'] < df['ema20']) & 
            (df['ema20'] < df['ema55']) & 
            (df['ema55'] < df['sma100'])   # Bearish Condition (🔴)
        ],
        ["🟢", "🔴 "],  # Labels for Bullish and Bearish States
        default="🔘 "  # Default for Neutral/Divergent State
    )

    # ADX State
    df['adx_state'] = np.select(
        [
            (df['adx'] > 23) & (df['adx_plus_di'] > df['adx_minus_di']),  # Strong Uptrend
            (df['adx'] > 23) & (df['adx_plus_di'] < df['adx_minus_di']),  # Strong Downtrend
            (df['adx'] <= 23)  # Weak Trend
        ],
        ["💪🟢 Strong", "💪🔴 Strong", "🔘Weak "],  # Labels for ADX States
        default="🔘 Undefined"  # Default for Undefined State
    )

    # RSI Status
    df['rsi_status'] = np.select(
        [
            df['rsi'] > 80,  # Overbought
            df['rsi'] < 30   # Oversold
        ],
        ["🧲", "🚩"],  # Labels for Overbought and Oversold
        default="🔘"  # Default for Neutral
    )
    return df

# Apply Strategy
def apply_strategy(df):
    """
    Apply the buy/sell strategy based on MACD, OBV, EMA, and ATR.
    :param df: DataFrame with calculated indicators.
    :return: DataFrame with buy/sell signals.
    """
    # Buy Signal (Last Fully Closed Candle)
    df['buy_signal'] = (
        # MACD crossing above the signal line with a significant gap and increasing histogram
        (df['macd_line'].shift(1) < df['signal_line'].shift(1)) &  # Previous candle MACD line < Signal line
        (df['macd_line'] > df['signal_line']) &                    # Current candle MACD line > Signal line
        ((df['macd_line'] - df['signal_line']).abs() > 0.0005) &   # Significant gap between MACD and Signal line
        (df['histogram'] > df['histogram'].shift(1)) &             # Histogram increasing (big gap up)
        (df['histogram'] > 0) &                                    # Positive histogram
        # OBV rising from below the oversold area in the last 3 candles
        (df['obv'] > df['obv'].shift(1)) &                         # OBV increasing
        (df['obv'].shift(1) > df['obv'].shift(2)) &                # OBV big gap up
        (df['obv'].shift(2) <= df['obv'].shift(3)) &               # OBV was previously in an oversold area
        # Price above EMA 10 and EMA 50
        (df['close'] > df['ema10']) &                              # Price above EMA 10
        (df['close'] > df['ema50'])                                # Price above EMA 50
    )

    # Sell Signal (Last Fully Closed Candle)
    df['sell_signal'] = (
        # MACD crossing below the signal line with a significant gap and decreasing histogram
        (df['macd_line'].shift(1) > df['signal_line'].shift(1)) &  # Previous candle MACD line > Signal line
        (df['macd_line'] < df['signal_line']) &                    # Current candle MACD line < Signal line
        ((df['macd_line'] - df['signal_line']).abs() > 0.0005) &   # Significant gap between MACD and Signal line
        (df['histogram'] < df['histogram'].shift(1)) &             # Histogram decreasing (big gap down)
        (df['histogram'] < 0) &                                    # Negative histogram
        # OBV falling from above the overbought area in the last 3 candles
        (df['obv'] < df['obv'].shift(1)) &                         # OBV decreasing
        (df['obv'].shift(1) < df['obv'].shift(2)) &                # OBV big gap down
        (df['obv'].shift(2) >= df['obv'].shift(3)) &               # OBV was previously in an overbought area
        # Price below EMA 10 and EMA 50
        (df['close'] < df['ema10']) &                              # Price below EMA 10
        (df['close'] < df['ema50'])                                # Price below EMA 50
    )

    # Entry, Targets, and Stop Loss using ATR
    df['entry'] = df['close']
    df['target1_buy'] = df['close'] + df['atr'] * 1
    df['target2_buy'] = df['close'] + df['atr'] * 2
    df['target3_buy'] = df['close'] + df['atr'] * 3
    df['stop_loss_buy'] = df['close'] - df['atr'] * 1.5
    df['target1_sell'] = df['close'] - df['atr'] * 1
    df['target2_sell'] = df['close'] - df['atr'] * 2
    df['target3_sell'] = df['close'] - df['atr'] * 3
    df['stop_loss_sell'] = df['close'] + df['atr'] * 1.5
    return df

# Fetch Data and Check Symbols
def fetch_and_check_all_symbols(exchange, excluded_symbols=[]):
    """
    Fetch market data for all USDT symbols, calculate indicators, and send alerts.
    :param exchange: CCXT exchange instance.
    :param excluded_symbols: List of symbols to exclude.
    """
    try:
        # Load all markets and filter for spot and futures markets
        markets = exchange.load_markets()
        symbols = [symbol for symbol in markets.keys() if markets[symbol].get('spot', False) or markets[symbol].get('future', False)]
        usdt_symbols = [
            symbol for symbol in symbols
            if symbol.endswith('/USDT') 
            and symbol not in excluded_symbols
            and not symbol.split('/')[0].endswith(('UP', 'DOWN', 'BEAR', 'BULL'))  # Skip symbols ending with UP, DOWN, BEAR, BULL
        ]
        timeframes = ['15m', '1h', '4h', '1d']  # Multi-timeframe analysis

        for symbol in usdt_symbols:
            try:
                all_timeframe_data = {}
                skip_symbol = False  # Flag to skip the symbol if data is insufficient

                for timeframe in timeframes:
                    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)

                    # Check if data is sufficient
                    if len(ohlcv_data) < 50:  # Minimum required candles
                        print(f"⚠️ Skipping {symbol} on {timeframe.upper()} due to insufficient data.")
                        skip_symbol = True
                        break  # Skip this symbol entirely

                    # Convert data to DataFrame
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                    # Calculate indicators
                    df = calculate_indicators(df)
                    df = add_support_resistance_with_volume(df, timeframe)  # Add professional support/resistance logic with volume
                    df = apply_strategy(df)

                    all_timeframe_data[timeframe] = df

                # If data is insufficient, skip to the next symbol
                if skip_symbol:
                    continue

                # Generate Market State Summary Across Timeframes
                market_state_summary = ""
                adx_state_summary = ""  # New summary for ADX states
                rsi_status_summary = ""  # New summary for RSI status
                obv_status_summary = ""  # New summary for OBV status
                support_resistance_summary = ""  # New summary for support/resistance status

                for timeframe, df in all_timeframe_data.items():
                    market_state_summary += f"{timeframe.upper()}: `{df['market_state'].iloc[-2]}` | "
                    adx_state_summary += f"{timeframe.upper()}: `{df['adx_state'].iloc[-2]}` | "
                    rsi_status_summary += f"{timeframe.upper()}: `{df['rsi_status'].iloc[-2]}` | "
                    obv_status_summary += f"{timeframe.upper()}: `{df['obv_level'].iloc[-2]}` | "
                    support_resistance_summary += f"{timeframe.upper()}: `{df['support_resistance'].iloc[-2]}` | "

                market_state_summary = market_state_summary.rstrip(" | ")
                adx_state_summary = adx_state_summary.rstrip("-")
                rsi_status_summary = rsi_status_summary.rstrip("-")
                obv_status_summary = obv_status_summary.rstrip("-")
                support_resistance_summary = support_resistance_summary.rstrip("-")

                # Check for Buy/Sell Signals
                alert_message = ""
                for timeframe, df in all_timeframe_data.items():
                    if df['buy_signal'].iloc[-1]:
                        alert_message += (
                            f"----------------------------------\n"
                            f"🚀 *BUY SIGNAL!*🟩 🟢\n"
                            f"*Symbol:📍* `{symbol}`\n"
                            f"*Timeframe:* `{timeframe.upper()}`\n"
                            f"*Market State:* `{df['market_state'].iloc[-2]}`\n"
                            f"*ADX :* `{df['adx_state'].iloc[-2]}`\n"
                            f"*RSI :* `{df['rsi_status'].iloc[-2]}`\n"
                            f"*OBV :* `{df['obv_level'].iloc[-2]}`\n"
                            f"*Support/Resistance:* `{df['support_resistance'].iloc[-2]}`\n"
                            f"*Volume Support:* `{df['volume_support'].iloc[-1]:.4f}`\n"
                            f"*Volume Resistance:* `{df['volume_resistance'].iloc[-1]:.4f}`\n"
                            f"*Entry Price:* `{df['entry'].iloc[-1]:.4f}`\n"
                            f"*Target 1:🎯* `{df['target1_buy'].iloc[-1]:.4f}`\n"
                            f"*Target 2:🎯* `{df['target2_buy'].iloc[-1]:.4f}`\n"
                            f"*Target 3:🎯* `{df['target3_buy'].iloc[-1]:.4f}`\n"
                            f"*Stop Loss:🛡* `{df['stop_loss_buy'].iloc[-1]:.4f}`\n"
                        )
                    elif df['sell_signal'].iloc[-1]:
                        alert_message += (
                            f"----------------------------------\n"
                            f"🚨 *SELL SIGNAL!*🟥 🔴\n"
                            f"*Symbol:📍* `{symbol}`\n"
                            f"*Timeframe:* `{timeframe.upper()}`\n"
                            f"*Market State:* `{df['market_state'].iloc[-2]}`\n"
                            f"*ADX :* `{df['adx_state'].iloc[-2]}`\n"
                            f"*RSI :* `{df['rsi_status'].iloc[-2]}`\n"
                            f"*OBV :* `{df['obv_level'].iloc[-2]}`\n"
                            f"*Support/Resistance:* `{df['support_resistance'].iloc[-2]}`\n"
                            f"*Volume Support:* `{df['volume_support'].iloc[-1]:.4f}`\n"
                            f"*Volume Resistance:* `{df['volume_resistance'].iloc[-1]:.4f}`\n"
                            f"*Entry Price:* `{df['entry'].iloc[-1]:.4f}`\n"
                            f"*Target 1:🎯* `{df['target1_sell'].iloc[-1]:.4f}`\n"
                            f"*Target 2:🎯* `{df['target2_sell'].iloc[-1]:.4f}`\n"
                            f"*Target 3:🎯* `{df['target3_sell'].iloc[-1]:.4f}`\n"
                            f"*Stop Loss:🛡* `{df['stop_loss_sell'].iloc[-1]:.4f}`\n"
                        )

                # Add Market State Summary, ADX State Summary, RSI Status Summary, OBV Status Summary, and Support/Resistance Summary to the Alert Message
                if alert_message:
                    alert_message = (
                        f"📊 *Market State:* {market_state_summary}\n"
                        f"📈 *ADX :* {adx_state_summary}\n"
                        f"----------------------------------\n"
                        f"🔰 *RSI :* {rsi_status_summary}\n"
                        f"----------------------------------\n"
                        f"🌊 *OBV Level:* {obv_status_summary}\n"
                        f"----------------------------------\n"
                        f"⛰️ *Support/Resistance:* {support_resistance_summary}\n"
                        + alert_message
                    )
                    send_telegram_message(alert_message)

            except Exception as e:
                print(f"Error fetching data for symbol {symbol}: {e}")
    except Exception as e:
        print(f"Error fetching data: {e}")

# Welcome message
welcome_message = (
    "👋 SCALPKING WHALE PUMP 🚀\n"
    "⚠️ Risks involved.\n"
    "This bot is programmed to catch whales while buying owls it hunts the currency immediately.\n"
    "You should know very well that such owls can lead to loss of your capital.\n"
    "⚠️ You must commit to stop loss to avoid price fluctuations.\n"
    "..Enjoy😊 .\n"
    "BOT RUNNING...⏳"
)
send_telegram_message(welcome_message)

# Schedule tasks for Binance only, excluding MULTI/USDT by default
excluded_symbols = [
    'MULTI/USDT', 'XEM/USDT','PROM/USDT' ,'BSV/USDT', 'STORM/USDT', '1000STAR/USDT', 'BTCST/USDT:USDT', 'MFT/USDT', 'MC/USDT','KEY/USDT','TOMO/USDT','TCT/USDT','VITE/USDT','MDT/USDT','AUD/USDT',
    'ATOM/USDT', 'PERL/USDT', 'STMX/USDT', 'BTT/USDT', 'XMR/USDT', 'NANO/USDT', 'MITH/USDT', 'MATIC/USDT', 'MDX/USDT','YFI/USDT','MIR/USDT','AUTO/USDT','POLS/USDT',
    'VIDT/USDT', 'BTCST/USDT', 'PAX/USDT', 'BTG/USDT', 'USDC/USDT', 'BCHABC/USDT', 'WAVES/USDT', 'PNT/USDT', 'MEME/USDT',
    'TVK/USDT', 'LEND/USDT', 'VEN/USDT', 'HNT/USDT', 'HC/USDT', 'USDSB/USDT', 'GTO/USDT', 'AEVOUSDT', 'ETHDOWNUSDT',
    'POLSUSDT', 'MDXUSDT', 'DOCKUSDT', 'BNBBULLUSDT', 'MULTIUSDT', 'BNBBEARUSDT', 'ETHBULLUSDT', 'XRPBEARUSDT', 'BCHSVUSDT','CLV/USDT',
    'TOMOUSDT', 'LTOUSDT', 'PNTUSDT', 'BCCUSDT', 'SUSHIUPUSDT', 'TVKUSDT', 'MOBUSDT', 'LINKDOWNUSDT', 'ADADOWNUSDT',
    'BTCDOWNUSDT', 'ETHUPUSDT', 'ERD/USDT', 'XTZDOWNUSDT', 'EOSUPUSDT', 'TRXDOWNUSDT', 'SKYUSDT', 'AVXUSDT', 'COTIUSDT',
    'NBSUSDT', 'MTAUSDT', 'MDAUSDT','WING/USDT' ,'AUTOUSDT', 'USTCUSDT', 'DOTDOWNUSDT', 'REPUSDT', 'DREPUSDT', 'XEMUSDT', 'WAVESUSDT'
]

# Schedule the task to run every 15 minutes, aligned with candle close times
schedule.every().hour.at(":00").do(fetch_and_check_all_symbols, exchange=binance_exchange, excluded_symbols=excluded_symbols)
schedule.every().hour.at(":15").do(fetch_and_check_all_symbols, exchange=binance_exchange, excluded_symbols=excluded_symbols)
schedule.every().hour.at(":30").do(fetch_and_check_all_symbols, exchange=binance_exchange, excluded_symbols=excluded_symbols)
schedule.every().hour.at(":45").do(fetch_and_check_all_symbols, exchange=binance_exchange, excluded_symbols=excluded_symbols)

while True:
    schedule.run_pending()
    time.sleep(300)