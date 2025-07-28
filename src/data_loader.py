import os
import pandas as pd
import yfinance as yf
import src.config as config


def download_and_preprocess_data():
    print(f"--- Downloading data for {config.TICKER_SYMBOL} ---")

    ticker = yf.Ticker(config.TICKER_SYMBOL)
    hist_data = ticker.history(period="5y", interval="1d")

    df = hist_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)

    print("--- Engineering Features ---")

    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()

    df['ROC'] = df['Close'].pct_change(periods=10) * 100

    print("--- Making Core Features Stationary ---")
    stationary_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in stationary_cols:
        df[col] = df[col].pct_change()

    df.dropna(inplace=True)

    processed_dir = os.path.dirname(config.PROCESSED_DATA_PATH)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")

    df.to_csv(config.PROCESSED_DATA_PATH)
    print(f"--- Processed data saved to {config.PROCESSED_DATA_PATH} ---")


if __name__ == '__main__':
    download_and_preprocess_data()
