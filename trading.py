import yfinance as yf
import pandas as pd

def prepare_trading_data(ticker="RELIANCE.BO"):
    print(f"Fetching data for {ticker}...")
    
    df = yf.download(ticker, start="2021-01-01", end="2026-03-28", auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if df.empty:
        print("No data fetched. Check your internet connection or the ticker symbol.")
        return None
        
    print("Data fetched successfully! Calculating features...")

    df['Returns'] = df['Close'].pct_change()
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100

    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))


    df['Vol_20MA'] = df['Volume'].rolling(window=20).mean()
    df['High_Volume'] = (df['Volume'] > df['Vol_20MA']).astype(int)


    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    
    df.dropna(inplace=True)

    print("\n--- Final Dataset Preview ---")
    print(df[['Close', 'MA50', 'RSI', 'High_Volume', 'Target']].tail())

    filename = f"{ticker.replace('.', '_')}_ml_data.csv"
    df.to_csv(filename)
    print(f"\nSuccess! Prepared data saved to '{filename}'")
    
    return df

if __name__ == "__main__":
    final_data = prepare_trading_data("RELIANCE.BO")