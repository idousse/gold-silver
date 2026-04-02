"""
Download historical gold and silver prices and save to CSV.
Run this script periodically (e.g., weekly) to update the data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def download_prices():
    """Download gold and silver futures prices and save to CSV."""
    
    print("Downloading gold futures (GC=F)...")
    gold = yf.download("GC=F", start="2000-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=True)
    
    print("\nDownloading silver futures (SI=F)...")
    silver = yf.download("SI=F", start="2000-01-01", end=datetime.now().strftime("%Y-%m-%d"), progress=True)
    
    # Extract close prices
    def get_close(df):
        if isinstance(df.columns, pd.MultiIndex):
            return df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        return df['Close']
    
    # Create combined dataframe
    prices = pd.DataFrame()
    prices['gold_price'] = get_close(gold)
    prices['silver_price'] = get_close(silver)
    prices = prices.dropna()
    
    # Calculate ratio
    prices['ratio'] = prices['gold_price'] / prices['silver_price']
    
    # Save to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, 'prices.csv')
    prices.to_csv(output_path)
    
    print(f"\n✅ Saved {len(prices)} rows to {output_path}")
    print(f"   Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return prices

if __name__ == "__main__":
    download_prices()
