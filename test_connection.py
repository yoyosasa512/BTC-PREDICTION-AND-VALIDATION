import pandas as pd
import yfinance as yf
from supabase import create_client, Client
import toml
import os
from datetime import datetime, timedelta

# Windowsコンソールの文字化け対策のため、絵文字を削除したテストスクリプト
def test():
    print("--- 1. secrets.toml read test ---")
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        supabase_url = secrets["connections"]["supabase"]["url"]
        supabase_key = secrets["connections"]["supabase"]["key"]
        print("[OK] secrets.toml read successful.")
    except Exception as e:
        print(f"[ERROR] secrets.toml read failed: {e}")
        return

    print("\n--- 2. Supabase connection test ---")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        # Check if table exists by counting rows
        res = supabase.table("btc_prices").select("count", count="exact").limit(1).execute()
        print(f"[OK] Supabase connection successful. Current row count: {res.count}")
    except Exception as e:
        print(f"[ERROR] Supabase connection or table reference failed: {e}")
        print("Please check if the table was created in the SQL Editor.")
        return

    print("\n--- 3. yfinance data fetch test ---")
    try:
        df = yf.download("BTC-JPY", period="1d", interval="1d", progress=False)
        if not df.empty:
            price = df['Close'].iloc[-1]
            # Handle possible MultiIndex or Series
            if hasattr(price, 'item'):
                price = price.item()
            print(f"[OK] yfinance fetch successful. Latest price: {price} JPY")
        else:
            print("[ERROR] yfinance returned empty data.")
    except Exception as e:
        print(f"[ERROR] yfinance fetch failed: {e}")

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    test()
