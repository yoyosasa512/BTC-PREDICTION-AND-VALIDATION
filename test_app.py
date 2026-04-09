from app import calculate_mape, add_technical_indicators
import pandas as pd
import numpy as np

def test_calculate_mape_accuracy():
    # 誤差が計算通りになるか確認
    actual = [100, 200]
    forecast = [110, 180] # 10%誤差 と 10%誤差
    assert calculate_mape(actual, forecast) == 10.0

def test_add_technical_indicators_columns():
    # 必要な列が正しく追加されているか確認
    df = pd.DataFrame({'y': np.random.randn(50)})
    df_result = add_technical_indicators(df)
    assert 'SMA25' in df_result.columns
    assert 'RSI' in df_result.columns

def test_rsi_range():
    # RSIが論理的な範囲(0-100)に収まっているか確認
    df = pd.DataFrame({'y': np.random.randn(100)})
    df_result = add_technical_indicators(df)
    # NaNを除外してチェック
    rsi_values = df_result['RSI'].dropna()
    assert rsi_values.min() >= 0
    assert rsi_values.max() <= 100