import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 0. 計算ロジック・関数の定義 ---

def calculate_mape(actual, forecast):
    """平均絶対誤差率 (MAPE) を計算する関数"""
    actual, forecast = np.array(actual), np.array(forecast)
    # 実績値が0の場合の除算エラーを避ける処理
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100

def add_technical_indicators(df):
    """テクニカル指標（SMA25, RSI）を追加する関数"""
    df_result = df.copy()
    # 25日移動平均線 (SMA25)
    df_result['SMA25'] = df_result['y'].rolling(window=25).mean()
    # RSI (相対力指数)
    delta = df_result['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_result['RSI'] = 100 - (100 / (1 + rs))
    return df_result

@st.cache_data
def get_data():
    """データを取得し、Prophet形式に整える関数"""
    df = yf.download("BTC-JPY", period="2y", interval="1d")
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df

# --- 1. アプリの基本設定 ---

st.set_page_config(page_title="BTC予測・答え合わせボード", layout="wide")
st.title("📈 ビットコイン予測：答え合わせ ＆ 未来予測")

# --- 2. サイドバーの設定 ---
st.sidebar.header("🛠️ ツール解説 & 使い方")
with st.sidebar:
    st.markdown("""
    ### 📈 グラフの読み方
    - **黒い実線**: 実際のビットコイン価格（実績）
    - **オレンジ点線**: 30日前のデータのみで予測した「答え合わせ用」の線
    - **太い青い線**: 最新データに基づく「これからの予測」
    
    ### 🔴🔵 テクニカル指標の境界線
    - **赤い点線 (RSI 70)**: 「買われすぎ」の目安。
    - **青い点線 (RSI 30)**: 「売られすぎ」の目安。
    ---
    ### 🧪 技術スタック
    - **AIモデル**: Prophet (Meta社製)
    - **自動テスト**: pytest + GitHub Actions
    """)

# --- 3. メイン処理：データの準備 ---

df = get_data()
# テクニカル指標の計算（関数呼び出し）
df_display = add_technical_indicators(df)

# --- 4. 予測ロジック ---

# A. 「30日前」の視点での答え合わせ予測
df_past = df.iloc[:-30] 
model_backtest = Prophet(daily_seasonality=True)
model_backtest.fit(df_past)
future_backtest = model_backtest.make_future_dataframe(periods=30)
forecast_backtest = model_backtest.predict(future_backtest)

# B. 「現在」の視点での未来予測
model_future = Prophet(daily_seasonality=True)
model_future.fit(df)
future_real = model_future.make_future_dataframe(periods=30)
forecast_future = model_future.predict(future_real)

# --- 5. グラフ作成 (Plotly) ---

fig = make_subplots(specs=[[{"secondary_y": True}]])

# ① 実績値
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績価格", line=dict(color='black', width=2)), secondary_y=False)

# ② SMA25
fig.add_trace(go.Scatter(x=df_display['ds'], y=df_display['SMA25'], name="SMA25", line=dict(color='green', width=1)), secondary_y=False)

# ③ RSI
fig.add_trace(go.Scatter(x=df_display['ds'], y=df_display['RSI'], name="RSI", line=dict(color='purple', width=1)), secondary_y=True)
fig.add_hline(y=70, line_dash="dot", line_color="red", secondary_y=True)
fig.add_hline(y=30, line_dash="dot", line_color="blue", secondary_y=True)

# ④ 未来予測（線を繋げる処理）
last_actual_point = df.iloc[[-1]][['ds', 'y']].copy()
last_actual_point.columns = ['ds', 'yhat']
future_only = forecast_future[forecast_future['ds'] > df['ds'].max()]
future_combined = pd.concat([last_actual_point, future_only], ignore_index=True)

fig.add_trace(go.Scatter(x=future_combined['ds'], y=future_combined['yhat'], name="今後の予測", line=dict(color='blue', width=3)), secondary_y=False)

# ⑤ 答え合わせ予測（線を繋げる処理）
last_past_point = df_past.iloc[[-1]][['ds', 'y']].copy()
last_past_point.columns = ['ds', 'yhat']
bt_only = forecast_backtest[forecast_backtest['ds'] > df_past['ds'].max()]
bt_combined = pd.concat([last_past_point, bt_only], ignore_index=True)

fig.add_trace(go.Scatter(x=bt_combined['ds'], y=bt_combined['yhat'], name="30日前の予測（答え合わせ）", line=dict(color='orange', dash='dot')), secondary_y=False)

# レイアウト設定
fig.update_layout(title="ビットコイン価格推移と予測比較", xaxis_title="日付", hovermode="x unified", template="plotly_white")
fig.update_yaxes(title_text="価格 (JPY)", secondary_y=False)
fig.update_yaxes(title_text="RSI", range=[0, 100], secondary_y=True)
last_date = df['ds'].max()
fig.update_xaxes(range=[last_date - timedelta(days=90), last_date + timedelta(days=35)])

st.plotly_chart(fig, use_container_width=True)

# --- 6. 精度指標 ---

st.subheader("📊 予測精度の自己採点（直近30日間）")
actual_last_30 = df.iloc[-30:]['y'].values
pred_eval = forecast_backtest.iloc[-30:]['yhat'].values

# MAPEの計算（関数呼び出し）
mape = calculate_mape(actual_last_30, pred_eval)

col1, col2 = st.columns(2)
col1.metric("平均誤差率 (MAPE)", f"{mape:.2f} %")
col2.write("※一般的に5%以内なら高精度、10%以内なら良好とされます。")

