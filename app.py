import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC予測・答え合わせボード", layout="wide")
st.title("📈 ビットコイン予測：答え合わせ ＆ 未来予測")

# --- 1. データ取得 ---
@st.cache_data
def get_data():
    df = yf.download("BTC-JPY", period="2y", interval="1d")
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

df = get_data()

# --- 2. 予測ロジック ---
# A. 「30日前」の視点での答え合わせ予測
df_past = df.iloc[:-30] # 直近30日を除外して学習
model_backtest = Prophet(daily_seasonality=True)
model_backtest.fit(df_past)
future_backtest = model_backtest.make_future_dataframe(periods=30)
forecast_backtest = model_backtest.predict(future_backtest)

# B. 「現在」の視点での未来予測
model_future = Prophet(daily_seasonality=True)
model_future.fit(df)
future_real = model_future.make_future_dataframe(periods=30)
forecast_future = model_future.predict(future_real)

# --- 3. グラフ作成 (Plotly) ---
fig = go.Figure()

# ① 実績値（実際の価格）
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績価格", line=dict(color='black', width=2)))

# ② 答え合わせ（30日前の予測結果）
# 直近30日分のみ抽出
bt_result = forecast_backtest.iloc[-30:]
fig.add_trace(go.Scatter(x=bt_result['ds'], y=bt_result['yhat'], 
                         name="30日前の予測（答え合わせ用）", 
                         line=dict(color='orange', dash='dot')))

# ③ 未来予測（今日から30日後）
future_result = forecast_future.iloc[-30:]
fig.add_trace(go.Scatter(x=future_result['ds'], y=future_result['yhat'], 
                         name="今後の予測", 
                         line=dict(color='blue', width=3)))

# 予測の幅（薄い青色）
fig.add_trace(go.Scatter(x=future_result['ds'], y=future_result['yhat_upper'], 
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_result['ds'], y=future_result['yhat_lower'], 
                         line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', 
                         showlegend=False))

fig.update_layout(title="ビットコイン価格推移と予測比較", xaxis_title="日付", yaxis_title="価格 (JPY)",
                  hovermode="x unified", template="plotly_white")

# 表示範囲を直近3ヶ月に絞る（初期状態）
last_date = df['ds'].max()
fig.update_xaxes(range=[last_date - timedelta(days=90), last_date + timedelta(days=30)])

st.plotly_chart(fig, use_container_width=True)

# --- 4. 精度指標の表示 ---
st.subheader("📊 予測精度の自己採点（直近30日間）")
# 簡易的な誤差計算
actual_last_30 = df.iloc[-30:]['y'].values
pred_last_30 = bt_result['yhat'].values
mape = (abs(actual_last_30 - pred_last_30) / actual_last_30).mean() * 100

col1, col2 = st.columns(2)
col1.metric("平均誤差率 (MAPE)", f"{mape:.2f} %")
col2.write("※一般的に5%以内なら高精度、10%以内なら良好とされます。")