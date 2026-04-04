import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="BTC予測・答え合わせボード", layout="wide")
st.title("📈 ビットコイン予測：答え合わせ ＆ 未来予測")

# --- サイドバーの設定 ---
st.sidebar.header("🛠️ ツール解説 & 使い方")

with st.sidebar:
    st.markdown("""
    ### 📈 グラフの読み方
    - **黒い実線**: 実際のビットコイン価格（実績）
    - **オレンジ点線**: 30日前のデータのみで予測した「答え合わせ用」の線
    - **太い青い線**: 最新データに基づく「これからの予測」
    
    ### 🔴🔵 テクニカル指標の境界線
    - **赤い点線 (RSI 70)**: 
        「買われすぎ」の目安。市場が過熱しており、反落に注意が必要なゾーンです。
    - **青い点線 (RSI 30)**: 
        「売られすぎ」の目安。価格が下がりすぎており、反発のチャンスを示唆します。
        
    ---
    ### 🧪 技術スタック
    - **AIモデル**: Prophet (Meta社製)
    - **データ元**: Yahoo Finance API
    - **UI**: Streamlit
    - **可視化**: Plotly (Interactive Chart)
    
    ### 💡 開発のポイント
    AIによる「未来予測」と、伝統的な「テクニカル分析」を組み合わせることで、多角的な判断ができるように設計しました。
    """)

# --- ここから下に既存のデータ取得やグラフ描画のコードを続ける ---

# --- 1. データ取得 ---
@st.cache_data
def get_data():
    # 2年分のデータを取得
    df = yf.download("BTC-JPY", period="2y", interval="1d")
    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    # 日付型を標準的なdatetimeに変換し、タイムゾーンを削除
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df

df = get_data()

# データの準備（テクニカル指標の計算用）
df_display = df.copy()

# 25日移動平均線 (SMA25)
df_display['SMA25'] = df_display['y'].rolling(window=25).mean()

# RSI (相対力指数) の計算
delta = df_display['y'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df_display['RSI'] = 100 - (100 / (1 + rs))

# --- 2. 予測ロジック ---
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

# --- 3. グラフ作成 (Plotly) ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# ① 実績値（黒い線）
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績価格", 
                         line=dict(color='black', width=2)),
              secondary_y=False)

# SMA25（緑の線）
fig.add_trace(go.Scatter(x=df_display['ds'], y=df_display['SMA25'], 
                         name="SMA25", line=dict(color='green', width=1)),
              secondary_y=False)

# RSI（紫の線・第2軸）
fig.add_trace(go.Scatter(x=df_display['ds'], y=df_display['RSI'], 
                         name="RSI", line=dict(color='purple', width=1)),
              secondary_y=True)

# RSIの境界線
fig.add_hline(y=70, line_dash="dot", line_color="red", secondary_y=True)
fig.add_hline(y=30, line_dash="dot", line_color="blue", secondary_y=True)

# --- 【重要】線を繋げるためのデータ結合処理 ---

# 1. 実績の最後の1点を抽出（青い線のスタート地点にする）
last_actual_point = df.iloc[[-1]][['ds', 'y']].copy()
last_actual_point.columns = ['ds', 'yhat'] # Prophetの出力名に合わせる

# 2. 未来予測の線を繋げる（今後の予測：青い線）
# 実績の最終日より「後」の予測データを取得
future_only = forecast_future[forecast_future['ds'] > df['ds'].max()]
# 実績の最後 + 未来予測 を結合
future_combined = pd.concat([last_actual_point, future_only], ignore_index=True)

# ③ 未来予測の描画
fig.add_trace(go.Scatter(x=future_combined['ds'], y=future_combined['yhat'], 
                         name="今後の予測", line=dict(color='blue', width=3)),
              secondary_y=False)

# 予測の不確実性（網掛け）も結合データを使用
fig.add_trace(go.Scatter(x=future_combined['ds'], y=forecast_future.loc[forecast_future['ds'] >= df['ds'].max(), 'yhat_upper'], 
                         line=dict(width=0), showlegend=False), secondary_y=False)
fig.add_trace(go.Scatter(x=future_combined['ds'], y=forecast_future.loc[forecast_future['ds'] >= df['ds'].max(), 'yhat_lower'], 
                         line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', 
                         showlegend=False), secondary_y=False)

# 3. 答え合わせの線を繋げる（オレンジの点線）
last_past_point = df_past.iloc[[-1]][['ds', 'y']].copy()
last_past_point.columns = ['ds', 'yhat']
bt_only = forecast_backtest[forecast_backtest['ds'] > df_past['ds'].max()]
bt_combined = pd.concat([last_past_point, bt_only], ignore_index=True)

# ② 答え合わせ予測の描画
fig.add_trace(go.Scatter(x=bt_combined['ds'], y=bt_combined['yhat'], 
                         name="30日前の予測（答え合わせ）", 
                         line=dict(color='orange', dash='dot')),
              secondary_y=False)

# レイアウト設定
fig.update_layout(title="ビットコイン価格推移と予測比較", xaxis_title="日付", 
                  hovermode="x unified", template="plotly_white")
fig.update_yaxes(title_text="価格 (JPY)", secondary_y=False)
fig.update_yaxes(title_text="RSI", range=[0, 100], secondary_y=True)

# 表示範囲を直近90日に設定
last_date = df['ds'].max()
fig.update_xaxes(range=[last_date - timedelta(days=90), last_date + timedelta(days=35)])

st.plotly_chart(fig, use_container_width=True)

# --- 4. 精度指標 ---
st.subheader("📊 予測精度の自己採点（直近30日間）")
actual_last_30 = df.iloc[-30:]['y'].values
pred_eval = forecast_backtest.iloc[-30:]['yhat'].values
mape = (abs(actual_last_30 - pred_eval) / actual_last_30).mean() * 100

col1, col2 = st.columns(2)
col1.metric("平均誤差率 (MAPE)", f"{mape:.2f} %")
col2.write("※一般的に5%以内なら高精度、10%以内なら良好とされます。")
with st.expander("🛠️ このサイトの技術的な構成について"):
    st.markdown("""
    ### 使用ライブラリ
    - **Prophet**: Meta社が開発した時系列予測モデル。トレンドの変化点を自動検知します。
    - **Streamlit**: データ分析アプリを迅速にWeb化するためのフレームワーク。
    - **Plotly**: ズームやホバーが可能な高機能グラフライブラリ。
    
    ### 仕組み
    1. `yfinance` で最新のBTC価格を取得。
    2. 過去2年間のデータを `Prophet` に学習させ、30日先を予測。
    3. 30日前の時点での予測値と実績値を比較し、誤差率（MAPE）を算出。
    """)