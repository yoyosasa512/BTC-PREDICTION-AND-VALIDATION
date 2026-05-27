import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from supabase import create_client

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

# Supabase接続の初期化（標準的な接続方法）
@st.cache_resource
def init_supabase():
    try:
        # secrets.toml の [connections.supabase] セクションから取得
        url = st.secrets["connections"]["supabase"]["url"]
        key = st.secrets["connections"]["supabase"]["key"]
    except Exception:
        # またはトップレベルの SUPABASE_URL / SUPABASE_KEY から取得
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    
    if not url or not key:
        st.error("❌ Supabaseの接続設定が見つかりません。")
        st.stop()
    return create_client(url, key)

supabase = init_supabase()

@st.cache_data(ttl=3600)
def get_data():
    """データをDBから取得し、未取得分をyfinanceで補完してDBに保存する関数"""
    # 1. Supabaseから既存データを取得
    try:
        res = supabase.table("btc_prices").select("*").execute()
        df_db = pd.DataFrame(res.data)
        if not df_db.empty:
            df_db['ds'] = pd.to_datetime(df_db['ds'])
            latest_date_in_db = df_db['ds'].max()
        else:
            latest_date_in_db = None
    except Exception as e:
        # 初回実行時などでテーブルがない場合
        st.warning(f"DB接続エラー（初回はテーブル作成が必要です）: {e}")
        df_db = pd.DataFrame()
        latest_date_in_db = None

    # 2. yfinanceから差分（または全件）を取得
    try:
        if latest_date_in_db is None:
            # DBが空の場合は3年分取得
            df_new = yf.download("BTC-JPY", period="3y", interval="1d", progress=False)
            st.info("初回データ取得を実行しました（3年分）。")
        else:
            # DBにある最新日の翌日から今日までを取得
            start_date = (latest_date_in_db + timedelta(days=1)).strftime('%Y-%m-%d')
            # yf.downloadに本日までの期間を指定
            df_new = yf.download("BTC-JPY", start=start_date, interval="1d", progress=False)

        if not df_new.empty:
            # データ整形
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            df_new = df_new.reset_index()
            date_col = 'Date' if 'Date' in df_new.columns else df_new.columns[0]
            df_new = df_new[[date_col, 'Close']]
            df_new.columns = ['ds', 'y']
            
            # もしアクセス制限などでデータが空だった場合は処理を停止し、警告を出す
            if df_new.empty:
                st.error("⚠️ Yahoo Financeのアクセス制限（Rate Limit）に達したため、データを取得できませんでした。数分〜1時間ほど時間をおいてから再度お試しください。")
                st.stop()
            
            df_new['ds'] = pd.to_datetime(df_new['ds']).dt.tz_localize(None)
            df_new['y'] = pd.to_numeric(df_new['y'], errors='coerce')
            df_new = df_new.dropna(subset=['y'])

            # 3. 新規データをSupabaseに保存 (Upsert)
            if not df_new.empty:
                upsert_data = df_new.copy()
                upsert_data['ds'] = upsert_data['ds'].dt.strftime('%Y-%m-%d')
                records = upsert_data.to_dict(orient='records')
                supabase.table("btc_prices").upsert(records).execute()
                
                # DBデータと結合
                df_result = pd.concat([df_db, df_new]).drop_duplicates(subset=['ds'])
            else:
                df_result = df_db
        else:
            df_result = df_db

        return df_result.sort_values('ds').reset_index(drop=True)

    except Exception as e:
        st.error(f"データ取得・保存中にエラーが発生しました: {e}")
        return df_db if not df_db.empty else pd.DataFrame()

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
    
    > **技術解説**: テクニカル指標（SMA/RSI）の計算に伴う初期データの欠損を補完するため、バックエンドでは3年分のデータを取得し、計算後の有効データのみを予測モデルに投入するパイプラインを構築しています。
    ---
    """)

# --- 3. メイン処理：データの準備 ---

df = get_data()

# 取得データのバリデーション
if df.empty:
    st.error("📉 データの取得に失敗しました。Yahoo Financeの接続制限か、一時的なネットワークエラーの可能性があります。しばらく時間を置いてから再度リロードしてみてください。")
    st.stop()

# バックテスト（過去30日分除外）に十分なデータがあるか確認
if len(df) < 35: # 2行以上の学習データ + 30日のバックテスト + 余裕
    st.warning(f"⚠️ データ件数が不足しています（現在 {len(df)} 件）。正常な予測には少なくとも35日分以上の過去データが必要です。")
    st.stop()

# テクニカル指標の計算
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

st.plotly_chart(fig, width="stretch")

# --- 6. 精度指標 ---

st.subheader("📊 予測精度の自己採点（直近30日間）")
actual_last_30 = df.iloc[-30:]['y'].values
pred_eval = forecast_backtest.iloc[-30:]['yhat'].values

# MAPEの計算（関数呼び出し）
mape = calculate_mape(actual_last_30, pred_eval)

col1, col2 = st.columns(2)
col1.metric("平均誤差率 (MAPE)", f"{mape:.2f} %")
col2.write("※一般的に5%以内なら高精度、10%以内なら良好とされます。")

