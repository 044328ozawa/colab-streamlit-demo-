import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
# from sqlalchemy import create_engine  # 実際にDWH接続する場合に使用

# タイトル部分（トップページ）
st.title("在庫管理・需要予測AIプロジェクト -データサイエンのデモ")

st.markdown("""
---
**プロジェクト概要**:  
- このアプリは、病院内のカテーテル在庫管理に関するAIモデルの運用イメージを凝縮したデモです。  
- フェーズ3 (モデル開発・検証) と フェーズ4 (本番運用・MLOps) のハイライトや、ビッグデータを想定した在庫分析例をまとめています。

""")

# 2つのメインタブ: フェーズ3 & 4 デモ  と  ビッグデータ在庫分析
tabs_main = st.tabs(["フェーズ3 & 4 デモ", "ビッグデータ在庫分析"])

################################################################################
# タブ1: フェーズ3 & 4 デモ
################################################################################
with tabs_main[0]:
    st.header("フェーズ3 & フェーズ4 - デモ画面")

    st.markdown("""
    **(A)** フェーズ3：モデル開発・検証  
    - データエクスプロレーション(EDA)  
    - 特徴量エンジニアリング  
    - モデル選定・実装 (Prophet, ARIMA, XGBoost, LightGBM etc.)  
    - ハイパーパラメータチューニング (交差検証, MLflow など)  
    - モデル評価 (RMSE, MAPE, Precision/Recall)  
    - パイロット運用・A/Bテスト  

    **(B)** フェーズ4：本番運用・MLOps確立  
    - モデルデプロイ (バッチ推論 / リアルタイムAPI / コンテナ化)  
    - サービング & オーケストレーション (Airflow, Argo, etc.)  
    - モデル監視・再学習サイクル (Concept Drift, 異常検知)  
    - ダッシュボード・アラート設計 (経営層向けKPI, 在庫担当向けリスク, リアルタイム手術室画面)
    """)

    st.subheader("デモ1: 需要予測サンプル")

    # ダミーデータ: 日付 + 需要(予測値)
    n_days = 30
    date_rng = pd.date_range(start="2025-01-01", periods=n_days, freq="D")
    demand = np.random.randint(50, 120, size=n_days)

    df_demand = pd.DataFrame({
        "date": date_rng,
        "predicted_demand": demand
    })

    limit_days = st.slider("表示する期間（日数）", 7, n_days, 14, key="slider_phase3")
    df_demand_limited = df_demand.head(limit_days)

    chart_demand = alt.Chart(df_demand_limited).mark_line().encode(
        x='date:T',
        y='predicted_demand:Q'
    ).properties(
        width=600,
        height=300,
        title="予測需要 (サンプル)"
    )
    st.altair_chart(chart_demand, use_container_width=True)

    st.subheader("デモ2: 在庫リスク モニタリング")

    item_names = ["カテーテルA", "カテーテルB", "カテーテルC", "カテーテルD"]
    stock_data = {
        "item": item_names,
        "current_stock": [100, 60, 30, 20],
        "risk_score": [0.1, 0.4, 0.7, 0.85]  # 0に近いほど低リスク
    }
    df_stock = pd.DataFrame(stock_data)

    st.dataframe(df_stock)

    threshold = st.slider("リスク閾値(0~1)", 0.0, 1.0, 0.5, key="slider_phase4")
    high_risk_items = df_stock[df_stock["risk_score"] >= threshold]

    if len(high_risk_items) > 0:
        st.warning(f"リスクスコアが {threshold} 以上の在庫品：")
        st.write(high_risk_items)
    else:
        st.success("リスクが高い在庫は現在ありません。")

################################################################################
# タブ2: ビッグデータ在庫分析
################################################################################
with tabs_main[1]:
    st.header("ビッグデータ在庫分析デモ")

    st.markdown("""
    こちらでは、大規模な在庫ログを扱う場合を想定したサンプルです。  
    本来はDWH(例: Redshift, BigQuery等)やData Lake(S3等)で集計し、Streamlitアプリは**要約結果のみ取得**します。  
    """)

    # サイドまたは画面上部で日付選択
    start_date = st.date_input("開始日", value=pd.to_datetime("2025-01-01"))
    end_date = st.date_input("終了日", value=pd.to_datetime("2025-01-31"))

    st.write("※ 実際にはSQL発行 or API呼び出しをして必要データのみ読み込みます。")

    if st.button("集計を実行"):
        # デモ用ダミーデータ
        date_range = pd.date_range(start_date, end_date)
        df_demo = pd.DataFrame({
            "date": np.repeat(date_range, 3),
            "item": ["カテーテルA","カテーテルB","カテーテルC"] * len(date_range),
            "usage": np.random.randint(50, 150, size=len(date_range)*3)
        })

        st.markdown("#### 集計結果 (ダミー)")
        st.dataframe(df_demo)

        st.subheader("日次使用量の推移 (カテーテルA)")

        df_a = df_demo[df_demo["item"]=="カテーテルA"]
        st.line_chart(data=df_a, x='date', y='usage')

st.markdown("---")
st.markdown("""
### ご利用ありがとうございました  
**© 2025 - In-Hospital AI Project**  
先端ソリューションデモでした。
""")
