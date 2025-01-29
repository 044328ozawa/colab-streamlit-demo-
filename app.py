import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
# from sqlalchemy import create_engine  # 実際にDWH接続する場合に使用

# タイトル部分（トップページ）
st.title("在庫管理・需要予測AIプロジェクト - データサイエンス デモ")

st.markdown("""
---
**プロジェクト概要**:  
- 病院内のカテーテル在庫管理を想定したAIモデルの運用イメージを凝縮したデモアプリです。  
- フェーズ3 (モデル開発・検証) & フェーズ4 (本番運用・MLOps) を簡単にデモし、ビッグデータ想定の在庫分析例やレポートまとめ画面を紹介します。

""")

# 3つのタブ: (1) フェーズ3 & 4 デモ, (2) ビッグデータ在庫分析, (3) ダッシュボード・レポートまとめ
tabs_main = st.tabs(["フェーズ3 & 4 デモ", "ビッグデータ在庫分析", "ダッシュボード・レポートまとめ"])

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
    - サービング & オーケストレーション (Airflow, Argo等)  
    - モデル監視・再学習サイクル (Concept Drift, 異常検知)  
    - ダッシュボード・アラート設計 (経営層向けKPI, 在庫担当向けリスク, etc.)
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
        st.warning(f"リスクスコア {threshold} 以上の在庫品：")
        st.write(high_risk_items)
    else:
        st.success("リスクが高い在庫は現在ありません。")

################################################################################
# タブ2: ビッグデータ在庫分析
################################################################################
with tabs_main[1]:
    st.header("ビッグデータ在庫分析デモ")

    st.markdown("""
    大規模な在庫ログを扱う場合を想定したサンプルです。  
    実際にはDWH(例: Redshift, BigQuery)やData Lake(S3)で集計し、Streamlitアプリは要約結果のみを取得します。
    """)

    start_date = st.date_input("開始日", value=pd.to_datetime("2025-01-01"))
    end_date = st.date_input("終了日", value=pd.to_datetime("2025-01-31"))

    st.write("※ 実際にはSQL発行 or API呼び出しで要約済みデータを取得。")

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

        st.subheader("日次使用量の推移 (カテーテルAのみ)")

        df_a = df_demo[df_demo["item"]=="カテーテルA"]
        st.line_chart(data=df_a, x='date', y='usage')

################################################################################
# タブ3: ダッシュボード・レポートまとめ
################################################################################
with tabs_main[2]:
    st.header("ダッシュボード・レポート - サマリー")

    st.markdown("""
    ここでは、実際の運用における**ダッシュボードやレポート例**をまとめてご紹介します。以下の機能が想定されます。

    1. 需要予測・在庫モニタリングダッシュボード
       - **(a) 週次／月次需要予測チャート**  
         過去実績と予測値を折れ線グラフで比較表示。  
         「予測 vs 実績」の乖離を視覚化し、精度の推移を把握。

       - **(b) カテーテル種類別 在庫状況サマリ**  
         カテーテル種類 / 現在庫数 / 推定消費日数(需要予測) / 安全在庫閾値 / 推奨発注数 / 廃棄リスク(High/Medium/Low)  
         ```plain
         A製品  100本  15日分  80本  なし        Low
         B製品   50本   5日分  80本  +30本発注  Medium
         C製品  200本  30日分  80本  在庫過多    High (期限近)
         ```
         - 推定消費日数: 需要予測モデルから、在庫がいつ尽きるか計算
         - 廃棄リスク: 使用頻度・期限などからスコアリング

       - **(c) 欠品リスク／廃棄リスク アラート**  
         カレンダーやゲージチャートで高リスク品をハイライト。  

    2. 減耗(廃棄)リスクレポート
       - **(a) 廃棄見込み本数・コスト予測**  
         今月 / 来月の見込み廃棄数＆推定コストをグラフ化  
         部署・医師別に原因ドリルダウン

       - **(b) 廃棄原因分析**  
         「滅菌期限切れ / 医師選好変更 / 緊急手術で別製品優先」など原因別件数を円グラフ・棒グラフで可視化  

    3. 改善施策シミュレーション画面
       - **(a) シナリオ比較**  
         「需要+10%」「リードタイム+2日」で欠品リスクや廃棄コストがどう変わるかを比較表 / グラフで提示  

       - **(b) 発注タイミング最適化**  
         安全在庫(Safety Stock)を変動させた際の在庫コスト / 欠品リスクのトレードオフを可視化  

    4. マネジメント向けハイレベルエグゼクティブレポート
       - **ROI / コスト削減レポート**  
         AI導入による廃棄コスト＆緊急調達コストの削減額を試算  
         前年比 / 前月比でインパクトを可視化

       - **経営指標ダッシュボード**  
         月次の削減率(%)、欠品率(%)、満足度などをKPIカード表示  
         経営層が全体を俯瞰しやすいレイアウト

    ---
    ここでは簡易的に「カテーテル種類別 在庫状況サマリ」をダミー表示します。
    """)

    ### デモ: 在庫状況サマリ (ダミー)
    st.subheader("在庫状況サマリ (ダミー)")
    # 適当なフラグデータ
    data_summary = {
        "種類": ["A製品","B製品","C製品"],
        "現在庫数": [100, 50, 200],
        "推定消費日数": ["15日分","5日分","30日分"],
        "安全在庫閾値": [80,80,80],
        "推奨発注数": ["なし","+30本","在庫過多"],
        "廃棄リスク": ["Low","Medium","High(期限近)"]
    }
    df_summary = pd.DataFrame(data_summary)
    st.dataframe(df_summary)

    st.markdown("""
    今後、このようなダッシュボードを**リアルタイム or バッチ更新**で自動生成し、
    欠品リスク・廃棄リスクを迅速に把握できる体制を作ることがゴールです。
    """)

st.markdown("---")
st.markdown("""
### ご利用ありがとうございました  
**© 2025 - In-Hospital AI Project**  
データサイエンスの最先端ソリューションデモでした。
""")
