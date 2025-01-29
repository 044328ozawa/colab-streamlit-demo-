import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("在庫管理AIプロジェクト - フェーズ3 & 4 デモ")

"""
このデモ画面では、フェーズ3（モデル開発・検証）とフェーズ4（本番運用・MLOps確立）の流れを
簡単にイメージできるように、最低限のインタラクションや可視化を用意しています。
"""

# タブ（またはセクション）でPhase3, Phase4を切り替え
tabs = st.tabs(["フェーズ3：モデル開発・検証", "フェーズ4：本番運用・MLOps確立"])

with tabs[0]:
    st.header("フェーズ3: モデル開発・検証")

    st.markdown("""
    **1. データエクスプロレーション(EDA)**  
    - データ基盤が整備されたら、Python環境で欠損値や分布、外れ値などを確認。
    - 季節性（例: 冬に増える心臓手術）や医師別嗜好の仮説を可視化。

    **2. 特徴量エンジニアリング**  
    - 在庫回転率や賞味期限フラグ、医師選好度などを計算。
    - リアルタイム連携する手術スケジュールとの突合や、連続使用実績を活かした特徴を作る。

    **3. モデル選定・実装**  
    - 需要予測: Prophet, ARIMA + 機械学習モデル(XGBoost, LightGBMなど)
    - 減耗予測: ロジスティック回帰やRandomForestで「廃棄になりそうか」を分類 or 回帰

    **4. ハイパーパラメータチューニング**  
    - 交差検証やローリングウィンドウ検証で精度を確認。
    - MLflowやWeights & Biasesなどで実験管理。

    **5. モデル評価**  
    - 需要予測: RMSE/MAE/MAPEなど
    - 廃棄予測: Precision/Recall, コスト換算
    - 現場フィードバック: 医師や在庫担当の意見を反映

    **6. パイロット運用・A/Bテスト**  
    - 部分導入して、廃棄コストや欠品件数などの改善度を測定
    """)

    st.subheader("デモ：需要予測のサンプル可視化")

    # ダミーデータ：日付と需要(予測値)を適当に作る
    n_days = 30
    date_rng = pd.date_range(start="2025-01-01", periods=n_days, freq="D")
    demand = np.random.randint(50, 120, size=n_days)  # ランダムな需要

    df_demand = pd.DataFrame({
        "date": date_rng,
        "predicted_demand": demand
    })

    # 予測期間をスライダーで制限
    limit_days = st.slider("表示する期間（日数）", 7, n_days, 14)
    df_demand_limited = df_demand.head(limit_days)

    # グラフ表示(Altair)
    chart_demand = alt.Chart(df_demand_limited).mark_line().encode(
        x='date:T',
        y='predicted_demand:Q'
    ).properties(
        width=600,
        height=300,
        title="予測需要 (サンプル)"
    )
    st.altair_chart(chart_demand, use_container_width=True)

with tabs[1]:
    st.header("フェーズ4: 本番運用・MLOps確立")

    st.markdown("""
    **1. モデルのデプロイ方法の決定**  
    - バッチ推論：例: 週1回需要予測を更新して在庫管理システムに自動連携
    - リアルタイム推論：緊急手術などのトリガーでオンデマンドAPI予測
    - コンテナ化：Docker + K8s(EKSなど) でスケーラビリティ確保

    **2. サービングとオーケストレーション**  
    - Airflow/Argo Workflowで「データ取得→モデル学習→推論→結果反映」をパイプライン化
    - DWHやBIツール(例: Redshift, Looker等)に接続し、ダッシュボードで一元管理

    **3. モデル監視・メンテナンス**  
    - 予測誤差(MAPE等)・データ分布の変化(Concept Drift)をモニタ
    - 異常検知で通知→再学習サイクルを自動化
    - 新医師や新製品追加時にモデルを再フィット

    **4. ダッシュボード・アラート設計**  
    - 経営層向け: KGI/KPIダッシュボード(廃棄コスト、削減率など)
    - 在庫担当向け: 使用率急上昇/滅菌期限切れアラート、発注推奨数
    - 手術室向けリアルタイム画面: 各ロケーションの在庫本数、欠品リスク通知

    """)

    st.subheader("デモ：在庫モニタリング (サンプル)")

    # ダミーの在庫データ
    item_names = ["カテーテルA", "カテーテルB", "カテーテルC", "カテーテルD"]
    stock_data = {
        "item": item_names,
        "current_stock": [100, 60, 30, 20],
        "risk_score": [0.1, 0.4, 0.7, 0.85]  # 0に近いほど低リスク
    }
    df_stock = pd.DataFrame(stock_data)

    st.dataframe(df_stock)

    # リスクスコア閾値
    threshold = st.slider("リスク閾値(0~1)", 0.0, 1.0, 0.5)
    high_risk_items = df_stock[df_stock["risk_score"] >= threshold]

    if len(high_risk_items) > 0:
        st.warning(f"リスクスコアが{threshold}以上の在庫品：")
        st.write(high_risk_items)
    else:
        st.success("リスクが高い在庫は現在ありません。")

st.markdown("---")
st.write("© 2025 In-Hospital AI Project - Demo Screen")
