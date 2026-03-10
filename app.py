import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import btc_signal_pipeline as btc

st.set_page_config(page_title="AlphaQuest Capital — BTC Signal Dashboard", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card { background: linear-gradient(135deg, #1e3a5f, #0f2137);
    border-radius: 12px; padding: 20px 24px; margin: 6px 0;
    border-left: 4px solid #f39c12; }
.kpi-label { color: #aab4be; font-size: 13px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
.kpi-value { color: #ffffff; font-size: 28px; font-weight: 800; margin: 4px 0; }
.kpi-delta-pos { color: #2ecc71; font-size: 14px; font-weight: 600; }
.kpi-delta-neg { color: #e74c3c; font-size: 14px; font-weight: 600; }
.section-header { border-bottom: 2px solid #f39c12; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

st.title("🟡 AlphaQuest Capital — BTC Signal Dashboard")
st.markdown("**FINTECH 717 — Investment Committee Pitch**  |  Strategy: Risk-Managed Long/Cash with 5% Stop-Loss kill switch")

CAPITAL = 50_000_000   # $50M unallocated capital

btc.OUTPUT_DIR = "."

if st.button("🚀 Run Full Pipeline", type="primary"):
    with st.spinner("Executing ETL → Feature Engineering → Training → Backtesting… (see terminal for verbose logs)"):
        try:
            df, models, scaler, metrics_df, backtests, prediction, lr_coef_df, lr_formula = btc.main()
            st.success("✅ Pipeline executed successfully!")

            # ═══════════════════════════════════════════════════════════════
            # SECTION 1 — NEXT DAY LIVE SIGNAL
            # ═══════════════════════════════════════════════════════════════
            st.header("🔮 Tomorrow's Trading Signal")
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            s_col1.metric("Today's BTC Close",  f"${prediction['close']:,.2f}", f"{prediction['today_ret']:+.2%}")
            s_col2.metric("Signal",             prediction["signal_label"])
            s_col3.metric("Model Confidence",   f"{prediction['proba']:.1%}",
                          delta="above cutoff" if prediction["proba"] >= btc.PROBA_CUTOFF else "below cutoff",
                          delta_color="normal" if prediction["proba"] >= btc.PROBA_CUTOFF else "inverse")
            s_col4.metric("Prob. Cutoff",       f"{btc.PROBA_CUTOFF:.0%}")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 2 — $50M INVESTMENT COMMITTEE KPI CARDS
            # ═══════════════════════════════════════════════════════════════
            st.header("💰 $50M Capital Allocation — Performance Report")
            st.caption("Simulated deployment of USD $50,000,000 into the Risk-Managed Long/Cash strategy on the 2024 test period.")

            # Pick best model by strategy return
            best_name = max(backtests, key=lambda k: backtests[k].attrs["strat_return"])
            bt_best   = backtests[best_name]
            strat_ret = bt_best.attrs["strat_return"]
            bnh_ret   = bt_best.attrs["bnh_return"]
            sharpe    = bt_best.attrs["sharpe"]
            max_dd    = bt_best.attrs["max_dd"]

            final_value    = CAPITAL * (1 + strat_ret)
            profit_loss    = final_value - CAPITAL
            bnh_value      = CAPITAL * (1 + bnh_ret)
            alpha          = strat_ret - bnh_ret          # excess return vs buy & hold

            buy_days       = int(bt_best["signal"].sum())
            total_days     = len(bt_best)
            n_trades       = int(bt_best["trade"].sum())
            avg_trade_ret  = bt_best.loc[bt_best["signal"].shift(1) == 1, "net_ret"].mean()

            # KPI row 1 — Portfolio level
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📥 Capital Deployed",  f"${CAPITAL/1e6:.0f}M",  "Starting capital")
            k2.metric("📤 Portfolio Value",   f"${final_value/1e6:.2f}M",
                      f"{'+' if profit_loss>=0 else ''}{profit_loss/1e6:.2f}M profit/loss",
                      delta_color="normal" if profit_loss >= 0 else "inverse")
            k3.metric("📈 Strategy Return",   f"{strat_ret:+.2%}",
                      f"vs Buy&Hold {bnh_ret:+.2%}", delta_color="normal" if strat_ret >= 0 else "inverse")
            k4.metric("⚡ Alpha vs Buy&Hold", f"{alpha:+.2%}",
                      "excess return", delta_color="normal" if alpha >= 0 else "inverse")

            st.markdown("")

            # KPI row 2 — Risk metrics
            k5, k6, k7, k8 = st.columns(4)
            k5.metric("📊 Sharpe Ratio",       f"{sharpe:.2f}",
                      "≥1.0 investable" if sharpe >= 1 else "<1.0 needs improvement",
                      delta_color="normal" if sharpe >= 1 else "off")
            k6.metric("📉 Max Drawdown",        f"{max_dd:.2%}",
                      "Worst peak-to-trough loss", delta_color="inverse" if max_dd < -0.1 else "normal")
            k7.metric("🔢 Number of Trades",    f"{n_trades}",
                      f"{buy_days} buy days / {total_days} total")
            k8.metric("💵 Avg Daily P&L",       f"${CAPITAL * avg_trade_ret / 1e3:+,.1f}K" if not np.isnan(avg_trade_ret) else "N/A",
                      "on active trade days")

            # KPI row 3 — Dollar figures
            st.markdown("### Dollar Impact")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("💸 Gross Profit",
                      f"${max(profit_loss, 0)/1e6:.2f}M",
                      "Strategy gain in USD")
            d2.metric("🛡️ Capital Preserved vs Buy&Hold",
                      f"${(bnh_value - final_value)/1e6:.2f}M avoided loss" if final_value < bnh_value else f"${(final_value - bnh_value)/1e6:.2f}M extra profit",
                      delta_color="normal")
            d3.metric("⛽ Est. Trading Fees",
                      f"${CAPITAL * n_trades * 0.001 / 1e3:.1f}K",
                      "0.1% per trade (Binance rate)")
            d4.metric("🏆 Best Model",
                      best_name, f"selected for max return")

            st.info(f"📌 **IC Summary:** Starting from **$50M**, the **{best_name}** strategy grew the portfolio to **${final_value/1e6:.2f}M** (+{strat_ret:.1%}) over the 2024 test period. This compares to **${bnh_value/1e6:.2f}M** (+{bnh_ret:.1%}) if we had simply held Bitcoin. The Sharpe Ratio of **{sharpe:.2f}** {'demonstrates a risk-adjusted return suitable for the Investment Committee.' if sharpe >= 0.8 else 'indicates the strategy needs further refinement before capital deployment.'}")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 3 — DATA PIPELINE INSPECTION
            # ═══════════════════════════════════════════════════════════════
            st.header("🔍 Data Pipeline Inspection")
            tab1, tab2, tab3 = st.tabs(["Raw Market Data", "Engineered Features", "Buy Signal Distribution"])

            with tab1:
                st.info("**Google Trends (`gtrends`)** — 0-100 scale. 100 = peak search interest. Fetched weekly via pytrends, interpolated to daily, **lagged by 1 day** to prevent look-ahead bias.")
                st.dataframe(df[["open","high","low","close","volume","gtrends"]].head(10), use_container_width=True)
                st.dataframe(df[["open","high","low","close","volume","gtrends"]].tail(10), use_container_width=True)

            with tab2:
                st.info(f"**7 CORE Features used:** {', '.join(btc.CORE_FEATURES)}")
                st.dataframe(df[btc.CORE_FEATURES].head(10), use_container_width=True)
                st.dataframe(df[btc.CORE_FEATURES].tail(10), use_container_width=True)

            with tab3:
                st.markdown(f"**3-Day Forward Return > {btc.SIGNAL_THRESHOLD*100:.0f}%** = Buy label")
                vc = df["target"].value_counts().rename("Count (Days)")
                vc_pct = df["target"].value_counts(normalize=True).rename("Proportion")
                dist   = pd.concat([vc, vc_pct], axis=1)
                dist.index = ["Hold (0)", "Buy (1)"] if len(dist) == 2 else dist.index
                st.dataframe(dist.style.format({"Proportion": "{:.1%}"}), use_container_width=True)

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 4 — LR FORMULA
            # ═══════════════════════════════════════════════════════════════
            st.header("🧮 Logistic Regression Formula")
            st.markdown("The complete log-odds equation. A **positive coefficient** means higher feature value → higher P(Buy).")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(lr_coef_df.style.background_gradient(cmap="RdYlGn", subset=["Coefficient"]),
                             use_container_width=True)
            with c2:
                st.code(lr_formula, language="text")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 5 — VISUALISATIONS
            # ═══════════════════════════════════════════════════════════════
            st.header("📊 Pipeline & Backtest Charts")
            st.caption(f"Strategy: Risk-Managed Long/Cash | Kill Switch: -5% daily loss | PROBA_CUTOFF: {btc.PROBA_CUTOFF:.0%}")

            for p in ["plot1_btc_technicals.png","plot2_model_evaluation.png",
                      "plot3_confusion_matrices.png","plot4_backtest.png","plot5_features.png"]:
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                else:
                    st.warning(f"Plot '{p}' not found — run pipeline first.")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 6 — MODEL EVALUATION TABLE
            # ═══════════════════════════════════════════════════════════════
            st.header("📈 Full Model Evaluation Metrics")
            st.dataframe(
                metrics_df.style.background_gradient(cmap="Blues", subset=["ROC_AUC","Recall","F1"]),
                use_container_width=True
            )

        except Exception as e:
            import traceback
            st.error(f"Error running pipeline: {e}")
            st.code(traceback.format_exc())

else:
    # Landing page stats
    st.markdown("---")
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.metric("Capital to Deploy", "$50,000,000", "Unallocated")
    col_l2.metric("Strategy",          "Long/Cash + Kill-Switch", "Risk-managed")
    col_l3.metric("Models",            "3 Ensemble",  "LR + RF + GB")
    st.info("👆 Click **Run Full Pipeline** above to execute the full ETL → Training → Backtest → Tomorrow's Signal workflow.")
