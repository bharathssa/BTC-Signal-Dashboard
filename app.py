import streamlit as st
import pandas as pd
import numpy as np
import os

import btc_signal_pipeline as btc

st.set_page_config(page_title="AlphaQuest Capital — Multi-Asset Dashboard", layout="wide")

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
.state-badge { font-size: 20px; font-weight: 800; padding: 12px 20px;
    border-radius: 10px; background: #1e3a5f; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🟡 AlphaQuest Capital — Multi-Asset Signal Dashboard")
st.markdown("**FINTECH 717 — Investment Committee Pitch**  |  Strategy: BTC + ETH + USDT · 4-State Portfolio · 5% Kill Switch")

CAPITAL = 50_000_000   # $50M unallocated capital
btc.OUTPUT_DIR = "."

if st.button("🚀 Run Full Pipeline", type="primary"):
    with st.spinner("Executing ETL → Feature Engineering → BTC + ETH Training → 4-State Portfolio Backtest…"):
        try:
            (df_btc, df_eth, models_btc, models_eth, scaler_btc, scaler_eth,
             metrics_btc, metrics_eth, backtests_btc, portfolio_bt,
             prediction_btc, prediction_eth, lr_coef_df, lr_formula) = btc.main()

            st.success("✅ Pipeline executed successfully — BTC + ETH dual-model complete!")

            # ═══════════════════════════════════════════════════════════════
            # SECTION 1 — TOMORROW'S PORTFOLIO DECISION
            # ═══════════════════════════════════════════════════════════════
            st.header("🔮 Tomorrow's Portfolio Decision")

            combined_key   = (prediction_btc["signal"], prediction_eth["signal"])
            combined_state = btc.PORTFOLIO_STATES[combined_key]["label"]
            w_btc = btc.PORTFOLIO_STATES[combined_key]["btc"]
            w_eth = btc.PORTFOLIO_STATES[combined_key]["eth"]
            w_usdt= btc.PORTFOLIO_STATES[combined_key]["usdt"]

            # Portfolio state banner
            st.markdown(f"""<div class='state-badge'>{combined_state}</div>""", unsafe_allow_html=True)
            st.markdown("")

            sig_col1, sig_col2, sig_col3, sig_col4, sig_col5 = st.columns(5)
            sig_col1.metric("₿ BTC Signal",
                            "🟢 BUY"  if prediction_btc["signal"] else "⚪ HOLD",
                            f"{prediction_btc['proba']:.1%} confidence")
            sig_col2.metric("Ξ ETH Signal",
                            "🟢 BUY"  if prediction_eth["signal"] else "⚪ HOLD",
                            f"{prediction_eth['proba']:.1%} confidence")
            sig_col3.metric("BTC Allocation",  f"{w_btc:.0%}",  f"of $50M = ${CAPITAL*w_btc/1e6:.1f}M")
            sig_col4.metric("ETH Allocation",  f"{w_eth:.0%}",  f"of $50M = ${CAPITAL*w_eth/1e6:.1f}M")
            sig_col5.metric("USDT (Safe-Haven)", f"{w_usdt:.0%}", f"of $50M = ${CAPITAL*w_usdt/1e6:.1f}M")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 2 — PORTFOLIO KPI CARDS
            # ═══════════════════════════════════════════════════════════════
            st.header("💰 $50M Portfolio — 4-State Backtest Results (2024)")
            st.caption("Blended daily returns from BTC + ETH based on model signals. 0.1% fee per trade. 5% kill-switch active.")

            strat_ret = portfolio_bt.attrs["strat_return"]
            btc_bnh   = portfolio_bt.attrs["btc_bnh"]
            eth_bnh   = portfolio_bt.attrs["eth_bnh"]
            sharpe    = portfolio_bt.attrs["sharpe"]
            max_dd    = portfolio_bt.attrs["max_dd"]

            final_value = CAPITAL * (1 + strat_ret)
            profit_loss = final_value - CAPITAL
            alpha_btc   = strat_ret - btc_bnh

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📥 Capital Deployed",  f"${CAPITAL/1e6:.0f}M", "Starting capital")
            k2.metric("📤 Portfolio Value",   f"${final_value/1e6:.2f}M",
                      f"{'+' if profit_loss>=0 else ''}{profit_loss/1e6:.2f}M P&L",
                      delta_color="normal" if profit_loss >= 0 else "inverse")
            k3.metric("📈 Portfolio Return",  f"{strat_ret:+.2%}",
                      f"vs BTC B&H {btc_bnh:+.2%}", delta_color="normal" if strat_ret >= 0 else "inverse")
            k4.metric("⚡ Alpha vs BTC B&H", f"{alpha_btc:+.2%}",
                      "excess return", delta_color="normal" if alpha_btc >= 0 else "inverse")

            st.markdown("")
            k5, k6, k7, k8 = st.columns(4)
            k5.metric("📊 Sharpe Ratio",  f"{sharpe:.2f}",
                      "≥1.0 investable" if sharpe >= 1 else "<1.0 needs improvement",
                      delta_color="normal" if sharpe >= 1 else "off")
            k6.metric("📉 Max Drawdown",  f"{max_dd:.2%}", "Worst peak-to-trough loss",
                      delta_color="inverse")
            k7.metric("₿ BTC Buy&Hold",  f"{btc_bnh:+.2%}", "2024 benchmark")
            k8.metric("Ξ ETH Buy&Hold",  f"{eth_bnh:+.2%}", "2024 benchmark")

            # Days per state
            st.markdown("### Portfolio State Distribution (2024)")
            state_counts = portfolio_bt["state_label"].value_counts()
            state_df = pd.DataFrame({
                "State": state_counts.index,
                "Days":  state_counts.values,
                "% of Year": (state_counts.values / len(portfolio_bt) * 100).round(1)
            })
            st.dataframe(state_df, use_container_width=True, hide_index=True)

            # IC Summary
            st.info(f"📌 **IC Summary:** Starting from **$50M**, the 4-state BTC+ETH+USDT portfolio "
                    f"grew to **${final_value/1e6:.2f}M** ({strat_ret:+.1%}) over the 2024 test period. "
                    f"BTC Buy&Hold returned {btc_bnh:+.1%}; ETH Buy&Hold returned {eth_bnh:+.1%}. "
                    f"The portfolio Sharpe Ratio of **{sharpe:.2f}** "
                    f"{'demonstrates risk-adjusted performance suitable for the IC.' if sharpe >= 0.8 else 'indicates further threshold tuning is recommended.'}")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 3 — TODAY'S PRICES
            # ═══════════════════════════════════════════════════════════════
            st.header("💲 Current Prices")
            p1, p2 = st.columns(2)
            p1.metric("Bitcoin (BTC)", f"${prediction_btc['close']:,.2f}", f"{prediction_btc['today_ret']:+.2%} today")
            p2.metric("Ethereum (ETH)", f"${prediction_eth['close']:,.2f}", f"{prediction_eth['today_ret']:+.2%} today")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 4 — VISUALISATIONS
            # ═══════════════════════════════════════════════════════════════
            st.header("📊 Pipeline & Backtest Charts")
            st.caption(f"BTC Model | PROBA_CUTOFF: {btc.PROBA_CUTOFF:.0%} | Kill Switch: -5% daily stop")

            for p in ["plot1_btc_technicals.png", "plot2_model_evaluation.png",
                      "plot3_confusion_matrices.png", "plot4_backtest.png", "plot5_features.png"]:
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                else:
                    st.warning(f"Plot '{p}' not found — run pipeline first.")

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 5 — MODEL EVALUATION TABLES
            # ═══════════════════════════════════════════════════════════════
            st.header("📈 Model Evaluation Metrics")
            tab_btc, tab_eth = st.tabs(["₿ BTC Model", "Ξ ETH Model"])
            with tab_btc:
                st.dataframe(metrics_btc.style.background_gradient(cmap="Blues", subset=["ROC_AUC","Recall","F1"]),
                             use_container_width=True)
            with tab_eth:
                st.dataframe(metrics_eth.style.background_gradient(cmap="Purples", subset=["ROC_AUC","Recall","F1"]),
                             use_container_width=True)

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 6 — DATA INSPECTION
            # ═══════════════════════════════════════════════════════════════
            st.header("🔍 Data Inspection")
            tab1, tab2 = st.tabs(["₿ BTC Features", "Ξ ETH Features"])
            with tab1:
                st.info(f"**7 CORE Features:** {', '.join(btc.CORE_FEATURES)}")
                st.dataframe(df_btc[btc.CORE_FEATURES].tail(15), use_container_width=True)
            with tab2:
                st.info(f"**7 CORE Features:** {', '.join(btc.CORE_FEATURES)}")
                st.dataframe(df_eth[btc.CORE_FEATURES].tail(15), use_container_width=True)

            st.divider()

            # ═══════════════════════════════════════════════════════════════
            # SECTION 7 — LR FORMULA
            # ═══════════════════════════════════════════════════════════════
            st.header("🧮 Logistic Regression Formula (BTC Model)")
            st.markdown("Positive coefficient → higher feature value → higher P(Buy)")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(lr_coef_df.style.background_gradient(cmap="RdYlGn", subset=["Coefficient"]),
                             use_container_width=True)
            with c2:
                st.code(lr_formula, language="text")

        except Exception as e:
            import traceback
            st.error(f"Error running pipeline: {e}")
            st.code(traceback.format_exc())

else:
    st.markdown("---")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.metric("Capital to Deploy", "$50,000,000", "Unallocated")
    col_l2.metric("Assets",            "BTC + ETH + USDT", "3-asset portfolio")
    col_l3.metric("Portfolio States",  "4",  "Full / BTC / ETH / Defensive")
    col_l4.metric("Models per Asset",  "5",  "LR + RF + GB + Ensemble + MLP")
    st.info("👆 Click **Run Full Pipeline** above to execute the full ETL → Dual-Model Training → 4-State Portfolio Backtest → Tomorrow's Signals workflow.")
