import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import date, datetime

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
.sidebar-header { color: #f39c12; font-size: 13px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px; margin-top: 8px; }
.whatif-note { color: #aab4be; font-size: 12px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.title("🟡 AlphaQuest Capital — Multi-Asset Signal Dashboard")
st.markdown("**FINTECH 717 — Investment Committee Pitch**  |  Strategy: BTC + ETH + USDT · 4-State Portfolio · Kill Switch")

btc.OUTPUT_DIR = "."

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Interactive Controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://em-content.zobj.net/source/google/387/bar-chart_1f4ca.png", width=40)
    st.title("⚙️ Strategy Controls")
    st.caption("Adjust parameters to explore what-if scenarios. Models re-use pre-trained weights — only the backtest reruns.")

    st.markdown("<div class='sidebar-header'>📅 Backtest Window</div>", unsafe_allow_html=True)
    bt_start = st.date_input(
        "Start Date",
        value=date(2025, 1, 1),
        min_value=date(2020, 1, 1),
        max_value=date(2026, 2, 28),
        help="Backtest evaluation start. Models are always trained on 2020–2023 data regardless of this setting.",
    )
    bt_end = st.date_input(
        "End Date",
        value=date(2025, 12, 31),
        min_value=date(2020, 1, 2),
        max_value=date(2026, 2, 28),
        help="Backtest evaluation end date. Default is full year 2025.",
    )

    st.divider()
    st.markdown("<div class='sidebar-header'>🎯 Signal Probability Cutoffs</div>", unsafe_allow_html=True)
    btc_cutoff = st.slider(
        "BTC Buy Threshold",
        min_value=0.40, max_value=0.75, value=0.55, step=0.01,
        format="%.2f",
        help="Minimum probability for a BTC buy signal. Higher = fewer but more selective trades. Default 0.55 is more conservative than the training cutoff.",
    )
    eth_cutoff = st.slider(
        "ETH Buy Threshold",
        min_value=0.40, max_value=0.75, value=0.63, step=0.01,
        format="%.2f",
        help="Minimum probability for an ETH buy signal. Default 0.63 is conservative — ETH is more volatile so we require high confidence.",
    )

    st.divider()
    st.markdown("<div class='sidebar-header'>⚖️ Allocation per Portfolio State</div>", unsafe_allow_html=True)
    st.caption("**State: Full Risk-On** (BTC signal=✅ AND ETH signal=✅)")
    full_btc_pct = st.slider("BTC %", 0, 100, 70, 5, key="full_btc",
                              help="BTC allocation when both signals are bullish. ETH fills the rest.")
    full_eth_pct = st.slider("ETH %", 0, 100 - full_btc_pct, min(30, 100 - full_btc_pct), 5, key="full_eth",
                              help="ETH allocation when both signals are bullish.")
    full_usdt_pct = 100 - full_btc_pct - full_eth_pct
    st.caption(f"↳ USDT (safe-haven): **{full_usdt_pct}%**")

    st.markdown("---")
    st.caption("**State: BTC Only** (BTC=✅, ETH=⚪)")
    btc_only_pct = st.slider("BTC % (BTC-only state)", 50, 100, 100, 5, key="btc_only",
                              help="BTC allocation when only BTC signal is active. Remainder goes to USDT.")
    btc_only_usdt = 100 - btc_only_pct
    st.caption(f"↳ USDT: **{btc_only_usdt}%**")

    st.markdown("---")
    st.caption("**State: ETH Partial** (BTC=⚪, ETH=✅)")
    eth_pct = st.slider("ETH % (ETH-partial state)", 0, 100, 60, 5, key="eth_partial",
                         help="ETH allocation when only ETH signal fires. Remainder = USDT hedge.")
    eth_usdt = 100 - eth_pct
    st.caption(f"↳ USDT hedge: **{eth_usdt}%**")

    st.markdown("---")
    st.caption("**State: Defensive** (BTC=⚪, ETH=⚪) → 100% USDT (fixed)")

    st.divider()
    st.markdown("<div class='sidebar-header'>💰 Portfolio & Risk Settings</div>", unsafe_allow_html=True)
    capital_m = st.slider(
        "Starting Capital ($M)", 1, 200, 50, 1,
        help="Portfolio starting value in USD millions.",
    )
    CAPITAL = capital_m * 1_000_000

    fee_bps = st.slider(
        "Transaction Fee (bps)", 0, 50, 10, 1,
        help="Fee charged on each trade (each side). 10 bps = 0.10% ≈ Binance rate.",
    )
    fee_rate = fee_bps / 10_000

    kill_pct_val = st.slider(
        "Kill-Switch Threshold (%)", 1, 15, 5, 1,
        help="If daily loss exceeds this %, force exit to USDT next day.",
    )
    kill_pct = kill_pct_val / 100

    st.divider()
    st.markdown("<div class='sidebar-header'>📊 What-If Table</div>", unsafe_allow_html=True)
    show_whatif = st.checkbox("Show Year-by-Year History Table", value=True,
                               help="Compute performance for each calendar year from 2020 to end date.")

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUTTON — cache the heavy ML training; backtest reruns via sidebar
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def _run_cached_pipeline():
    """Cache trained models + raw data for 1 hour. Backtest reruns separately."""
    return btc.main()

if st.button("🚀 Run Full Pipeline", type="primary"):
    with st.spinner("Executing ETL → Feature Engineering → BTC + ETH Training → 4-State Portfolio Backtest… (data cached after first run)"):
        try:
            (df_btc, df_eth, models_btc, models_eth, scaler_btc, scaler_eth,
             metrics_btc, metrics_eth, backtests_btc, portfolio_bt,
             prediction_btc, prediction_eth, lr_coef_df, lr_formula) = _run_cached_pipeline()

            # ── Store trained artefacts in session_state so sidebar can reuse them ──
            st.session_state["pipeline_done"]   = True
            st.session_state["df_btc"]          = df_btc
            st.session_state["df_eth"]          = df_eth
            st.session_state["models_btc"]      = models_btc
            st.session_state["models_eth"]      = models_eth
            st.session_state["scaler_btc"]      = scaler_btc
            st.session_state["scaler_eth"]      = scaler_eth
            # Use CORE_FEATURES — the exact 7 features the models were trained on.
            # Do NOT recompute from column difference; that produces extra columns
            # that the fitted model has never seen (causes sklearn feature-name error).
            core_feats = [f for f in btc.CORE_FEATURES if f in df_btc.columns]
            st.session_state["features_btc"]    = core_feats
            st.session_state["features_eth"]    = [f for f in btc.CORE_FEATURES if f in df_eth.columns]
            st.session_state["metrics_btc"]     = metrics_btc
            st.session_state["metrics_eth"]     = metrics_eth
            st.session_state["backtests_btc"]   = backtests_btc
            st.session_state["prediction_btc"]  = prediction_btc
            st.session_state["prediction_eth"]  = prediction_eth
            st.session_state["lr_coef_df"]      = lr_coef_df
            st.session_state["lr_formula"]      = lr_formula

            st.success("✅ Pipeline executed successfully — BTC + ETH dual-model complete!")

        except Exception as e:
            import traceback
            st.error(f"Error running pipeline: {e}")
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — renders whenever pipeline results are in session_state
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("pipeline_done"):

    df_btc        = st.session_state["df_btc"]
    df_eth        = st.session_state["df_eth"]
    models_btc    = st.session_state["models_btc"]
    models_eth    = st.session_state["models_eth"]
    scaler_btc    = st.session_state["scaler_btc"]
    scaler_eth    = st.session_state["scaler_eth"]
    features_btc  = st.session_state["features_btc"]
    features_eth  = st.session_state["features_eth"]
    metrics_btc   = st.session_state["metrics_btc"]
    metrics_eth   = st.session_state["metrics_eth"]
    backtests_btc = st.session_state["backtests_btc"]
    prediction_btc= st.session_state["prediction_btc"]
    prediction_eth= st.session_state["prediction_eth"]
    lr_coef_df    = st.session_state["lr_coef_df"]
    lr_formula    = st.session_state["lr_formula"]

    # ── Re-run backtest with sidebar params (fast — no ML training) ───────────
    custom_bt = btc.run_portfolio_backtest_custom(
        models_btc, scaler_btc, features_btc,
        models_eth, scaler_eth, features_eth,
        df_btc, df_eth,
        start_date       = str(bt_start),
        end_date         = str(bt_end),
        alloc_full_btc   = full_btc_pct / 100,
        alloc_full_eth   = full_eth_pct / 100,
        alloc_btc_only   = btc_only_pct / 100,
        alloc_eth_eth    = eth_pct / 100,
        btc_cutoff       = btc_cutoff,
        eth_cutoff       = eth_cutoff,
        fee_rate         = fee_rate,
        kill_pct         = kill_pct,
    )

    strat_ret   = custom_bt.attrs["strat_return"]
    btc_bnh     = custom_bt.attrs["btc_bnh"]
    eth_bnh     = custom_bt.attrs["eth_bnh"]
    sharpe      = custom_bt.attrs["sharpe"]
    max_dd      = custom_bt.attrs["max_dd"]
    final_value = CAPITAL * (1 + strat_ret)
    profit_loss = final_value - CAPITAL
    alpha_btc   = strat_ret - btc_bnh

    period_label = f"{bt_start.strftime('%d %b %Y')} → {bt_end.strftime('%d %b %Y')}"

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1 — TOMORROW'S PORTFOLIO DECISION
    # ═══════════════════════════════════════════════════════════════
    st.header("🔮 Tomorrow's Portfolio Decision")

    combined_key   = (prediction_btc["signal"], prediction_eth["signal"])
    combined_state = btc.PORTFOLIO_STATES[combined_key]["label"]
    w_btc  = btc.PORTFOLIO_STATES[combined_key]["btc"]
    w_eth  = btc.PORTFOLIO_STATES[combined_key]["eth"]
    w_usdt = btc.PORTFOLIO_STATES[combined_key]["usdt"]

    st.markdown(f"""<div class='state-badge'>{combined_state}</div>""", unsafe_allow_html=True)
    st.markdown("")

    sig_col1, sig_col2, sig_col3, sig_col4, sig_col5 = st.columns(5)
    sig_col1.metric("₿ BTC Signal",
                    "🟢 BUY"  if prediction_btc["signal"] else "⚪ HOLD",
                    f"{prediction_btc['proba']:.1%} confidence")
    sig_col2.metric("Ξ ETH Signal",
                    "🟢 BUY"  if prediction_eth["signal"] else "⚪ HOLD",
                    f"{prediction_eth['proba']:.1%} confidence")
    sig_col3.metric("BTC Allocation",  f"{w_btc:.0%}",  f"of ${capital_m}M = ${CAPITAL*w_btc/1e6:.1f}M")
    sig_col4.metric("ETH Allocation",  f"{w_eth:.0%}",  f"of ${capital_m}M = ${CAPITAL*w_eth/1e6:.1f}M")
    sig_col5.metric("USDT (Safe-Haven)", f"{w_usdt:.0%}", f"of ${capital_m}M = ${CAPITAL*w_usdt/1e6:.1f}M")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2 — PORTFOLIO KPI CARDS
    # ═══════════════════════════════════════════════════════════════
    st.header(f"💰 ${capital_m}M Portfolio — Backtest Results ({period_label})")
    st.caption(f"Blended daily returns from BTC + ETH based on model signals. {fee_bps}bps fee per trade. {kill_pct_val}% kill-switch active.")

    st.markdown("### 🏆 Top Performing Models (Selected by ROC-AUC)")
    m1, m2 = st.columns(2)
    m1.metric("🧠 Best BTC Model", prediction_btc.get('model_name', 'Unknown'))
    m2.metric("🧠 Best ETH Model", prediction_eth.get('model_name', 'Unknown'))
    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📥 Capital Deployed",  f"${capital_m}M", "Starting capital")
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
    k7.metric("₿ BTC Buy&Hold",  f"{btc_bnh:+.2%}", f"{bt_start.year}–{bt_end.year} benchmark")
    k8.metric("Ξ ETH Buy&Hold",  f"{eth_bnh:+.2%}", f"{bt_start.year}–{bt_end.year} benchmark")

    # Days per state
    if len(custom_bt) > 0:
        st.markdown("### Portfolio State Distribution")
        state_counts = custom_bt["state_label"].value_counts()
        state_df = pd.DataFrame({
            "State":    state_counts.index,
            "Days":     state_counts.values,
            "% of Period": (state_counts.values / len(custom_bt) * 100).round(1)
        })
        st.dataframe(state_df, use_container_width=True, hide_index=True)

    # IC Summary
    st.info(f"📌 **IC Summary:** Starting from **${capital_m}M**, the 4-state BTC+ETH+USDT portfolio "
            f"grew to **${final_value/1e6:.2f}M** ({strat_ret:+.1%}) over {period_label}. "
            f"BTC Buy&Hold returned {btc_bnh:+.1%}; ETH Buy&Hold returned {eth_bnh:+.1%}. "
            f"Portfolio Sharpe Ratio: **{sharpe:.2f}** | Max Drawdown: **{max_dd:.2%}**.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3 — WHAT-IF HISTORY TABLE
    # ═══════════════════════════════════════════════════════════════
    if show_whatif and len(custom_bt) > 0:
        st.header("📆 What-If: Year-by-Year Performance")
        st.caption(
            f"How would ${capital_m}M have performed each calendar year using our strategy with current slider settings? "
            f"Uses the same trained models, signal cutoffs ({btc_cutoff:.2f}/{eth_cutoff:.2f}), and allocation weights as above."
        )

        first_year = max(2020, bt_end.year - 2)   # last 3 calendar years from end date
        last_year  = bt_end.year
        year_rows  = []
        compound_value = CAPITAL   # tracks compounded portfolio value year by year

        for yr in range(first_year, last_year + 1):
            yr_start = str(max(date(yr, 1, 1), bt_start))
            yr_end   = str(min(date(yr, 12, 31), bt_end))

            if yr_start > yr_end:
                continue

            ybt = btc.run_portfolio_backtest_custom(
                models_btc, scaler_btc, features_btc,
                models_eth, scaler_eth, features_eth,
                df_btc, df_eth,
                start_date       = yr_start,
                end_date         = yr_end,
                alloc_full_btc   = full_btc_pct / 100,
                alloc_full_eth   = full_eth_pct / 100,
                alloc_btc_only   = btc_only_pct / 100,
                alloc_eth_eth    = eth_pct / 100,
                btc_cutoff       = btc_cutoff,
                eth_cutoff       = eth_cutoff,
                fee_rate         = fee_rate,
                kill_pct         = kill_pct,
            )
            if len(ybt) == 0:
                continue

            yr_ret   = ybt.attrs["strat_return"]
            yr_btc   = ybt.attrs["btc_bnh"]
            yr_eth   = ybt.attrs["eth_bnh"]
            yr_sh    = ybt.attrs["sharpe"]
            yr_dd    = ybt.attrs["max_dd"]
            yr_final = compound_value * (1 + yr_ret)
            alpha_y  = yr_ret - yr_btc

            # Count actual trading days available (some years may be partial)
            days_in_period = len(ybt)

            year_rows.append({
                "Year":           yr,
                "Period":         f"{yr_start} → {yr_end}",
                "Strategy":       f"{yr_ret:+.1%}",
                "BTC B&H":        f"{yr_btc:+.1%}",
                "ETH B&H":        f"{yr_eth:+.1%}",
                "Alpha":          f"{alpha_y:+.1%}",
                "Sharpe":         f"{yr_sh:.2f}",
                "Max DD":         f"{yr_dd:.1%}",
                "Days":           days_in_period,
                "Start Value":    f"${compound_value/1e6:.2f}M",
                "End Value":      f"${yr_final/1e6:.2f}M",
                # raw for coloring
                "_ret":           yr_ret,
                "_alpha":         alpha_y,
                "_sharpe":        yr_sh,
            })
            compound_value = yr_final   # compound into next year

        if year_rows:
            # Summary row (full period)
            total_ret_all = compound_value / CAPITAL - 1
            summary_row = {
                "Year":       "📊 TOTAL",
                "Period":     f"{bt_start} → {bt_end}",
                "Strategy":   f"{total_ret_all:+.1%}",
                "BTC B&H":    "",
                "ETH B&H":    "",
                "Alpha":      "",
                "Sharpe":     "",
                "Max DD":     "",
                "Days":       sum(r["Days"] for r in year_rows),
                "Start Value": f"${CAPITAL/1e6:.2f}M",
                "End Value":  f"${compound_value/1e6:.2f}M",
                "_ret":       total_ret_all,
                "_alpha":     0,
                "_sharpe":    0,
            }
            all_rows = year_rows + [summary_row]

            display_cols = ["Year", "Period", "Strategy", "BTC B&H", "ETH B&H",
                            "Alpha", "Sharpe", "Max DD", "Days", "Start Value", "End Value"]
            df_whatif = pd.DataFrame(all_rows)[display_cols]

            # Style the table
            def color_strategy(val):
                try:
                    v = float(val.replace("%","").replace("+",""))
                    return f"color: {'#2ecc71' if v >= 0 else '#e74c3c'}; font-weight: 700"
                except Exception:
                    return ""

            def color_alpha(val):
                try:
                    v = float(val.replace("%","").replace("+",""))
                    return f"color: {'#2ecc71' if v >= 0 else '#e74c3c'}"
                except Exception:
                    return ""

            styled = (
                df_whatif.style
                .applymap(color_strategy, subset=["Strategy"])
                .applymap(color_alpha,    subset=["Alpha"])
                .set_properties(**{"text-align": "center"})
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#1e3a5f"),
                                                  ("color", "#f39c12"),
                                                  ("font-weight", "700"),
                                                  ("text-align", "center")]},
                    {"selector": "tr:last-child td",
                     "props": [("background-color", "#1e3a5f"),
                                ("color", "#ffffff"),
                                ("font-weight", "800"),
                                ("border-top", "2px solid #f39c12")]},
                ])
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Highlight the compounded result
            delta_emoji = "🚀" if compound_value > CAPITAL else "📉"
            st.success(
                f"{delta_emoji} **Compounded Result:** ${capital_m}M invested at **{bt_start}** "
                f"→ **${compound_value/1e6:.2f}M** by **{bt_end}** "
                f"({total_ret_all:+.1%} total return)"
            )
        else:
            st.warning("No data available for the selected date range.")
        st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4 — TODAY'S PRICES
    # ═══════════════════════════════════════════════════════════════
    st.header("💲 Current Prices")
    p1, p2 = st.columns(2)
    p1.metric("Bitcoin (BTC)", f"${prediction_btc['close']:,.2f}", f"{prediction_btc['today_ret']:+.2%} on {prediction_btc['date']}")
    p1.caption(f"🧠 Selected Model: **{prediction_btc.get('model_name', 'Unknown')}**")

    p2.metric("Ethereum (ETH)", f"${prediction_eth['close']:,.2f}", f"{prediction_eth['today_ret']:+.2%} on {prediction_eth['date']}")
    p2.caption(f"🧠 Selected Model: **{prediction_eth.get('model_name', 'Unknown')}**")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5 — VISUALISATIONS
    # ═══════════════════════════════════════════════════════════════
    st.header("📊 Pipeline & Backtest Charts")
    st.caption(f"BTC Cutoff: {btc_cutoff:.0%} | ETH Cutoff: {eth_cutoff:.0%} | Kill Switch: -{kill_pct_val}% daily stop")

    # ── Cumulative Return (most important — show first) ──────────────────
    st.subheader("📈 Cumulative Return: Portfolio vs BTC vs ETH Buy & Hold")
    p7 = "plot7_cumulative_returns.png"
    if os.path.exists(p7):
        st.image(p7, use_container_width=True)
    else:
        st.warning("Run the pipeline to generate the cumulative return chart.")

    st.divider()

    # ── BTC Technicals ──────────────────────────────────────────────────
    st.subheader("₿ BTC Technical Indicator Overview")
    p1img = "plot1_btc_technicals.png"
    if os.path.exists(p1img):
        st.image(p1img, use_container_width=True)
    else:
        st.warning("BTC Technicals chart not found — run pipeline first.")

    # ── ETH Technicals ──────────────────────────────────────────────────
    st.subheader("Ξ ETH Technical Indicator Overview")
    p6 = "plot6_eth_technicals.png"
    if os.path.exists(p6):
        st.image(p6, use_container_width=True)
    else:
        st.warning("ETH Technicals chart not found — run pipeline first.")

    st.divider()

    # ── Remaining charts ─────────────────────────────────────────────────
    st.subheader("🧪 Model Diagnostics")
    for p in ["plot2_model_evaluation.png", "plot3_confusion_matrices.png",
              "plot4_backtest.png", "plot5_features.png"]:
        if os.path.exists(p):
            st.image(p, use_container_width=True)
        else:
            st.warning(f"'{p}' not found — run pipeline first.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6 — MODEL EVALUATION TABLES
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
    # SECTION 7 — DATA INSPECTION
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
    # SECTION 8 — LR FORMULA
    # ═══════════════════════════════════════════════════════════════
    st.header("🧮 Logistic Regression Formula (BTC Model)")
    st.markdown("Positive coefficient → higher feature value → higher P(Buy)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(lr_coef_df.style.background_gradient(cmap="RdYlGn", subset=["Coefficient"]),
                     use_container_width=True)
    with c2:
        st.code(lr_formula, language="text")

else:
    st.markdown("---")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.metric("Capital to Deploy", f"${capital_m}M", "Configurable in sidebar")
    col_l2.metric("Assets",            "BTC + ETH + USDT", "3-asset portfolio")
    col_l3.metric("Portfolio States",  "4",  "Full / BTC / ETH / Defensive")
    col_l4.metric("Models per Asset",  "5",  "LR + RF + GB + Ensemble + MLP")
    st.info("👆 Set your parameters in the **sidebar**, then click **Run Full Pipeline** to execute the full ETL → Dual-Model Training → 4-State Portfolio Backtest → Tomorrow's Signals workflow.")
    st.info("💡 **Tip:** After the pipeline runs, all sidebar sliders update the backtest **instantly** — no retraining needed!")
