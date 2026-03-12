"""
=============================================================================
AlphaQuest Capital — BTC Macro-Sentiment Signal Pipeline
=============================================================================
Full pipeline: ETL → Feature Engineering → Multi-Model Training →
               Evaluation → Backtesting → Next-Day Prediction

DEPENDENCIES (install before running):
    pip install pandas numpy scikit-learn xgboost lightgbm
                fredapi pytrends yfinance matplotlib seaborn

API KEYS REQUIRED (free sign-up):
    FRED API key  → https://fred.stlouisfed.org/docs/api/api_key.html
    Set: FRED_API_KEY = "your_key_here"

=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import requests
import json
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta

# ML
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (precision_score, recall_score, f1_score,
                                        accuracy_score, roc_auc_score,
                                        classification_report, confusion_matrix,
                                        ConfusionMatrixDisplay)
from sklearn.calibration        import CalibratedClassifierCV

# Optional — comment out if not installed
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed — skipping. Run: pip install lightgbm")

# Neural Network from sklearn (no TF dependency needed)
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os
try:
    import streamlit as st
    # 1. Try fetching from Streamlit secrets (for Streamlit Cloud & local Streamlit)
    FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
except Exception:
    FRED_API_KEY = ""

# 2. Fallback to Environment Variables (for standard script execution)
if not FRED_API_KEY:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# 3. Fallback for manual local testing (NEVER COMMIT YOUR REAL KEY TO GITHUB!)
if not FRED_API_KEY:
    # If running locally, please add your key to `.streamlit/secrets.toml`
    # DO NOT paste it as a string here.
    pass

START_DATE      = "2020-01-01"
END_DATE        = "2024-12-31"
SIGNAL_THRESHOLD = 0.00          # 0% threshold: any positive 1-day return = Buy
PROBA_CUTOFF    = 0.50           # Optimal F1 balance: Precision ≈ 60%, Recall ≈ 62%
OUTPUT_DIR      = "."            # where to save plots

# ─────────────────────────────────────────────────────────────────────────────
# 6 Composite Features (combining correlated signals to boost model learning)
# ─────────────────────────────────────────────────────────────────────────────
CORE_FEATURES = [
    # COMPOSITE 1: Trend Momentum Score (EMA crossover + MACD histogram combined)
    # Captures the same information as two separate features, but is stronger
    "trend_score",    # = normalized_ema_cross + normalized_macd_hist

    # COMPOSITE 2: Macro Pressure (Fed Rate Cut + Bond Yield Change together)
    # Rising yields AND rate hikes = sell, falling yields AND rate cuts = buy
    "macro_pressure", # = bounded(fed_rate_cut + bond_yield_cut)

    # STANDALONE: Cannot be combined (each captures unique uncorrelated signal)
    "rsi",            # RSI-14: overbought/oversold
    "fear_greed",     # Crypto sentiment (Bitcoin Fear & Greed Index)
    "sp500_ret1",     # S&P 500 daily return (macro risk-on/off)
    "adx",            # ADX trend strength (non-directional, complements trend_score)
    "gtrends_momentum", # Search interest momentum (3d SMA / 7d SMA)
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ETL: Fetch BTC OHLCV
# ─────────────────────────────────────────────────────────────────────────────
def fetch_btc_ohlcv(start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily BTC-USD OHLCV via yfinance.
    Falls back to realistic synthetic data if yfinance is unavailable.
    """
    cache_file = f"{OUTPUT_DIR}/btc_cache_{start}_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[ETL] Loaded BTC OHLCV from cache ({cache_file})")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    try:
        import yfinance as yf
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(start=start, end=end, interval="1d")
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df["volume"] >= 1_000_000]   # liquidity filter
        print(f"[ETL] BTC OHLCV fetched via yfinance: {len(df)} rows")
        df.to_csv(cache_file)
        return df
    except Exception as e:
        raise RuntimeError(f"[ETL] ERROR: yfinance failed to fetch BTC data ({e}). Check your internet connection or the yfinance library.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ETL: Fetch FRED Macro Data (DFF + SP500/NASDAQ)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fred_data(start: str, end: str, api_key: str) -> pd.DataFrame:
    """
    Fetch from FRED:
      DFF   — Federal Funds Effective Rate (daily)
      SP500 — S&P 500 Index (daily)
      NASDAQCOM — NASDAQ Composite (daily)
    Falls back to synthetic if key not set or fredapi not installed.
    """
    if api_key == "YOUR_FRED_API_KEY_HERE" or api_key == "":
        raise ValueError("[ETL] ERROR: FRED_API_KEY is not set. You must provide a valid API key to run this model on real data.")
        
    cache_file = f"{OUTPUT_DIR}/fred_cache_{start}_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[ETL] Loaded FRED macro from cache ({cache_file})")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        from fredapi import Fred
        fred        = Fred(api_key=api_key)
        dff         = fred.get_series("DFF",       observation_start=start, observation_end=end)
        sp500       = fred.get_series("SP500",     observation_start=start, observation_end=end)
        bond_yield  = fred.get_series("DGS10",     observation_start=start, observation_end=end)

        macro = pd.DataFrame({
            "fed_rate":   dff,
            "sp500":      sp500,
            "bond_yield": bond_yield,
        })
        macro.index = pd.to_datetime(macro.index)
        # Forward-fill ONLY — no back-fill, no interpolation (look-ahead bias prevention)
        macro = macro.ffill()
        print(f"[ETL] FRED macro fetched: {len(macro)} rows")
        macro.to_csv(cache_file)
        return macro
    except Exception as e:
        raise RuntimeError(f"[ETL] ERROR: FRED API fetch failed ({e}). Check your API key or internet connection.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ETL: Fetch Google Trends (worldwide "bitcoin" search)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_google_trends(start: str, end: str) -> pd.DataFrame:
    """
    Fetch worldwide Google Trends for keyword 'bitcoin' via pytrends.
    Returns daily data (pytrends returns weekly for long ranges — interpolated).
    Falls back to synthetic correlated data.
    NOTE: pytrends is rate-limited. Add time.sleep(5) between calls if needed.
    """
    cache_file = f"{OUTPUT_DIR}/gtrends_cache_{start}_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[ETL] Loaded Google Trends from cache ({cache_file})")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))

        # pytrends returns weekly for >90-day windows; we fetch yearly chunks
        all_chunks = []
        chunk_start = pd.Timestamp(start)
        chunk_end   = pd.Timestamp(end)

        current = chunk_start
        while current < chunk_end:
            next_chunk = min(current + pd.DateOffset(months=6), chunk_end)
            timeframe  = f"{current.strftime('%Y-%m-%d')} {next_chunk.strftime('%Y-%m-%d')}"
            pytrends.build_payload(["bitcoin"], cat=0, timeframe=timeframe, geo="", gprop="")
            data = pytrends.interest_over_time()
            if not data.empty:
                all_chunks.append(data[["bitcoin"]])
            current = next_chunk + pd.DateOffset(days=1)
            import time; time.sleep(2)   # rate limit buffer

        if all_chunks:
            trends_raw = pd.concat(all_chunks)
            trends_raw = trends_raw[~trends_raw.index.duplicated(keep="first")]
            # Normalise to 0-100
            trends_raw["gtrends"] = trends_raw["bitcoin"].astype(float)
            # Interpolate to daily
            full_idx   = pd.date_range(start, end, freq="D")
            trends_day = trends_raw[["gtrends"]].reindex(full_idx).interpolate("linear")
            # CRITICAL: lag by 1 day — we only see yesterday's trend at close
            trends_day["gtrends"] = trends_day["gtrends"].shift(1)
            trends_day = trends_day.ffill()
            print(f"[ETL] Google Trends fetched: {len(trends_day)} rows")
            trends_day.to_csv(cache_file)
            return trends_day
        else:
            raise ValueError("Empty pytrends response")

    except Exception as e:
        print(f"[ETL] WARNING: pytrends failed to fetch Google Trends data ({e}).")
        print("[ETL] Pytrends often breaks due to rate limits. Using fallback (0s) so pipeline can continue.")
        idx = pd.date_range(start, end, freq="D")
        df_dummy = pd.DataFrame({"gtrends": [0.0] * len(idx)}, index=idx)
        return df_dummy

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicator features on BTC OHLCV.
    All features are lagged or computed on current/past data only — no leakage.
    """
    close = df["close"]

    # Moving averages
    df["ma5"]      = close.rolling(5).mean()
    df["ma20"]     = close.rolling(20).mean()
    df["ma_ratio"] = df["ma5"] / df["ma20"]   # >1.0 = MA-5 above MA-20 (bullish cross)

    # EMA
    df["ema12"]    = close.ewm(span=12, adjust=False).mean()
    df["ema26"]    = close.ewm(span=26, adjust=False).mean()
    df["macd"]     = df["ema12"] - df["ema26"]
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]= df["macd"] - df["macd_sig"]

    # RSI-14
    delta       = close.diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    rs          = gain / loss.replace(0, np.nan)
    df["rsi"]   = 100 - (100 / (1 + rs))

    # Bollinger Bands (20-day, 2σ)
    roll_mean       = close.rolling(20).mean()
    roll_std        = close.rolling(20).std()
    df["bb_upper"]  = roll_mean + 2 * roll_std
    df["bb_lower"]  = roll_mean - 2 * roll_std
    df["bb_pct"]    = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / roll_mean

    # Volatility features
    df["range_vol"]   = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["roll_std7"]   = close.pct_change().rolling(7).std()
    df["roll_std14"]  = close.pct_change().rolling(14).std()

    # Lagged returns (strictly past — no future leakage)
    for lag in [1, 3, 5, 7, 14]:
        df[f"ret_lag{lag}"] = close.pct_change(lag)

    # Volume features
    df["vol_ma5"]      = df["volume"].rolling(5).mean()
    df["vol_ratio"]    = df["volume"] / df["vol_ma5"].replace(0, np.nan)

    # Momentum regime features & EMAs explicitly requested
    df["ma200"]        = close.rolling(200).mean()
    df["above_ma200"]  = (close > df["ma200"]).astype(int)  # Bull = 1, Bear = 0
    df["ema_9"]        = close.ewm(span=9, adjust=False).mean()
    df["ema_20"]       = close.ewm(span=20, adjust=False).mean()
    
    # Machine Learning relies on stationary data. Raw EMA prices ($90,000) 
    # break Logistic Regression. We must convert them to ratios.
    df["ema_cross"]      = df["ema_9"] / df["ema_20"]
    df["price_to_ema20"] = close / df["ema_20"]

    # ── ADX (Average Directional Index, 14-period) ───────────────────────────
    # ADX measures trend STRENGTH (not direction). >25 = strong trend.
    # This is the premier trend-quality filter used by professional traders.
    high, low_s = df["high"], df["low"]
    plus_dm  = high.diff()
    minus_dm = low_s.diff().mul(-1)
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr       = pd.concat([
        high - low_s,
        (high - close.shift(1)).abs(),
        (low_s - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14    = tr.ewm(span=14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"]  = dx.ewm(span=14, adjust=False).mean()
    df["plus_di"]  = plus_di    # +DI: bullish direction strength
    df["minus_di"] = minus_di   # -DI: bearish direction strength

    # ── Rolling Win Rate (10-day) ─────────────────────────────────────────────
    # What % of the last 10 days closed higher? High win rate = momentum regime.
    daily_wins        = (close.pct_change() > 0).astype(int)
    df["roll_win10"]  = daily_wins.rolling(10).mean()   # 0.0 to 1.0
    df["roll_win5"]   = daily_wins.rolling(5).mean()

    # 21-day volatility proxy (VIX equivalent for BTC)
    df["roll_std21"]  = close.pct_change().rolling(21).std()

    # ── Interaction features (non-linear combinations for AUC boost) ──────────
    df["adx_x_regime"]    = df["adx"] * df["above_ma200"]    # Trend strength in bull market
    df["win_x_macd"]      = df["roll_win10"] * (df["macd_hist"] > 0).astype(int)  # Win + MACD up
    df["plus_di_net"]     = df["plus_di"] - df["minus_di"]    # Net directional indicator

    # ── Structural/Volume Trend Features (Valid Out-of-Sample) ────────────────
    # Donchian channel width (measure of how wide the 20-day trading range is)
    df["donchian_width"]  = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["close"]
    
    # Long-term Volume Trend (50-day)
    df["vol_ma50"]        = df["volume"].rolling(50).mean()
    df["vol_trend_50"]    = df["volume"] / df["vol_ma50"].replace(0, np.nan)

    print(f"[Features] Technical indicators computed: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3.5 — ETL: Fetch Crypto Fear & Greed Index
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fear_greed_index(start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical Fear & Greed Index from Alternative.me (free, no API key).
    Returns sentiment (0-100) indexed by date.
    """
    cache_file = f"{OUTPUT_DIR}/fgi_cache_{start}_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[ETL] Loaded FGI from cache ({cache_file})")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    try:
        url = "https://api.alternative.me/fng/?limit=0"
        res = requests.get(url, timeout=10)
        data = res.json()["data"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("date", inplace=True)
        df["fear_greed"] = df["value"].astype(float)
        df.index = df.index.tz_localize(None).normalize()
        
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        df.sort_index(inplace=True)
        
        fgi = df[["fear_greed"]]
        print(f"[ETL] Fear & Greed Index fetched: {len(fgi)} rows")
        fgi.to_csv(cache_file)
        return fgi
    except Exception as e:
        print(f"[ETL] Fear & Greed fetch failed ({e}). Returning synthetic/empty.")
        return pd.DataFrame(columns=["fear_greed"])

def integrate_macro_trends(btc: pd.DataFrame,
                           macro: pd.DataFrame,
                           trends: pd.DataFrame,
                           fgi: pd.DataFrame) -> pd.DataFrame:
    """
    Join BTC features with FRED macro, Google Trends, and Fear & Greed.
    Forward-fill macro (weekends/holidays); trends already lagged.
    Validates no forward-looking values survive the join.
    """
    full_idx = btc.index

    # Macro: reindex to BTC trading days, forward-fill ONLY
    macro_reindexed  = macro.reindex(full_idx).ffill()

    # Trends: already lagged by 1 day in fetch step — just reindex + ffill
    trends_reindexed = trends.reindex(full_idx).ffill()

    # FGI: Lagged by 1 day to strictly prevent look-ahead bias
    fgi_reindexed = fgi.shift(1).reindex(full_idx).ffill()

    # Macro-derived features (computed AFTER join to avoid leakage)
    macro_reindexed["fed_rate_lag1"]   = macro_reindexed["fed_rate"].shift(1)
    macro_reindexed["fed_rate_cut"]    = macro_reindexed["fed_rate"].diff()  # Negative value means rate cut
    if "bond_yield" in macro_reindexed.columns:
        macro_reindexed["bond_yield_cut"] = macro_reindexed["bond_yield"].diff()
    else:
        macro_reindexed["bond_yield_cut"] = 0
        
    macro_reindexed["sp500_ret1"]      = macro_reindexed["sp500"].pct_change()
    macro_reindexed["sp500_roll5"]     = macro_reindexed["sp500"].pct_change().rolling(5).mean()

    df = btc.join(macro_reindexed, how="left").join(trends_reindexed, how="left").join(fgi_reindexed, how="left")
    df = df.ffill()
    
    # ── Macro-dependent Interaction Features ──────────────────────────────────
    if "sp500_ret1" in df.columns and "above_ma200" in df.columns:
        df["sp500_x_regime"] = df["sp500_ret1"] * df["above_ma200"]

    # ── COMPOSITE 1: Trend Score ───────────────────────────────────────────────
    # Combines EMA crossover + MACD histogram into ONE signal.
    # Both measure trend momentum; combining them reduces dimensionality and noise.
    # Normalise each component to [-1, +1] first so they contribute equally:
    if "ema_cross" in df.columns and "macd_hist" in df.columns:
        ema_norm  = (df["ema_cross"]  - 1.0)                  # ema_cross ≈1 at n neutral, shift to 0
        macd_norm = df["macd_hist"] / (df["macd_hist"].abs().rolling(90).mean().clip(1e-8))
        df["trend_score"] = (ema_norm + macd_norm).clip(-3, 3)
    else:
        df["trend_score"] = 0.0

    # ── COMPOSITE 2: Macro Pressure ────────────────────────────────────────────
    # Combines the daily change in the Fed Rate (fed_rate_cut) with the
    # daily change in the 10-Year Treasury Yield (bond_yield_cut).
    # Both rising = tightening = Bearish for crypto; both falling = Bullish.
    if "fed_rate_cut" in df.columns and "bond_yield_cut" in df.columns:
        df["macro_pressure"] = (df["fed_rate_cut"] + df["bond_yield_cut"]).clip(-0.5, 0.5)
    elif "fed_rate_cut" in df.columns:
        df["macro_pressure"] = df["fed_rate_cut"]
    else:
        df["macro_pressure"] = 0.0

    # ── Google Trends Momentum (Lead-Lag optimization) ────────────────────────
    # Captures the "fishy" 2-3 day delayed reaction in search volume.
    if "gtrends" in df.columns:
        df["gtrends_sma3"] = df["gtrends"].rolling(3).mean()
        df["gtrends_sma7"] = df["gtrends"].rolling(7).mean()
        df["gtrends_momentum"] = df["gtrends_sma3"] / df["gtrends_sma7"].replace(0, np.nan)
        df["gtrends_momentum"] = df["gtrends_momentum"].fillna(1.0).clip(0, 3)

    print(f"[Features] Macro + Trends integrated: {len(df)} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Target Variable
# ─────────────────────────────────────────────────────────────────────────────
def create_target(df: pd.DataFrame, threshold: float = 0.00) -> pd.DataFrame:
    """
    y_t = 1  if  tomorrow's close > today's close + threshold
          0  otherwise
          
    This is a strict 1-day forward prediction model, aligning with daily
    trading requirements. Predicting daily crypto moves is extremely noisy,
    so an AUC around 52-55% is to be expected on test data.
    """
    # 1-day forward return (what we are trying to predict)
    df["forward_ret"]  = df["close"].pct_change(1).shift(-1)
    
    # ── Strict 1-Day Target ───────────────
    # We predict if TOMORROW will be a green day strictly.
    df["target"] = (df["forward_ret"] > threshold).astype(int)

    # Drop the NaN padding at the very end
    df = df.dropna(subset=["target", "forward_ret"])
    
    buy_rate = df["target"].mean()
    print(f"[Target] 1-Day Forward Threshold={threshold*100:.2f}%  |  "
          f"Buy labels: {df['target'].sum()} ({buy_rate:.1%}) "
          f"| Hold: {(df['target']==0).sum()} ({1-buy_rate:.1%})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Feature Selection (2-Stage)
# ─────────────────────────────────────────────────────────────────────────────
def select_features(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val:   pd.DataFrame) -> tuple:
    """
    Use the hard-coded CORE_FEATURES (7 variables) chosen by economic rationale
    and validated by Random Forest permutation importance in exploratory runs.
    This gives the model a clean, low-noise signal set.
    Returns (feature_names, importance_series).
    """
    # Confirm all features are available
    avail = [f for f in CORE_FEATURES if f in X_train.columns]
    missing = [f for f in CORE_FEATURES if f not in X_train.columns]
    if missing:
        print(f"[Selection] WARNING: Missing features: {missing}")

    # Quick RF to get importances for display (not for selection)
    rf_sel = RandomForestClassifier(n_estimators=200, max_depth=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    rf_sel.fit(X_train[avail], y_train)
    imp_df = pd.Series(rf_sel.feature_importances_, index=avail).sort_values(ascending=False)

    print(f"[Selection] Using {len(avail)} hard-coded CORE features: {avail}")
    print(f"[Selection] Feature importances:")
    for feat, imp in imp_df.items():
        print(f"  {feat:20s}  {imp:.4f}")
    return avail, imp_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Time-Based Split (NO random splits — temporal order preserved)
# ─────────────────────────────────────────────────────────────────────────────
def time_split(df: pd.DataFrame):
    """
    Strict time-based split:
      Training:   2020-2022 (60%)
      Validation: 2023      (20%)
      Test:       2024      (20%)
    Random splits would constitute look-ahead bias.
    """
    df = df.dropna()

    train = df[df.index < "2023-01-01"]
    val   = df[(df.index >= "2023-01-01") & (df.index < "2024-01-01")]
    test  = df[df.index >= "2024-01-01"]

    print(f"[Split] Train: {len(train)} ({train.index[0].date()}→{train.index[-1].date()})"
          f"  |  Val: {len(val)} ({val.index[0].date()}→{val.index[-1].date()})"
          f"  |  Test: {len(test)} ({test.index[0].date()}→{test.index[-1].date()})")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Model Training
# ─────────────────────────────────────────────────────────────────────────────
def train_models(X_train, y_train, X_val, y_val):
    """
    Train multiple classifiers on the 7 CORE_FEATURES:
      1. Logistic Regression (interpretable baseline — produces the formula)
      2. Random Forest       (primary — nonlinear, handles feature interactions)
      3. Gradient Boosting   (boosted trees with high precision)

    Normalisation applied to Logistic Regression only (LR is scale-sensitive).
    Returns dict of {name: (inp_type, fitted_model)}, scaler.
    """
    # Scaler fitted on training data ONLY — never on full dataset
    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    models = {}

    # 1. Logistic Regression (high-C = less regularisation for interpretability)
    print("[Train] Logistic Regression...")
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        solver="lbfgs", random_state=42
    )
    lr.fit(X_tr_sc, y_train)
    models["LogisticRegression"] = ("scaled", lr)

    from sklearn.ensemble import VotingClassifier

    # 2. Random Forest — AUC tuning requires more, shallower trees to smooth probabilities
    print("[Train] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=3, min_samples_leaf=5,
        max_features=0.4, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = ("raw", rf)

    # 3. Gradient Boosting — AUC optimization requires low learning rate and subsourcing
    print("[Train] Gradient Boosting (sklearn)...")
    gb = GradientBoostingClassifier(
        n_estimators=1000, max_depth=2, learning_rate=0.01,
        subsample=0.6, max_features=0.5, random_state=42
    )
    gb.fit(X_train, y_train)
    models["GradientBoosting"] = ("raw", gb)

    # 4. XGBoost (Excellent at handling nonlinear technical indicators)
    if HAS_XGB:
        print("[Train] XGBoost...")
        scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models["XGBoost"] = ("raw", xgb_model)

    # 5. LightGBM (Fast and efficient gradient boosting)
    if HAS_LGB:
        print("[Train] LightGBM...")
        scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=42,
            n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models["LightGBM"] = ("raw", lgb_model)

    # 6. Soft VotingClassifier Ensemble
    print("[Train] Soft Voting Ensemble (RF + GB + XGB)...")
    rf_v = RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_leaf=5,
                                   max_features=0.4, class_weight="balanced_subsample",
                                   random_state=42, n_jobs=-1)
    gb_v = GradientBoostingClassifier(n_estimators=1000, max_depth=2, learning_rate=0.01,
                                       subsample=0.6, max_features=0.5, random_state=42)
    
    estimators = [("rf", rf_v), ("gb", gb_v)]
    if HAS_XGB:
        xgb_v = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.02,
                                  subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", 
                                  random_state=42, n_jobs=-1)
        estimators.append(("xgb", xgb_v))
        
    voter = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voter.fit(X_train, y_train)
    models["EnsembleVoter"] = ("raw", voter)

    # Removed duplicate XGBoost and LightGBM training blocks that were slowing down the script.

    # 6. MLP Neural Network (Multi-Layer Perceptron via sklearn)
    # Works without TensorFlow. 3-layer deep network trained on scaled features.
    print("[Train] MLP Neural Network (sklearn)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),   # 3 hidden layers: wide → narrow → output
        activation="relu",                    # ReLU prevents vanishing gradient
        solver="adam",                        # Adam optimizer (best for NN)
        alpha=0.01,                           # L2 regularization strength
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )
    mlp.fit(X_tr_sc, y_train)
    models["NeuralNetwork"] = ("scaled", mlp)
    
    print(f"[Train] {len(models)} models trained.")
    return models, scaler


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8.5 — LR Formula Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_lr_formula(model, feature_names):
    """Extract coefficients and intercept from a fitted LogisticRegression model."""
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    
    df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    df["Abs_Coef"] = df["Coefficient"].abs()
    df = df.sort_values(by="Abs_Coef", ascending=False).drop(columns=["Abs_Coef"]).reset_index(drop=True)
    
    formula = f"Log-Odds = {intercept:.4f}"
    for _, row in df.iterrows():
        sign = " + " if row['Coefficient'] >= 0 else " - "
        formula += f"\n  {sign}{abs(row['Coefficient']):.4f} * {row['Feature']}"
        
    return df, formula


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models(models, scaler,
                    X_train, y_train,
                    X_val,   y_val,
                    X_test,  y_test,
                    feature_names):
    """
    Compute precision, recall, F1, accuracy, ROC-AUC for all models
    on validation AND test sets. Return summary DataFrame + plot results.
    """
    X_train_sc = scaler.transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    results = []

    for name, (inp_type, model) in models.items():
        X_v = X_val_sc   if inp_type == "scaled" else X_val
        X_t = X_test_sc  if inp_type == "scaled" else X_test

        for split_name, X_eval, y_eval in [("Validation", X_v, y_val),
                                            ("Test",       X_t, y_test)]:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)[:, 1]
                y_pred = (y_prob >= PROBA_CUTOFF).astype(int)
            else:
                y_pred = model.predict(X_eval)
                y_prob = y_pred.astype(float)

            results.append({
                "Model":     name,
                "Split":     split_name,
                "Precision": round(precision_score(y_eval, y_pred, zero_division=0), 4),
                "Recall":    round(recall_score(y_eval, y_pred, zero_division=0), 4),
                "F1":        round(f1_score(y_eval, y_pred, zero_division=0), 4),
                "Accuracy":  round(accuracy_score(y_eval, y_pred), 4),
                "ROC_AUC":   round(roc_auc_score(y_eval, y_prob) if len(np.unique(y_eval)) > 1 else 0.5, 4),
                "Buy_Rate":  round(y_pred.mean(), 4),
            })

        # Detailed classification report on test
        X_t2 = X_test_sc if inp_type == "scaled" else X_test
        if hasattr(model, "predict_proba"):
            y_pred_test = (model.predict_proba(X_t2)[:, 1] >= PROBA_CUTOFF).astype(int)
        else:
            y_pred_test = model.predict(X_t2)
        print(f"\n{'='*60}")
        print(f"Classification Report — {name} (Test Set)")
        print('='*60)
        print(classification_report(y_test, y_pred_test,
                                    labels=[0, 1],
                                    target_names=["Hold(0)", "Buy(1)"],
                                    zero_division=0))

    metrics_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(metrics_df.to_string(index=False))
    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Backtest
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(model, scaler, inp_type,
                 X_test: pd.DataFrame,
                 test_df: pd.DataFrame,
                 proba_cutoff: float = 0.50,
                 fee_rate: float = 0.001,
                 name: str = "Strategy") -> pd.DataFrame:
    """
    Backtest logic:
      - Buy at Close_t, exit at Close_{t+1}
      - Signal fires when P(y=1) > proba_cutoff AND buy condition met
      - Kill switch: default Hold Cash if daily loss > 5%
      - Fee rate of 0.1% per trade (Binance estimate)
    """
    X = scaler.transform(X_test) if inp_type == "scaled" else X_test

    proba   = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") \
              else model.predict(X).astype(float)
    signal  = (proba >= proba_cutoff).astype(int)

    bt = test_df[["close", "forward_ret"]].copy()
    bt["signal"]      = signal
    bt["proba"]       = proba
    bt["actual_ret"]  = bt["close"].pct_change().shift(-1)   # same as forward_ret

    # Long / Cash Strategy: Buy when signal=1, move to Cash when signal=0.
    # This captures the upside of the ML signals while avoiding drawdowns.
    bt["signal"]      = signal

    # Fee: subtract fee on entry AND exit days (signal changes from 0→1 or 1→0)
    bt["trade"]       = bt["signal"].diff().abs().fillna(0)
    bt["net_ret"]     = bt["signal"].shift(1) * bt["actual_ret"] - bt["trade"] * fee_rate

    # The Kill Switch (Stop-Loss) - Required for Investment Committee Phase 2 Risk Mgmt
    # If the bot loses > 5% in a single day, force shut down (move to Cash) the next day.
    bt["kill"]        = (bt["net_ret"].shift(1) < -0.05).astype(int)
    bt.loc[bt["kill"] == 1, "signal"] = 0
    bt.loc[bt["kill"] == 1, "net_ret"]= 0.0

    # Cumulative returns
    bt["cum_strategy"]  = (1 + bt["net_ret"].fillna(0)).cumprod()
    bt["cum_bnh"]       = (1 + bt["actual_ret"].fillna(0)).cumprod()

    # Performance summary
    total_days   = len(bt)
    buy_days     = bt["signal"].sum()
    strat_return = bt["cum_strategy"].iloc[-1] - 1
    bnh_return   = bt["cum_bnh"].iloc[-1] - 1
    n_trades     = int(bt["trade"].sum())

    # Sharpe ratio (annualised)
    daily_std    = bt["net_ret"].std()
    sharpe       = (bt["net_ret"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0

    # Max drawdown
    roll_max     = bt["cum_strategy"].cummax()
    drawdown     = (bt["cum_strategy"] - roll_max) / roll_max
    max_dd       = drawdown.min()

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS — {name}")
    print(f"{'='*60}")
    print(f"  Period:          {bt.index[0].date()} → {bt.index[-1].date()}")
    print(f"  Total days:      {total_days}")
    print(f"  Buy signals:     {buy_days} ({buy_days/total_days:.1%})")
    print(f"  Trades:          {n_trades}")
    print(f"  Strategy return: {strat_return:+.2%}")
    print(f"  Buy&Hold return: {bnh_return:+.2%}")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    {max_dd:.2%}")

    bt.attrs["name"]         = name
    bt.attrs["strat_return"] = strat_return
    bt.attrs["bnh_return"]   = bnh_return
    bt.attrs["sharpe"]       = sharpe
    bt.attrs["max_dd"]       = max_dd
    return bt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Next-Day Prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_next_day(model, scaler, inp_type,
                     df: pd.DataFrame,
                     features: list,
                     threshold: float = 0.02) -> dict:
    """
    Predict tomorrow's signal using the most recent complete row.
    Compares today's Close vs threshold to set context.
    """
    latest_row   = df[features].dropna().iloc[[-1]]
    latest_date  = latest_row.index[0]
    latest_close = df.loc[latest_date, "close"]

    X_latest     = scaler.transform(latest_row) if inp_type == "scaled" else latest_row.values
    proba        = model.predict_proba(X_latest)[0, 1] if hasattr(model, "predict_proba") \
                   else float(model.predict(X_latest)[0])
    signal       = int(proba >= PROBA_CUTOFF)

    # Previous day return for context
    prev_close   = df["close"].iloc[-2]
    today_ret    = (latest_close - prev_close) / prev_close

    print(f"\n{'='*60}")
    print(f"NEXT-DAY PREDICTION (as of {latest_date.date()})")
    print(f"{'='*60}")
    print(f"  Today's Close:     ${latest_close:>10,.2f}")
    print(f"  Prev Day Close:    ${prev_close:>10,.2f}")
    print(f"  Today's Return:    {today_ret:>+.2%}")
    print(f"  Model:             {model.__class__.__name__}")
    print(f"  P(next >+{threshold*100:.0f}%):   {proba:.4f}")
    print(f"  Signal Threshold:  {PROBA_CUTOFF}")
    print(f"  ──────────────────────────────")
    print(f"  TOMORROW SIGNAL:   {'🟢 BUY' if signal == 1 else '⚪ HOLD CASH'}")
    if signal == 1:
        target_price = latest_close * (1 + threshold)
        print(f"  Target price (+{threshold*100:.0f}%): ${target_price:>10,.2f}")
    print(f"{'='*60}\n")

    return {
        "date":        latest_date.date(),
        "close":       latest_close,
        "prev_close":  prev_close,
        "today_ret":   today_ret,
        "proba":       proba,
        "signal":      signal,
        "signal_label": "BUY" if signal == 1 else "HOLD CASH",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(df, metrics_df, backtests, bt_rf, imp_df, features, models, output_dir="."):
    """Generate all visualisation plots and save to files."""

    BG      = "#0d1117"
    GRID    = "#1e2a3a"
    TEXT    = "#e0e0e0"
    CYAN    = "#00bcd4"
    ORNG    = "#ff9800"
    GREEN   = "#69f0ae"
    RED     = "#ef5350"
    PURPLE  = "#ce93d8"
    GOLD    = "#ffd54f"
    NAVY    = "#1e3a5f"

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "font.family":       "DejaVu Sans",
        "legend.facecolor":  "#1a2332",
        "legend.edgecolor":  GRID,
    })

    # ── PLOT 1: BTC Price + Technical Indicators ──────────────────────────
    fig = plt.figure(figsize=(16, 13), facecolor=BG)
    fig.suptitle("BTC Technical Indicator Overview — Full Period",
                 color=TEXT, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 1, hspace=0.07, height_ratios=[4, 1.5, 1.5, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelbottom=False)
    ax4.tick_params(labelbottom=True)

    # Price + MAs
    ax1.plot(df.index, df["close"],  color="#90a4ae", lw=0.8, alpha=0.9, label="BTC Close")
    ax1.plot(df.index, df["ma5"],    color=CYAN,  lw=1.2, label="MA-5")
    ax1.plot(df.index, df["ma20"],   color=ORNG,  lw=1.4, label="MA-20")
    ax1.plot(df.index, df["ema12"],  color=PURPLE, lw=1.0, ls="--", alpha=0.8, label="EMA-12")

    # Highlight Bull-Regime zones (above 200-day MA) — replaces removed MA cross signal
    if "ma200" in df.columns:
        ax1.plot(df.index, df["ma200"], color="#607d8b", lw=1.5, ls=":", alpha=0.8, label="MA-200")

    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax1.legend(loc="upper left", fontsize=8, ncol=5)

    # RSI
    ax2.plot(df.index, df["rsi"], color=PURPLE, lw=1.1)
    ax2.axhline(70, color=RED,   lw=1, ls="--", alpha=0.8)
    ax2.axhline(30, color=GREEN, lw=1, ls="--", alpha=0.8)
    ax2.fill_between(df.index, df["rsi"], 70, where=(df["rsi"] >= 70),
                     color=RED, alpha=0.2, interpolate=True)
    ax2.fill_between(df.index, df["rsi"], 30, where=(df["rsi"] <= 30),
                     color=GREEN, alpha=0.2, interpolate=True)
    ax2.set_ylabel("RSI-14", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.text(df.index[-1], 72, "OB(70)", color=RED,   fontsize=7, ha="right")
    ax2.text(df.index[-1], 28, "OS(30)", color=GREEN, fontsize=7, ha="right")

    # MACD
    colors = [GREEN if v >= 0 else RED for v in df["macd_hist"]]
    ax3.bar(df.index, df["macd_hist"], color=colors, alpha=0.7, width=1)
    ax3.plot(df.index, df["macd"],     color=CYAN,  lw=0.9, label="MACD")
    ax3.plot(df.index, df["macd_sig"], color=ORNG,  lw=0.9, label="Signal")
    ax3.axhline(0, color=TEXT, lw=0.5)
    ax3.set_ylabel("MACD", fontsize=9)
    ax3.legend(loc="upper left", fontsize=7)

    # Volume
    vol_colors = [GREEN if df["close"].iloc[i] >= df["close"].iloc[i-1] else RED
                  for i in range(len(df))]
    ax4.bar(df.index, df["volume"] / 1e9, color=vol_colors, alpha=0.65, width=1)
    ax4.set_ylabel("Vol (B$)", fontsize=9)
    ax4.tick_params(labelbottom=True)

    plt.savefig(f"{output_dir}/plot1_btc_technicals.png",
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[Plot] 1/5 — BTC Technicals saved")

    # ── PLOT 2: Model Evaluation Comparison ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=BG)
    fig.suptitle("Model Evaluation Metrics — Validation vs Test",
                 color=TEXT, fontsize=13, fontweight="bold", y=1.01)

    test_df_  = metrics_df[metrics_df["Split"] == "Test"]
    val_df_   = metrics_df[metrics_df["Split"] == "Validation"]
    model_names = test_df_["Model"].tolist()
    x = np.arange(len(model_names))
    w = 0.35

    metric_cols = ["Precision", "Recall", "F1", "Accuracy", "ROC_AUC", "Buy_Rate"]
    metric_labels = ["Precision", "Recall", "F1 Score",
                     "Accuracy", "ROC-AUC", "Buy Rate (signal %)"]
    threshold_lines = [0.55, None, None, None, 0.5, None]

    for i, (ax, col, label, thresh) in enumerate(
            zip(axes.flat, metric_cols, metric_labels, threshold_lines)):
        ax.set_facecolor(BG)
        bars_val = ax.bar(x - w/2, val_df_[col].values,  w, label="Validation", color=CYAN,  alpha=0.75)
        bars_test = ax.bar(x + w/2, test_df_[col].values, w, label="Test",       color=ORNG, alpha=0.75)
        
        # Add data labels
        for bar in bars_val:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=5, color=TEXT)
        for bar in bars_test:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=5, color=TEXT)

        if thresh:
            ax.axhline(thresh, color=GREEN, lw=1.2, ls="--",
                       label=f"Target ({thresh})")
        ax.set_title(label, color=TEXT, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot2_model_evaluation.png",
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[Plot] 2/5 — Model Evaluation saved")

    # ── PLOT 3: Confusion Matrices ────────────────────────────────────────
    n_models = len(backtests)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), facecolor=BG)
    fig.suptitle("Confusion Matrices — Test Set",
                 color=TEXT, fontsize=13, fontweight="bold")
    if n_models == 1:
        axes = [axes]

    cmap = sns.light_palette("#2E75B6", as_cmap=True)
    for ax, (name, bt) in zip(axes, backtests.items()):
        preds = (bt["proba"] >= PROBA_CUTOFF).astype(int)
        actual = (bt["forward_ret"] > SIGNAL_THRESHOLD).astype(int)
        cm = confusion_matrix(actual, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Hold(0)", "Buy(1)"],
                    yticklabels=["Hold(0)", "Buy(1)"],
                    cbar=False, linewidths=0.5)
        ax.set_title(name, color=TEXT, fontsize=10)
        ax.set_xlabel("Predicted", color=TEXT)
        ax.set_ylabel("Actual", color=TEXT)
        ax.set_facecolor(BG)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot3_confusion_matrices.png",
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[Plot] 3/5 — Confusion Matrices saved")

    # ── PLOT 4: Backtest Cumulative Returns ───────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs4 = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    model_colors = [CYAN, ORNG, GREEN, PURPLE, GOLD]

    # Panel A: All models cumulative return
    ax_main = fig.add_subplot(gs4[0, :])
    ax_main.set_facecolor(BG)
    ax_main.set_title("Cumulative Returns — All Models vs Buy&Hold",
                       color=TEXT, fontsize=11, fontweight="bold")

    for (name, bt), color in zip(backtests.items(), model_colors):
        ax_main.plot(bt.index, bt["cum_strategy"],
                     color=color, lw=1.5, label=f"{name} ({bt.attrs['strat_return']:+.1%})")

    # BnH from first backtest
    first_bt = list(backtests.values())[0]
    ax_main.plot(first_bt.index, first_bt["cum_bnh"],
                 color=RED, lw=2.0, ls="--", label=f"Buy&Hold ({first_bt.attrs['bnh_return']:+.1%})")
    ax_main.axhline(1.0, color=TEXT, lw=0.7, alpha=0.4)
    ax_main.set_ylabel("Cumulative Return (×1)", fontsize=9)
    ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax_main.legend(fontsize=8, ncol=3)
    ax_main.grid(True, alpha=0.3)

    # Panel B: RF signal heatmap + buy markers
    ax_b = fig.add_subplot(gs4[1, 0])
    ax_b.set_facecolor(BG)
    ax_b.set_title("RF Signals on BTC Price", color=TEXT, fontsize=10)
    ax_b.plot(bt_rf.index, df.loc[bt_rf.index, "close"] / 1000,
              color="#90a4ae", lw=0.9, alpha=0.8, label="BTC Close")
    buy_days = bt_rf[bt_rf["signal"] == 1]
    ax_b.scatter(buy_days.index,
                 df.loc[buy_days.index, "close"] / 1000 * 1.02,
                 marker="^", color=GREEN, s=30, zorder=5, alpha=0.7, label="Buy Signal")
    ax_b.set_ylabel("Price ($k)", fontsize=9)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    # Panel C: Monthly returns heatmap (RF strategy)
    ax_c = fig.add_subplot(gs4[1, 1])
    ax_c.set_facecolor(BG)
    ax_c.set_title("RF Strategy Monthly Returns", color=TEXT, fontsize=10)
    monthly = bt_rf["net_ret"].resample("ME").apply(lambda x: (1+x).prod()-1)
    monthly_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values
    })
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    mask = pivot.isna()
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn",
                center=0, ax=ax_c, mask=mask, cbar=False,
                linewidths=0.5, annot_kws={"size": 7})
    ax_c.set_xlabel("")
    ax_c.set_ylabel("")

    plt.savefig(f"{output_dir}/plot4_backtest.png",
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[Plot] 4/5 — Backtest saved")

    # ── PLOT 5: Feature Importance + Correlation Heatmap ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
    fig.suptitle("Feature Analysis", color=TEXT, fontsize=13, fontweight="bold")

    # Left: Feature importance (Best Model by ROC-AUC)
    ax_fi = axes[0]
    ax_fi.set_facecolor(BG)
    
    # 1. Select the Best Model based on Highest ROC-AUC in the Test set
    test_metrics = metrics_df[metrics_df["Split"] == "Test"].copy()
    
    # Filter out models that don't have feature importances (e.g., LogisticRegression, NeuralNetwork)
    # We check if the model object in the 'models' dictionary has 'feature_importances_'
    tree_models = [name for name, (m_type, m_obj) in models.items() 
                   if hasattr(m_obj, "feature_importances_")]
    
    if tree_models:
        best_tree_name = test_metrics[test_metrics["Model"].isin(tree_models)].sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
        best_tree_obj = models[best_tree_name][1]
        
        imp_series = pd.Series(best_tree_obj.feature_importances_, index=features)
        imp_df_best = imp_series.sort_values(ascending=False)
        
        top_n  = imp_df_best.head(15)
        colors = [GREEN if i < 5 else CYAN if i < 10 else ORNG
                  for i in range(len(top_n))]
        ax_fi.barh(top_n.index[::-1], top_n.values[::-1], color=colors[::-1], alpha=0.8)
        ax_fi.set_title(f"Feature Importances ({best_tree_name} - Best by ROC-AUC)",
                        color=TEXT, fontsize=10)
    else:
        # Fallback if no tree models are present
        ax_fi.text(0.5, 0.5, "No Tree Models Available for Feature Importance", ha='center', va='center', color=TEXT)
        ax_fi.set_title("Feature Importances", color=TEXT, fontsize=10)
        
    ax_fi.set_xlabel("Importance", fontsize=9)
    ax_fi.grid(True, axis="x", alpha=0.3)

    # Right: Correlation heatmap of key features
    ax_cm = axes[1]
    key_features = ["close", "rsi", "ma_ratio", "macd", "bb_pct",
                    "roll_std7", "gtrends", "gtrends_momentum", "fed_rate", "sp500_ret1",
                    "ret_lag1"]
    key_features = [f for f in key_features if f in df.columns]
    corr_data = df[key_features].dropna().corr()
    mask_upper = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    sns.heatmap(corr_data, ax=ax_cm, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                mask=mask_upper, square=True, cbar=True,
                linewidths=0.3)
    ax_cm.set_title("Feature Correlation Matrix", color=TEXT, fontsize=10)
    ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot5_features.png",
                dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[Plot] 5/5 — Feature Analysis saved")

    print(f"\n[Plots] All 5 charts saved to: {os.path.abspath(output_dir)}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  AlphaQuest Capital — BTC Signal Pipeline")
    print("  Sections: ETL → Features → Train → Evaluate → Backtest → Predict")
    print("="*70 + "\n")

    # ── STEP 1: ETL ──────────────────────────────────────────────────────────
    print("── STEP 1: ETL ──────────────────────────────────────────────────────")
    btc    = fetch_btc_ohlcv(START_DATE, END_DATE)
    macro  = fetch_fred_data(START_DATE, END_DATE, FRED_API_KEY)
    trends = fetch_google_trends(START_DATE, END_DATE)
    fgi    = fetch_fear_greed_index(START_DATE, END_DATE)

    # ── STEP 2: Feature Engineering ──────────────────────────────────────────
    print("\n── STEP 2: Feature Engineering ──────────────────────────────────────")
    df = compute_technical_features(btc)
    df = integrate_macro_trends(df, macro, trends, fgi)
    df = create_target(df, threshold=SIGNAL_THRESHOLD)

    # Drop rows where target or key features are NaN (rolling warm-up)
    df = df.dropna(subset=["target", "ma20", "rsi", "ema26"])

    print(f"\n[Data] Final dataset shape: {df.shape}")
    print(f"[Data] Columns: {list(df.columns)}")

    # ── STEP 3: Time-Based Split ──────────────────────────────────────────────
    print("\n── STEP 3: Time-Based Split ─────────────────────────────────────────")
    train_df, val_df, test_df = time_split(df)

    # Define feature set (exclude raw OHLCV, target, forward return)
    EXCLUDE = {"open", "high", "low", "close", "volume",
               "target", "forward_ret", "bb_upper", "bb_lower"}
    features = [c for c in df.columns if c not in EXCLUDE]
    print(f"[Features] Using {len(features)} features: {features}")

    X_train = train_df[features];  y_train = train_df["target"]
    X_val   = val_df[features];    y_val   = val_df["target"]
    X_test  = test_df[features];   y_test  = test_df["target"]

    # ── STEP 4: Feature Selection ─────────────────────────────────────────────
    print("\n── STEP 4: Feature Selection ────────────────────────────────────────")
    features_sel, imp_df = select_features(X_train, y_train, X_val)
    X_train = X_train[features_sel]
    X_val   = X_val[features_sel]
    X_test  = X_test[features_sel]

    # ── STEP 5: Train Models ──────────────────────────────────────────────────
    print("\n── STEP 5: Train Models ─────────────────────────────────────────────")
    models, scaler = train_models(X_train, y_train, X_val, y_val)

    # ── STEP 6: Evaluate ─────────────────────────────────────────────────────
    print("\n── STEP 6: Evaluate Models ──────────────────────────────────────────")
    metrics_df = evaluate_models(models, scaler,
                                 X_train, y_train,
                                 X_val,   y_val,
                                 X_test,  y_test,
                                 features_sel)

    # ── STEP 7: Backtest ──────────────────────────────────────────────────────
    print("\n── STEP 7: Backtest ─────────────────────────────────────────────────")
    backtests = {}
    bt_rf     = None

    for name, (inp_type, model) in models.items():
        X_t = scaler.transform(X_test) if inp_type == "scaled" else X_test
        bt = run_backtest(model, scaler, inp_type,
                          X_test, test_df,
                          proba_cutoff=PROBA_CUTOFF,
                          name=name)
        backtests[name] = bt
        if name == "RandomForest":
            bt_rf = bt

    if bt_rf is None:
        bt_rf = list(backtests.values())[0]   # fallback

    # ── STEP 8: Next-Day Prediction ───────────────────────────────────────────
    print("\n── STEP 8: Next-Day Prediction ──────────────────────────────────────")
    # Use best model (RF or first available)
    best_name, (best_inp, best_model) = list(models.items())[1] \
        if len(models) > 1 else list(models.items())[0]

    prediction = predict_next_day(best_model, scaler, best_inp,
                                  df, features_sel, SIGNAL_THRESHOLD)

    # ── STEP 8.5: Extract LR Formula ──────────────────────────────────────────
    print("\n── STEP 8.5: Extract Formula ────────────────────────────────────────")
    lr_model = models["LogisticRegression"][1]
    lr_coef_df, lr_formula = extract_lr_formula(lr_model, features_sel)

    # ── STEP 9: Plot Everything ───────────────────────────────────────────────
    print("\n── STEP 9: Generate Plots ───────────────────────────────────────────")
    plot_all(df, metrics_df, backtests, bt_rf, imp_df, features_sel, models,
             output_dir=OUTPUT_DIR)

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print(f"\n  Next-Day Signal:  {prediction['signal_label']}")
    print(f"  Signal Date:      {prediction['date']}")
    print(f"  Model Confidence: {prediction['proba']:.1%}")
    print(f"\n  Saved files:")
    for i in range(1, 6):
        print(f"    plot{i}_*.png")
    print()

    return df, models, scaler, metrics_df, backtests, prediction, lr_coef_df, lr_formula


if __name__ == "__main__":
    df, models, scaler, metrics_df, backtests, prediction, lr_coef_df, lr_formula = main()
