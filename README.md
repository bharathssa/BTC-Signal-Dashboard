# 🚀 BTC Macro-Sentiment Signal Dashboard

**AlphaQuest Capital — Bitcoin Trading Signal Pipeline**

A full-stack machine learning pipeline that predicts daily Bitcoin price movements using macro-economic indicators, technical analysis, and market sentiment data.

## 📊 Features

- **6 Composite Indicators**: Trend Score, Macro Pressure, RSI, Fear & Greed Index, ADX, S&P 500 Returns
- **5 ML Models**: Logistic Regression, Random Forest, Gradient Boosting, Neural Network (MLP), Ensemble Voter
- **Live Backtesting**: Historical simulation with $50M capital allocation
- **Interactive Dashboard**: Streamlit-powered UI with confusion matrices, equity curves, and model comparison

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a FRED API Key (Free)
1. Sign up at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Set it as an environment variable:
```bash
export FRED_API_KEY="your_key_here"
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

### 4. Run the Pipeline (CLI)
```bash
python btc_signal_pipeline.py
```

## 📈 Model Performance (2024 Backtest)

| Model | Return | Trades | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| Random Forest | +100.00% | 87 | 1.60 | -27.87% |
| Neural Network | +48.12% | 84 | 0.90 | -40.23% |
| Gradient Boosting | +44.95% | 121 | 0.96 | -30.89% |
| Ensemble Voter | +23.81% | 115 | 0.62 | -41.65% |

## 🔧 Data Sources

- **Yahoo Finance** — BTC-USD OHLCV data
- **FRED API** — Federal Funds Rate, S&P 500, 10Y Treasury Yield
- **Alternative.me** — Crypto Fear & Greed Index
- **Google Trends** — Bitcoin search interest

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. Past performance does not guarantee future results.
