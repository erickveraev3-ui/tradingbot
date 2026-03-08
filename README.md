# TradingBot — Quantitative Bitcoin Trading System

Quantitative trading system for Bitcoin designed to emulate professional discretionary trading logic using machine learning, structural market features, and probabilistic trade selection.

The system combines:

- pattern detection
- swing structure analysis
- meta-labeling models
- expected value filtering
- regime detection
- systematic backtesting

The goal is not to predict price directly but to **identify high-probability trading setups** similar to those used by professional traders.

---

# Strategy Philosophy

Most trading bots attempt to predict the next price movement.

This system takes a different approach:

1. Detect potential **market setups**
2. Estimate **probability of success**
3. Estimate **expected value**
4. Filter trades using **market regime**
5. Execute only **high-quality trades**

This architecture is inspired by quantitative trading approaches used in professional trading desks.

---

# Core Concepts

The bot combines several trading ideas:

### Market Structure
Detection of structural price behaviour:

- swing highs
- swing lows
- break of structure
- double tops
- double bottoms

### Pattern Engine
Detection of repeatable patterns used by traders.

### Meta-Labeling
Separate machine learning models evaluate:

- probability of long trade success
- probability of short trade success

### Expected Value Engine
Trades are only taken when the **expected value is positive**.

### Market Regime Engine
The system adapts behaviour based on market conditions:

- `trend_up`
- `trend_down`
- `range`

### Trade Selection
Only the highest-ranked setups are executed.

---

# Repository Structure
tradingbot/

scripts/
01_download_data.py
02_build_triple_barrier_dataset.py
03_train_meta_label_model_v3.py
07_debug_trade_funnel.py
08_backtest_meta_model_v6.py
09_trade_attribution.py

src/
features/
    feature_builder.py
    structure_features.py
    pattern_engine.py

structure/
    swing_structure.py

strategy/
    expected_value_engine.py
    regime_engine.py
    setup_ranking.py
    position_sizer.py
    
---

# Data Pipeline

The pipeline follows these steps:
Raw Market Data
↓
Feature Engineering
↓
Pattern Detection
↓
Swing Structure Analysis
↓
Triple Barrier Labeling
↓
Meta-Label Model Training
↓
Expected Value Calculation
↓
Regime Filtering
↓
Trade Selection
↓
Backtesting

---

# Dataset

Example dataset characteristics:

- ~48,700 rows
- ~160+ engineered features
- candidate trades:
candidate_long ≈ 3.7%
candidate_short ≈ 3.5%


Triple barrier outcomes:
tb_long_win ≈ 1.28%
tb_short_win ≈ 1.22%


---

# Current Backtest Results

Example results from `meta_model_v6`:
Initial capital: 10000
Final capital: 11754
Total return: +17.5%

Sharpe ratio: 2.77
Max drawdown: -2.04%

Trades: 55
Win rate: 65%
Long trades: 29
Short trades: 26

Trade attribution shows that the system performs best during:
trend_down regimes

and high expected-value setups.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/erickveraev3-ui/tradingbot.git
cd tradingbot
Create virtual environment:
python -m venv .venv
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt

Running the Pipeline
1 Download data
python scripts/01_download_data.py
2 Build dataset
python scripts/02_build_triple_barrier_dataset.py
3 Train models
python scripts/03_train_meta_label_model_v3.py
4 Debug trade funnel
python scripts/07_debug_trade_funnel.py
5 Backtest strategy
python scripts/08_backtest_meta_model_v6.py
6 Trade attribution analysis
python scripts/09_trade_attribution.py
Strategy Components
Swing Structure Engine

Provides structural market understanding:

swing highs

swing lows

breakout detection

double tops / bottoms

This allows the model to capture chart patterns used by discretionary traders.
Meta-Label Models

Two independent models:

meta_model_long
meta_model_short

Each estimates the probability that a candidate setup will be profitable.

Expected Value Engine

Trades are only executed when:

Expected Value > threshold

This prevents the model from trading low-quality setups.
Current Development Stage

The system currently operates in the research and validation phase.

Next development steps include:

walk-forward validation

robustness testing

stress testing with slippage and fees

paper trading

Disclaimer

This project is for research and educational purposes.

Trading cryptocurrencies involves substantial risk.

No financial advice is provided.

Author

Erick Vera
AI & Quantitative Trading Research