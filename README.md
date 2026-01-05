# Climate-Aware Volatility Forecasting & Trading

This repository contains the source code for the research project: **"Impact of Climate Change on Financial Market Volatility"**.

It implements a machine learning pipeline that uses granular climate data (ERA5) and disaster statistics (EM-DAT) to predict realized volatility in financial markets, with a specific focus on the Agriculture sector.

##  Project Structure

- **`builder/`**: Scripts to process raw data (ERA5, EM-DAT, Yahoo Finance) and build the training panels.
- **`LSTM Agriculture/`**: The core Deep Learning model (LSTM with Temporal Attention) for agricultural volatility.
- **`LSTM Sectors/`**: LSTM models for other S&P 500 sectors.
- **`Random Forest/`**: Monthly benchmark models using Random Forests.
- **`Strategy/`**: Investment strategy backtesting (Volatility Breakout) and live trading logic.
- **`robustness/`**: Scripts for ablation studies and stability checks.

## Dataset
The dataset used for this analysis (ERA5 climate data + EM-DAT disaster records) is too large to be hosted on GitHub.
**It has been provided separately via email/ swiss transfert link**
Please place the `data/` folder at the root of this repository to run the scripts.

## Getting Started

### Prerequisites
- Python 3.8+
- Recommended: Create a virtual environment.

### Installation
```bash
pip install -r requirements.txt
```

### Usage

**1. Train the Agriculture Model:**
```bash
python "LSTM Agriculture/train_agriculture.py"
```
This will train the LSTM model and save it to `bot/models/`.

**2. Run the Backtest:**
```bash
python "Strategy/backtest.py"
```
This will execute the "Volatility Breakout" strategy (200% Long / -100% Short) and generate performance metrics.

##  Key Results

- **Directional Accuracy (Agriculture):** ~90%
- **Strategy Performance (2015-2024):**
  - Cumulative Return: **+52.2%** (vs +19.7% Buy & Hold)
  - Max Drawdown: **-21.0%** (vs -46.0%)
  - Sharpe Ratio: **0.35** (vs 0.19)

##  Author
Charlie Ormond
