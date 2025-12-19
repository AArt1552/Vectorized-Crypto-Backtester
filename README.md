# ⚡ High-Performance Quantitative Crypto Backtester

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

A highly optimized, **event-driven quantitative backtesting engine** designed specifically for **Altcoin trading strategies** that utilize Bitcoin (BTC) as a market driver.

Built with **Numba JIT compilation** and **multiprocessing**, this tool is capable of simulating **millions of parameter combinations** in record time. It allows traders to mathematically validate complex strategies involving correlation-based entries, dynamic leverage, and sentiment analysis (Fear & Greed Index) before risking capital.

## 🚀 Key Features

* **BTC-Correlated Strategy:** Designed to trade Altcoins (e.g., ADA, ETH, SOL) by analyzing Bitcoin's price action as a primary trend filter and entry trigger.
* **Quantitative Permutation:** Test infinite variations of entries, exits, and risk settings. The engine generates a Cartesian product of all your config parameters to find the statistical "sweet spot."
* **Blazing Fast Execution:** Core simulation logic is compiled to machine code using `numba.njit`, making iteration over years of 1-minute candle data nearly instantaneous.
* **Dynamic Leverage & Risk:** Supports advanced logic where leverage and position sizing adapt automatically based on the **Crypto Fear & Greed Index**.
* **Parallel Processing:** Automatically distributes the workload across all available CPU cores.
* **Smart Data Handling:** Includes automated tools to fetch historical data (Binance) and cache it as optimized `.feather` files for rapid reloading.

## 🧠 The Strategy Logic

Unlike simple indicator-based bots (RSI/MACD), this engine simulates a sophisticated **Price Action & Correlation Model**:

1.  **Bitcoin as the Leader:** The engine monitors BTC price movements on micro-timeframes. It detects specific volatility patterns (pumps or dumps) in Bitcoin to predict subsequent moves in the target Altcoin.
2.  **Altcoin Execution:** Trades are executed on the Altcoin (Asset) based on the "lag effect" or correlation with Bitcoin, combined with the Altcoin's own local price action.
3.  **Dynamic Entries:** Supports multiple entry types:
    * **BTC Trigger:** Enter Altcoin when BTC moves $X$% in $Y$ minutes.
    * **Re-entries (DCA):** Configurable safety nets to average down price if the market moves against the position.
    * **Panic Exits:** Safety triggers based on rapid BTC crashes.
4.  **Sentiment-Based Params:** The system can switch entire parameter sets (Leverage, Take Profit targets, Stop Losses) depending on whether the global market is in "Extreme Fear" or "Greed".

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
    cd REPO_NAME
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy numba tqdm requests pyarrow
    ```

## ⚙️ Usage Workflow

### 1. Fetch Historical Data
Since this quantitative engine requires high-resolution data, you must download the history first. The included `get_data.py` script creates the necessary directory structure and fetches data from Binance.

**To download the last 365 days of data for BTC (Leader) and ADA (Target):**
```bash
python get_data.py 365 BTC ADA

This fetches 1-minute candles and the daily Fear & Greed Index, saving them to datasets/.

2. Configure the Matrix
Create a config.json file to define your search space. You can rename the provided config.example.json to start quickly.

Example of a quantitative search grid:

JSON

{
  "candles_file_BTC": ["datasets/BTC_val.json"],
  "candles_file_asset": ["datasets/ADA_val.json"],
  "candles_file_fear": ["datasets/fear_index.json"],
  
  "initial_balance": [1000.0],
  
  // Test different triggers: Does a 0.5% or 1.0% BTC move work better?
  "btc_entry_trigger_value": [0.005, 0.010],
  
  // Test leverage impact
  "leverage_BTC": [10, 20, 50],
  
  // Risk Management
  "stop_loss": [0.03, 0.05],
  "take_profit_dynamic": [true]
}
The engine will automatically generate and test every unique combination of these lists.

📘 Documentation: For a deep dive into every parameter and how the dynamic leverage logic works, read the Configuration Parameters Guide.

3. Run the Backtest
Execute the runner. It will calculate the total combinations and begin the parallel simulation.

Bash

python main_runner.py config.json
Advanced Usage:

Bash

# Test only the last 30 days using 8 CPU cores with detailed logging
python main_runner.py config.json 30 8 log
📂 Project Structure
main_runner.py: The quantitative orchestrator. Loads data and manages the multiprocessing pool.

vectorized_backtest.py: The Numba-compiled core engine. Contains the complex logic for BTC correlation, Fear & Greed overrides, and trade management.

get_data.py: Utility to fetch and normalize historical data from Binance and Alternative.me.

param_generator.py: Generates the Cartesian product of all parameters in your config file.

results_consolidator.py: Aggregates the parallel results into a readable performance report.

docs/CONFIGURATION_GUIDE.md: Detailed documentation of all configuration parameters.

🤝 Contributing
This is an open-source framework intended to help the community build robust, statistically validated trading strategies. If you have ideas for new entry triggers or optimization techniques:

Fork the repository.

Create your feature branch (git checkout -b feature/NewTrigger).

Commit your changes.

Open a Pull Request.

👤 Author
Vicente Dorsa Network & Security Specialist | Algorithmic Trading Enthusiast

I combine my professional background in Network Engineering and Cybersecurity with a strong passion for Python programming and quantitative financial markets. This project is the result of applying rigorous performance optimization to crypto market analysis.

🌐 LinkedIn: linkedin.com/in/vicentedorsa

📧 Email: vicente.dorsa@gmail.com

🚀 Live Trading Engine: This repository houses the backtesting framework. I maintain a private, institutional-grade version of this bot ("The Live Trader") capable of executing these strategies in real-time.

Interested in the Live Bot? Reach out via email with the subject "Live Bot Inquiry" for licensing or partnership opportunities.

⚠️ Disclaimer
This software is for educational and research purposes only. Past performance indicated by this backtester does not guarantee future results. Cryptocurrency trading involves significant risk. The authors are not responsible for any financial losses incurred.

📄 License
Distributed under the MIT License. See LICENSE for more information.


⚠️ Disclaimer
This software is for educational and research purposes only. Past performance indicated by this backtester does not guarantee future results. Cryptocurrency trading involves significant risk. The authors are not responsible for any financial losses incurred.

📄 License
Distributed under the MIT License. See LICENSE for more information.
