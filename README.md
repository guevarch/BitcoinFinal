# Bitcoin Analytics Dashboard with Flask

## Overview
This Flask application serves as a comprehensive Bitcoin analytics dashboard, providing deep-dive data visualizations and predictive analytics. By merging historical CSV data with live market feeds, the platform offers a multi-faceted view of Bitcoin’s valuation, market cycles, and institutional correlations.

---

## Key Features

### 📈 Market Valuation & Technicals
* **Bitcoin Prophet Model + Buy Zones**: Combines Facebook's Prophet forecasting with standard deviation "Buy Zones" to identify market extremes.
* **Moving Average Suite**: Visualizes 7D, 50D, 200D, and 300D moving averages, including a "Moving Average Cloud" for trend strength analysis.
* **MACD Analysis**: Detailed tracking of the Moving Average Convergence Divergence, including Signal and Histogram data.
* **Current Valuation Gauge**: A real-time indicator showing the standard deviation percent move from the mean average.

### 🌈 Cycle & Regression Analysis
* **Rainbow Charts**: Logarithmic regression bands provided in both "Standard" and "Black" (dark-themed) modes to track long-term price exhaustion.
* **Halving Tracker**: A cycle comparison tool that scales previous historical cycles (2012, 2016, 2020) against the current 2024–2028 era.

### 🏦 Institutional & Macro Metrics
* **MSTR to Bitcoin Ratio**: Analyzes the relationship between MicroStrategy stock and BTC price, including Bollinger Band analysis on the ratio's volatility.
* **Inter-Market Correlation Matrix**: A heatmap measuring Bitcoin's correlation with Gold, the S&P 500, Nasdaq, and the DXY.
* **30-Day Correlation Bars**: Dynamic bar charts tracking rolling correlations against major global indexes.
* **YTD Returns Comparison**: A performance leaderboard comparing Bitcoin’s Year-to-Date returns against Gold, Oil, Treasuries, and major stock indices.

---

## Technical Architecture

### Data Processing
* **Data Sourcing**: Leverages the `yfinance` API for live price updates and `Resources/btcjoin.csv` for historical context.
* **Statistical Analysis**: Uses `pandas` and `numpy` for rolling mean calculations, exponential smoothing (EMA), and curve fitting for regression bands.
* **Forecasting**: Employs the `Prophet` library to generate 60-day price predictions based on log-transformed historical data.

### Tech Stack
* **Backend**: `Flask`
* **Analysis**: `pandas`, `pandas_ta`, `scipy`, `numpy`
* **Machine Learning**: `prophet`
* **Visualization**: `Plotly`, `Seaborn`, `Matplotlib` (rendered via `mpld3`)

---

## Installation & Usage

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install flask pandas plotly pandas-ta prophet yfinance seaborn mpld3 scipy
```

### Running the Dashboard
1. Clone the repository to your local machine.
2. Execute the application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to: `http://127.0.0.1:5000/`

---

## Project Structure
* `app.py`: The core engine containing data ingestion, indicator logic, and Plotly JSON serialization.
* `templates/bar.html`: The frontend dashboard layout that renders the interactive Plotly objects.
* `static/`: Stores generated assets like the monthly returns heatmap.