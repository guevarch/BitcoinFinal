# Bitcoin Analytics Dashboard with Flask

## Overview
This Flask application presents a comprehensive dashboard for Bitcoin analytics leveraging data visualization techniques. It covers several aspects of Bitcoin data examining price trends, predicting future values using the Prophet model, and assessing financial metrics like moving averages and year-to-date returns.

---

## Features
The dashboard includes the following features visualized through Plotly:

- **Bitcoin Buy Zones**: Graph showing different valuation zones for Bitcoin.
- **Moving Averages**: Visualization of various moving averages like 200-day, 300-day, and 50-day.
- **MACD Analysis**: A graph showing Moving Average Convergence Divergence (MACD) data for Bitcoin.
- **Indicators**: A gauge indicator representing the current standard deviation percent move from the mean average.
- **Year To Date Returns**: A horizontal bar chart comparing the Year To Date (YTD%) returns of different assets including Bitcoin.
- **Correlation Matrix**: A heatmap showing the correlation between Bitcoin and other financial assets.
- **30 Day Correlation Bars**: Bar charts representing the 30-day correlation coefficient between Bitcoin and certain indexes/assets.

Additional features include:
- Extrapolation of future Bitcoin values using the Prophet model.
- Calculation of RSI, an important technical indicator for trading analysis.
- Visualization of a 'Black Rainbow', showcasing logarithmic regression bands on a dark theme.
- A modified 'rainbow' chart portraying logarithmic growth over time.

---

## Data Processing
- The app fetches historical Bitcoin data using `yfinance`.
- Bitcoin prices are combined with other datasets considering date and wallet information.
- Rolling means and EMA calculations are performed to get moving averages and MACD indicators.
- JSON serialization is used to prepare data for Plotly visualization.

## Predictions
- Predictive modeling is accomplished using Facebook's Prophet model.
- In-sample predictions generate future data points for Bitcoin pricing.

---

## Visualization
- Plotly and Plotly Express are utilized for creating interactive charts.
- Logarithmic scales are applied where appropriate for better representation of financial data.
- Color palettes are carefully chosen to differentiate data points in scatter plots, heatmaps, and bar charts.

## Technical Stack
- Libraries: `Flask`, `pandas`, `numpy`, `Plotly`, `pandas_ta`, `yfinance`, and related dependencies for predictive analytics.
- Hosting: The app is designed to be run as a Flask web server.

---

## Running the Application
To run this dashboard locally, execute the following command in the terminal:

```bash
python app.py
```

Once the server starts, navigate to `http://localhost:5000/` in your web browser to view the dashboard.

---