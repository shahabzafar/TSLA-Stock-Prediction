# Tesla Stock Trading with Machine Learning

A machine learning project for predicting and trading Tesla (TSLA) stock using various ML models and strategies. This project includes a web dashboard for visualizing the performance of trading strategies.

## Project Overview

This project applies machine learning techniques to develop a trading agent capable of predicting Tesla's stock movements and making informed trading decisions. The system simulates stock trading with specific rules, aiming to maximize portfolio value.

## Features

- **Historical Data Analysis**: Process and analyze historical Tesla stock data from 2010-2022
- **Feature Engineering**: Create technical indicators and features for machine learning models
- **Multiple ML Models**: Train various models including LSTM, Ensemble approaches
- **Trading Simulation**: Simulate trading based on ML model predictions
- **Performance Metrics**: Calculate key metrics such as total return, Sharpe ratio, max drawdown
- **Web Dashboard**: Interactive dashboard to visualize trading performance and history
- **Project Simulation**: Special simulation mode for the March 24-28, 2025 trading period matching project requirements

## Project Requirements Implementation

The project has been fully implemented according to the official requirements:

### March 2025 Project Simulation
- **Simulation Period**: March 24-28, 2025 (5 trading days)
- **Daily Order Submission**: Orders generated at 9:00 AM EST
- **Order Execution**: Orders executed at 10:00 AM EST
- **Starting Capital**: $10,000 USD
- **Transaction Fee**: 1% fee applied to each Buy/Sell order
- **Order Format**:
  - Buy: Dollar amount to invest
  - Sell: Number of shares to sell
  - Hold: No transaction

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Dashboard

```
python app.py
```

This will start the Flask web application. Open your browser and navigate to `http://localhost:5000` to access the dashboard.

### Dashboard Features

1. **Performance Metrics**: View key performance metrics of your trading strategy
2. **Visualizations**: Interactive charts showing portfolio performance and trading signals
3. **Trading History**: Detailed history of all trading decisions and portfolio values
4. **March 2025 Project Simulation**: Run a separate simulation specifically for the project requirements

### Running the March 2025 Project Simulation

From the dashboard, click the "Run Project Simulation" button in the March 2025 Project Simulation section.

Alternatively, you can run it directly:

```
python run_simulation_2025.py
```

This will:
1. Generate simulated Tesla data for the March 24-28, 2025 period
2. Apply your ML model to make trading decisions at 9:00 AM each day
3. Execute trades at 10:00 AM according to project specifications
4. Calculate and save performance metrics
5. Generate visualizations

## Files and Structure

- `app.py`: Flask web application for the dashboard
- `run_final_simulation.py`: Script for running the main historical simulation
- `run_simulation_2025.py`: Script for running the March 2025 project simulation
- `templates/`: HTML templates for the web dashboard
- `data/`: Directory for Tesla stock data
- `models/`: Directory for trained machine learning models
- `results/`: Directory for simulation results
- `visualizations/`: Directory for generated charts and visualizations

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 