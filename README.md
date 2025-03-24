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
- **Dynamic Date Simulation**: Select any date in history to run a 5-day trading simulation
- **March 2023 Simulation**: Run a special simulation for March 2023 to test the model on historical data
- **Realistic Strategy**: Implement a trading strategy using current market price
- **Next Days Prediction**: Predict Tesla stock performance for the next 5 trading days from today

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
5. **March 2023 Simulation**: Test the model with historical data from March 2023
6. **Realistic Strategy Simulation**: Run a realistic trading strategy with current market prices
7. **Dynamic Date Simulation**: Select any date and run a trading simulation for the following 5 days
8. **Next Days Prediction**: Forecast Tesla stock behavior for the next 5 trading days

### Running Specific Simulations

#### March 2025 Project Simulation
From the dashboard, click the "Run Project Simulation" button, or run directly:
```
python run_simulation_2025.py
```

#### March 2023 Simulation
From the dashboard, click the "Run March 2023 Simulation" button, or run directly:
```
python run_simulation_march_2023.py
```

#### Realistic Strategy Simulation
From the dashboard, click the "Run Realistic Strategy" button, or run directly:
```
python simple_realistic_strategy.py
```

#### Dynamic Date Simulation
From the dashboard, select a date using the date picker and click "Run Simulation", or run directly:
```
python dynamic_simulation.py --year YYYY --month MM
```

#### Next Days Prediction
From the dashboard, enter the current Tesla stock price and click "Generate Prediction", or run directly:
```
python next_days_prediction.py --price XXX.XX
```

## Models

The project utilizes several machine learning models:

1. **Ensemble Model**: Combines predictions from multiple models to improve accuracy
2. **LSTM Model**: Deep learning model specialized for sequential data like time series
3. **Improved Model**: Enhanced version with additional features and optimization

## Files and Structure

- `app.py`: Flask web application for the dashboard
- `run_final_simulation.py`: Script for running the main historical simulation
- `run_simulation_2025.py`: Script for running the March 2025 project simulation
- `run_simulation_march_2023.py`: Script for simulating trading in March 2023
- `simple_realistic_strategy.py`: Implementation of a realistic trading strategy
- `dynamic_simulation.py`: Script for running simulations with user-selected dates
- `next_days_prediction.py`: Script for predicting near-future stock performance
- `templates/`: HTML templates for the web dashboard
- `data/`: Directory for Tesla stock data
- `models/`: Directory for trained machine learning models
  - `models/lstm_model/`: LSTM model files
  - `models/improved/`: Enhanced model files
  - `models/ensemble_model.py`: Ensemble model implementation
- `results/`: Directory for simulation results
- `visualizations/`: Directory for generated charts and visualizations
- `static/`: Static files for the web dashboard

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 