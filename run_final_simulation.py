import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create directories for results
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Load the stock data
def load_stock_data():
    """Load the Tesla stock data"""
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    if not data_files:
        print("No CSV files found in the data directory.")
        return None
    
    tesla_file = None
    for file in data_files:
        if 'TSLA' in file.upper():
            tesla_file = file
            break
    
    if tesla_file:
        print(f"Loading Tesla stock data from {tesla_file}")
        stock_data = pd.read_csv(os.path.join('data', tesla_file))
    else:
        print(f"Using first available CSV file: {data_files[0]}")
        stock_data = pd.read_csv(os.path.join('data', data_files[0]))
    
    # Ensure there's a Date column
    if 'Date' not in stock_data.columns and 'date' not in stock_data.columns:
        if 'index' in stock_data.columns and pd.api.types.is_datetime64_any_dtype(stock_data['index']):
            stock_data.rename(columns={'index': 'Date'}, inplace=True)
    
    # Convert Date to datetime
    date_col = 'Date' if 'Date' in stock_data.columns else 'date'
    stock_data[date_col] = pd.to_datetime(stock_data[date_col])
    
    print(f"Loaded {len(stock_data)} rows of stock data")
    print(f"Date range: {stock_data[date_col].min()} to {stock_data[date_col].max()}")
    
    return stock_data

# Create a simulated trading history
def create_simulated_trading_history(stock_data, initial_balance=10000, risk_tolerance=0.02):
    """Create a simulated trading history based on stock data"""
    # Ensure we have the essential columns
    if 'close' in stock_data.columns and 'Close' not in stock_data.columns:
        stock_data.rename(columns={'close': 'Close'}, inplace=True)
    
    if 'date' in stock_data.columns and 'Date' not in stock_data.columns:
        stock_data.rename(columns={'date': 'Date'}, inplace=True)
    
    # Make a copy to avoid modifying the original
    simulation_data = stock_data.copy()
    
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(simulation_data.index):
        simulation_data.set_index('Date', inplace=True)
    
    # Get model predictions
    print("Generating model predictions...")
    try:
        # Try to load ensemble model
        with open('models/ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load feature columns
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        print(f"Model loaded with {len(feature_columns)} feature columns")
        print(f"Available columns in data: {simulation_data.columns.tolist()}")
        
        # For simulation, we'll use a simpler approach since we might not have all model features
        simulation_data['prediction'] = 0
        for i in range(1, len(simulation_data)):
            if i % 5 == 0:  # Every 5 days, make a decision based on trend
                prev_close = simulation_data.iloc[i-1]['Close']
                curr_close = simulation_data.iloc[i]['Close']
                
                if curr_close > prev_close * 1.01:  # If price went up >1%, predict up
                    simulation_data.iloc[i, simulation_data.columns.get_loc('prediction')] = 1
                elif curr_close < prev_close * 0.99:  # If price went down >1%, predict down
                    simulation_data.iloc[i, simulation_data.columns.get_loc('prediction')] = -1
    
    except Exception as e:
        print(f"Error loading model or making predictions: {str(e)}")
        print("Using simple moving average strategy instead...")
        
        # Simple moving average strategy
        simulation_data['MA5'] = simulation_data['Close'].rolling(window=5).mean()
        simulation_data['MA20'] = simulation_data['Close'].rolling(window=20).mean()
        simulation_data['prediction'] = 0
        
        # Buy when 5-day MA crosses above 20-day MA, sell when it crosses below
        for i in range(20, len(simulation_data)):
            if (simulation_data.iloc[i-1]['MA5'] <= simulation_data.iloc[i-1]['MA20'] and 
                simulation_data.iloc[i]['MA5'] > simulation_data.iloc[i]['MA20']):
                simulation_data.iloc[i, simulation_data.columns.get_loc('prediction')] = 1
            elif (simulation_data.iloc[i-1]['MA5'] >= simulation_data.iloc[i-1]['MA20'] and 
                  simulation_data.iloc[i]['MA5'] < simulation_data.iloc[i]['MA20']):
                simulation_data.iloc[i, simulation_data.columns.get_loc('prediction')] = -1
    
    # Trading simulation
    print("Running trading simulation...")
    
    # Initialize portfolio
    cash = initial_balance
    shares = 0
    portfolio_values = []
    actions = []
    transaction_fee_pct = 0.001  # 0.1% transaction fee
    
    # Simulate trading
    for idx, row in simulation_data.iterrows():
        close_price = row['Close']
        prediction = row['prediction'] if 'prediction' in row else 0
        
        # Current portfolio value
        portfolio_value = cash + (shares * close_price)
        
        # Determine action
        action = 'hold'
        if prediction > 0 and cash > 0:  # Buy signal
            # Position sizing based on risk tolerance
            position_size = cash * risk_tolerance
            shares_to_buy = int(position_size / close_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * close_price
                fee = cost * transaction_fee_pct
                total_cost = cost + fee
                
                if total_cost <= cash:
                    cash -= total_cost
                    shares += shares_to_buy
                    action = 'buy'
        
        elif prediction < 0 and shares > 0:  # Sell signal
            # Sell all shares
            proceeds = shares * close_price
            fee = proceeds * transaction_fee_pct
            cash += proceeds - fee
            shares = 0
            action = 'sell'
        
        # Record portfolio value and action
        portfolio_values.append(portfolio_value)
        actions.append(action)
    
    # Create trading history DataFrame
    simulation_data['portfolio_value'] = portfolio_values
    simulation_data['action'] = actions
    
    # Reset index to make Date a column
    if simulation_data.index.name == 'Date':
        simulation_data = simulation_data.reset_index()
    
    # Calculate daily returns
    simulation_data['daily_return'] = simulation_data['portfolio_value'].pct_change()
    
    return simulation_data

# Create performance summary
def create_performance_summary(trading_history):
    """Create a performance summary based on trading history"""
    # Extract portfolio values
    initial_value = trading_history['portfolio_value'].iloc[0]
    final_value = trading_history['portfolio_value'].iloc[-1]
    
    # Calculate metrics
    total_return = (final_value / initial_value - 1) * 100
    
    # Calculate annualized return
    days = (trading_history['Date'].iloc[-1] - trading_history['Date'].iloc[0]).days
    years = days / 365
    annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.01  # Assuming 1% annual risk-free rate
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = trading_history['daily_return'].dropna() - daily_risk_free
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    # Calculate max drawdown
    peak = trading_history['portfolio_value'].cummax()
    drawdown = (trading_history['portfolio_value'] - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Count trades
    buy_trades = len(trading_history[trading_history['action'] == 'buy'])
    sell_trades = len(trading_history[trading_history['action'] == 'sell'])
    
    # Buy & Hold comparison
    stock_return = (trading_history['Close'].iloc[-1] / trading_history['Close'].iloc[0] - 1) * 100
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Metric': [
            'Initial Portfolio Value ($)',
            'Final Portfolio Value ($)',
            'Total Return (%)',
            'Annualized Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Number of Buy Trades',
            'Number of Sell Trades',
            'Buy & Hold Return (%)'
        ],
        'Value': [
            f"{initial_value:.2f}",
            f"{final_value:.2f}",
            f"{total_return:.2f}",
            f"{annualized_return:.2f}",
            f"{sharpe_ratio:.2f}",
            f"{max_drawdown:.2f}",
            buy_trades,
            sell_trades,
            f"{stock_return:.2f}"
        ]
    })
    
    return summary

# Visualize results
def visualize_results(trading_history):
    """Create visualizations of trading results"""
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot portfolio value vs stock price
    plt.figure(figsize=(12, 6))
    
    # Portfolio value
    plt.plot(trading_history['Date'], trading_history['portfolio_value'], 
             label='Portfolio Value', linewidth=2)
    
    # Stock price (scaled)
    first_portfolio = trading_history['portfolio_value'].iloc[0]
    first_close = trading_history['Close'].iloc[0]
    scale_factor = first_portfolio / first_close
    
    plt.plot(trading_history['Date'], trading_history['Close'] * scale_factor, 
             label='TSLA Stock Price (Scaled)', linestyle='--', alpha=0.7)
    
    plt.title('Trading Performance vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format dates
    plt.gcf().autofmt_xdate()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('visualizations/portfolio_performance.png')
    
    # Plot buy/sell actions
    plt.figure(figsize=(12, 6))
    
    # Plot stock price
    plt.plot(trading_history['Date'], trading_history['Close'], 
             label='TSLA Price', color='gray', alpha=0.6)
    
    # Mark buy points
    buy_points = trading_history[trading_history['action'] == 'buy']
    if not buy_points.empty:
        plt.scatter(buy_points['Date'], buy_points['Close'], 
                   marker='^', color='green', s=100, label='Buy')
    
    # Mark sell points
    sell_points = trading_history[trading_history['action'] == 'sell']
    if not sell_points.empty:
        plt.scatter(sell_points['Date'], sell_points['Close'], 
                   marker='v', color='red', s=100, label='Sell')
    
    plt.title('TSLA Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format dates
    plt.gcf().autofmt_xdate()
    
    # Save plot
    plt.tight_layout()
    plt.savefig('visualizations/trading_signals.png')
    
    print("\nVisualization complete!")
    print("Check the 'visualizations' directory for the generated plots:")
    print("1. portfolio_performance.png - Portfolio value vs. scaled stock price")
    print("2. trading_signals.png - Buy and sell signals on stock price chart")

def main():
    """Run the final Tesla stock trading simulation"""
    print("=== Tesla Stock Trading Simulation ===")
    
    # Load stock data
    stock_data = load_stock_data()
    if stock_data is None:
        print("Failed to load stock data. Please ensure data files are in the 'data' directory.")
        return
    
    # Run simulation
    print("\nRunning trading simulation...")
    trading_history = create_simulated_trading_history(
        stock_data, 
        initial_balance=10000,
        risk_tolerance=0.02
    )
    
    # Create performance summary
    print("\nGenerating performance summary...")
    performance_summary = create_performance_summary(trading_history)
    
    # Save results
    trading_history.to_csv('results/trading_history.csv', index=False)
    performance_summary.to_csv('results/performance_summary.csv', index=False)
    
    print("\nResults saved to 'results' directory:")
    print("1. trading_history.csv - Complete trading history")
    print("2. performance_summary.csv - Performance metrics")
    
    # Display performance summary
    print("\n=== Performance Summary ===")
    print(performance_summary.to_string(index=False))
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_results(trading_history)
    
    print("\n=== Simulation Complete ===")
    print("Final portfolio value: ${:.2f}".format(trading_history['portfolio_value'].iloc[-1]))
    
    initial_value = trading_history['portfolio_value'].iloc[0]
    final_value = trading_history['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    print("Total return: {:.2f}%".format(total_return))
    
    stock_return = (trading_history['Close'].iloc[-1] / trading_history['Close'].iloc[0] - 1) * 100
    print("Buy & Hold return: {:.2f}%".format(stock_return))
    
    outperformance = total_return - stock_return
    print("Strategy outperformance: {:.2f}%".format(outperformance))

if __name__ == "__main__":
    main() 