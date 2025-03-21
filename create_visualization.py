import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def visualize_trading_results():
    """
    Create visualizations for the trading simulation results
    """
    # Check if results directory exists
    if not os.path.exists('results'):
        print("Results directory not found. Please run the simulation first.")
        return
    
    # Check for performance summary file
    perf_path = os.path.join('results', 'performance_summary.csv')
    if not os.path.exists(perf_path):
        print(f"Performance summary file not found at {perf_path}")
        print("Looking for trading history...")
    else:
        # Load performance summary
        performance = pd.read_csv(perf_path)
        print("Loaded performance summary:")
        print(performance)
        print("\n")
    
    # Check for trading history file
    history_path = os.path.join('results', 'trading_history.csv')
    if not os.path.exists(history_path):
        print(f"Trading history file not found at {history_path}")
        return
    
    # Load trading history
    trading_history = pd.read_csv(history_path)
    
    # Convert date columns to datetime
    if 'Date' in trading_history.columns:
        trading_history['Date'] = pd.to_datetime(trading_history['Date'])
    elif 'date' in trading_history.columns:
        trading_history['date'] = pd.to_datetime(trading_history['date'])
        trading_history.rename(columns={'date': 'Date'}, inplace=True)
    
    print(f"Loaded trading history with {len(trading_history)} entries")
    print("First few rows:")
    print(trading_history.head())
    print("\nColumns:", trading_history.columns.tolist())
    
    # Create visualizations directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Plot account value over time
    plt.figure(figsize=(12, 6))
    
    if 'portfolio_value' in trading_history.columns:
        plt.plot(trading_history['Date'], trading_history['portfolio_value'], 
                 label='Portfolio Value', linewidth=2)
    elif 'account_value' in trading_history.columns:
        plt.plot(trading_history['Date'], trading_history['account_value'], 
                 label='Account Value', linewidth=2)
    
    if 'Close' in trading_history.columns:
        # Normalize stock price to start at the same point as account value
        first_account_value = trading_history.iloc[0]['portfolio_value'] if 'portfolio_value' in trading_history.columns else trading_history.iloc[0]['account_value']
        first_close = trading_history.iloc[0]['Close']
        scale_factor = first_account_value / first_close
        
        plt.plot(trading_history['Date'], trading_history['Close'] * scale_factor, 
                 label='TSLA Stock Price (Scaled)', linestyle='--', alpha=0.7)
    
    plt.title('Trading Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format the date axis
    date_format = DateFormatter("%Y-%m-%d")
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/trading_performance.png')
    
    # Plot trade actions
    plt.figure(figsize=(12, 6))
    
    # Plot stock price
    if 'Close' in trading_history.columns:
        plt.plot(trading_history['Date'], trading_history['Close'], 
                 label='TSLA Price', color='gray', alpha=0.6)
    
    # Mark buy and sell points
    if 'action' in trading_history.columns:
        buy_points = trading_history[trading_history['action'] == 'buy']
        sell_points = trading_history[trading_history['action'] == 'sell']
        
        if not buy_points.empty:
            plt.scatter(buy_points['Date'], buy_points['Close'], 
                     marker='^', color='green', s=100, label='Buy')
        if not sell_points.empty:
            plt.scatter(sell_points['Date'], sell_points['Close'], 
                     marker='v', color='red', s=100, label='Sell')
    
    plt.title('TSLA Trading Actions')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('visualizations/trading_actions.png')
    
    # Print summary
    print("\nVisualization complete!")
    print("Check the 'visualizations' directory for the following plots:")
    print("1. trading_performance.png - Portfolio value vs. Stock price")
    print("2. trading_actions.png - Buy and sell points on stock price chart")

if __name__ == "__main__":
    visualize_trading_results() 