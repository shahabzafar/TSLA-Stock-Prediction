"""Visualization utilities for the Tesla ML Trading Agent."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os
import json


def plot_stock_prices(df: pd.DataFrame, 
                     title: str = "Tesla Stock Price",
                     save_path: Optional[str] = None) -> None:
    """
    Plot stock prices from DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot closing price
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    
    # Add moving averages if available
    if 'MA50' in df.columns:
        plt.plot(df.index, df['MA50'], label='50-day MA', color='orange', alpha=0.7)
    if 'MA200' in df.columns:
        plt.plot(df.index, df['MA200'], label='200-day MA', color='red', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_trading_signals(df: pd.DataFrame, 
                        buy_signals: List[pd.Timestamp], 
                        sell_signals: List[pd.Timestamp],
                        title: str = "Trading Signals",
                        save_path: Optional[str] = None) -> None:
    """
    Plot stock prices with buy and sell signals.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        buy_signals (List[pd.Timestamp]): List of buy signal dates.
        sell_signals (List[pd.Timestamp]): List of sell signal dates.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot closing price
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    
    # Plot buy signals
    for date in buy_signals:
        if date in df.index:
            plt.scatter(date, df.loc[date, 'Close'], 
                       color='green', marker='^', s=100, label='Buy')
    
    # Plot sell signals
    for date in sell_signals:
        if date in df.index:
            plt.scatter(date, df.loc[date, 'Close'], 
                       color='red', marker='v', s=100, label='Sell')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_portfolio_performance(portfolio_history: pd.DataFrame,
                              benchmark_data: Optional[pd.DataFrame] = None,
                              title: str = "Portfolio Performance",
                              save_path: Optional[str] = None) -> None:
    """
    Plot portfolio performance over time.
    
    Args:
        portfolio_history (pd.DataFrame): DataFrame containing portfolio history.
        benchmark_data (pd.DataFrame, optional): DataFrame with benchmark data.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot portfolio value
    portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
    portfolio_history.set_index('timestamp', inplace=True)
    plt.plot(portfolio_history.index, portfolio_history['portfolio_value'], 
            label='Portfolio Value', color='blue')
    
    # Plot benchmark if available
    if benchmark_data is not None:
        # Normalize benchmark to start at initial portfolio value
        initial_value = portfolio_history['portfolio_value'].iloc[0]
        norm_factor = initial_value / benchmark_data['Close'].iloc[0]
        plt.plot(benchmark_data.index, benchmark_data['Close'] * norm_factor, 
                label='Benchmark (TSLA)', color='gray', alpha=0.7, linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_predictions(actual_prices: List[float],
                          predicted_prices: List[float],
                          dates: List[pd.Timestamp],
                          title: str = "Model Predictions vs Actual Prices",
                          save_path: Optional[str] = None) -> None:
    """
    Plot model predictions against actual prices.
    
    Args:
        actual_prices (List[float]): List of actual prices.
        predicted_prices (List[float]): List of predicted prices.
        dates (List[pd.Timestamp]): List of dates.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, actual_prices, label='Actual Price', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Price', color='orange', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(feature_names: List[str],
                           importances: List[float],
                           title: str = "Feature Importance",
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_names (List[str]): List of feature names.
        importances (List[float]): List of feature importance values.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the figure.
    """
    # Sort features by importance
    indices = np.argsort(importances)
    features = [feature_names[i] for i in indices]
    importance_values = [importances[i] for i in indices]
    
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    plt.barh(range(len(features)), importance_values, align='center')
    plt.yticks(range(len(features)), features)
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_performance_summary(results_df: pd.DataFrame,
                              output_dir: str = 'results') -> Dict[str, Any]:
    """
    Create a performance summary with visualizations.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results.
        output_dir (str): Directory to save visualizations.
        
    Returns:
        Dict[str, Any]: Performance summary metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check column names and adjust if needed (different versions might use different column names)
    date_col = 'date' if 'date' in results_df.columns else 'Date'
    action_col = 'action' if 'action' in results_df.columns else 'Action'
    portfolio_value_col = 'portfolio_value' if 'portfolio_value' in results_df.columns else 'Portfolio_Value'
    current_price_col = 'current_price' if 'current_price' in results_df.columns else 'Current_Price'
    
    # Extract data for plotting
    dates = results_df[date_col].tolist()
    portfolio_values = results_df[portfolio_value_col].tolist()
    
    # Create performance plots
    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio_values, label='Portfolio Value', color='blue')
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Value (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'portfolio_performance.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create price and signals plot
    buy_signals = results_df[results_df[action_col] == 'buy'][date_col].tolist()
    sell_signals = results_df[results_df[action_col] == 'sell'][date_col].tolist()
    
    # Create a DataFrame for plotting
    price_df = pd.DataFrame({
        'Date': dates,
        'Close': results_df[current_price_col].tolist()
    })
    price_df.set_index('Date', inplace=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(price_df.index, price_df['Close'], label='Price', color='blue')
    
    # Plot buy signals
    for date in buy_signals:
        price = price_df.loc[date, 'Close'] if date in price_df.index else None
        if price is not None:
            plt.scatter(date, price, color='green', marker='^', s=100)
    
    # Plot sell signals
    for date in sell_signals:
        price = price_df.loc[date, 'Close'] if date in price_df.index else None
        if price is not None:
            plt.scatter(date, price, color='red', marker='v', s=100)
    
    plt.title('Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend(['Price', 'Buy', 'Sell'])
    plt.savefig(os.path.join(output_dir, 'trading_signals.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate performance metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Calculate daily returns
    daily_returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                     for i in range(1, len(portfolio_values))]
    
    # Calculate Sharpe ratio (using 0% risk-free rate for simplicity)
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0
    
    # Calculate max drawdown
    running_max = 0
    drawdowns = []
    for value in portfolio_values:
        if value > running_max:
            running_max = value
        drawdown = (running_max - value) / running_max if running_max > 0 else 0
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns) * 100
    
    # Count trades
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    
    # Create summary
    summary = {
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'total_days': len(portfolio_values)
    }
    
    # Save summary to file
    with open(os.path.join(output_dir, 'performance_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary 