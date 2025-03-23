"""
Generate realistic synthetic stock data for Tesla simulation.
This module provides functions to create realistic stock data for backtesting when
real data is not available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_trading_dates(start_date_str, end_date_str):
    """
    Generate a list of trading dates (excluding weekends).
    
    Args:
        start_date_str: Start date string in 'YYYY-MM-DD' format
        end_date_str: End date string in 'YYYY-MM-DD' format
        
    Returns:
        List of datetime objects representing trading days
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # Generate all dates between start and end
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # Filter out weekends (Saturday = 5, Sunday = 6)
    trading_dates = [date for date in all_dates if date.weekday() < 5]
    
    return trading_dates

def generate_realistic_stock_data(symbol, start_date, end_date, base_price=150, 
                                 volatility=0.02, trend=0.0, price_range=None):
    """
    Generate realistic synthetic stock data with OHLCV format.
    
    Args:
        symbol: Stock symbol (e.g. 'TSLA')
        start_date: Start date string in 'YYYY-MM-DD' format
        end_date: End date string in 'YYYY-MM-DD' format
        base_price: Starting price for the simulation
        volatility: Daily volatility (standard deviation of returns)
        trend: Daily trend factor (positive for upward, negative for downward)
        price_range: Tuple of (min_price, max_price) to constrain prices
        
    Returns:
        DataFrame with synthetic stock data
    """
    # Get trading dates
    trading_dates = generate_trading_dates(start_date, end_date)
    
    # Number of trading days
    n_days = len(trading_dates)
    
    # Generate daily returns with the specified volatility and trend
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(trend, volatility, n_days)
    
    # Calculate daily prices
    prices = [base_price]
    for ret in daily_returns:
        # Calculate new price based on previous price and return
        new_price = prices[-1] * (1 + ret)
        
        # Apply price constraints if specified
        if price_range:
            min_price, max_price = price_range
            new_price = max(min(new_price, max_price), min_price)
            
        prices.append(new_price)
    
    # Remove the first element (it was just the base price)
    prices = prices[1:]
    
    # Generate OHLC data
    data = []
    for i, date in enumerate(trading_dates):
        # Get the closing price for the day
        close_price = prices[i]
        
        # Generate realistic intraday volatility
        daily_vol = volatility * close_price
        
        # Generate open, high, low prices
        open_price = close_price * (1 + np.random.normal(0, 0.005))  # Small variation from close
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))  # Always higher
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))   # Always lower
        
        # Generate volume - higher on more volatile days
        price_change_pct = abs((close_price - open_price) / open_price)
        volume_factor = 1 + 5 * price_change_pct  # More volume on days with larger price changes
        volume = int(np.random.gamma(2, 1000000 * volume_factor))
        
        # Add to data
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'Symbol': symbol
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df

def generate_tesla_march_2023_data():
    """
    Generate synthetic Tesla stock data for March 2023 with realistic characteristics.
    
    Returns:
        DataFrame with synthetic Tesla data for March 2023
    """
    # Real Tesla price from early March 2023 was around $180-200
    return generate_realistic_stock_data(
        symbol='TSLA',
        start_date='2023-03-01',
        end_date='2023-03-31',
        base_price=190,  # Starting price around $190
        volatility=0.025,  # Slightly higher volatility
        trend=0.001,       # Slight upward trend
        price_range=(150, 220)  # Reasonable price range for that period
    )

def generate_tesla_march_2025_data():
    """
    Generate synthetic Tesla stock data for March 24-28, 2025 with realistic characteristics.
    
    Returns:
        DataFrame with synthetic Tesla data for March 2025 (1 week)
    """
    # Use current price as a baseline and project forward
    current_price = 200  # Example, should be replaced with a more realistic estimate
    
    return generate_realistic_stock_data(
        symbol='TSLA',
        start_date='2025-03-24',
        end_date='2025-03-28',
        base_price=current_price,
        volatility=0.022,  # Typical Tesla volatility
        trend=0.0,        # Neutral trend
        price_range=(current_price * 0.9, current_price * 1.1)  # +/- 10% from baseline
    )

def interpolate_price_data(start_price, end_price, n_days, volatility=0.02, trend_strength=0.7):
    """
    Interpolate between two price points with realistic price movements.
    
    Args:
        start_price: Starting price point
        end_price: Ending price point
        n_days: Number of days between points
        volatility: Daily volatility
        trend_strength: How strongly to bias toward the end price (0-1)
        
    Returns:
        List of interpolated prices
    """
    # Calculate the trend component
    total_return = (end_price / start_price) - 1
    daily_trend = (1 + total_return) ** (1 / n_days) - 1
    
    # Generate random walk with drift
    np.random.seed(42)  # For reproducibility
    random_returns = np.random.normal(0, volatility, n_days)
    
    # Combine trend and random components
    returns = [(daily_trend * trend_strength) + (ret * (1 - trend_strength)) for ret in random_returns]
    
    # Calculate prices
    prices = [start_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Adjust final price
    adjustment_factor = end_price / prices[-1]
    adjusted_prices = [p * adjustment_factor for p in prices]
    
    return adjusted_prices[1:]  # Remove the first element (it was just the start price)

def generate_data_with_specific_pattern(pattern_type, symbol='TSLA', start_date='2023-03-01', 
                                       end_date='2023-03-31', base_price=190):
    """
    Generate synthetic data with a specific price pattern.
    
    Args:
        pattern_type: String identifier for the pattern ('uptrend', 'downtrend', 'volatile', etc.)
        symbol: Stock symbol
        start_date: Start date string
        end_date: End date string
        base_price: Starting price
        
    Returns:
        DataFrame with synthetic stock data exhibiting the specified pattern
    """
    # Get trading dates
    trading_dates = generate_trading_dates(start_date, end_date)
    n_days = len(trading_dates)
    
    # Generate prices based on pattern
    if pattern_type == 'uptrend':
        end_price = base_price * 1.3  # 30% increase
        trend = 0.003
        volatility = 0.015
    elif pattern_type == 'downtrend':
        end_price = base_price * 0.7  # 30% decrease
        trend = -0.003
        volatility = 0.015
    elif pattern_type == 'volatile':
        end_price = base_price * 1.1  # 10% increase but with high volatility
        trend = 0.001
        volatility = 0.03
    elif pattern_type == 'sideways':
        end_price = base_price * 1.02  # Almost flat
        trend = 0.0
        volatility = 0.01
    elif pattern_type == 'v_shape':
        # Create a V-shaped recovery
        mid_point = n_days // 2
        first_half = interpolate_price_data(base_price, base_price * 0.7, mid_point, volatility=0.015)
        second_half = interpolate_price_data(base_price * 0.7, base_price * 1.1, n_days - mid_point, volatility=0.015)
        prices = first_half + second_half
        return _create_dataframe_from_prices(prices, trading_dates, symbol)
    elif pattern_type == 'inverted_v':
        # Create an inverted V pattern
        mid_point = n_days // 2
        first_half = interpolate_price_data(base_price, base_price * 1.3, mid_point, volatility=0.015)
        second_half = interpolate_price_data(base_price * 1.3, base_price * 0.9, n_days - mid_point, volatility=0.015)
        prices = first_half + second_half
        return _create_dataframe_from_prices(prices, trading_dates, symbol)
    else:
        # Default to realistic random data
        end_price = base_price * 1.1
        trend = 0.001
        volatility = 0.02
    
    # For simple patterns, use the regular generator with appropriate parameters
    return generate_realistic_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        base_price=base_price,
        volatility=volatility,
        trend=trend,
        price_range=None  # No explicit price range
    )

def _create_dataframe_from_prices(prices, dates, symbol):
    """
    Create a complete OHLCV DataFrame from a list of closing prices.
    
    Args:
        prices: List of closing prices
        dates: List of dates
        symbol: Stock symbol
        
    Returns:
        DataFrame with OHLCV data
    """
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Generate realistic intraday volatility
        daily_vol = 0.02 * close_price
        
        # Generate open, high, low prices
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        # Generate volume
        price_change_pct = abs((close_price - (prices[i-1] if i > 0 else close_price)) / close_price)
        volume_factor = 1 + 5 * price_change_pct
        volume = int(np.random.gamma(2, 1000000 * volume_factor))
        
        # Add to data
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'Symbol': symbol
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("Generating sample data...")
    
    # 1. Simple realistic data
    df_march_2023 = generate_tesla_march_2023_data()
    print(f"Generated data for March 2023: {len(df_march_2023)} trading days")
    
    # 2. Generate specific pattern
    df_uptrend = generate_data_with_specific_pattern('uptrend')
    print(f"Generated uptrend data: {len(df_uptrend)} trading days")
    
    # Save to CSV for inspection
    df_march_2023.to_csv('sample_tesla_march_2023.csv')
    print("Data saved to sample_tesla_march_2023.csv") 