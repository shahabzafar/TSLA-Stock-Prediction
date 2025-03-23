"""Enhanced feature engineering for the Tesla ML Trading Agent."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import yfinance as yf  # For downloading additional market data
from datetime import datetime, timedelta
import requests
import json
import os

def add_enhanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add enhanced technical indicators to the DataFrame beyond the basics.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        
    Returns:
        pd.DataFrame: DataFrame with added enhanced technical indicators.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_features = df.copy()
    
    # Bollinger Bands Percentage B (%B)
    # Tells where price is in relation to the bands
    # Check if BB columns use different naming convention
    if 'BB_Upper' in df_features.columns:
        bb_upper = 'BB_Upper'
        bb_lower = 'BB_Lower'
        bb_middle = 'BB_Middle'
    else:
        # Try alternative names from feature_engineering.py
        bb_upper = 'Bollinger_Upper'
        bb_lower = 'Bollinger_Lower'
        bb_middle = 'MA20'  # BB Middle is typically the 20-day MA
    
    # Ensure the columns exist before using them
    if bb_upper in df_features.columns and bb_lower in df_features.columns and bb_middle in df_features.columns:
        df_features['BB_width'] = (df_features[bb_upper] - df_features[bb_lower]) / df_features[bb_middle]
        df_features['BB_pct_b'] = (df_features['Close'] - df_features[bb_lower]) / (df_features[bb_upper] - df_features[bb_lower])
    else:
        print("Warning: Bollinger Band columns not found. Skipping BB width and %B calculations.")
    
    # Ichimoku Cloud indicators
    df_features['Tenkan_sen'] = (df_features['High'].rolling(window=9).max() + df_features['Low'].rolling(window=9).min()) / 2
    df_features['Kijun_sen'] = (df_features['High'].rolling(window=26).max() + df_features['Low'].rolling(window=26).min()) / 2
    df_features['Senkou_span_a'] = ((df_features['Tenkan_sen'] + df_features['Kijun_sen']) / 2).shift(26)
    df_features['Senkou_span_b'] = ((df_features['High'].rolling(window=52).max() + df_features['Low'].rolling(window=52).min()) / 2).shift(26)
    df_features['Chikou_span'] = df_features['Close'].shift(-26)
    
    # Average Directional Index (ADX) - measure of trend strength
    df_features['DMplus'] = np.where(
        (df_features['High'] - df_features['High'].shift(1)) > (df_features['Low'].shift(1) - df_features['Low']),
        np.maximum(df_features['High'] - df_features['High'].shift(1), 0),
        0
    )
    df_features['DMminus'] = np.where(
        (df_features['Low'].shift(1) - df_features['Low']) > (df_features['High'] - df_features['High'].shift(1)),
        np.maximum(df_features['Low'].shift(1) - df_features['Low'], 0),
        0
    )
    df_features['TR'] = np.maximum(
        df_features['High'] - df_features['Low'],
        np.maximum(
            abs(df_features['High'] - df_features['Close'].shift(1)),
            abs(df_features['Low'] - df_features['Close'].shift(1))
        )
    )
    df_features['ATR14'] = df_features['TR'].rolling(window=14).mean()
    df_features['DMplus14'] = df_features['DMplus'].rolling(window=14).mean()
    df_features['DMminus14'] = df_features['DMminus'].rolling(window=14).mean()
    df_features['DIplus14'] = 100 * df_features['DMplus14'] / df_features['ATR14']
    df_features['DIminus14'] = 100 * df_features['DMminus14'] / df_features['ATR14']
    df_features['DIdiff'] = abs(df_features['DIplus14'] - df_features['DIminus14'])
    df_features['DIsum'] = df_features['DIplus14'] + df_features['DIminus14']
    df_features['DX'] = 100 * df_features['DIdiff'] / df_features['DIsum']
    df_features['ADX'] = df_features['DX'].rolling(window=14).mean()
    
    # Stochastic Oscillator
    df_features['Stoch_K'] = 100 * ((df_features['Close'] - df_features['Low'].rolling(window=14).min()) / 
                                    (df_features['High'].rolling(window=14).max() - df_features['Low'].rolling(window=14).min()))
    df_features['Stoch_D'] = df_features['Stoch_K'].rolling(window=3).mean()
    
    # Chaikin Money Flow (CMF)
    df_features['MFM'] = ((df_features['Close'] - df_features['Low']) - (df_features['High'] - df_features['Close'])) / (df_features['High'] - df_features['Low'])
    df_features['MFV'] = df_features['MFM'] * df_features['Volume']
    df_features['CMF'] = df_features['MFV'].rolling(window=20).sum() / df_features['Volume'].rolling(window=20).sum()
    
    # Volume-Weighted Average Price (VWAP)
    df_features['TP'] = (df_features['High'] + df_features['Low'] + df_features['Close']) / 3
    df_features['CumVol'] = df_features['Volume'].cumsum()
    df_features['CumVolPrice'] = (df_features['TP'] * df_features['Volume']).cumsum()
    df_features['VWAP'] = df_features['CumVolPrice'] / df_features['CumVol']
    
    # Check if OBV exists
    if 'OBV' in df_features.columns:
        # On-Balance Volume (OBV) and OBV Momentum
        df_features['OBV_ROC'] = df_features['OBV'].pct_change(periods=10) * 100
    else:
        # Calculate OBV if it doesn't exist
        obv = 0
        obv_series = []
        for i, row in df_features.iterrows():
            if i > 0:
                if row['Close'] > df_features['Close'].iloc[i-1]:
                    obv += row['Volume']
                elif row['Close'] < df_features['Close'].iloc[i-1]:
                    obv -= row['Volume']
            obv_series.append(obv)
        df_features['OBV'] = obv_series
        df_features['OBV_ROC'] = df_features['OBV'].pct_change(periods=10) * 100
    
    # Momentum Indicators - Triple Exponential Moving Average (TEMA)
    df_features['EMA'] = df_features['Close'].ewm(span=9, adjust=False).mean()
    df_features['EMA_of_EMA'] = df_features['EMA'].ewm(span=9, adjust=False).mean()
    df_features['EMA_of_EMA_of_EMA'] = df_features['EMA_of_EMA'].ewm(span=9, adjust=False).mean()
    df_features['TEMA'] = 3 * df_features['EMA'] - 3 * df_features['EMA_of_EMA'] + df_features['EMA_of_EMA_of_EMA']
    
    # Relative Volume
    df_features['Vol_10MA'] = df_features['Volume'].rolling(window=10).mean()
    df_features['Relative_Vol'] = df_features['Volume'] / df_features['Vol_10MA']
    
    # Drop intermediate calculation columns
    columns_to_drop = ['DMplus', 'DMminus', 'MFM', 'MFV', 'CumVol', 'CumVolPrice', 
                       'DIsum', 'DIdiff', 'DX', 'EMA_of_EMA', 'EMA_of_EMA_of_EMA',
                       'DIplus14', 'DIminus14', 'DMplus14', 'DMminus14']
    
    # Keep only essential columns to avoid too many features
    df_features = df_features.drop(columns=columns_to_drop, errors='ignore')
    
    return df_features


def add_market_indicators(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Add market and sector indicators to enhance prediction capabilities.
    Downloads comparable data if not provided.
    
    Args:
        df (pd.DataFrame): DataFrame containing Tesla stock data.
        start_date: Start date for the data.
        end_date: End date for the data.
        
    Returns:
        pd.DataFrame: DataFrame with added market indicators.
    """
    # Create a copy of the DataFrame
    df_with_market = df.copy()
    
    # If dates not provided, use date range from input DataFrame
    if start_date is None and end_date is None:
        if isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index.min() - timedelta(days=30)  # Add buffer for calculations
            end_date = df.index.max()
        else:
            # Try to find a date column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                start_date = df[date_cols[0]].min() - timedelta(days=30)
                end_date = df[date_cols[0]].max()
            else:
                raise ValueError("Could not determine date range from DataFrame. Please provide start_date and end_date.")
    
    # Define tickers to download
    market_tickers = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ',
        'XLI': 'Industrial Sector',
        'ARKK': 'Innovation ETF',
        'LIT': 'Lithium ETF',
        'DRIV': 'EV ETF'
    }
    
    # Competitor tickers
    competitor_tickers = {
        'F': 'Ford',
        'GM': 'General Motors',
        'LCID': 'Lucid',
        'RIVN': 'Rivian',
        'NIO': 'NIO'
    }
    
    # Create cache directory if it doesn't exist
    os.makedirs('data/market_data', exist_ok=True)
    
    # Function to download data with caching
    def download_with_cache(ticker, start, end, cache_path):
        cache_file = os.path.join(cache_path, f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")
        
        if os.path.exists(cache_file):
            try:
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            except Exception:
                pass
        
        # Download if not cached or cache is invalid
        data = yf.download(ticker, start=start, end=end, progress=False)
        if not data.empty:
            data.to_csv(cache_file)
        return data
    
    # Download market data
    market_data = {}
    for ticker in market_tickers:
        try:
            market_data[ticker] = download_with_cache(ticker, start_date, end_date, 'data/market_data')
        except Exception as e:
            print(f"Error downloading {ticker} data: {str(e)}")
            continue
    
    # Download competitor data
    competitor_data = {}
    for ticker in competitor_tickers:
        try:
            competitor_data[ticker] = download_with_cache(ticker, start_date, end_date, 'data/market_data')
        except Exception as e:
            print(f"Error downloading {ticker} data: {str(e)}")
            continue
    
    # Ensure we have a DatetimeIndex to align data
    if not isinstance(df_with_market.index, pd.DatetimeIndex):
        if 'Date' in df_with_market.columns:
            df_with_market = df_with_market.set_index('Date')
        elif 'date' in df_with_market.columns:
            df_with_market = df_with_market.set_index('date')
    
    # Add market indicators
    for ticker, ticker_data in market_data.items():
        if ticker_data.empty:
            continue
            
        # Calculate returns for correlation
        ticker_returns = ticker_data['Close'].pct_change()
        
        # Resample to match our data frequency if needed
        if df_with_market.index.freq != ticker_data.index.freq:
            ticker_returns = ticker_returns.reindex(df_with_market.index, method='ffill')
        
        # Add return and relative return vs Tesla
        df_with_market[f'{ticker}_Return'] = ticker_returns
        
        # Calculate 10-day correlation
        if 'Close' in df_with_market.columns:
            tesla_returns = df_with_market['Close'].pct_change()
            df_with_market[f'{ticker}_Corr_10D'] = tesla_returns.rolling(10).corr(ticker_returns)
    
    # Add competitor indicators
    for ticker, ticker_data in competitor_data.items():
        if ticker_data.empty:
            continue
            
        # Calculate returns
        ticker_returns = ticker_data['Close'].pct_change()
        
        # Resample to match our data frequency if needed
        if df_with_market.index.freq != ticker_data.index.freq:
            ticker_returns = ticker_returns.reindex(df_with_market.index, method='ffill')
        
        # Add competitor returns
        df_with_market[f'{ticker}_Return'] = ticker_returns
    
    # Calculate market breadth indicator (% of stocks above their 50-day MA)
    if 'SPY' in market_data and not market_data['SPY'].empty:
        df_with_market['SPY_Above_50MA'] = (market_data['SPY']['Close'] > market_data['SPY']['Close'].rolling(50).mean()).astype(int)
    
    # Add VIX (volatility index) if available
    try:
        vix_data = download_with_cache('^VIX', start_date, end_date, 'data/market_data')
        if not vix_data.empty:
            vix_close = vix_data['Close'].reindex(df_with_market.index, method='ffill')
            df_with_market['VIX'] = vix_close
            df_with_market['VIX_MA10'] = vix_close.rolling(10).mean()
            df_with_market['VIX_Ratio'] = vix_close / df_with_market['VIX_MA10']
    except Exception as e:
        print(f"Error downloading VIX data: {str(e)}")
    
    # Calculate cross-asset indicators (e.g., gold, dollar index, crude oil)
    for asset in ['GLD', 'UUP', 'USO']:
        try:
            asset_data = download_with_cache(asset, start_date, end_date, 'data/market_data')
            if not asset_data.empty:
                asset_close = asset_data['Close'].reindex(df_with_market.index, method='ffill')
                df_with_market[f'{asset}'] = asset_close
                df_with_market[f'{asset}_Return'] = asset_close.pct_change()
        except Exception as e:
            print(f"Error downloading {asset} data: {str(e)}")
    
    # Clean up - drop rows with NaNs or fill with appropriate values
    df_with_market.fillna(method='ffill', inplace=True)
    
    return df_with_market


def add_calendar_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar effects features to capture time-based patterns in stock behavior.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        
    Returns:
        pd.DataFrame: DataFrame with added calendar features.
    """
    # Create a copy of the DataFrame
    df_calendar = df.copy()
    
    # Ensure we have a DatetimeIndex
    if not isinstance(df_calendar.index, pd.DatetimeIndex):
        if 'Date' in df_calendar.columns:
            date_col = df_calendar['Date'].copy()
            date_series = pd.to_datetime(date_col)
        elif 'date' in df_calendar.columns:
            date_col = df_calendar['date'].copy()
            date_series = pd.to_datetime(date_col)
        else:
            # Try to find a date column
            date_cols = [col for col in df_calendar.columns if 'date' in col.lower()]
            if date_cols:
                date_col = df_calendar[date_cols[0]].copy()
                date_series = pd.to_datetime(date_col)
            else:
                raise ValueError("Could not find date column in DataFrame")
    else:
        date_series = df_calendar.index
    
    # Add day of week (0=Monday, 6=Sunday)
    df_calendar['Day_of_Week'] = date_series.dayofweek
    
    # Add day of month
    df_calendar['Day_of_Month'] = date_series.day
    
    # Add month
    df_calendar['Month'] = date_series.month
    
    # Add quarter
    df_calendar['Quarter'] = date_series.quarter
    
    # Is month end
    df_calendar['Is_Month_End'] = date_series.is_month_end.astype(int)
    
    # Is quarter end
    df_calendar['Is_Quarter_End'] = date_series.is_quarter_end.astype(int)
    
    # Is year end
    df_calendar['Is_Year_End'] = date_series.is_year_end.astype(int)
    
    # Is Monday
    df_calendar['Is_Monday'] = (date_series.dayofweek == 0).astype(int)
    
    # Is Friday
    df_calendar['Is_Friday'] = (date_series.dayofweek == 4).astype(int)
    
    return df_calendar


def combine_all_features(df: pd.DataFrame, include_market_data: bool = True) -> pd.DataFrame:
    """
    Combine all enhanced features into a single DataFrame.
    
    Args:
        df (pd.DataFrame): Original DataFrame with stock data.
        include_market_data (bool): Whether to include market data (requires yfinance).
        
    Returns:
        pd.DataFrame: DataFrame with all enhanced features.
    """
    from src.preprocessing.feature_engineering import add_technical_indicators
    
    # Start with base technical indicators
    df_enhanced = add_technical_indicators(df)
    
    # Add enhanced technical indicators
    df_enhanced = add_enhanced_technical_indicators(df_enhanced)
    
    # Add calendar effects
    df_enhanced = add_calendar_effects(df_enhanced)
    
    # Add market indicators if requested
    if include_market_data:
        try:
            df_enhanced = add_market_indicators(df_enhanced)
        except Exception as e:
            print(f"Warning: Could not add market indicators: {str(e)}")
            print("Continuing without market indicators...")
    
    # Drop rows with NaN values, which typically come from the lookback period
    # at the beginning of the dataset
    df_enhanced.dropna(inplace=True)
    
    return df_enhanced 