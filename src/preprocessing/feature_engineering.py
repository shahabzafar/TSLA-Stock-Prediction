"""Feature engineering for the Tesla ML Trading Agent."""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_features = df.copy()
    
    # Moving averages
    df_features['MA5'] = df['Close'].rolling(window=5).mean()
    df_features['MA10'] = df['Close'].rolling(window=10).mean()
    df_features['MA20'] = df['Close'].rolling(window=20).mean()
    df_features['MA50'] = df['Close'].rolling(window=50).mean()
    df_features['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential moving averages
    df_features['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df_features['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df_features['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative strength index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df_features['MACD'] = ema12 - ema26
    df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()
    df_features['MACD_Hist'] = df_features['MACD'] - df_features['MACD_Signal']
    
    # Bollinger Bands
    df_features['MA20_std'] = df['Close'].rolling(window=20).std()
    df_features['Bollinger_Upper'] = df_features['MA20'] + (df_features['MA20_std'] * 2)
    df_features['Bollinger_Lower'] = df_features['MA20'] - (df_features['MA20_std'] * 2)
    df_features['Bollinger_Width'] = (df_features['Bollinger_Upper'] - df_features['Bollinger_Lower']) / df_features['MA20']
    df_features['Bollinger_pct_b'] = (df_features['Close'] - df_features['Bollinger_Lower']) / (df_features['Bollinger_Upper'] - df_features['Bollinger_Lower'])
    
    # Price momentum
    df_features['Price_Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    
    # Volume features
    df_features['Volume_1d_Change'] = df['Volume'] / df['Volume'].shift(1)
    df_features['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df_features['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    df_features['Volume_Ratio'] = df['Volume'] / df_features['Volume_MA10']
    
    # Price rate of change
    df_features['ROC_5'] = (df['Close'] / df['Close'].shift(5) - 1) * 100
    df_features['ROC_10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
    df_features['ROC_20'] = (df['Close'] / df['Close'].shift(20) - 1) * 100
    
    # Gaps
    df_features['Gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    # Daily returns
    df_features['Daily_Return'] = df['Close'] / df['Close'].shift(1) - 1
    
    # Log returns
    df_features['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (standard deviation of returns)
    df_features['Volatility_5'] = df_features['Log_Return'].rolling(window=5).std()
    df_features['Volatility_10'] = df_features['Log_Return'].rolling(window=10).std()
    df_features['Volatility_20'] = df_features['Log_Return'].rolling(window=20).std()
    
    # Advanced Oscillators
    # Stochastic Oscillator
    df_features['Stoch_K'] = 100 * ((df_features['Close'] - df_features['Low'].rolling(window=14).min()) / 
                                    (df_features['High'].rolling(window=14).max() - df_features['Low'].rolling(window=14).min()))
    df_features['Stoch_D'] = df_features['Stoch_K'].rolling(window=3).mean()
    
    # Triple Exponential Average (TEMA)
    ema1 = df['Close'].ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    ema3 = ema2.ewm(span=9, adjust=False).mean()
    df_features['TEMA'] = 3 * ema1 - 3 * ema2 + ema3
    
    # Average True Range (ATR)
    true_range = pd.DataFrame()
    true_range['tr1'] = abs(df['High'] - df['Low'])
    true_range['tr2'] = abs(df['High'] - df['Close'].shift(1))
    true_range['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    true_range['TR'] = true_range[['tr1', 'tr2', 'tr3']].max(axis=1)
    df_features['ATR'] = true_range['TR'].rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    obv = 0
    obv_list = []
    df_reset = df.reset_index(drop=True)  # Reset index to avoid timestamp issues
    
    for i in range(len(df_reset)):
        if i > 0:
            if df_reset['Close'].iloc[i] > df_reset['Close'].iloc[i-1]:
                obv += df_reset['Volume'].iloc[i]
            elif df_reset['Close'].iloc[i] < df_reset['Close'].iloc[i-1]:
                obv -= df_reset['Volume'].iloc[i]
        obv_list.append(obv)
    
    df_features['OBV'] = obv_list
    df_features['OBV_ROC'] = df_features['OBV'].pct_change(periods=10) * 100
    
    # Add profit-maximizing features
    
    # Price Acceleration - capture rapid price movements
    df_features['Price_Acceleration'] = df_features['Close'].diff().diff()
    
    # Mean Reversion Indicators
    df_features['Deviation_From_MA50'] = (df_features['Close'] - df_features['MA50']) / df_features['MA50']
    df_features['Deviation_From_MA200'] = (df_features['Close'] - df_features['MA200']) / df_features['MA200']
    
    # Volume Breakout Signals - capture major moves
    df_features['Volume_MA20'] = df_features['Volume'].rolling(window=20).mean()
    df_features['Volume_Breakout'] = df_features['Volume'] / df_features['Volume_MA20']
    
    # RSI Extremes - better identify overbought/oversold conditions
    df_features['RSI_Extreme'] = 0
    df_features.loc[df_features['RSI'] > 80, 'RSI_Extreme'] = 1  # Strong overbought
    df_features.loc[df_features['RSI'] < 20, 'RSI_Extreme'] = -1  # Strong oversold
    
    # Profitability trend indicators
    df_features['Close_Shifted'] = df_features['Close'].shift(1)
    df_features['Daily_Return'] = (df_features['Close'] / df_features['Close_Shifted'] - 1) * 100
    df_features['Return_MA5'] = df_features['Daily_Return'].rolling(window=5).mean()
    df_features['Return_MA10'] = df_features['Daily_Return'].rolling(window=10).mean()
    df_features['Momentum_Signal'] = (df_features['Return_MA5'] > df_features['Return_MA10']).astype(int)
    
    # Gap detection - often indicates significant moves
    df_features['Gap_Up'] = (df_features['Open'] > df_features['Close_Shifted']).astype(int)
    df_features['Gap_Down'] = (df_features['Open'] < df_features['Close_Shifted']).astype(int)
    df_features['Gap_Size'] = (df_features['Open'] - df_features['Close_Shifted']) / df_features['Close_Shifted'] * 100
    
    # Volatility-based position sizing signals
    df_features['Volatility_20d'] = df_features['Daily_Return'].rolling(window=20).std()
    df_features['Volatility_Signal'] = 0
    df_features.loc[df_features['Volatility_20d'] < df_features['Volatility_20d'].rolling(window=60).mean(), 'Volatility_Signal'] = 1
    df_features.loc[df_features['Volatility_20d'] > df_features['Volatility_20d'].rolling(window=60).mean() * 1.5, 'Volatility_Signal'] = -1
    
    # Chaikin Money Flow (CMF)
    mf_multiplier = ((df_features['Close'] - df_features['Low']) - (df_features['High'] - df_features['Close'])) / (df_features['High'] - df_features['Low']).replace(0, np.nan)
    mf_volume = mf_multiplier * df_features['Volume']
    df_features['CMF'] = mf_volume.rolling(window=20).sum() / df_features['Volume'].rolling(window=20).sum()
    
    # Vortex Indicator
    df_features['TR'] = true_range['TR']
    pdm = (df['High'] - df['High'].shift(1)).clip(lower=0)
    ndm = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    df_features['PDI14'] = pdm.rolling(window=14).sum() / df_features['TR'].rolling(window=14).sum() * 100
    df_features['NDI14'] = ndm.rolling(window=14).sum() / df_features['TR'].rolling(window=14).sum() * 100
    
    # Calendar Effects
    # Extract date components
    if isinstance(df.index, pd.DatetimeIndex):
        date_index = df.index
    else:
        date_index = pd.to_datetime(df.index)
    
    df_features['Day_of_Week'] = date_index.dayofweek
    df_features['Day_of_Month'] = date_index.day
    df_features['Month'] = date_index.month
    df_features['Quarter'] = date_index.quarter
    df_features['Is_Month_End'] = date_index.is_month_end.astype(int)
    df_features['Is_Month_Start'] = date_index.is_month_start.astype(int)
    
    # Historical volatility context
    # 10-day historical volatility
    df_features['HV10'] = df_features['Log_Return'].rolling(window=10).std() * np.sqrt(252) * 100
    # 30-day historical volatility
    df_features['HV30'] = df_features['Log_Return'].rolling(window=30).std() * np.sqrt(252) * 100
    
    # Price Range Features
    df_features['Daily_Range'] = (df['High'] - df['Low']) / df['Open'] * 100  # Daily range as percentage
    df_features['Range_MA5'] = df_features['Daily_Range'].rolling(window=5).mean()
    
    # Market Regime Features
    df_features['Bull_Bear'] = (df_features['Close'] > df_features['MA200']).astype(int)  # 1 if bullish (above 200MA), 0 if bearish
    df_features['Price_Trend'] = ((df_features['MA5'] - df_features['MA20']) / df_features['MA20'] * 100)  # Short-term trend indicator
    
    # Trend Persistence
    df_features['Trend_Persistence'] = ((df_features['Close'] - df_features['MA50']) / df_features['MA50'] * 100)
    
    # Drop NaN values
    df_features.dropna(inplace=True)
    
    return df_features


def create_target_variable(df: pd.DataFrame, prediction_horizon: int = 1) -> pd.DataFrame:
    """
    Create target variables for price prediction.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        prediction_horizon (int): Number of days ahead to predict.
        
    Returns:
        pd.DataFrame: DataFrame with added target variables.
    """
    df_with_target = df.copy()
    
    # Future price
    df_with_target[f'Future_Close_{prediction_horizon}'] = df['Close'].shift(-prediction_horizon)
    
    # Price movement (up/down)
    df_with_target[f'Price_Up_{prediction_horizon}'] = (
        df_with_target[f'Future_Close_{prediction_horizon}'] > df['Close']).astype(int)
    
    # Percentage change
    df_with_target[f'Return_{prediction_horizon}'] = (
        df_with_target[f'Future_Close_{prediction_horizon}'] / df['Close'] - 1)
    
    # Log return
    df_with_target[f'Log_Return_{prediction_horizon}'] = np.log(
        df_with_target[f'Future_Close_{prediction_horizon}'] / df['Close'])
    
    # Drop NaN values (will be at the end due to shifting)
    df_with_target.dropna(inplace=True)
    
    return df_with_target


def prepare_data_for_training(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    sequence_length: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for training machine learning models.
    
    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        target_column (str): Name of the target column.
        feature_columns (List[str], optional): List of feature column names. If None, use all columns except the target.
        test_size (float): Proportion of data to use for testing.
        sequence_length (int): Length of sequences for time series models.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    # Select features
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column and not col.startswith('Future_')]
    
    # Convert to numpy arrays
    X = df[feature_columns].values
    y = df[target_column].values
    
    # Calculate split point
    split_idx = int(len(X) * (1 - test_size))
    
    # Create sequences for time series models
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Split into train and test sets
    split_idx_seq = int(len(X_seq) * (1 - test_size))
    X_train = X_seq[:split_idx_seq]
    X_test = X_seq[split_idx_seq:]
    y_train = y_seq[:split_idx_seq]
    y_test = y_seq[split_idx_seq:]
    
    return X_train, X_test, y_train, y_test


def normalize_data(train_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data for training machine learning models.
    
    Args:
        train_data (np.ndarray): Training data.
        test_data (np.ndarray): Test data.
        
    Returns:
        Tuple: (normalized_train_data, normalized_test_data, data_mean, data_std)
    """
    # Calculate mean and standard deviation from training data
    data_mean = np.mean(train_data, axis=0)
    data_std = np.std(train_data, axis=0)
    
    # Avoid division by zero
    data_std = np.where(data_std == 0, 1, data_std)
    
    # Normalize data
    normalized_train_data = (train_data - data_mean) / data_std
    normalized_test_data = (test_data - data_mean) / data_std
    
    return normalized_train_data, normalized_test_data, data_mean, data_std 