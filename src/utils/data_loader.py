"""Data loader utility for Tesla ML Trading Agent."""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timedelta


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing stock data.
        
    Returns:
        DataFrame with stock data, with Date as the index.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df


def get_data_for_simulation(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get a subset of data for a specific time period for simulation.
    
    Args:
        data: DataFrame with stock data.
        start_date: Start date for the simulation period (format: 'YYYY-MM-DD').
        end_date: End date for the simulation period (format: 'YYYY-MM-DD').
        
    Returns:
        DataFrame with stock data for the specified period.
    """
    # Convert dates to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Filter the data for the specified period
    return data.loc[start_date:end_date]


def get_training_data(data: pd.DataFrame, end_date: str, lookback_days: int) -> pd.DataFrame:
    """
    Get data for training models, up to a specific end date with a lookback period.
    
    Args:
        data: DataFrame with stock data.
        end_date: End date for the training data (format: 'YYYY-MM-DD').
        lookback_days: Number of days to look back for training data.
        
    Returns:
        DataFrame with stock data for training.
    """
    # Convert end_date to datetime if it's a string
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate the start date (business days)
    start_date = end_date - pd.tseries.offsets.BDay(lookback_days)
    
    # Filter the data for the specified period
    return data.loc[start_date:end_date]


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.
    
    Args:
        data: DataFrame with stock data.
        
    Returns:
        DataFrame with added technical indicators.
    """
    df = data.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # Price Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Average True Range (ATR)
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Volume indicators
    df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Target variables (next day's return)
    df['Next_Day_Return'] = df['Close'].pct_change(periods=1).shift(-1)
    df['Next_Day_Direction'] = np.where(df['Next_Day_Return'] > 0, 1, 0)
    
    # Additional target variables (multi-day returns)
    df['Next_3D_Return'] = df['Close'].pct_change(periods=3).shift(-3)
    df['Next_5D_Return'] = df['Close'].pct_change(periods=5).shift(-5)
    
    # Drop unnecessary columns that aren't in our feature list
    columns_to_keep = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20',
        'EMA_5', 'EMA_10', 'EMA_20',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'ROC', 'ATR', 'Volume_ROC', 'OBV',
        'Next_Day_Return', 'Next_Day_Direction',
        'Next_3D_Return', 'Next_5D_Return'
    ]
    
    return df[columns_to_keep]


def prepare_lstm_data(
        data: pd.DataFrame, 
        feature_columns: List[str], 
        target_column: str, 
        sequence_length: int, 
        test_size: float = 0.2, 
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM model training, validation, and testing.
    
    Args:
        data: DataFrame with stock data and features.
        feature_columns: List of column names to use as features.
        target_column: Name of the column to predict.
        sequence_length: Number of time steps to look back.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of data to use for validation.
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Arrays for training, validation, and testing.
    """
    # Extract features and target
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Create sequences for LSTM
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences).reshape(-1, 1)
    
    # Split into train, validation, and test sets
    train_size = int(len(X_sequences) * (1 - test_size - val_size))
    val_size = int(len(X_sequences) * val_size)
    
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    
    X_val = X_sequences[train_size:train_size+val_size]
    y_val = y_sequences[train_size:train_size+val_size]
    
    X_test = X_sequences[train_size+val_size:]
    y_test = y_sequences[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_ensemble_data(
        data: pd.DataFrame, 
        feature_columns: List[str], 
        target_column: str, 
        test_size: float = 0.2, 
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for ensemble model training, validation, and testing.
    
    Args:
        data: DataFrame with stock data and features.
        feature_columns: List of column names to use as features.
        target_column: Name of the column to predict.
        test_size: Fraction of data to use for testing.
        val_size: Fraction of data to use for validation.
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Arrays for training, validation, and testing.
    """
    # Drop rows with NaN values
    data = data.dropna()
    
    # Extract features and target
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Split into train, validation, and test sets
    train_size = int(len(X) * (1 - test_size - val_size))
    val_size = int(len(X) * val_size)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Normalize data using min-max scaling.
    
    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Testing features.
        
    Returns:
        X_train_norm, X_val_norm, X_test_norm, normalize_params: Normalized arrays and normalization parameters.
    """
    # For LSTM data (3D arrays)
    if len(X_train.shape) == 3:
        # Reshape to 2D for normalization
        n_samples_train, n_timesteps, n_features = X_train.shape
        n_samples_val = X_val.shape[0]
        n_samples_test = X_test.shape[0]
        
        X_train_reshaped = X_train.reshape(n_samples_train * n_timesteps, n_features)
        X_val_reshaped = X_val.reshape(n_samples_val * n_timesteps, n_features)
        X_test_reshaped = X_test.reshape(n_samples_test * n_timesteps, n_features)
        
        # Calculate min and max
        X_min = X_train_reshaped.min(axis=0)
        X_max = X_train_reshaped.max(axis=0)
        
        # Normalize
        X_train_norm = (X_train_reshaped - X_min) / (X_max - X_min)
        X_val_norm = (X_val_reshaped - X_min) / (X_max - X_min)
        X_test_norm = (X_test_reshaped - X_min) / (X_max - X_min)
        
        # Reshape back to 3D
        X_train_norm = X_train_norm.reshape(n_samples_train, n_timesteps, n_features)
        X_val_norm = X_val_norm.reshape(n_samples_val, n_timesteps, n_features)
        X_test_norm = X_test_norm.reshape(n_samples_test, n_timesteps, n_features)
    
    # For ensemble data (2D arrays)
    else:
        # Calculate min and max
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        
        # Normalize
        X_train_norm = (X_train - X_min) / (X_max - X_min)
        X_val_norm = (X_val - X_min) / (X_max - X_min)
        X_test_norm = (X_test - X_min) / (X_max - X_min)
    
    # Store normalization parameters
    normalize_params = {
        'min': X_min,
        'max': X_max
    }
    
    return X_train_norm, X_val_norm, X_test_norm, normalize_params


def normalize_new_data(data: np.ndarray, normalize_params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Normalize new data using saved normalization parameters.
    
    Args:
        data: New data to normalize.
        normalize_params: Dictionary containing min and max values for normalization.
        
    Returns:
        Normalized data.
    """
    X_min = normalize_params['min']
    X_max = normalize_params['max']
    
    # For LSTM data (3D arrays)
    if len(data.shape) == 3:
        # Reshape to 2D for normalization
        n_samples, n_timesteps, n_features = data.shape
        data_reshaped = data.reshape(n_samples * n_timesteps, n_features)
        
        # Normalize
        data_norm = (data_reshaped - X_min) / (X_max - X_min)
        
        # Reshape back to 3D
        data_norm = data_norm.reshape(n_samples, n_timesteps, n_features)
    
    # For ensemble data (2D arrays)
    else:
        # Normalize
        data_norm = (data - X_min) / (X_max - X_min)
    
    return data_norm


def prepare_sequence_for_prediction(
        data: pd.DataFrame, 
        feature_columns: List[str], 
        sequence_length: int, 
        normalize_params: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
    """
    Prepare a sequence for making predictions.
    
    Args:
        data: DataFrame with stock data and features.
        feature_columns: List of column names to use as features.
        sequence_length: Number of time steps to look back.
        normalize_params: Dictionary containing min and max values for normalization.
        
    Returns:
        Sequence ready for prediction with shape (1, sequence_length, n_features).
    """
    # Extract features
    X = data[feature_columns].values
    
    # Create sequence
    if len(X) < sequence_length:
        raise ValueError(f"Not enough data for sequence length {sequence_length}. Needed: {sequence_length}, Available: {len(X)}")
    
    sequence = X[-sequence_length:].reshape(1, sequence_length, len(feature_columns))
    
    # Normalize if necessary
    if normalize_params is not None:
        sequence = normalize_new_data(sequence, normalize_params)
    
    return sequence 