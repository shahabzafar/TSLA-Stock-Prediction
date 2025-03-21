"""Training script for machine learning models for stock price prediction."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from utils.data_loader import (
    load_stock_data, 
    get_training_data, 
    add_technical_indicators, 
    prepare_lstm_data, 
    prepare_ensemble_data, 
    normalize_data
)
from models.lstm_model import create_lstm_model
from models.ensemble_model import create_ensemble_model


def train_models(
    data_path: str, 
    output_dir: str = 'models',
    test_size: float = 0.2,
    prediction_horizon: int = 1,
    sequence_length: int = 20,
    lstm_epochs: int = 10,  # Reduced epochs for faster training
    verbose: int = 1
) -> dict:
    """
    Train machine learning models for stock price prediction.
    
    Args:
        data_path (str): Path to the CSV file containing stock data.
        output_dir (str): Directory to save the trained models.
        test_size (float): Proportion of data to use for testing.
        prediction_horizon (int): Number of days ahead to predict.
        sequence_length (int): Length of sequences for time series models.
        lstm_epochs (int): Number of epochs for LSTM training.
        verbose (int): Verbosity level.
        
    Returns:
        dict: Dictionary containing the trained models and metrics.
    """
    print("Starting model training process...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load stock data
    print(f"Loading stock data from {data_path}...")
    df = load_stock_data(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Set end_date to the last date in the dataset
    end_date = df.index[-1]
    
    # Prepare training data - use a smaller lookback period
    print(f"Preparing training data...")
    training_data = get_training_data(df, end_date, lookback_days=365)  # Reduced to 1 year of data
    print(f"Training data shape: {training_data.shape}")
    
    # Add technical indicators
    print(f"Adding technical indicators...")
    enhanced_data = add_technical_indicators(training_data)
    print(f"Enhanced data shape: {enhanced_data.shape}")
    
    # Create target variables
    print(f"Creating target variables...")
    if prediction_horizon > 1:
        target_column = f'Next_{prediction_horizon}D_Return'
    else:
        target_column = 'Next_Day_Return'
    
    # Define feature columns
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20',
        'EMA_5', 'EMA_10', 'EMA_20',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'ROC', 'ATR', 'Volume_ROC', 'OBV'
    ]
    
    # Save feature columns for later use
    with open(os.path.join(output_dir, 'feature_columns.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)
    
    # Remove rows with NaN values
    enhanced_data = enhanced_data.dropna()
    print(f"After dropping NaN values, shape: {enhanced_data.shape}")
    
    # Prepare data for training
    print("Preparing data for LSTM model...")
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = prepare_lstm_data(
        enhanced_data, feature_columns, target_column, sequence_length, test_size=test_size
    )
    print(f"LSTM data shapes: X_train={X_train_lstm.shape}, y_train={y_train_lstm.shape}")
    
    # Normalize data
    print("Normalizing data...")
    X_train_lstm_norm, X_val_lstm_norm, X_test_lstm_norm, norm_params = normalize_data(
        X_train_lstm, X_val_lstm, X_test_lstm
    )
    
    # Save normalization parameters
    with open(os.path.join(output_dir, 'normalization_params.pkl'), 'wb') as f:
        pickle.dump(norm_params, f)
    
    # Train LSTM model
    print("Training LSTM model...")
    lstm_model, lstm_history = create_lstm_model(
        X_train_lstm_norm, y_train_lstm,
        X_val_lstm_norm, y_val_lstm,
        lstm_units=64,
        dropout_rate=0.2,
        epochs=lstm_epochs,
        batch_size=32,
        patience=5  # Reduced patience for faster training
    )
    
    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    lstm_metrics = lstm_model.evaluate(X_test_lstm_norm, y_test_lstm)
    
    # Save LSTM model
    print(f"Saving LSTM model to {output_dir}...")
    lstm_model.save(os.path.join(output_dir, 'lstm_model'))
    
    # Prepare data for ensemble models
    print("Preparing data for ensemble models...")
    X_train_ens, y_train_ens, X_val_ens, y_val_ens, X_test_ens, y_test_ens = prepare_ensemble_data(
        enhanced_data, feature_columns, target_column, test_size=test_size
    )
    print(f"Ensemble data shapes: X_train={X_train_ens.shape}, y_train={y_train_ens.shape}")
    
    # Normalize data
    X_train_ens_norm, X_val_ens_norm, X_test_ens_norm, _ = normalize_data(
        X_train_ens, X_val_ens, X_test_ens
    )
    
    # Train ensemble model
    print("Training ensemble model...")
    ensemble_model, ensemble_metrics = create_ensemble_model(
        X_train_ens_norm, y_train_ens,
        X_val_ens_norm, y_val_ens,
        X_test_ens_norm, y_test_ens
    )
    
    # Save ensemble model
    print(f"Saving ensemble model to {output_dir}...")
    ensemble_model.save(os.path.join(output_dir, 'ensemble_model.pkl'))
    
    # Return results
    results = {
        'lstm_model': lstm_model,
        'ensemble_model': ensemble_model,
        'lstm_metrics': lstm_metrics,
        'ensemble_metrics': ensemble_metrics,
        'feature_columns': feature_columns,
        'normalization_params': norm_params
    }
    
    print("Training complete!")
    print(f"LSTM Metrics: {lstm_metrics}")
    print(f"Ensemble Metrics: {ensemble_metrics}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for stock price prediction")
    parser.add_argument("--data", type=str, required=True, help="Path to the stock data CSV file")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--horizon", type=int, default=1, help="Number of days ahead to predict")
    parser.add_argument("--seq-length", type=int, default=20, help="Sequence length for LSTM model")
    parser.add_argument("--lstm-epochs", type=int, default=10, help="Number of epochs for LSTM training")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()
    
    train_models(
        data_path=args.data,
        output_dir=args.models_dir,
        test_size=args.test_size,
        prediction_horizon=args.horizon,
        sequence_length=args.seq_length,
        lstm_epochs=args.lstm_epochs,
        verbose=args.verbose
    ) 