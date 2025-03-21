"""Run trading simulation for the Tesla ML Trading Agent."""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.utils.data_loader import load_stock_data, get_data_for_simulation
from src.preprocessing.feature_engineering import add_technical_indicators
from src.models.lstm_model import LSTMModel
from src.models.ensemble_model import EnsembleModel
from src.trading.trading_agent import TradingAgent, TradingSimulation
from src.utils.visualization import create_performance_summary


def load_trained_models(model_dir: str = "models"):
    """
    Load trained models for simulation.
    
    Args:
        model_dir (str): Directory containing trained models.
        
    Returns:
        dict: Dictionary containing loaded models and parameters.
    """
    # Load LSTM model
    lstm_path = os.path.join(model_dir, 'lstm_model')
    if os.path.exists(lstm_path):
        lstm_model = tf.keras.models.load_model(lstm_path)
    else:
        # Try the old path format
        old_lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        if os.path.exists(old_lstm_path):
            lstm_model = tf.keras.models.load_model(old_lstm_path)
        else:
            raise FileNotFoundError(f"LSTM model not found at {lstm_path} or {old_lstm_path}")
    
    # Load ensemble model
    ensemble_path = os.path.join(model_dir, 'ensemble_model.pkl')
    if os.path.exists(ensemble_path):
        with open(ensemble_path, 'rb') as f:
            ensemble_model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Ensemble model not found at {ensemble_path}")
    
    # Load feature columns
    feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
    if os.path.exists(feature_columns_path):
        with open(feature_columns_path, 'rb') as f:
            feature_columns = pickle.load(f)
    else:
        raise FileNotFoundError(f"Feature columns not found at {feature_columns_path}")
    
    # Load normalization parameters
    norm_params_path = os.path.join(model_dir, 'normalization_params.pkl')
    if os.path.exists(norm_params_path):
        with open(norm_params_path, 'rb') as f:
            norm_params = pickle.load(f)
    else:
        raise FileNotFoundError(f"Normalization parameters not found at {norm_params_path}")
    
    return {
        'lstm_model': lstm_model,
        'ensemble_model': ensemble_model,
        'feature_columns': feature_columns,
        'normalization_params': norm_params
    }


def run_simulation(data_path: str = "data/TSLA.csv",
                  model_dir: str = "models",
                  output_dir: str = "results",
                  start_date: str = None,
                  end_date: str = None,
                  initial_balance: float = 10000.0,
                  transaction_fee_pct: float = 0.001,
                  risk_tolerance: float = 0.02,
                  sequence_length: int = 20,
                  verbose: bool = True):
    """
    Run a trading simulation using trained models.
    
    Args:
        data_path (str): Path to the stock data CSV file.
        model_dir (str): Directory containing trained models.
        output_dir (str): Directory to save simulation results.
        start_date (str): Start date for simulation in format YYYY-MM-DD.
        end_date (str): End date for simulation in format YYYY-MM-DD.
        initial_balance (float): Initial account balance in USD.
        transaction_fee_pct (float): Transaction fee as a percentage.
        risk_tolerance (float): Risk tolerance for position sizing (0-1).
        sequence_length (int): Length of sequences for time series models.
        verbose (bool): Whether to print progress information.
        
    Returns:
        dict: Dictionary containing simulation results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("Loading data...")
    
    # Load stock data
    df = load_stock_data(data_path)
    
    # Filter data if dates are provided
    if start_date and end_date:
        df = get_data_for_simulation(df, start_date, end_date)
    
    if verbose:
        print(f"Simulation data shape: {df.shape}")
    
    if df.shape[0] < sequence_length + 5:
        raise ValueError(f"Not enough data for simulation. Need at least {sequence_length + 5} rows.")
    
    if verbose:
        print("Adding technical indicators...")
    
    # Add technical indicators
    sim_data_features = add_technical_indicators(df)
    
    # Check if features were added successfully
    if sim_data_features.shape[1] <= df.shape[1]:
        raise ValueError("Failed to add technical indicators to the data.")
    
    # Handle NaN values
    sim_data_features = sim_data_features.fillna(method='ffill').fillna(method='bfill')
    
    if verbose:
        print(f"Features data shape: {sim_data_features.shape}")
    
    if verbose:
        print("Loading models...")
    
    # Load trained models
    models_data = load_trained_models(model_dir)
    
    # Extract models and parameters
    lstm_model = models_data['lstm_model']
    ensemble_model = models_data['ensemble_model']
    feature_columns = models_data['feature_columns']
    norm_params = models_data['normalization_params']
    
    # Create trading agent
    agent = TradingAgent(
        lstm_model=lstm_model,
        ensemble_model=ensemble_model,
        initial_balance=initial_balance,
        transaction_fee_pct=transaction_fee_pct,
        risk_tolerance=risk_tolerance,
        feature_columns=feature_columns,
        normalization_params=norm_params
    )
    
    # Create simulation
    simulation = TradingSimulation(
        agent=agent,
        data=sim_data_features,
        feature_columns=feature_columns,
        price_column='Close',
        sequence_length=sequence_length
    )
    
    if verbose:
        print("Running simulation...")
    
    # Run simulation
    results_df = simulation.run_simulation()
    
    # Debug: Print DataFrame info
    print("Results DataFrame columns:", results_df.columns.tolist())
    if not results_df.empty:
        print("Results DataFrame first row:", results_df.iloc[0].to_dict())
    else:
        print("WARNING: Results DataFrame is empty!")
    
    if verbose:
        print("Simulation completed.")
    
    # Create a simple summary if the performance visualization fails
    summary = {
        'initial_value': initial_balance,
        'final_value': initial_balance,  # Default if no simulation results
        'total_return_pct': 0.0,
        'max_drawdown_pct': 0.0,
        'sharpe_ratio': 0.0,
        'buy_count': 0,
        'sell_count': 0,
        'total_days': 0
    }
    
    # Only create performance summary if we have results
    if not results_df.empty:
        if verbose:
            print("Creating performance summary...")
        try:
            summary = create_performance_summary(results_df, output_dir)
        except Exception as e:
            print(f"WARNING: Failed to create performance summary: {e}")
    
    if verbose:
        print("Final portfolio value:", summary['final_value'])
        print("Total return:", summary['total_return_pct'], "%")
        print("Results saved to:", output_dir)
    
    # Save simulation results
    if not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)
    
    return {
        'results_df': results_df,
        'summary': summary,
        'agent': agent
    }


def prepare_simulation_for_tesla_project():
    """
    Prepare simulation specifically for the Tesla project.
    Sets the simulation to run for the last week of March 2025.
    """
    # Project-specific simulation dates
    simulation_start = "2025-03-24"  # Monday of last week of March
    simulation_end = "2025-03-28"    # Friday of last week of March
    
    # For this project, we'll use historical data and "pretend" it's 2025
    # We'll use the most recent 5 days of data as a substitute
    df = load_stock_data("data/TSLA.csv")
    
    # Get the last 5 trading days from the available data
    last_date = df.index.max()
    first_date = last_date - pd.Timedelta(days=10)  # Get more days than needed to account for non-trading days
    recent_data = df.loc[first_date:last_date].tail(5)
    
    # Create a new DataFrame with dates adjusted to match project simulation dates
    simulation_dates = pd.date_range(start=simulation_start, end=simulation_end, freq='B')
    
    # Ensure we have exactly 5 dates
    if len(simulation_dates) != 5:
        print("Warning: Expected 5 trading days, but generated", len(simulation_dates))
    
    # Map the recent data to the simulation dates
    simulation_data = recent_data.copy()
    simulation_data.index = simulation_dates[:len(recent_data)]
    
    # Save the simulation data to a temporary file
    sim_data_path = "data/simulation_data.csv"
    simulation_data = simulation_data.reset_index()
    simulation_data = simulation_data.rename(columns={'index': 'Date'})
    simulation_data.to_csv(sim_data_path, index=False)
    
    print(f"Created simulation data for {simulation_start} to {simulation_end}")
    print(f"Data saved to {sim_data_path}")
    
    return sim_data_path, simulation_start, simulation_end


def generate_daily_advice(data_path: str = "data/TSLA.csv",
                         model_dir: str = "models",
                         date: str = None,
                         sequence_length: int = 10,
                         initial_balance: float = 10000.0,
                         shares_owned: int = 0):
    """
    Generate trading advice for a specific day based on trained models.
    
    Args:
        data_path (str): Path to the CSV file containing stock data.
        model_dir (str): Directory containing trained models.
        date (str): Date for advice in format YYYY-MM-DD.
        sequence_length (int): Length of sequences for time series models.
        initial_balance (float): Current account balance in USD.
        shares_owned (int): Current number of shares owned.
        
    Returns:
        dict: Trading advice.
    """
    # Load data
    df = load_stock_data(data_path)
    
    # Set date
    if not date:
        date = df.index.max().strftime('%Y-%m-%d')
    
    # Convert date to datetime
    date = pd.to_datetime(date)
    
    # Prepare data
    start_date = date - timedelta(days=sequence_length * 2)
    data = get_data_for_simulation(df, start_date=start_date, end_date=date)
    
    # Add technical indicators
    data_features = add_technical_indicators(data)
    
    # Load models and parameters
    models_data = load_trained_models(model_dir)
    
    # Create trading agent
    agent = TradingAgent(
        initial_balance=initial_balance,
        models={
            'lstm': models_data['lstm_model'],
            'ensemble': models_data['ensemble_model']
        }
    )
    
    # Update agent state to match provided values
    agent.balance = initial_balance
    agent.shares = shares_owned
    
    # Prepare features for the current day
    if len(data_features) < sequence_length:
        raise ValueError(f"Not enough data for prediction. Need at least {sequence_length} days.")
    
    feature_columns = models_data['feature_columns']
    
    # Get the sequence of data
    sequence = data_features.iloc[-sequence_length:][feature_columns].values
    
    # Reshape for model input (add batch dimension)
    sequence = np.expand_dims(sequence, axis=0)
    
    # Get current price
    current_price = data_features.iloc[-1]['Close']
    
    # Make prediction
    predicted_price, confidence = agent.predict(sequence)
    
    # Generate advice
    advice = agent.generate_trading_advice(current_price, predicted_price, confidence)
    
    return advice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Tesla stock trading simulation')
    parser.add_argument('--data', type=str, default='data/TSLA.csv', 
                        help='Path to the CSV file containing stock data')
    parser.add_argument('--models', type=str, default='models', 
                        help='Directory containing trained models')
    parser.add_argument('--output', type=str, default='results', 
                        help='Directory to save simulation results')
    parser.add_argument('--start-date', type=str, 
                        help='Start date for simulation in format YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, 
                        help='End date for simulation in format YYYY-MM-DD')
    parser.add_argument('--balance', type=float, default=10000.0, 
                        help='Initial account balance in USD')
    parser.add_argument('--fee', type=float, default=0.01, 
                        help='Transaction fee as a percentage')
    parser.add_argument('--risk', type=float, default=0.2, 
                        help='Risk tolerance for position sizing (0-1)')
    parser.add_argument('--seq-length', type=int, default=10, 
                        help='Length of sequences for time series models')
    parser.add_argument('--project', action='store_true', 
                        help='Run simulation specifically for Tesla project dates')
    parser.add_argument('--daily-advice', action='store_true', 
                        help='Generate trading advice for today')
    parser.add_argument('--quiet', action='store_true', 
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    if args.daily_advice:
        # Generate advice for today
        advice = generate_daily_advice(
            data_path=args.data,
            model_dir=args.models,
            sequence_length=args.seq_length,
            initial_balance=args.balance
        )
        
        print("\nTESLA TRADING AGENT DAILY ADVICE")
        print("================================")
        print(f"Date: {advice['date']}")
        print(f"Current Price: ${advice['current_price']:.2f}")
        print(f"Predicted Price: ${advice['predicted_price']:.2f}")
        print(f"Confidence: {advice['confidence']:.2f}")
        print(f"Expected Change: {advice['expected_pct_change']:.2f}%")
        print(f"Recommended Action: {advice['action'].upper()}")
        
        if advice['action'] == 'buy':
            print(f"Buy: ${advice['amount']:.2f} ({advice['position_size']} shares)")
        elif advice['action'] == 'sell':
            print(f"Sell: {advice['position_size']} shares")
    
    elif args.project:
        # Run simulation specifically for Tesla project
        data_path, start_date, end_date = prepare_simulation_for_tesla_project()
        
        run_simulation(
            data_path=data_path,
            model_dir=args.models,
            output_dir=args.output,
            start_date=start_date,
            end_date=end_date,
            initial_balance=args.balance,
            transaction_fee_pct=args.fee,
            risk_tolerance=args.risk,
            sequence_length=args.seq_length,
            verbose=not args.quiet
        )
    
    else:
        # Run regular simulation
        run_simulation(
            data_path=args.data,
            model_dir=args.models,
            output_dir=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_balance=args.balance,
            transaction_fee_pct=args.fee,
            risk_tolerance=args.risk,
            sequence_length=args.seq_length,
            verbose=not args.quiet
        ) 