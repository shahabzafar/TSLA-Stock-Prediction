"""Main script for the Tesla ML Trading Agent project."""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.train_models import train_models
from src.run_simulation import run_simulation, prepare_simulation_for_tesla_project


def main(args):
    """
    Main function to run the Tesla ML Trading Agent project.
    
    Args:
        args: Command line arguments.
    """
    print("\nTESLA ML TRADING AGENT")
    print("=====================\n")
    
    # Create output directories if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Step 1: Train models if needed
    if args.train or not os.path.exists(os.path.join(args.models_dir, 'lstm_model')):
        print("Step 1: Training models...")
        train_models(
            data_path=args.data,
            output_dir=args.models_dir,
            test_size=args.test_size,
            prediction_horizon=args.horizon,
            sequence_length=args.seq_length,
            lstm_epochs=args.epochs,
            verbose=1 if not args.quiet else 0
        )
        print("Model training completed.")
    else:
        print("Step 1: Using existing trained models.")
    
    # Step 2: Run simulation
    print("\nStep 2: Running trading simulation...")
    
    if args.project:
        # Run simulation specifically for Tesla project
        data_path, start_date, end_date = prepare_simulation_for_tesla_project()
        
        simulation_results = run_simulation(
            data_path=data_path,
            model_dir=args.models_dir,
            output_dir=args.results_dir,
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
        simulation_results = run_simulation(
            data_path=args.data,
            model_dir=args.models_dir,
            output_dir=args.results_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_balance=args.balance,
            transaction_fee_pct=args.fee,
            risk_tolerance=args.risk,
            sequence_length=args.seq_length,
            verbose=not args.quiet
        )
    
    # Step 3: Print summary
    summary = simulation_results['summary']
    
    print("\nSimulation Results Summary")
    print("=========================")
    print(f"Initial Portfolio Value: ${summary['initial_value']:.2f}")
    print(f"Final Portfolio Value: ${summary['final_value']:.2f}")
    print(f"Total Return: {summary['total_return_pct']:.2f}%")
    print(f"Maximum Drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Buy Trades: {summary['buy_count']}")
    print(f"Sell Trades: {summary['sell_count']}")
    print(f"Total Trading Days: {summary['total_days']}")
    
    print("\nResults have been saved to:", args.results_dir)
    
    return simulation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tesla ML Trading Agent')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data/TSLA.csv', 
                        help='Path to the CSV file containing stock data')
    
    # Training parameters
    parser.add_argument('--train', action='store_true', 
                        help='Retrain models even if they already exist')
    parser.add_argument('--test-size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--horizon', type=int, default=1, 
                        help='Number of days ahead to predict')
    parser.add_argument('--seq-length', type=int, default=20, 
                        help='Length of sequences for time series models')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs for LSTM training')
    
    # Simulation parameters
    parser.add_argument('--start-date', type=str, 
                        help='Start date for simulation in format YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, 
                        help='End date for simulation in format YYYY-MM-DD')
    parser.add_argument('--balance', type=float, default=10000.0, 
                        help='Initial account balance in USD')
    parser.add_argument('--fee', type=float, default=0.001, 
                        help='Transaction fee as a percentage')
    parser.add_argument('--risk', type=float, default=0.02, 
                        help='Risk tolerance for position sizing (0-1)')
    
    # Project-specific parameters
    parser.add_argument('--project', action='store_true', 
                        help='Run simulation specifically for Tesla project dates')
    
    # Output parameters
    parser.add_argument('--models-dir', type=str, default='models', 
                        help='Directory to save/load trained models')
    parser.add_argument('--results-dir', type=str, default='results', 
                        help='Directory to save simulation results')
    
    # Other parameters
    parser.add_argument('--quiet', action='store_true', 
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    main(args) 