"""Trading agent for Tesla stock trading simulation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import os
import json

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMModel
from models.ensemble_model import EnsembleModel


class TradingAgent:
    """Trading agent for Tesla stock trading simulation."""
    
    def __init__(self, 
                lstm_model: Any = None,
                ensemble_model: Any = None,
                initial_balance: float = 10000.0,
                transaction_fee_pct: float = 0.001,
                risk_tolerance: float = 0.02,
                feature_columns: List[str] = None,
                normalization_params: Dict[str, Any] = None,
                models: Optional[Dict[str, Any]] = None):
        """
        Initialize the trading agent.
        
        Args:
            lstm_model (Any): The LSTM model for prediction.
            ensemble_model (Any): The ensemble model for prediction.
            initial_balance (float): Initial account balance in USD.
            transaction_fee_pct (float): Transaction fee as a percentage.
            risk_tolerance (float): Risk tolerance for position sizing (0-1).
            feature_columns (List[str]): Columns to use as features.
            normalization_params (Dict[str, Any]): Parameters for normalizing data.
            models (Dict[str, Any], optional): Dictionary of trained models (legacy parameter).
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares = 0
        self.transaction_fee_pct = transaction_fee_pct
        self.risk_tolerance = risk_tolerance
        self.feature_columns = feature_columns or []
        self.normalization_params = normalization_params or {}
        
        # Set up models
        self.models = models or {}
        if lstm_model is not None:
            self.models['lstm'] = lstm_model
        if ensemble_model is not None:
            self.models['ensemble'] = ensemble_model
            
        self.transaction_history = []
        self.portfolio_history = []
        
        # Performance metrics
        self.cumulative_return = 0.0
        self.max_drawdown = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        
    def add_model(self, model_name: str, model: Any) -> None:
        """
        Add a model to the trading agent.
        
        Args:
            model_name (str): Name of the model.
            model (Any): The model object.
        """
        self.models[model_name] = model
    
    def _calculate_position_size(self, confidence: float, current_price: float) -> int:
        """
        Calculate the position size based on confidence and risk tolerance.
        
        Args:
            confidence (float): Confidence in the prediction (0-1).
            current_price (float): Current stock price.
            
        Returns:
            int: Number of shares to buy.
        """
        # Calculate the amount to invest based on risk tolerance and confidence
        amount_to_invest = self.balance * self.risk_tolerance * confidence
        
        # Calculate number of shares
        num_shares = int(amount_to_invest / current_price)
        
        return num_shares
    
    def _apply_transaction_fee(self, amount: float) -> float:
        """
        Apply transaction fee to the amount.
        
        Args:
            amount (float): Amount of the transaction.
            
        Returns:
            float: Amount after fee.
        """
        fee = amount * self.transaction_fee_pct
        return amount - fee
    
    def predict(self, 
               features: np.ndarray, 
               model_name: Optional[str] = None) -> Tuple[float, float]:
        """
        Make a prediction using the specified model or ensemble.
        
        Args:
            features (np.ndarray): Input features.
            model_name (str, optional): Name of the model to use. If None, use ensemble.
            
        Returns:
            Tuple[float, float]: Predicted price and confidence.
        """
        if not self.models:
            raise ValueError("No models available for prediction.")
        
        if model_name and model_name in self.models:
            model = self.models[model_name]
            prediction = model.predict(features)
            
            # Simple confidence measure (can be improved)
            confidence = 0.7  # Default confidence
            
            return prediction[0][0] if isinstance(prediction, np.ndarray) and prediction.ndim > 1 else prediction[0], confidence
        else:
            # Use ensemble or weighted average of all models
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                pred = model.predict(features)
                pred_value = pred[0][0] if isinstance(pred, np.ndarray) and pred.ndim > 1 else pred[0]
                predictions.append(pred_value)
                weights.append(1.0)  # Equal weights by default
            
            # Weighted average
            weighted_pred = np.average(predictions, weights=weights)
            
            # Calculate confidence based on agreement between models
            std_dev = np.std(predictions)
            mean_pred = np.mean(predictions)
            coefficient_of_variation = std_dev / mean_pred if mean_pred != 0 else 0
            
            # Convert to confidence (inverse relationship with CV)
            confidence = max(0.5, 1.0 - min(coefficient_of_variation, 0.5))
            
            return weighted_pred, confidence
    
    def decide_action(self, 
                     current_price: float, 
                     predicted_price: float, 
                     confidence: float) -> str:
        """
        Decide on trading action based on prediction and confidence.
        
        Args:
            current_price (float): Current stock price.
            predicted_price (float): Predicted stock price.
            confidence (float): Confidence in the prediction.
            
        Returns:
            str: Trading action ('buy', 'sell', or 'hold').
        """
        # Calculate predicted percentage change
        predicted_change = (predicted_price / current_price - 1) * 100
        
        # Define thresholds for decision making
        buy_threshold = 0.5  # Percentage increase to trigger buy
        sell_threshold = -0.5  # Percentage decrease to trigger sell
        
        # Adjust thresholds based on confidence
        adjusted_buy_threshold = buy_threshold / (confidence ** 2)
        adjusted_sell_threshold = sell_threshold / (confidence ** 2)
        
        # Make decision
        if predicted_change > adjusted_buy_threshold:
            return 'buy'
        elif predicted_change < adjusted_sell_threshold:
            return 'sell'
        else:
            return 'hold'
    
    def execute_trade(self, 
                     action: str, 
                     current_price: float, 
                     amount: Optional[float] = None, 
                     shares: Optional[int] = None,
                     timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a trade based on the action.
        
        Args:
            action (str): Trading action ('buy', 'sell', or 'hold').
            current_price (float): Current stock price.
            amount (float, optional): Amount to invest for buy order.
            shares (int, optional): Number of shares to sell for sell order.
            timestamp (str, optional): Timestamp for the trade.
            
        Returns:
            Dict[str, Any]: Transaction details.
        """
        transaction = {
            'timestamp': timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'price': current_price,
            'balance_before': self.balance,
            'shares_before': self.shares,
        }
        
        if action == 'buy':
            # Determine amount to invest
            if amount is None and shares is None:
                # Use all available balance
                amount = self.balance
            elif shares is not None:
                # Calculate amount based on shares
                amount = shares * current_price
            
            # Apply transaction fee
            amount_after_fee = self._apply_transaction_fee(amount)
            
            # Calculate number of shares
            shares_to_buy = int(amount_after_fee / current_price)
            
            # Update balance and shares
            cost = shares_to_buy * current_price
            fee = cost * self.transaction_fee_pct
            self.balance -= (cost + fee)
            self.shares += shares_to_buy
            
            # Update transaction details
            transaction.update({
                'amount': amount,
                'fee': fee,
                'shares': shares_to_buy,
                'balance_after': self.balance,
                'shares_after': self.shares,
            })
            
        elif action == 'sell':
            # Determine number of shares to sell
            if shares is None:
                # Sell all shares
                shares_to_sell = self.shares
            else:
                # Sell specified number of shares
                shares_to_sell = min(shares, self.shares)
            
            # Calculate proceeds
            proceeds = shares_to_sell * current_price
            fee = proceeds * self.transaction_fee_pct
            proceeds_after_fee = proceeds - fee
            
            # Update balance and shares
            self.balance += proceeds_after_fee
            self.shares -= shares_to_sell
            
            # Update transaction details
            transaction.update({
                'shares': shares_to_sell,
                'proceeds': proceeds,
                'fee': fee,
                'balance_after': self.balance,
                'shares_after': self.shares,
            })
            
        self.transaction_history.append(transaction)
        self.total_trades += 1 if action != 'hold' else 0
        
        # Update portfolio value
        portfolio_value = self.balance + (self.shares * current_price)
        self.portfolio_history.append({
            'timestamp': transaction['timestamp'],
            'balance': self.balance,
            'shares': self.shares,
            'price': current_price,
            'portfolio_value': portfolio_value
        })
        
        # Update performance metrics
        self._update_performance_metrics(transaction, portfolio_value)
        
        return transaction
    
    def _update_performance_metrics(self, transaction: Dict[str, Any], portfolio_value: float) -> None:
        """
        Update performance metrics after a trade.
        
        Args:
            transaction (Dict[str, Any]): Transaction details.
            portfolio_value (float): Current portfolio value.
        """
        # Calculate returns
        self.cumulative_return = (portfolio_value / self.initial_balance) - 1
        
        # Calculate max drawdown
        if self.portfolio_history:
            peak_value = max(item['portfolio_value'] for item in self.portfolio_history)
            drawdown = (peak_value - portfolio_value) / peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Track win/loss for completed trades
        if transaction['action'] == 'sell' and 'shares' in transaction and transaction['shares'] > 0:
            # Simple win/loss calculation
            if 'proceeds' in transaction and transaction['proceeds'] > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the trading agent.
        
        Returns:
            Dict[str, Any]: Performance metrics.
        """
        win_rate = self.win_count / self.total_trades if self.total_trades > 0 else 0
        
        metrics = {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'shares_held': self.shares,
            'portfolio_value': self.balance + (self.shares * self.portfolio_history[-1]['price'] if self.portfolio_history else 0),
            'cumulative_return': self.cumulative_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'transaction_count': len(self.transaction_history)
        }
        
        return metrics
    
    def save_results(self, output_dir: str = 'results') -> None:
        """
        Save trading results to files.
        
        Args:
            output_dir (str): Directory to save results.
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save transaction history
        transactions_df = pd.DataFrame(self.transaction_history)
        transactions_df.to_csv(os.path.join(output_dir, 'transactions.csv'), index=False)
        
        # Save portfolio history
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
        
        # Save performance metrics
        metrics = self.get_performance_metrics()
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def load_models(self, model_paths: Dict[str, str]) -> None:
        """
        Load models from file paths.
        
        Args:
            model_paths (Dict[str, str]): Dictionary mapping model names to file paths.
        """
        for name, path in model_paths.items():
            if 'lstm' in name.lower():
                model = LSTMModel.load(path)
            else:
                model = EnsembleModel.load(path)
            
            self.models[name] = model
    
    def generate_trading_advice(self, 
                              current_price: float, 
                              predicted_price: float, 
                              confidence: float,
                              features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate trading advice for the current day.
        
        Args:
            current_price (float): Current stock price.
            predicted_price (float): Predicted stock price.
            confidence (float): Confidence in the prediction.
            features (np.ndarray, optional): Input features for additional analysis.
            
        Returns:
            Dict[str, Any]: Trading advice.
        """
        # Decide action
        action = self.decide_action(current_price, predicted_price, confidence)
        
        # Calculate expected profit/loss
        expected_change = predicted_price - current_price
        expected_pct_change = (predicted_price / current_price - 1) * 100
        
        # Calculate position size
        if action == 'buy':
            position_size = self._calculate_position_size(confidence, current_price)
            amount = position_size * current_price
        elif action == 'sell':
            position_size = self.shares
            amount = position_size * current_price
        else:
            position_size = 0
            amount = 0
        
        # Generate advice
        advice = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'current_price': current_price,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'action': action,
            'position_size': position_size,
            'amount': amount,
            'expected_change': expected_change,
            'expected_pct_change': expected_pct_change,
            'current_balance': self.balance,
            'current_shares': self.shares,
            'portfolio_value': self.balance + (self.shares * current_price)
        }
        
        return advice


class TradingSimulation:
    """Simulates a trading period using the trading agent."""
    
    def __init__(self, 
                agent: TradingAgent,
                data: pd.DataFrame,
                feature_columns: List[str],
                price_column: str = 'Close',
                sequence_length: int = 10):
        """
        Initialize the trading simulation.
        
        Args:
            agent (TradingAgent): Trading agent to use.
            data (pd.DataFrame): DataFrame containing simulation data.
            feature_columns (List[str]): Columns to use as features.
            price_column (str): Column containing the price.
            sequence_length (int): Length of input sequences for the model.
        """
        self.agent = agent
        self.data = data
        self.feature_columns = feature_columns
        self.price_column = price_column
        self.sequence_length = sequence_length
        self.current_day = 0
        self.current_price = None
        self.results = []
        
    def prepare_features(self, day_index: int) -> np.ndarray:
        """
        Prepare features for the given day.
        
        Args:
            day_index (int): Index of the day in the simulation data.
            
        Returns:
            np.ndarray: Features for the model.
        """
        if day_index < self.sequence_length:
            raise ValueError(f"Not enough data for day index {day_index}. Need at least {self.sequence_length} days of data.")
        
        # Get the sequence of data
        sequence = self.data.iloc[day_index - self.sequence_length:day_index][self.feature_columns].values
        
        # Reshape for model input (add batch dimension)
        sequence = np.expand_dims(sequence, axis=0)
        
        return sequence
    
    def run_day(self, day_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single day of the simulation.
        
        Args:
            day_index (int, optional): Index of the day to run. If None, use current_day.
            
        Returns:
            Dict[str, Any]: Results for the day.
        """
        if day_index is None:
            day_index = self.current_day
        
        # Check if day is valid
        if day_index >= len(self.data):
            raise ValueError(f"Day index {day_index} is out of bounds for data with {len(self.data)} rows.")
        
        # Get current price
        self.current_price = self.data.iloc[day_index][self.price_column]
        
        # Prepare features
        if day_index >= self.sequence_length:
            features = self.prepare_features(day_index)
            
            # Get prediction
            predicted_price, confidence = self.agent.predict(features)
            
            # Generate advice
            advice = self.agent.generate_trading_advice(
                self.current_price, predicted_price, confidence, features)
            
            # Execute the trade
            transaction = self.agent.execute_trade(
                advice['action'],
                self.current_price,
                amount=advice['amount'] if advice['action'] == 'buy' else None,
                shares=advice['position_size'] if advice['action'] == 'sell' else None,
                timestamp=self.data.index[day_index].strftime('%Y-%m-%d')
            )
            
            # Store results
            result = {
                'date': self.data.index[day_index],
                'day_index': day_index,
                'current_price': self.current_price,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'action': advice['action'],
                'balance': self.agent.balance,
                'shares': self.agent.shares,
                'portfolio_value': self.agent.balance + (self.agent.shares * self.current_price),
                'transaction': transaction
            }
        else:
            # Not enough data for prediction, just track price
            result = {
                'date': self.data.index[day_index],
                'day_index': day_index,
                'current_price': self.current_price,
                'action': 'hold',
                'balance': self.agent.balance,
                'shares': self.agent.shares,
                'portfolio_value': self.agent.balance + (self.agent.shares * self.current_price)
            }
        
        self.results.append(result)
        self.current_day += 1
        
        return result
    
    def run_simulation(self) -> pd.DataFrame:
        """
        Run the full simulation.
        
        Returns:
            pd.DataFrame: DataFrame containing simulation results.
        """
        self.results = []
        self.current_day = 0
        
        # Run each day
        for day_index in range(len(self.data)):
            self.run_day(day_index)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        return results_df
    
    def save_results(self, output_dir: str = 'results') -> None:
        """
        Save simulation results to files.
        
        Args:
            output_dir (str): Directory to save results.
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save simulation results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(output_dir, 'simulation_results.csv'), index=False)
        
        # Save agent results
        self.agent.save_results(output_dir) 