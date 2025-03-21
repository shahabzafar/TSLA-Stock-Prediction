"""Ensemble models for stock price prediction."""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class EnsembleModel:
    """Ensemble model for stock price prediction."""
    
    def __init__(self):
        """Initialize the ensemble model."""
        self.models = {}
        self.model_weights = {}
        self.feature_importances_ = None
    
    def add_model(self, model_name: str, model: Any, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_name (str): Name of the model.
            model (Any): The model object.
            weight (float): Weight of the model in the ensemble.
        """
        self.models[model_name] = model
        self.model_weights[model_name] = weight
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray, optional): Validation features.
            y_val (np.ndarray, optional): Validation targets.
            **kwargs: Additional arguments for model training.
            
        Returns:
            Dict[str, Dict[str, float]]: Training metrics for each model.
        """
        metrics = {}
        
        for name, model in self.models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Calculate metrics if validation data is provided
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                metrics[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
        
        # Set feature importances from random forest model if available
        if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
            self.feature_importances_ = self.models['random_forest'].feature_importances_
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Weighted average predictions from all models.
        """
        if not self.models:
            raise ValueError("No models in the ensemble. Add models before prediction.")
        
        predictions = {}
        total_weight = sum(self.model_weights.values())
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Calculate weighted average
        weighted_avg = np.zeros_like(predictions[list(predictions.keys())[0]])
        for name, pred in predictions.items():
            weight = self.model_weights[name] / total_weight
            weighted_avg += weight * pred
        
        return weighted_avg
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble on test data.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the ensemble model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        Load an ensemble model from a file.
        
        Args:
            filepath (str): Path to the saved model.
            
        Returns:
            EnsembleModel: Loaded model.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_ensemble_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    random_forest_params: Optional[Dict] = None,
    gbm_params: Optional[Dict] = None
) -> Tuple[EnsembleModel, Dict[str, Dict[str, float]]]:
    """
    Create and train an ensemble model.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        X_test (np.ndarray, optional): Test features.
        y_test (np.ndarray, optional): Test targets.
        random_forest_params (Dict, optional): Parameters for Random Forest.
        gbm_params (Dict, optional): Parameters for Gradient Boosting.
        
    Returns:
        Tuple[EnsembleModel, Dict]: Trained ensemble model and metrics.
    """
    # Default parameters
    if random_forest_params is None:
        random_forest_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    if gbm_params is None:
        gbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
    
    # Create models
    rf_model = RandomForestRegressor(**random_forest_params)
    gbm_model = GradientBoostingRegressor(**gbm_params)
    svr_model = SVR(kernel='rbf', gamma='scale')
    lr_model = LinearRegression()
    
    # Create ensemble
    ensemble = EnsembleModel()
    
    # Add models with weights (can be tuned based on validation performance)
    ensemble.add_model('random_forest', rf_model, weight=0.4)
    ensemble.add_model('gradient_boosting', gbm_model, weight=0.3)
    ensemble.add_model('svr', svr_model, weight=0.2)
    ensemble.add_model('linear_regression', lr_model, weight=0.1)
    
    # Train ensemble
    metrics = ensemble.train(X_train, y_train, X_val, y_val)
    
    # Adjust weights based on validation performance if needed
    if X_val is not None and y_val is not None:
        # Example: set weights inversely proportional to MSE
        for name in ensemble.models:
            if name in metrics:
                mse = metrics[name]['mse']
                if mse > 0:
                    ensemble.model_weights[name] = 1.0 / mse
    
        # Normalize weights
        total_weight = sum(ensemble.model_weights.values())
        for name in ensemble.model_weights:
            ensemble.model_weights[name] /= total_weight
    
    return ensemble, metrics 