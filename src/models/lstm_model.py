"""LSTM model for stock price prediction."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Dict, Any, Optional


class LSTMModel:
    """LSTM model for stock price prediction."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 output_dim: int = 1,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape (Tuple[int, int]): Shape of input data (sequence_length, num_features).
            output_dim (int): Dimension of output (default 1 for regression).
            lstm_units (int): Number of LSTM units.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build and compile the LSTM model.
        
        Returns:
            Sequential: Compiled Keras model.
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=self.lstm_units, 
                       return_sequences=True,
                       input_shape=self.input_shape))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        
        # Dense output layer
        model.add(Dense(units=self.output_dim))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              X_val: np.ndarray, 
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              patience: int = 10,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            patience (int): Patience for early stopping.
            verbose (int): Verbosity level.
            
        Returns:
            Dict[str, Any]: Training history.
        """
        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', 
                                      patience=patience, 
                                      restore_best_weights=True)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test targets.
            
        Returns:
            float: Mean squared error on test data.
        """
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'LSTMModel':
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model.
            
        Returns:
            LSTMModel: Loaded model.
        """
        # Load the Keras model
        loaded_keras_model = tf.keras.models.load_model(filepath)
        
        # Get input shape from the model
        input_shape = (loaded_keras_model.input_shape[1], loaded_keras_model.input_shape[2])
        
        # Create a new instance with the correct input shape
        instance = cls(input_shape=input_shape)
        
        # Replace the model
        instance.model = loaded_keras_model
        
        return instance


def create_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lstm_units: int = 50,
    dropout_rate: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10
) -> Tuple[LSTMModel, Dict[str, Any]]:
    """
    Create and train an LSTM model.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        patience (int): Patience for early stopping.
        
    Returns:
        Tuple[LSTMModel, Dict[str, Any]]: Trained model and training history.
    """
    # Get input shape from training data
    seq_length, num_features = X_train.shape[1], X_train.shape[2]
    
    # Create model
    model = LSTMModel(
        input_shape=(seq_length, num_features),
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience
    )
    
    return model, history 