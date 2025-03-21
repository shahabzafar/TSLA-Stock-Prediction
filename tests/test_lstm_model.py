"""Tests for the LSTM model."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.models.lstm_model import LSTMModel, create_lstm_model


class TestLSTMModel(unittest.TestCase):
    """Tests for the LSTM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock data
        self.sequence_length = 10
        self.n_features = 5
        self.X_train = np.random.random((100, self.sequence_length, self.n_features))
        self.y_train = np.random.random((100, 1))
        self.X_val = np.random.random((20, self.sequence_length, self.n_features))
        self.y_val = np.random.random((20, 1))
        
        # Create model
        self.model = LSTMModel(input_shape=(self.sequence_length, self.n_features))
    
    def test_init(self):
        """Test initialization of LSTM model."""
        self.assertEqual(self.model.input_shape, (self.sequence_length, self.n_features))
        self.assertEqual(self.model.output_dim, 1)
        self.assertEqual(self.model.lstm_units, 50)  # Default value
        self.assertEqual(self.model.dropout_rate, 0.2)  # Default value
        self.assertIsNotNone(self.model.model)
    
    def test_build_model(self):
        """Test building the LSTM model."""
        # Create model with custom parameters
        model = LSTMModel(
            input_shape=(self.sequence_length, self.n_features),
            lstm_units=64,
            dropout_rate=0.3
        )
        
        # Assert the model is built
        self.assertIsNotNone(model.model)
        self.assertIsInstance(model.model, tf.keras.Model)
        
        # Check the model structure
        self.assertEqual(len(model.model.layers), 5)  # LSTM, Dropout, LSTM, Dropout, Dense
        
        # Check input and output shapes
        self.assertEqual(model.model.input_shape, (None, self.sequence_length, self.n_features))
        self.assertEqual(model.model.output_shape, (None, 1))
    
    def test_train(self):
        """Test training the LSTM model."""
        # Train the model
        history = self.model.train(
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            epochs=2, batch_size=16, 
            patience=5, verbose=0
        )
        
        # Check if history is returned
        self.assertIsInstance(history, dict)
        
        # Check if history contains loss and validation loss
        self.assertTrue('loss' in history)
        self.assertTrue('val_loss' in history)
        self.assertEqual(len(history['loss']), 2)  # 2 epochs
    
    def test_predict(self):
        """Test prediction with the LSTM model."""
        # Train the model minimally
        self.model.train(
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            epochs=1, batch_size=16, 
            patience=5, verbose=0
        )
        
        # Make predictions
        X_test = np.random.random((5, self.sequence_length, self.n_features))
        predictions = self.model.predict(X_test)
        
        # Check predictions shape
        self.assertEqual(predictions.shape, (5, 1))
    
    def test_evaluate(self):
        """Test evaluating the LSTM model."""
        # Train the model minimally
        self.model.train(
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            epochs=1, batch_size=16, 
            patience=5, verbose=0
        )
        
        # Evaluate the model
        X_test = np.random.random((10, self.sequence_length, self.n_features))
        y_test = np.random.random((10, 1))
        mse = self.model.evaluate(X_test, y_test)
        
        # Check if metrics are returned
        self.assertIsInstance(mse, float)
    
    @patch('os.makedirs')
    @patch('tensorflow.keras.models.Sequential.save')
    def test_save(self, mock_save, mock_makedirs):
        """Test saving the LSTM model."""
        # Save the model
        output_path = "test_output/lstm_model"
        self.model.save(output_path)
        
        # Check if model was saved
        mock_save.assert_called_once_with(output_path)
    
    @patch('tensorflow.keras.models.load_model')
    def test_load(self, mock_load_model):
        """Test loading the LSTM model."""
        # Mock the loaded model with input shape property
        mock_model = MagicMock()
        mock_model.input_shape = (None, self.sequence_length, self.n_features)
        mock_load_model.return_value = mock_model
        
        # Load the model
        model_path = "test_model_path"
        loaded_model = LSTMModel.load(model_path)
        
        # Check if model was loaded
        mock_load_model.assert_called_once_with(model_path)
        self.assertEqual(loaded_model.model, mock_model)
    
    @patch('src.models.lstm_model.LSTMModel')
    def test_create_lstm_model(self, mock_lstm_class):
        """Test the create_lstm_model function."""
        # Mock the LSTMModel instance
        mock_lstm = MagicMock()
        mock_lstm.train.return_value = {'loss': [0.1, 0.05], 'val_loss': [0.2, 0.1]}
        mock_lstm_class.return_value = mock_lstm
        
        # Call create_lstm_model
        model, history = create_lstm_model(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            lstm_units=64,
            dropout_rate=0.2,
            epochs=5,
            batch_size=32,
            patience=3
        )
        
        # Check if LSTMModel was instantiated correctly
        mock_lstm_class.assert_called_once_with(
            input_shape=(self.sequence_length, self.n_features),
            lstm_units=64,
            dropout_rate=0.2
        )
        
        # Check if train method was called
        mock_lstm.train.assert_called_once()
        
        # Check if returned values are correct
        self.assertEqual(model, mock_lstm)
        self.assertEqual(history, {'loss': [0.1, 0.05], 'val_loss': [0.2, 0.1]})


if __name__ == '__main__':
    unittest.main() 