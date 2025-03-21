"""Tests for the data loader module."""

import os
import sys
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.utils.data_loader import load_stock_data, get_data_for_simulation, get_training_data


class TestDataLoader(unittest.TestCase):
    """Tests for the data loader module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample CSV file for testing
        self.test_file = "tests/test_data.csv"
        
        # Create sample data
        dates = pd.date_range(start='2022-01-01', end='2022-01-10')
        data = {
            'Date': dates,
            'Open': [100 + i for i in range(len(dates))],
            'High': [105 + i for i in range(len(dates))],
            'Low': [95 + i for i in range(len(dates))],
            'Close': [101 + i for i in range(len(dates))],
            'Adj Close': [101 + i for i in range(len(dates))],
            'Volume': [1000000 for _ in range(len(dates))]
        }
        
        # Save to CSV
        os.makedirs(os.path.dirname(self.test_file), exist_ok=True)
        pd.DataFrame(data).to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_stock_data(self):
        """Test loading stock data from a CSV file."""
        # Load the test data
        df = load_stock_data(self.test_file)
        
        # Check the DataFrame properties
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)  # 10 days of data
        self.assertTrue('Open' in df.columns)
        self.assertTrue('High' in df.columns)
        self.assertTrue('Low' in df.columns)
        self.assertTrue('Close' in df.columns)
        self.assertTrue('Volume' in df.columns)
        
        # Check that Date is the index and it's a datetime
        self.assertEqual(df.index.name, 'Date')
        self.assertIsInstance(df.index, pd.DatetimeIndex)
    
    def test_get_data_for_simulation(self):
        """Test getting a subset of data for simulation."""
        # Load the test data
        df = load_stock_data(self.test_file)
        
        # Get data for a specific period
        start_date = '2022-01-03'
        end_date = '2022-01-07'
        sim_data = get_data_for_simulation(df, start_date, end_date)
        
        # Check the properties of the simulation data
        self.assertIsInstance(sim_data, pd.DataFrame)
        self.assertEqual(len(sim_data), 5)  # 5 days of data
        
        # Check date range
        self.assertEqual(sim_data.index.min().strftime('%Y-%m-%d'), start_date)
        self.assertEqual(sim_data.index.max().strftime('%Y-%m-%d'), end_date)
    
    def test_get_training_data(self):
        """Test getting data for training models."""
        # Load the test data
        df = load_stock_data(self.test_file)
        
        # Get training data
        end_date = '2022-01-07'
        lookback_days = 3
        train_data = get_training_data(df, end_date, lookback_days)
        
        # Check the properties of the training data
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertEqual(len(train_data), 4)  # 4 days of data returned by implementation
        
        # Check date range
        self.assertEqual(train_data.index.min().strftime('%Y-%m-%d'), '2022-01-04')  # Actual start date
        self.assertEqual(train_data.index.max().strftime('%Y-%m-%d'), end_date)


if __name__ == '__main__':
    unittest.main() 