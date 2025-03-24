import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Default simulation parameters
INITIAL_BALANCE = 10000.0        # $10,000 USD starting capital
TRANSACTION_FEE_PCT = 0.01       # 1% transaction fee
ORDER_SUBMISSION_TIME = "09:00"  # Orders must be submitted by 9:00 AM EST
ORDER_EXECUTION_TIME = "10:00"   # Orders are executed at 10:00 AM EST

def get_month_date_range(year, month):
    """
    Get the start and end dates for a given month and year.
    
    Args:
        year (int): Year
        month (int): Month (1-12)
        
    Returns:
        tuple: (start_date, end_date) as YYYY-MM-DD strings
    """
    # Get first day of the month
    first_day = datetime(year, month, 1)
    
    # Get last day of the month
    _, last_day_num = calendar.monthrange(year, month)
    last_day = datetime(year, month, last_day_num)
    
    # Format as strings
    start_date = first_day.strftime('%Y-%m-%d')
    end_date = last_day.strftime('%Y-%m-%d')
    
    return start_date, end_date

def load_best_model():
    """
    Load the best prediction model from the models directory.
    Tries to load ensemble model first, then LSTM model.
    If no models are found, creates a simple dummy model.
    
    Returns:
        tuple: (model, model_type, normalization_params, feature_columns)
    """
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    model = None
    model_type = None
    norm_params = None
    feature_columns = None
    
    # Try to load normalization parameters
    try:
        with open('models/normalization_params.pkl', 'rb') as f:
            norm_params = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load normalization parameters: {e}")
        # Create default normalization parameters
        norm_params = {
            'min': np.array([-1, -1, -1, -1, -1]),
            'scale': np.array([2, 2, 2, 2, 2])
        }
    
    # Try to load feature columns
    try:
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load feature columns: {e}")
        # Default feature columns - most important ones
        feature_columns = ['Open', 'Close', 'High', 'Low', 'Volume']
    
    # Try to load ensemble model first (typically better for this task)
    try:
        with open('models/ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
            model_type = 'ensemble'
            print("Loaded ensemble model successfully")
    except Exception as e:
        print(f"Could not load ensemble model: {e}")
        
        # Try to load LSTM model as backup
        try:
            model = tf.keras.models.load_model('models/lstm_model')
            model_type = 'lstm'
            print("Loaded LSTM model successfully")
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
            
            # Create a dummy model if no models could be loaded
            print("No models found. Creating a simple prediction model.")
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit([[0, 0, 0, 0, 0]], [0])  # Dummy fit
            model_type = 'ensemble'
    
    return model, model_type, norm_params, feature_columns

def create_or_get_data(start_date, end_date):
    """
    Create or load Tesla stock data for the specified date range.
    First tries to load historical data for the date range if available
    If not, will generate simulated data based on available historical data.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        DataFrame: Stock data for the period
    """
    # Check if we have Tesla data first
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if not data_files:
        raise FileNotFoundError("No CSV files found in the data directory.")
    
    tesla_file = None
    for file in data_files:
        if 'TSLA' in file.upper():
            tesla_file = file
            break
    
    if not tesla_file:
        raise FileNotFoundError("No Tesla stock data file found.")
    
    print(f"Using Tesla data from {tesla_file}")
    historical_data = pd.read_csv(os.path.join('data', tesla_file))
    
    # Convert date column to datetime
    date_col = 'Date' if 'Date' in historical_data.columns else 'date'
    historical_data[date_col] = pd.to_datetime(historical_data[date_col])
    
    # Try to extract data for the specified date range if it exists in the file
    period_data = historical_data[
        (historical_data[date_col] >= start_date) & 
        (historical_data[date_col] <= end_date)
    ]
    
    # If we found actual data for the period, use it
    if len(period_data) > 0:
        print(f"Found {len(period_data)} actual trading days for the period in the data")
        
        # Create 9:00 AM and 10:00 AM prices if they don't exist
        if 'Price_9AM' not in period_data.columns:
            period_data['Price_9AM'] = period_data['Open'] * (1 + np.random.normal(0, 0.005, len(period_data)))
        
        if 'Price_10AM' not in period_data.columns:
            period_data['Price_10AM'] = period_data['Open'] * (1 + np.random.normal(0.002, 0.008, len(period_data)))
        
        # Ensure the columns we need are present
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Price_9AM', 'Price_10AM', 'Volume']
        for col in required_columns:
            if col not in period_data.columns and col.lower() in period_data.columns:
                period_data[col] = period_data[col.lower()]
        
        return period_data
    else:
        print(f"No actual data found for the period {start_date} to {end_date}. Generating simulated data.")
        return generate_simulated_data(historical_data, start_date, end_date)

def generate_simulated_data(historical_data, start_date, end_date):
    """
    Generate simulated data based on historical data for a specified date range.
    
    Args:
        historical_data (DataFrame): Historical stock data
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        DataFrame: Simulated stock data for the period
    """
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate business days in the range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    # Get recent volatility to make realistic price movements
    historical_data['Returns'] = historical_data['Close'].pct_change()
    recent_volatility = historical_data['Returns'].std()
    
    # Use the most recent price as the starting point, or a reasonable default
    if not historical_data.empty:
        last_close = historical_data['Close'].iloc[-1]
    else:
        # If no historical data, use a reasonable default for Tesla
        last_close = 500.0  
    
    # Generate simulated data
    sim_data = []
    
    for date in date_range:
        # Simulate intraday prices
        daily_volatility = recent_volatility * np.random.uniform(0.5, 1.5)
        
        # Morning movement (pre-market to 9:00 AM)
        morning_change = np.random.normal(0, daily_volatility)
        price_9am = last_close * (1 + morning_change)
        
        # Additional movement to 10:00 AM
        hour_change = np.random.normal(0, daily_volatility / 2)
        price_10am = price_9am * (1 + hour_change)
        
        # Rest of day movement
        day_change = np.random.normal(0, daily_volatility)
        close_price = price_10am * (1 + day_change)
        
        # Generate other price data
        open_price = last_close * (1 + np.random.normal(0, daily_volatility / 3))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, daily_volatility / 2)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, daily_volatility / 2)))
        
        # Generate volume
        volume = int(np.random.uniform(5000000, 20000000))
        
        # Store the data
        sim_data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Price_9AM': round(price_9am, 2),
            'Price_10AM': round(price_10am, 2),
            'Volume': volume
        })
        
        # Update last close for next day
        last_close = close_price
    
    # Create DataFrame from simulated data
    sim_df = pd.DataFrame(sim_data)
    
    print(f"Created simulated Tesla stock data for {len(sim_df)} trading days")
    print(f"Date range: {sim_df['Date'].min()} to {sim_df['Date'].max()}")
    print(f"Price range: ${sim_df['Low'].min():.2f} to ${sim_df['High'].max():.2f}")
    
    return sim_df

def add_technical_indicators(df):
    """
    Add technical indicators to the data for feature engineering.
    
    Args:
        df (DataFrame): Stock price data
        
    Returns:
        DataFrame: Enhanced data with technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])
    
    # Set Date as index for easier calculation
    data.set_index('Date', inplace=True)
    
    # Moving Averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (std_dev * 2)
    data['BB_Lower'] = data['BB_Middle'] - (std_dev * 2)
    
    # Rate of Change (ROC)
    data['ROC'] = data['Close'].pct_change(periods=10) * 100
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    
    # Volume Rate of Change
    data['Volume_ROC'] = data['Volume'].pct_change(periods=1) * 100
    
    # On-Balance Volume (OBV)
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    
    # Create target for next-day return
    data['Next_Day_Return'] = data['Close'].pct_change(periods=1).shift(-1)
    
    # Reset index to get Date back as a column
    data.reset_index(inplace=True)
    
    return data

def prepare_data_for_prediction(df, feature_columns, model_type, sequence_length=20, norm_params=None):
    """
    Prepare data for prediction based on the model type.
    
    Args:
        df (DataFrame): Stock data with technical indicators
        feature_columns (list): List of feature column names to use
        model_type (str): 'lstm' or 'ensemble'
        sequence_length (int): Sequence length for LSTM model
        norm_params (dict): Normalization parameters
        
    Returns:
        tuple: (X, scaler) where X is prepared features
    """
    # Fill any missing values (important for technical indicators at the beginning)
    df_filled = df.copy()
    df_filled[feature_columns] = df_filled[feature_columns].fillna(method='bfill').fillna(method='ffill')
    
    # Normalize data
    scaler = MinMaxScaler()
    if norm_params:
        try:
            # Try to set the scaler parameters from norm_params
            scaler.min_ = norm_params['min']
            scaler.scale_ = norm_params['scale']
        except (KeyError, AttributeError) as e:
            print(f"Warning: Could not set scaler parameters from norm_params: {e}")
            # Fit the scaler on the data if parameters couldn't be set
            scaler.fit(df_filled[feature_columns])
    else:
        # If no norm_params provided, fit the scaler on the data
        scaler.fit(df_filled[feature_columns])
    
    # Transform the data
    try:
        X_normalized = scaler.transform(df_filled[feature_columns])
    except Exception as e:
        print(f"Error transforming data: {e}")
        # Fall back to fitting and transforming if transform fails
        scaler = MinMaxScaler()
        scaler.fit(df_filled[feature_columns])
        X_normalized = scaler.transform(df_filled[feature_columns])
    
    if model_type == 'lstm':
        # For LSTM, we need sequences
        X = []
        for i in range(len(df_filled) - sequence_length + 1):
            X.append(X_normalized[i:(i + sequence_length)])
        
        X = np.array(X)
        
        # For the first few days where we don't have enough history, we'll pad with the first available data
        pad_sequences = []
        for i in range(sequence_length - 1):
            # Create a sequence with repeated first rows to reach sequence_length
            padding = np.repeat(X_normalized[:1], sequence_length - i - 1, axis=0)
            seq = np.vstack([padding, X_normalized[:i+1]])
            pad_sequences.append(seq)
        
        X_padded = np.array(pad_sequences)
        
        # Combine padded sequences with regular sequences
        X = np.vstack([X_padded, X])
        
        return X, scaler
    else:
        # For ensemble, we just return the normalized features
        return X_normalized, scaler

def get_model_prediction(stock_data, day_index, model, model_type, feature_columns, norm_params=None):
    """
    Get the model's prediction for the given day.
    
    Args:
        stock_data (DataFrame): Stock data with technical indicators
        day_index (int): Index of the day to predict
        model: The prediction model (LSTM or ensemble)
        model_type (str): 'lstm' or 'ensemble'
        feature_columns (list): List of feature column names
        norm_params (dict): Normalization parameters
    
    Returns:
        tuple: (action, confidence) where action is 'buy', 'sell', or 'hold'
    """
    try:
        # Prepare the data for prediction
        X, _ = prepare_data_for_prediction(
            stock_data.iloc[:day_index+1], 
            feature_columns, 
            model_type, 
            sequence_length=20, 
            norm_params=norm_params
        )
        
        # Make prediction
        if model_type == 'lstm':
            # For LSTM, use the last sequence
            pred = model.predict(X[-1:], verbose=0)
            confidence = abs(float(pred[0][0]))
        else:
            # For ensemble, use the last data point
            try:
                pred = model.predict(X[-1:].reshape(1, -1))
            except Exception:
                # Handle case when reshape fails
                pred = model.predict(X[-1:])
            confidence = abs(float(pred[0]))
        
        # Convert prediction to action
        if pred > 0.02:  # Predicted return > 2%
            action = 'buy'
            confidence = min(confidence * 1.5, 0.95)  # Scale confidence
        elif pred < -0.02:  # Predicted return < -2%
            action = 'sell'
            confidence = min(confidence * 1.5, 0.95)  # Scale confidence
        else:
            action = 'hold'
            confidence = 0.5
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        # If prediction fails, generate a random prediction
        # with bias towards holding (less risky)
        import random
        rand_val = random.random()
        if rand_val > 0.7:
            action = 'buy'
            confidence = 0.6
        elif rand_val < 0.2:
            action = 'sell'
            confidence = 0.6
        else:
            action = 'hold'
            confidence = 0.7
        print(f"Using fallback prediction: {action} (confidence: {confidence:.2f})")
    
    return action, confidence 

def determine_order_amount(action, confidence, portfolio):
    """
    Determine the amount to buy or sell based on the action, confidence, and portfolio state.
    
    Returns:
        tuple: (action, amount) where amount is:
               - dollar amount for 'buy' orders
               - number of shares for 'sell' orders
               - None for 'hold' orders
    """
    cash = portfolio['cash']
    shares = portfolio['shares']
    current_price = portfolio['current_price']
    
    if action == 'buy' and cash > 0:
        # Risk-based position sizing
        # Use higher confidence to allocate more of available cash
        risk_factor = min(confidence * 1.5, 0.9)  # Cap at 90% of available cash
        amount = cash * risk_factor
        return 'buy', amount
    
    elif action == 'sell' and shares > 0:
        # Risk-based position sizing for sell
        # Use higher confidence to sell more of available shares
        risk_factor = min(confidence * 1.5, 0.9)  # Cap at 90% of available shares
        shares_to_sell = int(shares * risk_factor)
        return 'sell', shares_to_sell
    
    else:
        return 'hold', None

def format_order(action, amount):
    """
    Format the trading order according to the required submission format.
    
    Returns:
        str: The formatted order string
    """
    if action == 'buy':
        return f"Buy: ${amount:.2f}"
    elif action == 'sell':
        return f"Sell: {amount} shares"
    else:
        return "Hold: No transaction"

def execute_order(action, amount, portfolio, execution_price, transaction_fee_pct):
    """
    Execute a trading order at the specified execution price.
    
    Args:
        action (str): 'buy', 'sell', or 'hold'
        amount: Dollar amount for buy orders, number of shares for sell orders
        portfolio (dict): The current portfolio state
        execution_price (float): The stock price at which to execute the order
        transaction_fee_pct (float): The transaction fee as a percentage
        
    Returns:
        dict: Updated portfolio state
    """
    cash = portfolio['cash']
    shares = portfolio['shares']
    
    if action == 'buy' and amount > 0:
        # Calculate how many shares can be bought with the specified dollar amount
        max_shares = int(amount / (execution_price * (1 + transaction_fee_pct)))
        
        if max_shares > 0:
            cost = max_shares * execution_price
            fee = cost * transaction_fee_pct
            total_cost = cost + fee
            
            if total_cost <= cash:
                cash -= total_cost
                shares += max_shares
                print(f"Executed BUY order: {max_shares} shares at ${execution_price:.2f} per share")
                print(f"Transaction fee: ${fee:.2f}")
                print(f"Total cost: ${total_cost:.2f}")
            else:
                print(f"Insufficient cash for BUY order. Available: ${cash:.2f}, Required: ${total_cost:.2f}")
        else:
            print(f"Unable to buy any shares with ${amount:.2f} at current price (${execution_price:.2f})")
    
    elif action == 'sell' and amount > 0:
        # Ensure we don't sell more shares than owned
        shares_to_sell = min(int(amount), shares)
        
        if shares_to_sell > 0:
            proceeds = shares_to_sell * execution_price
            fee = proceeds * transaction_fee_pct
            net_proceeds = proceeds - fee
            
            cash += net_proceeds
            shares -= shares_to_sell
            print(f"Executed SELL order: {shares_to_sell} shares at ${execution_price:.2f} per share")
            print(f"Transaction fee: ${fee:.2f}")
            print(f"Net proceeds: ${net_proceeds:.2f}")
    
    # Update portfolio
    portfolio['cash'] = cash
    portfolio['shares'] = shares
    
    return portfolio

def run_simulation(year, month, initial_balance=INITIAL_BALANCE, transaction_fee_pct=TRANSACTION_FEE_PCT):
    """
    Run a trading simulation for the specified month and year.
    
    Args:
        year (int): Year
        month (int): Month (1-12)
        initial_balance (float): Initial portfolio value
        transaction_fee_pct (float): Transaction fee percentage
        
    Returns:
        tuple: (results_df, performance_summary)
    """
    print(f"\nRunning trading simulation for {calendar.month_name[month]} {year}")
    
    # Get date range for the month
    start_date, end_date = get_month_date_range(year, month)
    
    # Load best model
    model, model_type, norm_params, feature_columns = load_best_model()
    
    # Get data for the simulation period
    simulation_data = create_or_get_data(start_date, end_date)
    
    # Add technical indicators
    simulation_data = add_technical_indicators(simulation_data)
    
    # Initialize portfolio
    portfolio = {
        'cash': initial_balance,
        'shares': 0,
        'current_price': simulation_data.iloc[0]['Open'],
        'portfolio_value': initial_balance
    }
    
    # Initialize results tracking
    results = []
    
    # Run the simulation for each day
    for day_idx, (_, day_data) in enumerate(simulation_data.iterrows()):
        date = day_data['Date']
        price_9am = day_data['Price_9AM']
        price_10am = day_data['Price_10AM']
        
        print(f"\n--- Trading Day {day_idx+1}: {date} ---")
        
        # Morning market check - 9:00 AM
        print(f"Morning Market Check (9:00 AM EST)")
        print(f"Tesla stock price: ${price_9am:.2f}")
        print(f"Current portfolio: ${portfolio['portfolio_value']:.2f} (Cash: ${portfolio['cash']:.2f}, Shares: {portfolio['shares']})")
        
        # Update portfolio with current price
        portfolio['current_price'] = price_9am
        
        # Get model prediction
        action, confidence = get_model_prediction(
            simulation_data, 
            day_idx, 
            model, 
            model_type, 
            feature_columns, 
            norm_params
        )
        print(f"Model prediction: {action.upper()} (confidence: {confidence:.2f})")
        
        # Determine order amount based on prediction and portfolio
        action, amount = determine_order_amount(action, confidence, portfolio)
        
        # Format the order for submission
        order = format_order(action, amount)
        print(f"Submitting order: {order}")
        
        # Order execution - 10:00 AM
        print(f"\nOrder Execution (10:00 AM EST)")
        print(f"Tesla stock price: ${price_10am:.2f}")
        
        # Execute the order
        portfolio = execute_order(action, amount, portfolio, price_10am, transaction_fee_pct)
        
        # End of day summary
        print(f"\nEnd of Day Summary:")
        print(f"Tesla closing price: ${day_data['Close']:.2f}")
        
        # Update portfolio with closing price
        portfolio['current_price'] = day_data['Close']
        portfolio['portfolio_value'] = portfolio['cash'] + (portfolio['shares'] * day_data['Close'])
        
        print(f"Portfolio value: ${portfolio['portfolio_value']:.2f}")
        print(f"Cash: ${portfolio['cash']:.2f}")
        print(f"Shares owned: {portfolio['shares']}")
        
        # Store daily results
        results.append({
            'Date': date,
            'Open': day_data['Open'],
            'High': day_data['High'],
            'Low': day_data['Low'],
            'Close': day_data['Close'],
            'Price_9AM': price_9am,
            'Price_10AM': price_10am,
            'Action': action,
            'Order_Amount': amount,
            'Portfolio_Value': portfolio['portfolio_value'],
            'Cash': portfolio['cash'],
            'Shares': portfolio['shares']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    initial_value = initial_balance
    final_value = portfolio['portfolio_value']
    total_return_pct = ((final_value / initial_value) - 1) * 100
    
    # Calculate buy & hold performance
    buy_hold_shares = int(initial_balance / (simulation_data.iloc[0]['Price_10AM'] * (1 + transaction_fee_pct)))
    buy_hold_cost = buy_hold_shares * simulation_data.iloc[0]['Price_10AM'] * (1 + transaction_fee_pct)
    buy_hold_value = buy_hold_shares * simulation_data.iloc[-1]['Close']
    buy_hold_return_pct = ((buy_hold_value / buy_hold_cost) - 1) * 100
    
    # Assemble performance summary
    performance_summary = pd.DataFrame([
        {'Metric': 'Initial Portfolio Value ($)', 'Value': f"{initial_value:.2f}"},
        {'Metric': 'Final Portfolio Value ($)', 'Value': f"{final_value:.2f}"},
        {'Metric': 'Total Return (%)', 'Value': f"{total_return_pct:.2f}"},
        {'Metric': 'Buy & Hold Return (%)', 'Value': f"{buy_hold_return_pct:.2f}"},
        {'Metric': 'Strategy vs. Buy & Hold (%)', 'Value': f"{total_return_pct - buy_hold_return_pct:.2f}"},
        {'Metric': 'Transaction Fee (%)', 'Value': f"{transaction_fee_pct * 100:.2f}"},
        {'Metric': 'Trading Period', 'Value': f"{start_date} to {end_date}"},
        {'Metric': 'Number of Trading Days', 'Value': f"{len(simulation_data)}"}
    ])
    
    # Print final summary
    print("\n=== Final Simulation Summary ===")
    print(f"Trading Period: {start_date} to {end_date}")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return_pct:.2f}%")
    print(f"Strategy vs. Buy & Hold: {total_return_pct - buy_hold_return_pct:.2f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Create unique filename based on the month and year
    month_name = calendar.month_name[month].lower()
    results_filename = f"tesla_{month_name}_{year}_simulation.csv"
    performance_filename = f"tesla_{month_name}_{year}_performance.csv"
    
    # Save to CSV
    results_df.to_csv(os.path.join('results', results_filename), index=False)
    performance_summary.to_csv(os.path.join('results', performance_filename), index=False)
    
    print(f"\nResults saved to results directory:")
    print(f"1. {results_filename} - Complete trading history")
    print(f"2. {performance_filename} - Performance summary")
    
    # Generate visualizations
    create_visualizations(results_df, month_name, year)
    
    return results_df, performance_summary

def create_visualizations(results_df, month_name, year):
    """Create visualizations for the simulation results"""
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(results_df['Date']):
        results_df['Date'] = pd.to_datetime(results_df['Date'])
    
    # 1. Portfolio value chart
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Portfolio_Value'], 
             marker='o', linestyle='-', linewidth=2, color='blue', label='Portfolio Value')
    
    # Highlight buy/sell points
    buy_points = results_df[results_df['Action'] == 'buy']
    sell_points = results_df[results_df['Action'] == 'sell']
    
    if not buy_points.empty:
        plt.scatter(buy_points['Date'], buy_points['Portfolio_Value'], 
                   color='green', s=100, marker='^', label='Buy')
    
    if not sell_points.empty:
        plt.scatter(sell_points['Date'], sell_points['Portfolio_Value'], 
                   color='red', s=100, marker='v', label='Sell')
    
    plt.title(f'Portfolio Value During Tesla Trading Simulation ({month_name.capitalize()} {year})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"{month_name}_{year}_portfolio_value.png"
    plt.savefig(f'visualizations/{filename}')
    plt.close()  # Close the figure to free memory
    
    # 2. Stock price with buy/sell signals
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], results_df['Close'], 
             marker='o', linestyle='-', linewidth=2, color='gray', alpha=0.7, label='TSLA Close Price')
    
    # Add 9AM and 10AM prices for context
    plt.plot(results_df['Date'], results_df['Price_9AM'], 
             marker='s', linestyle='--', linewidth=1, color='blue', alpha=0.6, label='9 AM Price')
    plt.plot(results_df['Date'], results_df['Price_10AM'], 
             marker='d', linestyle='--', linewidth=1, color='purple', alpha=0.6, label='10 AM Price (Execution)')
    
    # Highlight buy/sell points on closing price
    if not buy_points.empty:
        plt.scatter(buy_points['Date'], buy_points['Price_10AM'], 
                   color='green', s=150, marker='^', label='Buy')
    
    if not sell_points.empty:
        plt.scatter(sell_points['Date'], sell_points['Price_10AM'], 
                   color='red', s=150, marker='v', label='Sell')
    
    plt.title(f'Tesla Stock Price and Trading Signals ({month_name.capitalize()} {year})')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"{month_name}_{year}_trading_signals.png"
    plt.savefig(f'visualizations/{filename}')
    plt.close()  # Close the figure to free memory
    
    # 3. Portfolio composition chart (cash vs. stock)
    plt.figure(figsize=(12, 6))
    
    # Stack plot of cash and stock value
    stock_value = results_df['Shares'] * results_df['Close']
    plt.stackplot(results_df['Date'], 
                  [results_df['Cash'], stock_value],
                  labels=['Cash', 'Tesla Stock Value'],
                  colors=['#1f77b4', '#ff7f0e'],
                  alpha=0.7)
    
    plt.title(f'Portfolio Composition During Tesla Trading Simulation ({month_name.capitalize()} {year})')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"{month_name}_{year}_portfolio_composition.png"
    plt.savefig(f'visualizations/{filename}')
    plt.close()  # Close the figure to free memory
    
    print("\nVisualizations saved to 'visualizations' directory:")
    print(f"1. {month_name}_{year}_portfolio_value.png - Portfolio value chart")
    print(f"2. {month_name}_{year}_trading_signals.png - Stock price with trading signals")
    print(f"3. {month_name}_{year}_portfolio_composition.png - Portfolio composition chart")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a dynamic Tesla stock trading simulation for a specific month and year.')
    parser.add_argument('--year', type=int, required=True, help='Year for the simulation (e.g., 2023)')
    parser.add_argument('--month', type=int, required=True, help='Month for the simulation (1-12)')
    parser.add_argument('--initial-balance', type=float, default=INITIAL_BALANCE, 
                       help=f'Initial portfolio balance (default: ${INITIAL_BALANCE})')
    parser.add_argument('--transaction-fee', type=float, default=TRANSACTION_FEE_PCT,
                       help=f'Transaction fee percentage (default: {TRANSACTION_FEE_PCT*100}%)')
    
    args = parser.parse_args()
    
    # Validate month input
    if args.month < 1 or args.month > 12:
        raise ValueError("Month must be between 1 and 12")
    
    print(f"Running simulation for {calendar.month_name[args.month]} {args.year}")
    
    try:
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        
        # Run the simulation with the provided arguments
        results_df, _ = run_simulation(
            year=args.year,
            month=args.month,
            initial_balance=args.initial_balance,
            transaction_fee_pct=args.transaction_fee
        )
        
        # Get month name for creating the URL
        month_name = calendar.month_name[args.month].lower()
        
        # Ensure the results file exists before trying to create visualizations
        results_file = f'results/tesla_{month_name}_{args.year}_simulation.csv'
        if os.path.exists(results_file):
            # Try to load the saved results file
            try:
                saved_results = pd.read_csv(results_file)
                create_visualizations(saved_results, month_name, args.year)
            except Exception as e:
                print(f"Warning: Could not create visualizations from saved file: {e}")
                # Use the results_df directly if loading fails
                create_visualizations(results_df, month_name, args.year)
        else:
            # Use the results_df directly if the file doesn't exist
            print(f"Warning: Results file not found at {results_file}")
            create_visualizations(results_df, month_name, args.year)
        
        print(f"Simulation completed for {calendar.month_name[args.month]} {args.year}")
        print(f"Results saved to results/tesla_{month_name}_{args.year}_simulation.csv")
        print(f"Performance metrics saved to results/tesla_{month_name}_{args.year}_performance.csv")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        exit(1) 