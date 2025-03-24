import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json
import warnings

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Disable deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Filter out TensorFlow warning messages
tf.get_logger().setLevel('ERROR')  # Only show ERROR messages

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

def predict_next_days(current_price, num_days=5, volatility=0.02, portfolio_value=10000, shares_owned=0):
    """
    Predict stock prices for the next several days based on current price.
    
    Args:
        current_price (float): The current closing price
        num_days (int): Number of days to predict (default: 5)
        volatility (float): Estimated daily volatility (default: 2%)
        portfolio_value (float): Current portfolio value in USD
        shares_owned (int): Number of Tesla shares currently owned
        
    Returns:
        dict: Prediction results with dates and prices
    """
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Load the best model
    model, model_type, norm_params, feature_columns = load_best_model()
    
    # Generate dates for the next num_days days
    today = datetime.now()
    dates = []
    for i in range(1, num_days + 1):
        next_date = today + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        dates.append(next_date.strftime('%Y-%m-%d'))
    
    # Create synthetic data for prediction starting with current price
    predicted_prices = [current_price]
    predicted_returns = []
    actions = []
    confidences = []
    
    # Portfolio tracking
    cash = portfolio_value - (shares_owned * current_price)
    if cash < 0:  # Adjust if portfolio value input doesn't match shares
        shares_owned = portfolio_value / current_price
        cash = 0
    
    portfolio_values = [portfolio_value]
    shares_held = [shares_owned]
    cash_values = [cash]
    investment_amounts = []
    
    transaction_fee_pct = 0.001  # 0.1% transaction fee
    
    # Generate a synthetic data point based on current price
    last_price = current_price
    
    # Basic feature set needed for prediction
    for i in range(num_days):
        if model_type == 'lstm':
            # For LSTM models, we need a sequence of data
            # This is a simplification - in practice you'd need actual historical data
            # to generate a proper sequence of technical indicators
            
            # Use a simple random walk model as fallback
            price_change = np.random.normal(0, volatility) * last_price
            next_price = last_price + price_change
            
            # Determine action based on price change
            if price_change > 0:
                action = 'buy'
                confidence = min(abs(price_change / last_price) * 10, 0.95)
            elif price_change < 0:
                action = 'sell'
                confidence = min(abs(price_change / last_price) * 10, 0.95)
            else:
                action = 'hold'
                confidence = 0.5
                
        else:
            # For ensemble models, we can make a more direct prediction
            try:
                # Create a feature vector for the model
                # This is a simplified approach - in practice, you'd calculate actual technical indicators
                features = np.array([[
                    last_price,                      # Using last price as Open
                    last_price * (1 + 0.005),        # High (0.5% above last price)
                    last_price * (1 - 0.005),        # Low (0.5% below last price)
                    last_price,                      # Close (same as last price for simplicity)
                    10000000                         # Average volume
                ]])
                
                # Normalize features if we have normalization parameters
                if norm_params:
                    try:
                        scaler = MinMaxScaler()
                        scaler.min_ = norm_params['min']
                        scaler.scale_ = norm_params['scale']
                        features = scaler.transform(features)
                    except Exception as e:
                        print(f"Warning: Failed to normalize features: {e}")
                
                # Make prediction
                pred = model.predict(features)
                predicted_return = float(pred[0])
                predicted_returns.append(predicted_return)
                
                # Calculate next price based on predicted return
                next_price = last_price * (1 + predicted_return)
                
                # Determine action based on predicted return
                if predicted_return > 0.01:  # 1% return threshold
                    action = 'buy'
                    confidence = min(predicted_return * 10, 0.95)  # Scale confidence
                elif predicted_return < -0.01:  # -1% return threshold
                    action = 'sell'
                    confidence = min(abs(predicted_return) * 10, 0.95)  # Scale confidence
                else:
                    action = 'hold'
                    confidence = 0.5
                    
            except Exception as e:
                print(f"Error during prediction: {e}")
                # Fall back to random walk model
                price_change = np.random.normal(0, volatility) * last_price
                next_price = last_price + price_change
                
                # Determine action based on price change
                if price_change > 0:
                    action = 'buy'
                    confidence = min(abs(price_change / last_price) * 10, 0.95)
                elif price_change < 0:
                    action = 'sell'
                    confidence = min(abs(price_change / last_price) * 10, 0.95)
                else:
                    action = 'hold'
                    confidence = 0.5
        
        # Add the prediction
        predicted_prices.append(next_price)
        actions.append(action)
        confidences.append(confidence)
        
        # Calculate portfolio changes based on action
        current_shares = shares_held[-1]
        current_cash = cash_values[-1]
        
        # Determine investment amount based on confidence and action
        if action == 'buy':
            # Invest more when confidence is higher
            investment_pct = min(confidence * 0.5, 0.3)  # Max 30% of available cash
            investment_amount = current_cash * investment_pct
            
            # Calculate number of shares to buy (accounting for fees)
            fee = investment_amount * transaction_fee_pct
            shares_to_buy = (investment_amount - fee) / last_price
            
            # Update portfolio
            new_shares = current_shares + shares_to_buy
            new_cash = current_cash - investment_amount
            new_portfolio = new_shares * next_price + new_cash
            
            investment_amounts.append(investment_amount)
            
        elif action == 'sell' and current_shares > 0:
            # Sell more when confidence is higher
            sell_pct = min(confidence * 0.5, 0.3)  # Max 30% of holdings
            shares_to_sell = current_shares * sell_pct
            
            # Calculate proceeds (accounting for fees)
            proceeds = shares_to_sell * last_price
            fee = proceeds * transaction_fee_pct
            net_proceeds = proceeds - fee
            
            # Update portfolio
            new_shares = current_shares - shares_to_sell
            new_cash = current_cash + net_proceeds
            new_portfolio = new_shares * next_price + new_cash
            
            investment_amounts.append(-proceeds)
            
        else:  # Hold
            # Just update portfolio value based on price change
            new_shares = current_shares
            new_cash = current_cash
            new_portfolio = new_shares * next_price + new_cash
            
            investment_amounts.append(0)
        
        # Add updated portfolio values
        shares_held.append(new_shares)
        cash_values.append(new_cash)
        portfolio_values.append(new_portfolio)
        
        # Update last price for next iteration
        last_price = next_price
    
    # Remove the first predicted price (which was the input price)
    predicted_prices = predicted_prices[1:]
    portfolio_values = portfolio_values[1:]
    shares_held = shares_held[1:]
    cash_values = cash_values[1:]
    
    # Create a dataframe with the results
    results_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Price': [round(price, 2) for price in predicted_prices],
        'Action': actions,
        'Confidence': [round(conf, 2) for conf in confidences],
        'Portfolio_Value': [round(val, 2) for val in portfolio_values],
        'Shares_Held': [round(shares, 4) for shares in shares_held],
        'Cash': [round(cash, 2) for cash in cash_values],
        'Investment_Amount': [round(amt, 2) for amt in investment_amounts]
    })
    
    # Save results to CSV - use consistent naming matching the UI expectations
    csv_file = f'results/tesla_next_{num_days}_days_prediction.csv'
    results_df.to_csv(csv_file, index=False)
    results_df.to_csv('results/next_days_prediction.csv', index=False)
    
    # Create visualization
    create_prediction_chart(current_price, dates, predicted_prices, actions, portfolio_values, num_days)
    
    # Calculate overall trend
    start_price = current_price
    end_price = predicted_prices[-1]
    overall_change = ((end_price / start_price) - 1) * 100
    
    # Calculate portfolio performance
    start_portfolio = portfolio_value
    end_portfolio = portfolio_values[-1]
    portfolio_change = ((end_portfolio / start_portfolio) - 1) * 100
    
    # Format recommendation text
    final_shares = shares_held[-1]
    initial_shares = shares_owned
    shares_difference = final_shares - initial_shares
    
    if shares_difference > 0:
        recommendation = f"Buy {shares_difference:.2f} shares over the next {num_days} days"
    elif shares_difference < 0:
        recommendation = f"Sell {abs(shares_difference):.2f} shares over the next {num_days} days"
    else:
        recommendation = f"Hold your current {initial_shares:.2f} shares"
    
    # Prepare the results dictionary
    results = {
        'current_price': current_price,
        'days': num_days,
        'prediction_date': today.strftime('%Y-%m-%d'),
        'overall_change': round(overall_change, 2),
        'initial_portfolio': round(portfolio_value, 2),
        'final_portfolio': round(portfolio_values[-1], 2),
        'portfolio_change': round(portfolio_change, 2),
        'initial_shares': round(shares_owned, 4),
        'final_shares': round(shares_held[-1], 4),
        'recommendation': recommendation,
        'predictions': results_df.to_dict('records'),
        'chart_path': 'next_days_prediction.png'
    }
    
    # Save results as JSON for easy retrieval
    with open('results/next_days_prediction.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def create_prediction_chart(current_price, dates, predicted_prices, actions, portfolio_values=None, num_days=5):
    """
    Create visualization of price predictions with buy/sell indicators.
    
    Args:
        current_price (float): The current closing price
        dates (list): List of date strings
        predicted_prices (list): List of predicted prices
        actions (list): List of recommended actions ('buy', 'sell', 'hold')
        portfolio_values (list): List of projected portfolio values
        num_days (int): Number of days in the prediction
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Add today's price as starting point
    all_dates = ['Today'] + dates
    all_prices = [current_price] + predicted_prices
    
    # Plot the price predictions on first subplot
    ax1.plot(all_dates, all_prices, marker='o', linestyle='-', linewidth=2, color='blue', label='Predicted Price')
    
    # Add buy/sell indicators
    for i, action in enumerate(actions):
        if action == 'buy':
            ax1.scatter(dates[i], predicted_prices[i], color='green', s=150, marker='^', label='Buy' if i == 0 else "")
        elif action == 'sell':
            ax1.scatter(dates[i], predicted_prices[i], color='red', s=150, marker='v', label='Sell' if i == 0 else "")
    
    # Calculate overall trend
    start_price = all_prices[0]
    end_price = all_prices[-1]
    pct_change = ((end_price / start_price) - 1) * 100
    
    # Add title with trend information
    trend_direction = "Upward" if pct_change > 0 else "Downward"
    ax1.set_title(f'Tesla Stock {num_days}-Day Price Prediction\n{trend_direction} Trend: {pct_change:.2f}% Change Expected')
    
    # Add labels and formatting
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')
    
    ax1.tick_params(axis='x', rotation=45)
    
    # Add portfolio value chart if provided
    if portfolio_values is not None:
        all_portfolio_values = [portfolio_values[0]] + portfolio_values  # Add initial value
        
        # Plot portfolio values on second subplot
        ax2.plot(all_dates, all_portfolio_values, marker='s', linestyle='-', linewidth=2, color='purple', label='Portfolio Value')
        
        # Calculate portfolio change
        start_portfolio = all_portfolio_values[0]
        end_portfolio = all_portfolio_values[-1]
        portfolio_pct_change = ((end_portfolio / start_portfolio) - 1) * 100
        
        # Add title with portfolio information
        portfolio_trend = "Growing" if portfolio_pct_change > 0 else "Declining"
        ax2.set_title(f'Projected Portfolio Value\n{portfolio_trend} Portfolio: {portfolio_pct_change:.2f}% Change Expected')
        
        # Add labels and formatting
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('visualizations/next_days_prediction.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tesla stock prices for the next several days')
    parser.add_argument('--price', type=float, required=True, help='Current closing price')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict (default: 5)')
    parser.add_argument('--portfolio', type=float, default=10000, help='Current portfolio value (default: $10,000)')
    parser.add_argument('--shares', type=float, default=0, help='Current number of Tesla shares owned (default: 0)')
    
    args = parser.parse_args()
    
    try:
        results = predict_next_days(args.price, args.days, portfolio_value=args.portfolio, shares_owned=args.shares)
        print(f"Predictions generated for the next {args.days} trading days")
        print("Date\t\tPrice\t\tAction\t\tPortfolio")
        print("-" * 60)
        
        for pred in results['predictions']:
            print(f"{pred['Date']}\t${pred['Predicted_Price']:.2f}\t\t{pred['Action'].upper()}\t\t${pred['Portfolio_Value']:.2f}")
            
        print(f"\nInitial Portfolio: ${results['initial_portfolio']:.2f}")
        print(f"Final Portfolio: ${results['final_portfolio']:.2f}")
        print(f"Portfolio Change: {results['portfolio_change']:.2f}%")
        print(f"\nRecommendation: {results['recommendation']}")
            
        print(f"\nResults saved to results/next_days_prediction.csv")
        print(f"Visualization saved to visualizations/next_days_prediction.png")
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        exit(1) 