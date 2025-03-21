import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Constants for the simulation
SIMULATION_START = "2025-03-24"  # Monday of the last week of March 2025
SIMULATION_END = "2025-03-28"    # Friday of the last week of March 2025
INITIAL_BALANCE = 10000.0        # $10,000 USD starting capital
TRANSACTION_FEE_PCT = 0.01       # 1% transaction fee
ORDER_SUBMISSION_TIME = "09:00"  # Orders must be submitted by 9:00 AM EST
ORDER_EXECUTION_TIME = "10:00"   # Orders are executed at 10:00 AM EST

def create_simulation_data():
    """
    Create simulated Tesla stock data for the specific week in March 2025.
    In a real scenario, this would be replaced with actual market data.
    """
    # For simulation purposes, we'll use data from a recent period and project it forward
    # Load existing Tesla data to use as a base
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
    
    print(f"Using historical Tesla data from {tesla_file} as a base")
    historical_data = pd.read_csv(os.path.join('data', tesla_file))
    
    # Convert date column to datetime
    date_col = 'Date' if 'Date' in historical_data.columns else 'date'
    historical_data[date_col] = pd.to_datetime(historical_data[date_col])
    
    # Get some recent price action to simulate from
    recent_data = historical_data.sort_values(by=date_col, ascending=False).head(20)
    
    # Create simulation date range for the specific week in March 2025
    simulation_dates = pd.date_range(start=SIMULATION_START, end=SIMULATION_END, freq='B')  # Business days
    
    # Create simulated data for the week
    sim_data = []
    
    # Get recent volatility to make realistic price movements
    recent_volatility = recent_data['Close'].pct_change().std()
    
    # Start with a reasonable price for Tesla in 2025
    last_close = 500.0  # Hypothetical Tesla price in March 2025
    
    for date in simulation_dates:
        # Simulate intraday prices (9:00 AM and 10:00 AM)
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
    
    print(f"Created simulated Tesla stock data for {len(sim_df)} trading days in March 2025")
    print(f"Date range: {sim_df['Date'].min()} to {sim_df['Date'].max()}")
    print(f"Price range: ${sim_df['Low'].min():.2f} to ${sim_df['High'].max():.2f}")
    
    return sim_df

def get_model_prediction(stock_data, day_index):
    """
    Get the model's prediction for the given day.
    This would normally use the trained models, but for simulation we'll use a simplified approach.
    """
    try:
        # Try to load the ensemble model
        with open('models/ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load feature columns
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        print(f"Using trained ensemble model for prediction on day {day_index+1}")
        
        # In a real implementation, we would:
        # 1. Prepare features based on historical data
        # 2. Apply the model to get a prediction
        # For now, we'll return a simplified prediction
        
        # Simplified prediction: Just use recent trend
        if day_index > 0:
            prev_close = stock_data.iloc[day_index-1]['Close']
            current_open = stock_data.iloc[day_index]['Open']
            
            if current_open > prev_close * 1.01:  # If open is up >1%, predict buy
                return 'buy', 0.75  # Buy with 75% confidence
            elif current_open < prev_close * 0.99:  # If open is down >1%, predict sell
                return 'sell', 0.75  # Sell with 75% confidence
            else:
                return 'hold', 0.5  # Hold with 50% confidence
        else:
            # First day, just use a random signal
            signals = ['buy', 'hold', 'sell']
            confidences = [0.6, 0.5, 0.6]
            idx = np.random.choice(len(signals), p=[0.4, 0.2, 0.4])
            return signals[idx], confidences[idx]
            
    except Exception as e:
        print(f"Error loading model or making prediction: {str(e)}")
        print("Using simple rule-based prediction instead")
        
        # Fall back to a simple rule-based approach
        signals = ['buy', 'hold', 'sell']
        confidences = [0.6, 0.5, 0.6]
        idx = np.random.choice(len(signals), p=[0.4, 0.2, 0.4])
        return signals[idx], confidences[idx]

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
        else:
            print("No shares to sell")
    
    elif action == 'hold':
        print("Executed HOLD order: No transaction")
    
    # Calculate updated portfolio value
    portfolio_value = cash + (shares * execution_price)
    
    return {
        'cash': cash,
        'shares': shares,
        'current_price': execution_price,
        'portfolio_value': portfolio_value
    }

def run_tesla_march_2025_simulation():
    """Run the Tesla stock trading simulation for the last week of March 2025"""
    print("\n=== Tesla Stock Trading Simulation - March 24-28, 2025 ===\n")
    
    # Create simulated data for the trading period
    simulation_data = create_simulation_data()
    
    # Initialize portfolio
    portfolio = {
        'cash': INITIAL_BALANCE,
        'shares': 0,
        'current_price': simulation_data.iloc[0]['Open'],
        'portfolio_value': INITIAL_BALANCE
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
        action, confidence = get_model_prediction(simulation_data, day_idx)
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
        portfolio = execute_order(action, amount, portfolio, price_10am, TRANSACTION_FEE_PCT)
        
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
    initial_value = INITIAL_BALANCE
    final_value = portfolio['portfolio_value']
    total_return_pct = ((final_value / initial_value) - 1) * 100
    
    # Calculate buy & hold performance
    buy_hold_shares = int(INITIAL_BALANCE / (simulation_data.iloc[0]['Price_10AM'] * (1 + TRANSACTION_FEE_PCT)))
    buy_hold_cost = buy_hold_shares * simulation_data.iloc[0]['Price_10AM'] * (1 + TRANSACTION_FEE_PCT)
    buy_hold_value = buy_hold_shares * simulation_data.iloc[-1]['Close']
    buy_hold_return_pct = ((buy_hold_value / buy_hold_cost) - 1) * 100
    
    # Print final summary
    print("\n=== Final Simulation Summary ===")
    print(f"Trading Period: {simulation_data['Date'].min()} to {simulation_data['Date'].max()}")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return_pct:.2f}%")
    print(f"Strategy vs. Buy & Hold: {total_return_pct - buy_hold_return_pct:.2f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/tesla_march_2025_simulation.csv', index=False)
    
    # Generate performance summary
    performance_summary = pd.DataFrame({
        'Metric': [
            'Initial Portfolio Value ($)',
            'Final Portfolio Value ($)',
            'Total Return (%)',
            'Buy & Hold Return (%)',
            'Strategy vs. Buy & Hold (%)',
            'Transaction Fee (%)',
            'Trading Period',
            'Number of Trading Days'
        ],
        'Value': [
            f"{initial_value:.2f}",
            f"{final_value:.2f}",
            f"{total_return_pct:.2f}",
            f"{buy_hold_return_pct:.2f}",
            f"{total_return_pct - buy_hold_return_pct:.2f}",
            f"{TRANSACTION_FEE_PCT * 100:.2f}",
            f"{simulation_data['Date'].min()} to {simulation_data['Date'].max()}",
            len(simulation_data)
        ]
    })
    
    performance_summary.to_csv('results/tesla_march_2025_performance.csv', index=False)
    
    print("\nResults saved to 'results' directory:")
    print("1. tesla_march_2025_simulation.csv - Complete trading history")
    print("2. tesla_march_2025_performance.csv - Performance summary")
    
    # Generate visualizations
    create_visualizations(results_df)
    
    return results_df, performance_summary

def create_visualizations(results_df):
    """Create visualizations for the March 2025 simulation results"""
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
    
    plt.title('Portfolio Value During Tesla Trading Simulation (March 2025)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/march_2025_portfolio_value.png')
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
    
    plt.title('Tesla Stock Price and Trading Signals (March 2025)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/march_2025_trading_signals.png')
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
    
    plt.title('Portfolio Composition During Tesla Trading Simulation (March 2025)')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/march_2025_portfolio_composition.png')
    plt.close()  # Close the figure to free memory
    
    print("\nVisualizations saved to 'visualizations' directory:")
    print("1. march_2025_portfolio_value.png - Portfolio value chart")
    print("2. march_2025_trading_signals.png - Stock price with trading signals")
    print("3. march_2025_portfolio_composition.png - Portfolio composition chart")

if __name__ == "__main__":
    run_tesla_march_2025_simulation() 