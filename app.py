from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import base64
from io import BytesIO
import subprocess
import time

app = Flask(__name__)

# Modified static route to serve visualization files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('visualizations', filename)

# Configure Flask to use visualizations directory for static files
app.static_folder = 'visualizations'

@app.route('/')
def index():
    """Render the dashboard homepage"""
    # Check if results exist
    if not os.path.exists('results/performance_summary.csv') or not os.path.exists('results/trading_history.csv'):
        return render_template('index.html', has_results=False)
    
    # Load performance summary
    performance = pd.read_csv('results/performance_summary.csv')
    perf_dict = dict(zip(performance['Metric'], performance['Value']))
    
    # Get trading history info
    trading_history = pd.read_csv('results/trading_history.csv')
    history_start = trading_history['Date'].iloc[0]
    history_end = trading_history['Date'].iloc[-1]
    num_days = len(trading_history)
    
    # Get buy/sell counts
    buy_count = len(trading_history[trading_history['action'] == 'buy'])
    sell_count = len(trading_history[trading_history['action'] == 'sell'])
    
    # Generate chart for portfolio value
    portfolio_chart = generate_portfolio_chart(trading_history)
    
    # Generate chart for trading signals
    signals_chart = generate_signals_chart(trading_history)
    
    # Check if project simulation has been run
    has_project_sim = os.path.exists('results/tesla_march_2025_simulation.csv')
    project_performance = None
    
    if has_project_sim:
        # Load project performance if available
        try:
            project_performance = pd.read_csv('results/tesla_march_2025_performance.csv')
            project_performance = dict(zip(project_performance['Metric'], project_performance['Value']))
        except Exception as e:
            print(f"Error loading project simulation results: {e}")
            has_project_sim = False
    
    return render_template('index.html', 
                          has_results=True,
                          performance=perf_dict,
                          history_start=history_start,
                          history_end=history_end,
                          num_days=num_days,
                          buy_count=buy_count,
                          sell_count=sell_count,
                          portfolio_chart=portfolio_chart,
                          signals_chart=signals_chart,
                          has_project_sim=has_project_sim,
                          project_performance=project_performance)

def generate_portfolio_chart(trading_history):
    """Generate base64 encoded image of portfolio performance chart"""
    plt.figure(figsize=(10, 5))
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(trading_history['Date']):
        trading_history['Date'] = pd.to_datetime(trading_history['Date'])
    
    # Portfolio value
    plt.plot(trading_history['Date'], trading_history['portfolio_value'], 
             label='Portfolio Value', linewidth=2)
    
    # Stock price (scaled)
    first_portfolio = trading_history['portfolio_value'].iloc[0]
    first_close = trading_history['Close'].iloc[0]
    scale_factor = first_portfolio / first_close
    
    plt.plot(trading_history['Date'], trading_history['Close'] * scale_factor, 
             label='TSLA Stock Price (Scaled)', linestyle='--', alpha=0.7)
    
    plt.title('Portfolio Performance vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format dates
    plt.gcf().autofmt_xdate()
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode to base64
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Close the figure to free memory
    
    encoded = base64.b64encode(image_png)
    return encoded.decode('utf-8')

def generate_signals_chart(trading_history):
    """Generate base64 encoded image of trading signals chart"""
    plt.figure(figsize=(10, 5))
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(trading_history['Date']):
        trading_history['Date'] = pd.to_datetime(trading_history['Date'])
    
    # Plot stock price
    plt.plot(trading_history['Date'], trading_history['Close'], 
             label='TSLA Price', color='gray', alpha=0.6)
    
    # Mark buy points
    buy_points = trading_history[trading_history['action'] == 'buy']
    if not buy_points.empty:
        plt.scatter(buy_points['Date'], buy_points['Close'], 
                   marker='^', color='green', s=100, label='Buy')
    
    # Mark sell points
    sell_points = trading_history[trading_history['action'] == 'sell']
    if not sell_points.empty:
        plt.scatter(sell_points['Date'], sell_points['Close'], 
                   marker='v', color='red', s=100, label='Sell')
    
    plt.title('TSLA Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format dates
    plt.gcf().autofmt_xdate()
    
    # Save to BytesIO buffer
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode to base64
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Close the figure to free memory
    
    encoded = base64.b64encode(image_png)
    return encoded.decode('utf-8')

@app.route('/run-simulation', methods=['POST'])
def run_simulation():
    """Run the trading simulation"""
    try:
        # Run the simulation script
        result = subprocess.run(['python', 'run_final_simulation.py'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Add a small delay to ensure files are written
        time.sleep(1)
        
        # Display success message on next page load
        print("Simulation completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        print(e.stderr)
    
    # Redirect back to the dashboard
    return redirect(url_for('index'))

@app.route('/run-project-simulation', methods=['POST'])
def run_project_simulation():
    """Run the March 2025 project-specific trading simulation"""
    try:
        # Run the project simulation script
        result = subprocess.run(['python', 'run_simulation_2025.py'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Add a small delay to ensure files are written
        time.sleep(1)
        
        print("Project simulation completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running project simulation: {e}")
        print(e.stderr)
    
    # Redirect back to the dashboard
    return redirect(url_for('index'))

@app.route('/data')
def get_data():
    """Return trading history data as JSON for interactive charts"""
    if not os.path.exists('results/trading_history.csv'):
        return json.dumps({'error': 'No data available'})
    
    trading_history = pd.read_csv('results/trading_history.csv')
    
    # Convert to format suitable for charts
    data = {
        'dates': trading_history['Date'].tolist(),
        'portfolio': trading_history['portfolio_value'].tolist(),
        'prices': trading_history['Close'].tolist(),
        'actions': trading_history['action'].tolist()
    }
    
    return json.dumps(data)

@app.route('/details')
def details():
    """Show detailed trading history"""
    if not os.path.exists('results/trading_history.csv'):
        return redirect(url_for('index'))
    
    trading_history = pd.read_csv('results/trading_history.csv')
    
    # Filter to only show important columns
    if 'portfolio_value' in trading_history.columns and 'Close' in trading_history.columns and 'action' in trading_history.columns:
        display_cols = ['Date', 'Close', 'portfolio_value', 'action']
        
        # Add any other trading-specific columns that exist
        for col in ['prediction', 'daily_return', 'MA5', 'MA20']:
            if col in trading_history.columns:
                display_cols.append(col)
        
        history_data = trading_history[display_cols].to_dict('records')
    else:
        history_data = trading_history.to_dict('records')
    
    return render_template('details.html', history=history_data)

@app.route('/project-details')
def project_details():
    """Show detailed project simulation trading history"""
    if not os.path.exists('results/tesla_march_2025_simulation.csv'):
        return redirect(url_for('index'))
    
    project_history = pd.read_csv('results/tesla_march_2025_simulation.csv')
    
    # Convert column names to match template expectations
    if 'Action' in project_history.columns:
        project_history['action'] = project_history['Action']
    
    if 'Portfolio_Value' in project_history.columns:
        project_history['portfolio_value'] = project_history['Portfolio_Value']
    
    history_data = project_history.to_dict('records')
    
    return render_template('project_details.html', history=history_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    # Ensure results and visualizations directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    print("Starting Tesla Stock Trading Dashboard...")
    print("Open your web browser and navigate to http://localhost:5000")
    app.run(debug=True) 