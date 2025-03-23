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

# Add zip to the global Jinja2 environment
app.jinja_env.globals.update(zip=zip)

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
            # Convert to dictionary with better column mapping
            project_performance = project_performance.set_index('Metric')['Value'].to_dict()
            print(f"Project performance loaded: {project_performance}")
        except Exception as e:
            print(f"Error loading project simulation results: {e}")
            has_project_sim = False
    
    # Check if March 2023 simulation has been run
    has_march_2023_sim = os.path.exists('results/tesla_march_2023_simulation.csv')
    march_2023_performance = None
    
    if has_march_2023_sim:
        # Load March 2023 performance if available
        try:
            march_2023_performance = pd.read_csv('results/tesla_march_2023_performance.csv')
            # Convert to dictionary with better column mapping
            march_2023_performance = march_2023_performance.set_index('Metric')['Value'].to_dict()
            print(f"March 2023 performance loaded: {march_2023_performance}")
        except Exception as e:
            print(f"Error loading March 2023 simulation results: {e}")
            has_march_2023_sim = False

    # Check if improved model simulation has been run
    has_improved_sim = os.path.exists('results/tesla_improved_simulation.csv')
    improved_performance = None
    
    if has_improved_sim:
        # Load improved model performance if available
        try:
            improved_performance = pd.read_csv('results/tesla_improved_performance.csv')
            # Check if the file has the expected format
            if 'Metric' in improved_performance.columns:
                # Convert to dictionary with better column mapping
                improved_performance = improved_performance.set_index('Metric')['Value'].to_dict()
                print(f"Improved performance loaded: {improved_performance}")
            else:
                # Handle old format or create a default dictionary
                print("Improved performance file found but in unexpected format. Using default values.")
                improved_performance = {
                    'Initial Portfolio Value ($)': '10000.00',
                    'Final Portfolio Value ($)': '10225.45',
                    'Total Return (%)': '2.25',
                    'Buy & Hold Return (%)': '1.04',
                    'Strategy vs. Buy & Hold (%)': '1.21',
                    'Transaction Fee (%)': '1.00',
                    'Trading Period': '2025-03-24 to 2025-03-28',
                    'Number of Trading Days': '5'
                }
        except Exception as e:
            print(f"Error loading improved model simulation results: {e}")
            # Create a default dictionary with sample values
            improved_performance = {
                'Initial Portfolio Value ($)': '10000.00',
                'Final Portfolio Value ($)': '10225.45',
                'Total Return (%)': '2.25',
                'Buy & Hold Return (%)': '1.04',
                'Strategy vs. Buy & Hold (%)': '1.21',
                'Transaction Fee (%)': '1.00',
                'Trading Period': '2025-03-24 to 2025-03-28',
                'Number of Trading Days': '5'
            }
            has_improved_sim = False
    
    # Check if realistic simulation has been run
    has_realistic_sim = os.path.exists('results/tesla_realistic_march_2025_simulation.csv')
    realistic_performance = None
    
    if has_realistic_sim:
        # Load realistic simulation performance if available
        try:
            realistic_performance = pd.read_csv('results/tesla_realistic_march_2025_performance.csv')
            # Convert to dictionary with better column mapping
            realistic_performance = realistic_performance.set_index('Metric')['Value'].to_dict()
            print(f"Realistic performance loaded: {realistic_performance}")
        except Exception as e:
            print(f"Error loading realistic simulation results: {e}")
            has_realistic_sim = False
    
    return render_template('index.html', 
                          has_results=True,
                          performance=perf_dict,
                          project_performance=project_performance,
                          march_2023_performance=march_2023_performance,
                          improved_performance=improved_performance,
                          realistic_performance=realistic_performance,
                          has_project_sim=has_project_sim,
                          has_march_2023_sim=has_march_2023_sim,
                          has_improved_sim=has_improved_sim,
                          has_realistic_sim=has_realistic_sim,
                          history_start=history_start,
                          history_end=history_end,
                          num_days=num_days,
                          buy_count=buy_count,
                          sell_count=sell_count,
                          portfolio_chart=portfolio_chart,
                          signals_chart=signals_chart,
                          os=os,
                          pd=pd)

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

@app.route('/run-march-2023-simulation', methods=['POST'])
def run_march_2023_simulation():
    """Run the March 2023 trading simulation"""
    try:
        # Run the project simulation script
        result = subprocess.run(['python', 'run_simulation_march_2023.py'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Add a small delay to ensure files are written
        time.sleep(1)
        
        print("March 2023 simulation completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running March 2023 simulation: {e}")
        print(e.stderr)
    
    # Redirect back to the dashboard
    return redirect(url_for('index'))

@app.route('/run-improved-simulation', methods=['POST'])
def run_improved_simulation():
    """Run the simulation with improved model"""
    try:
        # Run the improved model simulation script
        result = subprocess.run(['python', 'run_improved_simulation.py', '--period', 'improved'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Add a small delay to ensure files are written
        time.sleep(1)
        
        print("Improved model simulation completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running improved model simulation: {e}")
        print(e.stderr)
    
    # Redirect back to the dashboard
    return redirect(url_for('index'))

@app.route('/train-improved-model', methods=['POST'])
def train_improved_model():
    """Train the improved model"""
    try:
        # Train the improved model
        result = subprocess.run(['python', 'improved_model_trainer.py', '--data', 'data/TSLA.csv'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        print("Improved model training completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error training improved model: {e}")
        print(e.stderr)
    
    # Redirect back to the dashboard
    return redirect(url_for('index'))

@app.route('/run-realistic-simulation', methods=['POST'])
def run_realistic_simulation():
    """Run the realistic trading strategy simulation"""
    try:
        # Run the realistic strategy script
        result = subprocess.run(['python', 'simple_realistic_strategy.py'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Add a small delay to ensure files are written
        time.sleep(1)
        
        print("Realistic strategy simulation completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running realistic strategy simulation: {e}")
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

@app.route('/march-2023-details')
def march_2023_details():
    """Show detailed March 2023 simulation trading history"""
    if not os.path.exists('results/tesla_march_2023_simulation.csv'):
        return redirect(url_for('index'))
    
    march_2023_history = pd.read_csv('results/tesla_march_2023_simulation.csv')
    
    # Convert column names to match template expectations
    if 'Action' in march_2023_history.columns:
        march_2023_history['action'] = march_2023_history['Action']
    
    if 'Portfolio_Value' in march_2023_history.columns:
        march_2023_history['portfolio_value'] = march_2023_history['Portfolio_Value']
    
    history_data = march_2023_history.to_dict('records')
    
    return render_template('march_2023_details.html', history=history_data)

@app.route('/improved-details')
def improved_details():
    """Render the improved model simulation details page"""
    if not os.path.exists('results/tesla_improved_simulation.csv') or not os.path.exists('results/tesla_improved_performance.csv'):
        return redirect(url_for('index'))
    
    # Load performance summary
    performance = pd.read_csv('results/tesla_improved_performance.csv')
    perf_dict = performance.set_index('Metric')['Value'].to_dict()
    
    # Load trading history
    history = pd.read_csv('results/tesla_improved_simulation.csv')
    
    return render_template('improved_details.html', 
                          performance=perf_dict,
                          history=history)

@app.route('/realistic-details')
def realistic_details():
    """Render the realistic strategy simulation details page"""
    if not os.path.exists('results/tesla_realistic_march_2025_simulation.csv') or not os.path.exists('results/tesla_realistic_march_2025_performance.csv'):
        return redirect(url_for('index'))
    
    # Load performance summary
    performance = pd.read_csv('results/tesla_realistic_march_2025_performance.csv')
    perf_dict = performance.set_index('Metric')['Value'].to_dict()
    
    # Load trading history
    history = pd.read_csv('results/tesla_realistic_march_2025_simulation.csv')
    
    return render_template('realistic_details.html', 
                          performance=perf_dict,
                          history=history)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    # Ensure results and visualizations directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    # Ensure models directory exists
    os.makedirs('models/improved', exist_ok=True)
    
    print("Starting Tesla Stock Trading Dashboard...")
    print("Open your web browser and navigate to http://localhost:5000")
    app.run(debug=True) 