from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session
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
import calendar

app = Flask(__name__)
app.secret_key = 'tsla_predictions_secret_key'  # Required for session

# Add zip to the global Jinja2 environment
app.jinja_env.globals.update(zip=zip)

# Modified static route to serve visualization files
@app.route('/<path:filename>')
def serve_visualizations(filename):
    return send_from_directory('visualizations', filename)

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
        return render_template('index.html', has_results=False, current_year=datetime.now().year)
    
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
                          realistic_performance=realistic_performance,
                          has_project_sim=has_project_sim,
                          has_march_2023_sim=has_march_2023_sim,
                          has_realistic_sim=has_realistic_sim,
                          history_start=history_start,
                          history_end=history_end,
                          num_days=num_days,
                          buy_count=buy_count,
                          sell_count=sell_count,
                          portfolio_chart=portfolio_chart,
                          signals_chart=signals_chart,
                          current_year=datetime.now().year,
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

@app.route('/run-dynamic-simulation', methods=['POST'])
def run_dynamic_simulation():
    """Run the dynamic date simulation for user-selected month and year"""
    try:
        # Get year and month from form data
        year = request.form.get('year', type=int)
        month = request.form.get('month', type=int)
        
        if not year or not month or month < 1 or month > 12:
            # Invalid input
            print("Invalid year or month input")
            return jsonify({'status': 'error', 'message': 'Invalid month or year'})
        
        # Get month name for file naming and messages
        month_name = calendar.month_name[month].lower()
        print(f"Starting dynamic simulation for {month_name} {year}...")
        
        # Run the dynamic simulation script with proper arguments
        cmd = ['python', 'dynamic_simulation.py', 
               '--year', str(year), 
               '--month', str(month)]
        
        print(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if the process was successful
        if result.returncode != 0:
            print(f"Error in simulation: {result.stderr}")
            return jsonify({
                'status': 'error',
                'message': f'Error running simulation: {result.stderr}'
            })
        
        # Add a small delay to ensure files are written
        time.sleep(1)
        
        print(f"Dynamic simulation for {month_name} {year} completed successfully")
        if result.stdout:
            print(result.stdout)
        
        # Store the completed simulation info in session
        session['last_simulation'] = {
            'month': month,
            'year': year,
            'month_name': month_name
        }
        
        return jsonify({
            'status': 'success', 
            'message': f'Simulation for {calendar.month_name[month]} {year} completed successfully',
            'redirect': f'/dynamic-simulation-details?month={month}&year={year}'
        })
        
    except Exception as e:
        print(f"Error running dynamic simulation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error running simulation: {str(e)}'
        })

@app.route('/check-simulation-status')
def check_simulation_status():
    """Check if there's a completed simulation in the session"""
    if 'last_simulation' in session:
        sim_info = session['last_simulation']
        # Clear from session after retrieving
        session.pop('last_simulation', None)
        return jsonify({
            'status': 'success',
            'simulation': sim_info
        })
    else:
        return jsonify({
            'status': 'none'
        })

@app.route('/dynamic-simulation-details')
def dynamic_simulation_details():
    """Show detailed results for dynamic date simulation"""
    # Get parameters from query string
    year = request.args.get('year', type=int)
    month = request.args.get('month', type=int)
    
    if not year or not month or month < 1 or month > 12:
        return redirect(url_for('index'))
    
    # Get month name for file naming
    month_name = calendar.month_name[month].lower()
    
    # Check if simulation results exist
    sim_file = f'results/tesla_{month_name}_{year}_simulation.csv'
    perf_file = f'results/tesla_{month_name}_{year}_performance.csv'
    
    if not os.path.exists(sim_file) or not os.path.exists(perf_file):
        return redirect(url_for('index'))
    
    # Load simulation data
    simulation_data = pd.read_csv(sim_file)
    performance_data = pd.read_csv(perf_file)
    
    # Convert performance data to dictionary
    perf_dict = performance_data.set_index('Metric')['Value'].to_dict()
    
    # Pass month and year to template for image paths
    return render_template(
        'dynamic_details.html',
        history=simulation_data.to_dict('records'),
        performance=perf_dict,
        month=month,
        month_name=month_name.capitalize(),
        year=year
    )

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

@app.route('/run-next-days-prediction', methods=['POST'])
def run_next_days_prediction():
    """Run a prediction for the next X days based on current price."""
    try:
        current_price = request.form.get('current_price')
        days = request.form.get('days', '5')
        portfolio_value = request.form.get('portfolio_value', '10000')
        shares_owned = request.form.get('shares_owned', '0')
        
        # Validate inputs
        if not current_price:
            return jsonify({'status': 'error', 'message': 'Current price is required'})
        
        try:
            current_price = float(current_price)
            days = int(days)
            portfolio_value = float(portfolio_value)
            shares_owned = float(shares_owned)
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid input values'})
        
        # Run the prediction script
        cmd = f"python next_days_prediction.py --price {current_price} --days {days}"
        if portfolio_value is not None and shares_owned is not None:
            cmd += f" --portfolio {portfolio_value} --shares {shares_owned}"
            
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running prediction: {result.stderr}")
            return jsonify({'status': 'error', 'message': 'Error running prediction'})
        
        # Store the prediction info in the session
        session['prediction'] = {
            'current_price': current_price,
            'days': days,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'shares_owned': shares_owned
        }
        
        return jsonify({
            'status': 'success',
            'redirect': '/next-days-prediction-details'
        })
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/check-prediction-status')
def check_prediction_status():
    """Check if a prediction has been completed."""
    if 'prediction' in session:
        prediction_data = session['prediction']
        # Check if there is a results file
        results_file = os.path.join('results', 'next_days_prediction.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    # Add portfolio data if available
                    if 'portfolio_value' in prediction_data and 'shares_owned' in prediction_data:
                        results['portfolio_value'] = prediction_data.get('portfolio_value')
                        results['shares_owned'] = prediction_data.get('shares_owned')
            except Exception as e:
                print(f"Error reading prediction results: {e}")
                
        return jsonify({
            'status': 'success',
            'prediction': {
                'current_price': prediction_data.get('current_price'),
                'days': prediction_data.get('days'),
                'date': prediction_data.get('date'),
                'portfolio_value': prediction_data.get('portfolio_value', 10000),
                'shares_owned': prediction_data.get('shares_owned', 0)
            }
        })
    return jsonify({'status': 'no_prediction'})

@app.route('/results/<path:filename>')
def serve_result_file(filename):
    """Serve files from the results directory"""
    return send_from_directory('results', filename)

@app.route('/next_days_prediction.png')
def serve_prediction_chart():
    """Serve the prediction chart image"""
    return send_from_directory('visualizations', 'next_days_prediction.png')

@app.route('/next-days-prediction-details')
def next_days_prediction_details():
    """Show detailed results for next days prediction"""
    # Check if prediction results exist
    json_file = 'results/next_days_prediction.json'
    
    if not os.path.exists(json_file):
        return redirect(url_for('index'))
    
    try:
        # Load JSON data for prediction details
        with open(json_file, 'r') as f:
            prediction_data = json.load(f)
        
        # Get prediction information
        current_price = prediction_data.get('current_price', 0)
        days = prediction_data.get('days', 5)
        prediction_date = prediction_data.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
        predictions = prediction_data.get('predictions', [])
        
        # Get portfolio information
        portfolio_value = prediction_data.get('portfolio_value', 10000)
        shares_owned = prediction_data.get('shares_owned', 0)
        
        # Calculate overall trend
        if predictions and len(predictions) > 0:
            first_price = predictions[0]['Predicted_Price']
            last_price = predictions[-1]['Predicted_Price']
            overall_change = ((last_price / current_price) - 1) * 100
            
            # Calculate portfolio values
            if 'Portfolio_Value' in predictions[-1]:
                initial_portfolio = portfolio_value
                final_portfolio = predictions[-1]['Portfolio_Value']
                portfolio_change = ((final_portfolio / initial_portfolio) - 1) * 100
            else:
                initial_portfolio = portfolio_value
                final_portfolio = initial_portfolio * (1 + (overall_change / 100))
                portfolio_change = overall_change
        else:
            overall_change = 0.0
            initial_portfolio = portfolio_value
            final_portfolio = portfolio_value
            portfolio_change = 0.0
        
        # Pass data to template
        return render_template(
            'next_days_prediction.html',
            predictions=predictions,
            current_price=current_price,
            prediction_date=prediction_date,
            days=days,
            overall_change=overall_change,
            initial_portfolio=initial_portfolio,
            final_portfolio=final_portfolio,
            portfolio_change=portfolio_change,
            shares_owned=shares_owned
        )
    except Exception as e:
        print(f"Error rendering prediction details: {str(e)}")
        return redirect(url_for('index'))

@app.route('/reset-prediction', methods=['POST'])
def reset_prediction():
    """Clear the prediction data from the session."""
    if 'prediction' in session:
        del session['prediction']
    return jsonify({'status': 'success', 'message': 'Prediction data cleared'})

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