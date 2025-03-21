import pickle
import os

print("Checking pickled files in models directory:")

# Check feature_columns.pkl
feature_columns_path = 'models/feature_columns.pkl'
if os.path.exists(feature_columns_path):
    try:
        with open(feature_columns_path, 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"Feature columns: {feature_columns}")
    except Exception as e:
        print(f"Error loading feature_columns.pkl: {e}")
else:
    print(f"File not found: {feature_columns_path}")

# Check normalization_params.pkl
norm_params_path = 'models/normalization_params.pkl'
if os.path.exists(norm_params_path):
    try:
        with open(norm_params_path, 'rb') as f:
            norm_params = pickle.load(f)
        print(f"Normalization params: {norm_params}")
    except Exception as e:
        print(f"Error loading normalization_params.pkl: {e}")
else:
    print(f"File not found: {norm_params_path}")

# Check ensemble_model.pkl
ensemble_path = 'models/ensemble_model.pkl'
if os.path.exists(ensemble_path):
    try:
        with open(ensemble_path, 'rb') as f:
            ensemble_model = pickle.load(f)
        print(f"Ensemble model loaded: {type(ensemble_model)}")
    except Exception as e:
        print(f"Error loading ensemble_model.pkl: {e}")
else:
    print(f"File not found: {ensemble_path}")

# List LSTM model files
lstm_path = 'models/lstm_model'
if os.path.exists(lstm_path):
    print(f"LSTM model directory exists: {lstm_path}")
    print("Contents:")
    for item in os.listdir(lstm_path):
        print(f"  - {item}")
else:
    print(f"Directory not found: {lstm_path}") 