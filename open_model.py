import pickle
import sys
import os

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Open the model file
with open('models/ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Print model information
print(f"Model type: {type(model)}")
print(f"Model keys: {model.models.keys()}")
print(f"Model weights: {model.model_weights}")

# Print more detailed information if available
if hasattr(model, 'feature_importances_'):
    print(f"Feature importances: {model.feature_importances_}") 