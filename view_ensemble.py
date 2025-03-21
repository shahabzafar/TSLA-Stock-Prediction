import pickle
import sys
import os

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to load the model
try:
    print("Loading ensemble model...")
    with open('models/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("\nModel Information:")
    print("-----------------")
    print(f"Type: {type(model)}")
    
    # Get the model's attributes
    print("\nModel Attributes:")
    print("-----------------")
    
    # Check model components
    if hasattr(model, 'models'):
        print(f"Models: {list(model.models.keys())}")
        
        # Print details of each sub-model
        print("\nSub-models:")
        print("-----------")
        for name, sub_model in model.models.items():
            print(f"{name}: {type(sub_model)}")
    else:
        print("No 'models' attribute found")
    
    # Check model weights
    if hasattr(model, 'model_weights'):
        print(f"\nModel Weights: {model.model_weights}")
    else:
        print("\nNo 'model_weights' attribute found")
    
    # Check feature importances
    if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
        print(f"\nFeature Importances: {model.feature_importances_}")
    else:
        print("\nNo feature importances available")
        
except Exception as e:
    print(f"Error loading model: {e}") 