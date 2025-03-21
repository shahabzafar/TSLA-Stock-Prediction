import pickle
import sys

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            print(f"Could not import {module}.{name}")
            return None
        except AttributeError:
            print(f"Could not find {name} in {module}")
            return None

# Try to load the pickle file and examine its contents
try:
    with open('models/ensemble_model.pkl', 'rb') as f:
        unpickler = CustomUnpickler(f)
        model = unpickler.load()
        
    # Print basic information
    print(f"\nSuccessfully loaded the model")
    print(f"Model type: {type(model)}")
    
    # Try to access various attributes
    for attr in ['models', 'model_weights', 'feature_importances_']:
        try:
            value = getattr(model, attr, "Not available")
            print(f"{attr}: {value}")
        except Exception as e:
            print(f"Could not access {attr}: {e}")
            
except Exception as e:
    print(f"Error loading pickle file: {e}") 