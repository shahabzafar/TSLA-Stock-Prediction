import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Add parent directory to sys.path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ModelViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ensemble Model Viewer")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Setup the UI
        self.setup_ui()
        
        # Load the model
        self.load_model()
        
    def setup_ui(self):
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Create frames for each tab
        self.overview_frame = ttk.Frame(self.notebook)
        self.submodels_frame = ttk.Frame(self.notebook)
        self.weights_frame = ttk.Frame(self.notebook)
        self.features_frame = ttk.Frame(self.notebook)
        
        # Add the frames to notebook
        self.notebook.add(self.overview_frame, text="Overview")
        self.notebook.add(self.submodels_frame, text="Sub-models")
        self.notebook.add(self.weights_frame, text="Weights")
        self.notebook.add(self.features_frame, text="Features")
        
        self.notebook.pack(expand=1, fill="both", padx=10, pady=10)
        
    def load_model(self):
        try:
            # Load the model
            with open('models/ensemble_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Populate the overview tab
            self.populate_overview()
            
            # Populate the submodels tab
            self.populate_submodels()
            
            # Populate the weights tab
            self.populate_weights()
            
            # Populate the features tab
            self.populate_features()
            
        except Exception as e:
            self.show_error(f"Error loading model: {str(e)}")
    
    def populate_overview(self):
        # Create a frame to hold the overview information
        frame = ttk.LabelFrame(self.overview_frame, text="Model Information")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add the model type
        ttk.Label(frame, text=f"Model Type: {type(self.model).__name__}").pack(anchor="w", padx=10, pady=5)
        
        # Add number of submodels
        ttk.Label(frame, text=f"Number of Sub-models: {len(self.model.models)}").pack(anchor="w", padx=10, pady=5)
        
        # Add model keys
        ttk.Label(frame, text="Model Keys:").pack(anchor="w", padx=10, pady=5)
        for key in self.model.models.keys():
            ttk.Label(frame, text=f"  • {key}").pack(anchor="w", padx=30, pady=2)
    
    def populate_submodels(self):
        # Create a frame to hold the submodels information
        frame = ttk.LabelFrame(self.submodels_frame, text="Sub-models Details")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a treeview to display the submodels
        columns = ("model_name", "model_type")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # Define headings
        tree.heading("model_name", text="Model Name")
        tree.heading("model_type", text="Model Type")
        
        # Define columns
        tree.column("model_name", width=150)
        tree.column("model_type", width=400)
        
        # Insert data
        for name, sub_model in self.model.models.items():
            tree.insert("", "end", values=(name, type(sub_model).__name__))
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        scrollbar.pack(side="right", fill="y")
        tree.pack(expand=True, fill="both")
    
    def populate_weights(self):
        # Create a frame to hold the weights information
        frame = ttk.LabelFrame(self.weights_frame, text="Model Weights")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Extract data
        names = list(self.model.model_weights.keys())
        weights = list(self.model.model_weights.values())
        
        # Create the bar chart
        bars = ax.bar(names, weights)
        
        # Add labels and title
        ax.set_xlabel('Model Name')
        ax.set_ylabel('Weight')
        ax.set_title('Model Weights in Ensemble')
        
        # Add the plot to the frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def populate_features(self):
        if not hasattr(self.model, 'feature_importances_') or self.model.feature_importances_ is None:
            ttk.Label(self.features_frame, text="No feature importances available").pack(anchor="center", padx=10, pady=10)
            return
        
        # Create a frame to hold the features information
        frame = ttk.LabelFrame(self.features_frame, text="Feature Importances")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Extract data
        feature_importances = self.model.feature_importances_
        feature_indices = np.arange(len(feature_importances))
        
        # Sort indices by importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        
        # Create the bar chart
        bars = ax.bar(
            [f"Feature {i}" for i in sorted_indices[:10]], 
            [feature_importances[i] for i in sorted_indices[:10]]
        )
        
        # Add labels and title
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        plt.xticks(rotation=45, ha='right')
        
        # Add the plot to the frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add a table view of all features
        ttk.Label(frame, text="All Feature Importances:").pack(anchor="w", padx=10, pady=5)
        
        # Create a frame for the table
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create a treeview to display the features
        columns = ("feature_id", "importance")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # Define headings
        tree.heading("feature_id", text="Feature ID")
        tree.heading("importance", text="Importance")
        
        # Define columns
        tree.column("feature_id", width=150)
        tree.column("importance", width=150)
        
        # Insert data
        for i, importance in enumerate(feature_importances):
            tree.insert("", "end", values=(f"Feature {i}", f"{importance:.6f}"))
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the widgets
        scrollbar.pack(side="right", fill="y")
        tree.pack(expand=True, fill="both")
    
    def show_error(self, message):
        for frame in [self.overview_frame, self.submodels_frame, self.weights_frame, self.features_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            ttk.Label(frame, text=message, foreground="red").pack(anchor="center", padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelViewerApp(root)
    root.mainloop() 