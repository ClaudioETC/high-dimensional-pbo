import torch
import numpy as np

# Import the trial runner
from pbo_trial import pbo_trial

# Import your benchmark of choice (e.g., Hartmann6 is a classic D=6 optimization benchmark)
from test_funcs import Hartmann6

def run():
    # 1. Setup the Objective Function
    input_dim = 6
    benchmark = Hartmann6(dim=input_dim)
    
    # BoTorch passes a 2D tensor (N, d). We must return a 1D tensor (N,)
    def obj_func_wrapper(X_tensor):
        # Convert PyTorch tensor to NumPy
        X_np = X_tensor.detach().numpy()
        
        # test_funcs.py has a .f() method that handles 2D numpy arrays
        # We negate it because BoTorch MAXIMIZES, but the benchmark MINIMIZES
        y_np = -benchmark.f(X_np) 
        
        # Convert back to PyTorch tensor
        return torch.tensor(y_np, dtype=torch.float32)

    print(f"Starting High-Dimensional qEUBO Trial on {benchmark.__class__.__name__}...")

    # 2. Run the Trial
    pbo_trial(
        problem="Hartmann6_Test",       # Name of the folder where results will save
        obj_func=obj_func_wrapper,      # Our PyTorch->NumPy wrapper
        input_dim=input_dim,
        noise_type="noiseless",         # Keep it noiseless for the first real test
        noise_level=0.0,
        algo="qeubo",                   # Use your qEUBO acquisition function
        num_alternatives=2,             # Pairwise duels (q=2)
        num_init_queries=20,            # How many random duels to start with
        num_algo_queries=50,            # How many BO steps to run
        trial=1,                        # Trial ID
        restart=False,                  # Start fresh
        model_type="variational_preferential_gp", # Triggers your custom model in utils.py
        add_baseline_point=False,
        ignore_failures=False,
        algo_params=None
    )
    
    print("Experiment Complete! Check the /experiments/results/ folder.")

if __name__ == "__main__":
    run()