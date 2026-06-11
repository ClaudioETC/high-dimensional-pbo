import torch
import numpy as np
import wandb
from high_dim_pbo.qeubo_utils import generate_random_queries
from test_funcs import Hartmann6, Cola #import benchmark of interest
#from src.high_dim_pbo.benchmarks.rover import Rover
from rover import Rover
from qeubo_utils import (
    generate_initial_data,
    fit_model,
    get_obj_vals,
    generate_responses,
    generate_random_queries,
    optimize_acqf_and_get_suggested_query,
    classify
)
from vanilla_qeubo import qExpectedUtilityOfBestOption
from botorch.sampling import SobolQMCNormalSampler
import numpy as np
import scipy
# Monkey-patch histogram back into scipy to fix guacamol compatibility
scipy.histogram = np.histogram
from guacamol_task import GuacamolObjective

#objective Function/Benchmark
torch.manual_seed(42)
benchmark = GuacamolObjective(guacamol_task_id="adip")#Rover(0.0) #instantiate the benchmark class of interest; noise is 

#only for Cola benchmark
# benchmark.init_normalize_X()  
# benchmark.init_normalize_Y()

#problem = benchmark.__class__.__name__
problem = f"Guacamol_{benchmark.__class__.__name__}_adip"
# data_dims = benchmark.dim
def run_clean_bo_experiment():
    wandb.init(
        project="high-dim-pbo-qeubo",
        name="qEUBO_Hartmann6_Trial_4",
        config={
            "problem": problem,
            "input_dim": benchmark.dim,
            "num_init_queries": 20, #consider N<D, N>D and N~~D
            "num_algo_queries": 150,
            "num_alternatives": 2,               
            "acquisition_function": "qEUBO",
            "model_type": "variational_preferential_gp",
            "noise": "noiseless", #check qeubo_utils.py; specifically corrupt_obj_vals()
            "use_whitening": False,       # True = Inducing-Point VI, False = Vanilla VI
            "model_optimizer": "L-BFGS-B", #or others like "Adam"  
            "adam_lr": 0.05,             
            "adam_epochs": 150,          
            "mc_samples": 64,            # MC Samples for qEUBO
            "acqf_restarts": 12,         # L-BFGS-B (optimizer in general) restarts
            "acqf_raw_samples": 360,      # Raw samples for L-BFGS-B init
            "num_test_points": 100,
        }
    )
    
    cfg = wandb.config
    # benchmark = Hartmann6(dim=cfg.input_dim)
    ######################################################
    # Benchmarks are implemented in Numpy arrays, we're on torch; need to "connect them"
    def obj_func_wrapper(X_tensor):
        X_np = X_tensor.detach().numpy()
        y_np = -benchmark.f(X_np) # Negate because BoTorch maximizes, test_funcs minimizes
        return torch.tensor(y_np, dtype=torch.float32)
    
    def rover_obj_func_wrapper(X_tensor):
        with torch.no_grad():
        # Returns a 1D tensor of shape (batch_size,)
            y_tensor = benchmark(X_tensor) 
        
        return y_tensor.to(torch.float32)
    
    def guacamol_obj_func_wrapper(X_tensor):
        """
        Bridges the normalized BO search space [0, 1] with the 
        unnormalized VAE latent space [-8, 8].
        """
        #Scale from [0, 1] bounds to [-8, 8] bounds
        # (X * range) + minimum
        X_scaled = (X_tensor * 16.0) - 8.0
    
        # No negation is applied because Guacamol tasks are maximized natively
        with torch.no_grad():
            y_tensor = benchmark(X_scaled) 
        
        return y_tensor.to(torch.float32)
    
    ######################################################
    print(f"Starting High-Dimensional qEUBO Trial on {problem}")

    # Generate Initial Random Preference Data
    queries, obj_vals, responses = generate_initial_data(
        num_queries=cfg.num_init_queries,
        num_alternatives=cfg.num_alternatives,
        input_dim=cfg.input_dim,
        obj_func=guacamol_obj_func_wrapper,
        noise_type=cfg.noise,
        noise_level=0.0,
        add_baseline_point=False,
        seed=42
    )

    # Log the best initial utility found by random guessing
    best_initial_utility = obj_vals.max().item()
    wandb.log({"iteration": 0, "max_utility_found": best_initial_utility})
    print(f"Best initial utility from random queries: {best_initial_utility:.4f}")

    #Bayesian Optimization Loop
    for iteration in range(cfg.num_algo_queries):
        current_step = iteration + 1
        print(f"\n--- Iteration {current_step}/{cfg.num_algo_queries} ---")
        
        #Fit the Model
        model = fit_model(
            queries, 
            responses, 
            model_type=cfg.model_type, 
            model_optimizer=cfg.model_optimizer,
            use_whitening=cfg.use_whitening,
            adam_lr=cfg.adam_lr,
            adam_epochs=cfg.adam_epochs
        )
        
        #Setup the Acquisition Function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([cfg.mc_samples]))
        acq_func = qExpectedUtilityOfBestOption(model=model, sampler=sampler)
        
        # Optimize the Acquisition Function 
        standard_bounds = torch.tensor([[0.0] * cfg.input_dim, [1.0] * cfg.input_dim])
        
        new_query = optimize_acqf_and_get_suggested_query(
            acq_func=acq_func,
            bounds=standard_bounds,
            batch_size=cfg.num_alternatives,
            num_restarts=cfg.acqf_restarts,       
            raw_samples=cfg.acqf_raw_samples,    
        ).unsqueeze(0)      
        
        # Query the True Function and simulate the user's preference choice
        new_obj_vals = get_obj_vals(new_query, guacamol_obj_func_wrapper)
        new_response = generate_responses(new_obj_vals, noise_type=cfg.noise, noise_level=0.0)

        #Update the dataset
        queries = torch.cat((queries, new_query))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)
        responses = torch.cat((responses, new_response))

        #Log to Weights & Biases
        current_max_utility = obj_vals.max().item()
        print(f"Max Utility Found So Far: {current_max_utility:.4f}")
        
        wandb.log({
            "iteration": current_step,
            "max_utility_found": current_max_utility,
            "total_queries_asked": cfg.num_init_queries + current_step
        })
    
    #classification
    accuracy, nll = classify(
        model=model,
        num_queries=cfg.num_test_points,
        alts_per_query=cfg.num_alternatives,
        alts_dim=cfg.input_dim,
        obj_func=guacamol_obj_func_wrapper,
    )

    print(f"Final Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Negative Log-Likelihood: {nll:.4f}")
    
    wandb.log({
        "test_accuracy": accuracy, 
        "test_nll": nll
    })

    # Close the run gracefully
    wandb.finish()
    print("Experiment Complete! Check your Weights & Biases dashboard.")

if __name__ == "__main__":
    run_clean_bo_experiment()