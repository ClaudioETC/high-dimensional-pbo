import argparse
#from guacamol_task import benchmark_suites
import torch
import numpy as np
import math
import wandb
from botorch.acquisition.analytic import PosteriorMean
from vanilla_qeubo import qExpectedUtilityOfBestOption
from botorch.sampling import SobolQMCNormalSampler
import scipy

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default = 42)
args = parser.parse_args()

#setting seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

from wrappers import BenchmarkWrapper

from qeubo_utils import (
    generate_initial_data,
    fit_model,
    get_obj_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
    classify,
    post_mean_max
)

scipy.histogram = np.histogram
torch.manual_seed(42)

# Change this to any wrapper from wrappers.py
BENCHMARK_SELECTION = "Alpine1" 

# Instantiate the dynamic wrapper
obj_wrapper = BenchmarkWrapper(BENCHMARK_SELECTION, guacamol_task_id="adip")

def run_clean_bo_experiment():
    wandb.init(
        entity="claudiotorrescantu-danmarks-tekniske-universitet-dtu",
        project="high-dim-pbo-qeubo",
        group=f"{obj_wrapper.display_name}_qEUBO_RBF",
        name=f"{obj_wrapper.display_name}_qEUBO_RBF;trial_seed_{args.seed}",
        config={
            "problem": obj_wrapper.display_name,
            "input_dim": obj_wrapper.dim,
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
            "mc_samples": 64,          # MC Samples for qEUBO   
            "acqf_restarts": obj_wrapper.dim * 2,     #again, hard-coded because of hartmann 6   # L-BFGS-B (optimizer in general) restarts
            "acqf_raw_samples": 30 * obj_wrapper.dim * 2,#num_alternatives last #was 360 because of Hartmann6 hardcoded # Raw samples for L-BFGS-B init
            "num_test_points": 1000,
            "kernel": "RBF" , #alternatives are spherical or other of choice
            "seed": args.seed,
            "log10scale": True
                }
    )
    
    cfg = wandb.config
    print(f"Starting High-Dimensional qEUBO Trial on {obj_wrapper.display_name}")

    # Generate Initial Random Preference Data using the wrapper
    queries, obj_vals, responses = generate_initial_data(
        num_queries=cfg.num_init_queries,
        num_alternatives=cfg.num_alternatives,
        input_dim=cfg.input_dim,
        obj_func=obj_wrapper,  # <
        noise_type=cfg.noise,
        noise_level=0.0,
        add_baseline_point=False,
        seed=42
    )

    #best initial utility found by random guessing
    best_initial_utility = obj_vals.max().item()
    wandb.log({"iteration": 0, "max_utility_found": best_initial_utility})
    print(f"Best initial utility from random queries: {best_initial_utility:.4f}")

    # Bayesian Optimization Loop
    for iteration in range(cfg.num_algo_queries):
        current_step = iteration + 1
        print(f"\n--- Iteration {current_step}/{cfg.num_algo_queries} ---")
        
        # Fit the Model
        model, end_elbo = fit_model(
            queries, 
            responses, 
            model_type=cfg.model_type, 
            model_optimizer=cfg.model_optimizer,
            use_whitening=cfg.use_whitening,
            adam_lr=cfg.adam_lr,
            adam_epochs=cfg.adam_epochs,
            kernel = cfg.kernel
        )

        # learned RBF hyperparameters
        if cfg.kernel == "RBF":
            # Extract the lengthscales (shape: [1, input_dim]) and average them, 
            # or log them individually 
            mean_lengthscale = model.covar_module.base_kernel.lengthscale.mean().item()
            outputscale = model.covar_module.outputscale.item()
            
            wandb.log({
                "mean_lengthscale": mean_lengthscale,
                "outputscale": outputscale,
            }, commit=False) # commit=False so it logs on the same step as your other metrics
        
        # Find Maximum of posterior mean using the wrapper
        posterior_max = post_mean_max(
            model,
            obj_func=obj_wrapper, # <
            bounds=obj_wrapper.bounds, # uniform [0,1] bounds
            num_points=1,
            num_restarts=6 * obj_wrapper.dim,
            raw_samples=180 * obj_wrapper.dim,
        )
        regret = obj_wrapper.max - posterior_max
        if cfg.log10scale == True:
            regret = math.log10(regret)
    
        # Setup the Acquisition Function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([cfg.mc_samples]))
        acq_func = qExpectedUtilityOfBestOption(model=model, sampler=sampler)
        
        # Optimize the Acquisition Function 
        new_query = optimize_acqf_and_get_suggested_query(
            acq_func=acq_func,
            bounds=obj_wrapper.bounds, # <
            batch_size=cfg.num_alternatives,
            num_restarts=cfg.acqf_restarts,       
            raw_samples=cfg.acqf_raw_samples,    
        ).unsqueeze(0)      
        
        # Query the True Function and simulate choice using the wrapper
        new_obj_vals = get_obj_vals(new_query, obj_wrapper) # <
        new_response = generate_responses(new_obj_vals, noise_type=cfg.noise, noise_level=0.0)

        # Update the dataset
        queries = torch.cat((queries, new_query))
        obj_vals = torch.cat([obj_vals, new_obj_vals], 0)
        responses = torch.cat((responses, new_response))

        # Log to Weights & Biases
        current_max_utility = obj_vals.max().item()
        print(f"Max Utility Found So Far: {current_max_utility:.4f}")
        print(f"True Utility at GP's Max Mean: {posterior_max:.4f}")
        print(f"Log Regret of Mean Posterior argmax: {regret:.4f}")

        wandb.log({
            "iteration": current_step,
            "max_utility_found in queries": current_max_utility,
            "true_utility_at_post_mean": posterior_max,
            "model_final_elbo": end_elbo,
            "total_queries_asked": cfg.num_init_queries + current_step

        })
    
    # Classification
    accuracy, nll = classify(
        model=model,
        num_queries=cfg.num_test_points,
        alts_per_query=cfg.num_alternatives,
        alts_dim=cfg.input_dim,
        obj_func=obj_wrapper, # <
    )

    print(f"Final Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Negative Log-Likelihood: {nll:.4f}")
    
    if cfg.log10scale == True:
        wandb.log({
        "test_accuracy": accuracy, 
        "test_nll": nll,
        "Log-Regret": regret
        })
    else:
        wandb.log({
        "test_accuracy": accuracy, 
        "test_nll": nll,
        "Regret": regret
        })

    wandb.finish()
    print("Experiment Complete")

if __name__ == "__main__":
    run_clean_bo_experiment()