import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.variational_elbo import VariationalELBO

# Import your newly merged class
from var_inf_approx import VariationalPreferentialGP

def run_posterior_sanity_check():
    torch.manual_seed(42)
    
    # ---------------------------------------------------------
    # Create a "Dummy" Linear Dataset
    # True latent utility: f(x) = 2*x1 - 1*x2 
    # (Dimension 1 is good because of its positive weight (2), Dimension 2 is bad because of its weight (-1))
    # this makes dim 1 increase the total score while dim 2 decreases it 
    # ---------------------------------------------------------
    num_queries = 150
    q_batch_size = 2 # Pairwise duels
    input_dim = 2 #x1 and x2
    
    # Random queries in the [0, 1] bounded square
    queries = torch.rand(num_queries, q_batch_size, input_dim)
    
    # Calculate true utility (the hidden function we want to learn)
    true_weights = torch.tensor([2.0, -1.0])
    true_utilities = torch.matmul(queries, true_weights)
    
    # Assign the "winner" of each pair based on which had higher utility
    # responses shape: (50, 1) containing 0 or 1
    responses = torch.argmax(true_utilities, dim=-1, keepdim=True)
    
    print(f"Generated {num_queries} preference pairs.")

    # ---------------------------------------------------------
    # 2. Train the Variational Preferential GP
    # ---------------------------------------------------------
    # Initialize your updated model
    model = VariationalPreferentialGP(
        queries=queries, 
        responses=responses, 
        use_withening=True 
    )
    model = model.to(torch.float64)
    
    # Define the ELBO objective
    mll = VariationalELBO(
        likelihood=model.likelihood,
        model=model,
        num_data=queries.shape[0] * queries.shape[1]
    )
    mll = mll.to(torch.float64)
    
    print("Fitting the Variational GP...")
    model.train()
    model.likelihood.train()

    #######
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.05)
    
    epochs = 150
    print("Starting custom training loop...")
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        # model.train_inputs[0] is your flattened (100, 2) tensor of queries
        output = model(model.train_inputs[0]) 
        
        # Calculate ELBO loss
        # We multiply by -1 because optimizers minimize, but we want to maximize ELBO
        loss = -mll(output, model.train_targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print out the loss to monitor convergence
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Epoch {i+1:3d}/{epochs} - Loss (Negative ELBO): {loss.item():.4f}")
            
    print("Model successfully fitted!")
    #######
    
    # # This automatically optimizes the ELBO using PyTorch L-BFGS-B
    # fit_gpytorch_mll(mll) 
    # print("Model successfully fitted! No gradient errors.")

    # ---------------------------------------------------------
    # 3. Sample from the Posterior
    # ---------------------------------------------------------
    model.eval()
    model.likelihood.eval()
    
    # Create 3 test points we want to evaluate
    # Point A: [1.0, 0.0] -> Should be very high utility (good x1, low x2)
    # Point B: [0.5, 0.5] -> Should be medium utility
    # Point C: [0.0, 1.0] -> Should be very low utility (low x1, high x2)
    test_X = torch.tensor([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0]
    ])
    
    with torch.no_grad():
        # Get the posterior distribution
        posterior_dist = model(test_X)
        
        # The Mean is the model's absolute best guess
        mean_utilities = posterior_dist.mean
        print("\n--- Posterior Mean (The True Learned Ranking) ---")
        print("Expected ranking: A > B > C")
        print(f"Mean utilities   | A: {mean_utilities[0]:.3f} | B: {mean_utilities[1]:.3f} | C: {mean_utilities[2]:.3f}")
        
        # The samples show how certain the model is about that mean
        samples = posterior_dist.sample(sample_shape=torch.Size([20]))
        print("\n--- Posterior Samples (Uncertainty Check) ---")
        for i in range(5):
            print(f"Sample {i+1} utilities | A: {samples[i][0]:.3f} | B: {samples[i][1]:.3f} | C: {samples[i][2]:.3f}")
if __name__ == "__main__":
    run_posterior_sanity_check()