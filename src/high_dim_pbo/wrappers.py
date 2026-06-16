import torch
from botorch.test_functions.synthetic import Hartmann

class BenchmarkWrapper:
    def __init__(self, benchmark_name, **kwargs):
        """
        Dynamically instantiates and wraps different benchmarks.
        """
        self.name = benchmark_name
        self.dim = None
        self.benchmark = None
        self.bounds = None
        self.max = None
        
        #Dynamically Instantiate the underlying Benchmark
        if "Guacamol" in benchmark_name:
            from guacamol_task import GuacamolObjective
            task_id = kwargs.get("guacamol_task_id", "adip")
            self.benchmark = GuacamolObjective(guacamol_task_id=task_id)
            self.dim = self.benchmark.dim
            self.display_name = f"Guacamol_{self.benchmark.__class__.__name__}_{task_id}"
            
            
        elif benchmark_name == "Rover":
            from rover import Rover
            self.benchmark = Rover(0.0)
            self.dim = self.benchmark.dim 
            self.display_name = "Rover"
            self.max = 5.0

        elif benchmark_name == "Hartmann":
            from botorch.test_functions.synthetic import Hartmann
            self.benchmark = Hartmann()
            self.dim = 6
            self.display_name = "Hartmann"
            self.max = 3.32237

        elif benchmark_name == "Cola":
            from test_funcs import Cola
            self.benchmark = Cola()
            self.dim = self.benchmark.dim
            self.display_name = "Cola"
            # Normalization Setup
            self.benchmark.init_normalize_X()
            self.benchmark.init_normalize_Y()
            self.max = 1607.73849331

        elif benchmark_name == "Alpine1":
            self.dim = 7
            self.benchmark = "analytic_internal" 
            self.display_name = "Alpine1"
            self.max = 0
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
        # Define the unified [0, 1] bounds for BoTorch optimization
        self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.float64)

    def __call__(self, X_tensor):
        """
        The universal objective wrapper function. 
        Accepts normalized X in [0, 1] and returns a 1D Torch Tensor.
        """
        if "Guacamol" in self.name:
            # Scale from [0, 1] to VAE latent space [-8, 8]
            X_scaled = (X_tensor * 16.0) - 8.0
            with torch.no_grad():
                y_tensor = self.benchmark(X_scaled)
            return y_tensor.to(torch.float64)

        elif self.name == "Rover":
            with torch.no_grad():
                y_tensor = self.benchmark(X_tensor)
            return y_tensor.to(torch.float64)

        elif self.name == "Cola":
            X_np = X_tensor.detach().numpy()
            # Negate because BoTorch maximizes, test_funcs minimizes
            y_np = -self.benchmark.f(X_np) #we use f() because Cola inherits from TestFunction(); revise test_funcs.py
            return torch.tensor(y_np, dtype=torch.float64)
        
        elif self.name == "Hartmann":
            hartmann = Hartmann()
            objective_X = -hartmann.evaluate_true(X_tensor)
            return torch.tensor(objective_X, dtype=torch.float64)

        elif self.name == "Alpine1":
            # Scale X from [0, 1] to [-10.0, 10.0]
            X_unnorm = 20.0 * X_tensor - 10.0
            
            with torch.no_grad():
                # Apply the analytic objective function
                y_tensor = -torch.abs(X_unnorm * torch.sin(X_unnorm) + 0.1 * X_unnorm).sum(dim=-1)
                
            return y_tensor.to(torch.float64)