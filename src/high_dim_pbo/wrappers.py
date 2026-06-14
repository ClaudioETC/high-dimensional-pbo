import torch

class BenchmarkWrapper:
    def __init__(self, benchmark_name, **kwargs):
        """
        Dynamically instantiates and wraps different benchmarks.
        """
        self.name = benchmark_name
        self.dim = None
        self.benchmark = None
        self.bounds = None
        
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
            
        elif benchmark_name == "Hartmann6":
            from test_funcs import Hartmann6
            self.benchmark = Hartmann6()
            self.dim = 6
            self.display_name = "Hartmann6"
            
        elif benchmark_name == "Cola":
            from test_funcs import Cola
            self.benchmark = Cola()
            self.dim = self.benchmark.dim
            self.display_name = "Cola"
            # Normalization Setup
            self.benchmark.init_normalize_X()
            self.benchmark.init_normalize_Y()
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

        elif self.name in ["Hartmann6", "Cola"]:
            X_np = X_tensor.detach().numpy()
            # Negate because BoTorch maximizes, test_funcs minimizes
            y_np = -self.benchmark.f(X_np) 
            return torch.tensor(y_np, dtype=torch.float64)