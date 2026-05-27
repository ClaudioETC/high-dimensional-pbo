from torch import nn, distributions
import gpytorch
import pymc as pm
from scipy.stats import multinomial
import jax.numpy as jnp
import jax.scipy as jsp
# class Model(nn.Module):
#     """Just a dummy model to show how to structure your code"""
#     def __init__(self):
#         super().__init__()
#         self.layer = nn.Linear(1, 1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layer(x)

# if __name__ == "__main__":
#     model = Model()
#     x = torch.rand(1)
#     print(f"Output shape of model: {model(x).shape}")