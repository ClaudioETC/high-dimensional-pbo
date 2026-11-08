#!/usr/bin/env python3
"""
Information-Gain and Posterior Halving Acquisition Functions for Preference-Based BO
with Spherical Linear Models.

Reference:
    "Preference-Based Bayesian Optimization with Spherical Linear Models"
    - Section 4.2: Imbalance Delta = |P(theta^T (z1 - z2) >= 0) - 1/2| & Entropy H(1/2 + Delta)
    - Section 5: Practical Mutual Information approximation under probit noise:
                 A(q) = -m^2 / (v + tau^{-2}) + lambda * log(1 + tau^2 * v)
"""

from __future__ import annotations

import math

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class Info_Gain_AF(AcquisitionFunction):
    r"""Information-Gain / Posterior-Halving Acquisition Function for pairwise
    preference-based Bayesian Optimization.

    Supports three complementary formulations from the paper:
    1. ``"entropy"`` (Section 4.2):
       Evaluates the binary entropy of the predicted pairwise comparison:
       .. math::
           p = \Phi\left(\frac{m}{\sqrt{v + \tau^{-2}}}\right)
           H(p) = -p \log_2(p) - (1-p) \log_2(1-p)
       Maximizing H(p) directly achieves posterior halving (bisection) where p = 0.5.

    2. ``"imbalance"`` (Section 4.2):
       Penalizes deviation from balanced bisection:
       .. math::
           \Delta = \left| p - \frac{1}{2} \right|, \quad A(q) = -\left(p - \frac{1}{2}\right)^2

    3. ``"practical_mi"`` (Section 5):
       Practical one-step mutual information approximation balancing posterior halving with SNR:
       .. math::
           A(q) = -\frac{m^2}{v + \tau^{-2}} + \lambda \ln(1 + \tau^2 v)

    Args:
        model: A fitted GP model whose forward call returns a distribution with
               `mean` and `covariance_matrix` attributes (e.g. `VariationalPreferentialGP`).
        mode: Acquisition mode, one of ``"entropy"`` (default), ``"imbalance"``,
              or ``"practical_mi"``.
        lambda_snr: Weight $\lambda \ge 0$ for the variance / SNR regularizer in
                    ``"practical_mi"`` mode (default: 1.0).
        eps: Small constant for numerical stability (default: 1e-7).
    """

    def __init__(
        self,
        model: Model,
        mode: str = "entropy",
        lambda_snr: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__(model=model)
        valid_modes = {"entropy", "imbalance", "practical_mi"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
        if lambda_snr < 0:
            raise ValueError(f"lambda_snr must be non-negative, got {lambda_snr}")

        self.mode = mode
        self.eps = eps
        self.register_buffer("lambda_snr", torch.tensor(lambda_snr, dtype=torch.float64))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the acquisition function on a batch of query pairs.

        Args:
            X: A `batch_shape x 2 x d` tensor where each entry along the
               q-dimension contains a pairwise candidate `[z1, z2]`.

        Returns:
            A `batch_shape`-dim tensor of acquisition values.
        """
        if X.shape[-2] != 2:
            raise ValueError(
                f"X must have exactly 2 alternatives per query (got {X.shape[-2]}). "
                "Info_Gain_AF is defined for pairwise queries."
            )

        # Obtain the joint GP posterior over the two query points.
        # Direct call to model(X) yields full 2x2 covariance matrix.
        posterior = self.model(X)
        mu = posterior.mean                  # shape: (..., 2)
        Sigma = posterior.covariance_matrix  # shape: (..., 2, 2)

        # Projected statistics for difference f(z1) - f(z2):
        # m = mu[0] - mu[1]
        # v = Var[f(z1) - f(z2)] = Sigma[0,0] + Sigma[1,1] - Sigma[0,1] - Sigma[1,0]
        m = mu[..., 0] - mu[..., 1]
        v = (
            Sigma[..., 0, 0]
            + Sigma[..., 1, 1]
            - Sigma[..., 0, 1]
            - Sigma[..., 1, 0]
        )
        v = torch.clamp(v, min=1e-12)

        if self.mode == "practical_mi":
            lambda_snr = self.lambda_snr.to(dtype=X.dtype, device=X.device)
            balance_term = -(m ** 2) / (v + 1)
            snr_term = lambda_snr * torch.log1p(v)
            return balance_term + snr_term

        # For "entropy" and "imbalance" modes:
        # Effective standard deviation of the difference under probit observation model
        eff_sigma = torch.sqrt(v + 1)
        z = m / eff_sigma
        # Normal CDF using erf for stable gradients: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        p = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)

        if self.mode == "entropy":
            # Binary entropy in bits: H(p) = -p log2(p) - (1-p) log2(1-p)
            entropy = -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)
            return entropy

        elif self.mode == "imbalance":
            # Negative squared imbalance: - (p - 0.5)^2 (maximized at p = 0.5)
            return -((p - 0.5) ** 2)


# Alias for compatibility
InformationGainAcquisitionFunction = Info_Gain_AF