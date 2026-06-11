#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import math
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation.gen import get_best_candidates
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from gpytorch.mlls.variational_elbo import VariationalELBO
from scipy.optimize import minimize
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel

#from src.models.variational_preferential_gp import VariationalPreferentialGP
from var_inf_approx import VariationalPreferentialGP

# In qeubo_utils.py
def fit_model(
    queries: Tensor,
    responses: Tensor,
    model_type: str,
    model_optimizer: str = "Adam",
    use_whitening: bool = True,
    adam_lr: float = 0.05,       
    adam_epochs: int = 150       
):
    if model_type == "variational_preferential_gp":
        # Pass parameters to GP
        model = VariationalPreferentialGP(queries, responses, use_withening=use_whitening)
        model = model.to(torch.float64)
        model.train()
        model.likelihood.train()
        
        mll = VariationalELBO(
            likelihood=model.likelihood,
            model=model,
            num_data=2 * model.num_data,
        )
        
        if model_optimizer == "Adam":
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': model.likelihood.parameters()},
            ], lr=adam_lr)
            
            for _ in range(adam_epochs):
                optimizer.zero_grad()
                output = model(model.train_inputs[0])
                loss = -mll(output, model.train_targets)
                loss.backward()
                optimizer.step()
                
        elif model_optimizer == "L-BFGS-B":
            mll = fit_gpytorch_mll(mll)

            # Compute the final ELBO after convergence
            model.eval()
            model.likelihood.eval()
            with torch.no_grad():
                output = model(model.train_inputs[0])
                # No negative sign needed here, mll returns the true ELBO
                final_elbo = mll(output, model.train_targets).item() 
                print(final_elbo)
                #return model, final_elbo
            
        model.eval()
        model.likelihood.eval()
        
    return model

def generate_initial_data(
    num_queries: int,
    num_alternatives: int,
    input_dim: int,
    obj_func,
    noise_type,
    noise_level,
    add_baseline_point: bool,
    seed: int = None,
):
    queries = generate_random_queries(num_queries, num_alternatives, input_dim, seed)
    if add_baseline_point:  # If true, this adds 30 queries including a
        # "high-quality baseline point". The baseline point is hardcoded in generate_queries_against_baseline
        queries_against_baseline = generate_queries_against_baseline(
            30, num_alternatives, input_dim, obj_func, seed
        )
        queries = torch.cat([queries, queries_against_baseline], dim=0)
    obj_vals = get_obj_vals(queries, obj_func)
    responses = generate_responses(obj_vals, noise_type, noise_level)
    return queries, obj_vals, responses


def generate_random_queries(
    num_queries: int, num_alternatives: int, input_dim: int, seed: int = None
):
    # Generate `num_queries` queries each constituted by `num_alternatives` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, num_alternatives, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, num_alternatives, input_dim])
    return queries


def generate_queries_against_baseline(
    num_queries: int, num_alternatives: int, input_dim: int, obj_func, seed: int = None
):
    baseline_point = torch.tensor([0.51] * input_dim)  # This baseline point was meant
    # to be used with the Alpine1 function (with normalized input space) exclusively
    queries = generate_random_queries(
        num_queries, num_alternatives - 1, input_dim, seed + 2
    )
    queries = torch.cat([baseline_point.expand_as(queries), queries], dim=1)
    return queries


def get_obj_vals(queries, obj_func):
    queries_2d = queries.reshape(
        torch.Size([queries.shape[0] * queries.shape[1], queries.shape[2]])
    )
    obj_vals = obj_func(queries_2d)
    obj_vals = obj_vals.reshape(torch.Size([queries.shape[0], queries.shape[1]]))
    return obj_vals


def generate_responses(obj_vals, noise_type, noise_level):
    # Generate simulated comparisons based on true underlying objective
    corrupted_obj_vals = corrupt_obj_vals(obj_vals, noise_type, noise_level)
    responses = torch.argmax(corrupted_obj_vals, dim=-1)
    return responses


def corrupt_obj_vals(obj_vals, noise_type, noise_level):
    # Noise in the decision-maker's responses is simulated by corrupting the objective values
    if noise_type == "noiseless":
        corrupted_obj_vals = obj_vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = normal.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=obj_vals.shape)
        corrupted_obj_vals = obj_vals + noise
    elif noise_type == "constant":
        corrupted_obj_vals = obj_vals.clone()
        n = obj_vals.shape[0]
        for i in range(n):
            coin_toss = Bernoulli(noise_level).sample().item()
            if coin_toss == 1.0:
                corrupted_obj_vals[i, 0] = obj_vals[i, 1]
                corrupted_obj_vals[i, 1] = obj_vals[i, 0]
    return corrupted_obj_vals


def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 2,
    init_batch_limit: Optional[int] = 30,
) -> Tensor:
    """Optimizes the acquisition function and returns the (approximate) optimum."""

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": batch_limit,
            "init_batch_limit": init_batch_limit,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )
    candidates = candidates.detach()
    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x


def get_noise_level(
    obj_func, input_dim, noise_type, target_error, top_proportion, num_samples
):
    X = torch.rand([num_samples, input_dim])
    Y = obj_func(X)
    target_Y = Y.sort().values[-int(num_samples * top_proportion) :]
    target_Y = target_Y[torch.randperm(target_Y.shape[0])]
    target_Y = target_Y.reshape(-1, 2)

    # estimate probit error
    true_comps = target_Y[:, 0] > target_Y[:, 1]

    res = minimize(
        error_rate_loss,
        x0=0.1,
        args=(target_Y, true_comps, target_error, noise_type),
    )
    print(res)

    noise_level = res.x[0]

    error_rate = estimate_error_rate(noise_level, target_Y, true_comps, noise_type)
    print(error_rate)
    return noise_level


def estimate_error_rate(noise_scale, obj_vals, true_comps, noise_type):
    if noise_type == "probit":
        std_norm = torch.distributions.normal.Normal(0, 1)
        prob0 = std_norm.cdf(
            (obj_vals[:, 0] - obj_vals[:, 1]) / (math.sqrt(2) * noise_scale)
        )
        prob1 = 1 - prob0
    elif noise_type == "logit":
        soft_max = torch.nn.Softmax(dim=-1)
        probs = soft_max(obj_vals / noise_scale)
        prob0 = probs[:, 0]
        prob1 = probs[:, 1]
    correct_prob = torch.cat((prob0[true_comps], prob1[~true_comps]))
    error_rate = 1 - correct_prob.mean()
    return error_rate.item()

def error_rate_loss(x, obj_vals, true_comps, target_error, noise_type):
    return abs(estimate_error_rate(x, obj_vals, true_comps, noise_type) - target_error)

def classify(model, num_queries, alts_per_query, alts_dim, obj_func):
    """_summary_

    Args:
        model (Class): _description_
        num_queries (int): _description_
        alts_per_query (int): _description_
        alts_dim (int): dimension 
        benchmark (Class): Class encapsulating benchmark of choice

    Returns:
        _type_: _description_
    """
    model.eval()
    model.likelihood.eval()
    test_queries = generate_random_queries(num_queries, alts_per_query, alts_dim)
    # test_points = test_points.view(num_queries * alts_per_query, alts_dim)
    obj_vals = get_obj_vals(test_queries, obj_func)
    true_labels = torch.argmax(obj_vals, dim=-1)

    # 3. Get Model Predictions
    # Flatten to 2D for the model forward pass
    test_points_2d = test_queries.view(num_queries * alts_per_query, alts_dim)
    
    with torch.no_grad():
        # Get the latent GP predictions (returns a MultivariateNormal)
        f_preds = model(test_points_2d)
        
        # Pass through the softmax likelihood to get probabilities
        # This draws MC samples and returns a Categorical distribution
        pred_dist = model.likelihood(f_preds)
        
        # Average the probabilities across the MC sample dimension (dim=0)
        pred_probs = pred_dist.probs.mean(dim=0)

    # 4. Calculate Accuracy and NLL
    predictions = torch.argmax(pred_probs, dim=-1)
    correct = (predictions == true_labels).sum().item()
    accuracy = correct / num_queries

    # NLL (Cross-Entropy) penalizes the model for being confidently wrong
    nll = torch.nn.functional.cross_entropy(pred_probs, true_labels).item()

    return accuracy, nll
