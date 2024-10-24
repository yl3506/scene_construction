import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from models import build_tree, \
            probabilistic_bfs, probabilistic_dfs, \
            deterministic_bfs, deterministic_dfs
from evaluation import compute_log_likelihood, compute_aic
from visualize import *


def fit_deterministic_model(participant_data, model_type):
    participant_id = participant_data['ParticipantID']
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor'] 
    scene = participant_data['Scene']
    properties = participant_data['Properties']
    responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
    ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}

    # Define the Pyro model inside this function
    def pyro_model():
        # Sample cognitive_resource from its prior
        cognitive_resource = pyro.sample('cognitive_resource', dist.Uniform(1.0, 30.0))
        
        # Compute effective time limit
        effective_time_limit = base_time_limit * scaling_factor * cognitive_resource
        
        # Build the tree
        root = build_tree(scene)
        
        # Perform deterministic traversal
        if model_type == 'BFS':
            visited = deterministic_bfs(root, effective_time_limit)
        elif model_type == 'DFS':
            visited, _ = deterministic_dfs(root, effective_time_limit)
        else:
            raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")
        
        # Compute the log-likelihood
        log_likelihood = compute_log_likelihood(visited, responses, ranks)
        
        # Use pyro.factor to include the likelihood in the model
        pyro.factor('likelihood', log_likelihood)
    
    # Set up the NUTS kernel
    nuts_kernel = NUTS(pyro_model)
    
    # Run MCMC
    mcmc = MCMC(nuts_kernel, num_samples=50, warmup_steps=10, num_chains=1)
    mcmc.run()
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    plot_trace(posterior_samples, 'cognitive_resource', participant_id, f'Deterministic {model_type}')
    plot_autocorrelation(posterior_samples, 'cognitive_resource', participant_id, f'Deterministic {model_type}')
    plot_posterior_density(posterior_samples, 'cognitive_resource', participant_id, f'Deterministic {model_type}')
    plot_posterior_kde(posterior_samples, 'cognitive_resource', participant_id, f'Deterministic {model_type}')
    
    # Compute summary statistics
    cognitive_resource_samples = posterior_samples['cognitive_resource'].detach().numpy()
    cognitive_resource_mean = np.mean(cognitive_resource_samples)
    cognitive_resource_ci = np.percentile(posterior_samples['cognitive_resource'], [2.5, 97.5])  # 95% credible interval
    
    # Compute log-likelihood at the mean estimate
    effective_time_limit = base_time_limit * scaling_factor * cognitive_resource_mean
    
    # Build the tree
    root = build_tree(scene)
    # Perform deterministic traversal with mean cognitive_resource
    if model_type == 'BFS':
        visited = deterministic_bfs(root, effective_time_limit)
    elif model_type == 'DFS':
        visited, _ = deterministic_dfs(root, effective_time_limit)
    else:
        raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")
    
    # Compute log-likelihood
    log_likelihood = compute_log_likelihood(visited, responses, ranks).item()
    
    # Compute AIC
    k = 1  # One parameter
    aic = compute_aic(log_likelihood, k)
    
    return {
        'ExpansionProb': None,
        'ExpansionProbCI': None,
        'CognitiveResource': cognitive_resource_mean,
        'CognitiveResourceCI': cognitive_resource_ci,
        'log_likelihood': log_likelihood,
        'AIC': aic,
        'posterior_samples': posterior_samples
    }



def fit_probabilistic_model(participant_data, model_type):
    participant_id = participant_data['ParticipantID']
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']
    properties = participant_data['Properties']
    responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
    ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}
    scene = participant_data['Scene']

    # Define the Pyro model inside this function
    def pyro_model():
        # Sample parameters from their priors
        expansion_prob = pyro.sample('expansion_prob', dist.Beta(2, 2))
        cognitive_resource = pyro.sample('cognitive_resource', dist.Uniform(1.0, 30.0))
        
        # Compute effective time limit
        effective_time_limit = base_time_limit * scaling_factor * cognitive_resource
        
        # Build the tree
        root = build_tree(scene)
        
        # Number of simulation runs to average over
        num_runs = 10  # Adjust as needed for convergence
        
        # Initialize variables to store log-likelihoods
        total_log_likelihood = torch.tensor(0.0)
        
        # Perform simulations and compute likelihood
        for _ in range(num_runs):
            # Perform probabilistic traversal
            if model_type == 'BFS':
                visited = probabilistic_bfs(root, effective_time_limit, expansion_prob)
            elif model_type == 'DFS':
                visited, _ = probabilistic_dfs(root, effective_time_limit, expansion_prob)
            else:
                raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")
            
            # Compute the log-likelihood for this run
            log_likelihood = compute_log_likelihood(visited, responses, ranks)
            total_log_likelihood += log_likelihood
        
        # Average log-likelihood over runs
        average_log_likelihood = total_log_likelihood / num_runs
        
        # Use pyro.factor to include the likelihood in the model
        pyro.factor('likelihood', average_log_likelihood)
    
    # Set up the NUTS kernel
    nuts_kernel = NUTS(pyro_model)
    
    # Run MCMC
    mcmc = MCMC(nuts_kernel, num_samples=50, warmup_steps=10, num_chains=1)
    mcmc.run()
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    plot_trace(posterior_samples, 'expansion_prob', participant_id, f'Probabilistic {model_type}')
    plot_trace(posterior_samples, 'cognitive_resource', participant_id,  f'Probabilistic {model_type}')    
    plot_autocorrelation(posterior_samples, 'expansion_prob', participant_id,  f'Probabilistic {model_type}')
    plot_autocorrelation(posterior_samples, 'cognitive_resource', participant_id,  f'Probabilistic {model_type}')
    plot_posterior_density(posterior_samples, 'expansion_prob', participant_id,  f'Probabilistic {model_type}')
    plot_posterior_density(posterior_samples, 'cognitive_resource', participant_id,  f'Probabilistic {model_type}')
    plot_posterior_kde(posterior_samples, 'expansion_prob', participant_id,  f'Probabilistic {model_type}')
    plot_posterior_kde(posterior_samples, 'cognitive_resource', participant_id,  f'Probabilistic {model_type}')

    # Compute summary statistics
    expansion_prob_samples = posterior_samples['expansion_prob'].detach().numpy()
    cognitive_resource_samples = posterior_samples['cognitive_resource'].detach().numpy()
    
    expansion_prob_mean = np.mean(expansion_prob_samples)
    cognitive_resource_mean = np.mean(cognitive_resource_samples)

    expansion_prob_ci = np.percentile(posterior_samples['expansion_prob'], [2.5, 97.5])
    cognitive_resource_ci = np.percentile(posterior_samples['cognitive_resource'], [2.5, 97.5])
    
    # Perform simulations and compute likelihood
    effective_time_limit = base_time_limit * scaling_factor * cognitive_resource_mean
    root = build_tree(scene)
    num_runs = 10  # Adjust as needed
    total_log_likelihood = 0.0
    for _ in range(num_runs):
        if model_type == 'BFS':
            visited = probabilistic_bfs(root, effective_time_limit, expansion_prob_mean)
        elif model_type == 'DFS':
            visited, _ = probabilistic_dfs(root, effective_time_limit, expansion_prob_mean)
        else:
            raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")
        log_likelihood = compute_log_likelihood(visited, responses, ranks)
        total_log_likelihood += log_likelihood.item()
    average_log_likelihood = total_log_likelihood / num_runs
    
    # Compute AIC
    log_likelihood = average_log_likelihood
    k = 2  # Number of parameters
    aic = compute_aic(log_likelihood, k)
    
    return {
        'ExpansionProb': expansion_prob_mean,
        'ExpansionProbCI': expansion_prob_ci,
        'CognitiveResource': cognitive_resource_mean,
        'CognitiveResourceCI': cognitive_resource_ci,
        'log_likelihood': log_likelihood,
        'AIC': aic,
        'posterior_samples': posterior_samples
    }



