import numpy as np
from scipy.optimize import minimize
from models import deterministic_bfs, deterministic_dfs, probabilistic_bfs, probabilistic_dfs, build_tree
from evaluation import compute_likelihood, compute_aic

def fit_deterministic_model(participant_data, model_type):
    scene = participant_data['Scene']
    root = build_tree(scene)

    # Parameters to fit: Cognitive Resource
    default_effective_time_limit = participant_data['DefaultEffectiveTimeLimit']
    scaling_factor = participant_data['ScalingFactor']

    # Objective function to minimize (negative log-likelihood)
    def objective_func(params):
        cognitive_resource = params[0]
        effective_time_limit = default_effective_time_limit * scaling_factor * cognitive_resource

        if model_type == 'BFS':
            visited = deterministic_bfs(root, effective_time_limit)
        elif model_type == 'DFS':
            visited, _ = deterministic_dfs(root, effective_time_limit)
        else:
            raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")

        # Prepare responses and ranks
        properties = participant_data['Properties']
        responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
        ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}

        # Compute negative log-likelihood
        neg_log_likelihood = -compute_likelihood(visited, responses, ranks)
        return neg_log_likelihood

    # Initial guess and bounds for Cognitive Resource
    initial_guess = [1.0]
    bounds = [(0.1, 2.0)]  # Assuming Cognitive Resource ranges from 0.1 to 2.0

    # Minimize
    result = minimize(
        objective_func,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )

    cognitive_resource = result.x[0]
    log_likelihood = -result.fun
    k = 1  # One parameter (Cognitive Resource)
    aic = compute_aic(log_likelihood, k)

    return {'CognitiveResource': cognitive_resource, 'log_likelihood': log_likelihood, 'AIC': aic}

def fit_probabilistic_model(participant_data, model_type):
    scene = participant_data['Scene']
    root = build_tree(scene)
    default_effective_time_limit = participant_data['DefaultEffectiveTimeLimit']
    scaling_factor = participant_data['ScalingFactor']

    # Objective function
    def objective_func(params):
        expansion_prob, cognitive_resource = params
        effective_time_limit = default_effective_time_limit * scaling_factor * cognitive_resource

        np.random.seed(42)  # For reproducibility

        if model_type == 'BFS':
            visited = probabilistic_bfs(root, effective_time_limit, expansion_prob)
        elif model_type == 'DFS':
            visited, _ = probabilistic_dfs(root, effective_time_limit, expansion_prob)
        else:
            raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")

        properties = participant_data['Properties']
        responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
        ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}

        neg_log_likelihood = -compute_likelihood(visited, responses, ranks)
        return neg_log_likelihood

    # Initial guesses and bounds
    initial_guess = [0.5, 1.0]  # [expansion_prob, cognitive_resource]
    bounds = [(0, 1), (0.1, 2.0)]  # expansion_prob between 0 and 1, cognitive_resource between 0.1 and 2.0

    # Minimize
    result = minimize(
        objective_func,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )

    expansion_prob, cognitive_resource = result.x
    log_likelihood = -result.fun
    k = 2  # Two parameters (expansion_prob and cognitive_resource)
    aic = compute_aic(log_likelihood, k)

    return {'ExpansionProb': expansion_prob, 'CognitiveResource': cognitive_resource, 'log_likelihood': log_likelihood, 'AIC': aic}
