import numpy as np
from collections import deque
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from models import deterministic_bfs, deterministic_dfs, probabilistic_bfs, probabilistic_dfs, build_tree
from evaluation import compute_likelihood, compute_aic
import matplotlib.pyplot as plt

def fit_deterministic_model(participant_data, model_type):
    scene = participant_data['Scene']
    root = build_tree(scene)

    # Parameters to fit: Cognitive Resource
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']

    # Objective function to minimize (negative log-likelihood)
    def objective_func(params):
        cognitive_resource = params[0]
        effective_time_limit = base_time_limit * scaling_factor * cognitive_resource
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
    bounds = [(0.1, 5.0)]  
    # Minimize
    result = minimize(
        objective_func,
        x0=initial_guess,
        bounds=bounds,
        method='L-BFGS-B', #'Nelder-Mead', 
        options={'disp': False, 'maxiter': 1000}
    )
    if not result.success:
        print("\t!!! Optimization failed.")
    cognitive_resource = result.x[0] # solution for the parameter
    log_likelihood = -result.fun # value of objective function
    nit = result.nit # number of iterations performed by the optimizer
    k = 1  # One parameter (Cognitive Resource)
    aic = compute_aic(log_likelihood, k)

    return {'CognitiveResource': cognitive_resource, 'log_likelihood': log_likelihood, 'AIC': aic, 'num_iter': nit}, result.success




def fit_probabilistic_model(participant_data, model_type):
    scene = participant_data['Scene']
    root = build_tree(scene)
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']
    participant_id = participant_data['ParticipantID']
    np.random.seed(participant_id)  

    def objective_func(params):
        # the simulation solution for the likelihood
        expansion_prob, cognitive_resource = params
        effective_time_limit = base_time_limit * scaling_factor * cognitive_resource
        num_runs = 50  # Number of iterations to average over, smooth out randomness for the optimization
        total_neg_log_likelihood = 0
        for _ in range(num_runs):
            if model_type == 'BFS':
                visited = probabilistic_bfs(root, effective_time_limit, expansion_prob)
            elif model_type == 'DFS':
                visited, _ = probabilistic_dfs(root, effective_time_limit, expansion_prob)
            # Prepare responses and ranks
            properties = participant_data['Properties']
            responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
            ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}
            total_neg_log_likelihood += -compute_likelihood(visited, responses, ranks)
        average_neg_log_likelihood = total_neg_log_likelihood / num_runs
        return average_neg_log_likelihood

    # def objective_func(params):
    #     # an anlytical solution for the likelihood
    #     expansion_prob, cognitive_resource = params
    #     effective_time_limit = base_time_limit * scaling_factor * cognitive_resource
    #     # Compute expected visit times
    #     node_expected_times = compute_expected_visit_times(root, effective_time_limit, expansion_prob)
    #     # Prepare responses
    #     properties = participant_data['Properties']
    #     responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
    #     ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}
    #     # Compute negative log-likelihood
    #     neg_log_likelihood = -compute_likelihood_with_ranks(node_expected_times, responses, ranks)
    #     return neg_log_likelihood

    # Initial guesses and bounds
    initial_guess = [0.5, 1.0]  # [expansion_prob, cognitive_resource]
    bounds = [(0, 1), (0.1, 5.0)] 

    # # Use gradient-based optimizer is obj function is deterministic
    # result = minimize(
    #     objective_func,
    #     x0=initial_guess,
    #     bounds=bounds,
    #     method='L-BFGS-B',
    #     options={'disp': False, 'maxiter': 1000}
    # )

    result = differential_evolution(
        objective_func,
        bounds=bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
    )

    if not result.success:
        print("\t*** Optimization failed.")
    expansion_prob, cognitive_resource = result.x
    log_likelihood = -result.fun
    nit = result.nit
    k = 2  # Two parameters (expansion_prob and cognitive_resource)
    aic = compute_aic(log_likelihood, k)

    return {'ExpansionProb': expansion_prob, 'CognitiveResource': cognitive_resource, 'log_likelihood': log_likelihood, 'AIC': aic, 'num_iter': nit}, result.success



def compute_expected_visit_times(root, effective_time_limit, expansion_prob):
    node_visit_probs = {}  # {node_name: {time: prob}}
    node_expected_times = {}
    queue = deque()
    queue.append((root, 1.0, 0))  # (node, probability, time_step)
    while queue:
        node, prob, time = queue.popleft()
        if time >= effective_time_limit:
            continue
        # Update visit probabilities for the node
        if node.name not in node_visit_probs:
            node_visit_probs[node.name] = {}
        if time not in node_visit_probs[node.name]:
            node_visit_probs[node.name][time] = prob
        else:
            node_visit_probs[node.name][time] += prob
        # Enqueue children
        for child in node.children:
            child_prob = prob * expansion_prob
            queue.append((child, child_prob, time + 1))
    # Compute expected visit times
    for node_name, time_probs in node_visit_probs.items():
        total_prob = sum(time_probs.values())
        expected_time = sum(t * p for t, p in time_probs.items()) / total_prob
        node_expected_times[node_name] = expected_time + 1
    return node_expected_times


def compute_likelihood_with_ranks(node_expected_times, responses, ranks, sigma=1.0):
    total_log_likelihood = 0

    for node_name, response in responses.items():
        if np.isnan(response):
            continue  # Skip missing responses

        expected_time = node_expected_times.get(node_name, None)
        if expected_time is None or np.isinf(expected_time):
            continue  # Node not reachable within time limit

        rank = ranks.get(node_name, None)
        if np.isnan(rank) or np.isinf(rank):
            continue  # No rank reported

        if response == 1:
            # Likelihood based on the difference between expected time and reported rank
            likelihood = norm.pdf(rank, loc=expected_time, scale=sigma) + 1e-9
        elif response == 0:
            # Likelihood of not reporting the node
            likelihood = 1 - norm.cdf(rank, loc=expected_time, scale=sigma) + 1e-9
        else:
            continue  # Skip invalid responses
        # Avoid likelihoods of zero
        likelihood = max(likelihood, 1e-9)
        total_log_likelihood += np.log(likelihood)

    return total_log_likelihood


def test_likelihood_sensitivity(participant_data):
    scene = participant_data['Scene']
    root = build_tree(scene)
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']
    properties = participant_data['Properties']
    responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
    ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}

    expansion_probs = np.linspace(0.01, 0.99, 50)
    cognitive_resources = np.linspace(0.1, 5.0, 50)
    neg_log_likelihoods = np.zeros((len(expansion_probs), len(cognitive_resources)))

    for i, exp_prob in enumerate(expansion_probs):
        for j, cog_res in enumerate(cognitive_resources):
            # the simulation solution for the likelihood
            expansion_prob, cognitive_resource = exp_prob, cog_res
            effective_time_limit = base_time_limit * scaling_factor * cognitive_resource
            num_runs = 30  # Number of iterations to average over, smooth out randomness for the optimization
            total_neg_log_likelihood = 0
            for _ in range(num_runs):
                visited = probabilistic_bfs(root, effective_time_limit, expansion_prob)
                # visited, _ = probabilistic_dfs(root, effective_time_limit, expansion_prob)
                properties = participant_data['Properties']
                responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
                ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}
                total_neg_log_likelihood += -compute_likelihood(visited, responses, ranks)
            average_neg_log_likelihood = total_neg_log_likelihood / num_runs
            neg_log_likelihoods[i, j] = average_neg_log_likelihood
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(cognitive_resources, expansion_probs)
    contour = plt.contourf(X, Y, neg_log_likelihoods, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Negative Log-Likelihood')
    plt.xlabel('Cognitive Resource')
    plt.ylabel('Expansion Probability')
    plt.title('Likelihood Sensitivity with Rank Information')
    plt.show()




# def test_likelihood_sensitivity(participant_data):
#     scene = participant_data['Scene']
#     root = build_tree(scene)
#     base_time_limit = participant_data['BaseTimeLimit']
#     scaling_factor = participant_data['ScalingFactor']
#     properties = participant_data['Properties']
#     responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
#     ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}

#     expansion_probs = np.linspace(0.01, 0.99, 50)
#     cognitive_resources = np.linspace(0.1, 5.0, 50)
#     neg_log_likelihoods = np.zeros((len(expansion_probs), len(cognitive_resources)))

#     for i, exp_prob in enumerate(expansion_probs):
#         for j, cog_res in enumerate(cognitive_resources):
#             effective_time_limit = base_time_limit * scaling_factor * cog_res
#             node_expected_times = compute_expected_visit_times(root, effective_time_limit, exp_prob)
#             # Optionally print expected times for key nodes
#             if i % 10 == 0 and j % 10 == 0:
#                 print(f"exp_prob: {exp_prob}, cog_res: {cog_res}, expected_time: {node_expected_times.get('loc(s1, s2)')}")
#             neg_log_likelihood = -compute_likelihood_with_ranks(node_expected_times, responses, ranks)
#             neg_log_likelihoods[i, j] = neg_log_likelihood

#     plt.figure(figsize=(10, 8))
#     X, Y = np.meshgrid(cognitive_resources, expansion_probs)
#     contour = plt.contourf(X, Y, neg_log_likelihoods, levels=50, cmap='viridis')
#     plt.colorbar(contour, label='Negative Log-Likelihood')
#     plt.xlabel('Cognitive Resource')
#     plt.ylabel('Expansion Probability')
#     plt.title('Likelihood Sensitivity with Rank Information')
#     plt.show()

