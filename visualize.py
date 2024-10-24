import matplotlib.pyplot as plt
from models import build_tree
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import numpy as np
from evaluation import compute_log_likelihood
from models import deterministic_bfs, deterministic_dfs, probabilistic_bfs, probabilistic_dfs


def log_likelihood_sensitivity_det(participant_data, model_type):
    participant_id = participant_data['ParticipantID']
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']
    properties = participant_data['Properties']
    responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
    ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}
    scene = participant_data['Scene']

    cognitive_resources = np.linspace(1.0, 30.0, 30)
    log_likelihoods = []
    for cr in cognitive_resources:
        effective_time_limit = base_time_limit * scaling_factor * cr
        root = build_tree(scene)
        if model_type == 'BFS':
            visited = deterministic_bfs(root, effective_time_limit)
        elif model_type == 'DFS':
            visited, _ = deterministic_dfs(root, effective_time_limit)
        else:
            raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")
        log_likelihood = compute_log_likelihood(visited, responses, ranks).item()
        log_likelihoods.append(log_likelihood)
    plt.figure(figsize=(8, 6))
    plt.plot(cognitive_resources, log_likelihoods, marker='o')
    plt.title(f'Log-Likelihood Sensitivity (Deterministic {model_type}), Participant {participant_id}')
    plt.xlabel('Cognitive Resource')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"logs/{participant_id}_loglikelihood_sensitivity_det_{model_type}.png"
    plt.savefig(filename)


def log_likelihood_sensitivity_prob(participant_data, model_type, param_grid=None, num_simulations=10):
    if param_grid is None:
        # Define default grids
        param_grid = {
            'cognitive_resource': np.linspace(1.0, 30.0, 30),
            'expansion_prob': np.linspace(0.01, 1.0, 30)
        }
    cognitive_resources = param_grid['cognitive_resource']
    expansion_probs = param_grid['expansion_prob']

    participant_id = participant_data['ParticipantID']
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']
    scene = participant_data['Scene']
    properties = participant_data['Properties']
    responses = {prop: participant_data[f'Response_{prop}'] for prop in properties}
    ranks = {prop: participant_data[f'Rank_{prop}'] for prop in properties}

    log_likelihood_matrix = np.zeros((len(cognitive_resources), len(expansion_probs)))

    # Iterate over grid points
    for i, cr in enumerate(cognitive_resources):
        for j, ep in enumerate(expansion_probs):
            # Compute effective time limit
            effective_time_limit = base_time_limit * scaling_factor * cr
            # Build the tree
            root = build_tree(scene)
            # Perform multiple traversal simulations to estimate log-likelihood
            total_log_likelihood = 0.0
            for _ in range(num_simulations):
                if model_type == 'BFS':
                    visited = probabilistic_bfs(root, effective_time_limit, ep)
                elif model_type == 'DFS':
                    visited, _ = probabilistic_dfs(root, effective_time_limit, ep)
                else:
                    raise ValueError("Invalid model type. Choose 'BFS' or 'DFS'.")
                # Compute log-likelihood for this traversal
                log_likelihood = compute_log_likelihood(visited, responses, ranks)
                total_log_likelihood += log_likelihood
            # Average log-likelihood
            average_log_likelihood = total_log_likelihood / num_simulations
            log_likelihood_matrix[i, j] = average_log_likelihood
    # Create contour plot
    CR, EP = np.meshgrid(expansion_probs, cognitive_resources)
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(EP, CR, log_likelihood_matrix, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.title(f'Log-Likelihood Sensitivity (Probabilistic {model_type}), Participant {participant_id}')
    plt.ylabel('Expansion Probability')
    plt.xlabel('Cognitive Resource')
    plt.tight_layout()

    filename = f"logs/{participant_id}_loglikelihood_sensitivity_prob_{model_type}.png"
    plt.savefig(filename)


def plot_autocorrelation(posterior_samples, param_name, participant_id, model_type, lags=40):
    samples = posterior_samples[param_name].detach().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(samples, ax=ax, lags=lags)
    ax.set_title(f'Autocorrelation for {param_name} (Participant {participant_id}, Model {model_type})')
    plt.tight_layout()

    filename = f"logs/{participant_id}_autocorrelation_{model_type}_{param_name}.png"
    plt.savefig(filename)

def plot_trace(posterior_samples, param_name, participant_id, model_type):
    samples = posterior_samples[param_name].detach().numpy()
    plt.figure(figsize=(12, 4))
    plt.plot(samples)
    plt.title(f'Trace Plot for {param_name} (Participant {participant_id}, Model {model_type})')
    plt.xlabel('Sample')
    plt.ylabel(param_name)
    plt.tight_layout()

    filename = f"logs/{participant_id}_trace_{model_type}_{param_name}.png"
    plt.savefig(filename)

def plot_posterior_density(posterior_samples, param_name, participant_id, model_type):
    samples = posterior_samples[param_name].detach().numpy()
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='skyblue')
    plt.title(f'Posterior Distribution of {param_name} (Participant {participant_id}, Model {model_type})')
    plt.xlabel(param_name)
    plt.ylabel('Density')
    plt.tight_layout()

    filename = f"logs/{participant_id}_posterior_density_{model_type}_{param_name}.png"
    plt.savefig(filename)

def plot_posterior_kde(posterior_samples, param_name, participant_id, model_type):
    samples = posterior_samples[param_name].detach().numpy()
    plt.figure(figsize=(8, 6))
    sns.kdeplot(samples, color='green')
    plt.title(f'Posterior KDE of {param_name} (Participant {participant_id}, Model {model_type})')
    plt.xlabel(param_name)
    plt.ylabel('Density')
    plt.tight_layout()

    filename = f"logs/{participant_id}_posterior_kde_{model_type}_{param_name}.png"
    plt.savefig(filename)

def plot_average_performance(results_df, filename='results_model_aic.png'):
    # Compute mean and standard error of AIC per model
    model_stats = results_df.groupby('Model')['AIC'].agg(['mean', 'sem']).reset_index()
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.bar(
        model_stats['Model'],
        model_stats['mean'],
        yerr=model_stats['sem'],
        capsize=5
    )
    plt.title('Average AIC per Model')
    plt.ylabel('AIC')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)


def plot_best_model_histogram(results_df, num_bootstrap=1000, filename='results_best_model.png'):
    import pandas as pd

    # Define the list of all models
    all_models = ['Deterministic BFS', 'Deterministic DFS', 'Probabilistic BFS', 'Probabilistic DFS']
    # Original counts
    best_models = results_df.loc[results_df.groupby('ParticipantID')['AIC'].idxmin()]
    model_counts = best_models['Model'].value_counts().reindex(all_models, fill_value=0)
    # Bootstrap
    bootstrap_counts = []
    for _ in range(num_bootstrap):
        sample = best_models.sample(frac=1, replace=True)
        counts = sample['Model'].value_counts().reindex(all_models, fill_value=0)
        bootstrap_counts.append(counts)
    # Convert to DataFrame
    bootstrap_df = pd.DataFrame(bootstrap_counts)
    count_means = bootstrap_df.mean()
    count_sems = bootstrap_df.sem()
    # Plotting
    plt.figure(figsize=(8, 6))
    count_means.plot(kind='bar', yerr=count_sems, capsize=5)
    plt.title('Number of Participants Best Explained by Each Model')
    plt.xlabel('Model')
    plt.ylabel('Number of Participants')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)




def simulate_traversals(participant_data, model_type, expansion_prob_samples, cognitive_resource_samples):
    base_time_limit = participant_data['BaseTimeLimit']
    scaling_factor = participant_data['ScalingFactor']
    scene = participant_data['Scene']
    root = build_tree(scene)
    
    simulated_traversals = []
    for exp_prob, cog_res in zip(expansion_prob_samples, cognitive_resource_samples):
        effective_time_limit = base_time_limit * scaling_factor * cog_res
        if model_type == 'BFS':
            visited = probabilistic_bfs(root, effective_time_limit, exp_prob)
        elif model_type == 'DFS':
            visited, _ = probabilistic_dfs(root, effective_time_limit, exp_prob)
        simulated_traversals.append(visited)
    return simulated_traversals

def visualize_tree(scene):
    import networkx as nx
    root = build_tree(scene)
    G = nx.Graph()

    def add_edges(node):
        for child in node.children:
            G.add_edge(node.name, child.name)
            add_edges(child)

    add_edges(root)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, font_weight='bold')
    plt.title(f"Tree Structure for {scene} Scene")
    plt.show()
