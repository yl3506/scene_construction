import matplotlib.pyplot as plt
from models import build_tree, deterministic_bfs, probabilistic_bfs

def simulate_tree_traversal(scene, model_type, cognitive_resource, expansion_prob=None):
    root = build_tree(scene)
    default_effective_time_limit = 5
    scaling_factor = 1  # For simplicity, adjust as needed
    effective_time_limit = default_effective_time_limit * scaling_factor * cognitive_resource

    if model_type == 'Deterministic BFS':
        visited = deterministic_bfs(root, effective_time_limit)
    elif model_type == 'Probabilistic BFS':
        if expansion_prob is None:
            expansion_prob = 0.5  # Default value
        visited = probabilistic_bfs(root, effective_time_limit, expansion_prob)
    else:
        raise ValueError("Invalid model type.")

    print(f"Visited nodes for {model_type} with cognitive_resource={cognitive_resource}, expansion_prob={expansion_prob}:")
    print(visited)



def main():
    # Simulate and visualize
    scenes = ['Drop', 'Push', 'Pull']
    cognitive_resource_values = [0.5, 1.0, 1.5]
    expansion_prob_values = [0.3, 0.6, 0.9]

    for scene in scenes:
        visualize_tree(scene)
        for cr in cognitive_resource_values:
            simulate_tree_traversal(scene, 'Deterministic BFS', cognitive_resource=cr)
            for ep in expansion_prob_values:
                simulate_tree_traversal(scene, 'Probabilistic BFS', cognitive_resource=cr, expansion_prob=ep)
