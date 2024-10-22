import numpy as np
import pandas as pd

def compute_likelihood(visited, responses, ranks):
    total_log_likelihood = 0

    # Map properties to their expansion order
    expansion_order = {prop: idx + 1 for idx, prop in enumerate(visited)}

    for prop in responses.keys():
        response = responses[prop]
        reported_rank = ranks[prop]

        if np.isnan(response):
            continue  # Skip if response is missing

        if response == 1:
            if prop in expansion_order:
                model_rank = expansion_order[prop]
                if not np.isnan(reported_rank):
                    rank_difference = abs(model_rank - reported_rank)
                    # Gaussian likelihood centered at zero rank difference
                    sigma = 1.0
                    rank_prob = np.exp(-0.5 * (rank_difference / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                else:
                    # Missing rank but participant said 'yes'
                    rank_prob = 1e-6
            else:
                # Model did not predict the property
                rank_prob = 1e-6
        elif response == 0:
            if prop in expansion_order:
                # Model predicted the property, participant did not
                rank_prob = 1e-6
            else:
                # Neither model nor participant includes property
                rank_prob = 1.0
        else:
            continue  # Skip if response is missing

        total_log_likelihood += np.log(rank_prob + 1e-9)  # Avoid log(0)

    return total_log_likelihood

def compute_aic(log_likelihood, k):
    aic = 2 * k - 2 * log_likelihood
    return aic
