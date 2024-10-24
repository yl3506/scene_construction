import torch
import numpy as np

def compute_log_likelihood(visited, responses, ranks):
    total_log_likelihood = torch.tensor(0.0)
    sigma = torch.tensor(1.0)
    max_rank = torch.tensor(9.0)

    # Map properties to their expansion order
    expansion_order = {prop: idx + 1 for idx, prop in enumerate(visited)}

    for prop in responses.keys():
        response = responses[prop]
        reported_rank = ranks[prop]

        if np.isnan(response):
            continue  # Participant response missing, skip

        if response == 1:
            # Participant responded 'yes'
            if prop in expansion_order:
                # Model predicted the property
                model_rank = expansion_order[prop]
                if not np.isnan(reported_rank):
                    rank_difference = abs(model_rank - reported_rank)
                    rank_difference = torch.tensor(rank_difference, dtype=torch.float32)
                    # Gaussian likelihood centered at zero rank difference
                    rank_prob = torch.exp(-0.5 * (rank_difference / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
                else:
                    # Missing rank data but participant said 'yes'
                    rank_prob = torch.tensor(1e-6)
            else:
                # Model did not predict the property
                rank_difference = max_rank / torch.tensor(reported_rank, dtype=torch.float32)
                rank_prob = torch.exp(-0.5 * (rank_difference / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
        elif response == 0:
            # Participant responded 'no'
            if prop in expansion_order:
                # Model predicted the property, participant did not
                model_rank = expansion_order[prop]
                rank_difference = max_rank / torch.tensor(model_rank, dtype=torch.float32)
                rank_prob = torch.exp(-0.5 * (rank_difference / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
            else:
                # Neither model nor participant includes property
                rank_prob = torch.tensor(1.0)
        else:
            continue  # Participant response invalid, skip

        total_log_likelihood += torch.log(rank_prob + 1e-9)  # Avoid log(0)

    return total_log_likelihood



def compute_aic(log_likelihood, k):
    aic = 2 * k - 2 * log_likelihood
    return aic


