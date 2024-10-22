import pandas as pd
from data_processing import load_data, preprocess_data
from fit_models import fit_deterministic_model, fit_probabilistic_model
import matplotlib.pyplot as plt

import sys


def main():
    # Load and preprocess data
    data = load_data('/Users/yichen/Downloads/scene_construction/version6_pilot1_cleaned.csv')  # Replace with your data file
    processed_data = preprocess_data(data)
    print(processed_data.head(20))

    # Initialize results list
    results = []

    # Iterate over participants
    for idx, participant_data in processed_data.iterrows():
        if idx >=2:
            break
        participant_id = participant_data['ParticipantID']
        print(f"fitting participant {participant_id}")

        # Deterministic BFS
        print("\tfitting det bfs...")
        det_bfs_result = fit_deterministic_model(participant_data, model_type='BFS')
        det_bfs_result['ParticipantID'] = participant_id
        det_bfs_result['Model'] = 'Deterministic BFS'
        print("\tdone")

        # Deterministic DFS
        print("\tfitting det dfs...")
        det_dfs_result = fit_deterministic_model(participant_data, model_type='DFS')
        det_dfs_result['ParticipantID'] = participant_id
        det_dfs_result['Model'] = 'Deterministic DFS'
        print("\tdone")

        # Probabilistic BFS
        print("\tfitting prob bfs...")
        prob_bfs_result = fit_probabilistic_model(participant_data, model_type='BFS')
        prob_bfs_result['ParticipantID'] = participant_id
        prob_bfs_result['Model'] = 'Probabilistic BFS'
        print("\tdone")

        # Probabilistic DFS
        print("\tfitting prob dfs...")
        prob_dfs_result = fit_probabilistic_model(participant_data, model_type='DFS')
        prob_dfs_result['ParticipantID'] = participant_id
        prob_dfs_result['Model'] = 'Probabilistic DFS'
        print("\tdone")

        # Append results
        results.extend([det_bfs_result, det_dfs_result, prob_bfs_result, prob_dfs_result])

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv('model_results.csv', index=False)

    # Analyze results
    analyze_results(results_df)



def analyze_results(results_df):
    # Compute average AIC per model
    model_aic = results_df.groupby('Model')['AIC'].mean().reset_index()

    # Plot AIC
    plt.figure(figsize=(8, 6))
    plt.bar(model_aic['Model'], model_aic['AIC'])
    plt.title('Average AIC per Model')
    plt.ylabel('AIC')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_aic.png')
    plt.show()



if __name__ == '__main__':
    main()
