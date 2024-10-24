import pandas as pd
from data_processing import load_data, preprocess_data
from fit_models import fit_deterministic_model, fit_probabilistic_model, test_likelihood_sensitivity
import matplotlib.pyplot as plt
import sys


def main():
    # Load and preprocess data
    data = load_data(datafile) 
    processed_data = preprocess_data(data)
    print(processed_data.head())

    # Initialize results list
    results = []

    # Iterate over participants
    for idx, participant_data in processed_data.iterrows():
        test_likelihood_sensitivity(participant_data)
        if idx>=1:
            break

        participant_id = participant_data['ParticipantID']
        print(f"fitting participant {participant_id}")

        # Deterministic BFS
        det_bfs_result, success = fit_deterministic_model(participant_data, model_type='BFS')
        if not success:
            continue
        det_bfs_result['ParticipantID'] = participant_id
        det_bfs_result['Model'] = 'Deterministic BFS'

        # Deterministic DFS
        det_dfs_result, success = fit_deterministic_model(participant_data, model_type='DFS')
        if not success:
            continue
        det_dfs_result['ParticipantID'] = participant_id
        det_dfs_result['Model'] = 'Deterministic DFS'

        # Probabilistic BFS
        prob_bfs_result, success = fit_probabilistic_model(participant_data, model_type='BFS')
        if not success:
            continue
        prob_bfs_result['ParticipantID'] = participant_id
        prob_bfs_result['Model'] = 'Probabilistic BFS'

        # Probabilistic DFS
        prob_dfs_result, success = fit_probabilistic_model(participant_data, model_type='DFS')
        if not success:
            continue
        prob_dfs_result['ParticipantID'] = participant_id
        prob_dfs_result['Model'] = 'Probabilistic DFS'

        # Append results
        results.extend([det_bfs_result, det_dfs_result, prob_bfs_result, prob_dfs_result])

    # save and analyze results
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results.csv', index=False)
    analyze_results(results_df)
    plot_best_model_histogram(results_df)



def analyze_results(results_df):
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
    plt.savefig('model_aic.png')
    

def plot_best_model_histogram(results_df, num_bootstrap=1000):
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
    plt.savefig('best_model.png')



if __name__ == '__main__':
    datafile = '/Users/yichen/Downloads/scene_construction/version6_pilot1_cleaned.csv'
    main()
