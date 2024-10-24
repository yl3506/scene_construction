import pandas as pd
from data_processing import load_data, preprocess_data
from fit_models import fit_probabilistic_model, fit_deterministic_model
from visualize import plot_average_performance, plot_best_model_histogram, \
        log_likelihood_sensitivity_det, log_likelihood_sensitivity_prob

def main():
    data = load_data(datafile)
    processed_data = preprocess_data(data)
    print(processed_data.head())

    # Initialize results list
    results = []

    # Iterate over participants
    for idx, participant_data in processed_data.iterrows():
        if idx >= 1:
            break
        participant_id = participant_data['ParticipantID']
        print(f"Fitting participant {participant_id}")

        log_likelihood_sensitivity_det(participant_data, model_type='DFS')
        log_likelihood_sensitivity_det(participant_data, model_type='BFS')
        log_likelihood_sensitivity_prob(participant_data, model_type='DFS')
        log_likelihood_sensitivity_prob(participant_data, model_type='BFS')

        # Probabilistic DFS
        prob_dfs_result = fit_probabilistic_model(participant_data, model_type='DFS')
        prob_dfs_result['ParticipantID'] = participant_id
        prob_dfs_result['Model'] = 'Probabilistic DFS'

        # Probabilistic BFS
        prob_bfs_result = fit_probabilistic_model(participant_data, model_type='BFS')
        prob_bfs_result['ParticipantID'] = participant_id
        prob_bfs_result['Model'] = 'Probabilistic BFS'

        # Deterministic DFS
        det_dfs_result = fit_deterministic_model(participant_data, model_type='DFS')
        det_dfs_result['ParticipantID'] = participant_id
        det_dfs_result['Model'] = 'Deterministic DFS'

        # Deterministic BFS
        det_bfs_result = fit_deterministic_model(participant_data, model_type='BFS')
        det_bfs_result['ParticipantID'] = participant_id
        det_bfs_result['Model'] = 'Deterministic BFS'

        # Append results
        results.extend([det_bfs_result, det_dfs_result, prob_bfs_result, prob_dfs_result])

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(resultsfile, index=False)

    # Plot the average performance of each model
    plot_average_performance(results_df)

    # Plot the histogram of best model
    plot_best_model_histogram(results_df)




if __name__== '__main__':
    datafile = '/Users/yichen/Downloads/scene_construction/version6_pilot1_cleaned.csv'
    resultsfile = 'results.csv'
    main()

