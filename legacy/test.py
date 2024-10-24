import unittest
import numpy as np
import pandas as pd
from data_processing import load_data, preprocess_data
from models import build_tree, deterministic_bfs, deterministic_dfs, probabilistic_bfs, probabilistic_dfs
from evaluation import compute_likelihood, compute_aic
from fit_models import fit_deterministic_model, fit_probabilistic_model



class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.mock_data = pd.DataFrame({
            'scene': ['Drop', 'Push', 'Pull', 'Pull'],
            'condition': ['250ms', '500ms', '1000ms', '2000ms'],
            'invalid': [np.nan, np.nan, 1, np.nan],
            'DropQ_0_GROUP_1': ['SthYes', '', np.nan, np.nan],
            'DropQ_0_1_RANK': [1, np.nan, np.nan, np.nan],
            'DropQ_0_GROUP_2': ['SthYes', '', np.nan, np.nan],
            'DropQ_0_2_RANK': [2, np.nan, np.nan, np.nan],
            'PushQ_0_GROUP_1': ['', 'SthYes', np.nan, np.nan],
            'PushQ_0_1_RANK': [np.nan, 1, np.nan, np.nan],
            'PushQ_1_GROUP_2': ['', 'SthNo', np.nan, np.nan],
            'PushQ_1_2_RANK': [np.nan, np.nan, np.nan, np.nan],
            'PullQ_0_GROUP_1': ['', '', '', ''],
            'PullQ_0_1_RANK': [np.nan, np.nan, np.nan, np.nan],
            
        })
        print("Testing Data Processing...")

    def test_load_data(self):
        self.data = load_data()

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.mock_data)

        # Test that invalid participants are excluded
        self.assertEqual(len(processed_data), 3)  # One participant is invalid

        # Test that ParticipantID is assigned correctly
        self.assertListEqual(processed_data['ParticipantID'].tolist(), [1, 2, 4])

        # Test that ScalingFactor is calculated correctly
        expected_scaling_factors = [1, 2, 8]
        self.assertListEqual(processed_data['ScalingFactor'].tolist(), expected_scaling_factors)

        # Test that properties are processed correctly
        participant_data = processed_data.iloc[0]
        self.assertIn('Response_loc(s1, s2)', participant_data)
        self.assertEqual(participant_data['Response_loc(s1, s2)'], 1)
        self.assertEqual(participant_data['Rank_loc(s1, s2)'], 1)
        self.assertEqual(participant_data['Response_size(s1)'], 1)
        self.assertEqual(participant_data['Rank_size(s1)'], 2)

        # Test that 'No' responses are handled correctly
        participant_data = processed_data.iloc[1]
        self.assertEqual(participant_data['Response_loc(s1, s2)'], 1)
        self.assertEqual(participant_data['Rank_loc(s1, s2)'], 1)
        self.assertEqual(participant_data['Response_weight(s1)'], 0)
        self.assertTrue(np.isnan(participant_data['Rank_weight(s1)']))

        # Test that missing data are handled correctly
        participant_data = processed_data.iloc[2]
        self.assertTrue(np.isnan(participant_data['Response_loc(s1, s2)']))
        self.assertTrue(np.isnan(participant_data['Rank_loc(s1, s2)']))


class TestModels(unittest.TestCase):
    def setUp(self):
        self.scenes = ['Drop', 'Push', 'Pull']
        self.cognitive_resource = 1.0
        self.expansion_prob = 0.5
        self.effective_time_limit = 5  # For simplicity
        print("Testing Models...")

    def test_build_tree(self):
        for scene in self.scenes:
            root = build_tree(scene)
            self.assertIsNotNone(root)
            self.assertEqual(root.name, scene)
            self.assertGreater(len(root.children), 0)

    def test_deterministic_bfs(self):
        root = build_tree('Drop')
        visited = deterministic_bfs(root, self.effective_time_limit)
        self.assertIsInstance(visited, list)
        self.assertLessEqual(len(visited), self.effective_time_limit)

    def test_deterministic_dfs(self):
        root = build_tree('Drop')
        visited, time_spent = deterministic_dfs(root, self.effective_time_limit)
        self.assertIsInstance(visited, list)
        self.assertLessEqual(time_spent, self.effective_time_limit)

    def test_probabilistic_bfs(self):
        root = build_tree('Drop')
        visited = probabilistic_bfs(root, self.effective_time_limit, self.expansion_prob)
        self.assertIsInstance(visited, list)
        self.assertLessEqual(len(visited), self.effective_time_limit)

    def test_probabilistic_dfs(self):
        root = build_tree('Drop')
        visited, time_spent = probabilistic_dfs(root, self.effective_time_limit, self.expansion_prob)
        self.assertIsInstance(visited, list)
        self.assertLessEqual(time_spent, self.effective_time_limit)


class TestEvaluation(unittest.TestCase):
    print("Testing Evaluations...")
    def test_compute_likelihood(self):
        # Example data
        visited = ['Drop', 'loc(s1, s2)', 'size(s1)', 'traj(s1)']
        responses = {'loc(s1, s2)': 1, 'size(s1)': 0, 'traj(s1)': 1, 'person': 1}
        ranks = {'loc(s1, s2)': 1, 'size(s1)': np.nan, 'traj(s1)': 2, 'person': 3}
        log_likelihood = compute_likelihood(visited, responses, ranks)
        self.assertIsInstance(log_likelihood, float)

    def test_compute_aic(self):
        log_likelihood = -10.0
        k = 2
        aic = compute_aic(log_likelihood, k)
        expected_aic = 2 * k - 2 * log_likelihood
        self.assertEqual(aic, expected_aic)


class TestFitModels(unittest.TestCase):
    def setUp(self):
        print("Testing Fit Models...")
        # Mock participant data
        self.participant_data = {
            'ParticipantID': 1,
            'Scene': 'Drop',
            'ScalingFactor': 1,
            'CognitiveResource': 1.0,  # Will be updated during fitting
            'DefaultEffectiveTimeLimit': 5,
            'Properties': ['loc(s1, s2)', 'size(s1)', 'traj(s1)', 'person'],
            'Response_loc(s1, s2)': 1,
            'Rank_loc(s1, s2)': 1,
            'Response_size(s1)': 0,
            'Rank_size(s1)': np.nan,
            'Response_traj(s1)': 1,
            'Rank_traj(s1)': 2,
            'Response_person': 0,
            'Rank_person': np.nan,
        }

    def test_fit_deterministic_model(self):
        participant_series = pd.Series(self.participant_data)
        result = fit_deterministic_model(participant_series, model_type='BFS')
        self.assertIn('CognitiveResource', result)
        self.assertIn('log_likelihood', result)
        self.assertIn('AIC', result)

    def test_fit_probabilistic_model(self):
        participant_series = pd.Series(self.participant_data)
        result = fit_probabilistic_model(participant_series, model_type='BFS')
        self.assertIn('ExpansionProb', result)
        self.assertIn('CognitiveResource', result)
        self.assertIn('log_likelihood', result)
        self.assertIn('AIC', result)


if __name__ == '__main__':
    unittest.main()
