import unittest
import pandas as pd
from fit_models import fit_deterministic_model, fit_probabilistic_model
import numpy as np

class TestFitModels(unittest.TestCase):
    def setUp(self):
        # Mock participant data
        self.participant_data = {
            'ParticipantID': 1,
            'Scene': 'Drop',
            'ScalingFactor': 1,
            'CognitiveResource': 1.0,  # Will be updated during fitting
            'DefaultEffectiveTimeLimit': 5,
            'Properties': ['loc(s1,s2)', 'size(s1)', 'traj(s1)', 'person'],
            'Response_loc(s1,s2)': 1,
            'Rank_loc(s1,s2)': 1,
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
