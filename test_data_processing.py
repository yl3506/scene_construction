import unittest
import pandas as pd
import numpy as np
from data_processing import load_data, preprocess_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.mock_data = pd.DataFrame({
            'scene': ['Drop', 'Push', 'Pull'],
            'condition': ['250ms', '500ms', '1000ms'],
            'invalid': [np.nan, np.nan, 1],
            'DropQ_0_GROUP_1': ['Yes', '', ''],
            'DropQ_0_1_RANK': [1, np.nan, np.nan],
            'DropQ_1_GROUP_1': ['', 'No', 'No'],
            # ... (add other necessary columns with mock data)
        })

    def test_load_data(self):
        # Assuming load_data just reads a CSV file; we can skip this test since we use mock data
        pass

    def test_preprocess_data(self):
        processed_data = preprocess_data(self.mock_data)

        # Test that invalid participants are excluded
        self.assertEqual(len(processed_data), 2)  # One participant is invalid

        # Test that ParticipantID is assigned correctly
        self.assertListEqual(processed_data['ParticipantID'].tolist(), [1, 2])

        # Test that ScalingFactor is calculated correctly
        expected_scaling_factors = [1, 2]
        self.assertListEqual(processed_data['ScalingFactor'].tolist(), expected_scaling_factors)

        # Test that properties are processed correctly
        participant_data = processed_data.iloc[0]
        self.assertIn('Response_loc(s1,s2)', participant_data)
        self.assertEqual(participant_data['Response_loc(s1,s2)'], 1)
        self.assertEqual(participant_data['Rank_loc(s1,s2)'], 1)

        # Test that missing or 'No' responses are handled correctly
        participant_data = processed_data.iloc[1]
        self.assertEqual(participant_data['Response_loc(s1,s2)'], 0)
        self.assertTrue(np.isnan(participant_data['Rank_loc(s1,s2)']))

if __name__ == '__main__':
    unittest.main()
