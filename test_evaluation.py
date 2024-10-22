import unittest
import numpy as np
from evaluation import compute_likelihood, compute_aic

class TestEvaluation(unittest.TestCase):
    def test_compute_likelihood(self):
        # Example data
        visited = ['Drop', 'loc(s1,s2)', 'size(s1)', 'traj(s1)']
        responses = {'loc(s1,s2)': 1, 'size(s1)': 0, 'traj(s1)': 1, 'person': 1}
        ranks = {'loc(s1,s2)': 1, 'size(s1)': np.nan, 'traj(s1)': 2, 'person': 3}

        log_likelihood = compute_likelihood(visited, responses, ranks)
        self.assertIsInstance(log_likelihood, float)

    def test_compute_aic(self):
        log_likelihood = -10.0
        k = 2
        aic = compute_aic(log_likelihood, k)
        expected_aic = 2 * k - 2 * log_likelihood
        self.assertEqual(aic, expected_aic)

if __name__ == '__main__':
    unittest.main()
