import unittest
from models import build_tree, deterministic_bfs, deterministic_dfs, probabilistic_bfs, probabilistic_dfs

class TestModels(unittest.TestCase):
    def setUp(self):
        self.scenes = ['Drop', 'Push', 'Pull']
        self.cognitive_resource = 1.0
        self.expansion_prob = 0.5
        self.effective_time_limit = 5  # For simplicity

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

if __name__ == '__main__':
    unittest.main()
