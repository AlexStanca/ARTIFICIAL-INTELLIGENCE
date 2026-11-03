"""Behavioural checks for rule priority, input handling and evaluation maths."""
import unittest
from car_ai import classify, load_data, metrics, split_data, ROOT


class CarAITests(unittest.TestCase):
    def test_safety_overrides_attractive_features(self):
        self.assertEqual(classify(('low','low','4','4','big','low'))[0], 'unacc')

    def test_capacity_overrides_attractive_features(self):
        self.assertEqual(classify(('low','low','4','2','big','high'))[0], 'unacc')

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            classify(('cheap','low','4','4','big','high'))
        with self.assertRaises(ValueError):
            classify(('low',))

    def test_metrics_against_hand_calculation(self):
        result = metrics(['unacc','acc','good','vgood'], ['unacc'] * 4)
        self.assertEqual(result['accuracy'], 0.25)
        self.assertAlmostEqual(result['macro_f1'], 0.1)
        self.assertEqual(result['balanced_accuracy'], 0.25)

    def test_perfect_predictions(self):
        labels = ['unacc','acc','good','vgood']
        self.assertEqual(metrics(labels, labels)['macro_f1'], 1.0)

    def test_split_is_disjoint_complete_and_reproducible(self):
        rows = load_data(ROOT / 'data' / 'car.data')
        dev, test = split_data(rows)
        self.assertFalse(set(dev) & set(test))
        self.assertEqual(set(dev) | set(test), set(range(len(rows))))
        self.assertEqual((dev, test), split_data(rows))
        self.assertEqual({rows[i][-1] for i in test}, {'unacc','acc','good','vgood'})


if __name__ == '__main__':
    unittest.main()
