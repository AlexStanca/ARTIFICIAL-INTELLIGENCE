"""Tests of induction, unseen branches and exhaustive rule coverage."""
import itertools
import json
import unittest
from car_ai import ROOT, DOMAINS, load_data
from evaluate import internal_split
from rule_learning import entropy, fit, export_rules, predict


class LearningTests(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(entropy([('unacc',)] * 4), 0.0)
        self.assertAlmostEqual(entropy([('unacc',), ('acc',)]), 1.0)

    def test_learns_simple_attribute(self):
        rows = [('low','low','4','4','big','low','unacc'),
                ('low','low','4','4','big','high','vgood')]
        rules = export_rules(fit(rows))
        for row in rows:
            self.assertEqual(predict(rules, row[:6])['label'], row[-1])
        result = predict(rules, ('low','low','4','4','big','med'))
        self.assertTrue(result['fallback'])
        self.assertEqual(result['training_support'], 0)

    def test_depth_zero_is_majority(self):
        rows = [('low','low','4','4','big','low','unacc'),
                ('low','low','4','4','big','high','vgood')]
        rules = export_rules(fit(rows, max_depth=0))
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]['label'], 'unacc')

    def test_empty_training_rejected(self):
        with self.assertRaises(ValueError):
            fit([])

    def test_every_valid_combination_has_one_rule(self):
        # Coverage is structural: do not look at target labels here.
        rows = load_data(ROOT / 'data' / 'car.data')
        split = json.loads((ROOT / 'data' / 'split_indices.json').read_text())
        rules = export_rules(fit([rows[i] for i in split['development']]))
        count = 0
        for values in itertools.product(*[sorted(d) for d in DOMAINS]):
            predict(rules, values)  # Raises on zero/multiple matches.
            count += 1
        self.assertEqual(count, 1728)

    def test_validation_never_contains_test_indices(self):
        rows = load_data(ROOT / 'data' / 'car.data')
        split = json.loads((ROOT / 'data' / 'split_indices.json').read_text())
        train, validation = internal_split(split['development'], rows)
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train + validation) & set(split['reserved_test']))
        self.assertEqual(sorted(train + validation), split['development'])


if __name__ == '__main__':
    unittest.main()
