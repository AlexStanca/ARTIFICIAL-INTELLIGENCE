"""Release checks for the frozen model and UI behaviour without a display.

These exercise real event-handler logic with lightweight widget substitutes;
they are not graphical layout tests. Run with the other unittest modules.
"""
import hashlib
import itertools
import json
import unittest
from unittest.mock import Mock, patch

try:
    from app import CarApp
except ImportError:  # Tkinter is optional for the CLI and core engine.
    CarApp = None
from car_ai import ROOT, DOMAINS, LABELS
from predict import compare
from presentation import TOKENS, display_value, display_conditions
from rule_learning import predict


class ReleaseTests(unittest.TestCase):
    def test_frozen_model_matches_reported_hash(self):
        recorded = json.loads((ROOT / 'results/final_metrics.json').read_text())
        digest = hashlib.sha256((ROOT / 'results/model.json').read_bytes()).hexdigest()
        self.assertEqual(digest, recorded['model_sha256'])

    def test_final_rules_cover_all_valid_inputs(self):
        model = json.loads((ROOT / 'results/model.json').read_text())
        for values in itertools.product(*[sorted(d) for d in DOMAINS]):
            with self.subTest(values=values):
                outcome = predict(model['rules'], values)
                self.assertIn(outcome['label'], LABELS)

    def test_display_values_round_trip_without_ambiguity(self):
        for tokens in TOKENS:
            names = [display_value(t) for t in tokens]
            self.assertEqual(len(names), len(set(names)))
            for token in tokens:
                self.assertEqual(tokens[names.index(display_value(token))], token)
        self.assertEqual(display_conditions({'conditions': [[3, 'more']]}),
                         ['Passenger capacity: More than 4'])

    @unittest.skipIf(CarApp is None, 'Tkinter is not installed')
    def test_live_evaluation_handler_matches_engine(self):
        # Construct the controller without creating a graphical Tk window.
        controller = CarApp.__new__(CarApp)
        model = json.loads((ROOT / 'results/model.json').read_text())
        controller.rules = {r['id']: r for r in model['rules']}
        controller.manual_card = Mock()
        controller.learned_card = Mock()
        controller.agreement = Mock()
        for values in [('low', 'low', '4', '4', 'big', 'high'),
                       ('vhigh', 'vhigh', '4', 'more', 'big', 'high'),
                       ('low', 'low', '4', '4', 'big', 'low')]:
            controller.values = [Mock(get=Mock(return_value=display_value(t))) for t in values]
            expected = compare(values)
            with patch('app.messagebox.showerror') as error:
                controller.evaluate()
                error.assert_not_called()
            self.assertEqual(controller.manual_card.set.call_args.args[0], expected['manual']['label'])
            self.assertEqual(controller.learned_card.set.call_args.args[0], expected['learned']['label'])
            self.assertEqual(controller.matched_id, expected['learned']['rule_id'])
            self.assertIn(expected['learned']['explanation'], controller.last_summary)

    def test_missing_model_has_actionable_error(self):
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Missing model'):
                compare(('low', 'low', '4', '4', 'big', 'high'))

    def test_duplicate_rules_fail_instead_of_silent_conflict(self):
        rule = {'id': 'duplicate', 'conditions': [], 'label': 'unacc',
                'support': 1, 'fallback': False}
        with self.assertRaisesRegex(ValueError, 'expected one match, got 2'):
            predict([rule, rule], ('low', 'low', '4', '4', 'big', 'high'))


if __name__ == '__main__':
    unittest.main()
