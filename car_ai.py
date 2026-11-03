"""Shared validation, manual rules and evaluation metrics.

The six manual rules are explicit policy hypotheses, not recovered DEX rules.
Use app.py or predict.py for inference and evaluate.py for final reproduction.
The original split helper is retained to verify the saved allocation.
"""

import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety')
LABELS = ('unacc', 'acc', 'good', 'vgood')
DOMAINS = (
    {'vhigh', 'high', 'med', 'low'}, {'vhigh', 'high', 'med', 'low'},
    {'2', '3', '4', '5more'}, {'2', '4', 'more'},
    {'small', 'med', 'big'}, {'low', 'med', 'high'},
)


def validate_features(values):
    """Reject unknown categories instead of quietly making a prediction."""
    if len(values) != 6:
        raise ValueError('Expected exactly six input attributes.')
    for field, value, allowed in zip(FEATURES, values, DOMAINS):
        if value not in allowed:
            raise ValueError(f'Invalid {field}: {value!r}; expected {sorted(allowed)}')


def load_data(path):
    rows = []
    with path.open(newline='', encoding='utf-8') as source:
        for line_number, row in enumerate(csv.reader(source), 1):
            if len(row) != 7:
                raise ValueError(f'Row {line_number}: expected seven columns.')
            validate_features(row[:6])
            if row[6] not in LABELS:
                raise ValueError(f'Row {line_number}: invalid target.')
            rows.append(tuple(row))
    if not rows:
        raise ValueError('Dataset is empty.')
    if len({row[:6] for row in rows}) != len(rows):
        raise ValueError('Repeated feature combinations need investigation.')
    return rows


def classify(values):
    """First matching rule wins; return both label and explanation.

    Rules R1/R2 encode the initial minimum suitability policy. Later rules
    are provisional hypotheses about costs and features. No target value
    is accepted by this function. Doors are deliberately unused in v1.
    """
    validate_features(values)
    buying, maint, doors, persons, luggage, safety = values
    if safety == 'low':
        return 'unacc', 'R1: low safety fails the minimum suitability policy'
    if persons == '2':
        return 'unacc', 'R2: two-person capacity fails the minimum capacity policy'
    if buying == 'vhigh' and maint in {'high', 'vhigh'}:
        return 'unacc', 'R3: very high purchase cost with high maintenance cost'
    if (buying == 'low' and maint in {'low', 'med'}
            and safety == 'high' and luggage == 'big'):
        return 'vgood', 'R4: low purchase cost, manageable upkeep, high safety, large boot'
    if (buying in {'low', 'med'} and maint in {'low', 'med'}
            and safety == 'high'):
        return 'good', 'R5: moderate or low costs with high safety'
    return 'acc', 'R6: default after exclusions and higher-category rules'


def split_data(rows, seed=42):
    """Reserve approximately 20% of each class with fixed randomisation.

    Stratification uses labels to preserve class balance, not to tune rules.
    Save indices so the allocation is auditable and exactly reproducible.
    """
    rng = random.Random(seed)
    development, reserved = [], []
    for label in LABELS:
        indices = [i for i, row in enumerate(rows) if row[-1] == label]
        rng.shuffle(indices)
        count = max(1, round(len(indices) * 0.2))
        reserved.extend(indices[:count])
        development.extend(indices[count:])
    return sorted(development), sorted(reserved)


def metrics(actual, predicted):
    """Macro F1 weights each class equally; absent predictions score zero."""
    if not actual or len(actual) != len(predicted):
        raise ValueError('Expected equal non-empty actual and predicted lists.')
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for truth, guess in zip(actual, predicted):
        matrix[LABELS.index(truth)][LABELS.index(guess)] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {
        'accuracy': sum(matrix[i][i] for i in range(4)) / len(actual),
        'macro_f1': sum(v['f1'] for v in per_class.values()) / 4,
        'balanced_accuracy': sum(v['recall'] for v in per_class.values()) / 4,
        'per_class': per_class, 'labels': LABELS,
        'confusion_matrix_rows_actual_columns_predicted': matrix,
    }
