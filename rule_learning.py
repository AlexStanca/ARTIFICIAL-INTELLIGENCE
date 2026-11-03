"""Categorical ID3 induction and explicit IF–THEN rule inference.

Reference: Quinlan (1986), Induction of decision trees.
Adds depth/minimum-branch-size stopping; not C4.5 or post-pruning.
"""
from collections import Counter
from math import log2
from car_ai import DOMAINS, FEATURES, LABELS, validate_features


def entropy(rows):
    counts = Counter(row[-1] for row in rows)
    return -sum((n / len(rows)) * log2(n / len(rows)) for n in counts.values()) if rows else 0.0


def majority(rows):
    counts = Counter(row[-1] for row in rows)
    return max(LABELS, key=lambda label: counts[label])  # Stable ties.


def fit(rows, max_depth=6, min_leaf=1, attributes=None, depth=0):
    """Maximum information gain; ties favour original feature order."""
    if not rows:
        raise ValueError('Cannot train with no records.')
    attributes = tuple(range(6)) if attributes is None else attributes
    node = {'label': majority(rows), 'support': len(rows),
            'counts': dict(Counter(row[-1] for row in rows))}
    if depth >= max_depth or len(node['counts']) == 1 or not attributes:
        return node
    candidates = []
    for feature in attributes:
        groups = {value: [row for row in rows if row[feature] == value]
                  for value in sorted(DOMAINS[feature])}
        nonempty = [group for group in groups.values() if group]
        if len(nonempty) < 2 or any(len(group) < min_leaf for group in nonempty):
            continue
        gain = entropy(rows) - sum(len(group) / len(rows) * entropy(group) for group in nonempty)
        candidates.append((gain, feature, groups))
    if not candidates:
        return node
    gain, feature, groups = max(candidates, key=lambda item: (item[0], -item[1]))
    if gain <= 1e-12:
        return node
    node.update(feature=feature, gain=gain, branches={})
    for value, group in groups.items():
        # Absent but valid categories receive an explicit majority fallback.
        node['branches'][value] = (fit(group, max_depth, min_leaf,
            tuple(i for i in attributes if i != feature), depth + 1) if group else
            {'label': node['label'], 'support': 0, 'counts': {}, 'fallback': True})
    return node


def export_rules(tree):
    rules = []

    def walk(node, conditions):
        if 'feature' not in node:
            rules.append({'id': f'L{len(rules) + 1:03d}', 'conditions': conditions,
                          'label': node['label'], 'support': node['support'],
                          'counts': node['counts'], 'fallback': node.get('fallback', False)})
            return
        for value, child in sorted(node['branches'].items()):
            walk(child, conditions + [[node['feature'], value]])
    walk(tree, [])
    return rules


def predict(rules, values):
    """Deployed inference scans exported symbolic rules, not the tree."""
    validate_features(values)
    matches = [rule for rule in rules
               if all(values[index] == value for index, value in rule['conditions'])]
    if len(matches) != 1:
        raise ValueError(f'Rule coverage error: expected one match, got {len(matches)}.')
    rule = matches[0]
    conditions = ' AND '.join(f'{FEATURES[i]} = {v}' for i, v in rule['conditions']) or 'TRUE'
    return {'label': rule['label'], 'rule_id': rule['id'],
            'explanation': f"IF {conditions} THEN {rule['label']}",
            'training_support': rule['support'], 'fallback': rule['fallback']}
