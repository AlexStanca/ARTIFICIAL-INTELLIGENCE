"""Select on validation, freeze, then evaluate the reserved test set.

Rerunning unchanged code reproduces one experiment, not independent tests.
Do not tune rules or hyperparameters after reading final test scores.
"""
import csv
import hashlib
import json
import platform
import random
from collections import Counter
from car_ai import ROOT, FEATURES, LABELS, classify, load_data, metrics
from rule_learning import fit, export_rules, predict


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def internal_split(indices, rows):
    rng = random.Random(17)
    train, validation = [], []
    for label in LABELS:
        group = [i for i in indices if rows[i][-1] == label]
        rng.shuffle(group)
        count = max(1, round(len(group) * 0.25))
        validation.extend(group[:count])
        train.extend(group[count:])
    return sorted(train), sorted(validation)


def score(rules, rows):
    return metrics([r[-1] for r in rows], [predict(rules, r[:6])['label'] for r in rows])


def main():
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    raw = ROOT / 'data' / 'car.data'
    rows = load_data(raw)
    split = json.loads((ROOT / 'data' / 'split_indices.json').read_text())
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if digest != split['dataset_sha256']:
        raise ValueError('Dataset changed: recorded allocation no longer valid.')
    dev, test = split['development'], split['reserved_test']
    if set(dev) & set(test) or sorted(dev + test) != list(range(len(rows))):
        raise ValueError('Split is not a disjoint complete partition.')
    train, validation = internal_split(dev, rows)
    training_rows = [rows[i] for i in train]
    validation_rows = [rows[i] for i in validation]
    candidates = []
    # Fixed before testing: eight candidates, chosen by validation macro F1.
    for depth in (3, 4, 5, 6):
        for minimum in (1, 3):
            rules = export_rules(fit(training_rows, depth, minimum))
            result = score(rules, validation_rows)
            candidates.append({'max_depth': depth, 'min_leaf': minimum,
                               'rule_count': len(rules), 'validation': result})
    chosen = max(candidates, key=lambda c: (c['validation']['macro_f1'],
                                           -c['rule_count'], -c['max_depth'], c['min_leaf']))
    development_rows = [rows[i] for i in dev]
    final_rules = export_rules(fit(development_rows, chosen['max_depth'], chosen['min_leaf']))
    model = {'algorithm': 'Categorical ID3 exported as mutually exclusive IF–THEN rules',
             'max_depth': chosen['max_depth'], 'min_leaf': chosen['min_leaf'],
             'dataset_sha256': digest, 'trained_on': 'development indices only',
             'rules': final_rules}
    # Save selected settings/model BEFORE inspecting the reserved evaluation.
    dump(output / 'selection.json', {'seed': 17, 'train_indices': train,
          'validation_indices': validation, 'candidates': candidates, 'chosen': chosen,
          'limitation': 'Initial manual rules had already been evaluated on all development rows; '
          'internal validation is for the new learner, not an independent audit of those rules.'})
    dump(output / 'model.json', model)
    model_digest = hashlib.sha256((output / 'model.json').read_bytes()).hexdigest()
    (output / 'rules.txt').write_text('\n'.join(
        f"{rule['id']}: IF " + (' AND '.join(f'{FEATURES[i]} = {v}' for i, v in rule['conditions']) or 'TRUE')
        + f" THEN {rule['label']} [training support={rule['support']}; fallback={rule['fallback']}]"
        for rule in final_rules), encoding='utf-8')
    majority_label = Counter(r[-1] for r in development_rows).most_common(1)[0][0]
    actual = [rows[i][-1] for i in test]
    learned = [predict(final_rules, rows[i][:6]) for i in test]
    manual = [classify(rows[i][:6]) for i in test]
    report = {'stage': 'FINAL RESERVED TEST — no tuning after this evaluation',
              'python_version': platform.python_version(), 'dataset_sha256': digest,
              'model_sha256': model_digest, 'total_rows': len(rows),
              'development_size': len(dev), 'selection_train_size': len(train),
              'validation_size': len(validation), 'test_size': len(test),
              'class_counts': dict(Counter(r[-1] for r in rows)),
              'selected_parameters': {k: chosen[k] for k in ('max_depth', 'min_leaf')},
              'learned_rule_count': len(final_rules),
              'max_rule_conditions': max(len(r['conditions']) for r in final_rules),
              'fallback_rule_count': sum(r['fallback'] for r in final_rules),
              'test_fallback_hits': sum(r['fallback'] for r in learned),
              'majority_baseline': metrics(actual, [majority_label] * len(test)),
              'manual_rules': metrics(actual, [r[0] for r in manual]),
              'learned_rules': metrics(actual, [r['label'] for r in learned])}
    dump(output / 'final_metrics.json', report)
    with (output / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row_index', *FEATURES, 'actual', 'manual_prediction',
                         'manual_reason', 'learned_prediction', 'learned_reason', 'fallback'])
        for i, m, l in zip(test, manual, learned):
            writer.writerow([i, *rows[i], *m, l['label'], l['explanation'], l['fallback']])
    print('Frozen parameters:', report['selected_parameters'])
    print('Rule count:', len(final_rules))
    for name in ('majority_baseline', 'manual_rules', 'learned_rules'):
        result = report[name]
        print(f"{name}: accuracy={result['accuracy']:.4f}, macro F1={result['macro_f1']:.4f}, "
              f"balanced accuracy={result['balanced_accuracy']:.4f}")


if __name__ == '__main__':
    main()
