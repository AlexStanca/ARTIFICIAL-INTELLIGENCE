"""Predict one car using the frozen symbolic rule model; does not retrain."""
import argparse
import json
from car_ai import ROOT, classify, validate_features
from rule_learning import predict


def compare(values):
    validate_features(values)
    path = ROOT / 'results' / 'model.json'
    if not path.exists():
        raise ValueError('Missing model. Run python evaluate.py to recreate the supplied model.')
    model = json.loads(path.read_text(encoding='utf-8'))
    learned = predict(model['rules'], values)
    label, reason = classify(values)
    return {'manual': {'label': label, 'explanation': reason}, 'learned': learned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('values', nargs=6, metavar='VALUE',
        help='Order: buying maint doors persons lug_boot safety')
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.values), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
