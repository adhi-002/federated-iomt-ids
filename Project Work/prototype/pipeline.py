"""Small FedAvg simulation. Synthetic mode demonstrates plumbing, not IoMT findings."""
from pathlib import Path
import argparse
import copy
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def aggregate(models, counts):
    if not models or len(models) != len(counts) or min(counts) <= 0:
        raise ValueError('Aggregation requires a positive sample count for every model.')
    weights = np.asarray(counts) / sum(counts)
    result = copy.deepcopy(models[0])
    for attr in ('coefs_', 'intercepts_'):
        layers = [getattr(m, attr) for m in models]
        setattr(result, attr, [sum(w * p[i] for w, p in zip(weights, layers))
                              for i in range(len(layers[0]))])
    return result


def partition(y, seed=42, alpha=0.5, clients=3):
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(clients)]
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        rng.shuffle(ids)
        chunks = np.split(ids, np.cumsum(rng.multinomial(len(ids), rng.dirichlet([alpha] * clients)))[:-1])
        for part, chunk in zip(parts, chunks):
            part.extend(chunk.tolist())
    if any(len(p) == 0 for p in parts):
        raise ValueError('Empty client; use more records or a less extreme alpha.')
    return [np.asarray(p) for p in parts]


def calibrate(confidence, false_unknown_budget=0.05):
    return float(np.quantile(confidence, false_unknown_budget, method='lower'))


def prepare(args):
    if not args.train_csv:
        x, y = make_classification(n_samples=6000, n_features=20, n_informative=12,
                                  n_redundant=4, n_classes=4, n_clusters_per_class=1,
                                  class_sep=1.3, random_state=args.seed)
        frame = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        frame['label'] = np.array(['benign_demo', 'attack_A_demo', 'attack_B_demo', 'withheld_demo'])[y]
        train, test = train_test_split(frame, test_size=.25, stratify=frame['label'], random_state=args.seed)
        label, unknown = 'label', 'withheld_demo'
        source = 'SYNTHETIC DEMONSTRATION — not CICIoMT2024 research results'
    else:
        if not args.test_csv or not args.unknown:
            raise ValueError('Real CSV mode requires --test-csv and --unknown.')
        train, test = pd.read_csv(args.train_csv), pd.read_csv(args.test_csv)
        label, unknown = args.label, args.unknown
        source = 'USER CSV DATA — verify provenance and original capture splits before research claims'
    for name, frame in [('train', train), ('test', test)]:
        if label not in frame or frame[label].isna().any():
            raise ValueError(f'{name}: missing label column or missing labels: {label}')
        frame[label] = frame[label].astype(str)
    train = train[train[label] != unknown].copy()
    classes = sorted(train[label].unique())
    if len(classes) < 2 or unknown not in set(test[label]):
        raise ValueError('Need at least two known classes and the withheld class in test data.')
    unexpected = set(test[label]) - set(classes) - {unknown}
    if unexpected:
        raise ValueError(f'Test contains additional unseen labels: {unexpected}; explicitly select classes first.')
    features = [c for c in train.select_dtypes(include='number').columns if c != label]
    if not features or not set(features).issubset(test.columns):
        raise ValueError('Need matching numeric feature columns in train and test.')
    # Conservative safeguard: identical feature records may not cross train/test.
    train_hash = set(pd.util.hash_pandas_object(train[features], index=False))
    overlap = pd.util.hash_pandas_object(test[features], index=False).isin(train_hash)
    removed_overlap = int(overlap.sum())
    test = test.loc[~overlap].copy()
    if unknown not in set(test[label]) or not test[label].isin(classes).any():
        raise ValueError('Deduplication left no known or unknown test examples.')
    train, val = train_test_split(train, test_size=.2, stratify=train[label], random_state=args.seed)
    # Remove exact validation duplicates from training before fitting preprocessing.
    val_hash = set(pd.util.hash_pandas_object(val[features], index=False))
    train = train.loc[~pd.util.hash_pandas_object(train[features], index=False).isin(val_hash)].copy()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    clean = lambda f: f[features].replace([np.inf, -np.inf], np.nan)
    xtrain = scaler.fit_transform(imputer.fit_transform(clean(train)))
    xval = scaler.transform(imputer.transform(clean(val)))
    xtest = scaler.transform(imputer.transform(clean(test)))
    mapping = {c: i for i, c in enumerate(classes)}
    ytrain = train[label].map(mapping).to_numpy()
    yval = val[label].map(mapping).to_numpy()
    return xtrain, ytrain, xval, yval, xtest, test[label].to_numpy(), classes, unknown, features, imputer, scaler, source, removed_overlap


def run(args):
    start = time.perf_counter()
    x, y, xv, yv, xt, yt, classes, unknown, features, imputer, scaler, source, removed = prepare(args)
    parts = partition(y, args.seed, args.alpha)
    # Initialize shared parameters; the one-record initialization update is discarded.
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='sgd',
                          learning_rate_init=.01, momentum=0, alpha=.0001,
                          batch_size=64, random_state=args.seed)
    model.partial_fit(x[:1], y[:1], classes=np.arange(len(classes)))
    rng = np.random.default_rng(args.seed)
    model.coefs_ = [rng.uniform(-np.sqrt(6/(w.shape[0]+w.shape[1])), np.sqrt(6/(w.shape[0]+w.shape[1])), w.shape) for w in model.coefs_]
    model.intercepts_ = [np.zeros_like(b) for b in model.intercepts_]
    history = []
    client_info = [{ 'hospital': chr(65+i), 'samples': len(ids),
                     'aggregation_weight': len(ids)/len(y),
                     'classes': {c: int(sum(y[ids] == j)) for j,c in enumerate(classes)}} for i,ids in enumerate(parts)]
    initial = float(f1_score(yv, model.predict(xv), average='macro', zero_division=0))
    for round_id in range(1, args.rounds + 1):
        locals_ = []
        for i, ids in enumerate(parts):
            local = copy.deepcopy(model)
            local.random_state = args.seed + round_id * 10 + i
            for _ in range(args.local_epochs):
                local.partial_fit(x[ids], y[ids])
            locals_.append(local)
        model = aggregate(locals_, [len(ids) for ids in parts])
        score = float(f1_score(yv, model.predict(xv), average='macro', zero_division=0))
        history.append({'round': round_id, 'known_validation_macro_f1': score})
        print(f'Round {round_id}: known validation macro-F1 = {score:.3f}', flush=True)
    threshold = calibrate(model.predict_proba(xv).max(axis=1), args.false_unknown_budget)
    probabilities = model.predict_proba(xt)
    confidence = probabilities.max(axis=1)
    reject = confidence < threshold
    actual_unknown = yt == unknown
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    known = ~actual_unknown
    metrics = {
        'known_class_macro_f1_before_rejection': float(f1_score(yt[known], predictions[known], labels=classes, average='macro', zero_division=0)),
        'known_class_macro_f1_after_rejection': float(f1_score(yt[known], np.where(reject[known], 'UNKNOWN', predictions[known]), labels=classes, average='macro', zero_division=0)),
        'unknown_recall': float(reject[actual_unknown].mean()),
        'false_unknown_rate_on_known': float(reject[known].mean()),
        'unknown_auroc': float(roc_auc_score(actual_unknown, 1-confidence)),
    }
    benign = np.array(['benign' in str(v).lower() or 'normal' in str(v).lower() for v in yt])
    if benign.any(): metrics['false_unknown_rate_on_benign'] = float(reject[benign].mean())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sample = {'hospital': 'A', 'round': args.rounds, 'training_sample_count': len(parts[0]),
              'note': 'Actual local parameters from final round; first few entries shown only.',
              'arrays': [{'name': f'layer_{i+1}_weights', 'shape': list(w.shape), 'first_values': w.ravel()[:8].tolist()} for i,w in enumerate(locals_[0].coefs_)],
              'biases': [{'shape': list(b.shape), 'first_values': b[:8].tolist()} for b in locals_[0].intercepts_]}
    report = {'source': source, 'config': vars(args), 'known_classes': classes, 'withheld_class': unknown,
              'features': features, 'clients': client_info, 'training_samples': len(y), 'validation_samples': len(yv),
              'known_test_samples': int(known.sum()), 'unknown_test_samples': int(actual_unknown.sum()),
              'removed_train_test_duplicates': removed, 'initial_validation_macro_f1': initial,
              'threshold': threshold, 'validation_false_unknown_rate': float((model.predict_proba(xv).max(axis=1)<threshold).mean()),
              'metrics': metrics, 'history': history, 'runtime_seconds': time.perf_counter()-start,
              'parameter_count': int(sum(w.size for w in model.coefs_+model.intercepts_)), 'payload_example': sample}
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame({'true_label':yt, 'mlp_prediction':predictions, 'confidence':confidence,
                  'decision':np.where(reject, 'UNKNOWN', predictions)}).to_csv(out/'predictions.csv', index=False)
    joblib.dump({'model':model, 'imputer':imputer, 'scaler':scaler, 'features':features,
                 'classes':classes, 'threshold':threshold}, out/'checkpoint.joblib')
    print(json.dumps(metrics, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-csv'); p.add_argument('--test-csv')
    p.add_argument('--label', default='label'); p.add_argument('--unknown')
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--false-unknown-budget', type=float, default=.05)
    p.add_argument('--output', default='results/synthetic')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.rounds < 1 or args.local_epochs < 1 or args.alpha <= 0 or not 0 < args.false_unknown_budget < 1:
        raise ValueError('Rounds, epochs and alpha must be positive; false unknown budget must lie between 0 and 1.')
    run(args)
