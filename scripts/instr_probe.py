"""Linear probe of the side (left/right) from standardised instruction embeddings.

Within each modality: logistic regression, 5-fold cross-validation grouped by episode (all variants
of an episode fall in the same fold).  Across modalities: train on every embedding of one modality,
test on every embedding of another.  Prints a Markdown table (rows: train modality, columns: test
modality; diagonal = cross-validated accuracy) and optionally writes the numbers as JSON.
Standardisation statistics are computed on the whole bundle (label-free), as the policy does.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leap.steer.bundle import INSTR_DIM, MODALITIES, load_embeddings, standardize_stats


class NumpyLogisticRegression:
    """Small L2-regularised logistic regression (gradient descent) used when sklearn is unavailable."""

    def __init__(self, l2=1e-2, steps=500, lr=0.1):
        self.l2, self.steps, self.lr = l2, steps, lr
        self.weight = None
        self.bias = 0.0

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.weight = np.zeros(x.shape[1])
        self.bias = 0.0
        for _ in range(self.steps):
            logits = x @ self.weight + self.bias
            prob = 1.0 / (1.0 + np.exp(-logits))
            gradient = x.T @ (prob - y) / len(y) + self.l2 * self.weight
            self.weight -= self.lr * gradient
            self.bias -= self.lr * float((prob - y).mean())
        return self

    def predict(self, x):
        return (np.asarray(x, dtype=np.float64) @ self.weight + self.bias > 0).astype(np.int64)


def make_classifier():
    try:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=5000, C=1.0)
    except ImportError:
        return NumpyLogisticRegression()


def flatten(array, side):
    """[E, V, D] -> rows [E*V, D], labels [E*V], episode ids [E*V]."""
    episodes, variants, dim = array.shape
    rows = array.reshape(-1, dim)
    labels = np.repeat(np.asarray(side), variants)
    groups = np.repeat(np.arange(episodes), variants)
    return rows, labels, groups


def cross_validate(rows, labels, groups, folds, seed):
    """Accuracy over folds that split episodes (groups), not rows."""
    episodes = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(episodes))
    fold_of_episode = np.empty(len(episodes), dtype=np.int64)
    fold_of_episode[order] = np.arange(len(episodes)) % folds
    correct = 0
    for fold in range(folds):
        test = fold_of_episode[groups] == fold
        if not test.any() or test.all():
            continue
        if len(np.unique(labels[~test])) < 2:
            continue
        model = make_classifier().fit(rows[~test], labels[~test])
        correct += int((model.predict(rows[test]) == labels[test]).sum())
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=None, help="Write the accuracies here")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    embeddings = load_embeddings(args.bundle)
    side = np.asarray(embeddings["side"])
    present = [m for m in MODALITIES if m in embeddings]
    mean, std = standardize_stats(embeddings)
    data = {m: flatten(((np.asarray(embeddings[m]) - mean) / std).astype(np.float32), side) for m in present}

    results = {"episodes": int(len(side)), "left": int((side == 0).sum()), "right": int((side == 1).sum()),
               "folds": args.folds, "accuracy": {}}
    for train in present:
        results["accuracy"][train] = {}
        rows, labels, groups = data[train]
        for test in present:
            if train == test:
                accuracy = cross_validate(rows, labels, groups, args.folds, args.seed)
            else:
                model = make_classifier().fit(rows, labels)
                accuracy = float((model.predict(data[test][0]) == data[test][1]).mean())
            results["accuracy"][train][test] = float(accuracy)

    chance = max(results["left"], results["right"]) / max(results["episodes"], 1)
    print(f"Side probe on {args.bundle} ({results['episodes']} episodes, left {results['left']}, "
          f"right {results['right']}, majority baseline {100 * chance:.0f} %, dim {INSTR_DIM})")
    print("| train \\ test | " + " | ".join(present) + " |")
    print("|---|" + "---|" * len(present))
    for train in present:
        print(f"| {train} | " + " | ".join(f"{100 * results['accuracy'][train][test]:.1f}" for test in present) + " |")
    print(f"(diagonal: {args.folds}-fold cross-validation by episode; off-diagonal: train on row, test on column)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
