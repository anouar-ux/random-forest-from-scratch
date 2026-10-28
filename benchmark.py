"""
Accuracy of the from-scratch forest vs. scikit-learn on built-in datasets.
scikit-learn is used only to load the data and as a reference.

    python benchmark.py
"""
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from model import accuracy, build_tree, predict_forest, predict_tree, train_forest

N_TREES, MAX_DEPTH, SEED = 25, 6, 0


def evaluate(name, X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    t0 = time.perf_counter()
    forest = train_forest(Xtr, ytr, num_trees=N_TREES, max_depth=MAX_DEPTH, random_state=SEED)
    fit_s = time.perf_counter() - t0
    ours = accuracy(predict_forest(forest, Xte), yte)
    tree = accuracy(predict_tree(build_tree(Xtr, ytr, max_depth=MAX_DEPTH), Xte), yte)
    ref = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=SEED).fit(Xtr, ytr)
    sk = ref.score(Xte, yte)
    print(f"| {name} | {X.shape[0]} x {X.shape[1]} | {tree:.3f} | {ours:.3f} | {sk:.3f} | {fit_s:.1f} s |")


def main():
    print(f"Trees: {N_TREES}, max depth: {MAX_DEPTH}, 70/30 stratified split, seed {SEED}\n")
    print("| Dataset | Samples x features | Single tree (ours) | Forest (ours) | scikit-learn forest | Fit time (ours) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, loader in [("Breast cancer", load_breast_cancer), ("Wine (3 classes)", load_wine)]:
        d = loader()
        evaluate(name, d.data, d.target)
    Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=SEED)
    evaluate("Two moons (noisy)", Xm, ym)


if __name__ == "__main__":
    main()
