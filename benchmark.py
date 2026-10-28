"""
Benchmark and visualisations for the from-scratch random forest (model.py).

    python benchmark.py

1. Accuracy of the from-scratch forest vs. scikit-learn on built-in datasets
   (scikit-learn is used only as a reference and for the datasets).
2. Decision regions of a single tree vs. the forest on 2-D toy problems
   -> figures/decision_regions.png
3. Test accuracy as a function of the number of trees
   -> figures/accuracy_vs_trees.png
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer, load_wine, make_circles, make_moons
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


def plot_regions(ax, model_predict, X, y, title):
    pad = 0.4
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220),
    )
    zz = model_predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(["#c6dbef", "#fdd0a2"]), alpha=0.9)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#2171b5", "#d94801"]), s=12, edgecolors="white", linewidths=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    print(f"Trees: {N_TREES}, max depth: {MAX_DEPTH}, 70/30 stratified split, seed {SEED}\n")
    print("| Dataset | Samples x features | Single tree (ours) | Forest (ours) | scikit-learn forest | Fit time (ours) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, loader in [("Breast cancer", load_breast_cancer), ("Wine (3 classes)", load_wine)]:
        d = loader()
        evaluate(name, d.data, d.target)
    Xm, ym = make_moons(n_samples=400, noise=0.3, random_state=SEED)
    evaluate("Two moons (noisy)", Xm, ym)

    # Decision regions. With only 2 features, sqrt(2) rounds to 1 feature per tree,
    # so the forest is trained with both features per tree to show a meaningful picture.
    problems = [("Two moons", make_moons(n_samples=300, noise=0.25, random_state=SEED)),
                ("Circles", make_circles(n_samples=300, noise=0.12, factor=0.5, random_state=SEED))]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.2))
    for row, (name, (X, y)) in enumerate(problems):
        tree = build_tree(X, y, max_depth=MAX_DEPTH)
        forest = train_forest(X, y, num_trees=N_TREES, max_depth=MAX_DEPTH, num_features_per_split=2, random_state=SEED)
        plot_regions(axes[row, 0], lambda Z, t=tree: predict_tree(t, Z), X, y, f"{name}: single tree (depth {MAX_DEPTH})")
        plot_regions(axes[row, 1], lambda Z, f=forest: predict_forest(f, Z), X, y, f"{name}: forest of {N_TREES} bagged trees")
    fig.suptitle("Bagging removes the spurious splits of a single overfitted tree", fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/decision_regions.png", dpi=150)

    # Accuracy vs number of trees (breast cancer)
    d = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.3, random_state=SEED, stratify=d.target)
    counts = [1, 2, 5, 10, 15, 25, 40]
    accs = [accuracy(predict_forest(train_forest(Xtr, ytr, num_trees=k, max_depth=MAX_DEPTH, random_state=SEED), Xte), yte) for k in counts]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(counts, accs, marker="o", color="#2171b5")
    ax.set_xlabel("Number of trees"); ax.set_ylabel("Test accuracy")
    ax.set_title("Breast cancer dataset: accuracy vs. forest size", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/accuracy_vs_trees.png", dpi=150)
    print("\nAccuracy vs trees:", dict(zip(counts, [round(float(a), 3) for a in accs])))


if __name__ == "__main__":
    main()
