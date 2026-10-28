"""
Feature engineering on two interleaved spirals: Cartesian vs. polar inputs.

Trees split along one axis at a time, so a spiral boundary is very costly to
draw in (x, y). In polar coordinates (r, theta) the same problem becomes much
more regular. The script trains one forest on each representation, compares
their probability maps, then shows how the polar forest's decision regions
settle as trees are added (1, 10, 40, 100).

    python examples/spirals_feature_engineering.py [--show]
"""
import numpy as np
from _common import finish, plt
from model import predict_tree, train_forest


def generate_spirals(n_samples=1000, noise=0.7, seed=0):
    """Two interleaved spirals (class 0 and class 1)."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    t = np.linspace(1, 4 * np.pi, n)
    x0, y0 = t * np.cos(t) + rng.normal(0, noise, n), t * np.sin(t) + rng.normal(0, noise, n)
    x1, y1 = -t * np.cos(t) + rng.normal(0, noise, n), -t * np.sin(t) + rng.normal(0, noise, n)
    X = np.vstack([np.append(x0, x1), np.append(y0, y1)]).T
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def to_polar(P):
    return np.column_stack((np.hypot(P[:, 0], P[:, 1]), np.arctan2(P[:, 1], P[:, 0])))


def tree_votes(forest, features):
    return np.array([predict_tree(d["tree"], features) for d in forest])


if __name__ == "__main__":
    X, y = generate_spirals()
    X_test, y_test = generate_spirals(seed=1)          # fresh points from the same distribution
    params = dict(num_trees=100, max_depth=12, min_samples_split=2, num_features_per_split=2, random_state=0)

    print("Training on Cartesian features...")
    forest_xy = train_forest(X, y, **params)
    print("Training on polar features...")
    forest_polar = train_forest(to_polar(X), y, **params)

    acc_xy = np.mean((tree_votes(forest_xy, X_test).mean(0) > 0.5) == y_test)
    acc_polar = np.mean((tree_votes(forest_polar, to_polar(X_test)).mean(0) > 0.5) == y_test)
    print(f"Test accuracy  Cartesian: {acc_xy:.1%}  polar: {acc_polar:.1%}")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    grid = np.c_[xx.ravel(), yy.ravel()]
    print("Predicting on the grid...")
    votes_xy = tree_votes(forest_xy, grid)
    votes_polar = tree_votes(forest_polar, to_polar(grid))   # the grid is converted too

    # Figure 1: same model, two representations of the same data
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, votes, name, acc in [(axes[0], votes_xy, "Cartesian features (x, y)", acc_xy),
                                 (axes[1], votes_polar, r"Polar features (r, $\theta$)", acc_polar)]:
        cs = ax.contourf(xx, yy, votes.mean(0).reshape(xx.shape), levels=20, cmap="RdBu", alpha=0.85, vmin=0, vmax=1)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", s=8, edgecolor="k", linewidths=0.2)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", s=8, edgecolor="k", linewidths=0.2)
        ax.set_title(f"{name}: test accuracy {acc:.0%}")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(cs, ax=axes.tolist(), fraction=0.025, label="P(class 1)")
    finish(fig, "spirals_cartesian_vs_polar.png")

    # Figure 2: polar forest, decision regions as trees are added
    steps = [1, 10, 40, 100]
    fig, axes = plt.subplots(1, len(steps), figsize=(17, 4.4))
    for ax, k in zip(axes, steps):
        z = (votes_polar[:k].mean(0) > 0.5).astype(int).reshape(xx.shape)
        ax.contourf(xx, yy, z, alpha=0.4, cmap="viridis")
        ax.scatter(X[:, 0], X[:, 1], c=y, s=8, edgecolor="k", linewidths=0.2, cmap="viridis")
        ax.set_title(f"{k} tree{'s' if k > 1 else ''} (polar features)")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    finish(fig, "spirals_polar_n_trees.png")
