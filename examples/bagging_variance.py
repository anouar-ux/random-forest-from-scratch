"""
Why a forest beats its trees: high-variance trees vs. their majority vote.

Data: a dense Gaussian cluster (class 0) inside a noisy ring (class 1).
Each tree is grown very deep (max_depth=15) on its own bootstrap sample, so it
overfits in its own way. The vote of 50 such trees cancels most of that noise,
and the share of trees voting "class 1" gives a probability / uncertainty map.

    python examples/bagging_variance.py [--show]
"""
import numpy as np
from _common import finish, plt
from model import predict_tree, train_forest


def generate_complex_data(n_samples=800):
    """Dense central cluster (class 0) and peripheral ring (class 1)."""
    rng = np.random.default_rng(42)
    X0 = rng.normal(0, 1, (n_samples // 2, 2))
    r = rng.normal(4, 0.5, n_samples // 2)
    theta = rng.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    return X, y


if __name__ == "__main__":
    X, y = generate_complex_data(800)

    # Very deep trees, on purpose, to make every individual tree overfit.
    # num_features_per_split=2: with only 2 features the default (round(sqrt(2)) = 1)
    # would give each tree a single coordinate (see the README design note).
    print("Training the forest...")
    forest = train_forest(X, y, num_trees=50, max_depth=15, min_samples_split=2,
                          num_features_per_split=2, random_state=0)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    print("Predicting on the grid...")
    tree_preds = np.array([predict_tree(d["tree"], grid) for d in forest])  # (trees, points)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()
    scatter = dict(c=y, s=8, edgecolor="k", linewidths=0.2, cmap="coolwarm", alpha=0.7)

    for i in range(4):  # four individual trees
        axes[i].contourf(xx, yy, tree_preds[i].reshape(xx.shape), alpha=0.4, cmap="coolwarm")
        axes[i].scatter(X[:, 0], X[:, 1], **scatter)
        axes[i].set_title(f"Individual tree {i + 1} (high variance)")

    vote = tree_preds.mean(axis=0)  # share of trees predicting class 1
    axes[4].contourf(xx, yy, (vote > 0.5).astype(int).reshape(xx.shape), alpha=0.4, cmap="coolwarm")
    axes[4].scatter(X[:, 0], X[:, 1], **scatter)
    axes[4].set_title("Forest of 50 trees (majority vote)")

    # vmin/vmax fixed so that white means 50 %, i.e. maximal uncertainty
    cs = axes[5].contourf(xx, yy, vote.reshape(xx.shape), 20, alpha=0.85, cmap="coolwarm", vmin=0, vmax=1)
    axes[5].scatter(X[:, 0], X[:, 1], **scatter)
    axes[5].set_title("Share of trees voting class 1 (uncertainty)")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.colorbar(cs, ax=axes.tolist(), fraction=0.02, pad=0.02, label="P(class 1)")
    finish(fig, "bagging_variance.png")
