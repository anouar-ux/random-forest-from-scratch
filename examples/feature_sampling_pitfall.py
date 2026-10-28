"""
Pitfall: sampling features once per tree when there are only 2 features.

train_forest draws round(sqrt(d)) features per tree (random-subspace style).
With d = 2 that is a single feature, so every tree only sees theta OR
theta_dot and can only draw stripes. Same pendulum data, same forest size,
only num_features_per_split changes.

    python examples/feature_sampling_pitfall.py [--show]
"""
import numpy as np
from _common import finish, plt
from model import predict_tree, train_forest
from pendulum_basin import simulate_pendulum

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(low=[-np.pi / 1.5, -8.0], high=[np.pi / 1.5, 8.0], size=(1500, 2))
    y = np.array([simulate_pendulum(t, d) for t, d in X])
    xx, yy = np.meshgrid(np.linspace(-np.pi / 1.5, np.pi / 1.5, 80), np.linspace(-8.0, 8.0, 80))
    grid = np.c_[xx.ravel(), yy.ravel()]
    truth = np.array([simulate_pendulum(t, d) for t, d in grid])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, k in zip(axes, [1, 2]):
        forest = train_forest(X, y, num_trees=50, max_depth=12, num_features_per_split=k, random_state=0)
        proba = np.array([predict_tree(d["tree"], grid) for d in forest]).mean(axis=0)
        acc = np.mean((proba > 0.5) == truth)
        print(f"features per tree = {k}: agreement {acc:.1%}")
        ax.contourf(xx, yy, proba.reshape(xx.shape), levels=20, cmap="RdYlGn", vmin=0, vmax=1)
        ax.contour(xx, yy, truth.reshape(xx.shape), levels=[0.5], colors="k", linestyles="--", linewidths=1)
        label = "default: 1 feature per tree" if k == 1 else "2 features per tree"
        ax.set_title(f"{label}: {acc:.0%} agreement", fontsize=10)
        ax.set_xlabel(r"$\theta_0$ (rad)")
    axes[0].set_ylabel(r"$\dot\theta_0$ (rad/s)")
    finish(fig, "feature_sampling_pitfall.png")
