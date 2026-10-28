"""
Learning the basin of attraction of a controlled inverted pendulum.

An inverted pendulum (g/L = 9.81 s^-2) is stabilised by a PD controller whose
command is saturated (|u| <= 6 rad/s^2, less than gravity at large angles).
Whether the controller recovers the pendulum depends nonlinearly on the
initial state (theta_0, theta_dot_0). We simulate 1,500 random initial
conditions, label each one stable / unstable, and let the random forest learn
the stability region from these labels alone.

The dashed black line is the true boundary, obtained by simulating every point
of the grid, so the learned map can be checked against the physics.

    python examples/pendulum_basin.py [--show]
"""
import numpy as np
from _common import finish, plt
from model import predict_tree, train_forest


def simulate_pendulum(theta_0, theta_dot_0, steps=60, dt=0.05):
    """1 if the saturated PD controller brings the pendulum back near upright within 3 s, else 0."""
    g, L = 9.81, 1.0
    Kp, Kd = 15.0, 5.0
    max_torque = 6.0  # actuator saturation (angular acceleration units, unit inertia)

    theta, theta_dot = theta_0, theta_dot_0
    for _ in range(steps):
        torque = np.clip(-Kp * theta - Kd * theta_dot, -max_torque, max_torque)
        theta_ddot = (g / L) * np.sin(theta) + torque   # upright position is unstable
        theta_dot += theta_ddot * dt                     # semi-implicit Euler
        theta += theta_dot * dt
        if abs(theta) > np.pi / 2:                       # the pendulum has fallen
            return 0
    return int(abs(theta) < 0.5 and abs(theta_dot) < 1.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Simulating 1,500 random initial conditions...")
    X = rng.uniform(low=[-np.pi / 1.5, -8.0], high=[np.pi / 1.5, 8.0], size=(1500, 2))
    y = np.array([simulate_pendulum(t, d) for t, d in X])

    print("Learning the stability region...")
    forest = train_forest(X, y, num_trees=50, max_depth=12, min_samples_split=2,
                          num_features_per_split=2, random_state=0)

    xx, yy = np.meshgrid(np.linspace(-np.pi / 1.5, np.pi / 1.5, 120), np.linspace(-8.0, 8.0, 120))
    grid = np.c_[xx.ravel(), yy.ravel()]
    print("Mapping the basin of attraction...")
    proba = np.array([predict_tree(d["tree"], grid) for d in forest]).mean(axis=0)
    truth = np.array([simulate_pendulum(t, d) for t, d in grid])
    agreement = np.mean((proba > 0.5) == truth)
    print(f"Agreement with the simulated ground truth on the grid: {agreement:.1%}")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    cs = ax.contourf(xx, yy, proba.reshape(xx.shape), levels=20, cmap="RdYlGn", alpha=0.85, vmin=0, vmax=1)
    ax.contour(xx, yy, truth.reshape(xx.shape), levels=[0.5], colors="k", linestyles="--", linewidths=1.2)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="darkgreen", s=6, alpha=0.5, label="Stable (simulated sample)")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="darkred", s=6, alpha=0.35, label="Falls (simulated sample)")
    ax.plot([], [], "k--", label="True boundary (full simulation)")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Initial angle $\theta_0$ (rad)")
    ax.set_ylabel(r"Initial angular velocity $\dot\theta_0$ (rad/s)")
    ax.set_title(f"Inverted pendulum under saturated PD control: learned basin of attraction "
                 f"({agreement:.0%} agreement)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(cs, ax=ax, label="Share of trees predicting 'stable'")
    finish(fig, "pendulum_basin.png")
