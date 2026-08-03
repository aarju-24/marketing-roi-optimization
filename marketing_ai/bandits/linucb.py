"""
LinUCB Contextual Multi-Armed Bandit Engine.
Simulates daily sequential budget allocation updates balancing exploration and exploitation.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class LinUCBBandit:
    """
    LinUCB Contextual Bandit for dynamic daily channel allocation exploration.
    """
    def __init__(self, n_actions: int = 6, d_features: int = 4, alpha: float = 0.5):
        self.n_actions = n_actions
        self.d_features = d_features
        self.alpha = alpha

        # A_a = d x d identity matrix per action
        self.A = [np.identity(d_features) for _ in range(n_actions)]
        # b_a = d x 1 zero vector per action
        self.b = [np.zeros((d_features, 1)) for _ in range(n_actions)]

    def select_action(self, context_vector: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Selects best action using LinUCB formula.
        UCB_a = theta_a^T * x + alpha * sqrt(x^T * A_a^-1 * x)
        """
        x = context_vector.reshape(-1, 1)
        p = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            variance = float((x.T @ A_inv @ x).item())
            p[a] = float((theta_a.T @ x).item()) + self.alpha * np.sqrt(max(0.0, variance))

        chosen_action = int(np.argmax(p))
        return chosen_action, p

    def update(self, action: int, context_vector: np.ndarray, reward: float):
        """
        Updates Ridge regression parameters for chosen action after observing reward.
        A_a += x * x^T
        b_a += reward * x
        """
        x = context_vector.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x


def run_bandit_simulation(days: int = 60, alpha: float = 0.5, seed: int = 42) -> Dict[str, Any]:
    """
    Runs a multi-day sequential bandit simulation over 6 channels.
    """
    np.random.seed(seed)
    bandit = LinUCBBandit(n_actions=6, d_features=4, alpha=alpha)

    cumulative_reward = []
    actions_taken = []
    rewards_history = []

    total_r = 0.0
    for day in range(days):
        # Context vector: [holiday, competitor_cpc_ratio, month_season, trend]
        cpc_ratio = np.random.uniform(0.8, 1.3)
        holiday = 1 if day in [15, 30, 45] else 0
        season = np.sin(2 * np.pi * day / 30)
        context = np.array([1.0, float(holiday), float(cpc_ratio), float(season)])

        action, p_scores = bandit.select_action(context)
        actions_taken.append(action)

        # True underlying reward function (noisy)
        base_rewards = [3.8, 3.2, 3.4, 2.9, 5.1, 2.2]
        true_reward = base_rewards[action] * (1.0 + 0.3 * holiday) * (2.10 / cpc_ratio) + np.random.normal(0, 0.2)
        
        bandit.update(action, context, true_reward)
        
        total_r += true_reward
        cumulative_reward.append(round(total_r, 2))
        rewards_history.append(round(true_reward, 2))

    return {
        "simulation_days": days,
        "total_cumulative_reward": round(total_r, 2),
        "cumulative_reward_curve": cumulative_reward,
        "actions_taken": actions_taken,
        "rewards_history": rewards_history
    }
