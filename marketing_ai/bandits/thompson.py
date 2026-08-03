"""
Thompson Sampling Bandit Engine.
Implements Bayesian posterior sampling over channel payoffs for continuous exploration.
"""

import numpy as np
from typing import Dict, List, Any


class ThompsonSamplingBandit:
    """
    Bayesian Thompson Sampling for Multi-Armed Bandit Allocation.
    """
    def __init__(self, n_actions: int = 6):
        self.n_actions = n_actions
        # Gaussian prior: Mean mu=0, Variance sigma=1.0
        self.mu = np.zeros(n_actions)
        self.sigma = np.ones(n_actions) * 2.0
        self.counts = np.zeros(n_actions)

    def select_action(self) -> int:
        """
        Samples action from posterior normal distributions.
        """
        samples = [np.random.normal(self.mu[a], self.sigma[a]) for a in range(self.n_actions)]
        return int(np.argmax(samples))

    def update(self, action: int, reward: float):
        """
        Updates Gaussian posterior mean and variance.
        """
        self.counts[action] += 1
        n = self.counts[action]
        # Online mean update
        old_mu = self.mu[action]
        self.mu[action] = old_mu + (reward - old_mu) / n
        # Shrink variance as sample count increases
        self.sigma[action] = max(0.2, self.sigma[action] * 0.95)
