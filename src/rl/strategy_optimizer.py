"""
TimeStone AI — RL Strategy Optimizer

Reinforcement Learning agent that autonomously explores the digital twin
to discover optimal transformation strategies. Turns TimeStone from a
simulator into an autonomous strategic advisor.

The agent:
1. Observes the company genome + causal graph state
2. Selects interventions (actions) from a strategy space
3. Runs simulations to estimate reward (ROI - risk penalty)
4. Learns a policy that maximizes long-term risk-adjusted value

Uses a simplified policy-gradient approach (REINFORCE with baseline)
that works without deep learning dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.causal_graph import CausalGraph


class ActionType(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    INVEST = "invest"
    HOLD = "hold"


@dataclass
class Action:
    """An intervention the RL agent can take on the digital twin."""
    variable: str
    action_type: ActionType
    magnitude: float  # percentage change or absolute value
    cost: float = 0.0
    description: str = ""


@dataclass
class State:
    """Observable state of the digital twin."""
    values: Dict[str, float]
    period: int
    regime: str = "normal"
    budget_remaining: float = float("inf")

    def to_vector(self) -> np.ndarray:
        return np.array(list(self.values.values()) + [self.period, self.budget_remaining])


@dataclass
class Transition:
    """A single (s, a, r, s') transition."""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


@dataclass
class Episode:
    """A full trajectory of transitions."""
    transitions: List[Transition]
    total_reward: float
    total_cost: float
    final_state: State


@dataclass
class OptimizationResult:
    """Result of RL optimization."""
    best_strategy: List[Action]
    expected_reward: float
    confidence_interval: Tuple[float, float]
    explored_strategies: int
    convergence_history: List[float]
    top_strategies: List[Tuple[List[Action], float]]
    computation_time_ms: int = 0


class StrategySpace:
    """
    Defines the space of possible interventions the agent can take.
    Each variable has a set of possible actions (increase, decrease, invest).
    """

    def __init__(self):
        self.actions: List[Action] = []
        self._actions_by_variable: Dict[str, List[Action]] = {}

    def add_action(self, action: Action) -> None:
        self.actions.append(action)
        self._actions_by_variable.setdefault(action.variable, []).append(action)

    def get_actions_for(self, variable: str) -> List[Action]:
        return self._actions_by_variable.get(variable, [])

    @classmethod
    def from_causal_graph(
        cls,
        graph: CausalGraph,
        budget: float = 10_000_000,
        granularity: int = 3,
    ) -> "StrategySpace":
        """Auto-generate strategy space from a causal graph's variables."""
        space = cls()

        influence_scores = graph.get_influence_scores()
        for var_name, var in graph.variables.items():
            if not var.observable:
                continue

            # More influential variables get finer granularity
            influence = influence_scores.get(var_name, 0)
            n_levels = granularity + (1 if influence > 0.3 else 0)

            current = var.current_value
            if current == 0:
                continue

            for level in range(1, n_levels + 1):
                pct = 0.05 * level  # 5%, 10%, 15%, 20%
                cost = budget * pct * 0.5  # rough cost model

                space.add_action(Action(
                    variable=var_name,
                    action_type=ActionType.INCREASE,
                    magnitude=pct,
                    cost=cost,
                    description=f"Increase {var_name} by {pct:.0%}",
                ))
                space.add_action(Action(
                    variable=var_name,
                    action_type=ActionType.DECREASE,
                    magnitude=pct,
                    cost=cost * 0.3,
                    description=f"Decrease {var_name} by {pct:.0%}",
                ))

        space.add_action(Action(
            variable="_hold",
            action_type=ActionType.HOLD,
            magnitude=0,
            cost=0,
            description="Hold — take no action this period",
        ))

        return space

    def sample_random(self, rng: np.random.Generator, n: int = 1) -> List[Action]:
        """Sample n random actions."""
        indices = rng.choice(len(self.actions), size=n, replace=False)
        return [self.actions[i] for i in indices]


class TwinEnvironment:
    """
    RL environment wrapper around the digital twin.

    Translates RL actions into causal graph interventions,
    runs simulation, and computes reward.
    """

    def __init__(
        self,
        graph: CausalGraph,
        strategy_space: StrategySpace,
        horizon: int = 12,
        risk_aversion: float = 0.5,
        budget: float = 10_000_000,
    ):
        self.graph = graph
        self.strategy_space = strategy_space
        self.horizon = horizon
        self.risk_aversion = risk_aversion
        self.initial_budget = budget
        self._initial_values = {n: v.current_value for n, v in graph.variables.items()}
        self.state: Optional[State] = None

    def reset(self) -> State:
        """Reset environment to initial state."""
        self.state = State(
            values=dict(self._initial_values),
            period=0,
            budget_remaining=self.initial_budget,
        )
        return self.state

    def step(self, action: Action) -> Tuple[State, float, bool]:
        """Execute an action and return (next_state, reward, done)."""
        if self.state is None:
            self.reset()

        if action.cost > self.state.budget_remaining:
            # Can't afford this action — penalty
            return self.state, -0.1, False

        # Apply action as intervention
        interventions = {}
        if action.action_type != ActionType.HOLD:
            current = self.state.values.get(action.variable, 0)
            if action.action_type == ActionType.INCREASE:
                new_val = current * (1 + action.magnitude)
            elif action.action_type == ActionType.DECREASE:
                new_val = current * (1 - action.magnitude)
            else:
                new_val = current
            interventions[action.variable] = new_val

        # Run one-step simulation through causal graph
        if interventions:
            trajectories = self.graph.do_intervention(
                interventions, time_horizon=3, stochastic=True,
            )
            new_values = {name: float(traj[-1]) for name, traj in trajectories.items()}
        else:
            new_values = dict(self.state.values)

        new_period = self.state.period + 1
        new_budget = self.state.budget_remaining - action.cost
        done = new_period >= self.horizon

        next_state = State(
            values=new_values,
            period=new_period,
            budget_remaining=new_budget,
        )

        reward = self._compute_reward(self.state, next_state, action)
        self.state = next_state
        return next_state, reward, done

    def _compute_reward(self, prev: State, curr: State, action: Action) -> float:
        """
        Reward = value_gain - risk_penalty - cost_penalty

        Value gain: improvement in key metrics (revenue, margin, etc.)
        Risk penalty: penalize high-variance outcomes
        Cost penalty: proportional to investment
        """
        # Value gain from key variables
        value_gain = 0.0
        for var_name in ["revenue", "profit_margin", "customer_satisfaction", "market_share"]:
            if var_name in prev.values and var_name in curr.values:
                old = prev.values[var_name]
                new = curr.values[var_name]
                if old != 0:
                    value_gain += (new - old) / abs(old)

        # Cost penalty (normalized by budget)
        cost_penalty = action.cost / (self.initial_budget + 1e-9) * 0.1

        # Risk penalty (penalize large swings)
        volatility = 0.0
        for var_name in curr.values:
            if var_name in prev.values:
                old = prev.values[var_name]
                new = curr.values[var_name]
                if old != 0:
                    volatility += abs(new - old) / abs(old)
        risk_penalty = self.risk_aversion * volatility * 0.1

        return value_gain - cost_penalty - risk_penalty


class REINFORCEOptimizer:
    """
    Policy gradient (REINFORCE with baseline) optimizer.

    Learns a softmax policy over action preferences.
    No neural network needed — uses tabular preferences
    with linear function approximation.
    """

    def __init__(
        self,
        env: TwinEnvironment,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        n_actions = len(env.strategy_space.actions)
        self.preferences = np.zeros(n_actions)
        self.baseline = 0.0
        self.episode_history: List[Episode] = []

    def _softmax_policy(self) -> np.ndarray:
        """Compute action probabilities via softmax."""
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))
        return exp_prefs / exp_prefs.sum()

    def _select_action(self) -> Tuple[int, Action]:
        probs = self._softmax_policy()
        idx = self.rng.choice(len(probs), p=probs)
        return idx, self.env.strategy_space.actions[idx]

    def run_episode(self) -> Episode:
        """Run one episode using current policy."""
        state = self.env.reset()
        transitions = []
        total_reward = 0.0
        total_cost = 0.0

        for _ in range(self.env.horizon):
            idx, action = self._select_action()
            next_state, reward, done = self.env.step(action)

            transitions.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ))
            total_reward += reward
            total_cost += action.cost
            state = next_state
            if done:
                break

        episode = Episode(
            transitions=transitions,
            total_reward=total_reward,
            total_cost=total_cost,
            final_state=state,
        )
        self.episode_history.append(episode)
        return episode

    def update_policy(self, episode: Episode) -> None:
        """Update preferences using REINFORCE with baseline."""
        T = len(episode.transitions)
        returns = np.zeros(T)

        # Compute discounted returns
        G = 0.0
        for t in reversed(range(T)):
            G = episode.transitions[t].reward + self.gamma * G
            returns[t] = G

        # Update baseline (running average)
        self.baseline = 0.9 * self.baseline + 0.1 * episode.total_reward

        # Policy gradient update
        probs = self._softmax_policy()
        for t in range(T):
            action = episode.transitions[t].action
            idx = self.env.strategy_space.actions.index(action)
            advantage = returns[t] - self.baseline

            # Gradient of log softmax
            grad = -probs.copy()
            grad[idx] += 1.0

            self.preferences += self.lr * advantage * grad

    def optimize(
        self,
        n_episodes: int = 500,
        top_k: int = 5,
    ) -> OptimizationResult:
        """Run full RL optimization."""
        import time
        start = time.time()

        convergence = []
        strategy_rewards: List[Tuple[List[Action], float]] = []

        for ep in range(n_episodes):
            episode = self.run_episode()
            self.update_policy(episode)

            convergence.append(episode.total_reward)
            actions = [t.action for t in episode.transitions]
            strategy_rewards.append((actions, episode.total_reward))

        # Find best strategies
        strategy_rewards.sort(key=lambda x: x[1], reverse=True)
        top_strategies = strategy_rewards[:top_k]

        best_actions, best_reward = top_strategies[0]
        all_rewards = [r for _, r in strategy_rewards]

        elapsed_ms = int((time.time() - start) * 1000)

        return OptimizationResult(
            best_strategy=best_actions,
            expected_reward=best_reward,
            confidence_interval=(
                float(np.percentile(all_rewards, 5)),
                float(np.percentile(all_rewards, 95)),
            ),
            explored_strategies=n_episodes,
            convergence_history=convergence,
            top_strategies=top_strategies,
            computation_time_ms=elapsed_ms,
        )


class StrategyOptimizer:
    """
    High-level interface combining environment setup + RL optimization.
    This is what the API calls.
    """

    @staticmethod
    def optimize_for_company(
        graph: CausalGraph,
        budget: float = 10_000_000,
        horizon: int = 12,
        risk_aversion: float = 0.5,
        n_episodes: int = 300,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """Find optimal transformation strategy for a company."""
        space = StrategySpace.from_causal_graph(graph, budget)
        env = TwinEnvironment(graph, space, horizon, risk_aversion, budget)
        optimizer = REINFORCEOptimizer(env, seed=seed)
        return optimizer.optimize(n_episodes=n_episodes)
