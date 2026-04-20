"""
TimeStone AI — Causal Graph Engine

Directed Acyclic Graph of causal business relationships with
do-calculus for intervention analysis. This is the core IP:
it maps HOW business decisions propagate through an organization,
enabling second- and third-order effect prediction.

Key concepts:
- Nodes represent business variables (revenue, churn, NPS, etc.)
- Edges encode causal strength, lag, and nonlinearity
- Interventions use do-calculus: P(Y | do(X)) ≠ P(Y | X)
- Confounders are tracked to avoid Simpson's paradox
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class VariableType(Enum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    MARKET = "market"
    CUSTOMER = "customer"
    TALENT = "talent"
    TECHNOLOGY = "technology"
    REGULATORY = "regulatory"
    STRATEGIC = "strategic"


class EdgeType(Enum):
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    THRESHOLD = "threshold"
    SATURATING = "saturating"      # S-curve / logistic
    EXPONENTIAL = "exponential"
    INVERSE = "inverse"


@dataclass
class CausalVariable:
    """A node in the causal graph representing a measurable business variable."""
    name: str
    var_type: VariableType
    current_value: float
    unit: str = ""
    min_value: float = float("-inf")
    max_value: float = float("inf")
    volatility: float = 0.1          # intrinsic noise σ
    mean_reversion_rate: float = 0.0  # Ornstein-Uhlenbeck θ
    long_run_mean: Optional[float] = None
    description: str = ""
    observable: bool = True           # can we measure it?

    def clamp(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


@dataclass
class CausalEdge:
    """
    A directed edge encoding causal influence from source → target.

    Attributes:
        strength: base coefficient (elasticity for LINEAR)
        lag_periods: how many time steps before the effect manifests
        edge_type: functional form of the relationship
        confidence: how certain we are this edge exists [0, 1]
        threshold_value: activation threshold for THRESHOLD type
        saturation_cap: asymptote for SATURATING type
        decay_rate: temporal decay of the effect (1.0 = no decay)
    """
    source: str
    target: str
    strength: float
    lag_periods: int = 0
    edge_type: EdgeType = EdgeType.LINEAR
    confidence: float = 0.8
    threshold_value: float = 0.0
    saturation_cap: float = 1.0
    decay_rate: float = 1.0
    description: str = ""

    def compute_effect(self, delta_source: float, source_value: float) -> float:
        """Compute the causal effect of a change in the source variable."""
        if self.edge_type == EdgeType.LINEAR:
            return self.strength * delta_source

        if self.edge_type == EdgeType.LOGARITHMIC:
            if source_value <= 0 or source_value + delta_source <= 0:
                return 0.0
            return self.strength * math.log1p(delta_source / source_value)

        if self.edge_type == EdgeType.THRESHOLD:
            if source_value + delta_source >= self.threshold_value > source_value:
                return self.strength
            if source_value + delta_source < self.threshold_value <= source_value:
                return -self.strength
            return 0.0

        if self.edge_type == EdgeType.SATURATING:
            new_val = source_value + delta_source
            old_saturated = self.saturation_cap / (1.0 + math.exp(-self.strength * source_value))
            new_saturated = self.saturation_cap / (1.0 + math.exp(-self.strength * new_val))
            return new_saturated - old_saturated

        if self.edge_type == EdgeType.EXPONENTIAL:
            return self.strength * (math.exp(delta_source * 0.01) - 1.0)

        if self.edge_type == EdgeType.INVERSE:
            if source_value == 0 or source_value + delta_source == 0:
                return 0.0
            return self.strength * (1.0 / (source_value + delta_source) - 1.0 / source_value)

        return self.strength * delta_source


class CausalGraph:
    """
    Directed Acyclic Graph encoding causal business relationships.

    Supports:
    - Topological propagation of interventions (do-calculus)
    - Confounded vs. unconfounded path separation
    - Multi-hop effect estimation with lag accumulation
    - Counterfactual reasoning: "What would have happened if...?"
    """

    def __init__(self):
        self.variables: Dict[str, CausalVariable] = {}
        self.edges: Dict[str, List[CausalEdge]] = {}   # source → [edges]
        self.reverse_edges: Dict[str, List[CausalEdge]] = {}  # target → [edges]
        self.confounders: Dict[Tuple[str, str], List[str]] = {}

    # ---- graph construction ----

    def add_variable(self, var: CausalVariable) -> None:
        self.variables[var.name] = var
        self.edges.setdefault(var.name, [])
        self.reverse_edges.setdefault(var.name, [])

    def add_edge(self, edge: CausalEdge) -> None:
        if edge.source not in self.variables or edge.target not in self.variables:
            raise ValueError(f"Both {edge.source} and {edge.target} must exist in the graph")
        if self._would_create_cycle(edge.source, edge.target):
            raise ValueError(f"Edge {edge.source} → {edge.target} would create a cycle")
        self.edges[edge.source].append(edge)
        self.reverse_edges[edge.target].append(edge)

    def add_confounder(self, var_a: str, var_b: str, confounder: str) -> None:
        key = tuple(sorted([var_a, var_b]))
        self.confounders.setdefault(key, [])
        if confounder not in self.confounders[key]:
            self.confounders[key].append(confounder)

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding source→target would create a cycle (DFS from target)."""
        visited: Set[str] = set()
        stack = [target]
        while stack:
            node = stack.pop()
            if node == source:
                return True
            if node in visited:
                continue
            visited.add(node)
            for edge in self.edges.get(node, []):
                stack.append(edge.target)
        return False

    # ---- topological ordering ----

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {v: 0 for v in self.variables}
        for edges in self.edges.values():
            for e in edges:
                in_degree[e.target] += 1

        queue = [v for v, d in in_degree.items() if d == 0]
        order: List[str] = []

        while queue:
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for edge in self.edges.get(node, []):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(order) != len(self.variables):
            raise RuntimeError("Cycle detected in causal graph")
        return order

    # ---- do-calculus intervention ----

    def do_intervention(
        self,
        interventions: Dict[str, float],
        time_horizon: int = 12,
        stochastic: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Apply do(X = x) interventions and propagate effects through the graph.

        Unlike conditioning P(Y|X=x), do-calculus severs all incoming edges
        to the intervened variable, preventing confounding bias.

        Args:
            interventions: {variable_name: new_value} for intervened variables
            time_horizon: number of periods to simulate forward
            stochastic: add noise based on variable volatility
            seed: random seed for reproducibility

        Returns:
            {variable_name: np.array of values over time_horizon}
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        topo_order = self.topological_sort()
        intervened_set = set(interventions.keys())

        # Initialize trajectories with current values
        trajectories: Dict[str, np.ndarray] = {}
        for name, var in self.variables.items():
            trajectories[name] = np.full(time_horizon, var.current_value, dtype=np.float64)

        # Pending effects buffer: (target, period, delta)
        pending: List[Tuple[str, int, float]] = []

        # Apply interventions and seed propagation from the intervention delta.
        for var_name, new_value in interventions.items():
            if var_name not in self.variables:
                continue
            baseline = self.variables[var_name].current_value
            delta_from_intervention = new_value - baseline
            trajectories[var_name][:] = new_value
            if delta_from_intervention == 0:
                continue
            for edge in self.edges.get(var_name, []):
                effect = edge.compute_effect(delta_from_intervention, baseline) * edge.confidence
                arrival = 1 + edge.lag_periods
                if arrival < time_horizon:
                    pending.append((edge.target, arrival, effect))

        for t in range(1, time_horizon):
            # Collect deltas from previous period
            period_deltas: Dict[str, float] = {}

            # Process pending lagged effects arriving at this period
            still_pending = []
            for target, arrival_t, delta in pending:
                if arrival_t == t:
                    period_deltas[target] = period_deltas.get(target, 0.0) + delta
                else:
                    still_pending.append((target, arrival_t, delta))
            pending = still_pending

            # Propagate in topological order
            for var_name in topo_order:
                var = self.variables[var_name]

                # Intervened variables are fixed — sever incoming edges (do-calculus)
                if var_name in intervened_set:
                    trajectories[var_name][t] = interventions[var_name]
                    continue

                # Start with previous value + any accumulated deltas
                base_value = trajectories[var_name][t - 1]
                accumulated_delta = period_deltas.get(var_name, 0.0)

                # Mean reversion (Ornstein-Uhlenbeck)
                if var.mean_reversion_rate > 0 and var.long_run_mean is not None:
                    mr_pull = var.mean_reversion_rate * (var.long_run_mean - base_value)
                    accumulated_delta += mr_pull

                # Stochastic noise
                if stochastic and var.volatility > 0:
                    noise = rng.normal(0, var.volatility * abs(base_value) if base_value != 0 else var.volatility)
                    accumulated_delta += noise

                new_value = var.clamp(base_value + accumulated_delta)
                trajectories[var_name][t] = new_value

                # Propagate effect to children
                delta_from_prev = new_value - trajectories[var_name][t - 1]
                for edge in self.edges.get(var_name, []):
                    if edge.target in intervened_set:
                        continue

                    effect = edge.compute_effect(delta_from_prev, base_value)
                    effect *= edge.confidence  # weight by confidence

                    # Apply temporal decay
                    if edge.decay_rate < 1.0:
                        effect *= edge.decay_rate ** t

                    if edge.lag_periods > 0:
                        arrival = t + edge.lag_periods
                        if arrival < time_horizon:
                            pending.append((edge.target, arrival, effect))
                    else:
                        period_deltas[edge.target] = period_deltas.get(edge.target, 0.0) + effect

        return trajectories

    # ---- counterfactual reasoning ----

    def counterfactual(
        self,
        factual_history: Dict[str, np.ndarray],
        counterfactual_interventions: Dict[str, float],
        intervention_time: int,
        time_horizon: int = 12,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Answer: "What would have happened if we had done X at time T?"

        Uses the Abduction-Action-Prediction framework:
        1. Abduction: infer exogenous noise from factual data
        2. Action: apply the counterfactual intervention
        3. Prediction: propagate forward with inferred noise
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        topo_order = self.topological_sort()

        # Step 1: Abduction — infer noise terms from factual history
        inferred_noise: Dict[str, np.ndarray] = {}
        for var_name in topo_order:
            var = self.variables[var_name]
            history = factual_history.get(var_name)
            if history is None:
                inferred_noise[var_name] = np.zeros(time_horizon)
                continue

            noise = np.zeros(len(history))
            for t in range(1, len(history)):
                predicted_delta = 0.0
                # Sum incoming causal effects
                for edge in self.reverse_edges.get(var_name, []):
                    source_history = factual_history.get(edge.source)
                    if source_history is not None and t - edge.lag_periods - 1 >= 0:
                        src_t = t - edge.lag_periods - 1
                        src_delta = source_history[src_t + 1] - source_history[src_t] if src_t + 1 < len(source_history) else 0
                        predicted_delta += edge.compute_effect(src_delta, source_history[src_t]) * edge.confidence

                actual_delta = history[t] - history[t - 1]
                noise[t] = actual_delta - predicted_delta
            inferred_noise[var_name] = noise

        # Step 2 & 3: Action + Prediction
        cf_trajectories: Dict[str, np.ndarray] = {}
        for name in self.variables:
            history = factual_history.get(name)
            if history is not None:
                cf_trajectories[name] = np.copy(history[:time_horizon])
            else:
                cf_trajectories[name] = np.full(time_horizon, self.variables[name].current_value)

        intervened_set = set(counterfactual_interventions.keys())

        for var_name, new_val in counterfactual_interventions.items():
            if var_name in cf_trajectories and intervention_time < time_horizon:
                cf_trajectories[var_name][intervention_time:] = new_val

        pending: List[Tuple[str, int, float]] = []

        for t in range(intervention_time + 1, time_horizon):
            period_deltas: Dict[str, float] = {}

            still_pending = []
            for target, arrival_t, delta in pending:
                if arrival_t == t:
                    period_deltas[target] = period_deltas.get(target, 0.0) + delta
                else:
                    still_pending.append((target, arrival_t, delta))
            pending = still_pending

            for var_name in topo_order:
                var = self.variables[var_name]

                if var_name in intervened_set:
                    cf_trajectories[var_name][t] = counterfactual_interventions[var_name]
                    continue

                base_value = cf_trajectories[var_name][t - 1]
                accumulated_delta = period_deltas.get(var_name, 0.0)

                # Add back the inferred noise from the factual world
                if var_name in inferred_noise and t < len(inferred_noise[var_name]):
                    accumulated_delta += inferred_noise[var_name][t]

                new_value = var.clamp(base_value + accumulated_delta)
                cf_trajectories[var_name][t] = new_value

                delta_from_prev = new_value - cf_trajectories[var_name][t - 1]
                for edge in self.edges.get(var_name, []):
                    if edge.target in intervened_set:
                        continue
                    effect = edge.compute_effect(delta_from_prev, base_value) * edge.confidence
                    if edge.lag_periods > 0:
                        arrival = t + edge.lag_periods
                        if arrival < time_horizon:
                            pending.append((edge.target, arrival, effect))
                    else:
                        period_deltas[edge.target] = period_deltas.get(edge.target, 0.0) + effect

        return cf_trajectories

    # ---- causal path analysis ----

    def find_all_causal_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> List[List[str]]:
        """Find all directed paths from source to target (up to max_depth)."""
        paths: List[List[str]] = []

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(list(path))
                return
            for edge in self.edges.get(current, []):
                if edge.target not in path:
                    path.append(edge.target)
                    dfs(edge.target, path, depth + 1)
                    path.pop()

        dfs(source, [source], 0)
        return paths

    def total_causal_effect(
        self,
        source: str,
        target: str,
        delta: float = 1.0,
    ) -> float:
        """
        Estimate the total causal effect of a unit change in source on target.
        Sums effects over all directed paths, accounting for edge strength and lag.
        """
        paths = self.find_all_causal_paths(source, target)
        total_effect = 0.0

        for path in paths:
            path_effect = delta
            for i in range(len(path) - 1):
                src, tgt = path[i], path[i + 1]
                edge = self._find_edge(src, tgt)
                if edge is None:
                    path_effect = 0.0
                    break
                path_effect = edge.compute_effect(path_effect, self.variables[src].current_value)
                path_effect *= edge.confidence
            total_effect += path_effect

        return total_effect

    def identify_confounded_paths(self, source: str, target: str) -> List[List[str]]:
        """Identify paths between source and target that pass through confounders."""
        all_paths = self.find_all_causal_paths(source, target)
        confounded = []
        for path in all_paths:
            for i in range(len(path)):
                for j in range(i + 1, len(path)):
                    key = tuple(sorted([path[i], path[j]]))
                    if key in self.confounders:
                        confounded.append(path)
                        break
                else:
                    continue
                break
        return confounded

    def _find_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        for edge in self.edges.get(source, []):
            if edge.target == target:
                return edge
        return None

    # ---- graph metrics ----

    def get_influence_scores(self) -> Dict[str, float]:
        """Compute influence score for each variable (weighted out-degree centrality)."""
        scores: Dict[str, float] = {}
        for name in self.variables:
            total = 0.0
            for edge in self.edges.get(name, []):
                total += abs(edge.strength) * edge.confidence
            scores[name] = total
        return scores

    def get_vulnerability_scores(self) -> Dict[str, float]:
        """Compute vulnerability score for each variable (weighted in-degree centrality)."""
        scores: Dict[str, float] = {}
        for name in self.variables:
            total = 0.0
            for edge in self.reverse_edges.get(name, []):
                total += abs(edge.strength) * edge.confidence
            scores[name] = total
        return scores

    def get_critical_paths(self, top_n: int = 5) -> List[Tuple[List[str], float]]:
        """Find the top-N most influential causal chains in the graph."""
        all_path_effects: List[Tuple[List[str], float]] = []

        for source in self.variables:
            for target in self.variables:
                if source == target:
                    continue
                effect = abs(self.total_causal_effect(source, target))
                if effect > 0:
                    paths = self.find_all_causal_paths(source, target)
                    if paths:
                        all_path_effects.append((paths[0], effect))

        all_path_effects.sort(key=lambda x: x[1], reverse=True)
        return all_path_effects[:top_n]

    # ---- serialization ----

    def to_dict(self) -> Dict:
        return {
            "variables": {
                name: {
                    "var_type": v.var_type.value,
                    "current_value": v.current_value,
                    "unit": v.unit,
                    "min_value": v.min_value if v.min_value != float("-inf") else None,
                    "max_value": v.max_value if v.max_value != float("inf") else None,
                    "volatility": v.volatility,
                    "mean_reversion_rate": v.mean_reversion_rate,
                    "long_run_mean": v.long_run_mean,
                    "description": v.description,
                }
                for name, v in self.variables.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "strength": e.strength,
                    "lag_periods": e.lag_periods,
                    "edge_type": e.edge_type.value,
                    "confidence": e.confidence,
                    "threshold_value": e.threshold_value,
                    "saturation_cap": e.saturation_cap,
                    "decay_rate": e.decay_rate,
                }
                for edges in self.edges.values()
                for e in edges
            ],
            "confounders": {
                f"{k[0]}|{k[1]}": v for k, v in self.confounders.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalGraph":
        graph = cls()
        for name, vdata in data["variables"].items():
            graph.add_variable(CausalVariable(
                name=name,
                var_type=VariableType(vdata["var_type"]),
                current_value=vdata["current_value"],
                unit=vdata.get("unit", ""),
                min_value=vdata["min_value"] if vdata.get("min_value") is not None else float("-inf"),
                max_value=vdata["max_value"] if vdata.get("max_value") is not None else float("inf"),
                volatility=vdata.get("volatility", 0.1),
                mean_reversion_rate=vdata.get("mean_reversion_rate", 0.0),
                long_run_mean=vdata.get("long_run_mean"),
                description=vdata.get("description", ""),
            ))
        for edata in data["edges"]:
            graph.add_edge(CausalEdge(
                source=edata["source"],
                target=edata["target"],
                strength=edata["strength"],
                lag_periods=edata.get("lag_periods", 0),
                edge_type=EdgeType(edata.get("edge_type", "linear")),
                confidence=edata.get("confidence", 0.8),
                threshold_value=edata.get("threshold_value", 0.0),
                saturation_cap=edata.get("saturation_cap", 1.0),
                decay_rate=edata.get("decay_rate", 1.0),
            ))
        for key_str, confs in data.get("confounders", {}).items():
            parts = key_str.split("|")
            if len(parts) == 2:
                for c in confs:
                    graph.add_confounder(parts[0], parts[1], c)
        return graph
