"""
TimeStone AI — Causal Discovery

Automatic construction of causal graphs from observational data.
Implements three complementary algorithms:

1. **PC algorithm** — Constraint-based, uses conditional independence tests.
   Discovers the Markov equivalence class of DAGs.

2. **NOTEARS** — Score-based continuous optimization, smooth acyclicity
   constraint. Faster for large graphs, returns a unique DAG.

3. **Granger causality** — Time series-specific, uses F-tests to detect
   temporal precedence. Good for lag discovery.

Combined, these let TimeStone auto-build causal graphs from client data
rather than relying on hand-crafted industry knowledge alone. This is
the automation moat: every client's graph gets personalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DiscoveredEdge:
    """An edge discovered from data."""
    source: str
    target: str
    strength: float
    confidence: float
    lag: int = 0
    method: str = "pc"
    p_value: float = 1.0


@dataclass
class DiscoveryResult:
    """Full result of causal discovery."""
    edges: List[DiscoveredEdge]
    variables: List[str]
    method: str
    n_samples: int
    skeleton: List[Tuple[str, str]] = field(default_factory=list)
    undirected: List[Tuple[str, str]] = field(default_factory=list)


# ---- Conditional Independence Test ----

def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute partial correlation of x, y given z.
    Returns (correlation, p-value).
    """
    if z is None or z.size == 0:
        corr, p = stats.pearsonr(x, y)
        return corr, p

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Residualize x and y on z via OLS
    def residualize(v: np.ndarray) -> np.ndarray:
        z_aug = np.column_stack([np.ones(len(z)), z])
        beta, *_ = np.linalg.lstsq(z_aug, v, rcond=None)
        return v - z_aug @ beta

    x_res = residualize(x)
    y_res = residualize(y)

    n = len(x)
    k = z.shape[1]

    corr, _ = stats.pearsonr(x_res, y_res)

    # Correct p-value using t-distribution with adjusted df
    if abs(corr) >= 1.0:
        return corr, 0.0
    df = n - k - 2
    if df <= 0:
        return corr, 1.0
    t_stat = corr * np.sqrt(df / (1 - corr ** 2))
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return corr, p


def is_independent(
    x: np.ndarray,
    y: np.ndarray,
    conditioning: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """Test conditional independence via partial correlation."""
    _, p = partial_correlation(x, y, conditioning)
    return p > alpha, p


# ---- PC Algorithm ----

class PCAlgorithm:
    """
    Peter-Clark algorithm for causal discovery.

    Phase 1: Build skeleton by testing conditional independence
    Phase 2: Orient edges using collider rules (v-structures)
    Phase 3: Apply Meek's orientation rules
    """

    def __init__(self, alpha: float = 0.05, max_conditioning_size: int = 3):
        self.alpha = alpha
        self.max_conditioning_size = max_conditioning_size

    def discover(self, df: pd.DataFrame) -> DiscoveryResult:
        """Run PC algorithm on a DataFrame."""
        variables = list(df.columns)
        n = len(variables)
        data = df.to_numpy()

        # Phase 1: Skeleton
        adjacencies: Dict[str, Set[str]] = {v: set(variables) - {v} for v in variables}
        separating_sets: Dict[Tuple[str, str], Set[str]] = {}

        for cond_size in range(self.max_conditioning_size + 1):
            edges_to_remove = []

            for i, var_i in enumerate(variables):
                for var_j in list(adjacencies[var_i]):
                    if var_j <= var_i:
                        continue
                    other_neighbors = list(adjacencies[var_i] - {var_j})
                    if len(other_neighbors) < cond_size:
                        continue

                    for cond_set in combinations(other_neighbors, cond_size):
                        cond_indices = [variables.index(v) for v in cond_set]
                        cond_array = data[:, cond_indices] if cond_indices else None

                        independent, p = is_independent(
                            data[:, i],
                            data[:, variables.index(var_j)],
                            cond_array,
                            self.alpha,
                        )
                        if independent:
                            edges_to_remove.append((var_i, var_j))
                            separating_sets[(var_i, var_j)] = set(cond_set)
                            separating_sets[(var_j, var_i)] = set(cond_set)
                            break

            for a, b in edges_to_remove:
                adjacencies[a].discard(b)
                adjacencies[b].discard(a)

        # Build skeleton edge list
        skeleton = []
        for v, neighbors in adjacencies.items():
            for n_ in neighbors:
                if v < n_:
                    skeleton.append((v, n_))

        # Phase 2: Orient v-structures (X → Z ← Y where X-Z-Y in skeleton and Z not in sep(X,Y))
        oriented: Set[Tuple[str, str]] = set()
        for a, b in skeleton:
            for c in variables:
                if c == a or c == b:
                    continue
                if c in adjacencies[a] and c in adjacencies[b] and b not in adjacencies[a]:
                    sep = separating_sets.get((a, b), set())
                    if c not in sep:
                        oriented.add((a, c))
                        oriented.add((b, c))

        # Phase 3: Apply Meek's rules until no more changes
        changed = True
        while changed:
            changed = False

            # Rule 1: if a→b and b-c with a not adj c, then b→c
            for a, b in list(oriented):
                for c in variables:
                    if c == a or c == b:
                        continue
                    if c in adjacencies[b] and c not in adjacencies[a]:
                        if (b, c) not in oriented and (c, b) not in oriented:
                            oriented.add((b, c))
                            changed = True

            # Rule 2: if a→b→c and a-c then a→c
            for a, b in list(oriented):
                for c in variables:
                    if c == a or c == b:
                        continue
                    if (b, c) in oriented and c in adjacencies[a]:
                        if (a, c) not in oriented and (c, a) not in oriented:
                            oriented.add((a, c))
                            changed = True

        # Build DiscoveredEdge list
        edges = []
        undirected = []

        for a, b in skeleton:
            if (a, b) in oriented and (b, a) not in oriented:
                # Directed a → b
                corr, p = stats.pearsonr(
                    data[:, variables.index(a)],
                    data[:, variables.index(b)],
                )
                edges.append(DiscoveredEdge(
                    source=a, target=b,
                    strength=float(corr),
                    confidence=max(0.0, 1.0 - p),
                    method="pc",
                    p_value=float(p),
                ))
            elif (b, a) in oriented and (a, b) not in oriented:
                corr, p = stats.pearsonr(
                    data[:, variables.index(b)],
                    data[:, variables.index(a)],
                )
                edges.append(DiscoveredEdge(
                    source=b, target=a,
                    strength=float(corr),
                    confidence=max(0.0, 1.0 - p),
                    method="pc",
                    p_value=float(p),
                ))
            else:
                undirected.append((a, b))

        return DiscoveryResult(
            edges=edges,
            variables=variables,
            method="pc",
            n_samples=len(df),
            skeleton=skeleton,
            undirected=undirected,
        )


# ---- NOTEARS Algorithm ----

class NOTEARS:
    """
    NOTEARS: DAGs with NO TEARS.

    Continuous optimization approach — minimizes least squares loss
    subject to smooth acyclicity constraint h(W) = tr(e^(W⊙W)) - d = 0.

    Uses gradient descent with augmented Lagrangian.
    """

    def __init__(
        self,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ):
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

    def discover(self, df: pd.DataFrame, lambda1: float = 0.1) -> DiscoveryResult:
        """Run NOTEARS."""
        variables = list(df.columns)
        # Standardize data
        X = df.to_numpy()
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        n, d = X.shape

        def _h(W: np.ndarray) -> float:
            """Smooth acyclicity: tr(e^(W⊙W)) - d"""
            W_sq = W * W
            # Approximation: tr((I + W⊙W/d)^d) - d (matrix exponential series)
            M = np.eye(d) + W_sq / d
            E = np.linalg.matrix_power(M, d)
            return float(np.trace(E) - d)

        def _loss(W: np.ndarray) -> Tuple[float, np.ndarray]:
            """Least squares loss and gradient."""
            M = X @ W
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
            grad = -1.0 / n * X.T @ R
            return loss, grad

        # Initialize with zero matrix
        W_est = np.zeros((d, d))
        rho, alpha, h = 1.0, 0.0, float("inf")

        for _ in range(self.max_iter):
            # Inner optimization: minimize loss + L1 + penalty
            W_new, h_new = self._optimize_inner(W_est, X, lambda1, rho, alpha)

            if h_new > 0.25 * h:
                rho *= 10
                if rho >= self.rho_max:
                    break
            else:
                W_est = W_new
                alpha += rho * h_new
                h = h_new
                if h <= self.h_tol:
                    break

        # Threshold
        W_est[np.abs(W_est) < self.w_threshold] = 0.0

        # Build edges
        edges = []
        for i in range(d):
            for j in range(d):
                if i != j and W_est[i, j] != 0:
                    edges.append(DiscoveredEdge(
                        source=variables[i],
                        target=variables[j],
                        strength=float(W_est[i, j]),
                        confidence=min(1.0, abs(float(W_est[i, j]))),
                        method="notears",
                    ))

        return DiscoveryResult(
            edges=edges,
            variables=variables,
            method="notears",
            n_samples=n,
        )

    def _optimize_inner(
        self,
        W: np.ndarray,
        X: np.ndarray,
        lambda1: float,
        rho: float,
        alpha: float,
    ) -> Tuple[np.ndarray, float]:
        """Inner proximal gradient descent."""
        n, d = X.shape
        W = W.copy()
        lr = 0.001

        for _ in range(200):
            # Compute gradients
            M = X @ W
            R = X - M
            grad_loss = -1.0 / n * X.T @ R

            # Acyclicity gradient
            W_sq = W * W
            E = np.linalg.matrix_power(np.eye(d) + W_sq / d, d - 1)
            grad_h = 2 * E.T * W
            h_val = float(np.trace(np.eye(d) + W_sq / d @ E)) - d

            grad = grad_loss + rho * h_val * grad_h + alpha * grad_h
            W_new = W - lr * grad

            # Soft thresholding for L1
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lambda1 * lr, 0)

            # Zero out diagonal
            np.fill_diagonal(W_new, 0)

            if np.max(np.abs(W_new - W)) < 1e-5:
                break
            W = W_new

        # Final h
        W_sq = W * W
        M = np.eye(d) + W_sq / d
        E = np.linalg.matrix_power(M, d)
        h = float(np.trace(E) - d)

        return W, h


# ---- Granger Causality ----

class GrangerDiscovery:
    """
    Granger causality for time series causal discovery.
    Tests whether lagged values of X improve prediction of Y beyond
    Y's own lagged values.
    """

    def __init__(self, max_lag: int = 4, alpha: float = 0.05):
        self.max_lag = max_lag
        self.alpha = alpha

    def discover(self, df: pd.DataFrame) -> DiscoveryResult:
        """Discover Granger-causal edges."""
        variables = list(df.columns)
        edges = []

        for y_var in variables:
            for x_var in variables:
                if x_var == y_var:
                    continue
                best_lag, best_p, best_strength = self._granger_test(
                    df[y_var].values, df[x_var].values
                )
                if best_p < self.alpha:
                    edges.append(DiscoveredEdge(
                        source=x_var,
                        target=y_var,
                        strength=float(best_strength),
                        confidence=max(0.0, 1.0 - best_p),
                        lag=best_lag,
                        method="granger",
                        p_value=float(best_p),
                    ))

        return DiscoveryResult(
            edges=edges,
            variables=variables,
            method="granger",
            n_samples=len(df),
        )

    def _granger_test(
        self, y: np.ndarray, x: np.ndarray,
    ) -> Tuple[int, float, float]:
        """
        Test if x Granger-causes y. Returns (best_lag, p_value, coefficient).
        """
        best_p = 1.0
        best_lag = 1
        best_coef = 0.0
        n = len(y)

        for lag in range(1, self.max_lag + 1):
            if n - lag < 10:
                continue

            y_target = y[lag:]

            # Restricted: y on its own lags
            y_lagged = np.column_stack([y[lag - k - 1:n - k - 1] for k in range(lag)])
            y_lagged_aug = np.column_stack([np.ones(len(y_lagged)), y_lagged])
            beta_r, *_ = np.linalg.lstsq(y_lagged_aug, y_target, rcond=None)
            resid_r = y_target - y_lagged_aug @ beta_r
            rss_r = (resid_r ** 2).sum()

            # Unrestricted: y on own lags + x's lags
            x_lagged = np.column_stack([x[lag - k - 1:n - k - 1] for k in range(lag)])
            full = np.column_stack([y_lagged_aug, x_lagged])
            beta_u, *_ = np.linalg.lstsq(full, y_target, rcond=None)
            resid_u = y_target - full @ beta_u
            rss_u = (resid_u ** 2).sum()

            # F-test
            df_num = lag
            df_den = n - lag - 2 * lag - 1
            if df_den <= 0 or rss_u <= 0:
                continue

            f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
            if f_stat <= 0:
                continue
            p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)

            if p_value < best_p:
                best_p = p_value
                best_lag = lag
                best_coef = float(np.mean(beta_u[-lag:]))

        return best_lag, best_p, best_coef


# ---- Ensemble ----

class EnsembleCausalDiscovery:
    """
    Combines PC, NOTEARS, and Granger for robust discovery.
    An edge is accepted if it's supported by majority of methods.
    """

    def __init__(self, min_votes: int = 2):
        self.min_votes = min_votes

    def discover(self, df: pd.DataFrame) -> DiscoveryResult:
        results = [
            PCAlgorithm().discover(df),
            NOTEARSDiscoveryWrapper().discover(df),
            GrangerDiscovery().discover(df),
        ]

        # Count votes per (source, target) pair
        edge_votes: Dict[Tuple[str, str], List[DiscoveredEdge]] = {}
        for result in results:
            for edge in result.edges:
                key = (edge.source, edge.target)
                edge_votes.setdefault(key, []).append(edge)

        # Accept edges with enough votes
        final_edges = []
        for (source, target), votes in edge_votes.items():
            if len(votes) >= self.min_votes:
                avg_strength = np.mean([e.strength for e in votes])
                avg_confidence = np.mean([e.confidence for e in votes])
                final_edges.append(DiscoveredEdge(
                    source=source,
                    target=target,
                    strength=float(avg_strength),
                    confidence=float(avg_confidence),
                    lag=max((e.lag for e in votes), default=0),
                    method="ensemble",
                ))

        return DiscoveryResult(
            edges=final_edges,
            variables=list(df.columns),
            method="ensemble",
            n_samples=len(df),
        )


class NOTEARSDiscoveryWrapper:
    """Wrapper that defensively catches NOTEARS numerical issues."""

    def discover(self, df: pd.DataFrame) -> DiscoveryResult:
        try:
            return NOTEARS().discover(df)
        except Exception:
            return DiscoveryResult(
                edges=[], variables=list(df.columns),
                method="notears", n_samples=len(df),
            )
