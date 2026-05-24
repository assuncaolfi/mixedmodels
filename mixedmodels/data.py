"""Build design matrices from a parsed formula + a DataFrame.

This is the only module that touches pandas / formulaic. Downstream code sees
JAX arrays and term metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
from formulaic import Formula

from .formula import ParsedFormula


@dataclass
class REBlock:
    """One `(expr | group)` random-effects term."""

    name: str  # human-readable, e.g. "(1 + x | subject)"
    group_name: str
    q: int  # number of random effects per group (columns of Ze)
    G: int  # number of groups
    col_names: list[str]
    group_levels: list[str]
    group_idx: np.ndarray  # shape (n,), int32, indices 0..G-1
    Ze: np.ndarray  # shape (n, q), float64


@dataclass
class ModelMatrices:
    y: np.ndarray  # shape (n,)
    X: np.ndarray  # shape (n, p)
    fixed_names: list[str]
    weights: np.ndarray  # shape (n,)
    re: list[REBlock]
    response_name: str

    @property
    def n(self) -> int:
        return self.y.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]


def build_matrices(
    parsed: ParsedFormula,
    data: pd.DataFrame,
    *,
    weights: np.ndarray | None = None,
) -> ModelMatrices:
    # Fixed effects via formulaic
    fixed_formula = f"{parsed.response} ~ {parsed.fixed_rhs}"
    y_df, X_df = Formula(fixed_formula).get_model_matrix(data)
    fixed_names = list(X_df.columns)
    X = np.asarray(X_df, dtype=np.float64)
    y = np.asarray(y_df, dtype=np.float64).reshape(-1)

    if weights is None:
        w = np.ones(y.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != y.shape[0]:
            raise ValueError("weights length must match response length")

    # Random effects
    re_blocks: list[REBlock] = []
    for expr, grp in parsed.random:
        # Build the random expression's model matrix (no intercept stripping;
        # formulaic adds intercept by default if not suppressed by `0+` or `-1`).
        # We parse as a one-sided formula and reuse columns directly.
        Ze_df = Formula(f"~ {expr}").get_model_matrix(data)
        # Drop the dummy response column if any (formulaic with one-sided gives a single df).
        Ze = np.asarray(Ze_df, dtype=np.float64)
        col_names = list(Ze_df.columns)

        if grp not in data.columns:
            raise ValueError(f"Grouping factor {grp!r} not found in data columns")
        cat = pd.Categorical(data[grp])
        group_idx = np.asarray(cat.codes, dtype=np.int32)
        group_levels = [str(x) for x in cat.categories]
        G = len(group_levels)
        q = Ze.shape[1]

        re_blocks.append(
            REBlock(
                name=f"({expr} | {grp})",
                group_name=grp,
                q=q,
                G=G,
                col_names=col_names,
                group_levels=group_levels,
                group_idx=group_idx,
                Ze=Ze,
            )
        )

    return ModelMatrices(
        y=y,
        X=X,
        fixed_names=fixed_names,
        weights=w,
        re=re_blocks,
        response_name=parsed.response,
    )


def to_jax(mm: ModelMatrices):
    """Convert ModelMatrices to a tuple of JAX arrays.

    Only arrays go through `jit`; static shapes (G_k, q_k) live on `ModelSpec`
    and on the Hessian structure. Each RE block contributes a `(Ze, group_idx)`
    pair here, in the order they appear in the formula.
    """
    re_jax = tuple((jnp.asarray(b.Ze), jnp.asarray(b.group_idx)) for b in mm.re)
    return (
        jnp.asarray(mm.y),
        jnp.asarray(mm.X),
        jnp.asarray(mm.weights),
        re_jax,
    )
