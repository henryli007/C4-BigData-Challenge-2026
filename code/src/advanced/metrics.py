from __future__ import annotations

import numpy as np
import pandas as pd

from .features import DATE_COL, ID_COL, LABEL_COL


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones_like(weights) / max(len(weights), 1)
    return weights / total


def cap_and_redistribute_weights(weights: np.ndarray, max_weight: float | None = None) -> np.ndarray:
    weights = _normalize_weights(weights)
    if max_weight is None or max_weight <= 0 or max_weight >= 1 or len(weights) == 0:
        return weights

    capped = weights.copy()
    for _ in range(len(capped) + 1):
        over = capped > max_weight
        if not over.any():
            break
        excess = float((capped[over] - max_weight).sum())
        capped[over] = max_weight
        under = ~over
        if not under.any() or excess <= 1e-12:
            break
        base = capped[under].sum()
        if base <= 1e-12:
            capped[under] += excess / under.sum()
        else:
            capped[under] += excess * capped[under] / base
    return _normalize_weights(np.minimum(capped, max_weight))


def weights_for_topk(day: pd.DataFrame, policy: str, top_k: int = 5, temperature: float = 0.02) -> np.ndarray:
    scores = day["score"].to_numpy(dtype=float)
    if policy == "equal":
        return np.ones(len(day), dtype=float) / len(day)
    if policy == "score_softmax":
        centered = scores - np.nanmax(scores)
        raw = np.exp(np.clip(centered / max(temperature, 1e-6), -50, 50))
        return _normalize_weights(raw)
    if policy == "risk_adjusted":
        centered = scores - np.nanmax(scores)
        score_part = np.exp(np.clip(centered / max(temperature, 1e-6), -50, 50))
        risk = day.get("risk_vol_20", pd.Series(0.03, index=day.index)).to_numpy(dtype=float)
        raw = score_part / np.sqrt(np.clip(risk, 1e-4, None))
        return _normalize_weights(raw)
    raise ValueError(f"Unknown weight policy: {policy}")


def select_portfolio(
    scored: pd.DataFrame,
    top_k: int = 5,
    policy: str = "risk_adjusted",
    temperature: float = 0.02,
    max_weight: float | None = None,
    max_per_industry: int = 0,
) -> pd.DataFrame:
    candidates = scored.sort_values("score", ascending=False).copy()
    if max_per_industry > 0 and "行业" in candidates.columns:
        industry_counts: dict[str, int] = {}
        keep_indices = []
        for idx, row in candidates.iterrows():
            industry = str(row.get("行业", "UNK"))
            if industry_counts.get(industry, 0) >= max_per_industry:
                continue
            keep_indices.append(idx)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if len(keep_indices) >= top_k:
                break
        ranked = candidates.loc[keep_indices].copy()
        if len(ranked) < top_k:
            needed = top_k - len(ranked)
            ranked = pd.concat([ranked, candidates.drop(index=ranked.index).head(needed)], axis=0)
    else:
        ranked = candidates.head(top_k).copy()
    if len(ranked) == 0:
        raise ValueError("No candidates to select")
    ranked["weight"] = cap_and_redistribute_weights(
        weights_for_topk(ranked, policy=policy, top_k=top_k, temperature=temperature),
        max_weight=max_weight,
    )
    return ranked[[ID_COL, "score", "weight"]]


def evaluate_scored_frame(
    scored: pd.DataFrame,
    top_k: int = 5,
    policy: str = "risk_adjusted",
    temperature: float = 0.02,
    max_weight: float | None = None,
    max_per_industry: int = 0,
) -> dict[str, float]:
    returns = []
    oracle = []
    random_base = []
    for _, day in scored.groupby(DATE_COL, sort=True):
        day = day.dropna(subset=[LABEL_COL, "score"])
        if len(day) < top_k:
            continue
        pred = select_portfolio(
            day,
            top_k=top_k,
            policy=policy,
            temperature=temperature,
            max_weight=max_weight,
            max_per_industry=max_per_industry,
        )
        merged = pred.merge(day[[ID_COL, LABEL_COL]], on=ID_COL, how="left")
        returns.append(float((merged["weight"] * merged[LABEL_COL]).sum()))
        oracle.append(float(day.nlargest(top_k, LABEL_COL)[LABEL_COL].mean()))
        random_base.append(float(day[LABEL_COL].mean()))

    if not returns:
        return {"portfolio_return": 0.0, "oracle_return": 0.0, "random_return": 0.0, "final_score": 0.0}

    p = float(np.mean(returns))
    o = float(np.mean(oracle))
    r = float(np.mean(random_base))
    final = (p - r) / (o - r + 1e-12) if abs(o - r) > 1e-12 else 0.0
    return {
        "portfolio_return": p,
        "oracle_return": o,
        "random_return": r,
        "final_score": float(final),
    }
