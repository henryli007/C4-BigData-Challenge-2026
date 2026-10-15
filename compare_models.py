from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_ranker(name: str, model_dir: str) -> dict:
    path = Path(model_dir)
    progress = read_json(path / "progress.json")
    metrics_path = path / "weight_policy_metrics.csv"
    row = {"name": name, "type": "ranker", "status": progress.get("status"), "stage": progress.get("stage")}
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path, index_col=0)
        best_policy = metrics["portfolio_return"].idxmax()
        best = metrics.loc[best_policy]
        row.update(
            {
                "best_policy": best_policy,
                "portfolio_return": float(best["portfolio_return"]),
                "final_score": float(best["final_score"]),
            }
        )
    elif "policy_metrics" in progress:
        policy_metrics = progress["policy_metrics"]
        best_policy = max(policy_metrics, key=lambda p: policy_metrics[p]["portfolio_return"])
        best = policy_metrics[best_policy]
        row.update(
            {
                "best_policy": best_policy,
                "portfolio_return": float(best["portfolio_return"]),
                "final_score": float(best["final_score"]),
            }
        )
    return row


def summarize_portfolio(name: str, model_dir: str) -> dict:
    path = Path(model_dir)
    progress = read_json(path / "progress.json")
    row = {"name": name, "type": "portfolio", "status": progress.get("status"), "stage": progress.get("stage")}
    history_path = path / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        best = history.sort_values("portfolio_return", ascending=False).iloc[0]
        row.update(
            {
                "best_epoch": int(best["epoch"]),
                "portfolio_return": float(best["portfolio_return"]),
                "final_score": float(best["final_score"]),
            }
        )
    elif "best_val_return" in progress:
        row["portfolio_return"] = float(progress["best_val_return"])
    return row


def main() -> None:
    rows = [
        summarize_ranker("advanced_ranker", "model/advanced_ranker"),
        summarize_ranker("advanced_ranker_conservative", "model/advanced_ranker_conservative"),
        summarize_portfolio("portfolio_direct", "model/portfolio_direct"),
        summarize_portfolio("portfolio_direct_regularized", "model/portfolio_direct_regularized"),
    ]
    df = pd.DataFrame(rows)
    cols = ["name", "type", "status", "stage", "best_policy", "best_epoch", "portfolio_return", "final_score"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

