from __future__ import annotations

import argparse
from pathlib import Path

from advanced.features import (
    FeatureConfig,
    add_market_context,
    enrich_with_metadata,
    filter_universe,
    load_price_data,
    load_universe_codes,
)
from advanced.portfolio_model import PortfolioConfig, train_portfolio_model
from advanced.tree_ranker import TreeRankerConfig, train_tree_ranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced CSI300 ranking and portfolio models.")
    parser.add_argument("--data", default="./data/stock_data.csv")
    parser.add_argument("--target-universe", default="./data/hs300_stock_list.csv")
    parser.add_argument("--ranker-output", default="./model/advanced_ranker")
    parser.add_argument("--portfolio-output", default="./model/portfolio_direct")
    parser.add_argument("--skip-portfolio", action="store_true")
    parser.add_argument("--skip-ranker", action="store_true")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--portfolio-val-folds", type=int, default=3)
    parser.add_argument("--portfolio-rolling-epochs", type=int, default=20)
    parser.add_argument("--ranker-max-rows-per-fold", type=int, default=0)
    parser.add_argument("--ranker-profile", choices=["default", "conservative"], default="default")
    parser.add_argument("--portfolio-profile", choices=["default", "regularized"], default="default")
    parser.add_argument("--fast", action="store_true", help="Use a lighter config for quick validation runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data).parent
    full_df = enrich_with_metadata(load_price_data(args.data), data_dir)
    full_df = add_market_context(full_df)
    target_codes = load_universe_codes(args.target_universe)
    raw_df = filter_universe(full_df, target_codes)
    print(f"full rows: {len(full_df)}, target rows: {len(raw_df)}, target stocks: {raw_df['股票代码'].nunique()}")

    feature_cfg = FeatureConfig(windows=(3, 5, 10, 20, 30, 60, 120), min_history=60, horizon=5, top_k=5)

    if not args.skip_ranker:
        conservative = args.ranker_profile == "conservative"
        ranker_cfg = TreeRankerConfig(
            output_dir=args.ranker_output,
            n_folds=2 if args.fast else args.folds,
            val_size=args.val_size,
            lgb_estimators=250 if args.fast else (900 if conservative else 700),
            xgb_estimators=220 if args.fast else (700 if conservative else 550),
            cat_estimators=250 if args.fast else (900 if conservative else 700),
            learning_rate=0.025 if conservative else 0.035,
            num_leaves=31 if conservative else 63,
            max_depth=6 if conservative else -1,
            min_child_samples=45 if conservative else 25,
            subsample=0.78 if conservative else 0.85,
            colsample_bytree=0.75 if conservative else 0.85,
            reg_alpha=0.15 if conservative else 0.05,
            reg_lambda=1.2 if conservative else 0.2,
            max_rows_per_fold=args.ranker_max_rows_per_fold,
            progress_path=str(Path(args.ranker_output) / "progress.json"),
        )
        train_tree_ranker(raw_df, feature_cfg=feature_cfg, ranker_cfg=ranker_cfg)

    if not args.skip_portfolio:
        regularized = args.portfolio_profile == "regularized"
        portfolio_cfg = PortfolioConfig(
            output_dir=args.portfolio_output,
            epochs=10 if args.fast else args.epochs,
            n_val_folds=1 if args.fast else args.portfolio_val_folds,
            rolling_val_epochs=5 if args.fast else args.portfolio_rolling_epochs,
            learning_rate=1e-4 if regularized else 2e-4,
            risk_lambda=0.25 if regularized else 0.15,
            concentration_gamma=0.15 if regularized else 0.05,
            entropy_beta=0.03 if regularized else 0.01,
            early_stopping_patience=6 if regularized else 8,
            progress_path=str(Path(args.portfolio_output) / "progress.json"),
        )
        train_portfolio_model(raw_df, feature_cfg=feature_cfg, portfolio_cfg=portfolio_cfg)

    Path("./output").mkdir(exist_ok=True)


if __name__ == "__main__":
    main()
