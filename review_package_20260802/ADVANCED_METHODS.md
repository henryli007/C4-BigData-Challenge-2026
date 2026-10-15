# Advanced Methods

This project now includes two complete post-data pipelines for the CSI300 portfolio task.

## Main Pipeline: Multi-Window Factors + Tree Rankers

Entrypoints:

```bash
sh train.sh
sh test.sh
```

`train.sh` runs `code/src/train_advanced.py`. The main model builds same-day stock features from historical data only, labels each stock by `T+5 open / T+1 open - 1`, trains LightGBM/XGBoost/CatBoost ranking models when available, evaluates them with rolling date validation, and saves the final ensemble to `model/advanced_ranker/tree_ranker.joblib`.

The default data source is `data/stock_data.csv`. It uses the full A-share panel to build market context factors, then filters the trainable and predictable target universe to `data/hs300_stock_list.csv`, matching the competition stock pool. The feature set includes multi-window momentum, reversal, volatility, liquidity, price-channel, trend-efficiency, ATR-like risk, money-flow, market/industry metadata when available, full-market context, and cross-sectional rank/z-score factors over windows `3, 5, 10, 20, 30, 60, 120`.

Prediction ranks all valid latest-day CSI300 candidates, selects at most 5 stocks, then uses the best validation policy among:

- `risk_adjusted`
- `score_softmax`
- `equal`

The result is written to `output/result.csv` with `stock_id,weight` and normalized weight sum.

Useful commands:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thu-bdc2026

# Full training: tree ranker + direct portfolio model.
sh train.sh

# Main tree-ranker only, useful when iterating on the primary submission model.
python code/src/train_advanced.py --skip-portfolio

# More regularized tree ranker. This is slower but has better rolling validation
# on the current full dataset. New runs also save oof_scores.csv for blend tuning.
python code/src/train_advanced.py \
  --skip-portfolio \
  --ranker-profile conservative \
  --ranker-output ./model/advanced_ranker_conservative

# Quick smoke test with lighter estimators and capped fold rows.
python code/src/train_advanced.py --fast --skip-portfolio --ranker-max-rows-per-fold 8000 --ranker-output ./model/smoke_ranker

# Predict final result.csv from the main tree-ranker artifact.
sh test.sh

# Predict from the conservative tree-ranker.
python code/src/predict_advanced.py \
  --method ranker \
  --ranker-artifact ./model/advanced_ranker_conservative/tree_ranker.joblib \
  --output ./output/result_ranker_conservative.csv
```

## Direct Portfolio Model

`train_advanced.py` also trains `model/portfolio_direct/portfolio_model.pth` by default. This model receives each date as a full cross-section:

```text
X_T = N stocks x multi-window factors
score_i = model(x_i)
w_i = softmax(score_i)
loss = -sum(w_i * r_i) + lambda * risk + gamma * concentration
```

It optimizes realized `T+1` to `T+5` open return directly with risk and Herfindahl concentration penalties. It is available for prediction via:

```bash
python code/src/predict_advanced.py --method portfolio
```

The competition output still allows only up to 5 stocks, so inference keeps the top 5 direct-portfolio scores and renormalizes their weights. Prediction also supports a single-stock max-weight cap to avoid an over-concentrated softmax output:

```bash
python code/src/predict_advanced.py \
  --method portfolio \
  --portfolio-artifact ./model/portfolio_direct/portfolio_artifact.joblib \
  --portfolio-model ./model/portfolio_direct/portfolio_model.pth \
  --output ./output/result_portfolio_default.csv \
  --max-weight 0.45 \
  --max-per-industry 2
```

Portfolio training now runs walk-forward validation before final training by default and writes `rolling_validation.csv`, `oof_scores.csv`, and `validation_scores.csv`. Control the cost with:

```bash
python code/src/train_advanced.py \
  --skip-ranker \
  --portfolio-output ./model/portfolio_direct_walkforward \
  --portfolio-val-folds 3 \
  --portfolio-rolling-epochs 20
```

## Blended Prediction

`predict_advanced.py --method blend` combines the tree-ranker and direct-portfolio scores after cross-sectional percentile normalization. This avoids mixing incompatible raw score scales. The current final `output/result.csv` is generated from the conservative tree-ranker plus default direct-portfolio model:

```bash
python code/src/predict_advanced.py \
  --method blend \
  --ranker-artifact ./model/advanced_ranker_conservative/tree_ranker.joblib \
  --portfolio-artifact ./model/portfolio_direct/portfolio_artifact.joblib \
  --portfolio-model ./model/portfolio_direct/portfolio_model.pth \
  --output ./output/result.csv \
  --blend-alpha 0.6 \
  --max-weight 0.45 \
  --max-per-industry 2
```

After retraining both models with the current code, tune the blend and risk controls on true out-of-fold scores:

```bash
python code/src/tune_blend.py \
  --ranker-oof ./model/advanced_ranker_conservative/oof_scores.csv \
  --portfolio-oof ./model/portfolio_direct/oof_scores.csv \
  --output ./model/blend_tuning.csv
```

Use `./check_training_status.sh` while jobs are running and `python compare_models.py` after training to compare saved validation metrics.

## Baseline Backup

The original Transformer baseline remains in:

- `code/src/train.py`
- `code/src/predict.py`

Run those files directly if a backup baseline is needed.
