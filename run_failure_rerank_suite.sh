#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

DATA="${DATA_PATH:-./data_submit/stock_data.csv}"
UNIVERSE="${UNIVERSE_PATH:-./data_submit/hs300_universe.csv}"
HOT="${HOT_PATH:-./model/extra_apr_data/ths_hot_stock.csv}"
EXTRA="${EXTRA_FACTOR_DIR:-./model/extra_apr_event_data}"

echo "[1/4] smoke: continuation + failure rerank, latest 2 weeks"
"$PYTHON" code/src/weekly_feature_ablation.py \
  --data "$DATA" \
  --target-universe "$UNIVERSE" \
  --extra-hot-data "$HOT" \
  --extra-factor-dir "$EXTRA" \
  --output-dir ./model/two_stage_failure_smoke2 \
  --start-date 2026-07-01 \
  --end-date 2026-07-24 \
  --max-weeks 2 \
  --min-train-weeks 52 \
  --models lightgbm \
  --profile fast \
  --relevance-mode rank \
  --event-text \
  --decay-risk \
  --risk-penalty-weight 0.01 \
  --stage1-recall \
  --stage1-candidates 50 \
  --stage1-recall-weight 0.18 \
  --continuation-model \
  --continuation-weight 0.16 \
  --failure-model \
  --failure-weight 0.14 \
  --spec 'two_stage_no_industry_relative||_ind_(z|rank)$|f_industry'

echo "[2/4] baseline refresh: no failure model, recent 14 weeks"
"$PYTHON" code/src/weekly_feature_ablation.py \
  --data "$DATA" \
  --target-universe "$UNIVERSE" \
  --extra-hot-data "$HOT" \
  --extra-factor-dir "$EXTRA" \
  --output-dir ./model/two_stage_baseline_recent14_refresh \
  --start-date 2026-04-20 \
  --end-date 2026-07-24 \
  --max-weeks 14 \
  --min-train-weeks 52 \
  --models lightgbm \
  --profile fast \
  --relevance-mode rank \
  --event-text \
  --decay-risk \
  --risk-penalty-weight 0.01 \
  --stage1-recall \
  --stage1-candidates 50 \
  --stage1-recall-weight 0.18 \
  --spec 'two_stage_all||' \
  --spec 'two_stage_no_industry_relative||_ind_(z|rank)$|f_industry'

echo "[3/4] continuation + failure rerank: recent 14 weeks"
"$PYTHON" code/src/weekly_feature_ablation.py \
  --data "$DATA" \
  --target-universe "$UNIVERSE" \
  --extra-hot-data "$HOT" \
  --extra-factor-dir "$EXTRA" \
  --output-dir ./model/two_stage_failure_recent14 \
  --start-date 2026-04-20 \
  --end-date 2026-07-24 \
  --max-weeks 14 \
  --min-train-weeks 52 \
  --models lightgbm \
  --profile fast \
  --relevance-mode rank \
  --event-text \
  --decay-risk \
  --risk-penalty-weight 0.01 \
  --stage1-recall \
  --stage1-candidates 50 \
  --stage1-recall-weight 0.18 \
  --continuation-model \
  --continuation-weight 0.16 \
  --failure-model \
  --failure-weight 0.14 \
  --spec 'two_stage_all||' \
  --spec 'two_stage_no_industry_relative||_ind_(z|rank)$|f_industry'

echo "[4/4] train/test formal artifact with failure_model"
echo "[tune] rerank weights on recent14 outputs"
for spec in two_stage_all two_stage_no_industry_relative; do
  if [ -f "./model/two_stage_failure_recent14/${spec}/walkforward_scores.csv" ]; then
    "$PYTHON" code/src/tune_rerank_weights.py \
      --scores "./model/two_stage_failure_recent14/${spec}/walkforward_scores.csv" \
      --output "./model/two_stage_failure_recent14/${spec}/rerank_weight_tuning.csv"
  fi
done

sh train.sh
sh test.sh

echo "Done. Compare:"
echo "  cat model/two_stage_baseline_recent14_refresh/summary.csv"
echo "  cat model/two_stage_failure_recent14/summary.csv"
echo "  for f in model/two_stage_failure_recent14/*/rank_diagnostics.csv; do echo \"\$f\"; tail -1 \"\$f\"; done"
echo "  for f in model/two_stage_failure_recent14/*/rerank_weight_tuning.csv; do echo \"\$f\"; head -5 \"\$f\"; done"
echo "  for f in model/two_stage_failure_recent14/*/error_reason_summary.csv; do echo \"\$f\"; cat \"\$f\"; done"
echo "  cat output/result.csv"
echo "  head -30 model/current_submission/scored_top50.csv"
