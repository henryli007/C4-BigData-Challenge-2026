#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python"
fi

DATA_PATH="${DATA_PATH:-./data_submit/stock_data.csv}"
UNIVERSE_PATH="${UNIVERSE_PATH:-./data_submit/hs300_universe.csv}"
HOT_PATH="${HOT_PATH:-./model/extra_apr_data/ths_hot_stock.csv}"
EXTRA_DIR="${EXTRA_DIR:-./model/extra_apr_event_data}"

echo "[1/6] smoke: two-stage recall + decay risk, recent 2 weeks"
"$PYTHON" code/src/weekly_feature_ablation.py \
  --data "$DATA_PATH" \
  --target-universe "$UNIVERSE_PATH" \
  --extra-hot-data "$HOT_PATH" \
  --extra-factor-dir "$EXTRA_DIR" \
  --output-dir ./model/two_stage_decay_smoke2 \
  --start-date 2026-07-01 \
  --end-date 2026-07-24 \
  --max-weeks 2 \
  --models lightgbm \
  --profile fast \
  --relevance-mode rank \
  --blend-weight 0.0 \
  --event-text \
  --decay-risk \
  --risk-penalty-weight 0.01 \
  --stage1-recall \
  --stage1-recall-weight 0.18 \
  --stage1-candidates 50 \
  --spec 'two_stage_all||' \
  --spec 'two_stage_no_event||ths_hot|limit_|top_|margin_|ths_board|tsf_text'

echo "[2/6] core 14-week validation"
"$PYTHON" code/src/weekly_feature_ablation.py \
  --data "$DATA_PATH" \
  --target-universe "$UNIVERSE_PATH" \
  --extra-hot-data "$HOT_PATH" \
  --extra-factor-dir "$EXTRA_DIR" \
  --output-dir ./model/two_stage_decay_recent14 \
  --start-date 2026-04-20 \
  --end-date 2026-07-24 \
  --max-weeks 14 \
  --models lightgbm \
  --profile fast \
  --relevance-mode rank \
  --blend-weight 0.0 \
  --event-text \
  --decay-risk \
  --risk-penalty-weight 0.01 \
  --stage1-recall \
  --stage1-recall-weight 0.18 \
  --stage1-candidates 50 \
  --spec 'two_stage_all||' \
  --spec 'two_stage_no_event||ths_hot|limit_|top_|margin_|ths_board|tsf_text' \
  --spec 'two_stage_no_industry_relative||_ind_(z|rank)$|f_industry'

echo "[3/6] 40-week validation, no-event backbone"
"$PYTHON" code/src/weekly_feature_ablation.py \
  --data "$DATA_PATH" \
  --target-universe "$UNIVERSE_PATH" \
  --extra-hot-data "$HOT_PATH" \
  --extra-factor-dir "$EXTRA_DIR" \
  --output-dir ./model/two_stage_decay_recent40_no_event \
  --start-date 2025-10-01 \
  --end-date 2026-07-24 \
  --max-weeks 40 \
  --models lightgbm \
  --profile fast \
  --relevance-mode rank \
  --blend-weight 0.0 \
  --event-text \
  --decay-risk \
  --risk-penalty-weight 0.01 \
  --stage1-recall \
  --stage1-recall-weight 0.18 \
  --stage1-candidates 50 \
  --spec 'two_stage_no_event||ths_hot|limit_|top_|margin_|ths_board|tsf_text'

echo "[4/6] feature importance and SHAP audit for current formal artifact"
"$PYTHON" code/src/analyze_current_feature_importance.py \
  --artifact ./model/current_submission/fusion_submission.joblib \
  --output-dir ./model/current_submission/feature_audit \
  --top-n 60 \
  --shap-sample-rows 120

echo "[5/6] train formal submission"
sh train.sh

echo "[6/6] replay formal submission"
sh test.sh

echo "done"
echo "Key files:"
echo "  ./output/result.csv"
echo "  ./model/two_stage_decay_smoke2/summary.csv"
echo "  ./model/two_stage_decay_recent14/summary.csv"
echo "  ./model/two_stage_decay_recent40_no_event/summary.csv"
echo "  ./model/current_submission/scored_top50.csv"
echo "  ./model/current_submission/feature_audit/"
