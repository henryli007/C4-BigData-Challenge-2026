# Commands To Run

目标：一次性跑完整的两阶段召回 + decay/lifecycle 风险 + SHAP/importance 审计，不再一小步一小步试。

## 0. 环境检查

```bash
cd /root/THU-BDC2026
./.venv/bin/python - <<'PY'
import lightgbm, shap, pandas, sklearn
print('lightgbm', lightgbm.__version__)
print('shap', shap.__version__)
print('pandas', pandas.__version__)
print('sklearn', sklearn.__version__)
PY
```

## 1. 一键完整套件

这条命令会依次跑：

- 2 周 smoke
- 14 周核心验证
- 40 周 no-event 主干长验证
- 当前 artifact 的 importance + SHAP
- `train.sh`
- `test.sh`

```bash
cd /root/THU-BDC2026
bash run_full_optimization_suite.sh
```

预计耗时较长。跑完重点看：

```bash
cat model/two_stage_decay_smoke2/summary.csv
cat model/two_stage_decay_recent14/summary.csv
cat model/two_stage_decay_recent40_no_event/summary.csv
cat output/result.csv
head -50 model/current_submission/scored_top50.csv
```

## 2. 如果只想先跑 14 周核心验证

新增 continuation_model + failure_model rerank 套件会写到单独目录，不覆盖正在跑的 `two_stage_decay_*` 结果：

```bash
cd /root/THU-BDC2026
bash run_failure_rerank_suite.sh
```

跑完后把这些贴回来：

```bash
cat model/two_stage_baseline_recent14_refresh/summary.csv
cat model/two_stage_failure_recent14/summary.csv
for f in model/two_stage_failure_recent14/*/rank_diagnostics.csv; do echo "$f"; tail -1 "$f"; done
for f in model/two_stage_failure_recent14/*/rerank_weight_tuning.csv; do echo "$f"; head -5 "$f"; done
for f in model/two_stage_failure_recent14/*/error_reason_summary.csv; do echo "$f"; cat "$f"; done
cat output/result.csv
head -30 model/current_submission/scored_top50.csv
```

如果只跑 14 周 continuation/failure rerank 验证：

```bash
cd /root/THU-BDC2026
./.venv/bin/python code/src/weekly_feature_ablation.py \
  --data ./data_submit/stock_data.csv \
  --target-universe ./data_submit/hs300_universe.csv \
  --extra-hot-data ./model/extra_apr_data/ths_hot_stock.csv \
  --extra-factor-dir ./model/extra_apr_event_data \
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
```

旧版不带 failure_model 的 14 周核心验证：

```bash
cd /root/THU-BDC2026
./.venv/bin/python code/src/weekly_feature_ablation.py \
  --data ./data_submit/stock_data.csv \
  --target-universe ./data_submit/hs300_universe.csv \
  --extra-hot-data ./model/extra_apr_data/ths_hot_stock.csv \
  --extra-factor-dir ./model/extra_apr_event_data \
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
```

跑完：

```bash
cat model/two_stage_decay_recent14/summary.csv
for f in model/two_stage_decay_recent14/*/rank_diagnostics.csv; do echo "$f"; tail -1 "$f"; done
```

## 3. 如果只想跑 40 周长验证

```bash
cd /root/THU-BDC2026
./.venv/bin/python code/src/weekly_feature_ablation.py \
  --data ./data_submit/stock_data.csv \
  --target-universe ./data_submit/hs300_universe.csv \
  --extra-hot-data ./model/extra_apr_data/ths_hot_stock.csv \
  --extra-factor-dir ./model/extra_apr_event_data \
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
```

跑完：

```bash
cat model/two_stage_decay_recent40_no_event/summary.csv
tail -1 model/two_stage_decay_recent40_no_event/two_stage_no_event/rank_diagnostics.csv
```

## 4. 重新训练正式提交

这条会重新训练三路 ranker 和 Stage1 recall classifier，并写 `output/result.csv`。

```bash
cd /root/THU-BDC2026
sh train.sh
sh test.sh
cat output/result.csv
head -50 model/current_submission/scored_top50.csv
```

## 5. 重新跑 importance + SHAP

```bash
cd /root/THU-BDC2026
./.venv/bin/python code/src/analyze_current_feature_importance.py \
  --artifact ./model/current_submission/fusion_submission.joblib \
  --output-dir ./model/current_submission/feature_audit \
  --top-n 80 \
  --shap-sample-rows 160
```

重点看：

```bash
head -40 model/current_submission/feature_audit/combined_feature_importance.csv
head -40 model/current_submission/feature_audit/base_rank_shap_importance.csv
head -40 model/current_submission/feature_audit/event_text_rank_shap_importance.csv
head -40 model/current_submission/feature_audit/event_board_rank_shap_importance.csv
cat model/current_submission/feature_audit/zero_gain_summary.csv
```

## 6. 跑完后发给我看的文件

请把下面文件内容贴回来：

```bash
cat model/two_stage_decay_recent14/summary.csv
cat model/two_stage_decay_recent40_no_event/summary.csv
tail -1 model/two_stage_decay_recent14/two_stage_all/rank_diagnostics.csv
tail -1 model/two_stage_decay_recent14/two_stage_no_event/rank_diagnostics.csv
tail -1 model/two_stage_decay_recent40_no_event/two_stage_no_event/rank_diagnostics.csv
cat output/result.csv
head -30 model/current_submission/scored_top50.csv
```
