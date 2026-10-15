# AGENTS

## Current Competition Workflow

- Use the `thu-bdc2026` conda environment:
  `source /root/miniconda3/etc/profile.d/conda.sh && conda activate thu-bdc2026`
- Primary latest-data file for the 2026-06-28 run:
  `data/hybrid_stock_data.csv`
- Current target universe:
  `data_submit/hs300_universe.csv`
- Current final submission:
  `output/result.csv`

## Data Notes

- BaoStock is the current data source. It is not guaranteed to be byte-for-byte identical to official ToShare/Tushare data, so always validate date coverage, universe membership, and output sensitivity before submitting.
- `get_stock_data.py` is incremental. It reads existing `stock_data.csv`, builds per-stock fetch plans, and only downloads missing ranges unless a schema backfill is needed.
- `data_submit/stock_data.csv` contains current HS300 data through 2026-06-26. It may include old constituent history; always filter by `data_submit/hs300_universe.csv`.
- `data/hybrid_stock_data.csv` combines the old full-A panel with updated HS300 rows through 2026-06-26. This preserves full-market context while giving current candidates fresh data.

## Model Notes

- Best current model artifact:
  `model/ranker_hybrid_20260626/tree_ranker.joblib`
- It was trained by:
  `python code/src/train_advanced.py --data ./data/hybrid_stock_data.csv --target-universe ./data_submit/hs300_universe.csv --skip-portfolio --ranker-profile conservative --ranker-output ./model/ranker_hybrid_20260626 --folds 5 --val-size 15`
- OOF policy winner is `risk_adjusted`.
- Backup candidate from older model on latest data:
  `output/result_existing_conservative_on_20260626.csv`

## Background Jobs

- Full A-share incremental fetch was started in a detached screen:
  `screen -r bdc_fetch_all`
- Log:
  `logs/fetch_all_screen_20260628.log`
- Progress file:
  `data/fetch_progress.json`
