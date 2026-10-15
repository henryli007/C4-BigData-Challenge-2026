# THU-BDC2026 Review Package

This is a lightweight review package for the current THU-BDC2026 stock prediction project.

The package is intended for another GPT/model to inspect the current approach and diagnose why the system is still far from the target 10% weekly return.

## What Is Included

- `code/`: current source code.
- `train.sh`, `test.sh`: current formal train/replay entry points.
- `AGENT.md`, `日志.md`: current handoff and work log.
- `output/result.csv`: current formal result.
- `output/current_scored_top50_with_decay_risk.csv`: latest scored candidates with model score and decay-risk diagnostics.
- `data_samples/`: small sample rows and schemas only, not full data.
- `experiment_summaries/`: selected summaries, diagnostics, feature importance, and risk-layer tuning files.

## What Is Not Included

- Full stock K-line data.
- Full TuShare event data.
- Trained model binaries.
- Python virtual environment.
- Long logs and large walk-forward score files.
- `.env.tushare` or any API token file.

## Current Formal Method

The formal model keeps LightGBM rankers as the main predictor and adds a post-rank risk-control layer.

Three model profiles are blended:

- `base_rank`: 0.60
- `event_text_rank`: 0.20
- `event_board_rank`: 0.20

Then `advanced.decay_risk` computes:

- `strength_decay_score`
- `runup_reversal_score`
- `event_proxy_risk`
- `total_decay_risk`

Final rank score:

```text
score = model_score_rank - risk_penalty
```

Default `risk_penalty_weight` is `0.02`. This is intentionally light; heavier penalties hurt Top5 equal-weight backtests.

Current portfolio weights are confidence/risk controlled:

```csv
stock_id,weight
301165,0.240000
301236,0.220000
300896,0.200000
002558,0.180000
002353,0.160000
```

## Current Known Problem

The model can find recent strong stocks, but it still struggles to find 5 stocks that will all remain strong over the next 5 trading days.

The risk layer improves rank1 reliability in some 14-week tests, but it does not yet solve Top5 recall.

Important recent numbers:

- Best historical 14-week Top5 blend without decay risk:
  - `no_event 0.60 + event_text 0.20 + extra_board 0.20`
  - Top5 equal return about 4.96%.
- With light decay risk penalty `0.02`:
  - Top5 equal return about 4.28%.
  - rank1 mean improves from about 3.92% to 4.71%.
  - rank1 `>=10%` weeks improve from 3/14 to 4/14.
- `no_event_decay` single-model 14-week test:
  - best policy return about 4.52%.
  - rank1 return about 5.59%.

This is still far from the desired 10%.

## Suggested Review Questions

Please inspect:

1. Whether `advanced.decay_risk` is correctly designed or too hand-tuned.
2. Whether the 5-day label design is aligned with the desired Top5 portfolio objective.
3. Whether the current LightGBM ranker objective is unsuitable for stable Top5 recall.
4. Whether feature importance suggests too many noisy/highly correlated features.
5. Whether the portfolio layer should optimize expected Top5 return directly instead of ranking individual stocks.
6. Whether event/text/board-hot data is being used in a way that leaks, lags, or fails to align with weekly cutoffs.
7. Whether the project should move to a two-stage system:
   - candidate generation for strong continuation
   - risk veto / decay detection
8. What is the most likely reason the system cannot reach 10% average weekly return.

## Repro Notes

This package is not meant to run full training because full data and model binaries are intentionally excluded.

To run the original project, restore full data files in the original locations:

- `data_submit/stock_data.csv`
- `data_submit/hs300_universe.csv`
- `model/extra_apr_data/ths_hot_stock.csv`
- `model/extra_apr_event_data/*.csv`

Then use:

```bash
sh train.sh
sh test.sh
```
