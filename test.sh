PYTHON="${PYTHON:-python}"
if [ -x "./.venv/bin/python" ]; then
  PYTHON="./.venv/bin/python"
fi

DATA_PATH="${DATA_PATH:-./data/stock_data}"
UNIVERSE_PATH="${UNIVERSE_PATH:-./data/hs300_stock_list.csv}"
if [ -f "./data_submit/stock_data.csv" ] && [ ! -d "./data/stock_data" ]; then
  DATA_PATH="./data_submit/stock_data.csv"
  UNIVERSE_PATH="./data_submit/hs300_universe.csv"
fi

"$PYTHON" code/src/train_current_fusion_submission.py \
  --mode predict \
  --data "$DATA_PATH" \
  --target-universe "$UNIVERSE_PATH" \
  --model-output ./model/current_submission \
  --output ./output/result.csv \
  --scored-output ./model/current_submission/scored_top50.csv \
  --extra-hot-data ./model/extra_apr_data/ths_hot_stock.csv \
  --extra-factor-dir ./model/extra_apr_event_data
