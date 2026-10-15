#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "== processes =="
ps -eo pid,etime,stat,%cpu,%mem,rss,cmd \
  | grep -E 'train_advanced.py|bdc_train_(ranker|portfolio)' \
  | grep -v grep || true

echo
echo "== screens =="
screen -ls || true

show_json() {
  local path="$1"
  echo
  echo "== ${path} =="
  if [[ -f "$path" ]]; then
    python -m json.tool "$path"
  else
    echo "missing"
  fi
}

show_json "model/advanced_ranker/progress.json"
show_json "model/portfolio_direct/progress.json"

echo
echo "== latest ranker log =="
ranker_log="$(ls -t logs/train_ranker_*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "$ranker_log" ]]; then
  echo "$ranker_log"
  tail -n 20 "$ranker_log"
else
  echo "missing"
fi

echo
echo "== latest portfolio log =="
portfolio_log="$(ls -t logs/train_portfolio_*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "$portfolio_log" ]]; then
  echo "$portfolio_log"
  tail -n 20 "$portfolio_log"
else
  echo "missing"
fi

