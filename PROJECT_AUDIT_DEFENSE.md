# THU-BDC2026 股票量化项目审计与答辩复盘

> 目标：读完这一份文档，即使不重新翻代码，也能说明这个项目要解决什么问题、数据从哪里来、模型和策略如何运行、为什么可能赚钱、哪里有风险，以及答辩时老师可能追问什么。
>
> 依据：本文基于当前服务器 `/root/THU-BDC2026` 与 `/root/THU-BDC2026_review_package_20260802` 中真实文件、代码、数据样例、日志和输出结果整理。凡是不能完全确认的地方会明确标注“不确定”或“推测”。

---

## 0. 最重要的版本结论

这个项目不是一个干净的单版本仓库，而是一个比赛期间反复迭代的完整工作环境。里面同时存在：

1. 早期 Transformer baseline；
2. 中期 LightGBM/XGBoost/CatBoost 排序模型；
3. 后期正式三路 LightGBM 融合模型；
4. 更后期 Top20 到 Top5 的选择层、meta selector 和各种 walk-forward 实验；
5. 多个阶段的数据、输出、备份结果、日志和审计包。

因此答辩时不能简单说“项目就是 Transformer”或“项目就是一个 LightGBM”。更准确的说法是：

> 早期用 Transformer 做排序学习 baseline；后期经过多轮回测与失败分析，正式主线转向以 LightGBM LambdaRank 为核心的多因子横截面排序策略，并叠加事件数据、风险衰减层和 Top5 组合选择逻辑。

当前可复现正式入口：

- `train.sh`
- `test.sh`
- 二者都调用 `code/src/train_current_fusion_submission.py`

当前根目录最终输出：

- `output/result.csv`
- 内容为：

```csv
stock_id,weight
688506,0.250000
002837,0.220000
300442,0.200000
300450,0.180000
300274,0.150000
```

但存在一个重要差异：

- `model/current_submission/scored_top50.csv` 和 `model/current_submission/fusion_submission.joblib` 对应的是三路融合模型；
- `output/result.csv` 的时间戳晚于 `model/current_submission/scored_top50.csv`，且结果不同；
- `/root/THU-BDC2026_review_package_20260802/output/result.csv` 中的正式结果又是：

```csv
stock_id,weight
301165,0.240000
301236,0.220000
300896,0.200000
002558,0.180000
002353,0.160000
```

所以：

> 能确认的是：三路 LightGBM 融合 + 风险层是当前最完整、最可解释的正式主线。当前根目录 `output/result.csv` 则可能经过了后期 `final_selection.py` / `top20_meta_selector.py` 一类选择层覆盖。仅凭当前文件，无法 100% 确认最终真实提交给赛事方的是哪一个 result。

答辩时建议这样说：

> 项目最终阶段有两个层次：主体模型负责产生带风险诊断的候选排序；最终提交文件可能由后处理选择层在候选 Top20/Top50 中再筛选。我们的核心可解释策略仍以三路 LightGBM 排序模型和风险衰减层为主。

---

## 1. 项目全貌

### 1.1 比赛任务是什么

比赛要求在给定股票池中选择最多 5 只股票，并给出每只股票的权重。随后赛事方用未来 5 个交易日的开盘价收益来评价组合表现。

本地评分脚本依据：

- `test/score_self.py`
- `test/score_docker.py`

核心计算逻辑：

```python
return = (end['开盘'] - start['开盘']) / start['开盘']
final_score = sum(stock_return * weight)
```

也就是说：

> 如果选中某只股票，收益不是按收盘到收盘算，而是按测试期第一条记录的开盘价买入、最后一条记录的开盘价卖出。

### 1.2 输入是什么

主要输入包括：

1. 沪深300股票日频行情；
2. 估值、市值、换手率、涨跌停价等日频字段；
3. 同花顺热股榜；
4. 涨跌停明细；
5. 龙虎榜；
6. 融资融券；
7. 同花顺热门板块映射；
8. 股票元数据和行业信息。

核心数据文件：

- `data_submit/stock_data.csv`
- `data_submit/hs300_universe.csv`
- `data_submit/stock_metadata.csv`
- `model/extra_apr_data/ths_hot_stock.csv`
- `model/extra_apr_event_data/limit_list_d.csv`
- `model/extra_apr_event_data/top_list.csv`
- `model/extra_apr_event_data/margin_detail.csv`
- `model/extra_apr_event_data/ths_board_stock_hot.csv`

### 1.3 输出是什么

输出文件是：

- `output/result.csv`

格式：

```csv
stock_id,weight
688506,0.250000
002837,0.220000
300442,0.200000
300450,0.180000
300274,0.150000
```

要求：

- 最多 5 只股票；
- 必须有 `stock_id` 和 `weight` 两列；
- 权重和必须在 0 到 1 之间；
- 代码中实际通常满仓，也就是权重和约等于 1。

### 1.4 最终评价指标是什么

本地脚本的 Final Score 是：

```text
Final Score = Σ wi * ((P_open_end_i - P_open_start_i) / P_open_start_i)
```

其中：

- `wi` 是第 i 只股票权重；
- `P_open_start_i` 是测试期该股票第一条记录的开盘价；
- `P_open_end_i` 是测试期该股票最后一条记录的开盘价。

这不是 Sharpe，也不是年化收益，也不是最大回撤，而是单期 5 日持仓加权收益。

### 1.5 程序最后在预测什么

程序不是直接预测价格，也不是预测大盘涨跌。

程序核心是在预测：

> 同一个交易日横截面上，哪几只沪深300股票未来 5 个交易日的开盘到开盘收益更高。

正式主线用的是排序模型，所以它关心的是相对顺序：

```text
股票 A 未来收益是否应该排在股票 B 前面
```

最终决策：

1. 给每只股票打分；
2. 按分数排序；
3. 经过风险扣分和过滤；
4. 选 Top5；
5. 分配权重；
6. 输出 `result.csv`。

### 1.6 真实项目数据流

不是模板化的“清洗-建模-预测”一句话，真实链路如下：

```text
TuShare / BaoStock / AkShare 原始行情与事件数据
→ 股票代码标准化、日期解析、复权行情字段统一
→ 合并 stock_metadata 行业/市场信息
→ 合并同花顺热股、涨跌停、龙虎榜、融资融券、板块热度
→ 添加市场上下文特征
→ 过滤到沪深300目标股票池
→ 按股票构造多窗口历史因子
→ 构造未来5日开盘收益 label_return
→ 对每日横截面做 rank / z-score 标准化
→ 训练三路 LightGBM LambdaRank 排序模型
→ 三路分数转日内百分位后按 0.60/0.20/0.20 融合
→ Stage1 recall 分类器辅助召回候选
→ continuation_model 给持续强势股加分
→ decay_risk 风险层扣掉高位衰减风险
→ failure_model 扣掉过去强但未来可能失败的股票
→ close_strength / ret_20 / total_decay_risk 风险过滤
→ Stage1 Top50 候选池
→ 行业最多2只约束
→ Top5 选股
→ 动态权重或固定模板权重
→ 写 output/result.csv
→ 用 test/score_self.py 或 docker 评分脚本计算未来5日加权收益
```

---

## 2. 目录和核心代码

### 2.1 根目录核心文件

#### `train.sh`

作用：

- 当前正式训练入口。

关键逻辑：

- 默认数据路径先设为 `./data/stock_data`；
- 如果存在 `./data_submit/stock_data.csv` 且不存在 `./data/stock_data` 目录，则改用 `./data_submit/stock_data.csv`；
- 调用：

```bash
python code/src/train_current_fusion_submission.py --mode train ...
```

它不是早期 README 里描述的 `code/src/train.py` Transformer 训练，而是后期融合模型训练。

#### `test.sh`

作用：

- 当前正式预测入口。

关键逻辑：

- 数据选择逻辑与 `train.sh` 一样；
- 调用：

```bash
python code/src/train_current_fusion_submission.py --mode predict ...
```

输出：

- `output/result.csv`
- `model/current_submission/scored_top50.csv`

### 2.2 正式主线代码

#### `code/src/train_current_fusion_submission.py`

这是当前最重要的策略文件。

负责：

- 读取数据；
- 合并事件数据；
- 构造三路模型；
- 训练 LightGBM Ranker；
- 训练 Stage1 recall、continuation、failure 分类器；
- 融合分数；
- 应用风险扣分；
- 选 Top5；
- 写出最终 result。

关键变量：

```python
PROFILES = [
    ("base_rank", "rank", 0.60, False, False, False),
    ("event_text_rank", "rank", 0.20, True, True, True),
    ("event_board_rank", "rank", 0.20, True, True, False),
]
```

含义：

- `base_rank`：只使用基础行情/估值/横截面特征，不使用额外事件数据；
- `event_text_rank`：使用热榜、事件、板块、文本关键词；
- `event_board_rank`：使用热榜、事件、板块，但不使用文本关键词；
- 三者权重为 0.60 / 0.20 / 0.20。

关键函数：

- `_train_one()`：训练一个 LightGBM ranker；
- `_train_stage1_recall()`：训练候选召回分类器；
- `_train_continuation_model()`：训练强势延续分类器；
- `_train_failure_model()`：训练强势失败分类器；
- `predict()`：加载 artifact，打分、融合、风险调整、选股、输出。

在整体链路中的位置：

> 这是正式策略主控脚本。

#### `code/src/advanced/features.py`

负责：

- 数据列名定义；
- 数据读取；
- 股票池过滤；
- 市场上下文；
- 股票级滚动特征；
- 横截面标准化；
- 行业相对特征；
- 未来收益标签。

关键函数：

- `load_price_data()`
- `filter_universe()`
- `add_market_context()`
- `enrich_with_metadata()`
- `_build_stock_features()`
- `add_forward_labels()`
- `build_feature_frame()`
- `filter_trainable_rows()`

最重要标签定义：

```python
open_t1 = grouped[OPEN_COL].shift(-1)
open_t5 = grouped[OPEN_COL].shift(-cfg.horizon)
df[LABEL_COL] = open_t5 / (open_t1 + 1e-12) - 1.0
```

解释：

> 用下一交易日开盘买入，用未来第 5 个交易日开盘卖出。

#### `code/src/advanced/tree_ranker.py`

负责：

- LightGBM / XGBoost / CatBoost 排序模型；
- relevance 标签转换；
- rolling validation；
- 输出 OOF 分数和验证指标。

正式主线主要用 `_fit_lightgbm()`：

```python
LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
)
```

relevance 的默认 `rank` 模式：

- 每天未来收益第 1 名相关性最高；
- 第 2、3、4、5 名递减；
- 第 5 名之后为 0。

这和比赛“选前 5 只股票”的目标一致。

#### `code/src/advanced/decay_risk.py`

负责：

- 识别“过去很强但可能开始衰退”的股票；
- 给强势衰减、高位反转、事件退潮、市场弱势等打风险分；
- 对最终排序分数扣分；
- 生成动态仓位模板。

关键风险分数：

- `strength_decay_score`
- `runup_reversal_score`
- `event_proxy_risk`
- `theme_lifecycle_risk`
- `market_regime_score`
- `total_decay_risk`

总风险：

```python
total_decay_risk =
0.34 * strength_decay_score
+ 0.27 * runup_reversal_score
+ 0.18 * event_proxy_risk
+ 0.11 * theme_lifecycle_risk
+ 0.10 * market_regime_score
```

风险扣分：

```python
score = model_score_rank - penalty_weight * risk_mix
```

#### `code/src/advanced/metrics.py`

负责：

- TopK 组合选择；
- 等权、softmax、risk_adjusted 权重；
- 内部回测指标。

内部评估：

```python
portfolio_return = 策略组合平均收益
oracle_return = 真实Top5平均收益
random_return = 全股票平均收益
final_score = (portfolio - random) / (oracle - random)
```

注意：

> 这个内部 final_score 不是比赛官方最终分，而是为了衡量策略距离理论最优 Top5 还有多远。

### 2.3 后处理选择层

#### `code/src/final_selection.py`

负责：

- 不重新训练主体模型；
- 读取历史验证分数和当前候选分数；
- 比较 Top5、分层选、行业多样性、meta 模型等策略；
- 选一个历史表现最好的策略输出。

权重固定：

```python
WEIGHTS = [0.25, 0.22, 0.20, 0.18, 0.15]
```

当前根目录 `output/result.csv` 很可能是这种后处理结果。

#### `code/src/top20_meta_selector.py`

负责：

- 从已有 Top20 候选中训练轻量分类器/排序器；
- 目标是预测候选是否属于未来真实 Top10 / Top5；
- 比较 HGB、Logistic、ExtraTrees、RandomForest、LightGBM 等模型。

日志显示：

- 14 周短窗有提升；
- 40 周验证不稳定；
- 不能视为稳定最终解。

### 2.4 早期 baseline

#### `code/src/model.py`

定义早期 `StockTransformer`：

- 单股票时间序列 Transformer；
- 时间维 FeatureAttention；
- 股票间 CrossStockAttention；
- 输出每只股票排序分数。

#### `code/src/train.py`

训练早期 Transformer 排序模型。

损失：

- listwise softmax 交叉熵；
- pairwise 排序损失；
- Top5 样本加权。

#### `code/src/utils.py`

早期技术指标和 Alpha158 风格特征：

- 39 个技术指标；
- 158 类 Alpha 特征；
- `158+39` 合并特征。

答辩时要注意：

> 这些是历史尝试，不是后期最强正式主线。

---

## 3. 数据详解

### 3.1 数据来源

项目中出现的数据源/API：

1. BaoStock：
   - `get_stock_data.py`
   - 用于全 A 股日线、元数据抓取；
   - 默认后复权。

2. TuShare：
   - `update_hs300_tushare.py`
   - `update_tushare_daily_basic.py`
   - `update_tushare_ths_hot.py`
   - `update_tushare_extra_factors.py`
   - `update_tushare_limits.py`
   - `update_tushare_moneyflow.py`

3. AKShare：
   - `update_hs300_tushare.py` 用 `ak.index_stock_cons_csindex(symbol="000300")` 获取沪深300成分；
   - 日志中提到 AKShare 网络环境不稳定，所以后期核心行情更多依赖 TuShare。

### 3.2 股票池范围

正式提交数据：

- `data_submit/hs300_universe.csv`
- 300 只沪深300股票；
- 更新日期 2026-07-31；
- 包含字段：
  - `updateDate`
  - `code`
  - `股票名称`
  - `code_name`
  - `纯代码`
  - `市场`
  - `ts_code`

训练脚本通过：

```python
filter_universe(full_df, maybe_load_universe(args.target_universe))
```

把数据过滤到目标股票池。

### 3.3 时间范围

实际文件：

1. `data_submit/stock_data.csv`
   - 300 只沪深300；
   - 2024-01-02 到 2026-07-31；
   - 185868 行；
   - 29 列。

2. `data_submit_long/stock_data.csv`
   - 300 只；
   - 2018-01-02 到 2026-07-31；
   - 575823 行；
   - 36 列。

3. `data/stock_data.csv`
   - 全 A 股约 5200 只；
   - 2024-01-02 到 2026-06-26；
   - 2939974 行；
   - 23 列。

4. `data/train.csv`
   - 早期本地训练集；
   - 2024-01-02 到 2026-03-06；
   - 300 只；
   - 156590 行。

5. `data/test.csv`
   - 早期本地测试集；
   - 2026-03-09 到 2026-03-13；
   - 300 只；
   - 1500 行。

### 3.4 每一行是什么

在主行情文件中：

> 每一行 = 某只股票在某一个交易日的一条日频记录。

例如 `data_submit/stock_data.csv` 中一行包含：

- 股票代码；
- 股票名称；
- 市场；
- 日期；
- 开盘、收盘、最高、最低、昨收；
- 成交量、成交额；
- 涨跌幅、涨跌额、振幅；
- 换手率、自由流通换手率、量比；
- PE/PB/PS；
- 总市值、流通市值；
- 涨停价、跌停价；
- 复权信息。

### 3.5 是否复权

是。

主要行情文件里：

- `复权标记 = 1`
- `复权方式 = 后复权`

TuShare 抓取脚本：

```python
ts.pro_bar(..., adj="hfq")
```

`hfq` 即后复权。

### 3.6 缺失值如何处理

数据读取：

- 数值列转为 numeric；
- 无法解析的变 NaN。

特征阶段：

```python
featured[feature_cols] = featured[feature_cols].replace([np.inf, -np.inf], np.nan)
featured[feature_cols] = featured.groupby(ID_COL)[feature_cols].ffill()
featured[feature_cols] = featured[feature_cols].fillna(0.0)
```

训练过滤：

- 标签为空的样本丢弃；
- 历史长度不足 `min_history=60` 的样本丢弃；
- 特征非有限值的样本丢弃。

### 3.7 停牌、涨跌停如何处理

项目没有完整模拟停牌和涨跌停的真实成交约束。

已有处理：

- `交易状态` 作为原始字段存在；
- `涨停价`、`跌停价` 被用于构造距离涨停、触及涨停、近窗口涨停次数等特征；
- `limit_list_d.csv` 提供涨跌停事件；
- 但最终回测/评分没有判断“买不进去”或“卖不出去”。

这是答辩时必须承认的局限。

### 3.8 是否存在未来数据泄漏风险

代码层面有防护：

- 标签用未来开盘收益；
- 训练时要求 `label_known_date <= latest_date`；
- 特征主要由当前及历史滚动窗口构成。

但仍有潜在风险：

1. 使用当前沪深300成分股回测历史，可能有幸存者偏差；
2. 同花顺热榜的 `rank_time` 多为晚间，如果决策时间早于热榜发布时间，则有时点争议；
3. 事件文本里可能含公告/新闻总结，必须确认比赛规则是否允许外部事件数据；
4. 横截面市场特征使用同一天全股票收盘后数据，如果比赛要求盘前提交，则也有时点风险；如果允许收盘后预测下一交易日开盘，则较合理。

---

## 4. 特征工程 / 因子

正式主线特征由 `code/src/advanced/features.py` 生成。

### 4.1 收益率 / 动量

代表特征：

- `f_ret_1`
- `f_close_ret_3`
- `f_close_ret_5`
- `f_close_ret_10`
- `f_close_ret_20`
- `f_close_ret_30`
- `f_close_ret_60`
- `f_close_ret_120`
- `f_mom_5_minus_20`
- `f_mom_20_minus_60`

数学定义：

```text
f_close_ret_w = close_t / close_{t-w} - 1
```

代码：

```python
feature_data[f"f_close_ret_{w}"] = close.pct_change(w)
```

金融含义：

- 描述过去 w 日累计涨跌；
- 如果短期涨幅高，可能代表资金强势；
- 如果涨幅过高，也可能代表追高风险。

越大越好吗？

- 在强趋势阶段通常越大越好；
- 在短期过热阶段不一定越大越好，所以后面加了 `decay_risk`。

标准化：

- 每日横截面生成 `_cs_z` 和 `_cs_rank`。

### 4.2 波动率 / 风险

代表特征：

- `f_ret_std_w`
- `f_ret_down_std_w`
- `f_atr_pct_w`
- `risk_vol_20`
- `risk_vol_60`

数学定义：

```text
f_ret_std_w = std(daily_return, w)
f_ret_down_std_w = std(min(daily_return, 0), w)
ATR_pct_w = mean(true_range, w) / close
```

代码位置：

- `features.py` 中滚动窗口；
- `true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))`。

金融含义：

- 波动率高说明股票活跃，可能有短期机会；
- 但也说明风险大，容易冲高回落。

特征重要性：

- 当前模型最重要特征里大量是振幅和 ATR。
- `combined_feature_importance.csv` 前几名包括：
  - `f_amplitude_mean_20_cs_z`
  - `f_atr_pct_3_cs_z`
  - `f_amplitude_mean_10_cs_rank`

说明模型真正很依赖“活跃/波动/振幅”类信号。

### 4.3 成交量 / 流动性

代表特征：

- `f_log_volume`
- `f_log_amount`
- `f_volume_rel_w`
- `f_amount_rel_w`
- `f_turnover`
- `f_turnover_mean_w`
- `f_amount_rel_5_minus_20`

数学定义：

```text
f_amount_rel_w = amount_t / mean(amount, w)
f_volume_rel_w = volume_t / mean(volume, w)
```

金融含义：

- 放量代表资金关注；
- 缩量可能代表热度下降；
- 短期放量上涨可能是强势延续；
- 放量下跌则是风险信号。

在风险层中：

- `amount_rel_20` 被用于识别放量；
- `high_volume_down` 表示放量下跌；
- `volume_price_divergence` 表示量价背离。

### 4.4 技术形态 / 价格位置

代表特征：

- `f_gap`
- `f_intraday`
- `f_high_low_pct`
- `f_upper_shadow`
- `f_lower_shadow`
- `f_channel_pos_w`
- `f_max_high_dist_w`
- `f_min_low_dist_w`
- `f_ma_close_rel_w`
- `f_ema_close_rel_w`

含义：

- `gap`：跳空；
- `intraday`：日内涨跌；
- `upper_shadow`：上影线；
- `channel_pos`：收盘处于近 w 日价格通道的位置；
- `ma_close_rel`：均线相对当前价格的偏离。

金融解释：

- 收盘接近高点、价格位于通道高位，通常代表强势；
- 长上影线、跌破均线，可能代表冲高失败。

### 4.5 涨跌停相关

代表特征：

- `f_close_to_up_limit`
- `f_hit_up_limit`
- `f_close_near_up_limit`
- `f_hit_up_limit_count_w`
- `f_close_near_up_limit_count_w`
- `f_limit_list_up`
- `f_limit_list_down`

金融含义：

- 涨停/接近涨停代表短期资金强度；
- 连续涨停也可能带来退潮风险。

### 4.6 估值 / 市值

代表特征：

- `f_pe_ttm`
- `f_log_pe_ttm`
- `f_pb`
- `f_ps_ttm`
- `f_total_mv`
- `f_log_total_mv`
- `f_circ_mv`
- `f_log_circ_mv`

金融含义：

- 市值、估值代表风格暴露；
- 某些市场阶段偏好小盘、高弹性、高估值或低估值修复。

特征重要性：

- `f_log_pe_ttm_cs_z`
- `f_total_mv`
- `f_circ_mv_cs_z`

都进入了重要性前列。

### 4.7 市场上下文 / Beta / Alpha

市场上下文特征：

- `f_mkt_ret_mean`
- `f_mkt_ret_median`
- `f_mkt_ret_std`
- `f_mkt_up_ratio`
- `f_mkt_amount_log`

相关性特征：

- `f_beta_mkt_w`
- `f_corr_mkt_w`
- `f_alpha_mkt_w`
- `f_excess_ret_mean_w`

含义：

- 判断股票相对市场是强还是弱；
- Beta 高的股票对市场更敏感；
- Alpha 表示扣除市场影响后的独立强弱。

注意：

> 项目没有用相关性筛选因子，只是把相关性和 beta 当作特征。

### 4.8 事件 / 热榜 / 题材因子

同花顺热股：

- `f_ths_hot_listed`
- `f_ths_hot_rank_inv`
- `f_ths_hot_rank_score`
- `f_ths_hot_listed_count_w`
- `f_ths_hot_rank_score_mean_w`
- `f_ths_hot_value_log`

定义：

```text
hot_rank_score = max(0, 101 - rank) / 100
```

含义：

- 排名越靠前，热度分越高；
- 过去窗口内上榜次数越多，说明题材持续被关注。

龙虎榜：

- `f_top_listed`
- `f_top_net_amount_rel`
- `f_top_listed_count_w`
- `f_top_net_amount_sum_rel_w`

融资融券：

- `f_margin_fin_balance_log`
- `f_margin_fin_balance_chg_5`
- `f_margin_fin_balance_chg_20`

板块热度：

- `f_ths_board_hot_count`
- `f_ths_board_best_rank_score`
- `f_ths_board_rank_score_sum`
- `f_ths_board_hot_log_max`
- `f_ths_board_pct_change_mean`

### 4.9 事件文本关键词

如果 `add_event_text=True`，使用 `ths_hot_reason + ths_hot_concept` 生成关键词特征：

- 风险词：停火、风险、监管、减持、亏损、下滑、终止、回撤等；
- 正向词：业绩预增、订单、中标、量产、获批、合作、受益、突破、创新药、算力、芯片等；
- 主题词：
  - 地缘/航运；
  - AI/芯片；
  - 医药；
  - 能源。

代表特征：

- `f_event_text_risk`
- `f_event_text_positive`
- `f_event_text_net`
- `f_event_text_geo_shipping_risk`
- `f_event_text_ai_chip`

结论：

- 文本特征作为辅助有一些价值；
- 单独使用不强；
- 无监督 SVD 语义实验没有进入正式提交。

### 4.10 横截面标准化

这是项目非常关键的一步。

对每个交易日、每个特征，生成：

```text
cs_z = (x - 当日均值) / 当日标准差
cs_rank = 当日百分位排名
```

原因：

- 策略做的是当日股票之间比较；
- 原始数值尺度不同，不便于模型学习；
- rank 比原值更稳健，减少极端值影响。

---

## 5. 赋权与综合评价

### 5.1 是否使用熵权法

没有发现熵权法实现。

全库搜索到的 `entropy` 只出现在 `portfolio_model.py` 的神经网络组合损失里：

```python
entropy = -(weights * log(weights)).sum()
loss = -return + risk_penalty + concentration - entropy_beta * entropy
```

它不是熵权法，而是鼓励组合不要过度集中的熵正则。

### 5.2 是否使用 CRITIC

没有发现 CRITIC 方法实现。

### 5.3 是否使用 TOPSIS

没有发现 TOPSIS 方法实现。

### 5.4 是否使用 PCA

没有发现 PCA。

但存在 TruncatedSVD：

- `code/src/build_event_semantic_factors.py`
- 用 TF-IDF + SVD 压缩同花顺热榜文本；
- 输出 `event_semantic_factors.csv`；
- 日志显示实验效果不佳，未进入正式融合脚本。

### 5.5 是否使用相关性筛选

没有发现系统性相关性筛选。

存在 `rolling.corr()`、beta、市场相关性因子，但它们是输入特征，不是筛选方法。

### 5.6 是否使用 IC / RankIC

没有发现 IC 或 RankIC 计算。

答辩可以说：

> 项目没有用传统因子研究里的 IC/RankIC 来筛选因子，而是用 LightGBM 的排序目标和回测结果间接评价特征有效性。

### 5.7 实际使用的赋权

项目真实使用的是人工设定 + 回测搜索：

1. 三路模型融合：
   - base_rank：0.60
   - event_text_rank：0.20
   - event_board_rank：0.20

2. 风险层：
   - strength_decay：0.34
   - runup_reversal：0.27
   - event_proxy：0.18
   - theme_lifecycle：0.11
   - market_regime：0.10

3. 最终股票仓位：
   - 动态模板；
   - 或固定模板 `0.25,0.22,0.20,0.18,0.15`。

---

## 6. 机器学习 / 预测模型

### 6.1 正式主模型：LightGBM LambdaRank

模型：

- `LGBMRanker`
- objective：`lambdarank`
- metric：`ndcg`

输入：

- 每只股票在某个交易日的一行宽表因子；
- 特征数取决于 profile：
  - 当前 artifact 中 `base_rank` 约 795 个；
  - `event_text_rank` 约 1290 个；
  - `event_board_rank` 约 1176 个。

标签：

```text
label_return = open_t5 / open_t1 - 1
```

训练目标：

- 不是拟合收益率数值；
- 而是优化同一交易日内股票排序；
- 希望未来收益高的股票被排到前面。

超参数：

```text
n_estimators = 180
learning_rate = 0.04
num_leaves = 31
max_depth = 5
min_child_samples = 50
subsample = 0.80
colsample_bytree = 0.75
reg_alpha = 0.15
reg_lambda = 1.2
```

### 6.2 Stage1 Recall 分类器

目标：

```text
future_rank <= 20 或 label_return >= 6%
```

含义：

> 它不是直接选最终 5 只，而是帮助找出更可能进入强势候选池的股票。

模型：

- LightGBM `LGBMClassifier`

作用：

- 和主排序分数融合；
- 默认权重约 0.18。

### 6.3 Continuation Model

训练对象：

- 过去已经强势或活跃的股票。

正样本：

```text
future_rank <= 10 或 label_return >= 6%
```

含义：

> 判断一个已经强的股票未来是否还能继续强。

### 6.4 Failure Model

训练对象：

- 过去强势或活跃的股票。

正样本：

```text
未来跑输市场，或未来收益为负，或未来排名较弱且收益低于3%
```

含义：

> 它专门识别“看起来很强，但接下来可能失效”的股票。

预测阶段：

```text
score = rank(score) - failure_weight * rank(failure_score)
```

### 6.5 早期 Transformer

早期模型 `StockTransformer` 包含：

- 输入投影；
- 位置编码；
- TransformerEncoder；
- 时间维注意力；
- 股票间注意力；
- 排序分数输出。

训练损失：

- listwise softmax；
- pairwise 排序；
- Top5 样本加权。

但后期日志和文件显示，正式主线已经转向 LightGBM。

---

## 7. 选股逻辑

### 7.1 正式主线如何从 300 只股票筛出 5 只

1. 读取最新交易日候选股票；
2. 三路模型分别打分；
3. 每路分数转成当日百分位 rank；
4. 加权融合：

```text
model_score = 0.6 * base_rank_pct
            + 0.2 * event_text_rank_pct
            + 0.2 * event_board_rank_pct
```

5. Stage1 recall 融合：

```text
alpha_score = 0.82 * model_rank + 0.18 * recall_rank
```

6. continuation 加分；
7. decay risk 扣分；
8. failure model 扣分；
9. 过滤风险过高或收盘弱的股票：

```text
close_strength > 0.25
ret_20 > -0.25
total_decay_risk < 0.78
```

10. 从模型 Top50 和召回 Top50 的并集中保留候选；
11. 按最终 score 排序；
12. 行业最多 2 只；
13. 取 Top5。

### 7.2 是否考虑行业

考虑，但分两层：

1. 特征层：
   - 行业相对 `_ind_z/_ind_rank` 有代码；
   - 正式默认关闭；
   - 因为日志显示行业相对特征有时会加噪。

2. 组合层：
   - 默认 `max_per_industry=2`；
   - 避免同一行业过度集中。

### 7.3 是否考虑相关性

没有做股票之间收益相关性矩阵的组合优化。

只使用了：

- 股票收益与市场收益的滚动相关；
- beta；
- 行业约束。

### 7.4 是否做风险控制

有风险控制，但不是传统组合波动率优化。

风险控制包括：

- 高位衰减风险；
- 追高反转风险；
- 事件退潮风险；
- 市场弱势风险；
- 行业最多 2 只；
- 动态权重降低高风险 Top1 的集中度。

---

## 8. 仓位和交易

### 8.1 等权还是按分数配权

历史上都尝试过：

- 等权；
- risk_adjusted；
- softmax；
- 固定模板；
- 动态模板。

当前根目录 `output/result.csv` 是固定递减模板：

```text
0.25, 0.22, 0.20, 0.18, 0.15
```

review package 中正式结果是：

```text
0.24, 0.22, 0.20, 0.18, 0.16
```

三路融合脚本中的动态模板由 `confidence_weight_template()` 决定：

- 分数差距大、风险低时更集中；
- 分数差距小、风险高时更分散。

### 8.2 一次买多少只股票

最多 5 只。

代码默认取 5 只。

### 8.3 调仓频率和持有期

比赛评分相当于：

- 提交一次组合；
- 持有未来 5 个交易日；
- 用开盘到开盘收益评分。

walk-forward 回测中模拟的是每周 cutoff 后的 5 日持有。

### 8.4 买卖价格

买入价格：

- 未来第 1 个交易日开盘价。

卖出价格：

- 未来第 5 个交易日开盘价。

### 8.5 手续费、滑点、涨跌停、停牌

没有完整计算：

- 手续费：无；
- 滑点：无；
- 涨跌停无法买卖：无真实约束；
- 停牌无法交易：无真实约束；
- 止损止盈：无；
- 做空：无；
- 空仓：评分允许权重和小于 1，但策略实际基本满仓。

这是真实交易与比赛回测的重要差距。

---

## 9. 回测与比赛指标

### 9.1 比赛官方/本地指标

本地评分脚本直接计算：

```text
预测组合未来5日开盘加权收益
```

没有年化，没有 Sharpe，没有最大回撤。

### 9.2 项目内部指标

内部指标主要来自 `advanced.metrics.evaluate_scored_frame()`：

- `portfolio_return`：策略 Top5 组合收益；
- `oracle_return`：真实未来 Top5 平均收益；
- `random_return`：全股票平均收益；
- `final_score`：策略相对随机与理论最优之间的位置。

公式：

```text
final_score = (portfolio_return - random_return) / (oracle_return - random_return)
```

### 9.3 服务器上能确认的结果

本地 docker 模拟：

- `test/result.csv`
- `Final Score = 0.0251794912169185`

14 周验证：

- `model/two_stage_decay_recent14/summary.csv`
- `two_stage_no_industry_relative`
  - best policy return：约 4.56%
  - Top5 equal return：约 3.70%

40 周验证：

- `model/two_stage_failure_recent40_no_industry/summary.csv`
  - best policy return：约 2.06%
  - Top5 equal return：约 1.68%

核心 ablation：

- `feature_ablation_recent14_core/no_event`
  - Top5 equal return：约 4.12%
- `feature_ablation_recent14_core/all`
  - Top5 equal return：约 2.12%

说明：

> 简单把事件数据全量堆进去不一定更好，反而可能加噪。

事件文本：

- `all_event_text`
  - best policy return：约 4.00%
- `only_event_text`
  - Top5 equal return：约 1.51%

说明：

> 文本不能单独作为主模型，但可能作为辅助有用。

---

## 10. 策略为什么可能赚钱

不能说“因为模型预测准确”。更准确的解释是：

这个策略真正押注的是：

> 沪深300内部短周期强势延续和资金关注度延续。

它认为未来 5 日表现好的股票，往往在当前或过去一段时间已经表现出某些迹象：

- 振幅和 ATR 提升；
- 成交额/换手率放大；
- 近期收益强于同日其它股票；
- 收盘位置较强；
- 接近涨停或曾触及涨停；
- 行业相对强；
- 热榜/题材/龙虎榜/融资融券显示关注度；
- 估值和市值处于某种当期市场偏好的风格。

同时，策略也知道“强势股容易退潮”，所以加了风险层：

- 高位但最近走弱；
- 放量下跌；
- 上影线增多；
- 热榜热度下降；
- 行业扩散变差；
- 市场整体转弱。

所以它不是纯动量，也不是纯反转，而是：

```text
多因子横截面排序
+ 短周期动量/量价强势
+ 题材热度辅助
+ 高位衰减风险控制
```

给量化研究员的三句话：

> 我们的 alpha 来源是沪深300横截面里的短期资金关注和强势延续。模型用多窗口量价、波动、成交、估值、市值、行业相对强弱和事件热度特征，学习未来5日收益排名。后处理风险层扣掉高位衰减、放量转弱和题材退潮的股票，最后形成 Top5 组合。

---

## 11. 失败和局限

### 11.1 过拟合风险

特征数量非常多：

- base 几百个；
- event profile 上千个；
- 但真正可验证的周样本只有几十周。

14 周验证表现好，不代表 40 周稳定。

日志中已经明确：

- 14 周一些策略可到 5% 左右；
- 40 周经常下降到 1%-2%；
- Top20 选择层没有稳定打赢主体模型。

### 11.2 幸存者偏差

回测可能使用当前沪深300成分股去回看历史，而不是当时真实成分股。

这会导致：

- 历史中已经被剔除的弱股不在样本里；
- 回测收益可能偏乐观。

### 11.3 未来数据泄漏风险

代码对标签有防护，但仍要注意：

- 同花顺热榜发布时间；
- 事件文本可用时间；
- 市场上下文是否要求收盘后才能知道；
- 比赛实际提交时点是否允许这些信息。

### 11.4 特征相关性过高

许多特征是同一类量价数据不同窗口变形：

- 3/5/10/20/30/60/120 日收益；
- 多窗口成交量；
- 多窗口波动；
- 多窗口 rank/z-score。

没有做 IC 稳定性筛选，也没有系统去相关。

### 11.5 事件数据未充分发挥

特征重要性显示：

- 主导特征仍是振幅、ATR、估值、市值、行业相对强弱；
- 事件/text 特征很多 zero gain；
- `event_text_rank` 中零增益特征约 478 个；
- `event_board_rank` 中零增益特征约 392 个。

说明：

> 事件数据接入了，但模型主要还是在吃量价波动类信号。

### 11.6 Top5 目标没有直接优化

LambdaRank 优化排序质量，不直接优化：

```text
最终5只股票的加权组合收益
```

因此可能出现：

- Top1 很强；
- Top2 到 Top5 质量不稳定；
- Top20 里有强股，但最终排序排不准。

### 11.7 真实交易不可直接套用

没有考虑：

- 手续费；
- 滑点；
- 涨停买不进；
- 跌停卖不出；
- 停牌；
- 市场冲击；
- 仓位容量。

所以这是比赛回测策略，不是可直接实盘策略。

### 11.8 为什么有些强势股票没选出来

可能原因：

1. 它们事件驱动明显，但 K 线特征没有提前体现；
2. 热榜数据滞后或没覆盖；
3. 模型认为它们高位衰减风险太高；
4. 行业约束限制了同板块股票；
5. 事件文本关键词没有抓住真实催化；
6. 当前市场风格和训练期不同。

---

## 12. 答辩知识清单

### A. 必须会

1. 股票收益率：
   - `(卖出价 - 买入价) / 买入价`
   - 本项目用开盘到开盘。

2. 复权：
   - 解决分红、送股、拆股导致的价格不连续；
   - 本项目主要用后复权。

3. 因子：
   - 用于解释或预测未来收益的变量；
   - 如动量、波动、成交量、估值、市值。

4. 横截面排序：
   - 同一天比较不同股票；
   - 不是预测单只股票涨跌。

5. 标准化：
   - `_cs_z`：同日 z-score；
   - `_cs_rank`：同日百分位排名。

6. LightGBM：
   - 梯度提升树；
   - 适合表格数据；
   - 本项目用 LambdaRank 排序目标。

7. NDCG / 排序学习：
   - 排名前面的股票更重要；
   - 与 Top5 选股目标接近。

8. 回测：
   - 用历史数据模拟策略；
   - 项目用 weekly walk-forward。

9. 数据泄漏：
   - 使用了当时不可能知道的信息；
   - 项目用 `label_known_date` 避免标签泄漏，但仍有事件时点风险。

10. 过拟合：
   - 在历史样本上调得很好，未来失效；
   - 本项目 14 周和 40 周差异就是例子。

### B. 深挖时应该会

1. 动量 vs 反转；
2. ATR 和波动率；
3. 成交量放大；
4. 行业相对强弱；
5. Beta、Alpha、相关系数；
6. LambdaRank 与普通回归区别；
7. Stage1 recall 为什么存在；
8. continuation/failure 模型为什么只看过去强势股票；
9. 衰减风险层各部分含义；
10. 为什么事件文本单独效果不好；
11. 为什么 Top1 强但 Top5 弱；
12. 为什么要做 walk-forward 而不是随机切分；
13. 为什么当前沪深300回测可能有幸存者偏差；
14. 为什么真实交易要考虑涨跌停和滑点。

### C. 高级扩展，知道即可

1. IC / RankIC：
   - 衡量因子与未来收益的相关性；
   - 本项目没有实现。

2. PCA / SVD：
   - 降维；
   - 本项目只用 SVD 做过事件文本语义实验。

3. 熵权法、CRITIC、TOPSIS：
   - 多指标赋权/综合评价方法；
   - 本项目没有使用。

4. 组合优化：
   - 均值方差、风险预算、约束优化；
   - 本项目主要用模板权重和简单风险调整。

5. Sharpe Ratio：
   - 超额收益 / 波动；
   - 本项目不是官方指标。

6. 最大回撤：
   - 从高点到低点的最大跌幅；
   - 项目选择层有用单周最差收益近似风险。

---

## 13. 模拟老师提问

### 基础问题

1. 这个比赛的任务是什么？
2. 你们的输入数据有哪些？
3. 输出文件格式是什么？
4. 最终分数是怎么计算的？
5. 为什么最多只能选 5 只股票？
6. 为什么用开盘价计算收益？
7. 股票池为什么是沪深300？
8. 你们的数据是日频还是分钟频？

### 数据问题

9. 你们的数据来自哪里？
10. 什么是后复权？为什么要复权？
11. 数据里包含哪些字段？
12. 是否使用了新闻或文本？
13. 停牌和涨跌停怎么处理？
14. 有没有缺失值？怎么处理？
15. 训练集和测试期怎么划分？
16. 有没有未来数据泄漏？

### 特征问题

17. 你们最重要的因子是什么？
18. 动量因子怎么计算？
19. 波动率因子怎么计算？
20. 成交量因子想表达什么？
21. 热榜特征怎么转成数值？
22. 文本关键词特征怎么做？
23. 为什么要做横截面 rank？
24. 行业因素怎么考虑？

### 模型问题

25. 为什么用 LightGBM？
26. LambdaRank 和回归有什么区别？
27. 标签 `label_return` 是什么？
28. NDCG 为什么适合这个任务？
29. 三路模型分别做什么？
30. Stage1 recall 有什么用？
31. failure model 为什么预测“失败”？
32. continuation model 解决什么问题？

### 选股和仓位问题

33. 最终怎样从 300 只选出 5 只？
34. 排序依据是什么？
35. 为什么行业最多 2 只？
36. 权重为什么不是等权？
37. 当前结果为什么是这 5 只？
38. 如果 Top1 很强，为什么不全仓？
39. 有没有止损止盈？
40. 有没有考虑交易成本？

### 回测和质疑问题

41. 回测结果最好是多少？
42. 为什么 14 周表现好，40 周变差？
43. 有没有过拟合？
44. 有没有幸存者偏差？
45. 事件数据为什么加了反而可能变差？
46. 如果真实交易买不进去怎么办？
47. 为什么不用熵权法/CRITIC/TOPSIS？
48. 有没有做 IC/RankIC？
49. 这个策略真正的 alpha 是什么？
50. 如果让你继续改，最应该改哪里？

---

## 14. 推荐答辩主线

如果老师让你整体介绍项目，可以按这个顺序讲：

1. 这是一个沪深300未来 5 日收益 Top5 选股问题；
2. 官方输出是最多 5 只股票和权重，评分是未来 5 日开盘到开盘加权收益；
3. 我们的数据以日频行情为核心，补充估值、市值、热榜、涨跌停、龙虎榜、融资融券和板块热度；
4. 特征工程把每只股票的历史量价、波动、成交、价格位置、估值、市值、事件热度做成多窗口因子；
5. 每天再做横截面 rank 和 z-score，因为目标是同一天股票之间排序；
6. 主模型是 LightGBM LambdaRank，直接学习未来 5 日收益的横截面排序；
7. 三路模型分别负责基础量价、事件文本、事件板块，分数按 0.6/0.2/0.2 融合；
8. 后面加 recall、continuation、failure 和 decay risk 层，解决强势股延续与退潮问题；
9. 最后经过风险过滤、行业约束和 Top5 权重模板输出结果；
10. 策略本质是短周期强势延续 + 题材热度辅助 + 高位衰减风险控制；
11. 局限是样本少、特征多、事件时点不确定、没有真实交易成本和涨跌停成交约束，且 14 周和 40 周回测稳定性差异明显。

---

## 15. 一句话总结

这个项目本质上是：

> 一个面向沪深300的短周期多因子横截面排序选股系统，用 LightGBM LambdaRank 学习未来 5 日开盘收益排名，用量价/波动/成交/估值/事件热度刻画资金关注和强势延续，再用衰减风险层和选择层尽量避开高位退潮股票，最终输出 5 只股票及权重。

