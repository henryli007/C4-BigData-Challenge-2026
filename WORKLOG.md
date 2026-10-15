# Worklog

## 2026-06-28

### 今日工作
- 将 `/root/UAV-RGBT-Detection/data` 和 `/root/UAV-RGBT-Detection/outputs` 迁移到 `/root/autodl-tmp/UAV-RGBT-Detection/`，并在原路径保留符号链接，系统盘可用空间从约 3.7G 提升到约 15G。
- 更新沪深300数据到 2026-06-26：`data_submit/stock_data.csv`，当前最新交易日覆盖 300 只股票，抓取无失败。
- 更新当前沪深300股票池：`data_submit/hs300_universe.csv`，成分更新时间为 2026-06-22。
- 生成训练用混合面板：`data/hybrid_stock_data.csv`，保留全市场历史上下文，并用最新沪深300数据覆盖到 2026-06-26。
- 训练最新 conservative tree ranker：`model/ranker_hybrid_20260626/tree_ranker.joblib`。
- 生成并校验正式提交：`output/result.csv`。
- 启动全市场增量抓取后台任务：`screen -r bdc_fetch_all`，日志为 `logs/fetch_all_screen_20260628.log`。

### 本轮结果
- 正式提交股票与权重：
  - `000657`: `0.3294140776145476`
  - `002938`: `0.1970756846060582`
  - `688521`: `0.176257858040447`
  - `600549`: `0.1709904053232043`
  - `601872`: `0.1262619744157428`
- 输出检查：5 行、5 个唯一股票、权重和 `0.9999999999999999`，全部属于当前沪深300股票池。
- 新模型 OOF：
  - `risk_adjusted` final_score `0.1205026594408905`
  - `score_softmax` final_score `0.11570813378859485`
  - `equal` final_score `0.10889274882870997`
- 最近一折验证窗口 2026-05-29 到 2026-06-18，final_score `0.299811`，明显优于早期窗口；说明近期市场状态变化较大，本次采用最新 hybrid ranker。

### 注意事项
- BaoStock 数据不能保证与官方 ToShare/Tushare 完全逐字段一致。提交前已做覆盖和股票池校验，但如果后续注册官方账号，应优先对比复权价格、停牌日、成分股日期和最终候选敏感性。
- `data_submit/stock_data.csv` 有 319 只股票历史，是因为保留了旧成分股；训练和预测必须通过 `data_submit/hs300_universe.csv` 过滤当前 300 只。

### A1/A2 回放与动态权重
- A1 严格回放：
  - cutoff 数据：`data_backtest/hybrid_cutoff_20260424.csv`
  - 真实收益：`data_backtest/a1_actual_returns_20260427_0430.csv`
  - 回放模型：`model/backtest_a1_ranker_20260424/tree_ranker.joblib`
  - OOF 最佳策略为 `risk_adjusted`，但 final_score `-0.00377`，最近一折 final_score `0.05359`。
  - A1 实际窗口预测组合整体为小负分，说明该阶段模型不具备稳定优势。
- A2 严格回放：
  - cutoff 数据：`data_backtest/hybrid_cutoff_20260529.csv`
  - 真实收益：`data_backtest/a2_actual_returns_20260601_0605.csv`
  - 回放模型：`model/backtest_a2_ranker_20260529/tree_ranker.joblib`
  - OOF 最佳策略为 `equal`，final_score `0.06517`。
  - A2 实际窗口回放分数：集中策略 `0.03877`，risk_adjusted `0.03697`，equal `0.03047`，低于线上提交 `0.07327519410548909`。
  - 线上 A2 分数主要由 `002384` 重仓命中贡献，不能直接视为模型稳定泛化能力。
- 动态权重策略脚本：
  - `code/src/tune_dynamic_policy.py`
  - 当前 A3 OOF 以 28 天半衰期做近期加权，最佳为 `concentrated`，当前推荐与正式 `output/result.csv` 一致。
  - 后续应把每周五模拟提交纳入统一 walk-forward 回测，继续训练“何时重仓、何时分散”的策略层。
- 周度模拟补充：
  - 使用 `model/ranker_hybrid_20260626/oof_scores.csv` 中每周最后一个验证交易日模拟周末提交。
  - 全部 16 个周度模拟里，`risk_adjusted temp=0.02` final_score `0.213885` 最稳。
  - 最近 5 个周度模拟里，`risk_adjusted temp=0.01` final_score `0.372989` 略优于 `temp=0.02` 的 `0.369937`。
  - 因 A3 是周末提交任务，正式 `output/result.csv` 从极端重仓 `temp=0.005` 调整为半重仓 `temp=0.01`：
    - `000657`: `0.4982751424238831`
    - `002938`: `0.161792...`
    - `688521`: `0.136882...`
    - `600549`: `0.132791...`
    - `601872`: `0.070260...`
  - 极端重仓备份：`output/result_hybrid_ra_t0005_extreme_backup.csv`
  - 分散备份：`output/result_hybrid_ra_t002_mpi2_backup.csv`
- 周频 ranker 尝试：
  - 新增脚本：`code/src/train_weekly_ranker.py`
  - 训练产物：`model/weekly_ranker_20260626/tree_ranker.joblib`
  - 思路：只取每周最后一个交易日作为股票-周样本，标签仍为未来 5 个交易日开盘收益，避免日频重叠标签让验证虚高。
  - 周频 OOF 最佳为 `semi_concentrated`，final_score `0.076588`；最近两折为正，但前两折较差，整体弱于当前 hybrid 日频 ranker 的周度评估。
  - A3 周频候选：`output/result_weekly_ranker_20260626.csv`
  - 结论：周频训练形式更贴近比赛，但当前第一版效果不如日频 ranker + 周度策略调参，因此正式提交仍使用 `output/result.csv` 的半重仓 hybrid 方案。
- 主模型 + 周度质量动量融合：
  - 搜索结果：`model/heuristic_search/model_factor_blend_weekly.csv`
  - 融合分数：`0.9 * 主模型周度百分位 + 0.1 * quality factor 百分位`
  - quality factor：`ret_10_rank + 0.3 * ret_20_rank + 0.2 * amt_rel_5_rank - 0.2 * vol_10_rank`
  - 周度 OOF 最佳：`risk_adjusted temp=0.005 max_per_industry=1`，全部周度 final_score `0.311545`，最近 5 周 final_score `0.532950`。
  - 该结果显著优于模型-only 周度评估，因此正式 `output/result.csv` 已更新为融合版：
    - `000657`: `0.4273971844845565`
    - `688521`: `0.3527218933621603`
    - `601872`: `0.1027286253410539`
    - `600549`: `0.0690854439946202`
    - `600522`: `0.0480668528176090`
  - 模型-only 半重仓备份：`output/result_hybrid_model_only_semi_backup.csv`

### 严格 weekly walk-forward 框架初版
- 新增脚本：`code/src/weekly_walkforward_backtest.py`
- 回测定义：
  - 每周最后一个交易日作为模拟提交 cutoff。
  - 每个 cutoff 独立重新训练模型。
  - 训练集只允许使用 `label_known_date <= cutoff` 的样本；`label_known_date` 按每只股票第 5 个未来可交易行计算，避免停牌/缺行导致标签提前可见。
  - 预测只使用 cutoff 当天特征，评分使用该日之后 5 个开盘收益。
  - 默认可选 `daily` 或 `weekly` 训练样本频率，支持 `lightgbm,xgboost,catboost` 子集。
- 烟测输出：`model/weekly_walkforward_smoke`
- 最近 16 个周度窗口轻量版输出：`model/weekly_walkforward_lgb_fast_16w`
  - 命令使用单 LightGBM、fast profile、最多 60000 训练行、当前沪深300股票池。
  - primary 配置为 `0.9 * model_pct + 0.1 * quality_pct`。
  - 融合版最佳：`div_t005_mpi2`，final_score `0.151257`。
  - 同一预测下 model-only 最佳：`balanced_t002_mpi2`，final_score `0.202995`。
  - 结论：严格 walk-forward 下，当前 0.1 质量动量融合不如 model-only；此前 OOF 融合收益存在乐观风险。下一步应优先基于该框架重新搜索 blend weight、训练频率、模型组合和动态权重策略。

### 严格 walk-forward 循环优化结果
- 目标口径：只看严格 weekly walk-forward 的历史模拟提交分数，不再用日频 OOF 作为优化目标。
- 主要实验：
  - `model/weekly_walkforward_lgb_fast_model_only_40w`
    - fast LightGBM，daily 训练样本，60000 行下采样。
    - 默认集中策略 final_score `0.130417`。
    - 策略层搜索后，model-only `temp=0.003, max_per_industry=1` 可到约 `0.149829`。
  - `model/weekly_walkforward_lgb_fast_weekly_train_40w`
    - weekly 训练样本。
    - 最佳 final_score `0.057408`，明显不如 daily 训练样本，放弃该方向。
  - `model/weekly_walkforward_xgb_fast_model_only_16w`
    - XGBoost 最近 16 周筛选。
    - 最佳 final_score `0.227081`，低于同期全量 LightGBM 的 `0.278401`，不扩展到 40 周。
  - `model/weekly_walkforward_lgb_fast_fullrows_16w`
    - fast LightGBM，daily 训练样本，不下采样。
    - 最近 16 周默认集中策略 final_score `0.278401`，显著优于 60000 行下采样版。
  - `model/weekly_walkforward_lgb_fast_fullrows_40w`
    - fast LightGBM，daily 训练样本，不下采样。
    - 默认集中策略 final_score `0.146158`。
    - 策略搜索最佳：`0.05 * q_old + 0.95 * model_pct`，`risk_adjusted temp=0.0015 max_per_industry=1`，final_score `0.200556`。
    - 逐周统计：mean weekly final `0.161142`，median `0.147427`，hit 0.1 rate `0.575`，最近 16 周 mean `0.297713`，最近 8 周 mean `0.608366`。
- 当前正式训练脚本：
  - `code/src/train_current_lgb_submission.py`
  - 命令：
    `python code/src/train_current_lgb_submission.py --data ./data/hybrid_stock_data.csv --target-universe ./data_submit/hs300_universe.csv --model-output ./model/current_lgb_fullrows_20260626 --output ./output/result.csv --scored-output ./output/current_lgb_fullrows_scored_20260626.csv --quality-weight 0.05 --temperature 0.0015 --max-per-industry 1`
- 当前正式提交：`output/result.csv`
  - `000657`: `0.9985023976206284`
  - `301269`: `0.0012842622738665`
  - `600549`: `0.0001247381291612`
  - `002938`: `0.0000867187230131`
  - `601872`: `0.0000018829373305`
- 当前打分文件：`output/current_lgb_fullrows_scored_20260626.csv`
- 当前模型：`model/current_lgb_fullrows_20260626/tree_ranker.joblib`
- 备份：
  - 稍分散权重：`output/result_lgb_fullrows_q005_t0005_mpi1_backup.csv`
  - 等权备份：`output/result_lgb_fullrows_q005_equal_mpi1_backup.csv`
- 注意：正式 `temp=0.0015` 权重极端集中，回测最优但单票风险高；若担心单票风险，可提交 `temp=0.005` 备份，它选股相同但最大权重约 `0.769`。

### 官方收益率口径修正与近期策略
- 重新确认官方本地评分脚本：
  - `test/score_self.py`
  - `test/score_docker.py`
  - `Final Score = sum(提交股票未来 5 条记录开盘收益率 * 权重)`。
  - 这就是官方收益率本身，不是此前内部使用的 `(portfolio-random)/(oracle-random)` 归一化相对分数。
- 因此后续优化目标改为 `portfolio_return` / 官方收益率。
- 关键发现：
  - 全历史长特征最优策略 40 周官方平均收益约 `3.84%`，最近 12 周约 `9.84%`，最近 8 周约 `14.03%`。
  - 只使用最近 365 天训练样本效果很差，40 周官方平均收益约 `1.17%`，说明不能简单砍掉旧样本。
  - 半衰期 90 天样本加权最近 16 周官方收益约 `2.96%`，也不如无加权全历史训练。
  - 去掉 120 天特征、只用 `3,5,10,20,30,60` 窗口后，最近 12 周策略搜索可达到官方收益约 `10.52%`，最近 8 周约 `12.94%`；40 周全局最优约 `4.83%`。
- 当前采用“近期 regime”策略，而非声称全历史 40 周稳定 10%：
  - 模型：fast LightGBM，daily 股票-日期样本，全历史训练样本，不下采样。
  - 特征窗口：`3,5,10,20,30,60`。
  - 策略选择依据：最近 12 周官方收益率接近/达到 10%。
  - 当前正式文件：`output/result.csv`
  - 当前正式权重采用 `quality_weight=0.30, risk_adjusted temp=0.005, max_per_industry=1`，对应最近 12 周官方收益约 `9.84%`，比 `temp=0.0015` 的约 `10.52%` 更少极端集中。
  - 当前提交：
    - `601872`: `0.9433834836145166`
    - `600460`: `0.0271855956976975`
    - `000657`: `0.0216520811714509`
    - `600522`: `0.0057930641070249`
    - `600549`: `0.0019857754093101`
- 备份：
  - 极端近期收益版本：`output/result_lgb_w60_q030_t0003_backup.csv`
  - 稍更分散：`output/result_lgb_w60_q030_t001_backup.csv`
  - 等权：`output/result_lgb_w60_q030_equal_backup.csv`

### B阶段前交接总结：当前模型问题与后续方向
- A阶段第三次提交已经完成，当前不再临时调整 A3，等待该周后续 5 个交易日官方收益率结果。
- 当前确认的官方评分口径：
  - 使用提交股票在测试窗口最后 5 条记录的开盘收益率。
  - `Final Score = sum(单股票未来开盘收益率 * 权重)`。
  - 后续所有优化必须以官方收益率 `portfolio_return` 为主目标，不能再把内部归一化分数当作最终目标。
- 当前 A3 实际提交文件：
  - `output/result.csv`
  - 采用短窗口特征 `3,5,10,20,30,60`，`quality_weight=0.30`，`risk_adjusted temp=0.005`，`max_per_industry=1`。
  - 当前提交为偏进攻型：
    - `601872`: `0.9433834836145166`
    - `600460`: `0.0271855956976975`
    - `000657`: `0.0216520811714509`
    - `600522`: `0.0057930641070249`
    - `600549`: `0.0019857754093101`
- 当前模型能做什么：
  - 使用沪深300候选股票的历史量价/K线数据训练排序模型。
  - 训练样本是“股票-交易日”级别，标签是未来 5 个交易日开盘收益。
  - 严格 weekly walk-forward 框架已经建立：每周最后一个交易日模拟提交，只用该日之前可见标签训练，再预测未来 5 日。
  - 短窗口特征版本在近期 8-12 周的官方收益率表现更好，说明近期市场 regime 对策略选择很重要。
- 当前模型的主要问题：
  - 模型过度奖励短期强动量，容易把“涨得快、成交活跃”的股票排到最前。
  - 模型本身不理解新闻、地缘政治、运价、政策、行业事件等因果信息。
  - 事件驱动股票容易出现“前期强涨、事件溢价消退后快速反杀”，当前模型对这类反转风险惩罚不足。
  - 招商轮船 `601872` 是典型案例：模型因强动量和高分排序重仓，但 K线审计显示其已有明显过热和最新日走弱风险。
  - 只用最近 365 天训练样本、或半衰期 90 天加权训练，历史回测效果反而变差；不能简单砍掉旧样本。
  - 全 40 周官方平均收益仍没有稳定达到 10%；接近 10% 主要来自最近 8-12 周的近期 regime。
- 已新增纯 K线风险审计：
  - 脚本：`code/src/audit_kline_risk.py`
  - 输出：`output/kline_risk_audit.csv`
  - 不使用新闻，只用量价/K线识别：
    - 10/20 日涨幅过热
    - 成交额显著放大
    - 短期波动高或波动升高
    - 最新日大跌
    - 最新日开盘后走弱
    - 收盘接近日内低位
    - 上影线较长
    - 强势股明显回撤
    - Top10 行业拥挤
  - 对 `601872` 的审计结果已经能标出高风险：10日涨幅过热、成交额放大、短期波动高、最新日大跌、开盘后走弱、收盘接近日内低位。
- 关于新闻/外部数据：
  - 当前公开材料更偏向“基于沪深300历史股价数据”，没有明确看到允许新闻、舆情、宏观、运价、地缘政治数据作为模型输入。
  - 后续正式模型应默认只使用规则内量价数据。
  - 可以做人工层面的外部信息审查，但不应把新闻作为自动训练特征，除非后续确认规则允许。
  - 更合规的做法是：用 K线风险审计作为新闻不可见情况下的代理风险控制。
- B阶段建议重点：
  - 不要只输出一个重仓方案，应固定生成三档提交：
    - `aggressive`：Top1 约 80%-95%，用于模型分差大、风险低、近期验证强时冲榜。
    - `balanced`：Top1 约 50%-65%，其余分散到 4 只，用于模型强但存在过热/事件不确定时。
    - `defensive`：最大单票约 25%-35%，接近等权，用于 Top1 风险高、行业拥挤、最新日反转明显时。
  - 仓位策略必须绑定置信度和风险审计：
    - 模型分差越大，允许更集中。
    - K线风险越高，必须降权或剔除。
    - 最新日大跌、开盘后走弱、收盘接近日低、短期过热同时出现时，禁止 90%+ 重仓。
  - 继续做严格 weekly walk-forward，但评估表必须同时展示：
    - 全 40 周官方平均收益
    - 最近 24/16/12/8 周官方平均收益
    - 单周收益中位数
    - 单周小于 0 的比例
    - 单周大亏情况
    - 最大单票权重
  - B阶段最终选择应根据目标决定：
    - 冲高名次：可用 aggressive，但必须通过 K线风险审计。
    - 稳定入围：更偏 balanced。
    - 若候选全是事件/题材驱动且风险审计高：用 defensive。
- 下一轮建议先做的代码任务：
  - 把三档仓位策略固化成脚本，自动生成 `result_aggressive.csv`、`result_balanced.csv`、`result_defensive.csv`。
  - 把 K线风险审计接入选股后处理：高风险股票降权、限权或剔除。
  - 在严格 walk-forward 中加入风险审计后处理，直接比较官方收益率，而不是只做当前日人工判断。
  - 等 A3 官方结果出来后，把真实结果加入回测样本，分析：
    - 模型 Top1 是否命中
    - K线风险审计是否提前预警
    - aggressive/balanced/defensive 哪个更合适
    - 是否需要调整过热阈值和最大权重规则。

## 2026-08-02 深夜优化结论

### 本轮目标
- 用户已经完成提交，不再受截止时间约束。
- 目标转为继续冲击 10% 平均收益，重点解决：
  - Top20 候选中能覆盖约 4.5 个未来 Top10，但最终 Top5 选不准；
  - 模型偏向“过去强势”，无法稳定识别未来 5 日持续强势。

### 新增/验证的代码
- 新增透明选择层搜索脚本：
  - `code/src/selection_layer_search.py`
  - 只读取已有 walk-forward scores，不重训主体模型。
  - 比较：
    - 直接 Top5；
    - Top20 分层；
    - MMR 多样性；
    - 低追高/低上影线/低 decay/低 failure/高 continuation 规则组合；
    - 多种权重模板。
- 继续使用已有 Top20 meta selector：
  - `code/src/top20_meta_selector.py`
  - 对候选 Top20 训练轻量分类/排序器，目标为真实 Top10/Top5。

### 关键实验结果
- 完整 40 周 event + continuation/failure + no industry relative walk-forward：
  - 输出目录：`model/two_stage_failure_recent40_no_industry`
  - `features=1398`
  - `best_policy_return=2.06%`
  - `rank1_return=2.58%`
  - `top5_equal_return=1.68%`
  - `avg_hit_true_top5=0.675`
  - `avg_hit_true_top10=1.275`
  - `avg_cover_true_top10_in_pred20=4.5`
- 透明 selection layer 在同一 40 周验证：
  - 输出目录：`model/selection_layer_failure40`
  - 最好约 `1.98%` 加权收益；
  - 没有打赢主体 best policy。
- Top20 meta selector 在同一 40 周验证：
  - 输出目录：`model/top20_meta_selector_failure40`
  - 最好约 `1.69%` 等权收益；
  - 也没有打赢主体 best policy。
- 已有 14 周短窗：
  - 一些后排/late 位置策略可到约 `5%~5.5%`；
  - 但 40 周 no-event 验证上明显失效，属于近期 regime/样本挖掘，不能当稳定 10% 解法。

### 当前结论
- “Top20 里有 4.5 个未来 Top10”是真实的，但最终选 5 个的问题不能靠简单后处理解决。
- 当前候选排序信号在 40 周上仍然太弱：
  - Top20 内真实强势股的位置不稳定；
  - 选择层模型学不到可泛化的 Top5 判别边界；
  - failure/decay/continuation 作为附加分数没有把收益提升到 10% 量级。
- 当前系统的核心瓶颈已经不是权重策略，也不是简单 selection rule，而是主体训练目标没有真正学到“未来 5 日强势持续性”。

### 下一步必须改变的方向
- 不再继续只调 Top20 -> Top5 后处理。
- 需要重做训练目标：
  - 从“未来收益排序”转为“候选强势股中的持续性/失效二分类或 pairwise 排序”；
  - 训练样本应限定在历史候选池或历史强势股内，而不是全 300 只直接排序；
  - 标签重点区分：
    - 当前强且未来继续进入 Top10；
    - 当前强但未来转弱/回撤。
- 需要把 walk-forward 改成逐周 checkpoint，避免 40 周验证 45 分钟后才统一落盘。
- 当前 `output/result.csv` 保留此前较可信的 14 周 hgb10b 结果；本轮两个新候选只保存为独立文件：
  - `output/result_top20_meta_selector_failure40.csv`
  - `output/result_selection_layer_failure40.csv`

## 2026-05-26

### 今日工作
- 阅读并梳理了 `THU-BDC2026` 项目结构、训练流程、标签定义和提交逻辑。
- 新建并验证了 `conda` 环境 `thu-bdc2026`，确认 `torch`、`TA-Lib`、`scikit-learn` 等依赖可用。
- 重新设计了抓数脚本 [get_stock_data.py](/root/THU-BDC2026/get_stock_data.py)，目标改为更完整的 A 股日线与元数据抓取。
- 验证了 `baostock` 的可用字段范围，确认日线主接口可稳定抓取量价、交易状态、ST 标记以及部分估值字段。
- 启动并检查了全市场抓取流程，定位到当前 `stock_data.csv` 存在旧 schema 与新 schema 混合的问题。
- 将抓数脚本改造成增量模式：支持断点续抓、按股票补缺日期、必要时对旧 schema 股票做定向回填。
- 增加了自动取最新交易日的参数支持，并保留可见进度条。

### 发现的问题
- `AKShare` 在当前服务器网络环境下不稳定，东财接口容易被代理或连接中断影响，不适合作为默认主数据源。
- 当前 `data/stock_data.csv` 不是本次抓取的最终结果，里面混有旧版数据，导致新增字段大面积为空。
- 当前旧数据最晚只到 `2026-03-13`，不是最新交易日。
- 增量元数据路径命中缓存时，遗漏了 `股票代码` 字段回填，已定位并修复。

### 下一步
- 继续完成并验证增量抓数脚本。
- 给出你可直接运行的正式抓数命令。
- 等你开始下一步训练或数据处理时，继续在本文件追加日报。
