# K4 训练诊断结果（2026-10-20）

正式运行清单：`runs/model_optimization_v1/k4_diagnostics_v2/diagnostic_suite.json`。五组均 completed、返回码 0，各完成 8 轮 x 8 episodes，共训练 episode seeds 30000–30063；每 2 轮在 seeds 230–239 自然配对验证，patience=0（固定预算）。

## 核验

- 五组从同一 K4 best 开始：原 iter=24，checkpoint SHA256 `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`；源 pilot log SHA256 `a320efd38c2ea83b217c838fa3bfc8984cc681e1c8236fd271b38bef00aecf94`。
- 五组 initial evaluation 的逐种子摘要相同，task config hash 相同；实际暴露/惩罚/anchor 参数与计划吻合。
- 每组源码快照完整性、best/latest 文件与日志 SHA256、运行状态全部匹配；suite 中代码文件与运行记录哈希一致。
- 脚本规则数据、红方、场景和仿真源码无改动。自然验证没有启用强制探针。

## 主要结果

所有数值均来自相同任务配置的自然验证 seeds 230–239。违规数为总违规数 / 总仿真步数。

| 组别 | exposure | penalty scale | KL 系数 | best iter | RL best 成功率 / 回报 / 违规每步 | RL latest 成功率 / 回报 / 违规每步 |
|---|---:|---:|---:|---:|---|---|
| control | 1 | 0.2 | 0 | -1 | 70% / -4891.3 / 6.8201 | 40% / -8479.8 / 8.2382 |
| natural_requests | 0 | 0.2 | 0 | -1 | 70% / -4891.3 / 6.8201 | 40% / -8541.9 / 8.5415 |
| full_penalty | 1 | 1.0 | 0 | -1 | 70% / -4891.3 / 6.8201 | 40% / -7515.2 / 7.4985 |
| natural_full_penalty | 0 | 1.0 | 0 | -1 | 70% / -4891.3 / 6.8201 | 70% / -7607.6 / 8.0607 |
| natural_full_penalty_anchor | 0 | 1.0 | 1 | -1 | 70% / -4891.3 / 6.8201 | 50% / -6527.3 / 7.4315 |

各组 HEUR 都是 **80% / -4032.2 / 7.6631**。所有 best 都是 `iter=-1`，即原始 K4 best；八轮 PPO 更新没有一组达到选模排名优于起点。K4 没有获得胜过 HEUR 的模型。

## 诊断解读

- 关闭强制冲突暴露并未单独改善结果：penalty=0.2 时，natural_requests 的 latest 仍为 40%，回报与违规数略差于 control。
- penalty scale 提到 1 后，在 exposure=1 时 latest 仍为 40%，但违规率较 control 低（7.4985 vs 8.2382）；回报仍远差于起点。natural+full penalty latest 成功率回到 70%，但回报 -7607.6、违规率 8.0607 仍明显差于起点。
- anchor 组的最后更新 KL 为 0.000519；无正则组末次 KL 为 0.036124–0.053742，显示 KL 约束有效限制策略分布漂移。但该组 latest 只有 50%，故漂移受限尚未变成验证优势。
- 最后一次更新的自然 RL 多候选责任仲裁从 control 的 1 次，到 natural_requests 11 次、full_penalty 5 次、natural_full_penalty 0 次、anchor 0 次。赢家变化对应为 1、3、1、0、0。数量少且与性能没有一致正相关；不能据此称冲突解决成功。
- 训练中的 rule-conflict 特征非零率约为 96%，它统计局部规则冲突特征，不等价于多个飞机竞争同一目标的责任仲裁事件。
- 这些是单训练随机种子的短程机制对照。验证种子 230–239 已参与选模，不能作为最终独立测试；最终保留种子 250–279。

## 结论与建议

目前确认的问题是 PPO 更新会在不同配置下使 K4 策略退化；提高违规惩罚、关闭强制暴露或加 KL 软约束，各自都没有单独产出优于 HEUR 的模型。KL anchor 适合限制漂移，但不能保证任务表现不退化。暂不把任何新组别作为最终候选，也不据此扩到更大 K。

下一步不应再按 230–239 选模型。若继续，只从这里表现较好的自然请求 + penalty=1 + anchor 机制做多个训练种子的复验，选定后用 250–279 做一次冻结候选的 RL/HEUR 配对评估，并核对自然冲突事件与赢家改变是否确实伴随任务收益。

正式完整日志、每组 latest/best、每组 provenance 与控制台输出都保存在 `runs/model_optimization_v1/k4_diagnostics_v2/`。Git 留存保留各组输入快照、当前 best/latest 和 checkpoint 哈希记录；逐轮历史权重留在工作区，但不重复提交，因为八轮全部劣于起点且日志含各轮完整训练/验证指标及权重 SHA256。
