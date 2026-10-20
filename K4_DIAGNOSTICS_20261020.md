# K4 退化诊断：固定任务合同的五组对照

## 问题与范围

前一轮从各自 best 精修时，K3 在验证种子 230–239 上保持 90% 成功率，回报由 -3163.4 改善到 -3011.9，违规/步由 4.7069 降至 4.5016；第 8 轮早停。K4 的新更新未超过初始 best（70% / -4891.3 / 6.8201），第 6 轮早停，latest 仅 40%。best 保护及起点 SHA 校验正常。以上只是已用于选模的验证集结果。

本轮诊断固定 K4、W=8、A=10、C04=80、sector=60、direct RL 仲裁。teacher、21 条规则及冲突对、红方、环境和任务配置文件保持不变；只修改训练器及实验启动/记录代码。后续各 K 可使用相同训练机制，没有按 K 切换策略实现。

## 实验设计

五组均从原 `curriculum_k4_w8a10/curr_k4_best.pt`（原 iter=24，SHA256 `386a7a8d6eabedd68d1275ae77f4a8d4ce88bcb893eb82d9d3866aaff3f1e797`）开始。所有组使用新的 optimizer、相同 policy 初始化种子和相同训练种子 30000–30063，每组 8 轮，每轮 8 episodes，lr=1e-5、epochs=2，每 2 轮在 230–239 自然配对验证。固定预算关闭早停，best 保护继续启用。成功会提前结束 episode，因此相同 episode/update 预算不代表相同环境步数。

| 输出子目录 | 训练冲突暴露概率 | 训练违规惩罚缩放 | 起点策略 KL 正则系数 |
|---|---:|---:|---:|
| control | 1 | 0.2 | 0 |
| natural_requests | 0 | 0.2 | 0 |
| full_penalty | 1 | 1 | 0 |
| natural_full_penalty | 0 | 1 | 0 |
| natural_full_penalty_anchor | 0 | 1 | 1 |

前四组为 2×2 对照，可分别比较暴露与惩罚缩放的影响及其交互；最后一组与 natural_full_penalty 比较 KL 正则的作用。本轮是单次训练种子的机制筛查；若有改善，还需多训练种子复验。预留独立测试种子 250–279，本轮不使用。

暴露会临时覆盖部分训练动作（轴修正、角色、档位、请求），并屏蔽相应任务动作梯度；因此它的作用包含行为干预和训练样本掩码，不能归因于单纯增加冲突样本。惩罚缩放 1 只对齐违规惩罚项，训练仍有 potential shaping、反事实优势等，不能称训练目标与评估完全相同。惩罚缩放不是稳定窗口 W，不修改 W8/A10。

## 可选 RL 起点约束

`policy_anchor_coef` 默认 0，关闭新正则。启用时固定本轮初始 RL checkpoint，计算 `KL(reference || current)`：连续三维动作、角色，以及按 reference 角色概率加权的合法档位/目标请求条件分布；不约束 bid。参考模型只读取局部编码观测，参数冻结，不调用 HEUR。它独立于既有轴修正 anchor。

新 best 精修在 load 后刷新参考；resume 保存并恢复该次参考。checkpoint 仍只恢复权重，optimizer/RNG 不恢复。本实验五组都监测更新后的起点累计 KL，系数为 0 也报告真实漂移，但不加入正则梯度。KL 是软约束，不能保证成功率不下降；小 KL 也可能改变离散 argmax。

## 运行

```powershell
Set-Location 'D:\Desktop\Aircraft_mumt_cc'
.\Aircraft_mumt\.venv\Scripts\python.exe _run_k4_diagnostics.py
```

命令顺序运行全部五组，共 40 轮。默认输出：`runs/model_optimization_v1/k4_diagnostics_v2/`。每组完整 stdout 写入对应 `.stdout.txt`，终端打印每组开始和完成摘要。

总表为 `diagnostic_suite.json`，每组有独立 `pilot_log.json`、best/latest、输入权重快照及源码归档。启动前拒绝覆盖非空目录；每组开始和结束检查源码、输入权重及源日志哈希；各组初始逐种子评估必须完全一致；子进程失败或核验失败会保存原因并停止后续组，不静默跳过。

预检只读取，不训练或创建输出目录：

```powershell
.\Aircraft_mumt\.venv\Scripts\python.exe _run_k4_diagnostics.py --preflight-only
```

## 核验与解释

280 项相关测试通过（其中 29 项覆盖 KL、19 项覆盖初始化接线、28 项覆盖实验套件）。五组各 1 次真实更新的完整时限烟测均通过，记录留存于 `runs/model_optimization_v1/k4_diagnostic_verification_20261020/`，用于验证执行流程，不用于评价性能。

读取正式结果时同时比较 initial、best、latest 的成功率/回报/违规率，以及更新后累计 KL、实际环境步数、暴露样本比例和自然责任仲裁事件。训练 `mean_return` 含 shaping，`mean_environment_return` 仍按该组训练惩罚缩放计分，不能用它们跨组判胜；统一使用自然验证结果。责任竞争事件与规则对冲突特征是不同指标，赢家变化次数本身不能证明改选有益。

本轮新增代码仅是训练诊断及稳定化机制；尚无本轮正式训练结果，不能宣称已解决 K4 或达到论文目标。

本地留存本轮代码/测试、前轮有效 best 和日志，以及五组烟测的日志、源码和核验摘要。烟测模型与重复逐轮权重不作为候选纳入提交；未改动用户原实验目录。网络推送留待后续处理。
