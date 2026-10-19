# 从 best 精修：修复与实验记录（2026-10-20）

旧的 `curr K 60 20` 表示从 latest 继续，不能用于“从最佳模型精修”。此前 K=3/K=4 的 bid-only 阶段实际加载 iter=59 latest；其验证结果没有改善。本次修复启动、选模和溯源流程，尚未证明性能提升。

## 本次改动

- `_run_best_refine.py` 从源日志定位同 K 的 best，核对 checkpoint SHA256、W8/A10 配置和 direct RL 模式；清理继承的 `MUMT_*`，拒绝覆盖既有实验目录。
- `_curriculum_guard.py` 将 fresh、best、resume 分开。显式初始化与 `iter0>0` 冲突直接报错；缺失 checkpoint 不再静默回退。resume 核对迭代连续性及日志中的 latest SHA256。
- PPO 前评估并保存初始策略（`iter=-1`）；只用验证种子按成功率、回报、负违规率的顺序选 best，相同或退化的结果保留原 best。分别记录 baseline、last validation、best，加入早停。
- 评估结束或出错时恢复训练环境。自然验证记录多候选仲裁事件和相对固定评分的赢家变化次数，不注入探针。
- teacher、21 条规则及冲突对、红方、任务场景、环境和模型源码均未修改。当前数值合同保持 W=8、A=10、C04=80、sector=60；direct RL 仲裁。`MUMT_W8_FIXED=0.2` 是违规奖励缩放，与稳定窗口 W 不同。
- checkpoint 仅恢复模型权重；优化器及 RNG 重新初始化，日志已明确记录，不能宣称精确续训。

## 验证

共 187 项测试通过：启动/选择/环境恢复/仲裁诊断 72 项，launcher、权重加载、真实短 rollout、溯源和 detect 相关回归 115 项。存在一条既有 PyTorch 张量构造性能警告。

最终 K=3、K=4 各运行 1 次真实 PPO 更新（1 episode，完整任务时限；验证种子 230、231）。两组均加载指定 best、完整结束且源码哈希未变化；更新后成功率下降，因此保留 iter=-1 的全部初始模型和 anchor 张量并按 patience=1 早停。这是流程验证，不是提升效果的实验。证据在 `runs/model_optimization_v1/best_refine_verification_20261020/`。

## 原始 best 的独立审计

固定原始 best，自然配对评估 seeds 240–249，RL/HEUR 使用相同任务配置。违规率为总违规数 / 总步数，可超过 1。

| K | RL 成功率 | HEUR 成功率 | RL 平均回报 | HEUR 平均回报 | RL 违规/步 | HEUR 违规/步 |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 70% | 100% | -4952.09 | -3776.86 | 5.7980 | 6.5689 |
| 4 | 40% | 70% | -9444.43 | -4398.14 | 9.0441 | 8.1055 |

RL 的自然多候选责任仲裁事件仅为 K3=0、K4=2，相对固定评分赢家变化均为 0。这类事件与“13 对规则同时激活”含义不同，不能互相替代。当前结果不能支持“RL 已稳定优于 HEUR”或“学会仲裁带来了收益”。数据和评估源码快照在 `runs/model_optimization_v1/pre_refine_holdout_20261020/`。

## 下一步命令

在 PowerShell 中顺序执行。两组分别从各自现有 best 精修，不是 K3 精修完成后自动接 K4 的课程链。默认联合 PPO、lr=1e-5、epochs=2、每轮 8 episodes、最多 20 轮，每 2 轮验证，连续 3 次验证无提升早停。保留 saved bid std，不强制覆盖。

```powershell
Set-Location 'D:\Desktop\Aircraft_mumt_cc'
.\Aircraft_mumt\.venv\Scripts\python.exe _run_best_refine.py --k 3 --source-run runs/model_optimization_v1/curriculum_k3_w8a10 --output runs/model_optimization_v1/best_refine_k3_w8a10_v1
.\Aircraft_mumt\.venv\Scripts\python.exe _run_best_refine.py --k 4 --source-run runs/model_optimization_v1/curriculum_k4_w8a10 --output runs/model_optimization_v1/best_refine_k4_w8a10_v1
```

训练种子 K3 从 10000、K4 从 20000 开始；选模仍使用 230–239。240–249 已用于本次原始候选独立审计，不作为本轮选模依据；新候选最终评估保留 250–279。先读取两组新 `pilot_log.json`，如没有改善再针对行为退化和自然冲突覆盖诊断，避免直接追加 bid-only 长训。

## 留存范围

本次提交保留修复代码、测试、原 K3/K4 best 与日志、源码/运行清单、起点快照、独立审计及最终烟测证据。逐轮冗余权重不纳入本次提交，旧日志仍保留其哈希；未删除用户既有实验或混入工作区其他变更。最终烟测目录下的验证报告列出本次明确纳入的文件与哈希。`.gitattributes` 禁止 Git 对实验产物做换行转换，避免检出后日志、运行清单与指标文件的 SHA256 失效；提交前逐一核验暂存区的产物字节与原文件一致。
