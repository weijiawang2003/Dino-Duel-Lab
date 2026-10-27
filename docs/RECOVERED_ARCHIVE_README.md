# v1.3 Counterfactual planning extension

**Start at `START_COUNTERFACTUAL.txt` and `counterfactual/report.html`.**
The latest experiment tests whether learned future predictions improve attack
choice. The original v1.2 trained policies and their results below are preserved;
they are the frozen source policies, not newly retrained actors.

New scripts, trained predictors, paired branch outcomes, raw data and live
planning-pilot results are included. `docs/COUNTERFACTUAL_METHOD.md` specifies
what was actually tested and its limitations.

---

# Dino Duel Lab 1.2 · Elimination benchmark

**普通 PPO vs 多步 JEPA 辅助 PPO。** 这版把限时比赛改成持续淘汰制，提供实际训练权重、可恢复训练状态、逐局评测数据和离线回放。实际结果见 [`elimination/report.html`](elimination/report.html)，不能仅凭某一场演示判断模型强弱。

## 这次改了什么

主环境 `max_seconds=null`：只在碰撞淘汰时结束；同时撞死才算真实平局。外部评测观察预算耗尽记为 `censored / 未决`，不算平局，不发终局奖励。训练在 rollout 边界延续当前对局，而不是重新开局。主环境没有自然障碍、没有强制加速，也没有突然缩小间距的机制；致死障碍来自对手购买。

共同基础跑跳网络经过可行动作蒸馏，解决了原模型被连续仙人掌很快击败的问题。然后两组都添加可训练的防守修正头，与攻击头共用编码器。基础网络固定，但这个修正头可以改变实际跑、跳、蹲决策。不能把它描述成“整个底层神经网络全部解冻”。

新难障碍：宽仙人掌的宽/高为 2.2/1.2 米。所有新购买都要通过当前已知队列的可达性检查：存在至少一种合法操作序列能够通过。它不替被试控制器做选择，也不承诺对局最终一定有人死亡。其他三种障碍沿用原形状。双方开局20金币，可选择积攒、组合、投放。

JEPA 条件在同一 PPO 主干上增加 EMA 目标编码器、动作序列编码器和潜变量预测器，预测未来 8、32、64 个操作步（约0.53、2.13、4.27秒）。辅助训练能影响攻防共同表征；推理仍由策略网络直接出招，没有搜索或规划。真实未来动作只用于训练标签，不能进入当时的策略输入；预测标签不跨越对局重置。

## 安装、运行

```bash
python3 setup_env.py --backend cpu --test
.venv/bin/python scripts/elimination_suite.py evaluate --out elimination --workers 3
.venv/bin/python scripts/elimination_suite.py train --out elimination --updates 1500 --workers 3
.venv/bin/python scripts/elimination_suite.py evaluate --out elimination --section all --workers 3
```

从解压后的项目根目录运行。Python 3.11–3.13，实际验证的版本记录在 `VALIDATION.json`。Windows 使用 `.venv\Scripts\python.exe`。`--updates` 是总更新目标；自动从 `latest.pt` 恢复。新输出目录会从共同初始状态重新训练。

安装后不需要联网：模型、共同基础技能、配置、脚本都在包内。可选的原版 Chrome Dino 适配器需要另外下载上游资源，并不是主实验依赖。Mac可选 `--backend mps`，NVIDIA可选 `--backend cuda126/cuda128`；训练加 `--device mps/cuda`。本次没有GPU硬件验证，不能保证跨后端逐位一致或一定加速。

主实验脚本固定使用公开的三个训练种子11、23、47；任意种子/模型/规则的单独训练可使用 `run.py train --help`，自定义配置为JSON，不需要改Python代码。

## 评测入口

```bash
# 普通对战、固定对手、未见攻击程序、初始模型
python scripts/elimination_suite.py evaluate --out elimination --section main --workers 3
# 跨训练种子交叉对战
python scripts/elimination_suite.py evaluate --out elimination --section cross --workers 3
# 历史检查点联赛
python scripts/elimination_suite.py evaluate --out elimination --section history --workers 3
# 速度/资源/间距改变
python scripts/elimination_suite.py evaluate --out elimination --section stress --workers 3
# 禁用左侧攻击，只测防守存活
python scripts/elimination_suite.py evaluate --out elimination --section defense --workers 3
# 同一冻结模型、同一种子，延长观察；不能当额外独立样本
python scripts/elimination_suite.py evaluate --out elimination --section extended --safety-seconds 900 --workers 3
# 从记录重新生成报告
python scripts/elimination_suite.py report --out elimination
```

`all` 包括 main/cross/history/stress/defense，不默认跑 extended。参数 `--num-seeds` 指环境种子数量，不是训练模型数量。每个环境种子交换两侧座位。`--safety-seconds` 是外部观察预算，绝不是游戏平局规则。

### 三种执行模式

主分析 `joint`：跑跳和攻击均按照策略分布采样，与训练一致。`attack`：跑跳取最大概率、攻击采样。`greedy`：两者取最大概率。三种不能混为一谈。每个策略拥有独立、可复现的随机数发生器，换座位保持策略身份随机种子。

### 未决如何统计

胜=1、真平=0.5、负=0。未决不赋值。对于胜W、真平D、未决C、总局数N，完整结局得分只能确定在：

```
[(W + 0.5 D)/N, (W + 0.5 D + C)/N]
```

只在已决对局里计算的得分另外标注为条件统计，不能替代主结果。报告同时给出两个边界的分组重采样包络，按训练种子和共同环境种子分别重采样，保留交换座位配对。只有三个独立训练重复，不应把几千局游戏当成几千次训练重复。

## 文件

```
assets/runner_bc.npz                 原共同跑跳模型
assets/runner_adversarial.npz        新共同跑跳模型
assets/warmstart/seed_*.pt           两组共用的旧baseline来源
configs/elimination.json            本轮规则与学习参数
elimination/PROTOCOL.json           比较范围、预算、种子、开发选择说明
elimination/runs/*/policy.pt        推理权重
elimination/runs/*/latest.pt        优化器/RNG/对手池/进行中对局，可恢复
elimination/runs/*/checkpoints/     历史检查点
elimination/runs/*/metrics.jsonl    逐更新原始日志
elimination/evaluations/*/          每组对战CSV、汇总与部分回放
elimination/defense_bootstrap/      共用防守技能来源与开发验收
elimination/diagnostics/            放弃的开发阶段、几何校准、数值错误记录
scripts/bootstrap_adversarial.py   重训共同防守技能的完整脚本
```

旧v1.1方法文档保留在 `docs/archive-v1.1/`。旧配置仍可用于复现实验，但它们的有限时间平局与本轮未决不是同一含义。

## 公平性与限制

配对模型的初始行动权重相同，行动参数量相同，环境交互步数相同，训练使用相同PPO设置；JEPA额外增加预测器训练，不能宣称总参数、训练时间或FLOPs相同。实际学习者样本数、优化minibatch数量、时间记录在日志里。

本轮热启动保留旧baseline编码器，新共同防守模型独立提供基础动作；攻击与价值头重新初始化，防守修正头从零开始。这样不会把旧版只会连续发便宜仙人掌的攻击策略硬塞进新规则。两个条件都不是从随机动作重新学基本跑跳。

共同技能的离线教师能读取完整**已生成**障碍队列，学生只看最近三个；教师不能看到未来购买。推理不用教师或未来标签。这是一项双方共有的训练先验，不应包装成完全无监督从零学会。

这是开发中多次发现问题并更改规则后的探索性比较，不是预注册研究。失败阶段没有混入最终平衡比较，日志保留。最终测试只比较规定轮数检查点，不按测试成绩挑“最佳”。这些结果不证明JEPA在一般博弈上更强，也不能区分“未来预测”与额外正则化/计算的全部贡献。

主任务使用53维公共结构化观察，不是像素输入。该项目没有接入Meta官方视觉JEPA checkpoint，没有推理时潜空间规划，也不承诺所有训练种子最终能互相击败。完整检查与实际结果以 `VALIDATION.json`、`elimination/comparison.json` 为准。

### 额外检查预测器

```bash
python scripts/evaluate_prediction.py --out elimination
python scripts/elimination_suite.py report --out elimination
```

在新的留出轨迹上比较预测器、潜状态保持基线、动作序列打乱。这个诊断提供真实未来联合动作作为预测条件，并非可直接部署的规划能力；真实策略出招看不到这些未来标签。数据和逐窗口损失会保存在 `elimination/prediction/`。
