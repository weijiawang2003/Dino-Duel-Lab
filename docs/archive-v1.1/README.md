# Dino Duel Lab 1.1 — 已训练的 JEPA vs 普通 PPO

完整的双人小恐龙跑酷研究项目：跑步获得金币，购买障碍投放给对手，比较普通 PPO 与加入未来潜变量预测的 PPO。基础跑酷权重已经包含在包内；第一次运行不用从随机动作开始学习跳跃。

**先看实际训练结果，不需要安装任何东西：打开 `trained/index.html`；观看对局打开 `trained/replay.html`。** 包内保存两种方法各三个训练种子的权重、完整恢复 checkpoint、日志及独立对局记录。`trained/final/report.html` 是完整主分析表，`examples/smoke/` 仍只是旧版流程验证。

## 本包新增的正式实训

主实验是普通 PPO 与 PPO 加 JEPA 辅助预测，各使用训练种子 `11, 23, 47`，相同初始行动权重、游戏规则、基础跑跳网络和 PPO 设置；每个模型训练 `300 × 8 × 256 = 614,400` 个双人环境步。六个模型共 `3,686,400` 步。主评估双方都从学习到的合法攻击分布采样，跑跳动作使用 argmax。

JEPA 的额外目标预测下一操作步的 latent 表示，时间跨度约 1/15 秒；没有多步规划。基础跑跳网络在这次对比中冻结，因此本次测试比较的是**相同防守能力下学出的攻击策略**。额外训练的固定随机对手诊断单独保存在 `trained/diagnostics/`，没有混入两组主结果。

最终模型固定为 update 300，未根据最终测试选择最佳 checkpoint。每个训练种子使用 50 个独立环境 seed、交换两边位置，与另一个方法直接对战，并分别挑战 `random / aggressive / save / burst` 四种共同对手。`burst` 是手写的双仙人掌强基线。完整协议和局限见 [`docs/TRAINING_COMPARISON_PROTOCOL.md`](docs/TRAINING_COMPARISON_PROTOCOL.md)，实测结论见 [`trained/RESULTS.md`](trained/RESULTS.md)。

已经有权重，观看模型不必重新训练。安装依赖后运行：

```bash
.venv/bin/python run.py replay \
  --left trained/runs/baseline_seed11/policy.pt \
  --right trained/runs/jepa_seed11/policy.pt \
  --stochastic-attacks --use-checkpoint-env \
  --seed 17179869184 --out runs/trained-replay.html
```

重新完成同样的多种子训练与评估，只需：

```bash
.venv/bin/python scripts/reproduce_comparison.py --workers 2 --out runs/reproduction
# 如中断，继续原预算：
.venv/bin/python scripts/reproduce_comparison.py --workers 2 --out runs/reproduction --resume
```

`trained/runs/*/latest.pt` 可以继续训练；每个目录的 `resume_config.json` 与原训练设置一致。下面把 seed 11 的普通 PPO 延长到累计 600 次更新：

```bash
.venv/bin/python run.py train \
  --config trained/runs/baseline_seed11/resume_config.json \
  --variant baseline --seed 11 --out trained/runs/baseline_seed11 \
  --resume --updates 600
```

上述命令会更新对应 run；若要保留包内原始实验，先复制整个 `trained` 目录。

## 以后有资源时，直接运行

解压后进入本目录。推荐 **Python 3.12**；支持的固定依赖组合为 Python 3.11–3.12。

```bash
# 1. 建立独立环境并安装 CPU 版本；不需要浏览器、API key 或任何在线账号。
python3 setup_env.py --backend cpu

# 2. 小规模检查整个流程。
.venv/bin/python run.py experiment --config smoke --out runs/smoke

# 3. 正式的小预算、多种子实验。
.venv/bin/python run.py experiment --config cpu --out runs/main
```

Windows 将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`。Windows 安装时可用 `py -3.12 setup_env.py --backend cpu`。

Apple Silicon 可将安装选项改为 `--backend mps`。NVIDIA 可用 `--backend cuda121` 或 `--backend cuda124`；需要机器已有兼容驱动。训练自动选择可用设备，也可加 `--device cpu/cuda/mps`。模拟器在 CPU 上运行，小型网络不一定在 GPU 上更快。

训练完成后打开：

```text
runs/main/report.html          汇总曲线、评测、统计说明
runs/main/replay.html          第一个训练种子的 baseline vs JEPA 对局
runs/main/comparison.json      不同训练种子的正面对战汇总
runs/main/runs/                每个模型、每个种子的权重和日志
runs/main/evaluations/         原始每局 CSV、逐种子成绩和置信区间
```

完整命令说明在 [`docs/COMMANDS.md`](docs/COMMANDS.md)。可编辑参数都在 `configs/*.json`，不必修改 Python 代码。

## 这一版包含什么

- 原创的无窗口双人模拟器：跳跃、下蹲、加速、碰撞、金币、四种付费障碍、购买间隔、动作掩码、同时死亡和限时平局。
- 相同的基础跑酷初始化：`assets/runner_bc.npz`，约 24 KB，可用纯 NumPy 推理；另有完全不需要训练的规则控制器。
- 真实 PPO 自博弈：当前策略双方对战，混入历史对手，避免只适应最新版本。
- 四种实验条件：`baseline`、`jepa`、`jepa_shuffled`、`baseline_large`。
- 两种训练范围：固定跑酷策略只训练攻击；或从共同跑酷权重开始联合训练跑酷与攻击。
- 完整 checkpoint、原子保存、SIGINT/SIGTERM 停止、STOP 文件、同设备恢复；训练预算表示**累计更新次数**。
- 保留评测 seed、交换左右位置、多训练种子、原始 CSV、分层 bootstrap 汇总。
- 无需服务器的 HTML 回放和报告。
- 可选的原版预训练 Dino 模型下载和运行适配器，有固定 commit 与 SHA-256 校验。
- 自动化测试、依赖安装脚本、CPU Docker 配方和诊断命令。

## 游戏的具体规则

默认研究配置中，**所有障碍都由对手购买**。每跑 1 米获得 1 金币，普通仙人掌花 10 金币；宽仙人掌、低飞鸟、高飞鸟分别花 16、18、12。双方默认从 0 金币开始，钱包上限 60。所有这些数值都可在 JSON 中修改；要更快进入丰富的开局，可将 `initial_coins` 设为 20 或 30。

跑酷操作默认每秒 15 次；购买机会每 6 次操作出现一次，约 0.4 秒一次，购买冷却为 0.3 秒。分开两个频率是为了让“等一会儿攒钱”成为可以探索的动作。否则每帧都尝试购买时，随机策略很容易刚攒到 10 金币就花掉，几乎见不到较贵障碍。

研究配置使用较紧的障碍间距，并开启可躲避检查：投放时，以目标玩家的当前状态为起点，在模拟器中检查是否至少存在一条合法动作序列，能够通过**当前已放置的障碍队列**。它不保证玩家实际选中这条路径，也不预测尚未购买的障碍。双方动作同时提交；对手可以利用你选择动作的时机制造压力。

每局双方共用一个由 seed 决定的起始速度，默认 7–8 米/秒，速度在观测中公开。这让无随机自然障碍的游戏仍有不同的评测场景。训练 episode seeds 位于 `[0,2**31)`，默认最终评测从 `2**32` 开始，二者分离。

一方先碰撞则失败；同一物理 tick 双方碰撞判平；60 秒到时也判平。不要只看排除平局后的胜率，报告同时给出平局率和 `胜=1、平=.5、负=0` 的平均得分。

精确规则、观测向量和可躲避检查边界见 [`docs/GAME_RULES.md`](docs/GAME_RULES.md)。

## 基础跑酷权重从哪里来

包内 `runner_bc.npz` 是本项目在对应物理环境中，使用规则教师、合成的近障碍状态和 DAgger 实际训练得到的**原创行为克隆网络**。没有把第三方截图模型直接装进一个不兼容的模拟器。

它通过了开发验收：在保守间距、每局 60 秒的自然障碍测试中，与规则教师进行 20 个 seed、交换位置的 40 局比赛，全部存活至时间上限；另有 20 局保守间距的被攻击测试也存活到上限。详细原始记录在 `assets/runner_evaluation/`。这些是已经用于开发的验收场景，**不是正式研究的保留测试集**，也不代表在紧密对抗障碍中不会失败。

训练来源、随机种子、验证曲线和 SHA-256 在 `assets/runner_bc.json`；重建命令：

```bash
.venv/bin/python run.py bootstrap --config bootstrap --out runs/rebuilt/runner_bc.npz
```

有默认权重时无需执行这一步。若修改重力、跳跃速度等底层物理，已有控制器可能不再适用；可以通过配置重新 bootstrap，或选择联合微调。规则教师本身针对默认物理设计，不保证适应任意物理修改。

## “JEPA”在这个项目里具体是什么意思

`baseline` 和 `jepa` 使用相同的可见信息与相同的行动网络结构。JEPA 条件增加训练用的未来潜变量预测：在线 encoder 编码当前状态，predictor 根据双方已经执行的动作预测下一状态的表征，EMA target encoder 提供停止梯度的目标，并用方差正则抑制表示塌缩。

这个辅助目标更新 PPO 的共享 encoder。**推理时不执行潜空间 MPC，也不读取对手尚未提交的动作。** 它是针对本游戏实现的动作条件 JEPA 风格实验，不是官方 I-JEPA/V-JEPA 的复现或预训练权重。

`jepa_shuffled` 打乱未来目标，帮助检查收益是否依赖正确的时间对应；`baseline_large` 将主网络宽度翻倍，用于容量敏感性分析，不能当成严格参数量或计算量匹配。

默认只训练攻击，因此比较的是共同跑酷能力下的对抗策略。使用 `--config joint`，双方从同一跑酷网络开始，同时训练移动与攻击。即便如此，本版 JEPA 辅助目标直接作用于攻击/价值 encoder；跑酷 head 是单独的小网络，它通过 PPO 回报更新。不要据此声称已验证通用视觉预测或 JEPA 直接改善视觉控制。

本版模型读取 53 维**结构化游戏状态**，不从截图识别物体。这样能把实验集中在预测与决策上，并显著减少基础视觉处理开销。完整方法、损失函数和比较边界在 [`docs/METHOD.md`](docs/METHOD.md)。

## 中断、继续和备份

正常按 Ctrl+C 后，训练会完成当前 PPO update，再保存。也可以在另一个终端执行：

```bash
.venv/bin/python run.py stop --out runs/main
```

继续前删除 `runs/main/STOP`（如果存在），然后：

```bash
.venv/bin/python run.py experiment --config cpu --out runs/main --resume
```

同样的命令不会重训已经完成的 run，也不会重做权重与配置未变化的评测。要延长每个 run 的总预算：

```bash
.venv/bin/python run.py experiment --config cpu --out runs/main --resume --updates 500
```

`--updates 500` 表示累计 500 次，不是再加 500 次。不能在同一输出目录里更换环境、网络结构、学习参数或共同跑酷初始化；另开目录以保留可解释的实验记录。不要同时让两个进程写同一 run。

备份**整个输出目录**就能迁移训练。experiment 会将共有基础模型保存在 `shared_runner.npz`，并写出可携带的 `experiment_config.json`。移动后用 `--config 新目录/experiment_config.json --out 新目录 --resume`。`latest.pt` 保存完整恢复状态；`policy.pt` 保存独立推理所需的数据，包含基础跑酷权重。CPU 同版本连续训练与断点恢复的逐项一致性已经测试；跨 CPU/GPU、操作系统或依赖版本迁移不能保证浮点位级一致。

## 预算预设

| 配置 | 用途 | 更新数 × 环境数 × 每次采样步数 | 训练种子 / 条件 |
|---|---|---:|---|
| `smoke` | 完整流程检查 | 2 × 2 × 64 | 1 / 3 |
| `cpu` | 小预算研究起点，固定跑酷 | 250 × 8 × 128 | 3 / 3 |
| `standard` | 更充分的多种子比较 | 1000 × 16 × 256 | 5 / 4 |
| `joint` | 联合训练跑酷和攻击 | 500 × 8 × 128 | 3 / 3 |
| `tight` | 联合训练的紧密障碍配置 | 500 × 8 × 128 | 继承 cpu |
| `trained_strength` | 本包实际跑过的两方法主比较 | 300 × 8 × 256 | 3 / 2 |
| `trained_random` | 训练机制诊断，固定随机对手 | 300 × 8 × 256 | 1 / 1 |

表中步数是**双人环境 step**；一次推进双方。历史对手混入后，可用于 learner 更新的 transition 数会变化，日志单独记录。上述是预算设置，不是收敛保证，也不是运行时间预测。用 `benchmark` 查看当前机器的模拟器吞吐，最终训练时间还包括神经网络更新与评测。

## 可选：运行前面找到的原版模型

这个路径与主对战实验独立，保留了原模型的像素输入与原环境。默认不会联网下载。

```bash
python3 setup_env.py --backend cpu --upstream
.venv/bin/python run.py upstream info
.venv/bin/python run.py upstream fetch --out external/aome510
.venv/bin/python run.py upstream run --out external/aome510 --episodes 5
# 有图形桌面时加 --show 观看原版。
```

适配器固定在 `aome510/chrome-dino-game-rl` 的已检查 commit，下载验证 SHA-256；加载原始旧格式 checkpoint 使用明确 allowlist 和 `weights_only=True`，不会自动启用 unrestricted pickle。它不承诺这些像素权重能够直接迁移到我们的结构化状态对战中。

## 验证范围与交付边界

代码、基础权重、六个正式训练模型、安装配置、训练/评测/回放入口和文档均已包含。`VALIDATION.json` 记录实际测试与运行环境。此次已经完成上面的 CPU 多种子训练；原 `standard` 大预算和 GPU 平台尚未运行。当前小环境、冻结跑跳和三对训练种子的结果，不能代替更广泛任务上的 JEPA 效果结论。

固定依赖旨在降低未来兼容性变化，但无法保证未来任意操作系统、GPU 驱动、失效下载链接或硬件组合都无需维护。主项目不依赖外部 checkpoint 下载，已有权重可离线使用。

## 来源与许可

主项目代码与原创基础权重采用 MIT，见 `LICENSE`。没有打包 Chrome 图像素材或外部仓库源码/权重。可选外部模型的来源和许可信息见 `docs/THIRD_PARTY.md`。

方法参考：[PPO](https://arxiv.org/abs/1707.06347)、[I-JEPA](https://arxiv.org/abs/2301.08243)、[VICReg](https://arxiv.org/abs/2105.04906)、[DAgger](https://arxiv.org/abs/1011.0686)。这些参考不意味着本项目复现了各论文的完整架构或实验。

