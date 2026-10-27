> **Version note (1.2):** This document describes legacy presets unless stated otherwise. For the elimination experiment, use `../README.md` and `ELIMINATION_METHOD.md`; time-limit draws in legacy presets are not censored outcomes in v1.2.

# 命令手册

以下假设已经在项目根目录完成 `python3 setup_env.py --backend cpu`。Windows 将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。所有输出路径都可自行指定。

## 诊断与预览

```bash
.venv/bin/python run.py doctor
.venv/bin/python run.py --help
.venv/bin/python run.py demo --out runs/demo/replay.html
.venv/bin/python run.py demo --runner teacher --left teacher:random --right teacher:save --seconds 30
.venv/bin/python run.py benchmark --config cpu --steps 10000 --out runs/benchmark.json
```

`demo --runner teacher` 使用规则跑酷；默认 `--runner bundled` 使用已训练的网络。此处 `teacher:random` 等字符串表示**攻击策略**，最终名称会显示 `bc:random` 或 `teacher:random`，避免把脚本攻击误认成训练出的策略。

只生成 demo 时可以只装核心依赖：`python3 setup_env.py --backend core`。包里的 `examples/replay.html` 可以直接打开，连核心依赖也不需要。

## 一键完整实验

```bash
.venv/bin/python run.py experiment --config smoke --out runs/smoke
.venv/bin/python run.py experiment --config cpu --out runs/main
.venv/bin/python run.py experiment --config standard --out runs/standard
.venv/bin/python run.py experiment --config joint --out runs/joint
```

顺序执行共享基础模型读取 → 各条件/训练 seed 的自博弈 → 固定对手评测 → 相同训练 seed 的各条件正面对战 → 跨训练 seed 汇总 → 报告与回放。

配置可以是预设名，也可以是自己的 JSON 路径。自定义 JSON 会与 `configs/cpu.json` 合并，未写出的键继承默认。设置都可在文件中完成，不需要新增 Python 代码。

```bash
.venv/bin/python run.py experiment --config cpu --out runs/pilot --updates 20 --seeds 11 23 --variants baseline jepa
```

这是缩减预算的探索，不能把少种子结果当作算法优越性的充分证据。

## 单独训练

```bash
.venv/bin/python run.py train --config cpu --variant baseline --seed 11 --out runs/baseline11
.venv/bin/python run.py train --config cpu --variant jepa --seed 11 --out runs/jepa11
.venv/bin/python run.py train --config cpu --variant jepa_shuffled --seed 11 --out runs/shuffled11
.venv/bin/python run.py train --config standard --variant baseline_large --seed 11 --out runs/large11
```

支持 `--device auto/cpu/cuda/mps`，以及 `--runner bundled/teacher/路径.npz`。默认 runner 由预设决定。joint 训练必须有共同的 `.npz` 初始化，不能用 `--runner teacher` 直接开始随机 motor 学习。

每个 run 写入：

| 文件 | 用途 |
|---|---|
| `config.json` | 解析后的完整配置、runner 来源、有效网络结构 |
| `metrics.jsonl` | 每次更新的真实损失、步数、时间、对局和潜变量诊断 |
| `latest.pt` | 完整恢复状态 |
| `policy.pt` | 独立推理权重，内嵌基础跑酷网络 |
| `checkpoints/update_XXXXXX.pt` | 定期保留的完整历史 checkpoint |
| `training_summary.json` | 完成/停止状态、预算和权重路径 |

## 停止与恢复

```bash
.venv/bin/python run.py stop --out runs/main
# 删除 runs/main/STOP 后：
.venv/bin/python run.py experiment --config cpu --out runs/main --resume

# 单个 run：
.venv/bin/python run.py train --config cpu --variant jepa --seed 11 --out runs/jepa11 --resume --updates 500
```

Ctrl+C 与 STOP 都在 PPO update 边界生效，可能需要等当前采样和更新完成。机器强制断电只能恢复最近一次已经保存的 checkpoint。

同一实验不允许并发写入。程序使用操作系统的非阻塞文件锁；`RUNNING.json` 只保存状态元数据，完成后标记 `released` 并保留。文件仍存在不代表任务还在运行，也不需要删除它。不要删除仍在运行任务的锁文件；这会破坏互斥保护。不同机器同时写共享网络目录仍应由任务调度器避免。

完整 experiment 输出包含 `shared_runner.npz` 和 `experiment_config.json`。移动整个目录后可以直接使用：

```bash
.venv/bin/python run.py experiment --config moved-output/experiment_config.json --out moved-output --resume
```

原始自定义 runner 文件删除后仍可恢复；单个 run 的 `latest.pt` 也已内嵌其 runner。

`--updates` 是累计更新上限。完整 checkpoint 会保存随机状态、环境、历史池和当前对手。恢复时允许修改运行设备、线程、保存间隔、总预算和墙钟预算等运行参数；改变物理、网络或学习目标应另开输出目录。

## 评测指定模型

```bash
.venv/bin/python run.py evaluate --left runs/baseline11/policy.pt --right runs/jepa11/policy.pt \
  --config cpu --num-seeds 100 --seed-start 4294967296 --out runs/head_to_head --record

.venv/bin/python run.py evaluate --left runs/jepa11/policy.pt --right teacher:random \
  --config cpu --num-seeds 30 --out runs/vs_random --use-checkpoint-env
```

`--use-checkpoint-env` 使用左侧模型保存的完整环境配置。没有该参数时，用给定 JSON 的配置，方便进行明确的分布变化测试。模型的基础 runner 随 checkpoint 保存；`--runner` 只控制脚本对手的基础 runner。

最终评测默认使用 >=`2**31` 的保留 seed 域。手工 evaluate 可以使用任意非负 seed 进行诊断，但如果使用训练域或调参用过的 seeds，不能再称为保留测试。

结果为 `episodes.csv`、`summary.json`，可选 `replay.html`。置信区间以**同一 seed 的两次换座对局**为抽样单位。单个 checkpoint 的区间不包含训练随机性的影响；完整 experiment 的 `comparison.json` 另外对训练 seed 进行分层重采样。

## 看对局和重新生成报告

```bash
.venv/bin/python run.py replay --left runs/baseline11/policy.pt --right runs/jepa11/policy.pt \
  --config cpu --seed 4294967296 --out runs/match.html --use-checkpoint-env

.venv/bin/python run.py report --runs runs/main
.venv/bin/python run.py report --runs runs/main --out runs/main/custom_report.html
```

回放保存了真实模拟状态，浏览器只负责播放，没有在网页中训练神经网络，也不会调用在线服务。

## 重新训练基础跑酷模型

```bash
.venv/bin/python run.py bootstrap --config bootstrap --out runs/bootstrap/runner_bc.npz
.venv/bin/python run.py train --config joint --variant jepa --seed 11 \
  --runner runs/bootstrap/runner_bc.npz --out runs/new_runner_trial
```

默认包内权重已经可用。仅当你希望重建或修改环境时再执行 bootstrap。过程包含真实教师轨迹、合成近障碍监督状态与 DAgger。输出 `.json` 会记录数据规模、种子和训练曲线；动作分类准确率不能替代闭环跑酷测试。

## 原版第三方 checkpoint

```bash
python3 setup_env.py --backend cpu --upstream
.venv/bin/python run.py upstream info
.venv/bin/python run.py upstream fetch --out external/aome510
.venv/bin/python run.py upstream run --out external/aome510 --episodes 5 --max-steps 10000
.venv/bin/python run.py upstream run --out external/aome510 --episodes 2 --show
```

`fetch` 验证固定源代码归档和 checkpoint 的 SHA-256。`run` 验证下载文件未被修改，使用原版单人游戏与原版像素输入。主实验不依赖它，不会把这份权重误当成自定义模拟器兼容权重。

## Docker（CPU 配方）

```bash
docker build -t dino-duel-lab .
docker run --rm dino-duel-lab doctor
docker run --rm -v "$PWD/runs:/app/runs" dino-duel-lab experiment --config cpu --out runs/main
```

Docker 配方已提供，但本次交付环境没有执行 Docker 构建；实际验证使用 Python 3.12 的独立环境。Windows 的 volume 路径请按所用终端格式填写。输出挂载在宿主机目录中，删除容器不会删除该目录。

## 1.1 实际训练模型与对照复现

六个主模型已经训练好，不需要重训才可以观看。`trained/final/report.html` 是采样攻击的主比较，`trained/secondary_greedy/report.html` 是 argmax 的次要比较。

```bash
# 生成与正式评估一致的采样攻击回放。
.venv/bin/python run.py replay --left trained/runs/baseline_seed11/policy.pt --right trained/runs/jepa_seed11/policy.pt --stochastic-attacks --use-checkpoint-env --seed 17179869184 --out runs/duel.html

# 从零复现两方法、三个种子的同预算比较。
.venv/bin/python scripts/reproduce_comparison.py --out runs/reproduction --workers 2

# 同一总预算下恢复。
.venv/bin/python scripts/reproduce_comparison.py --out runs/reproduction --workers 2 --resume

# 重新评估现成六个模型。
.venv/bin/python scripts/compare_trained.py --runs trained/runs --out runs/retest --training-seeds 11 23 47 --expected-update 300 --num-seeds 50 --workers 2 --record

# 确定性攻击只是次要评估；使用独立输出目录。
.venv/bin/python scripts/compare_trained.py --runs trained/runs --out runs/retest-greedy --training-seeds 11 23 47 --expected-update 300 --num-seeds 50 --workers 2 --attack-mode greedy
```

`--stochastic-attacks` 也适用于 `evaluate` 和 `demo`。它只从 learned attack head 的 masked softmax 采样；motor 使用 argmax，原脚本对手保持原行为。每个策略有独立的可重置随机数发生器，强制等待不消耗随机数。

`trained_strength` 开启 `training.decision_only_actor=true`：冻结跑跳时，只在实际有攻击选择的步上计算 actor 目标和标准化 advantage；critic 与 JEPA 仍学习全部转移。`gamma=.999, gae_lambda=.995` 延长攻击延迟的回报归因。此设置对普通 PPO 与 JEPA 完全相同。

## 常见问题

- **报缺 torch**：运行安装脚本后，使用它输出的 `.venv` Python，避免不小心继续用系统 Python。
- **OOM 或不够快**：降低 `training.num_envs`、`rollout_steps`、`minibatch_size`；更改采样设置需要新输出目录。先使用 `cpu` 或更小的自定义配置。
- **CUDA 不可用**：使用 CPU，或安装与硬件/驱动匹配的已支持 CUDA wheel。安装了 CUDA wheel 不等于机器有可用 GPU。
- **一直平局**：查看是否误用了保守 bootstrap 配置；确认主实验 `natural_obstacles=false`、`min_gap_seconds=.30`、`ensure_reachable=true`、`attack_every_n_steps=6`、`attack_cooldown=.30`。即使参数正确，学出的策略仍可能造成高平局率，这本身是实验结果，不能强行解释成谁更强。
- **mean_episode_return 为 0**：零和双方的平均回报本来就抵消。看实际评测、平局率、局长以及绝对回报，不要用这个均值判断收敛。
- **恢复时配置不匹配**：使用原配置和原 seed/variant；需要改变研究条件时另开实验目录。不要修改 checkpoint 的签名来绕过检查。
- **外部下载失效**：主实验仍可离线运行。可选 upstream 工作流依赖其原站点和固定字节，不能保证第三方永久可用。

