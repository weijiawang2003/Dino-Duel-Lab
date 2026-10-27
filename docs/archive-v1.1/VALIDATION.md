# 实际验证范围

机器可读的最终结果在项目根目录 `VALIDATION.json`；自动化测试明细在
`validation/pytest-trained.xml`。旧版记录保留在 `validation/release_1.0.0.json` 与 `validation/pytest.xml`。

## 1.1 新增的实际训练与测试

- 普通 PPO 和 JEPA 辅助 PPO 各三个训练种子，六个模型各完成 300 次更新、614,400 个双人环境步，主实验共 3,686,400 步。
- 正式主分析 2,700 局，确定性出招的次要分析 2,700 局，初始模型控制 600 局；开发验证另外 288 局。逐局记录都在 `trained/`。
- 统计复核把原嵌套重采样改为适合共享环境种子的交叉重采样；重新生成结果后，54 个主/次要 matchup 的逐局 CSV 与原始记录字节完全一致。
- 最终 **82 项自动化测试全部通过**，包括真实决策样本归一化、采样策略随机数隔离、交叉重采样回归测试。
- 一键复现脚本已实际完成小规模训练和评估；再次 resume 后，8 个模型及日志文件的哈希和修改时间均保持不变。
- 新版真实回放和综合报告经过 Chromium 桌面/窄屏检查；播放、暂停和拖动正常，无 JavaScript 错误或页面横向溢出。
- 包含六个正式模型及单列的随机对手诊断模型；并非只有基础 runner 和 smoke checkpoint。结论与局限见 `trained/RESULTS.md`。

## 1.0 延续的基础验证

1. 在全新、独立的 Python 3.12 环境运行交付的 `setup_env.py --backend cpu --test`。
   安装 NumPy 2.3.5、PyTorch 2.5.1+cpu 和 pytest，随后运行全部测试并通过 doctor。
2. 完整 `smoke` 实验：baseline、JEPA、打乱目标的 JEPA；各自两次真实 PPO 更新，
   包含完整 episode 边界、历史池和 checkpoint；接着进行固定对手及正面对战，
   自动生成 CSV、JSON、HTML 报告和回放。
3. CPU 连续训练与中断恢复对照：模型、优化器、随机状态、环境、历史池、当前对手
   逐项一致。测试也覆盖已完成 resume 不改写权重、不使评测缓存失效。
4. 联合 motor+attack 训练从共同 NPZ 初始化，更新后能加载；checkpoint 嵌入了基础
   runner，删除原始来源文件后仍可独立推理。
5. 可躲避检查与实际引擎穷举短动作序列对照；检查同时碰撞、金币、购买窗口、
   冷却、动作掩码、保存恢复和限时平局的语义。
6. 接受的基础跑酷模型在保守间距自然障碍中跑完 40/40 个 60 秒对局；另有
   20/20 个保守被攻击场景跑完上限。原始记录在 `assets/runner_evaluation/`。
7. 主对战配置的少量规则校准：关闭自然障碍后，攻击不会再通过挤占自然队列来帮助
   对手；紧密但可行的组合能造成基础控制器失误。记录在 `validation/game_calibration.json`。
8. 可选原版 aome510 模型：校验固定 checkpoint，使用受限 allowlist 成功加载；
   在原版环境实际运行两局，分别在 260 和 243 个 wrapper step 后碰撞，累计奖励
   为 27 和 20。记录在 `validation/upstream_adapter.json`。这是接口验证，不是广泛跑分。
9. 用真实 headless Chromium 打开离线回放和报告，检查桌面与窄屏布局、拖动进度、
   播放/暂停。没有 JavaScript 错误，窄屏没有水平溢出。截图仅用于交付前检查。
10. 交付 ZIP 进行 CRC 和逐文件 SHA-256 校验。可用 `python scripts/verify_release.py`
    检查未修改的发行包。
11. 构建并安装 Python wheel，在源码目录外运行 doctor，确认安装包也包含默认配置和基础权重。

1.0 发布时自动化测试结果为 **74 passed**；1.1 已增至上面的 82 项。原完整 smoke 共执行 3 个 run、768 个双人环境 step，
随后完成 6 组对阵、36 局评测。smoke 使用 `2**31` 起始的开发检查 seed；正式预设
使用 `2**32` 起始的评测 seed。基础模型验收与规则校准另有明确记录的开发 seeds。

## 不据此作出的结论

- smoke 结果不说明 JEPA 优于 baseline。两次更新和少量评测远不足以回答研究问题。
- 基础模型的保守场景验收不说明紧密对抗中的最优性。验收 seeds 曾用于开发，不能
  再当作未见的最终测试集。
- 方差正则不保证所有长期运行都不会表示塌缩；报告记录 `latent_std` 等指标用于检查。
- Linux CPU 已执行验证。CUDA、Apple MPS、Windows 和 Docker 路径已经实现，
  但本次没有对应硬件/平台的实际运行记录。依赖安装、驱动可用性与位级重现需分别看待。
- 1.1 包含完成指定预算的六个主模型，但不把它们称为竞技冠军或最优策略。原 `standard` 大预算、完整联合 motor 训练与更多 JEPA 机制消融尚未执行。
- 游戏价格、冷却和间距经过功能校准，并非已经证明平衡。允许通过配置研究不同设置。

## 复核命令

```bash
.venv/bin/python -m pytest -q
.venv/bin/python run.py doctor
.venv/bin/python run.py experiment --config smoke --out runs/independent_smoke
.venv/bin/python run.py experiment --config smoke --out runs/independent_smoke --resume
python3 scripts/verify_release.py
```

运行生成的新 `runs/` 文件不影响发行清单；修改原始配置或代码会被完整性检查标记为 changed，
这不表示用户修改有错误。

