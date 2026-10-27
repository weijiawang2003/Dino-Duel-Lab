# 1.2 实际验证记录

机器可读信息：`../VALIDATION.json`。本文件只描述这次实际执行的验证；旧版保存在 `archive-v1.1/VALIDATION.md`。

## 真实训练与评测

六个主模型（普通PPO、JEPA辅助PPO，各3训练种子）各完成300更新、614,400双人环境步，共 **3,686,400新环境步**。共同底层技能另外训练36,000环境步，并重跑验证权重、元数据逐项完全一致。旧baseline编码器是双方共同热启动，攻击/价值头重新初始化。

主环境无限时、无自然障碍、无强制加速。3,620局主/诊断评测，加上同样300局直接对战的900秒延长观察复测，共 **3,920条记录**；后300局不是新增独立样本。另40条完全重复的初始共同防守记录保留在diagnostics，不重复计数。自动审计核实：所有已观测死亡都来自玩家购买障碍；没有clock终局；未决没有胜者或得分。

匹配初始化：每个种子16个共同参数张量逐项一致；行动参数量同为30,411。JEPA有额外训练模块，训练计算不匹配。所有最终权重有限，推理权重与恢复checkpoint相同；共同motor未更新，motor修正权重确实改变。所有更新指标有限。

## 代码与可迁移性

**94项pytest全部通过**，XML见 `../validation/pytest-elimination.xml`。覆盖无时限游戏、终局与未决区分、长对局跨rollout续接、多步标签不跨reset、表征梯度到攻防头、NumPy/PyTorch推理一致、合法队列、精确恢复、旧checkpoint兼容。

实际将JEPA第300轮恢复到301轮，故意令外部原始warmstart和runner路径不存在，仍能从嵌入权重和现场状态完成更新；原正式模型未改。记录在 `../elimination/portable_resume_check.json`。

成功构建并在独立venv安装wheel，从项目目录以外运行doctor和已训练模型的联合采样回放。此次venv共享容器已有依赖，不声称重新联网安装了所有依赖。

Chromium检查了报告筛选、回放播放/暂停/拖动、终局状态；无JS错误，无外部网络请求。容器禁止直接file导航，因此将原HTML原样传入浏览器set_content进行功能测试；独立报告的内嵌JSON下载和回放弹窗也已验证。具体记录在 `../elimination/browser_check.json`。

## 实际平台与局限

Linux CPU，Python 3.13.5，NumPy 2.3.5，PyTorch 2.10.0+cpu。GPU/MPS/Windows路径未在相应硬件执行。没有执行等计算量消融或推理时规划；这里的JEPA是结构化状态的多步潜变量辅助训练。

统计独立单位不是视频帧或交换座位后的单局。只有3个训练种子，环境种子配对且各模型共享；延长观察保持同样随机轨迹。所有原先已决对局在延长观察中逐行完全一致。不能把JEPA某一组领先当作普遍优越性。

## 复核

```bash
python -m pytest -q
python scripts/validate_elimination.py --out elimination
python scripts/elimination_suite.py report --out elimination
python scripts/verify_release.py
```

自动测试和完整性检验在用户修改配置后可以出现有解释的差异；修改不代表文件损坏。
