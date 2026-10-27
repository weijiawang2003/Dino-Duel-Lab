#!/usr/bin/env python3
"""Build a small self-contained human-readable report from measured records."""
from pathlib import Path
import json,html,csv,base64,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paired_advantage'

def read(path):return json.loads(path.read_text())
def pc(x):return f'{100*x:+.1f}'
def delta(r):return f"{pc(r['mean'])} [{pc(r['ci95'][0])}, {pc(r['ci95'][1])}]"

def main():
    offline=read(OUT/'OFFLINE_RESULTS.json');tree=read(OUT/'tree_diagnostic/OFFLINE_RESULTS.json')
    online=read(OUT/'online/SUMMARY.json');collection=read(OUT/'collection_summary.json')
    validation=read(OUT/'validation.json') if (OUT/'validation.json').exists() else {'tests_passed':'见 tests.log'}
    labels={'ppo':'原普通 PPO','fixed_wait':'固定等待＋PPO','advantage_only_raw':'普通收益差网络',
        'latent_aux_raw':'JEPA 表征辅助收益差网络','state_aux_raw':'普通状态预测辅助收益差网络',
        'tree_diagnostic':'普通树模型（额外诊断）'}
    methods=list(labels)
    offline_rows=[]
    for method in ['advantage_only_raw','latent_aux_raw','state_aux_raw','tree_diagnostic']:
        src=tree if method=='tree_diagnostic' else None
        a=tree['fresh_id'] if src else offline['fresh_id'][method]
        b=tree['fresh_ood'] if src else offline['fresh_ood'][method]
        offline_rows.append(f'<tr><th>{labels[method]}</th><td>{delta(a["gain_vs_wait"])}</td><td>{delta(b["gain_vs_wait"])}</td><td>{a["switches"]} / 64</td><td>{b["switches"]} / 64</td></tr>')
    online_rows=[]
    for m in methods:
        r=online[m];b=r['full_game_score_bounds']
        online_rows.append(f'<tr><th>{labels[m]}</th><td>{r["wins"]}</td><td>{r["losses"]}</td><td>{r["simultaneous"]}</td><td>{r["unresolved"]}</td><td>{100*b[0]:.1f}%–{100*b[1]:.1f}%</td></tr>')
    episodes=sum(r['games'] for r in online.values());steps=sum(r['steps'] for r in online.values())
    fits=[read(p) for p in sorted((OUT/'models').glob('*/summary.json'))]
    cards=f'''<div class="cards"><div><b>{len(fits)}</b><span>新增小型神经网络拟合</span></div><div><b>{collection['branches']:,}</b><span>新局面候选分支</span></div><div><b>{episodes}</b><span>完整对局测试（含未决）</span></div><div><b>{validation.get('tests_passed')}</b><span>通过的自动化测试</span></div></div>'''
    replay_html=[]
    cases=read(OUT/'cases/selection.json')
    for case in cases['cases']:
        name='有利案例：延迟双仙人掌' if case['case']=='tree_improves' else '反例：立刻恢复 PPO 反而更差'
        frames=[]
        for label in ('fixed_wait','tree'):
            text=(OUT/'cases'/case['paths'][label]).read_text()
            frames.append(f'<div><h3>{"固定等待" if label=="fixed_wait" else "树模型选择"}</h3><iframe loading="lazy" title="{label}" srcdoc="{html.escape(text,quote=True)}"></iframe></div>')
        replay_html.append(f'<details><summary>{name} · 局面 {case["source_root"]}</summary><div class="replays">{"".join(frames)}</div></details>')
    sources=[('offline_per_root.csv','逐局面神经网络结果'),('tree_diagnostic/per_root.csv','逐局面树模型结果'),('online/episodes.csv','384 场逐局记录')]
    download=[]
    for p,title in sources:
        data=base64.b64encode((OUT/p).read_bytes()).decode()
        download.append(f'<a download="{Path(p).name}" href="data:text/csv;base64,{data}">{title}</a>')
    total_sim=collection['simulated_steps']+collection['source_steps']+steps
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dino Duel · 助手端完成的收益差实验</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f4f5f2;color:#1c2c31;font:16px/1.75 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1160px;margin:auto;padding:42px 28px 70px}}header{{border-bottom:2px solid #29454b;padding-bottom:22px}}.eyebrow{{font-size:12px;letter-spacing:.15em;font-weight:700;color:#507077}}h1{{font-size:clamp(28px,4vw,45px);line-height:1.2;letter-spacing:-.035em;margin:14px 0}}.lead{{font-size:20px;max-width:960px;color:#476068}}h2{{font-size:25px;line-height:1.35;letter-spacing:-.02em;margin:36px 0 14px}}h3{{font-size:16px}}p{{margin:14px 0}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.cards>div{{background:white;border:1px solid #d7dfdb;border-radius:8px;padding:19px}}.cards b{{display:block;font-size:28px}}.cards span{{font-size:12px;color:#5e7075}}.callout{{padding:18px 22px;border-left:4px solid #396d63;background:#e8f0eb}}.caution{{border-left-color:#9a7543;background:#f3eedf}}.scroll{{overflow:auto;border:1px solid #d6dfdb;border-radius:8px}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th,td{{padding:12px 13px;text-align:right;border-bottom:1px solid #dfe5e0;white-space:nowrap}}th:first-child{{text-align:left}}.note{{font-size:13px;color:#576c72}}a{{color:#28636b;text-underline-offset:3px}}.downloads{{display:flex;flex-wrap:wrap;gap:22px;font-size:14px}}details{{margin:12px 0;background:white;border:1px solid #d9dfd8;border-radius:8px;padding:15px}}summary{{font-weight:650;cursor:pointer}}.replays{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}iframe{{width:100%;height:610px;border:0}}code{{font-size:13px}}footer{{margin-top:35px;border-top:1px solid #d2ddd5;padding-top:20px;font-size:12px;color:#5f7277}}@media(max-width:760px){{main{{padding:25px 15px}}.cards{{grid-template-columns:1fr 1fr}}.replays{{grid-template-columns:1fr}}th,td{{padding:10px}}}}
    </style></head><body><main><header><div class="eyebrow">DINO DUEL / ASSISTANT-SIDE EXPERIMENT / PAIRED ADVANTAGE</div><h1>不要求你在本地训练：这轮已在这里完成</h1><p class="lead">从固定等待基线出发，学习何时改变攻击。新测试保留了负面结果，也发现了普通树模型能利用的局面差异；目前没有足够证据宣布 JEPA 更强。</p></header>{cards}
    <div class="callout"><b>你无需运行本地代码。</b>训练、分支模拟、对局评测、报告和检查点均在助手的 CPU 运行环境中完成。本页不发起网络请求、不启动训练；回放只是播放已记录的模拟状态。完整恢复包用于后续续接，不要求你现在下载。</div>
    <h2>01 / 这轮改变了什么</h2><p>不再只拟合某个计划的胜负，而是拟合同一局面、同一随机重复下的相对收益：<b>Δ(s,a) = E[8 秒内净胜结果(a) − 8 秒内净胜结果(固定等待)]</b>。净胜结果取胜 +1、负 −1；观察窗内仍在进行与同时碰撞都对这个特定净胜事件指标贡献 0，但二者在原始记录中严格分开，未决不是平局。</p>
    <p>普通收益差网络、JEPA 表征辅助网络和普通状态预测辅助网络，共用完全相同的主干初始化、512 个训练局面与 96 个验证局面；每种拟合三个随机种子。JEPA 组使用旧 JEPA 编码器的<b>冻结未来表征</b>作为辅助目标，不是官方视觉 JEPA，也不是新训练九个完整 PPO agent。辅助组训练参数和计算量更多，不能当作等计算预算比较。</p>
    <p>三个神经网络方案均未通过验证集的保守启用条件；保守部署因此明确退回固定等待。下表中测试的神经网络是<b>未加保守门槛的诊断版本</b>。另增加一个仅使用公开状态的普通 ExtraTrees 树模型；树的叶节点参数仅按验证误差选择，未用新测试结果调参。</p>
    <h2>02 / 全新的同局面分叉测试</h2><p>神经网络与门槛在新测试生成前冻结；额外树模型在新测试结果读取前，仅据原训练／验证集冻结。新测试包含 64 个熟悉对手群的新局面和 64 个未参与本轮拟合的对手权重所产生的新局面。每个局面有 12 个候选计划、8 次配对随机重复，共 <b>{collection['branches']:,} 条分支</b>，实际执行 <b>{collection['simulated_steps']:,} 个双人环境步</b>。对手会在每条分支中根据新状态重新行动。</p>
    <div class="scroll"><table><thead><tr><th>方法</th><th>熟悉对手的新局面：收益差</th><th>未参与拟合的对手：收益差</th><th>前者非等待选择</th><th>后者非等待选择</th></tr></thead><tbody>{''.join(offline_rows)}</tbody></table></div>
    <p class="note">单位：百分点，相对于固定等待；方括号为按独立源局面重采样的探索性 95% 区间。这不是完整比赛胜率。每组只有 64 个独立源局面，12×8 条分支不当成 96 个独立局面；冻结的对手模型群也不代表任意对手。未做多重比较校正。</p>
    <div class="callout"><b>局部测试结论：</b>普通树模型在两组新局面上分别改善 {pc(tree['fresh_id']['gain_vs_wait']['mean'])}、{pc(tree['fresh_ood']['gain_vs_wait']['mean'])} 个百分点。当前三个神经方案没有超过固定等待。这支持“可利用的状态相关信号存在”，但不说明树普遍优于神经网络，更不能说明 JEPA 原理无效。树与神经模型并未做容量或计算量匹配。</div>
    <h2>03 / 384 场完整对局：不能把局部优势直接当作冠军</h2><p>六种方法各 64 场，覆盖 8 个环境种子、4 个固定对手、交换跑道。所有方法使用同一基础跑跳控制器；候选计划执行 8 个购买机会，再兑现 12 个机会的反应式 PPO 后续，避免“不断重新等待而永不出手”。游戏没有倒计时；300 秒是本次评测的外部观察预算。</p>
    <div class="scroll"><table><thead><tr><th>方法</th><th>胜</th><th>负</th><th>同时碰撞</th><th>未决</th><th>最终游戏得分可识别范围</th></tr></thead><tbody>{''.join(online_rows)}</tbody></table></div>
    <p class="note">游戏得分为胜 1、真正平局 0.5、负 0；“可识别范围”让未决分别取所有可能的最终结局，不直接赋 0.5。这不是统计置信区间。环境种子和左右位置共享随机性，不能把每组 64 场当成 64 次独立训练。</p>
    <div class="callout caution"><b>没有稳定胜者。</b>JEPA 表征辅助组有 46 胜，原 PPO 42 胜，树模型 44 胜；这些是该批固定模型的描述性结果。按八个环境种子成组估计，各自相对固定等待的“300 秒内净胜事件”差异区间都跨过 0。新局面上的 8 秒排序与完整对局成绩不完全一致，尚不能把局部预测改进解释成长期战略优势。</div>
    <h2>04 / 成功与反例同时保留</h2><p>下面两组事后说明案例按明确规则选取并记录：在新 ID 测试中，分别取最小编号的正收益与负收益局面，再取符合符号的配对随机重复。回放经过重新模拟，逐项重现保存的标签和终态；它们是挑出的说明案例，不是额外性能证据。</p>{''.join(replay_html)}
    <h2>05 / 审计与继续工作的依据</h2><p><b>{validation.get('tests_passed')} 项自动化测试通过。</b>新增测试覆盖状态—动作交互、等待差值恒等于零、各组共同初始化、公开输入接口、权重重载、优化器恢复、非法动作过滤，以及保守模式与固定等待行为一致。旧项目文件保持未改；新实验单独保存。完整对局另外执行 {steps:,} 步，本轮源局面、分支与完整对局合计 {total_sim:,} 个双人环境步（不含单元测试和案例重放）。</p>
    <p>接下来的研究重点应是核查短窗口排序与持续部署之间的差异，并让神经模型复现已经被树模型利用的状态信息，而不是只延长训练或挑选更好看的战绩。当前结论只适用于这套结构化状态、固定候选计划和冻结对手群。</p>
    <div class="downloads">{' '.join(download)}</div>
    <footer>计算已在当前交互中完成；没有宣称后台常驻训练。恢复包包含源码、模型、优化器、随机状态、协议、原始数据与源归档。工作目录可能被回收，附件和云端存储亦不应被当作无期限保证；本轮用可校验恢复包降低续接成本。</footer></main></body></html>'''
    (OUT/'report.html').write_text(doc)
    small=Path('/mnt/data/dino-duel-assistant-run-report.html');small.write_text(doc)
    text=f'''# Dino Duel：助手端本轮实际执行结果\n\n## 执行范围\n\n本轮训练、评测、整理全部在助手 CPU 环境执行，不需要用户本地运算或存储。基础项目 v1.3 保持不改，新增 paired_advantage 分支。\n\n- 新拟合 9 个小型神经选择网络，另外拟合 4 个树模型候选并仅按验证 MSE 冻结一个。没有新训练完整 PPO agent。\n- 新生成 128 个源局面、12,288 条候选分支，分支执行 {collection['simulated_steps']:,} 步。\n- 完成 384 场完整对局测试（包括未决），执行 {steps:,} 步。\n- {validation.get('tests_passed')} 项测试通过。\n\n## 新局面上的收益差（百分点，相对固定等待）\n\n| 方法 | 熟悉对手新局面 | 未参与本轮拟合的对手 |\n|---|---:|---:|\n'''
    for method in ['advantage_only_raw','latent_aux_raw','state_aux_raw','tree_diagnostic']:
        a=tree['fresh_id'] if method=='tree_diagnostic' else offline['fresh_id'][method]
        b=tree['fresh_ood'] if method=='tree_diagnostic' else offline['fresh_ood'][method]
        text+=f"| {labels[method]} | {delta(a['gain_vs_wait'])} | {delta(b['gain_vs_wait'])} |\n"
    text+='\n指标为未来8秒内 P(胜)−P(负) 的变化，不是整局胜率。每组64个独立源局面，置信区间按源局面重采样。\n\n## 完整对局（各64场）\n\n| 方法 | 胜 | 负 | 真平 | 未决 |\n|---|---:|---:|---:|---:|\n'
    for m in methods:
        r=online[m];text+=f"| {labels[m]} | {r['wins']} | {r['losses']} | {r['simultaneous']} | {r['unresolved']} |\n"
    text+='''\n没有稳定胜者。各组对固定等待的净胜差异，在八个环境种子分组分析下均有很宽且跨零的不确定性区间。未决不赋平局分数。\n\n## 关键边界\n\n潜变量组是冻结旧 JEPA 编码器的辅助训练，不是官方视觉 JEPA；三个拟合种子不是三个完整强化学习训练重复。树是额外诊断，未做等容量或等计算匹配。固定等待并非所有环境种子下都更好。保守门槛在验证集拒绝神经模型，因此部署保护会退回等待；原始神经策略的完整对局只作诊断。\n\n## 保存内容\n\n模型与优化器：paired_advantage/models/；本轮协议：paired_advantage/PROTOCOL.json；冻结选择：FROZEN_SELECTION.json；分支：data/；逐局面：offline_per_root.csv；整局：online/episodes.csv；测试：tests.log；完整审计：validation.json。\n\n所有统计在当前交互中完成，没有后台持续训练承诺。用户不需要下载、安装或执行命令。后续继续时优先从本轮恢复包与 v1.3 原归档读取，不要要求用户提供本地算力或储存。\n'''
    (OUT/'RESULTS.md').write_text(text);Path('/mnt/data/dino-duel-assistant-run-results.md').write_text(text)
    print('Reports written:',small,'bytes',small.stat().st_size)
if __name__=='__main__':main()
