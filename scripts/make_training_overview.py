#!/usr/bin/env python3
"""Create a standalone offline overview from packaged, completed results."""
import argparse
import base64
import html
import json
from pathlib import Path
from report_fonts import embed_report_font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", default="trained")
    args = parser.parse_args()
    root = Path(args.trained)
    load = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
    main_result = load("final/comparison.json")
    secondary = load("secondary_greedy/comparison.json")
    initial = load("initial_final/matrix.json")
    training = load("training_summary.json")
    comparison = main_result["comparison"]
    h2h = comparison["head_to_head"]
    counts = h2h["descriptive_pooled_counts_for_baseline"]
    ci = h2h["jepa_score"]["crossed_ci95"]
    fmt = lambda x: f"{100*x:.1f}%"
    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    before = {style: sum(r["mean_score"] for r in initial["matchups"] if r["opponent"] == 'bc:' + style) / 3
              for style in ('random', 'burst')}
    labels = {'random': '随机攻击', 'aggressive': '尽快购买', 'save': '优先昂贵障碍', 'burst': '存钱后双仙人掌连招', 'macro': '四对手等权平均'}
    rows = []
    for style in ('random', 'aggressive', 'save', 'burst', 'macro'):
        rows.append([labels[style], fmt(before[style]) if style in before else '未测',
                     fmt(comparison['fixed_opponents']['baseline'][style]['mean']),
                     fmt(comparison['fixed_opponents']['jepa'][style]['mean'])])
    fixed = table(['共同对手', '初始攻击网络', '普通 PPO', 'PPO + JEPA'], rows)
    direct = table(['训练种子', 'PPO 胜', '平', 'JEPA 胜', 'JEPA 得分'],
                   [[s, r['wins'], r['draws'], r['losses'], fmt(1-r['baseline_score'])]
                    for s, r in h2h['per_training_seed'].items()])
    def download(relative, filename, label, mime='application/json'):
        encoded = base64.b64encode((root / relative).read_bytes()).decode('ascii')
        return f'<a class="button" download="{html.escape(filename, quote=True)}" href="data:{mime};base64,{encoded}">{html.escape(label)}</a>'
    replays = ''.join([
        download('replay.html', 'PPO-vs-JEPA-replay.html', '下载 PPO × JEPA 对局', 'text/html'),
        download('final/fixed/baseline/seed_11/burst/replay.html', 'PPO-vs-burst-replay.html', '下载 PPO × 连招对手', 'text/html'),
        download('final/fixed/jepa/seed_11/burst/replay.html', 'JEPA-vs-burst-replay.html', '下载 JEPA × 连招对手', 'text/html'),
    ])
    records = ''.join([
        download('final/comparison.json', 'primary-comparison.json', '主结果 JSON'),
        download('secondary_greedy/comparison.json', 'greedy-comparison.json', '确定性出招 JSON'),
        download('final/matchups.csv', 'primary-matchups.csv', '主结果 CSV', 'text/csv'),
        download('training_summary.json', 'training-summary.json', '训练量与模型哈希'),
    ])
    per_run = [r for r in training['runs'] if r['main_comparison']]
    elapsed = {v: sum(r['elapsed_seconds'] for r in per_run if r['variant'] == v)/3
               for v in ('baseline', 'jepa')}
    greedy_h2h = secondary['comparison']['head_to_head']
    macro_gap = comparison['paired_fixed_score_differences']['macro']
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JEPA vs 普通 PPO · 实际训练结果</title>
<style>:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#0b111a;color:#e5edf7;font:16px/1.75 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:44px 24px 70px}}.eyebrow{{color:#62dfc4;letter-spacing:2px;font-size:12px}}h1{{font-size:clamp(29px,4vw,44px);line-height:1.25;margin:14px 0 18px}}h2{{font-size:24px;margin:35px 0 12px}}p{{margin:12px 0}}.lead{{font-size:19px;color:#c8d8e9}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:25px 0}}.card,.box{{background:#131e2c;border:1px solid #2a3c51;border-radius:12px;padding:20px}}.card strong{{display:block;color:#62dfc4;font-size:27px;line-height:1.4}}.muted,.card span{{color:#a5b6ca}}.box{{border-left:4px solid #62dfc4}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0}}th,td{{border-bottom:1px solid #2a3c51;padding:12px 15px;text-align:left;white-space:nowrap}}th{{color:#62dfc4;background:#131e2c}}.buttons{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}.button{{display:inline-block;padding:8px 14px;border:1px solid #436277;border-radius:8px;color:#9aebda;text-decoration:none;background:#152335}}code{{background:#172435;padding:1px 5px;border-radius:4px}}footer{{border-top:1px solid #2a3c51;margin-top:34px;padding-top:20px;color:#95a8bd;font-size:13px}}@media(max-width:600px){{main{{padding:25px 16px 45px}}.grid{{grid-template-columns:1fr}}.card{{padding:14px 18px}}.card strong{{font-size:24px}}.lead{{font-size:17px}}}}</style></head><body><main>
<div class="eyebrow">DINO DUEL LAB 1.1 · 实际训练与保留种子评估</div>
<h1>JEPA vs 普通 PPO<br>训练有效，JEPA 的稳定优势尚未确立</h1>
<p class="lead">两组都学到了更强的攻击策略。JEPA 在采样出招的直接对战中点估计略高，但共同对手综合成绩接近，确定性出招时优势也没有保持。</p>
<div class="grid"><div class="card"><strong>{training['main_training_steps']:,}</strong><span>主实验双人环境步</span></div><div class="card"><strong>2 方法 × 3 种子</strong><span>相同初始行动网络与训练预算</span></div><div class="card"><strong>6,000 局</strong><span>主、次要与初始模型测试</span></div></div>
<div class="box"><strong>阅读分数：</strong>胜 1、平 0.5、负 0。表中的百分数是平均得分，不是去掉平局后的胜率。相同环境种子会换边、跨模型重复使用；6,000 局不等于 6,000 次独立训练。</div>
<h2>1. 两组确实变强了</h2><p>每个已训练表项有 300 局：3 个训练种子 × 50 个保留环境种子 × 双座位。初始网络使用同样三组权重初始化；基础跑跳模型始终相同。</p>{fixed}
<p class="muted">“双仙人掌连招”是手写的强测试对手：攒够两个小仙人掌的钱，再连续购买。它不是训练成果。对 random 和 burst 的初始控制共 600 局。</p>
<h2>2. JEPA 与普通 PPO 直接对战</h2><p>采样出招的 300 局中，<strong>JEPA {counts['losses']} 胜，PPO {counts['wins']} 胜，{counts['draws']} 平</strong>。JEPA 得分 {fmt(h2h['jepa_score']['mean'])}，普通 PPO 得分 {fmt(h2h['baseline_score']['mean'])}。</p>{direct}
<p>考虑训练种子与共享环境种子的交叉 bootstrap，JEPA 得分 95% 区间为 <strong>{fmt(ci[0])}–{fmt(ci[1])}</strong>，仍包含 50%。只有三个训练种子，这个结果不足以确认稳定优势。</p>
<p>四种固定对手的宏平均得分，JEPA 比普通 PPO 高 <strong>{100*macro_gap['mean']:.2f} 个百分点</strong>；分差的交叉区间跨过 0。原始记录保留全部胜平负。</p>
<h2>3. 确定性出招是另一幅图景</h2><p>选择概率最大的攻击动作后，JEPA 直接对战得分为 <strong>{fmt(greedy_h2h['jepa_score']['mean'])}</strong>，PPO 为 <strong>{fmt(greedy_h2h['baseline_score']['mean'])}</strong>，300 局有 {greedy_h2h['descriptive_pooled_counts_for_baseline']['draws']} 局平局。该模式下两组基本打平。</p>
<p class="muted">主分析事先固定为与训练一致的采样攻击；确定性模式是单列的次要分析，没有根据最终成绩挑选更有利的模式。</p>
<h2>4. 这次的 JEPA 做了什么</h2><p>普通组是 masked PPO 自博弈。JEPA 组在同一 PPO 上增加动作条件的未来 latent 预测：在线 encoder、predictor、停止梯度的 EMA target encoder 和方差正则。两组的初始行动及价值权重逐 tensor 相同。</p>
<p><strong>预测跨度只有下一操作步，约 1/15 秒；推理时没有多步规划。</strong> 输入为 53 维结构化状态。此次跑跳网络冻结，增强的是攻击策略；能赢双仙人掌对手不代表已经学会躲开该连招，也可能是先击败对方。</p>
<p>两边行动参数均为 30,024；可训练参数为 PPO 24,198、JEPA 59,270。每个模型平均记录耗时分别为 {elapsed['baseline']:.1f} 秒和 {elapsed['jepa']:.1f} 秒。多个任务在同一 CPU 机器上并行执行，这些数字不是用户等待时间或严格硬件吞吐基准。</p>
<h2>5. 看真实对局与下载原始结果</h2><p>以下均为训练种子 11、第一个保留环境种子的真实记录，按固定顺序取首局，没有为演示选择胜局。下载回放 HTML 后可离线播放、暂停和拖动。</p><div class="buttons">{replays}</div>
<p>模型权重、完整继续训练状态、逐局 CSV、代码及一键复现命令都在完整项目 ZIP 内。下列数据也已嵌入此报告，可直接下载。</p><div class="buttons">{records}</div>
<footer>数据来源：本项目真实日志与模拟对局。所有主模型固定为 update 300，未从最终测试挑选 checkpoint。统计复核修正了原嵌套重采样对共享环境相关性的处理；点估计和逐局 CSV 均未改变。正式协议、开发验证及完整边界见项目内 trained/RESULTS.md 与 docs/TRAINING_COMPARISON_PROTOCOL.md。本报告无需网络。</footer>
</main></body></html>'''
    (root / "index.html").write_text(embed_report_font(document), encoding="utf-8")
    print(root / 'index.html')


if __name__ == '__main__':
    main()
