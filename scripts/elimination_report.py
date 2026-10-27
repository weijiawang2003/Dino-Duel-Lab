"""Generate an audit-friendly offline report solely from recorded files."""
from __future__ import annotations
import csv,html,json,hashlib
from pathlib import Path
import numpy as np


def read_rows(path):
    rows=[]
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            row['left_score']=None if row['left_score']=='' else float(row['left_score'])
            row['seconds']=float(row['seconds']);row['seed']=int(row['seed'])
            row['left_seat']=int(row['left_seat']);row['censored']=row.get('censored','False')=='True'
            rows.append(row)
    return rows


def group_result(directories):
    raw=[read_rows(d/'episodes.csv') for d in directories]
    rows=[r for group in raw for r in group]
    if not rows:return {}
    w=sum(x['left_score']==1 for x in rows);l=sum(x['left_score']==0 for x in rows)
    d=sum(x['left_score']==.5 for x in rows);c=sum(x['left_score'] is None for x in rows);n=len(rows)
    result={'games':n,'wins':w,'losses':l,'draws':d,'censored':c,
            'score_bounds':[(w+.5*d)/n,(w+.5*d+c)/n],
            'resolved_score':(w+.5*d)/(n-c) if n>c else None,
            'seconds_quantiles':dict(zip(['min','median','p90','max'],np.quantile([x['seconds'] for x in rows],[0,.5,.9,1]).tolist())),
            'games_beyond_60s':sum(x['seconds']>60+1e-6 for x in rows),
            'bootstrap95_envelope':None,'duration_note':'Observed time, with administrative right censoring; not uncensored full match length.'}
    # Paired training-seed rows, common environment columns, both seats grouped.
    seed_sets=[sorted({x['seed'] for x in group}) for group in raw]
    if len(raw)>=2 and all(x==seed_sets[0] for x in seed_sets):
        lower=[];upper=[]
        for group in raw:
            lo=[];hi=[]
            for seed in seed_sets[0]:
                scores=[x['left_score'] for x in group if x['seed']==seed]
                lo.append(np.mean([0 if x is None else x for x in scores]))
                hi.append(np.mean([1 if x is None else x for x in scores]))
            lower.append(lo);upper.append(hi)
        lower=np.asarray(lower);upper=np.asarray(upper)
        rng=np.random.default_rng(20261025);los=[];his=[]
        for _ in range(4000):
            rr=rng.integers(0,len(raw),len(raw));cc=rng.integers(0,len(seed_sets[0]),len(seed_sets[0]))
            los.append(lower[rr][:,cc].mean());his.append(upper[rr][:,cc].mean())
        result['bootstrap95_envelope']=[float(np.quantile(los,.025)),float(np.quantile(his,.975))]
        result['uncertainty_note']='Crossed resampling of training-seed rows and shared environment-seed columns. Only three training seeds; censor envelope includes unknown outcomes, not a score estimate.'
    return result


def make_report(root):
    root=Path(root);evaluation=root/'evaluations'
    results={};summaries={}
    for folder in sorted(evaluation.glob('*')) if evaluation.exists() else []:
        if (folder/'summary.json').exists():summaries[folder.name]=json.loads((folder/'summary.json').read_text())
    for mode in ('joint','attack','greedy','extended_joint'):
        dirs=[evaluation/f'h2h_{mode}_seed{s}' for s in (11,23,47)]
        if all((d/'episodes.csv').exists() for d in dirs):results[mode]=group_result(dirs)
    diagnostic_groups={}
    for category,variants,styles in (
        ('fixed',('baseline','jepa'),('aggressive','burst','cycle')),
        ('unseen',('baseline','jepa'),('wide','alternating')),
        ('defense',('initial','baseline','jepa'),('wide','alternating'))):
        for variant in variants:
            for style in styles:
                source_seeds=(11,) if category=='defense' and variant=='initial' else (11,23,47)
                dirs=[evaluation/f'{category}_{variant}_seed{seed}_{style}' for seed in source_seeds]
                if all((d/'episodes.csv').exists() for d in dirs):
                    diagnostic_groups[f'{category}/{variant}/{style}']=group_result(dirs)
    for key,data in results.items():
        dirs=[evaluation/f'h2h_{key}_seed{s}' for s in (11,23,47)]
        rows=[r for d in dirs for r in read_rows(d/'episodes.csv')]
        data['purchases']={}
        for side in ('left','right'):
            data['purchases'][side]={name:sum(int(r.get(f'{side}_{name}_purchases',0)) for r in rows)
                                     for name in ('cactus','wide_cactus','low_bird','high_bird')}
        data['collision_sources']={}
        for r in rows:
            for side in ('left','right'):
                source=r.get(f'{side}_collision_source','')
                if source:data['collision_sources'][source]=data['collision_sources'].get(source,0)+1
    training=[]
    for file in sorted((root/'runs').glob('*/training_summary.json')):
        row=json.loads(file.read_text());p=file.parent
        row['run_name']=p.name
        row['policy_sha256']=hashlib.sha256((p/'policy.pt').read_bytes()).hexdigest()
        metrics=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
        row['last_metrics']=metrics[-1] if metrics else {}
        training.append(row)
    aggregate={'schema':'dino_duel.elimination_report.v1','main':results,'diagnostics':diagnostic_groups,'training':training,
               'evaluated_games':sum(s['n_games'] for s in summaries.values()),
               'uncertainty':'Three trained seeds, exploratory result. Unknown games are NOT draws. No official pretrained JEPA or latent planner.'}
    (root/'comparison.json').write_text(json.dumps(aggregate,indent=2))
    def pct(v):return f'{100*v:.1f}%'
    def bounds(v):return pct(v[0]) if abs(v[0]-v[1])<1e-10 else pct(v[0])+' – '+pct(v[1])
    def escape(x):return html.escape(str(x))
    mainrows=''
    for mode,data in results.items():
        ci=data['bootstrap95_envelope']
        mainrows+=f"<tr><th>{escape(mode)}{' · 主分析' if mode=='joint' else ''}</th><td>{data['wins']} / {data['draws']} / {data['losses']}</td><td>{data['censored']}</td><td>{bounds(data['score_bounds'])}</td><td>{bounds(ci) if ci else '—'}</td><td>{data['seconds_quantiles']['median']:.1f} s</td><td>{data['games_beyond_60s']}</td></tr>"
    pairedrows=''
    for seed in (11,23,47):
        d=evaluation/f'h2h_joint_seed{seed}'
        if (d/'episodes.csv').exists():
            r=group_result([d])
            pairedrows+=f"<tr><th>{seed}</th><td>{r['wins']}</td><td>{r['losses']}</td><td>{r['draws']}</td><td>{r['censored']}</td><td>{bounds(r['score_bounds'])}</td></tr>"
    detailrows=''
    for name,summary in summaries.items():
        replay=f'<a href="evaluations/{escape(name)}/replay.html">回放</a>' if (evaluation/name/'replay.html').exists() else ''
        detailrows+=f"<tr><td>{escape(name)}</td><td>{summary['wins']} / {summary['draws']} / {summary['losses']}</td><td>{summary.get('censored',0)}</td><td>{bounds(summary.get('left_score_identification_bounds',[summary['left_mean_score']]*2))}</td><td>{summary['mean_seconds']:.1f} s</td><td><a href='evaluations/{escape(name)}/episodes.csv'>CSV</a> {replay}</td></tr>"
    diagnosticrows=''
    for key,data in diagnostic_groups.items():
        category,variant,style=key.split('/')
        meaning=(f"{data['censored']}/{data['games']} 撑过观察窗口" if category=='defense' else bounds(data['score_bounds']))
        diagnosticrows+=f"<tr><td>{escape(category)}</td><td>{escape(variant)}</td><td>{escape(style)}</td><td>{data['wins']} / {data['draws']} / {data['losses']} / {data['censored']}</td><td>{meaning}</td><td>{data['seconds_quantiles']['median']:.1f} s</td></tr>"
    trainrows=''
    for r in training:
        m=r['last_metrics']
        trainrows+=f"<tr><td>{escape(r['run_name'])}</td><td>{r['global_step']:,}</td><td>{r['update']}</td><td>{r['acting_parameter_count']:,}</td><td>{r['elapsed_seconds']:.1f} s</td><td>{m.get('latent_std',0):.3f}</td><td><a href='runs/{escape(r['run_name'])}/policy.pt'>权重</a></td></tr>"
    predictions=[]
    for file in sorted((root/'prediction').glob('seed*/summary.json')):
        predictions.append(json.loads(file.read_text()))
    aggregate['prediction_diagnostics']=predictions
    (root/'comparison.json').write_text(json.dumps(aggregate,indent=2))
    predrows=''
    def number(value):return '—' if value is None else f'{value:.5f}'
    for p in predictions:
        for m in p['horizons']:
            predrows+=f"<tr><td>{p['seed']}</td><td>{m['horizon_seconds']:.2f} s</td><td>{m['windows']}</td><td>{number(m['learned_huber'])}</td><td>{number(m['persistence_huber'])}</td><td>{number(m['shuffled_actions_huber'])}</td></tr>"
    frozen_config=root/'runs/baseline_seed11/config.json'
    cfg=json.loads(frozen_config.read_text()) if frozen_config.exists() else {'model':{},'env':{}}
    env=cfg['env'];hs=cfg['model'].get('prediction_horizons',[])
    hz=env.get('dt',1/60)*env.get('action_repeat',4)
    horizon_description='、'.join(map(str,hs))+' 个操作步（'+ '、'.join(f'{h*hz:.2f}' for h in hs)+' 秒）'
    geometry_description=f"宽仙人掌为宽 {env.get('wide_cactus_width',1.4)} m、高 {env.get('wide_cactus_height',.9)} m"
    totalsteps=sum(r['global_step'] for r in training)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dino Duel Lab 1.2 · 淘汰制实验</title>
<style>body{{margin:0;font:15px/1.7 system-ui,sans-serif;background:#f5f6f7;color:#1b2635}}main{{max-width:1200px;margin:auto;padding:42px 24px}}h1{{font-size:38px;letter-spacing:-.04em;line-height:1.15}}h2{{font-size:23px;margin-top:42px}}.eyebrow{{font-size:12px;letter-spacing:.16em}}.lead{{font-size:19px;max-width:850px}}.cards{{display:flex;gap:20px;flex-wrap:wrap;margin:28px 0}}.card{{background:white;padding:20px 26px;border-radius:12px;flex:1;min-width:160px}}.card b{{display:block;font-size:29px}}.note{{padding:18px 22px;border-left:3px solid #536478;background:white}}table{{border-collapse:collapse;width:100%;font-size:13px;background:white}}th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid #e2e5e9}}th{{font-weight:650}}.scroll{{overflow:auto}}a{{color:#28526d}}.muted{{color:#626c77}}input{{padding:10px;width:340px;max-width:90%;margin:12px 0;border:1px solid #ccd1d8;border-radius:5px}}code{{font-size:13px}}footer{{margin-top:40px;font-size:12px}}</style>
<main><div class="eyebrow">DINO DUEL LAB / VERSION 1.2</div><h1>不靠倒计时判平的对抗</h1><p class="lead">普通 PPO 与多步 JEPA 辅助 PPO。从共同基础技能出发，训练攻击时机和防守调整；胜负由碰撞产生，计算预算耗尽单独记录为未决。</p>
<div class="cards"><div class="card">本轮已完成环境步<b>{totalsteps:,}</b></div><div class="card">已完成训练模型<b>{len(training)}</b></div><div class="card">本报告评测局数<b>{aggregate['evaluated_games']:,}</b></div></div>
<div class="note"><b>边界：</b>这是结构化状态上的 JEPA 风格辅助训练，不是官方视觉 JEPA 权重，也没有推理时潜空间规划。两组行动参数相同、环境交互预算相同，但 JEPA 额外消耗训练计算。只有三个训练种子，结果属于探索性证据。未训练等计算预算、随机标签辅助损失等完整消融，不能排除额外优化或正则化造成的差异。</div>
<h2>真正改变的实验条件</h2><p>主环境没有倒计时、没有自然障碍、没有强制加速。只有双方购买的障碍会造成死亡。{geometry_description}，其他三类不变。新障碍需通过可达性检查：从当前状态至少存在一条可以通过已知障碍队列的动作序列。这个检查不保证被试模型能找到该序列。</p><p>双方共用经过可行动作蒸馏的基础跑跳网络（离线教师可见完整已生成队列），攻击与价值头重新初始化；基础跑跳网络仍作为共同先验，但可学习的防守修正头与攻击头共享表征。本轮 JEPA 预测 {horizon_description}后的潜变量。目标使用 EMA 编码器；辅助标签不跨过对局重置。</p>
<h2>主比较：JEPA 在左，普通 PPO 在右</h2><div class="scroll"><table><tr><th>执行模式</th><th>JEPA 胜 / 真平 / PPO 胜</th><th>未决</th><th>JEPA 得分或可识别区间</th><th>95% 重采样包络</th><th>中位观察时长</th><th>超过60秒</th></tr>{mainrows}</table></div>
<p class="muted">得分为胜1、真平0.5、负0；不是胜率。未决不赋0.5，报告所有可能后续结局对应的得分区间。joint 对攻击和跑跳均采样，与训练一致；attack 只采样攻击；greedy 全取最大概率动作。若无未决，包络退化为分组 bootstrap 区间。三个训练种子很少，不能把大量游戏局数当成大量独立训练重复。extended_joint 是同一对局改用更长外部观察预算的重放，不是额外独立样本。</p>
<h2>主比较按训练种子展开</h2><div class="scroll"><table><tr><th>训练种子</th><th>JEPA胜</th><th>PPO胜</th><th>同时死亡</th><th>未决</th><th>JEPA得分范围</th></tr>{pairedrows}</table></div><p class="muted">不能只选表现最好的种子。这三个配对是独立训练重复；交换跑道和重复环境种子不增加训练重复数。</p><p><a href="showcase/replay.html">观看按时长选择的长对局</a> · <a href="showcase/selection.json">回放选择规则与精确复现记录</a></p><h2>分离攻击与防守的诊断</h2><div class="scroll"><table><tr><th>测试</th><th>模型</th><th>对手程序</th><th>胜 / 真平 / 负 / 未决</th><th>得分范围或存活数</th><th>中位观察时长</th></tr>{diagnosticrows}</table></div><p class="muted">防守测试禁用自身购买，不能获胜；撑过外部观察窗口是存活，不是把未决算胜。initial 防守是同一个共同底座，只统计 seed11 那一份，避免把相同初始化复制三次当三次重复。其他测试同时允许攻防，不能单凭赢局把提升归因于防守。所有组共用更强的基础动作，这部分收益不属于JEPA独有。</p><h2>训练预算与权重</h2><div class="scroll"><table><tr><th>模型</th><th>新环境步</th><th>更新数</th><th>行动参数</th><th>记录耗时</th><th>末段潜变量标准差</th><th>文件</th></tr>{trainrows}</table></div><p class="muted">任务在共享 CPU 上部分并行，耗时不是独占硬件吞吐基准。训练来源、环境配置、参数哈希、全部日志与可恢复 checkpoint 都在压缩包中。预测误差下降或标准差非零并不证明学到了有用的战略。</p>
<h2>全部对战与原始记录</h2><p>fixed 使用新的共同防守底座；unseen 是训练时没有抽样的固定攻击程序；old 是原 v1.1 模型；initial 是本轮未训练初始化。defense 禁用左侧攻击、只测防守存活，未决表示撑过观察预算，不是获胜。cross 为跨训练种子配对，league 为历史检查点对战，stress 改变速度、资源或间距；这些与主比较分开解释。</p><input id="filter" placeholder="筛选：cross / fixed / initial / stress / league…"><div class="scroll"><table id="details"><tr><th>实验</th><th>左胜 / 真平 / 右胜</th><th>未决</th><th>左得分范围</th><th>平均观察时长</th><th>原始记录</th></tr>{detailrows}</table></div>
<h2>预测器是否真的在预测</h2><p>在额外留出的轨迹上，比较学习到的未来潜变量预测、保持当前目标编码不变的基线，以及打乱未来动作序列的预测。Huber误差越低越好；不同模型的潜空间不同，不能跨模型直接比较绝对误差。</p><div class="scroll"><table><tr><th>种子</th><th>跨度</th><th>有效窗口</th><th>学习预测</th><th>状态保持</th><th>动作打乱</th></tr>{predrows}</table></div><p class="muted">这里给预测器提供真实未来双方动作，是离线条件预测诊断；实际出招时没有这些信息。重叠窗口并非独立样本。这不能证明反事实规划准确，也不能把任何胜率差直接归因于世界模型。</p><h2>复现</h2><p><code>python scripts/elimination_suite.py evaluate --out elimination --workers 3 --section all</code></p><p>延长观察使用 <code>--safety-seconds 900</code>；它只改变评测预算，不改变游戏规则。继续训练使用 <code>train --updates 1500</code>，此值是总更新目标而不是追加更新数。缓存包含策略和代码哈希，不会把旧 checkpoint 的评测误用于新模型。</p>
<footer>本页由本地实际记录生成，无联网依赖、无嵌入字体。离线附件需与项目目录一起保留。<br><a href="comparison.json">汇总 JSON</a> · <a href="PROTOCOL.json">协议与范围</a></footer></main><script>document.getElementById('filter').oninput=e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('#details tr').forEach((r,i)=>{{if(i)r.hidden=!r.textContent.toLowerCase().includes(q)}})}};</script></html>'''
    (root/'report.html').write_text(doc,encoding='utf-8')
    return root/'report.html'
