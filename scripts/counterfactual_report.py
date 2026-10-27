#!/usr/bin/env python3
"""Build one offline, self-contained Chinese report from executed artifacts."""
from pathlib import Path
import json,sys,html,base64
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def esc(x):return html.escape(str(x))
def link(path,label,mime='application/json'):
    path=Path(path);data=base64.b64encode(path.read_bytes()).decode()
    return f'<a download="{esc(path.name)}" href="data:{mime};base64,{data}">{esc(label)}</a>'
def pct(x):return f'{100*x:.1f}%'
def pp(x):return f'{100*x:+.1f}'

def build(out):
    out=Path(out);s=json.loads((out/'evaluation/summary.json').read_text());c=json.loads((out/'collection_summary.json').read_text())
    train=json.loads((out/'training_summary.json').read_text());rule=json.loads((out/'rule_audit.json').read_text())
    families={'ppo':'原 PPO · 不规划','best_fixed':'验证集选出的固定计划',
        'uniform_random_macro':'随机选择计划','empirical_oracle':'模拟器参照 · 独立重复验收',
        'latent_additive_ensemble':'加法潜变量预测 · 旧结构对照',
        'latent_nonlinear_ensemble':'非线性潜变量预测 · JEPA 表征',
        'state_nonlinear_ensemble':'普通状态预测 · 同样选招',
        'outcome_only_ensemble':'只学结局 · 无未来表征损失'}
    sections=[]
    for split,title in [('test','未见局面 / 熟悉的对手群'),('ood','未见局面 / 留出的对手权重')]:
        group=s['splits'][split];met=group['metrics'];rows=[]
        for key,name in families.items():
            m=met[key];lo,hi=m['delta_ci95'];rates=m['rates']
            rows.append(f'<tr class="{"focus" if key=="latent_nonlinear_ensemble" else ""}"><th>{name}</th><td>{pct(m["net_win_within_horizon"])}</td><td>{pp(m["delta_vs_ppo"])} pp</td><td>[{pp(lo)}, {pp(hi)}] pp</td><td>{pct(rates["win"])}</td><td>{pct(rates["loss"])}</td><td>{pct(rates["ongoing"])}</td></tr>')
        sections.append(f'<section><div class="kicker">{esc(split.upper())} · {group["root_count"]} 个独立源局面</div><h2>{title}</h2><div class="tablewrap"><table><thead><tr><th>选招方式</th><th>8秒净胜事件率</th><th>相对 PPO 变化</th><th>95% 配对区间</th><th>8秒内赢</th><th>8秒内输</th><th>8秒时仍在进行</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class="caption">净胜事件率 = P(8秒内击败对手) − P(8秒内被击败)，不是完整对局胜率。每个局面用后4次独立随机重复验收；区间按局面配对重采样，不把分支当独立模型。</p></section>')
    notes=['<p><strong>主要结论：这轮没有证明 JEPA 选招更强。</strong>离线收益主要被一个简单的固定等待计划解释；接到每0.4秒重规划的控制器后，模型甚至一直等待，没有购买障碍。</p>']
    for split,label in [('test','熟悉对手群'),('ood','留出对手')]:
        m=s['splits'][split]['metrics'];v=m['latent_nonlinear_ensemble'];fixed=m['best_fixed'];oracle=m['empirical_oracle']
        notes.append(f'<p>{label}：潜变量模型相对 PPO 改善 <b>{pp(v["delta_vs_ppo"])} pp</b>，固定等待为 <b>{pp(fixed["delta_vs_ppo"])} pp</b>。独立重复验收的模拟器选招参照为 {pp(oracle["delta_vs_ppo"])} pp，说明候选中还有改善空间。</p>')
    online=''
    for commitment,heading in [(1,'第一次闭环测试：不断重新规划，却一直不出手'),(20,'事后诊断：兑现等待之后的攻击，能否修复失效？')]:
        op=out/f'online_commit{commitment}/summary.json'
        if not op.exists():continue
        data=json.loads(op.read_text());rows=[]
        for k,name in [('ppo','原 PPO'),('fixed_wait','固定等待 + PPO · 无预测器'),('latent_nonlinear','潜变量规划'),('state_nonlinear','状态预测规划'),('outcome_only','结局预测规划')]:
            if k not in data:continue
            m=data[k];n=m['games'];lower=(m['wins']+.5*m['simultaneous'])/n;upper=lower+m['unresolved']/n
            rows.append(f'<tr><th>{name}</th><td>{m["games"]}</td><td>{m["wins"]}</td><td>{m["losses"]}</td><td>{m["simultaneous"]}</td><td>{m["unresolved"]}</td><td>{pct(lower)}–{pct(upper)}</td></tr>')
        intro=('<p>只执行计划的第一步，每0.4秒重新选招。三种预测组在这64局中均<b>从未购买障碍</b>：模型假设等待3.2秒后会恢复 PPO，但每次又把恢复点推迟。离线目标与实际执行的后续策略不一致。</p>' if commitment==1 else
               '<p><b>明确标为事后修复诊断：</b>保留原模型不重训，用另一组全新种子，先执行完整3.2秒宏计划，再兑现4.8秒反应式 PPO。总共8秒后才重新规划。加上完全相同节奏的固定等待对照，不把承诺机制的收益归给 JEPA。</p>')
        online+=f'<section><div class="kicker">CLOSED LOOP / COMMIT {commitment} PURCHASE OPPORTUNITIES</div><h2>{heading}</h2>{intro}<div class="tablewrap"><table><tr><th>控制器</th><th>局数</th><th>胜</th><th>负</th><th>同时撞死</th><th>未决</th><th>最终得分可识别范围</th></tr>{"".join(rows)}</table></div><p class="caption">模型和跑跳控制器保持冻结。胜1、真实平局0.5、负0；未决不赋分，上下界涵盖所有可能后续结局。外部观察预算300秒，不改变游戏规则。每组仅8个环境种子、两边跑道、4个对手权重，不能当作64次独立强化学习训练。两个协议使用不同新种子，不直接跨表比较差值。</p>{link(out/f"online_commit{commitment}/episodes.csv","下载该协议逐局记录","text/csv")}</section>'
    cases=''
    cp=out/'cases/selection.json'
    if cp.exists():
        info=json.loads(cp.read_text());parts=[]
        for case in info['cases']:
            name='正向案例' if case['case']=='success' else '失败案例'
            frames=[]
            for key,label in [('ppo','原 PPO 的后续'),('planned','模型选择的后续')]:
                text=(out/'cases'/case['paths'][key]).read_text()
                frames.append(f'<div><h4>{label}</h4><iframe title="{name} {label}" srcdoc="{esc(text)}" loading="lazy"></iframe></div>')
            parts.append(f'<details><summary>{name} · {esc(case["root"])} · {esc(case["plan"])} · 配对收益变化 {pp(case["mean_heldout_delta"])} pp</summary><div class="replays">{"".join(frames)}</div></details>')
        cases=f'<section><div class="kicker">SAME STATE / TWO FUTURES</div><h2>同一个局面，实际执行两种选择</h2><p>以下是明确筛选的说明性案例，不是无偏抽样：分别取测试集中编号最小的正向与负向案例，再取第一个相应的验收重复。每条回放已与保存的分支结果逐项复现。</p>{"".join(parts)}{link(cp,"案例选择规则与复现记录")}</section>'
    tr=''.join(f'<tr><td>{esc(r["variant"])}</td><td>{r["seed"]}</td><td>{r["parameters"]:,}</td><td>{r["best_epoch"]} / {r["epochs"]}</td><td>{r["best_validation_nll"]:.4f}</td></tr>' for r in train)
    methods=f'''<section><div class="kicker">METHOD / 必须分开的概念</div><h2>这轮到底训练了什么</h2><p>新训练的是 <b>12个小预测器</b>，不是12个新 PPO agent。所有选招方式共用原普通 PPO 的跑跳和默认行动；潜变量目标使用旧 JEPA 编码器的冻结表征。没有继续训练目标编码器的 EMA，也没有改变游戏难度。</p><p>非线性预测器修复了旧模型里“相同攻击差异不随状态改变”的结构限制。普通状态预测、加法结构和去掉预测损失的结局学习均保留为对照。训练用同一批分支，但不同模型的参数量、继承的预训练和计算量不同，不作等算力胜负声明。</p><p>选择依据为预测的输赢概率；结局解码器在真实未来特征上训练后冻结。预测器同时使用未来特征误差与结局监督，因此这也不是纯自监督实验。</p><details><summary>查看12个实际训练记录</summary><div class="tablewrap"><table><tr><th>条件</th><th>拟合种子</th><th>预测器参数</th><th>验证集选中轮次 / 总轮次</th><th>验证 NLL</th></tr>{tr}</table></div></details></section>'''
    reasons=rule['reasons']
    audit=f'<section><div class="kicker">RULE AUDIT / 没有为了分胜负改游戏</div><h2>攻击规则不是凭感觉改的</h2><p>检查 {rule["roots"]} 个训练快照的四种购买：{reasons["allowed"]} 个合法，{reasons["funds"]} 个金币不足，{reasons["reachability"]} 个被可达性检查拦下。这里只检查当前购买边界，不能推广成所有未来攻击。</p><p>模型不接触真实模拟器状态，也不知道对手未来动作。每条分支的对手都会根据新局面重新行动。模拟器只负责数据与验收参照，不给某个参赛方法额外权限。</p></section>'
    downloads=' · '.join(([link(out/'live_replay.html','首个完整对局回放','text/html')] if (out/'live_replay.html').exists() else [])+[link(out/'evaluation/summary.json','完整机器可读结果'),link(out/'evaluation/root_selections.csv','逐局面选招 CSV','text/csv'),link(out/'PROTOCOL.json','实验协议'),link(out/'training_summary.json','训练汇总')])
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dino Duel v1.3 · 预测能改善出招吗？</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f4ef;color:#19252b;font:15px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1240px;margin:auto;padding:44px 30px 70px}}header{{border-bottom:2px solid #172c36;padding-bottom:28px}}.kicker{{font-size:11px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#536b72}}h1{{font-size:clamp(29px,4vw,48px);line-height:1.2;margin:12px 0 18px;letter-spacing:-.04em}}.lead{{font-size:19px;max-width:920px;color:#3c5058}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}.card{{background:#fff;padding:19px 21px;border:1px solid #dedfd9;border-radius:9px}}.card b{{display:block;font-size:29px;line-height:1.3;letter-spacing:-.03em}}.card span{{color:#5d6c70;font-size:12px}}.box{{background:#e8efed;border-left:4px solid #2e6860;padding:18px 23px}}.box p{{margin:5px 0}}section{{margin:39px 0}}h2{{font-size:26px;line-height:1.3;margin:8px 0 16px;letter-spacing:-.025em}}.tablewrap{{overflow:auto;border:1px solid #dadfdc;border-radius:8px}}table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}th,td{{padding:12px 14px;text-align:right;border-bottom:1px solid #e3e6e2;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th,table>tbody>tr:first-child:has(th){{font-weight:650}}.focus{{background:#eef5f1}}.caption{{font-size:12px;color:#5d696d}}details{{background:white;border:1px solid #daddd7;padding:14px 18px;border-radius:8px;margin:12px 0}}summary{{cursor:pointer;font-weight:600}}.replays{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}iframe{{width:100%;height:540px;border:0;border-radius:8px}}a{{color:#245c65;text-underline-offset:3px}}footer{{font-size:12px;border-top:1px solid #cdd6d1;padding-top:22px;color:#5f7076}}@media(max-width:760px){{main{{padding:24px 15px}}.cards{{grid-template-columns:1fr 1fr}}.replays{{grid-template-columns:1fr}}th,td{{padding:10px}}iframe{{height:580px}}}}
</style></head><body><main><header><div class="kicker">DINO DUEL LAB / V1.3 / COUNTERFACTUAL TEST</div><h1>预测有了，出招还没有更聪明。</h1><p class="lead">同一局面、12个攻击计划、独立随机重复验收。把离线收益、固定等待对照和整局失效一起公开，而不是只展示有利的回放。</p></header><div class="cards"><div class="card"><b>{c["roots"]:,}</b><span>独立源对局快照</span></div><div class="card"><b>{c["branches"]:,}</b><span>实际模拟的候选分支 · 非完整对局</span></div><div class="card"><b>{len(train)}</b><span>实际训练的小预测器</span></div><div class="card"><b>8 秒</b><span>向前预测窗口 · 不是比赛倒计时</span></div></div><div class="box">{"".join(notes)}</div>{"".join(sections)}{online}{cases}{methods}{audit}<section><h2>证据边界</h2><p>这轮只比较固定控制器、固定候选计划和冻结的对手群。没有在线对手适应，没有新增大规模自博弈，也没有证明潜变量规划优于任何普通世界模型。离线选招改善不自动意味着重新规划的整局表现改善。三次预测器拟合共享数据，不能当作三次完整强化学习重复。</p><p>数据生成实际执行 <b>{c["simulated_steps"]:,}</b> 个双人环境步，另有 {c["source_steps"]:,} 个源轨迹步；它们不是同等数量的新增 PPO 训练步。所有旧模型、原始分支、训练日志和恢复状态在完整项目中保留。</p></section><section><h2>原始记录</h2><p>{downloads}</p><p class="caption">本文件完全离线；表格来自保存的结果，回放来自实际模拟。没有外部字体、图片或网络请求。</p></section><footer>Local decision experiment, not a claim of human-like cognition or a general JEPA advantage. Source archive and policy hashes are recorded in PROTOCOL.json.</footer></main></body></html>'''
    (out/'report.html').write_text(doc,encoding='utf-8')
    return out/'report.html'
if __name__=='__main__':print(build(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'counterfactual'))
