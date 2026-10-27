"""Dependency-light, offline HTML report from real JSONL and evaluation files."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


PALETTE = ["#2c6eaa", "#8b53b9", "#d78c2d", "#278c75", "#bd5972", "#6b7280"]
PREFERRED_METRICS = [
    "mean_episode_return", "draw_rate", "episode_length", "policy_loss",
    "value_loss", "entropy", "aux_loss", "prediction_loss", "latent_std",
    "approx_kl", "clip_fraction", "grad_norm",
]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _read_metrics(path: Path, warnings: list[str]) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"Skipped incomplete or malformed metric row: {path.name}, line {i}.")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _svg_plot(metric: str, runs: list[dict]) -> str:
    series = []
    for i, run in enumerate(runs):
        points = []
        for row in run["metrics"]:
            x = row.get("global_step", row.get("update"))
            y = row.get(metric)
            if _finite_number(x) and _finite_number(y):
                points.append((float(x), float(y)))
        if points:
            # Display thinning only. Full-resolution JSONL remains untouched.
            if len(points) > 500:
                n = len(points)
                points = [points[round(k * (n - 1) / 499)] for k in range(500)]
            series.append((run["name"], points, PALETTE[i % len(PALETTE)]))
    if not series:
        return ""
    xs = [x for _, pts, _ in series for x, _ in pts]
    ys = [y for _, pts, _ in series for _, y in pts]
    x0, x1 = min(0.0, min(xs)), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        padding = max(abs(y0) * .1, .1)
        y0 -= padding
        y1 += padding
    else:
        padding = (y1 - y0) * .08
        y0 -= padding
        y1 += padding
    def sx(x):
        return 65 + (x - x0) / (x1 - x0) * 655
    def sy(y):
        return 235 - (y - y0) / (y1 - y0) * 200
    parts = [f'<section class="chart"><h3>{html.escape(metric.replace("_", " "))}</h3>',
             f'<svg viewBox="0 0 750 280" role="img" aria-label="{html.escape(metric)} training curve">']
    for k in range(5):
        value = y0 + (y1 - y0) * k / 4
        y = sy(value)
        parts.append(f'<line x1="65" y1="{y:.2f}" x2="720" y2="{y:.2f}" stroke="#e5ebf1"/>')
        parts.append(f'<text x="56" y="{y + 4:.2f}" text-anchor="end" class="axis">{value:.3g}</text>')
    for k in range(4):
        value = x0 + (x1 - x0) * k / 3
        parts.append(f'<text x="{sx(value):.2f}" y="255" text-anchor="middle" class="axis">{value:.3g}</text>')
    parts.append('<text x="390" y="275" text-anchor="middle" class="axis">joint environment action steps</text>')
    for name, points, color in series:
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.1"/>')
        if len(points) == 1:
            x, y = points[0]
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
    parts.append('</svg><div class="legend">')
    for name, _, color in series:
        parts.append(f'<span><i style="background:{color}"></i>{html.escape(name)}</span>')
    parts.append('</div></section>')
    return "".join(parts)


def _relative_href(path: Path, output_dir: Path) -> str:
    import os
    from urllib.parse import quote
    return quote(os.path.relpath(path, output_dir).replace("\\", "/"), safe="/.")


def make_report(root_dir: str | Path, out_path: str | Path | None = None) -> Path:
    root_dir = Path(root_dir).resolve()
    if not root_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {root_dir}")
    output = Path(out_path) if out_path is not None else root_dir / "report.html"
    output = output.resolve()
    warnings: list[str] = []
    runs = []
    for path in sorted(root_dir.rglob("metrics.jsonl")):
        metrics = _read_metrics(path, warnings)
        if metrics:
            runs.append({"name": str(path.parent.relative_to(root_dir)) or ".",
                         "path": path, "metrics": metrics})
    evaluations = []
    for path in sorted(root_dir.rglob("summary.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            warnings.append(f"Skipped unreadable summary: {path.relative_to(root_dir)}.")
            continue
        if isinstance(value, dict) and value.get("schema") in {"dino_duel.evaluation.v1", "dino_duel.evaluation.v2"}:
            evaluations.append((path, value))
    total_games = sum(x.get("n_games", 0) for _, x in evaluations)
    total_steps = sum(run["metrics"][-1].get("global_step", 0) for run in runs)
    comparisons = []
    comparison_path = root_dir / "comparison.json"
    if comparison_path.is_file():
        try:
            value = json.loads(comparison_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("schema_version") == 1:
                comparisons = value.get("comparisons", [])
        except (json.JSONDecodeError, OSError):
            warnings.append("Skipped unreadable comparison.json.")
    parts = [_HEADER,
             '<div class="eyebrow">DINO DUEL LAB / EXPERIMENT REPORT</div>',
             '<h1>Training and evaluation</h1>',
             '<p class="lead">A report of recorded runs. No synthetic scores or inferred victories are added.</p>',
             '<div class="stats">',
             f'<div><strong>{len(runs)}</strong><span>training runs</span></div>',
             f'<div><strong>{total_steps:,}</strong><span>joint environment steps</span></div>',
             f'<div><strong>{total_games:,}</strong><span>evaluation games</span></div>',
             f'<div><strong>{len(evaluations)}</strong><span>matchups</span></div></div>',
             '<section class="notice"><strong>Interpretation boundary</strong><p>Scores are wins + ½ draws, averaged over held-out environment-seed pairs with both seat assignments. Per-checkpoint 95% percentile bootstrap intervals resample complete pairs; they exclude uncertainty from training seeds. The training-seed comparison section, when present, separately reports hierarchical resampling. Small smoke runs verify software behavior and cannot establish which method is better.</p><p>The main benchmark uses structured simulator observations and an identical basic runner for both treatments unless the configuration explicitly enables motor learning. A JEPA-style auxiliary objective does not by itself imply model-based planning or reproduce an official I-JEPA/V-JEPA model.</p></section>']
    if comparisons:
        parts.append('<h2>Comparison across training seeds</h2><p class="note">Each training seed contributes one score averaged over complete environment-seed pairs. Hierarchical intervals resample training seeds, then environment-seed pairs within those seeds. These are distinct from the per-checkpoint intervals below.</p><div class="scroll"><table><thead><tr><th>Comparison</th><th>Training seeds</th><th>Pairs per seed</th><th>Mean left score</th><th>Hierarchical 95% interval</th><th>Scores by training seed</th></tr></thead><tbody>')
        for comparison in comparisons:
            ci = comparison.get("hierarchical_ci95")
            interval = f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else 'unavailable · exploratory'
            seed_scores = list(zip(comparison.get("training_seeds", []), comparison.get("per_training_seed_scores", [])))
            label = f'{comparison.get("left", "left")} vs {comparison.get("right", "right")}'
            scores = ", ".join(f"{seed}: {score:.3f}" for seed, score in seed_scores)
            parts.append('<tr>' + f'<td>{html.escape(label)}<small>{html.escape(str(comparison.get("note", "")))}</small></td>' +
                         f'<td>{_fmt(comparison.get("n_training_seeds"))}</td>' +
                         f'<td>{_fmt(comparison.get("n_eval_seed_pairs_per_training_seed"))}</td>' +
                         f'<td>{_fmt(comparison.get("mean_score"))}</td><td>{interval}</td><td>{html.escape(scores)}</td></tr>')
        parts.append(f'</tbody></table></div><p class="note"><a href="{_relative_href(comparison_path, output.parent)}">Read comparison.json</a>. Few training seeds give weak information about training variability even when many evaluation games are used.</p>')
    parts.append('<h2>Frozen-policy matchups</h2>')
    if evaluations:
        parts.append('<div class="scroll"><table><thead><tr><th>Matchup</th><th>Pairs / games</th><th>Left score</th><th>95% interval</th><th>W / D / L</th><th>Draw reasons</th><th>Files</th></tr></thead><tbody>')
        for path, summary in evaluations:
            label = f'{summary.get("left_name", "left")} vs {summary.get("right_name", "right")}'
            ci = summary.get("left_score_ci95")
            interval = f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else 'unavailable (one pair)'
            reasons = ", ".join(f"{key}: {value}" for key, value in summary.get("draw_reasons", {}).items()) or "—"
            links = [f'<a href="{_relative_href(path, output.parent)}">JSON</a>']
            for fname, caption in [("episodes.csv", "CSV"), ("replay.html", "Replay")]:
                candidate = path.parent / fname
                if candidate.is_file():
                    links.append(f'<a href="{_relative_href(candidate, output.parent)}">{caption}</a>')
            parts.append('<tr>' +
                         f'<td>{html.escape(label)}<small>{html.escape(str(path.parent.relative_to(root_dir)))}</small></td>' +
                         f'<td>{summary.get("n_seed_pairs", 0)} / {summary.get("n_games", 0)}</td>' +
                         f'<td>{_fmt(summary.get("left_mean_score"))}</td><td>{interval}</td>' +
                         f'<td>{summary.get("wins", 0)} / {summary.get("draws", 0)} / {summary.get("losses", 0)}</td>' +
                         f'<td>{html.escape(reasons)}</td><td>{" · ".join(links)}</td></tr>')
        parts.append('</tbody></table></div>')
        if any(x.get("small_sample_warning", False) for _, x in evaluations):
            parts.append('<p class="note">At least one matchup has fewer than 20 seed pairs. Treat its interval as descriptive and potentially unstable; even a zero-width empirical interval does not establish zero uncertainty.</p>')
    else:
        parts.append('<p class="empty">No completed evaluation summaries found yet. Run the evaluation or experiment command to populate this section.</p>')
    parts.append('<h2>Training diagnostics</h2><p class="note">Curves show logged updates, with at most 500 displayed points per series. Complete values remain in metrics.jsonl. Self-play return and seat-0 win rate are diagnostics, not a frozen-opponent strength measurement.</p>')
    if runs:
        parts.append('<div class="plots">')
        for metric in PREFERRED_METRICS:
            parts.append(_svg_plot(metric, runs))
        parts.append('</div><h2>Run accounting</h2><div class="scroll"><table><thead><tr><th>Run</th><th>Update</th><th>Joint steps</th><th>Agent steps</th><th>Seconds</th><th>Total / acting parameters</th><th>Log</th></tr></thead><tbody>')
        for run in runs:
            last = run["metrics"][-1]
            parts.append('<tr>' + f'<td>{html.escape(run["name"])}</td>' +
                         ''.join(f'<td>{_fmt(last.get(key))}</td>' for key in ["update", "global_step", "agent_steps", "elapsed_seconds"]) +
                         f'<td>{_fmt(last.get("parameter_count"))} / {_fmt(last.get("acting_parameter_count"))}</td>' +
                         f'<td><a href="{_relative_href(run["path"], output.parent)}">JSONL</a></td></tr>')
        parts.append('</tbody></table></div>')
    else:
        parts.append('<p class="empty">No training logs found. Evaluation of built-in policies can still be reported above.</p>')
    if warnings:
        parts.append('<h2>Read warnings</h2><ul>' + ''.join(f'<li>{html.escape(w)}</li>' for w in warnings) + '</ul>')
    parts.append('<h2>Before drawing a research conclusion</h2><ul><li>Use multiple independent training seeds, and keep evaluation seeds separate from tuning.</li><li>Compare identical observation access, runner initialization, reward, environment interaction budget, and opponent schedule.</li><li>Report elapsed training time and parameter counts as well as samples; predictive auxiliary losses require extra computation.</li><li>Include the shuffled-target control, fixed opponents, and cross-play. High draw rates or a weak motor controller can obscure strategic differences.</li><li>Inspect latent variance and held-out prediction error before attributing a change to useful prediction.</li></ul><footer>Generated entirely from local files. This report and its charts work offline; linked logs and replays should be kept alongside it.</footer></main></body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(parts), encoding="utf-8")
    temporary.replace(output)
    return output


_HEADER = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dino Duel Lab — Experiment report</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f8fb;color:#1d2b3d;font:14px/1.6 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:42px 26px}.eyebrow{font-size:11px;letter-spacing:.18em;font-weight:700;color:#2b7b68}h1{font-size:38px;letter-spacing:-.04em;margin:7px 0}h2{font-size:22px;letter-spacing:-.02em;margin:34px 0 12px}h3{font-size:14px;text-transform:capitalize;margin:0 0 10px}.lead,.note{color:#627185}.lead{font-size:16px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.stats>div,.chart{background:white;border:1px solid #dfe6ee;border-radius:12px;padding:20px}.stats strong{display:block;font-size:28px;font-weight:650}.stats span{font-size:12px;color:#66758a}.notice{background:#eaf2f1;border-left:3px solid #318c76;padding:17px 21px;border-radius:0 10px 10px 0}.notice p{margin:7px 0;font-size:13px}.scroll{overflow-x:auto;border:1px solid #dfe6ee;border-radius:10px;background:white}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;padding:12px 14px;border-bottom:1px solid #e8edf3;vertical-align:top}th{background:#edf2f7;font-size:11px;color:#596a80;font-weight:600;white-space:nowrap}td small{display:block;color:#8190a3;font-size:10px}a{color:#2c6eaa;text-decoration:none}a:hover{text-decoration:underline}.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.chart{padding:17px 14px}.chart svg{display:block;width:100%;height:auto}.axis{font:11px system-ui;fill:#79889a}.legend{display:flex;gap:6px 12px;flex-wrap:wrap;font-size:10px;color:#66758a}.legend i{display:inline-block;width:13px;height:3px;margin:0 5px 3px 0}.empty{padding:20px;background:white;border:1px dashed #ccd6e2;border-radius:9px;color:#79889a}.note{font-size:12px}li{margin:7px 0}footer{margin-top:34px;padding-top:15px;border-top:1px solid #dfe6ee;color:#7b899b;font-size:11px}@media(max-width:760px){main{padding:25px 14px}h1{font-size:29px}.stats{grid-template-columns:repeat(2,1fr)}.plots{grid-template-columns:1fr}}
</style></head><body><main>'''
