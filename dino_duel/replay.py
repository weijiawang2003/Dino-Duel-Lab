"""Self-contained HTML spectator for recorded, actually simulated duels."""

from __future__ import annotations

import html
import json
from pathlib import Path


def write_replay(frames: list[dict], episode: dict, out_path: str | Path) -> Path:
    if not frames:
        raise ValueError("A replay needs at least one frame")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Protect a standalone script data block from policy names containing HTML.
    payload = json.dumps({"frames": frames, "episode": episode}, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    title = html.escape(f"{episode.get('left_name', 'Left')} vs {episode.get('right_name', 'Right')}")
    rendered = _HTML.replace("__TITLE__", title).replace("__DATA__", payload)
    temporary = out_path.with_name(out_path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(out_path)
    return out_path


def make_replay(left, right, env_config, seed: int, out_path: str | Path,
                max_frames: int = 3000, safety_seconds: float = 300.0) -> dict:
    from .evaluation import play_episode
    episode, frames = play_episode(left, right, env_config, seed,
                                   record=True, max_frames=max_frames, safety_seconds=safety_seconds)
    path = write_replay(frames, episode, out_path)
    return {"path": str(path), "frames": len(frames), "episode": episode}


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dino Duel — __TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0c121b;--card:#131d2a;--line:#273447;--ink:#e9f0f8;--muted:#98a9bf;--a:#5de1c4;--b:#ffb37a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:32px 22px 44px}header{display:flex;justify-content:space-between;align-items:center;gap:20px}
.eyebrow{font-size:11px;letter-spacing:.2em;color:var(--a);font-weight:700}h1{font-size:clamp(24px,4vw,38px);letter-spacing:-.04em;margin:6px 0}
.muted{color:var(--muted)}.badge{border:1px solid var(--line);padding:7px 12px;border-radius:30px;font-size:12px;white-space:nowrap}
.players{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:24px 0 14px}.player{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--a);padding:16px 20px;border-radius:12px}.player.b{border-top-color:var(--b)}
.name{font-size:17px;font-weight:650;word-break:break-word}.stats{display:flex;flex-wrap:wrap;gap:18px;margin-top:10px}.stats span{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}.stats strong{display:block;color:var(--ink);font-size:20px;letter-spacing:0;font-weight:550}
.stage{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#101a27;position:relative}canvas{display:block;width:100%;height:auto;aspect-ratio:1100/420}
.controls{padding:16px;background:var(--card);border-top:1px solid var(--line)}.row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.timeline{width:100%;accent-color:var(--a);margin:0 0 13px}button,select{background:#202e40;color:var(--ink);border:1px solid #3c4b60;border-radius:7px;padding:8px 13px;font:inherit;cursor:pointer}button:hover{background:#2b3c52}button.primary{background:var(--a);color:#0b211d;font-weight:700;border-color:var(--a)}.spacer{flex:1}#clock{font-variant-numeric:tabular-nums}.notes{display:grid;grid-template-columns:1.2fr 1fr;gap:24px;margin-top:22px}.notes h2{font-size:14px;margin:0 0 8px}.notes p{font-size:13px;color:var(--muted);margin:0}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:12px}.dot{display:inline-block;width:8px;height:8px;margin-right:5px;border-radius:50%}.result{font-size:13px;color:var(--muted);max-width:600px}footer{margin-top:28px;font-size:11px;color:#6f829b}
@media(max-width:640px){main{padding:20px 12px}header{align-items:flex-start}.players{gap:8px}.player{padding:12px}.stats{gap:10px}.stats strong{font-size:17px}.notes{grid-template-columns:1fr}.badge{display:none}.row{gap:8px}button,select{padding:7px 10px}}
</style></head><body><main>
<header><div><div class="eyebrow">DINO DUEL LAB / SPECTATOR</div><h1>Two tracks. One duel.</h1><div class="muted">Recorded simulation · fully offline replay</div></div><div class="badge" id="seedBadge"></div></header>
<section class="players"><div class="player"><div class="name" id="name0"></div><div class="stats"><span>Distance<strong id="dist0">0 m</strong></span><span>Wallet<strong id="coins0">0</strong></span><span>Action<strong id="action0">Run</strong></span></div></div><div class="player b"><div class="name" id="name1"></div><div class="stats"><span>Distance<strong id="dist1">0 m</strong></span><span>Wallet<strong id="coins1">0</strong></span><span>Action<strong id="action1">Run</strong></span></div></div></section>
<div class="stage"><canvas id="scene" width="1100" height="420" aria-label="Two Dino racing tracks"></canvas><div class="controls"><input id="timeline" class="timeline" type="range" min="0" value="0" step="1" aria-label="Replay timeline"><div class="row"><button id="play" class="primary">Play</button><button id="restart">Restart</button><select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select><span id="clock" class="muted"></span><div class="spacer"></div><span class="result" id="status"></span></div></div></div>
<div class="legend"><span><i class="dot" style="background:#5de1c4"></i>Seat 0</span><span><i class="dot" style="background:#ffb37a"></i>Seat 1</span><span><i class="dot" style="background:#8ca1bb"></i>Natural obstacle</span><span><i class="dot" style="background:#f38fa5"></i>Purchased obstacle</span></div>
<div class="notes"><div><h2>What you are watching</h2><p>The distance clock earns currency. A legal purchase sends an obstacle to the other track, subject to reaction time, spacing, and cooldown. This replay contains recorded states from the actual simulator; the browser only renders them.</p></div><div><h2>Reading the result</h2><p id="resultNote"></p></div></div><footer>Original Dino Duel Lab simulator · structured-state experiment · no network requests, external assets, or live training in this file. Space toggles playback; arrow keys step frames.</footer>
</main><script id="replay-data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('replay-data').textContent), frames=data.frames, episode=data.episode;
const canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),slider=document.getElementById('timeline');
let index=0,playing=false,playTime=Number(frames[0].time||0),lastWall=null;
const colors=['#5de1c4','#ffb37a'], names=episode.left_seat===1?[episode.right_name,episode.left_name]:[episode.left_name,episode.right_name];
document.getElementById('name0').textContent='01 / '+names[0];document.getElementById('name1').textContent='02 / '+names[1];
document.getElementById('seedBadge').textContent='SEED '+episode.seed+' · '+frames.length+' FRAMES';slider.max=frames.length-1;
const finishTime=Number(frames[frames.length-1].time||0);
document.getElementById('resultNote').textContent=(episode.censored?'UNRESOLVED — observation budget exhausted':(episode.winner_policy==='draw'?'Draw':(episode.winner_policy==='left'?episode.left_name:episode.right_name)+' won'))+' — '+episode.reason+'. Duration '+Number(episode.seconds||finishTime).toFixed(2)+' s. A single replay illustrates behavior; use paired multi-seed evaluation for comparisons.';
function rect(x,y,w,h,color){ctx.fillStyle=color;ctx.fillRect(x,y,w,h)}
function dino(x,ground,p,scale,color,t){
 const y=ground-Number(p.y||0)*scale,duck=Boolean(p.duck),w=Number(p.width||.7)*scale,h=Number(p.height||(duck?.55:1.25))*scale;
 ctx.save();ctx.globalAlpha=p.alive===false?.42:1;ctx.fillStyle=color;
 if(duck){rect(x-3,y-h,w+15,h-2,color);rect(x+w+7,y-h-5,14,h*.7,color)}else{rect(x,y-h*.75,w*.8,h*.72,color);rect(x+w*.34,y-h,w*.86,h*.45,color);rect(x-6,y-h*.48,10,h*.19,color);rect(x+w*.7,y-h*.45,10,4,color)}
 rect(x+w*.87,y-h+(duck?3:5),3,3,'#0b1923');
 if(!p.y){const phase=Math.floor(t*12)%2;rect(x+3,y-2,5,phase?7:4,color);rect(x+w*.55,y-2,5,phase?4:7,color)}
 ctx.restore();
}
function obstacle(o,ground,x0,scale){
 const x=x0+Number(o.x)*scale,y=ground-(Number(o.y||0)+Number(o.height))*scale,w=Number(o.width)*scale,h=Number(o.height)*scale;
 if(x+w<0||x>1100)return;const purchased=!(o.source===undefined||o.source===null||o.source==='natural'||o.source===-1);const color=purchased?'#f38fa5':'#8ca1bb';
 const bird=String(o.kind).toLowerCase().includes('bird')||Number(o.y)>0;
 if(bird){rect(x+w*.2,y+h*.3,w*.65,h*.5,color);rect(x+w*.7,y,w*.3,h*.55,color);ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x+w*.1,y+h*.5);ctx.lineTo(x-w*.2,y);ctx.lineTo(x+w*.45,y+h*.3);ctx.fill();rect(x+w*.84,y+h*.18,2,2,'#101a27')}
 else{rect(x+w*.28,y,w*.42,h,color);rect(x,y+h*.35,w*.32,h*.17,color);rect(x,y+h*.15,w*.15,h*.35,color);rect(x+w*.65,y+h*.52,w*.35,h*.16,color);rect(x+w*.84,y+h*.3,w*.16,h*.36,color)}
}
function draw(){
 const f=frames[index],time=Number(f.time||0),horizon=Number(f.horizon||40),scale=Math.min(34,950/horizon),x0=100;
 ctx.clearRect(0,0,1100,420);ctx.fillStyle='#101a27';ctx.fillRect(0,0,1100,420);
 ctx.strokeStyle='#24354b';ctx.beginPath();ctx.moveTo(0,210);ctx.lineTo(1100,210);ctx.stroke();
 for(let s=0;s<2;s++){
  const ground=166+s*210,p=f.players[s],dist=Array.isArray(f.distance)?Number(f.distance[s]||0):Number(f.distance||0);
  ctx.fillStyle='#7e91aa';ctx.font='11px system-ui';ctx.fillText('TRACK '+(s+1),22,30+s*210);
  ctx.strokeStyle='#35465d';ctx.beginPath();ctx.moveTo(0,ground+7);ctx.lineTo(1100,ground+7);ctx.stroke();
  for(let k=0;k<14;k++){let xx=((k*96-dist*scale)%1344+1344)%1344-96;rect(xx,ground+13+(k%3)*4,12+(k%4)*6,1,'#29394d')}
  for(const o of (f.tracks[s]||[]))obstacle(o,ground,x0,scale);
  dino(x0,ground,p,scale,colors[s],time);
  document.getElementById('dist'+s).textContent=dist.toFixed(1)+' m';document.getElementById('coins'+s).textContent=Number(p.coins||0).toFixed(1);
  const action=f.actions?f.actions[s]:[0,0],motors=['Run','Jump','Duck'],attacks=['','Cactus','Wide cactus','Low bird','High bird'];
  document.getElementById('action'+s).textContent=p.alive===false?'Out':(action[1]?'Buy '+attacks[action[1]]:motors[action[0]]);
  if(p.alive===false){ctx.fillStyle=colors[s];ctx.font='700 18px system-ui';ctx.fillText('COLLISION',145,75+s*210)}
 }
 ctx.fillStyle='#899bb2';ctx.font='12px system-ui';ctx.fillText('Speed '+Number(f.speed||0).toFixed(1)+' m/s',930,28);
 slider.value=index;document.getElementById('clock').textContent=time.toFixed(2)+' / '+finishTime.toFixed(2)+' s';
 document.getElementById('status').textContent=index===frames.length-1?(episode.censored?'Unresolved · observation budget':episode.winner_policy==='draw'?'Draw · '+episode.reason:(episode.winner_policy==='left'?episode.left_name:episode.right_name)+' wins'):'Recorded match';
 document.getElementById('play').textContent=playing?'Pause':'Play';
}
function setIndex(n){index=Math.max(0,Math.min(frames.length-1,n));playTime=Number(frames[index].time||0);draw()}
function toggle(){if(index===frames.length-1&&!playing)setIndex(0);playing=!playing;lastWall=null;draw()}
document.getElementById('play').onclick=toggle;document.getElementById('restart').onclick=()=>{playing=false;setIndex(0)};
slider.oninput=()=>{playing=false;setIndex(Number(slider.value))};
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT'||e.target.tagName==='INPUT')return;if(e.code==='Space'){e.preventDefault();toggle()}if(e.code==='ArrowRight'){playing=false;setIndex(index+1)}if(e.code==='ArrowLeft'){playing=false;setIndex(index-1)}});
function tick(now){
 if(playing&&lastWall!==null){playTime+=(now-lastWall)/1000*Number(document.getElementById('speed').value);while(index<frames.length-1&&Number(frames[index+1].time||0)<=playTime)index++;if(index===frames.length-1)playing=false;draw()}
 lastWall=now;requestAnimationFrame(tick)
}
draw();requestAnimationFrame(tick);
</script></body></html>'''
