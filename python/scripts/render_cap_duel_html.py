"""Render a cap_duel inspector JSON episode as a self-contained HTML viewer.

The standard xushi2 viewer cannot faithfully render cap_duel replays because
cap_duel is a Python-only mini-game with its own world model (spawns near
origin within ``point_radius``, scores in Python, tracks alive/HP in Python).
The C++ Phase 4 sim that the viewer reconstructs from the action stream
has different spawn positions, a different objective location, and does not
honor the cap_duel scoring rule.

This script reads an inspector JSON produced by
``scripts/inspect_cap_duel_rollout.py``, picks one episode, and emits a
self-contained HTML file with an embedded canvas animation that shows the
*actual* cap_duel world: origin-centered objective, both active agents
within ``point_radius`` of origin, score/kill/hit events as they happened
in the Python env.

Open the output ``.html`` file in any browser; no server needed.

Usage:
    py -3.13 scripts/render_cap_duel_html.py \\
        --input runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_greedy.json \\
        --episode 1 \\
        --output runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/view_greedy_ep1.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_POINT_RADIUS = 0.18

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>cap_duel viewer — {title}</title>
<style>
  body {{ background: #0a0c12; color: #d8dde6; font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; padding: 24px; }}
  h1 {{ font-size: 16px; margin: 0 0 12px; font-weight: 600; letter-spacing: 0.02em; }}
  .container {{ display: flex; gap: 24px; align-items: flex-start; }}
  canvas {{ background: #111722; border: 1px solid #222a3a; border-radius: 8px; }}
  .panel {{ min-width: 320px; font-size: 13px; line-height: 1.55; }}
  .panel .kv {{ display: grid; grid-template-columns: 12em 1fr; column-gap: 8px; }}
  .panel .kv b {{ color: #9fb6ff; font-weight: 500; }}
  .panel .section {{ margin-top: 16px; padding-top: 12px; border-top: 1px solid #222a3a; }}
  .controls {{ margin: 12px 0; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
  button {{ background: #1a2334; color: #d8dde6; border: 1px solid #2a3550; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px; }}
  button:hover {{ background: #243154; }}
  input[type=range] {{ flex: 1; min-width: 200px; }}
  .legend span {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: middle; margin-right: 6px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; letter-spacing: 0.04em; }}
  .badge.win {{ background: #2a4d2a; color: #c4f5b6; }}
  .badge.loss {{ background: #4d2a2a; color: #f5b6b6; }}
  .badge.draw {{ background: #2a2a4d; color: #b6b6f5; }}
  .fire {{ color: #ff8b6b; }}
  .hit  {{ color: #ffd06b; }}
  .kill {{ color: #ff5b5b; font-weight: 700; }}
  .score {{ color: #6bff8b; font-weight: 700; }}
  .selfdied {{ color: #6b8bff; font-weight: 700; }}
  .event-log {{ max-height: 200px; overflow-y: auto; font-family: ui-monospace, monospace; font-size: 11px; background: #0d1320; padding: 8px; border-radius: 6px; border: 1px solid #1a2334; }}
  .event-log div {{ padding: 1px 0; }}
</style>
</head>
<body>
<h1>cap_duel viewer — {title} <span class="badge {winner_cls}">{winner_label}</span></h1>
<div class="container">
  <div>
    <canvas id="cv" width="560" height="560"></canvas>
    <div class="controls">
      <button id="play">Play</button>
      <button id="step-b">&#x25C0; step</button>
      <button id="step-f">step &#x25B6;</button>
      <button id="restart">restart</button>
      <label>speed
        <select id="speed">
          <option value="2">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="0.5">2x</option>
          <option value="0.25">4x</option>
          <option value="0.1">10x</option>
        </select>
      </label>
    </div>
    <input type="range" id="scrub" min="0" value="0" step="1" />
  </div>
  <div class="panel">
    <div class="kv">
      <b>checkpoint</b><span>{checkpoint}</span>
      <b>mode</b><span>{mode}</span>
      <b>seed</b><span>{seed}</span>
      <b>match_type</b><span>{match_type}</span>
      <b>point_radius</b><span>{point_radius}</span>
      <b>enemy_recontest_delay</b><span>{recontest_delay}</span>
      <b>score_ticks_to_clear</b><span>{score_to_clear}</span>
      <b>episode steps</b><span>{n_steps}</span>
    </div>
    <div class="section">
      <div class="kv">
        <b>step</b><span id="p-step">0</span>
        <b>score A / B</b><span id="p-score">0 / 0</span>
        <b>self_on_point</b><span id="p-sop">-</span>
        <b>enemy_on_point</b><span id="p-eop">-</span>
        <b>enemy_alive</b><span id="p-ea">-</span>
        <b>enemy_off_point_dec</b><span id="p-eopd">-</span>
        <b>self_score_ready (next)</b><span id="p-ssr">-</span>
        <b>A status</b><span id="p-astatus">-</span>
        <b>self HP / enemy HP</b><span id="p-hp">-</span>
        <b>kills A / hits A / fires A</b><span id="p-counts">-</span>
        <b>action (fire/aim)</b><span id="p-act">-</span>
      </div>
    </div>
    <div class="section">
      <div class="legend">
        <span style="background:#ff6b6b"></span> Team A (learner)
        &nbsp; <span style="background:#6b9bff"></span> Team B
        &nbsp; <span style="background:#444; border:1px dashed #888"></span> point
      </div>
    </div>
    <div class="section">
      <b>events</b>
      <div id="log" class="event-log"></div>
    </div>
  </div>
</div>
<script>
const DATA = {data_json};
DATA.recontest_delay = {recontest_delay_js};
const POINT_RADIUS = {point_radius};
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;
const PAD = 30;
// World coords are in [-1, 1]. Map to canvas with padding.
function wx(x) {{ return PAD + (x + 1) * 0.5 * (W - 2 * PAD); }}
function wy(y) {{ return H - (PAD + (y + 1) * 0.5 * (H - 2 * PAD)); }}
function rad(r) {{ return r * 0.5 * (W - 2 * PAD); }}

function draw(i) {{
  ctx.fillStyle = '#111722'; ctx.fillRect(0, 0, W, H);
  // grid
  ctx.strokeStyle = '#1a2334'; ctx.lineWidth = 1;
  for (let k = -1; k <= 1; k += 0.25) {{
    ctx.beginPath(); ctx.moveTo(wx(k), wy(-1)); ctx.lineTo(wx(k), wy(1)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(wx(-1), wy(k)); ctx.lineTo(wx(1), wy(k)); ctx.stroke();
  }}
  // axes
  ctx.strokeStyle = '#2a3550';
  ctx.beginPath(); ctx.moveTo(wx(0), wy(-1)); ctx.lineTo(wx(0), wy(1)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(wx(-1), wy(0)); ctx.lineTo(wx(1), wy(0)); ctx.stroke();
  // point circle
  ctx.strokeStyle = '#88aaff'; ctx.setLineDash([6, 4]); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(wx(0), wy(0), rad(POINT_RADIUS), 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);
  // trails — skip ticks where the agent was dead
  const trail = 8;
  const start = Math.max(0, i - trail);
  for (let j = start; j <= i; j++) {{
    const e = DATA.per_tick[j];
    const a = (j - start) / Math.max(1, trail);
    if (e.self_hp > 0) {{
      ctx.fillStyle = `rgba(255, 107, 107, ${{a * 0.5}})`;
      ctx.beginPath(); ctx.arc(wx(e.self_pos[0]), wy(e.self_pos[1]), 3, 0, Math.PI * 2); ctx.fill();
    }}
    if (e.enemy_alive) {{
      ctx.fillStyle = `rgba(107, 155, 255, ${{a * 0.5}})`;
      ctx.beginPath(); ctx.arc(wx(e.enemy_pos[0]), wy(e.enemy_pos[1]), 3, 0, Math.PI * 2); ctx.fill();
    }}
  }}
  const e = DATA.per_tick[i];
  // self — render as gray X when dead, solid red circle when alive
  if (e.self_hp > 0) {{
    ctx.fillStyle = '#ff6b6b';
    ctx.beginPath(); ctx.arc(wx(e.self_pos[0]), wy(e.self_pos[1]), 9, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('A', wx(e.self_pos[0]), wy(e.self_pos[1]) + 4);
  }} else {{
    ctx.strokeStyle = '#774444'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(wx(e.self_pos[0]), wy(e.self_pos[1]), 9, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#aa6666'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('x', wx(e.self_pos[0]), wy(e.self_pos[1]) + 4);
  }}
  // enemy
  if (e.enemy_alive) {{
    ctx.fillStyle = '#6b9bff';
    ctx.beginPath(); ctx.arc(wx(e.enemy_pos[0]), wy(e.enemy_pos[1]), 9, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText('B', wx(e.enemy_pos[0]), wy(e.enemy_pos[1]) + 4);
  }} else {{
    ctx.strokeStyle = '#555'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(wx(e.enemy_pos[0]), wy(e.enemy_pos[1]), 9, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#777'; ctx.fillText('x', wx(e.enemy_pos[0]), wy(e.enemy_pos[1]) + 4);
  }}
  // fire flash on A
  if (e.action[3] >= 0.5) {{
    ctx.strokeStyle = 'rgba(255, 139, 107, 0.9)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(wx(e.self_pos[0]), wy(e.self_pos[1]), 14, 0, Math.PI * 2); ctx.stroke();
  }}
  // Persistent displacement arrows and banners — look back up to LINGER frames.
  const LINGER = 6;
  function eventAt(j) {{
    const ev = {{ kill: false, ekill: false, pushed: 0, respawn: false, score: false }};
    if (j < 0 || j >= DATA.per_tick.length) return ev;
    const cur = DATA.per_tick[j];
    ev.kill = cur.kill_this_step;
    ev.ekill = cur.enemy_killed_self_this_step;
    ev.score = cur.score_event_this_step;
    if (j > 0) {{
      const pr = DATA.per_tick[j - 1];
      const drop = pr.self_hp - cur.self_hp;
      if (drop > 0 && !cur.enemy_killed_self_this_step) ev.pushed = drop;
      if (pr.self_hp <= 0 && cur.self_hp > 0) ev.respawn = true;
    }}
    return ev;
  }}
  // Draw lingering arrows for any push/respawn within the LINGER window.
  for (let back = 0; back <= LINGER; back++) {{
    const j = i - back;
    if (j <= 0) break;
    const ev = eventAt(j);
    const prev = DATA.per_tick[j - 1];
    const cur = DATA.per_tick[j];
    const alpha = 1.0 - (back / (LINGER + 1));
    if (ev.respawn) {{
      ctx.strokeStyle = `rgba(140, 220, 255, ${{alpha * 0.85}})`; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(wx(prev.self_pos[0]), wy(prev.self_pos[1]));
      ctx.lineTo(wx(cur.self_pos[0]), wy(cur.self_pos[1])); ctx.stroke();
      ctx.setLineDash([]);
    }}
    if (ev.pushed > 0) {{
      ctx.strokeStyle = `rgba(255, 200, 80, ${{alpha * 0.85}})`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(wx(prev.self_pos[0]), wy(prev.self_pos[1]));
      ctx.lineTo(wx(cur.self_pos[0]), wy(cur.self_pos[1])); ctx.stroke();
    }}
  }}
  // Lingering banners.
  ctx.font = 'bold 18px sans-serif'; ctx.textAlign = 'left';
  let banner_y = 30;
  function drawBanner(text, baseColor, alpha) {{
    ctx.fillStyle = baseColor.replace('ALPHA', alpha.toFixed(2));
    ctx.fillText(text, 18, banner_y); banner_y += 24;
  }}
  for (let back = 0; back <= LINGER; back++) {{
    const j = i - back;
    if (j < 0) break;
    const ev = eventAt(j);
    const alpha = 1.0 - (back / (LINGER + 1));
    if (ev.kill)    drawBanner('KILL (A killed B)',                'rgba(255, 91, 91, ALPHA)',  alpha);
    if (ev.ekill)   drawBanner('A WAS KILLED',                     'rgba(107, 139, 255, ALPHA)', alpha);
    if (ev.pushed)  drawBanner(`A HIT BY B (pushed, hp -${{ev.pushed}})`, 'rgba(255, 200, 80, ALPHA)', alpha);
    if (ev.respawn) drawBanner('A RESPAWN',                        'rgba(140, 220, 255, ALPHA)', alpha);
    if (ev.score)   drawBanner('A SCORES',                         'rgba(107, 255, 139, ALPHA)', alpha);
  }}
  // step indicator
  ctx.fillStyle = '#9aa5b8'; ctx.font = '12px ui-monospace, monospace'; ctx.textAlign = 'right';
  ctx.fillText(`step ${{i + 1}} / ${{DATA.per_tick.length}}`, W - 18, 24);
}}

function updatePanel(i) {{
  const e = DATA.per_tick[i];
  document.getElementById('p-step').innerText = i + 1;
  document.getElementById('p-score').innerText =
    `${{e.cap_duel_score_ticks_total}} / ${{e.cap_duel_enemy_score_ticks_total}}`;
  document.getElementById('p-sop').innerText = e.self_on_point ? 'YES' : 'no';
  document.getElementById('p-eop').innerText = e.enemy_on_point ? 'YES' : 'no';
  document.getElementById('p-ea').innerText = e.enemy_alive ? 'YES' : 'no';
  document.getElementById('p-eopd').innerText = e.enemy_off_point_decisions;
  document.getElementById('p-ssr').innerText = e.self_score_ready_after_step ? 'YES' : 'no';
  // A status: alive or dead with countdown to respawn (recontest_delay + 1 ticks total).
  let astatus = 'ALIVE';
  if (e.self_hp <= 0) {{
    // Count back to find the death tick.
    let deathStep = -1;
    for (let k = i; k >= 0; k--) {{ if (DATA.per_tick[k].self_hp <= 0) {{ deathStep = k; }} else {{ break; }} }}
    const RESPAWN_AT = (DATA.recontest_delay !== undefined ? DATA.recontest_delay : 12) + 1;
    const elapsed = (deathStep >= 0 ? (i - deathStep + 1) : 0);
    const remaining = Math.max(0, RESPAWN_AT - elapsed);
    astatus = `DEAD (respawn in ${{remaining}} ticks)`;
  }}
  document.getElementById('p-astatus').innerText = astatus;
  document.getElementById('p-hp').innerText = `${{e.self_hp}} / ${{e.enemy_hp}}`;
  document.getElementById('p-counts').innerText =
    `${{e.cap_duel_kills_total}} / ${{e.cap_duel_hits_total}} / ${{e.cap_duel_fires_total}}`;
  document.getElementById('p-act').innerText =
    `fire=${{e.action[3].toFixed(2)}} aim=${{e.action[2].toFixed(2)}} mv=(${{e.action[0].toFixed(2)}}, ${{e.action[1].toFixed(2)}})`;
}}

function buildLog() {{
  const log = document.getElementById('log');
  log.innerHTML = '';
  DATA.per_tick.forEach((e, i) => {{
    const tags = [];
    if (e.action[3] >= 0.5) tags.push('<span class="fire">fire</span>');
    if (e.hit_this_step) tags.push('<span class="hit">HIT</span>');
    if (e.kill_this_step) tags.push('<span class="kill">KILL</span>');
    if (e.enemy_killed_self_this_step) tags.push('<span class="selfdied">A-DIED</span>');
    if (i > 0) {{
      const prev = DATA.per_tick[i - 1];
      const hp_drop = prev.self_hp - e.self_hp;
      if (hp_drop > 0 && !e.enemy_killed_self_this_step) {{
        tags.push(`<span class="hit">PUSHED(hp-${{hp_drop}})</span>`);
      }}
      if (prev.self_hp <= 0 && e.self_hp > 0) {{
        tags.push('<span class="score">RESPAWN</span>');
      }}
    }}
    if (e.score_event_this_step) tags.push('<span class="score">SCORE</span>');
    if (tags.length === 0 && !e.self_on_point && !e.enemy_on_point) return;
    const dot = e.self_on_point ? '●' : '○';
    const dot2 = e.enemy_on_point ? '●' : '○';
    const row = document.createElement('div');
    row.innerHTML = `step ${{String(i + 1).padStart(3, ' ')}} A${{dot}} B${{dot2}}  ${{tags.join(' ')}}`;
    row.dataset.step = i;
    row.style.cursor = 'pointer';
    row.onclick = () => {{ paused = true; document.getElementById('play').innerText = 'Play'; idx = i; render(); }};
    log.appendChild(row);
  }});
}}

let idx = 0;
let paused = true;
let speed = 1.0;  // seconds-per-step multiplier; 1.0 = 0.15s/step

function render() {{
  draw(idx);
  updatePanel(idx);
  document.getElementById('scrub').value = idx;
}}

document.getElementById('play').onclick = (ev) => {{
  paused = !paused;
  ev.target.innerText = paused ? 'Play' : 'Pause';
  if (!paused) tick();
}};
document.getElementById('step-b').onclick = () => {{
  paused = true; document.getElementById('play').innerText = 'Play';
  if (idx > 0) {{ idx -= 1; render(); }}
}};
document.getElementById('step-f').onclick = () => {{
  paused = true; document.getElementById('play').innerText = 'Play';
  if (idx < DATA.per_tick.length - 1) {{ idx += 1; render(); }}
}};
document.getElementById('restart').onclick = () => {{ idx = 0; render(); }};
document.getElementById('speed').onchange = (ev) => {{ speed = parseFloat(ev.target.value); }};
document.getElementById('scrub').onchange = (ev) => {{ idx = parseInt(ev.target.value, 10); render(); }};
document.getElementById('scrub').oninput  = (ev) => {{ idx = parseInt(ev.target.value, 10); render(); }};

document.getElementById('scrub').max = DATA.per_tick.length - 1;

function tick() {{
  if (paused) return;
  if (idx < DATA.per_tick.length - 1) {{
    idx += 1; render();
    setTimeout(tick, 150 * speed);
  }} else {{
    paused = true;
    document.getElementById('play').innerText = 'Play';
  }}
}}

buildLog();
render();
</script>
</body>
</html>
"""


def _winner_label(s: dict) -> tuple[str, str]:
    w = s.get("winner", "")
    if w == "A":
        return "Team A win", "win"
    if w == "B":
        return "Team A loss", "loss"
    return "Draw", "draw"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="inspect_*.json")
    p.add_argument(
        "--episode",
        type=int,
        default=0,
        help="episode index (0-based); see the JSON's 'episodes' array",
    )
    p.add_argument("--output", type=Path, required=True, help="output .html file")
    p.add_argument(
        "--point-radius",
        type=float,
        default=_DEFAULT_POINT_RADIUS,
        help="cap_duel point_radius for circle drawing; default 0.18 matches the v1 config",
    )
    args = p.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if args.episode < 0 or args.episode >= len(data["episodes"]):
        raise SystemExit(
            f"episode {args.episode} out of range (have {len(data['episodes'])} episodes)"
        )
    episode = data["episodes"][args.episode]
    s = episode["summary"]
    top = data["summary"]
    winner_label, winner_cls = _winner_label(s)
    title = (
        f"seed {s['seed']} ({top.get('mode', '?')}) — "
        f"score {s['team_a_score_ticks']}-{s['team_b_score_ticks']}, "
        f"{s['team_a_kills']} kills, {s['team_a_hits']} hits"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        _HTML_TEMPLATE.format(
            title=title,
            winner_label=winner_label,
            winner_cls=winner_cls,
            checkpoint=top.get("checkpoint", ""),
            mode=top.get("mode", ""),
            seed=s["seed"],
            match_type=s.get("match_type", ""),
            point_radius=args.point_radius,
            recontest_delay=top.get("enemy_recontest_delay", "?"),
            score_to_clear=top.get("score_ticks_to_clear", "?"),
            n_steps=s["steps"],
            data_json=json.dumps(episode),
            recontest_delay_js=int(top.get("enemy_recontest_delay", 12)),
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
