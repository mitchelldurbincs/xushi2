"""Verify every captured decision and summarize objective access from native state.

Run from the repository root after extracting captures.tar.gz beside this file:
    python docs/reports/2026-09-19-sampled-replays/analyze.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
from scripts.analyze_replay_combat import _config_from_header, _load_replay, analyze_replay
from xushi2 import xushi2_cpp as cpp
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice


def inspect(path, record):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    header, decisions = _load_replay(path)
    sim = cpp.Sim(_config_from_header(header))
    assert f'0x{sim.state_hash:016x}' == record['initial_state_hash'], path
    sidecar = json.loads(path.with_suffix('.replay.json').read_text())
    rows = []
    counts = defaultdict(float)
    first_a = first_b = None
    actor = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    hp = actor_field_slice('own_hp').start
    point = actor_field_slice('self_on_point').start
    previous_tick = 0
    uncontested_run_a = 0.0
    counts['longest_uncontested_seconds_a'] = 0.0
    for decision, (tick, state_hash) in zip(decisions, sidecar['state_hashes'], strict=True):
        assert sim.tick == decision.tick, (path, decision.tick)
        sim.step_decision(decision.actions)
        assert sim.tick == tick and f'0x{sim.state_hash:016x}' == state_hash, (path, tick)
        alive = []
        on = []
        for slot in range(6):
            cpp.build_actor_obs(sim, slot, actor)
            alive.append(bool(actor[hp] > 0))
            on.append(bool(actor[hp] > 0 and actor[point] > .5))
        aa, ab, oa, ob = sum(alive[:3]), sum(alive[3:]), sum(on[:3]), sum(on[3:])
        seconds = min(tick - previous_tick, max(0, tick - sim.objective_unlock_ticks)) / cpp.TICK_HZ
        previous_tick = tick
        if oa and first_a is None:
            first_a = tick / cpp.TICK_HZ
        if ob and first_b is None:
            first_b = tick / cpp.TICK_HZ
        if tick >= sim.objective_unlock_ticks:
            counts['unlocked_seconds'] += seconds
            counts['alive_agent_seconds_a'] += aa * seconds
            counts['alive_agent_seconds_b'] += ab * seconds
            counts['on_point_agent_seconds_a'] += oa * seconds
            counts['on_point_agent_seconds_b'] += ob * seconds
            counts['alive_but_off_point_seconds_a'] += seconds * bool(aa and not oa)
            counts['alive_but_off_point_seconds_b'] += seconds * bool(ab and not ob)
            counts['alive_advantage_off_point_seconds_a'] += seconds * bool(aa > ab and not oa)
            counts['on_point_seconds_a'] += seconds * bool(oa)
            counts['on_point_seconds_b'] += seconds * bool(ob)
            counts['uncontested_seconds_a'] += seconds * bool(oa and not ob)
            counts['uncontested_seconds_b'] += seconds * bool(ob and not oa)
            uncontested_run_a = uncontested_run_a + seconds if oa and not ob else 0.0
            counts['longest_uncontested_seconds_a'] = max(counts['longest_uncontested_seconds_a'], uncontested_run_a)
            counts['contested_seconds'] += seconds * bool(oa and ob)
        if tick % (5 * cpp.TICK_HZ) == 0 or sim.episode_over:
            rows.append(dict(seconds=tick / cpp.TICK_HZ, alive_a=aa, alive_b=ab,
                             on_point_a=oa, on_point_b=ob,
                             score_a=sim.team_a_score, score_b=sim.team_b_score,
                             kills_a=sim.team_a_kills, kills_b=sim.team_b_kills))
    for key in ('tick', 'team_a_score', 'team_b_score', 'team_a_kills', 'team_b_kills'):
        assert getattr(sim, key) == record['final'][key], (path, key)
    assert sim.episode_over
    return dict(replay=str(path.relative_to(HERE)), verified_decisions=len(decisions),
                outcome={'A':'win', 'B':'loss'}.get(record['final']['winner'], 'draw'),
                final=record['final'], first_on_point_seconds_a=first_a,
                first_on_point_seconds_b=first_b, objective=dict(counts), timeline=rows)


def main():
    results = []
    selected = []
    summary = []
    for directory in sorted((HERE / 'captures').glob('*/cell-000')):
        index = json.loads((directory / 'index.json').read_text())
        cell = directory.parent.name
        episodes = [inspect(directory / record['replay'], record) for record in index['episodes']]
        results.extend(dict(cell=cell, **episode) for episode in episodes)
        for outcome in ('win', 'loss', 'draw'):
            group = [ep for ep in episodes if ep['outcome'] == outcome]
            if not group:
                continue
            # Predeclared rule: first episode of each outcome, never choose by appearance.
            chosen = dict(cell=cell, **group[0])
            chosen['combat'] = analyze_replay(HERE / chosen['replay'])
            chosen['combat']['replay'] = chosen['replay']
            selected.append(chosen)
            summary.append(dict(cell=cell, outcome=outcome, episodes=len(group),
                never_on_point_a=sum(ep['first_on_point_seconds_a'] is None for ep in group),
                mean_score_a=sum(ep['final']['team_a_score'] for ep in group)/len(group),
                mean_score_b=sum(ep['final']['team_b_score'] for ep in group)/len(group),
                mean_kills_a=sum(ep['final']['team_a_kills'] for ep in group)/len(group),
                mean_kills_b=sum(ep['final']['team_b_kills'] for ep in group)/len(group),
                objective_means={key:sum(ep['objective'].get(key,0) for ep in group)/len(group)
                                 for key in group[0]['objective']}))
        print(cell, len(episodes), 'episodes verified', flush=True)
    assert len(results) == 480, len(results)
    payload = dict(verified_episodes=len(results), verified_decisions=sum(ep['verified_decisions'] for ep in results),
                   measurement='Post-decision native actor state, all six slots. Objective duration metrics start at unlock; decision-time approximation, not exact tick-integrated occupancy. Combat diagnostics reuse analyze_replay_combat, whose aim error is pre-action and to the nearest visible aim direction, not a measure of shot accuracy.',
                   selection='First episode of each outcome in completion order for each cell.', groups=summary, selected=selected)
    (HERE/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    (HERE/'episode_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
    print('VERIFIED',payload['verified_episodes'],payload['verified_decisions'],flush=True)

if __name__ == '__main__':
    main()
