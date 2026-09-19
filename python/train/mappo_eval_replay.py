"""Write exact counted evaluation episodes, without running another policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def recording_env(env_fn):
    env = env_fn()
    setter = getattr(env, "set_replay_capture", None)
    if setter is None:
        env.close()
        raise ValueError("this environment does not support exact replay capture")
    setter(True)
    return env


class EvalReplayWriter:
    def __init__(self, directory: Path, metadata: dict):
        self.directory = Path(directory)
        # A failed or repeated capture must never mix old and new episodes.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = metadata
        self.episodes: list[dict] = []
        self.lanes: list[dict] = []

    @staticmethod
    def _lane(info: dict, episode: int) -> dict:
        return {
            "header": info["replay_header"],
            "initial_state_hash": info["state_hash"],
            "lane_episode": episode,
            "decisions": [],
            "state_hashes": [],
        }

    def reset(self, infos: list[dict]) -> None:
        self.lanes = [self._lane(info, 0) for info in infos]

    def append(self, infos: list[dict]) -> None:
        for lane, info in zip(self.lanes, infos, strict=True):
            final = info.get("final_info", info)
            lane["decisions"].append(final["replay_decision"])
            lane["state_hashes"].append([int(final["tick"]), final["state_hash"]])

    def complete(
        self, lane_index: int, info: dict, *, reward: float, terminated: bool, truncated: bool
    ) -> None:
        lane = self.lanes[lane_index]
        final = info.get("final_info", info)
        name = f"episode-{len(self.episodes):04d}.replay"
        payload = ("\n".join([lane["header"], *lane["decisions"]]) + "\n").encode("ascii")
        (self.directory / name).write_bytes(payload)
        record = {
            "episode": len(self.episodes),
            "vector_lane": lane_index,
            "lane_episode": lane["lane_episode"],
            "replay": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "initial_state_hash": lane["initial_state_hash"],
            "state_hashes": lane["state_hashes"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "final": {
                key: final[key]
                for key in (
                    "tick",
                    "state_hash",
                    "winner",
                    "learner_team",
                    "team_a_score",
                    "team_b_score",
                    "team_a_kills",
                    "team_b_kills",
                )
            },
        }
        sidecar = name + ".json"
        (self.directory / sidecar).write_text(json.dumps(record, indent=2) + "\n")
        self.episodes.append(
            {key: value for key, value in record.items() if key != "state_hashes"}
            | {"sidecar": sidecar}
        )
        self.lanes[lane_index] = self._lane(info["reset_info"], lane["lane_episode"] + 1)

    def finish(self) -> None:
        (self.directory / "index.json").write_text(
            json.dumps(
                {
                    **self.metadata,
                    "completed_episodes": len(self.episodes),
                    "episodes": self.episodes,
                },
                indent=2,
            )
            + "\n"
        )
