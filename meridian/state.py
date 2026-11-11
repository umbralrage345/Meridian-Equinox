"""Mission state persistence and schema definition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class _Subscriptable:
    """Helper allowing dataclasses to support dict-style subscripting."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ResourcesState(_Subscriptable):
    fuel_pct: float = 100.0
    sensor_noise: float = 0.0
    hull_health: float = 100.0


@dataclass
class Act1Target(_Subscriptable):
    star_id: str = ""
    is_real_planet: bool = False
    depth: float = 0.0
    period_days: float = 0.0
    gravity: float = 7.0
    wind: float = 2.0
    trip_days: int = 100
    label: str = "unscouted world"


@dataclass
class Act1State(_Subscriptable):
    done: bool = False
    score: float = 0.0
    auc: float = 0.0
    target: Act1Target = field(default_factory=Act1Target)


@dataclass
class Act2State(_Subscriptable):
    done: bool = False
    score: float = 0.0
    n_faults: int = 0
    caught: int = 0
    missed: int = 0
    false_alarms: int = 0
    mean_delay: float = 0.0
    alarm_coverage: float = 0.0


@dataclass
class Act3State(_Subscriptable):
    done: bool = False
    score: float = 0.0
    accuracy: float = 0.0
    value_ratio: float = 0.0
    backend: str = "sklearn"


@dataclass
class Act4State(_Subscriptable):
    done: bool = False
    score: float = 0.0
    landing_rate: float = 0.0
    mean_reward: float = 0.0
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissionState(_Subscriptable):
    """Encapsulates the state of a team's probe mission across all acts."""

    team_id: str
    schema_version: int = 1
    resources: ResourcesState = field(default_factory=ResourcesState)
    act1: Act1State = field(default_factory=Act1State)
    act2: Act2State = field(default_factory=Act2State)
    act3: Act3State = field(default_factory=Act3State)
    act4: Act4State = field(default_factory=Act4State)

    @property
    def total_score(self) -> float:
        """Total cumulative mission score across all 4 acts."""
        return float(self.act1.score + self.act2.score + self.act3.score + self.act4.score)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a nested dictionary representation."""
        data = asdict(self)
        data["total_score"] = self.total_score
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionState:
        """Construct MissionState instance from a nested dictionary."""
        team_id = str(data.get("team_id", "unknown"))
        schema_version = int(data.get("schema_version", 1))

        res_data = data.get("resources", {})
        resources = ResourcesState(
            fuel_pct=float(res_data.get("fuel_pct", 100.0)),
            sensor_noise=float(res_data.get("sensor_noise", 0.0)),
            hull_health=float(res_data.get("hull_health", 100.0)),
        )

        a1_data = data.get("act1", {})
        t_data = a1_data.get("target", {})
        target = Act1Target(
            star_id=str(t_data.get("star_id", "")),
            is_real_planet=bool(t_data.get("is_real_planet", False)),
            depth=float(t_data.get("depth", 0.0)),
            period_days=float(t_data.get("period_days", 0.0)),
            gravity=float(t_data.get("gravity", 7.0)),
            wind=float(t_data.get("wind", 2.0)),
            trip_days=int(t_data.get("trip_days", 100)),
            label=str(t_data.get("label", "unscouted world")),
        )
        act1 = Act1State(
            done=bool(a1_data.get("done", False)),
            score=float(a1_data.get("score", 0.0)),
            auc=float(a1_data.get("auc", 0.0)),
            target=target,
        )

        a2_data = data.get("act2", {})
        act2 = Act2State(
            done=bool(a2_data.get("done", False)),
            score=float(a2_data.get("score", 0.0)),
            n_faults=int(a2_data.get("n_faults", 0)),
            caught=int(a2_data.get("caught", 0)),
            missed=int(a2_data.get("missed", 0)),
            false_alarms=int(a2_data.get("false_alarms", 0)),
            mean_delay=float(a2_data.get("mean_delay", 0.0)),
        )

        a3_data = data.get("act3", {})
        act3 = Act3State(
            done=bool(a3_data.get("done", False)),
            score=float(a3_data.get("score", 0.0)),
            accuracy=float(a3_data.get("accuracy", 0.0)),
            value_ratio=float(a3_data.get("value_ratio", 0.0)),
            backend=str(a3_data.get("backend", "sklearn")),
        )

        a4_data = data.get("act4", {})
        act4 = Act4State(
            done=bool(a4_data.get("done", False)),
            score=float(a4_data.get("score", 0.0)),
            landing_rate=float(a4_data.get("landing_rate", 0.0)),
            mean_reward=float(a4_data.get("mean_reward", 0.0)),
            conditions=dict(a4_data.get("conditions", {})),
        )

        return cls(
            team_id=team_id,
            schema_version=schema_version,
            resources=resources,
            act1=act1,
            act2=act2,
            act3=act3,
            act4=act4,
        )

    def save(self, path: Path | str | None = None) -> Path:
        """Persist state to JSON file."""
        target_path = Path(path) if path is not None else Path(f"./mission_state_{self.team_id}.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path

    @classmethod
    def load(cls, team_id: str, path: Path | str | None = None) -> MissionState:
        """Load state from JSON file or create and save default state if absent."""
        target_path = Path(path) if path is not None else Path(f"./mission_state_{team_id}.json")
        if not target_path.exists():
            default_state = cls(team_id=team_id)
            default_state.save(target_path)
            return default_state

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_state(team_id: str, path: Path | str | None = None) -> MissionState:
    """Convenience helper to load or create a MissionState for the given team_id."""
    return MissionState.load(team_id=team_id, path=path)


def save_state(state: MissionState, path: Path | str | None = None) -> Path:
    """Convenience helper to save a MissionState to disk."""
    return state.save(path=path)

