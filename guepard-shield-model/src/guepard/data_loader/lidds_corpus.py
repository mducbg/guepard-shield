import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tqdm import tqdm

from .corpus import SequenceMeta

logger = logging.getLogger(__name__)


@dataclass
class LiddsRecordingMeta(SequenceMeta):
    """Extended metadata for LID-DS-2021 recordings."""

    scenario: str = ""
    has_exploit: bool = False
    exploit_times_ns: list[float] = field(default_factory=list)
    recording_time: float = 0.0


@lru_cache(maxsize=16)
def read_sc_tokens_and_timestamps(
    file_path_str: str,
) -> tuple[List[str], List[float]]:
    """Parse .sc exit events, return (tokens, timestamps_ns).

    LRU-cached with maxsize=16 to support up to 16 DataLoader workers.
    Single parse pass shared by both token and timestamp consumers.
    """
    tokens: list[str] = []
    timestamps: list[float] = []
    with open(file_path_str, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            # Format: timestamp uid pid proc_name tid syscall_name direction [args...]
            # Keep only exit events (direction = "<")
            if len(parts) >= 7 and parts[6] == "<":
                timestamps.append(float(parts[0]))
                tokens.append(parts[5])
    return tokens, timestamps


def read_sc_tokens(file_path_str: str) -> List[str]:
    """Return syscall name list for .sc file. Backed by read_sc_tokens_and_timestamps cache."""
    return read_sc_tokens_and_timestamps(file_path_str)[0]


def exploit_window_label(
    window_start: int,
    window_end: int,
    timestamps: List[float],
    exploit_times_ns: List[float],
) -> int:
    """Return 1 if any event in [window_start, window_end) occurred after the exploit start.

    Following LID-DS literature convention: a window is attack if it contains at
    least one syscall with timestamp >= min(exploit_times_ns).  This labels all
    post-exploit activity as attack, giving enough attack windows per recording
    for the sampler to reliably include them.
    """
    if not exploit_times_ns:
        return 0
    window_ts = timestamps[window_start:window_end]
    if not window_ts:
        return 0
    exploit_start = min(exploit_times_ns)
    return 1 if any(ts >= exploit_start for ts in window_ts) else 0


def lidds_label_fn(seq_meta, win_start: int, win_end: int) -> int:
    """Window-level label for LID-DS: 1 if window overlaps post-exploit activity.

    For normal recordings (no exploit_times_ns) returns seq_meta.label (= 0).
    For attack recordings returns 1 for any window on or after the first exploit
    event — post-exploit syscalls carry the attack context.
    """
    if (
        not isinstance(seq_meta, LiddsRecordingMeta)
        or not seq_meta.exploit_times_ns
    ):
        return seq_meta.label
    _, timestamps = read_sc_tokens_and_timestamps(str(seq_meta.file_path))
    return exploit_window_label(win_start, win_end, timestamps, seq_meta.exploit_times_ns)


class LiddsCorpus:
    """Parser for the LID-DS-2021 dataset.

    Directory structure per scenario::

        scenario_name/
        ├── training/               # normal recordings only
        │   └── sample_dir/
        │       ├── sample.sc
        │       └── sample.json
        ├── validation/             # normal recordings only
        │   └── ...
        └── test/
            ├── normal/             # normal recordings
            │   └── ...
            └── normal_and_attack/  # mixed: exploit=True/False per .json
                └── ...

    Parameters
    ----------
    data_dir : str | Path
        Root directory of LID-DS-2021 (contains scenario subdirectories).
    scenarios : list[str] | None
        Scenario names to load. None = all scenarios found.
    """

    _SPLITS = [
        ("training", "training"),
        ("validation", "validation"),
        ("test_normal", "test/normal"),
        ("test_attack", "test/normal_and_attack"),
    ]

    def __init__(self, data_dir: str | Path, scenarios: list[str] | None = None):
        self.data_dir = Path(data_dir)
        self.metadata: list[LiddsRecordingMeta] = []
        self._scenarios = scenarios
        self._build_index()

    # ------------------------------------------------------------------
    # Index building with persistent cache
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        scenario_dirs = self._resolve_scenario_dirs()
        cache_path = self.data_dir / "_lidds_seq_lengths.json"
        cache = self._load_cache(cache_path)
        cache_dirty = False

        found = 0
        for scenario_dir in tqdm(scenario_dirs, desc="Indexing LID-DS scenarios"):
            scenario = scenario_dir.name

            for split_name, split_subdir in self._SPLITS:
                split_dir = scenario_dir / split_subdir
                if not split_dir.exists():
                    continue

                for sample_dir in sorted(split_dir.iterdir()):
                    if not sample_dir.is_dir():
                        continue

                    sc_files = list(sample_dir.glob("*.sc"))
                    if not sc_files:
                        continue
                    sc_path = sc_files[0]

                    # Parse .json metadata
                    has_exploit, exploit_times, recording_time = self._parse_json(
                        sample_dir
                    )

                    # seq_length: cached or computed
                    cache_key = str(sc_path.relative_to(self.data_dir))
                    if cache_key in cache:
                        seq_length = cache[cache_key]
                    else:
                        seq_length = _count_exit_events(sc_path)
                        cache[cache_key] = seq_length
                        cache_dirty = True

                    label = 1 if has_exploit else 0

                    self.metadata.append(
                        LiddsRecordingMeta(
                            seq_id=sample_dir.name,
                            bug_name=scenario,
                            seq_class=split_name,
                            label=label,
                            seq_length=seq_length,
                            file_path=sc_path,
                            scenario=scenario,
                            has_exploit=has_exploit,
                            exploit_times_ns=exploit_times,
                            recording_time=recording_time,
                        )
                    )
                    found += 1

        if cache_dirty:
            self._save_cache(cache_path, cache)

        n_scenarios = len(scenario_dirs)
        n_attack = sum(1 for m in self.metadata if m.label == 1)
        logger.info(
            f"Indexed {found} recordings ({n_attack} attack) "
            f"across {n_scenarios} scenarios"
        )

    def _resolve_scenario_dirs(self) -> list[Path]:
        if self._scenarios:
            dirs = []
            for s in self._scenarios:
                d = self.data_dir / s
                if d.is_dir():
                    dirs.append(d)
                else:
                    logger.warning(f"Scenario directory not found: {d}")
            return dirs
        return sorted(
            d
            for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
        )

    @staticmethod
    def _parse_json(
        sample_dir: Path,
    ) -> Tuple[bool, list[float], float]:
        json_files = list(sample_dir.glob("*.json"))
        if not json_files:
            return False, [], 0.0
        try:
            with open(json_files[0]) as f:
                meta = json.load(f)
            has_exploit = meta.get("exploit", False)
            # JSON "absolute" is in Unix seconds; .sc timestamps are nanoseconds.
            exploit_times = [
                e.get("absolute", 0) * 1e9
                for e in meta.get("time", {}).get("exploit", [])
                if isinstance(e, dict)
            ]
            recording_time = float(meta.get("recording_time", 0))
            return has_exploit, exploit_times, recording_time
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Error parsing JSON in {sample_dir}: {e}")
            return False, [], 0.0

    @staticmethod
    def _load_cache(path: Path) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    @staticmethod
    def _save_cache(path: Path, cache: dict) -> None:
        try:
            with open(path, "w") as f:
                json.dump(cache, f)
        except OSError as e:
            logger.warning(f"Could not write cache: {e}")

    # ------------------------------------------------------------------
    # Public API (mirrors DongTingCorpus interface)
    # ------------------------------------------------------------------

    def get_split(self, split_name: str) -> list[LiddsRecordingMeta]:
        """Get recordings by split name.

        Supported names: ``training``, ``validation``, ``test_normal``,
        ``test_attack``, ``test`` (= test_normal + test_attack),
        or any custom name assigned via ``seq_class`` mutation.
        """
        target = split_name.lower()
        if target == "test":
            return [m for m in self.metadata if m.seq_class.startswith("test")]
        return [m for m in self.metadata if m.seq_class == target]

    def iter_sequences(
        self, split_name: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[Tuple[str, int, List[str]]]:
        metas = self.get_split(split_name) if split_name else self.metadata
        if limit is not None:
            metas = metas[:limit]
        for meta in metas:
            try:
                tokens = read_sc_tokens(str(meta.file_path))
                yield meta.seq_id, meta.label, tokens
            except Exception as e:
                logger.warning(f"Error reading {meta.file_path}: {e}")

    def iter_corpus(self, limit: Optional[int] = None) -> Iterator[List[str]]:
        for _, _, tokens in self.iter_sequences(limit=limit):
            yield tokens


def _count_exit_events(sc_path: Path) -> int:
    """Count exit events in a .sc file without full parsing."""
    count = 0
    with open(sc_path, "rb") as f:  # binary mode for speed
        for line in f:
            if b" < " in line:
                count += 1
    return count
