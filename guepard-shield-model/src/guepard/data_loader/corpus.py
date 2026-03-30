import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SequenceMeta:
    seq_id: str
    bug_name: str
    seq_class: str
    label: int  # 1 for abnormal, 0 for normal
    seq_length: int
    file_path: Path


class DongTingCorpus:
    """
    Parses the DongTing dataset structure consisting of a Baseline.csv
    metadata index and a directory of .log files containing syscall traces.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.baseline_csv = self.data_dir / "Baseline.csv"

        if not self.baseline_csv.exists():
            raise FileNotFoundError(f"Baseline index not found at {self.baseline_csv}")

        self.metadata: List[SequenceMeta] = []
        self._build_index()

    def _build_index(self):
        # 1. Gather all log files
        bug_to_path: Dict[str, Path] = {}
        for p in self.data_dir.rglob("*.log"):
            name = p.stem
            # Map exact name
            bug_to_path[name] = p
            # Map name without sy_ prefix
            if name.startswith("sy_"):
                bug_to_path[name[3:]] = p

        # 2. Parse Baseline.csv
        found = 0
        missing = 0

        with open(self.baseline_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bug_name = row["kcb_bug_name"]
                if bug_name.endswith(".log"):
                    bug_name = bug_name[:-4]

                seq_id = row["kcb_seq_id"]
                seq_class = row["kcb_seq_class"]  # e.g., DTDS-train
                label_str = row["kcb_seq_lables"]  # e.g., Attach, Normal

                label = 0 if label_str.lower() == "normal" else 1
                seq_length = int(row.get("kcb_syscall_counts", 0))

                # Try finding the file corresponding to the buggy logic sequence
                if bug_name in bug_to_path:
                    path = bug_to_path[bug_name]
                elif f"sy_{bug_name}" in bug_to_path:
                    path = bug_to_path[f"sy_{bug_name}"]
                else:
                    missing += 1
                    continue

                found += 1
                self.metadata.append(
                    SequenceMeta(
                        seq_id=seq_id,
                        bug_name=bug_name,
                        seq_class=seq_class,
                        label=label,
                        seq_length=seq_length,
                        file_path=path,
                    )
                )

        logger.info(
            f"Loaded {found} sequences from corpus index. {missing} files were missing."
        )

    def get_split(self, split_name: str) -> List[SequenceMeta]:
        """
        Get metadata for a specific dataset split ('train', 'validation', 'test').
        """
        target = split_name.lower()
        return [m for m in self.metadata if target in m.seq_class.lower()]

    def iter_sequences(
        self, split_name: Optional[str] = None, limit: Optional[int] = None
    ) -> Iterator[Tuple[str, int, List[str]]]:
        """
        Iterates over the sequences, yielding (seq_id, label, tokens).
        """
        if split_name:
            metas = self.get_split(split_name)
        else:
            metas = self.metadata

        if limit is not None:
            metas = metas[:limit]

        for meta in metas:
            try:
                with open(meta.file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    tokens = content.split("|")
                    yield meta.seq_id, meta.label, tokens
            except Exception as e:
                logger.warning(f"Error reading {meta.file_path}: {e}")

    def iter_corpus(self, limit: Optional[int] = None) -> Iterator[List[str]]:
        """
        Iterates over the tokens of each sequence. Useful for vocabulary building.
        """
        for _, _, tokens in self.iter_sequences(limit=limit):
            yield tokens
