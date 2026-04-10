from pathlib import Path

import grain
import grain.experimental
import msgpack
import numpy as np


def _parse(raw: bytes, max_len: int) -> dict:
    rec = msgpack.loads(raw)
    ids = rec["ids"][:max_len]
    pad = max_len - len(ids)
    return {
        "input_ids": np.array(ids + [0] * pad, dtype=np.int32),
        "length": np.int32(min(rec["length"], max_len)),
        "label": np.int32(rec["label"]),
    }


def make_loader(
    path: str | Path,
    *,
    batch_size: int,
    max_len: int,
    seed: int,
    shuffle: bool,
    ram_budget_mb: int = 2048,
) -> grain.IterDataset:
    """Build a Grain IterDataset from an ArrayRecord file.

    Records are lazily read — only prefetch_buffer_size records in RAM at a time.
    ram_budget_mb controls the auto-tuned prefetch buffer size.

    Yields batches of:
        input_ids: (batch_size, max_len) int32 — zero-padded syscall numbers
        length:    (batch_size,)         int32 — true sequence length for masking
        label:     (batch_size,)         int32 — 0 = normal, 1 = attack
    """
    source = grain.sources.ArrayRecordDataSource(str(path))
    ds = grain.MapDataset.source(source)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.map(lambda x: _parse(x, max_len))

    perf = grain.experimental.pick_performance_config(
        ds=ds,
        ram_budget_mb=ram_budget_mb,
    )
    return (
        ds.batch(batch_size)
        .to_iter_dataset(read_options=perf.read_options)
        .mp_prefetch(perf.multiprocessing_options)
    )
