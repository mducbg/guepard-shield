from pathlib import Path

import msgpack
from array_record.python.array_record_data_source import array_record_module
from gp.data_loader.recording import Recording
from gp.data_pipeline.syscall_table import encode


def write_arrayrecord(
    recordings: list[Recording],
    table: dict[str, int],
    out_path: Path,
) -> None:
    """Convert a list of Recordings to an ArrayRecord file.

    Each record is msgpack-serialized: {"ids": list[int], "length": int, "label": int}
    group_size:1 enables O(1) random access required for shuffled training.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = array_record_module.ArrayRecordWriter(str(out_path), "group_size:1")
    for rec in recordings:
        ids = encode(rec.syscalls, table)
        writer.write(
            msgpack.dumps({"ids": ids, "length": len(ids), "label": rec.label})
        )
    writer.close()
