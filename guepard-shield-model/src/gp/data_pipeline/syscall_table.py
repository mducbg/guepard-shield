from pathlib import Path

from gp.config import TBL_PATH

UNK_ID = 335  # unused kernel slot (335-423 gap in syscall_64.tbl)


def load_syscall_table(tbl_path: Path = TBL_PATH) -> dict[str, int]:
    """Parse syscall_64.tbl → {name: number}. Skips x32 entries."""
    table: dict[str, int] = {}
    for line in tbl_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        num, abi, name = int(parts[0]), parts[1], parts[2]
        if abi == "x32":
            continue
        if name not in table:  # prefer common over 64 on collision
            table[name] = num
    return table


def encode(syscall_names: list[str], table: dict[str, int]) -> list[int]:
    return [table.get(name, UNK_ID) for name in syscall_names]
