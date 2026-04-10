# %% [markdown]
# # P1 — ArrayRecord Conversion
#
# Converts LID-DS-2019 and LID-DS-2021 into ArrayRecord format for Grain training.
# Run this once before P2 Teacher training.
#
# Output: `data/processed/lidds_2019/` and `data/processed/lidds_2021/`

# %% Setup
import random

from rich.console import Console
from rich.table import Table

from gp.config import PROCESSED_DATA_DIR
from gp.data_loader.lidds_2019_loader import LIDDS2019Dataset
from gp.data_loader.lidds_2021_loader import LIDDS2021Dataset
from gp.data_pipeline.convert import write_arrayrecord
from gp.data_pipeline.syscall_table import load_syscall_table

console = Console()
table = load_syscall_table()
console.print(f"Syscall table loaded: {len(table)} entries")


# %% Helper
def split_recordings(recordings, *, train=0.70, val=0.15, seed=42):
    """Random stratified split by label."""
    rng = random.Random(seed)
    normal = [r for r in recordings if r.label == 0]
    attack = [r for r in recordings if r.label == 1]
    rng.shuffle(normal)
    rng.shuffle(attack)

    def _split(xs):
        n = len(xs)
        i1 = int(n * train)
        i2 = int(n * (train + val))
        return xs[:i1], xs[i1:i2], xs[i2:]

    n_tr, n_va, n_te = _split(normal)
    a_tr, a_va, a_te = _split(attack)
    return n_tr + a_tr, n_va + a_va, n_te + a_te


def show_split_stats(name, splits: dict[str, list]):
    t = Table(title=f"{name} splits")
    t.add_column("Split")
    t.add_column("Total", justify="right")
    t.add_column("Normal", justify="right")
    t.add_column("Attack", justify="right")
    for split_name, recs in splits.items():
        n = sum(1 for r in recs if r.label == 0)
        a = sum(1 for r in recs if r.label == 1)
        t.add_row(split_name, str(len(recs)), str(n), str(a))
    console.print(t)


# %% Convert LID-DS-2021
console.rule("LID-DS-2021")
lidds_2021_splits = {}
for split in ["train", "val", "test"]:
    recs = list(LIDDS2021Dataset(splits=[split]))
    lidds_2021_splits[split] = recs
    out = PROCESSED_DATA_DIR / "lidds_2021" / f"{split}.arrayrecord"
    write_arrayrecord(recs, table, out)
    console.print(f"[green]✓[/green] {split}: {len(recs)} recordings → {out}")

show_split_stats("LID-DS-2021", lidds_2021_splits)

# %% Convert LID-DS-2019
console.rule("LID-DS-2019")
all_2019 = list(LIDDS2019Dataset())
console.print(f"Total recordings: {len(all_2019)}")

train_recs, val_recs, test_recs = split_recordings(all_2019)
lidds_2019_splits = {"train": train_recs, "val": val_recs, "test": test_recs}

for split_name, recs in lidds_2019_splits.items():
    out = PROCESSED_DATA_DIR / "lidds_2019" / f"{split_name}.arrayrecord"
    write_arrayrecord(recs, table, out)
    console.print(f"[green]✓[/green] {split_name}: {len(recs)} recordings → {out}")

show_split_stats("LID-DS-2019", lidds_2019_splits)

console.print("\n[bold green]Conversion complete.[/bold green]")
console.print(f"Output directory: {PROCESSED_DATA_DIR}")
