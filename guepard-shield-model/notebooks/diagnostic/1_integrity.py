# %% [markdown]
# # D01 — File Integrity
# Scans all corpus files for missing/empty/mismatched entries.

# %%
from rich.console import Console

from guepard.evaluation.corpus_stats import scan_corpus_integrity

from config import CORPUS as corpus, COUNT_TOKENS

console = Console()

# %%
console.rule("[bold blue]Loading Corpus")
console.print(f"Total sequences in index: [bold]{len(corpus.metadata)}[/bold]")

# %%
console.rule("[bold blue]File Integrity Scan")
seq_data, file_issues = scan_corpus_integrity(corpus.metadata, count_tokens=COUNT_TOKENS)

valid = [s for s in seq_data if s["actual_len"] is not None]
console.print(f"Valid sequences: [bold]{len(valid)}[/bold] / {len(seq_data)} total")

if file_issues:
    console.print(f"[red]{len(file_issues)} issues found:[/red]")
    for seq_id, issue in file_issues[:20]:
        console.print(f"  {seq_id}: {issue}")
    if len(file_issues) > 20:
        console.print(f"  ... and {len(file_issues) - 20} more")
else:
    console.print("[green]OK — no missing/empty/mismatched files[/green]")
