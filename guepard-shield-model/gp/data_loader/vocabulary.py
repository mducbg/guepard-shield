"""Syscall vocabulary for LID-DS-2021 and cross-dataset mapping."""

from __future__ import annotations

from pathlib import Path

# Special tokens
PAD_TOKEN = "<pad>"      # ID 0
UNK_TOKEN = "<unk>"      # ID 1 (OOV from other datasets/test)
UNKNOWN_TOKEN = "<unknown>"  # ID 2 (Sysdig-specific unresolved syscalls in LID-DS)

class SyscallVocabulary:
    """Mapping between syscall names and integer IDs."""
    
    def __init__(self, syscall_list: list[str]):
        # Start with special tokens
        self.id_to_syscall = [PAD_TOKEN, UNK_TOKEN, UNKNOWN_TOKEN]
        
        # Add provided syscalls (excluding specials if they are in the list)
        for s in sorted(syscall_list):
            if s not in self.id_to_syscall:
                self.id_to_syscall.append(s)
        
        self.syscall_to_id = {s: i for i, s in enumerate(self.id_to_syscall)}

    @property
    def size(self) -> int:
        return len(self.id_to_syscall)

    def encode(self, syscalls: list[str]) -> list[int]:
        """Convert a list of syscall names to IDs."""
        return [self.syscall_to_id.get(s, self.syscall_to_id[UNK_TOKEN]) for s in syscalls]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert a list of IDs back to syscall names."""
        return [self.id_to_syscall[i] if i < self.size else UNK_TOKEN for i in ids]

    @classmethod
    def from_file(cls, path: Path) -> SyscallVocabulary:
        """Load vocabulary from a newline-separated file."""
        with open(path) as f:
            syscalls = [line.strip() for line in f if line.strip()]
        return cls(syscalls)

    def save(self, path: Path):
        """Save the full vocabulary (including specials) to a file."""
        with open(path, "w") as f:
            for s in self.id_to_syscall:
                f.write(f"{s}\n")
