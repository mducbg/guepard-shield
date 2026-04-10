"""Recording dataclass for syscall datasets.

This module provides the shared Recording data structure used across all loaders.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Recording:
    """A single syscall recording (sequence).

    Attributes:
        path: Path to the recording file or directory
        label: 0 = normal, 1 = attack
        syscalls: List of syscall names in sequence order
        timestamps: Optional list of timestamps (for rich features)
        tid: Optional list of thread IDs (for rich features)
        args: Optional list of syscall arguments (for rich features)
        metadata: Optional dict with additional metadata
    """

    path: str
    label: int
    syscalls: List[str]
    timestamps: Optional[List[float]] = None
    tid: Optional[List[int]] = None
    args: Optional[List[str]] = None
    metadata: Optional[dict] = None

    def to_numpy(self) -> tuple[np.ndarray, int]:
        """Convert recording to numpy array format.

        Returns:
            Tuple of (syscalls as numpy array of strings, label)
        """
        return np.array(self.syscalls, dtype=object), self.label

    def __len__(self) -> int:
        """Return the number of syscalls in this recording."""
        return len(self.syscalls)
