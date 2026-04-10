"""Phase segmenter for syscall recordings.

Divides a recording into four phases based on syscall rate over time:
  startup  — initial burst of activity (process initialization)
  active   — sustained high-rate processing (main IDS target)
  idle     — low-rate windows (waiting/sleeping)
  shutdown — trailing low-activity cleanup

Requires timestamps (available in LID-DS-2019 and LID-DS-2021).
"""

from enum import Enum

import numpy as np


class Phase(str, Enum):
    STARTUP = "startup"
    ACTIVE = "active"
    IDLE = "idle"
    SHUTDOWN = "shutdown"


def segment(
    timestamps: list[float],
    *,
    window_sec: float = 0.5,
    idle_percentile: float = 10.0,
    startup_fraction: float = 0.20,
    shutdown_fraction: float = 0.20,
) -> list[Phase]:
    """Assign a phase label to each syscall.

    Args:
        timestamps:        Per-syscall timestamps in seconds (same length as syscalls list).
        window_sec:        Duration of each rate-measurement window in seconds.
        idle_percentile:   Windows with rate below this percentile of all window rates
                           are classified as idle.
        startup_fraction:  Fraction of total recording duration treated as startup zone.
        shutdown_fraction: Fraction of total recording duration treated as shutdown zone.

    Returns:
        List of Phase values, one per syscall (same length as timestamps).
    """
    n = len(timestamps)
    if n == 0:
        return []
    if n == 1:
        return [Phase.ACTIVE]

    ts = np.asarray(timestamps, dtype=np.float64)
    t_start, t_end = ts[0], ts[-1]
    duration = t_end - t_start
    if duration <= 0:
        return [Phase.ACTIVE] * n

    # Build windows and compute rate
    edges = np.arange(t_start, t_end + window_sec, window_sec)
    counts, _ = np.histogram(ts, bins=edges)
    rates = counts / window_sec  # syscalls per second

    # Idle threshold from non-empty windows
    nonzero_rates = rates[rates > 0]
    idle_threshold = (
        float(np.percentile(nonzero_rates, idle_percentile))
        if len(nonzero_rates) > 0
        else 0.0
    )

    startup_end = t_start + startup_fraction * duration
    shutdown_start = t_end - shutdown_fraction * duration

    # Assign phase per window
    window_phases: list[Phase] = []
    for i, rate in enumerate(rates):
        w_mid = edges[i] + window_sec / 2
        if rate <= idle_threshold:
            window_phases.append(Phase.IDLE)
        elif w_mid <= startup_end:
            window_phases.append(Phase.STARTUP)
        elif w_mid >= shutdown_start:
            window_phases.append(Phase.SHUTDOWN)
        else:
            window_phases.append(Phase.ACTIVE)

    # Map each syscall to its window's phase
    window_indices = np.searchsorted(edges[1:], ts, side="right")
    window_indices = np.clip(window_indices, 0, len(window_phases) - 1)
    return [window_phases[i] for i in window_indices]
