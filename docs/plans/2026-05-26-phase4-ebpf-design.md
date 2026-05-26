# Phase 4: Dynamic DFA eBPF Enforcement — Design Document

**Date:** 2026-05-26
**Status:** Approved

## Mục tiêu

Kết nối DFA artifact từ Phase 3 (`results/p3/dfa_config.json`) với eBPF kernel enforcement. Userspace load file DFA và cập nhật động vào BPF maps; eBPF program chạy state machine trên mỗi syscall, emit alert khi DFA reject.

## Quyết định thiết kế

| Câu hỏi | Quyết định |
|---------|-----------|
| Alert mechanism | `aya-log` đơn giản — log `[ALERT] pid=X syscall=Y` |
| State tracking | Per-process (tgid) — mỗi process một DFA state |
| Unknown syscall | Emit alert + reset về start_state |
| DFA file format | JSON export từ Python (`dfa_config.json`) |

## Kiến trúc tổng thể

```
notebooks/p3/export_dfa.py
        ↓
results/p3/dfa_config.json
        ↓
guepard-shield --dfa dfa_config.json   (userspace)
        ↓ populate maps
┌─────────────────────────────────┐
│  BPF Maps                       │
│  TRANSITION_TABLE[64][102]      │
│  SYSCALL_TO_TOKEN[512]          │
│  PROCESS_STATE{tgid → state}    │
└─────────────────────────────────┘
        ↑ lookup
guepard-shield-ebpf (tracepoint raw_syscalls/sys_enter)
        ↓ alert
aya-log → stdout
```

## Phần 1: Python Export Script

**File:** `notebooks/p3/export_dfa.py`

**Input:**
- `results/p3/dfa_s1/K64_S3/transitions.npz` — arrays `src`, `tok`, `dst` (1973 entries), scalars `n_states=64`, `n_trans=1973`
- `data/processed/p2/vocab.json` — mapping `syscall_name → token_id` (102 entries)
- `ausyscall --dump` — mapping `syscall_nr → syscall_name` (Linux x86_64)
- `results/p3/clusters/K64/start_state_s1.txt` — start_state=59

**Output:** `results/p3/dfa_config.json`

```json
{
  "n_states": 64,
  "n_tokens": 102,
  "start_state": 59,
  "transition_table": [
    [-1, -1, 12, ...],
    ...
  ],
  "syscall_to_token": {
    "0": 5,
    "1": 12,
    "...": -1
  }
}
```

`transition_table` là dense array 64×102 (int, default -1 = no transition).
`syscall_to_token` map syscall_nr (string key) → token_id (-1 nếu không trong vocab).

## Phần 2: Common Types

**File:** `guepard-shield-common/src/lib.rs`

```rust
#![no_std]

#[repr(C)]
pub struct TransitionRow {
    pub dst: [i32; 102],  // -1 = no transition (reject)
}
```

`repr(C)` đảm bảo layout đồng nhất giữa eBPF (no_std) và userspace khi đọc/ghi BPF maps.

`DfaMeta` không cần struct riêng — `start_state=59` được hard-code là constant trong eBPF program.

## Phần 3: BPF Maps

**File:** `guepard-shield-ebpf/src/main.rs`

```rust
// Transition table: state_id (0..64) → row of next states per token
#[map]
static TRANSITION_TABLE: Array<TransitionRow> = Array::with_max_entries(64, 0);

// Syscall number → token_id (-1 nếu unknown)
#[map]
static SYSCALL_TO_TOKEN: Array<i32> = Array::with_max_entries(512, 0);

// Per-process DFA state: tgid → current_state
#[map]
static PROCESS_STATE: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);
```

Tổng memory: `64 × 102 × 4 = ~25 KB` (TRANSITION_TABLE) + `512 × 4 = 2 KB` (SYSCALL_TO_TOKEN) — trong giới hạn eBPF.

## Phần 4: eBPF Program Logic

**File:** `guepard-shield-ebpf/src/main.rs`

```
fn try_guepard_shield(ctx):
  tgid = bpf_get_current_pid_tgid() >> 32
  syscall_nr = ctx.read_at(offset=8)   // args[0] trong raw_syscalls/sys_enter

  token = SYSCALL_TO_TOKEN[syscall_nr]
  if token == -1:
    log ALERT "unknown syscall {nr} pid={tgid}"
    PROCESS_STATE[tgid] = START_STATE   // = 59
    return OK

  state = PROCESS_STATE[tgid] ?? START_STATE
  row = TRANSITION_TABLE[state]
  next_state = row.dst[token]

  if next_state == -1:
    log ALERT "reject: state={state} syscall={nr} pid={tgid}"
    PROCESS_STATE[tgid] = START_STATE
    return OK

  PROCESS_STATE[tgid] = next_state
  return OK
```

`START_STATE = 59u32` là compile-time constant trong eBPF crate.

## Phần 5: Userspace CLI

**File:** `guepard-shield/src/main.rs`

**CLI:** `guepard-shield --dfa <path>`

**Flow:**
1. Parse `--dfa` arg (dùng `clap`)
2. Đọc `dfa_config.json` → `transition_table[64][102]`, `syscall_to_token`
3. Load eBPF binary (`include_bytes_aligned!`)
4. Init `EbpfLogger`
5. Populate maps:
   - `TRANSITION_TABLE`: 64 lần `map.set(state_id, TransitionRow { dst })`
   - `SYSCALL_TO_TOKEN`: iterate entries, `map.set(nr, token_id)`
6. Attach tracepoint `raw_syscalls/sys_enter`
7. Log `"DFA loaded: 64 states, 1973 transitions. Monitoring..."`
8. Chờ `Ctrl-C` → exit

**Dependency bổ sung:** `clap` với feature `derive` (thêm vào `guepard-shield/Cargo.toml`).

## File changes summary

| File | Thay đổi |
|------|---------|
| `notebooks/p3/export_dfa.py` | Tạo mới |
| `guepard-shield-common/src/lib.rs` | Thêm `TransitionRow` struct |
| `guepard-shield-ebpf/src/main.rs` | Thêm maps + DFA state machine logic |
| `guepard-shield/src/main.rs` | Thêm CLI arg parsing + map population |
| `guepard-shield/Cargo.toml` | Thêm `clap` dependency |
