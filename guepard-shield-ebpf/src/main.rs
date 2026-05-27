#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::{bpf_get_current_pid_tgid, bpf_ktime_get_ns},
    macros::{map, tracepoint},
    maps::{Array, HashMap, PerCpuArray, RingBuf},
    programs::TracePointContext,
};
use guepard_shield_common::{DfaEvent, TransitionRow};

const START_STATE: u32 = 59;

const KIND_SUSPECT: u8 = 1;
const KIND_ATTACK: u8 = 2;

#[map]
static TRANSITION_TABLE: Array<TransitionRow> = Array::with_max_entries(64, 0);

#[map]
static SYSCALL_TO_TOKEN: Array<i32> = Array::with_max_entries(512, 0);

/// Per-state classification: 0 = NORMAL, 1 = SUSPECT.
/// Populated from state_class array in dfa_config.json at startup.
#[map]
static STATE_CLASS: Array<u8> = Array::with_max_entries(64, 0);

#[map]
static PROCESS_STATE: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

#[map]
static TARGET_TGID: Array<u32> = Array::with_max_entries(1, 0);

// 32 buckets x 100 ns; bucket 31 = overflow (>= 3100 ns). PerCpu avoids atomic ops.
#[map]
static LATENCY_HIST: PerCpuArray<u64> = PerCpuArray::with_max_entries(32, 0);

/// Ring buffer for SUSPECT and ATTACK events forwarded to the Rust agent.
#[map]
static EVENTS: RingBuf = RingBuf::with_byte_size(1 << 17, 0);

#[tracepoint]
pub fn guepard_shield(ctx: TracePointContext) -> u32 {
    let tgid = (bpf_get_current_pid_tgid() >> 32) as u32;
    let target_tgid = TARGET_TGID.get(0).copied().unwrap_or(0);
    if target_tgid != 0 && tgid != target_tgid {
        return 0;
    }

    let t0 = unsafe { bpf_ktime_get_ns() };
    let ret = match try_guepard_shield(ctx, tgid) {
        Ok(ret) => ret,
        Err(ret) => ret,
    };
    let elapsed = unsafe { bpf_ktime_get_ns() }.saturating_sub(t0);
    let bucket = ((elapsed / 100).min(31)) as u32;
    if bucket < 32 {
        if let Some(count) = LATENCY_HIST.get_ptr_mut(bucket) {
            unsafe { *count += 1 };
        }
    }
    ret
}

fn try_guepard_shield(ctx: TracePointContext, tgid: u32) -> Result<u32, u32> {
    let syscall_nr = unsafe { ctx.read_at::<i64>(8).map_err(|_| 0u32)? } as u32;

    let token = match SYSCALL_TO_TOKEN.get(syscall_nr) {
        Some(t) => *t,
        None => {
            PROCESS_STATE.insert(&tgid, &START_STATE, 0).ok();
            return Ok(0);
        }
    };

    if token == -1 {
        PROCESS_STATE.insert(&tgid, &START_STATE, 0).ok();
        return Ok(0);
    }

    let idx = token as usize;
    if idx >= 102 {
        return Ok(0);
    }

    let state = unsafe { PROCESS_STATE.get(&tgid) }.copied().unwrap_or(START_STATE);
    let row = match TRANSITION_TABLE.get(state) {
        Some(r) => r,
        None => return Ok(0),
    };

    let next_state = row.dst[idx];
    if next_state == -1 {
        // ATTACK: no valid transition for (state, token) — DFA rejects this sequence.
        emit_event(tgid, state, idx as u32, u32::MAX, KIND_ATTACK);
        PROCESS_STATE.insert(&tgid, &START_STATE, 0).ok();
        return Ok(0);
    }

    let next_u32 = next_state as u32;
    PROCESS_STATE.insert(&tgid, &next_u32, 0).ok();

    // SUSPECT: next state is a low-frequency state (rare but seen in training).
    let class = STATE_CLASS.get(next_u32).copied().unwrap_or(0);
    if class == 1 {
        emit_event(tgid, state, idx as u32, next_u32, KIND_SUSPECT);
    }

    Ok(0)
}

#[inline(always)]
fn emit_event(tgid: u32, state: u32, token: u32, next_state: u32, kind: u8) {
    let event = DfaEvent {
        tgid,
        state,
        token,
        next_state,
        kind,
        _pad: [0; 3],
    };
    EVENTS.output::<DfaEvent>(&event, 0).ok();
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[unsafe(link_section = "license")]
#[unsafe(no_mangle)]
static LICENSE: [u8; 13] = *b"Dual MIT/GPL\0";
