#![no_std]
#![no_main]

use aya_ebpf::{
    helpers::bpf_get_current_pid_tgid,
    macros::{map, tracepoint},
    maps::{Array, HashMap},
    programs::TracePointContext,
};
use aya_log_ebpf::warn;
use guepard_shield_common::TransitionRow;

const START_STATE: u32 = 59;

#[map]
static TRANSITION_TABLE: Array<TransitionRow> = Array::with_max_entries(64, 0);

#[map]
static SYSCALL_TO_TOKEN: Array<i32> = Array::with_max_entries(512, 0);

#[map]
static PROCESS_STATE: HashMap<u32, u32> = HashMap::with_max_entries(1024, 0);

#[tracepoint]
pub fn guepard_shield(ctx: TracePointContext) -> u32 {
    match try_guepard_shield(ctx) {
        Ok(ret) => ret,
        Err(ret) => ret,
    }
}

fn try_guepard_shield(ctx: TracePointContext) -> Result<u32, u32> {
    let tgid = (bpf_get_current_pid_tgid() >> 32) as u32;
    let syscall_nr = unsafe { ctx.read_at::<i64>(8).map_err(|_| 0u32)? } as u32;

    let token = match SYSCALL_TO_TOKEN.get(syscall_nr) {
        Some(t) => *t,
        None => {
            warn!(&ctx, "[ALERT] unknown syscall {} pid={}", syscall_nr, tgid);
            PROCESS_STATE.insert(&tgid, &START_STATE, 0).ok();
            return Ok(0);
        }
    };

    if token == -1 {
        warn!(&ctx, "[ALERT] unknown syscall {} pid={}", syscall_nr, tgid);
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
        warn!(
            &ctx,
            "[ALERT] reject: state={} syscall={} pid={}", state, syscall_nr, tgid
        );
        PROCESS_STATE.insert(&tgid, &START_STATE, 0).ok();
        return Ok(0);
    }

    PROCESS_STATE.insert(&tgid, &(next_state as u32), 0).ok();
    Ok(0)
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[unsafe(link_section = "license")]
#[unsafe(no_mangle)]
static LICENSE: [u8; 13] = *b"Dual MIT/GPL\0";
