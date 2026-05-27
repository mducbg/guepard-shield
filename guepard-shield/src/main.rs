use std::collections::HashMap as StdHashMap;

use aya::maps::Array;
use aya::programs::TracePoint;
use clap::Parser;
#[rustfmt::skip]
use log::{debug, warn};
use serde::Deserialize;
use tokio::signal;

use guepard_shield_common::TransitionRow;

#[derive(Parser)]
struct Opt {
    #[arg(long)]
    dfa: std::path::PathBuf,
}

#[derive(Deserialize)]
struct DfaConfig {
    n_states: usize,
    n_tokens: usize,
    #[allow(dead_code)]
    start_state: u32,
    transition_table: Vec<Vec<i32>>,
    syscall_to_token: StdHashMap<String, i32>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let opt = Opt::parse();

    env_logger::init();

    // Bump the memlock rlimit. This is needed for older kernels that don't use the
    // new memcg based accounting, see https://lwn.net/Articles/837122/
    let rlim = libc::rlimit {
        rlim_cur: libc::RLIM_INFINITY,
        rlim_max: libc::RLIM_INFINITY,
    };
    let ret = unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &rlim) };
    if ret != 0 {
        debug!("remove limit on locked memory failed, ret is: {ret}");
    }

    let mut ebpf = aya::Ebpf::load(aya::include_bytes_aligned!(concat!(
        env!("OUT_DIR"),
        "/guepard-shield"
    )))?;
    match aya_log::EbpfLogger::init(&mut ebpf) {
        Err(e) => {
            warn!("failed to initialize eBPF logger: {e}");
        }
        Ok(logger) => {
            let mut logger =
                tokio::io::unix::AsyncFd::with_interest(logger, tokio::io::Interest::READABLE)?;
            tokio::task::spawn(async move {
                loop {
                    let mut guard = logger.readable_mut().await.unwrap();
                    guard.get_inner_mut().flush();
                    guard.clear_ready();
                }
            });
        }
    }

    // Load DFA config and populate maps before attaching the tracepoint.
    let config: DfaConfig =
        serde_json::from_reader(std::fs::File::open(&opt.dfa)?)?;
    anyhow::ensure!(
        config.n_states == 64,
        "expected 64 states, got {}",
        config.n_states
    );
    anyhow::ensure!(
        config.n_tokens == 102,
        "expected 102 tokens, got {}",
        config.n_tokens
    );
    anyhow::ensure!(
        config.transition_table.len() == config.n_states,
        "transition_table length {} != n_states {}",
        config.transition_table.len(),
        config.n_states
    );

    let mut transition_table: Array<_, TransitionRow> = Array::try_from(
        ebpf.map_mut("TRANSITION_TABLE")
            .ok_or_else(|| anyhow::anyhow!("map TRANSITION_TABLE not found"))?,
    )?;
    for (i, row_vec) in config.transition_table.iter().enumerate() {
        let dst: [i32; 102] = row_vec
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("row {} has wrong length {}", i, row_vec.len()))?;
        transition_table.set(i as u32, TransitionRow { dst }, 0)?;
    }

    let mut syscall_to_token: Array<_, i32> = Array::try_from(
        ebpf.map_mut("SYSCALL_TO_TOKEN")
            .ok_or_else(|| anyhow::anyhow!("map SYSCALL_TO_TOKEN not found"))?,
    )?;
    for (nr_str, token_id) in &config.syscall_to_token {
        let nr: u32 = nr_str.parse()?;
        syscall_to_token.set(nr, *token_id, 0)?;
    }

    let n_transitions: usize = config
        .transition_table
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v != -1)
        .count();

    let program: &mut TracePoint = ebpf.program_mut("guepard_shield").unwrap().try_into()?;
    program.load()?;
    program.attach("raw_syscalls", "sys_enter")?;

    println!(
        "DFA loaded: {} states, {} transitions. Monitoring...",
        config.n_states, n_transitions
    );

    let ctrl_c = signal::ctrl_c();
    ctrl_c.await?;
    println!("Exiting...");

    Ok(())
}
