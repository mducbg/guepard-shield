use std::collections::HashMap as StdHashMap;

use aya::maps::{Array, PerCpuArray};
use aya::programs::TracePoint;
use clap::Parser;
#[rustfmt::skip]
use log::{debug, warn};
use serde::Deserialize;
use tokio::signal;
use tokio::time::{interval, Duration, MissedTickBehavior};

use guepard_shield_common::TransitionRow;

#[derive(Parser)]
struct Opt {
    #[arg(long)]
    dfa: std::path::PathBuf,

    #[arg(long, default_value = "results/p4/e1_ebpf_histogram.txt")]
    latency_out: std::path::PathBuf,

    #[arg(long)]
    target_tgid: Option<u32>,
}

#[derive(Deserialize)]
struct DfaConfig {
    n_states: usize,
    n_tokens: usize,
    #[allow(dead_code)]
    start_state: u32,
    transition_table: Vec<Vec<i32>>,
    syscall_to_token: StdHashMap<String, i32>,
    /// Per-state classification: 0 = NORMAL, 1 = SUSPECT.
    /// Absent entries default to 0 (NORMAL).
    #[serde(default)]
    state_class: Vec<u8>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let opt = Opt::parse();

    env_logger::init();

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

    // Populate STATE_CLASS: 0 = NORMAL, 1 = SUSPECT.
    let mut state_class_map: Array<_, u8> = Array::try_from(
        ebpf.map_mut("STATE_CLASS")
            .ok_or_else(|| anyhow::anyhow!("map STATE_CLASS not found"))?,
    )?;
    let n_suspect: usize = config.state_class.iter().filter(|&&c| c == 1).count();
    for (i, &cls) in config.state_class.iter().enumerate() {
        state_class_map.set(i as u32, cls, 0)?;
    }
    println!(
        "State classes: {} NORMAL, {} SUSPECT",
        config.state_class.len().saturating_sub(n_suspect),
        n_suspect
    );

    let mut target_tgid: Array<_, u32> = Array::try_from(
        ebpf.map_mut("TARGET_TGID")
            .ok_or_else(|| anyhow::anyhow!("map TARGET_TGID not found"))?,
    )?;
    target_tgid.set(0, opt.target_tgid.unwrap_or(0), 0)?;

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
        "DFA loaded: {} states, {} transitions. Monitoring; printing NORMAL/SUSPECT/ATTACK every second.",
        config.n_states, n_transitions
    );

    let mut previous_counts = [0u64; 3];
    let mut elapsed_seconds = 0u64;
    let mut summary_timer = interval(Duration::from_secs(1));
    summary_timer.set_missed_tick_behavior(MissedTickBehavior::Skip);
    summary_timer.tick().await;
    loop {
        tokio::select! {
            _ = signal::ctrl_c() => break,
            _ = summary_timer.tick() => {
                let counts = read_class_counts(&ebpf)?;
                elapsed_seconds += 1;
                print_class_summary(elapsed_seconds, previous_counts, counts);
                previous_counts = counts;
            }
        }
    }
    println!("Exiting...");

    // Dump latency histogram collected during run.
    if let Some(map_ref) = ebpf.map("LATENCY_HIST") {
        if let Ok(hist) = PerCpuArray::<_, u64>::try_from(map_ref) {
            let mut counts = [0u64; 32];
            for bucket in 0..32u32 {
                if let Ok(per_cpu) = hist.get(&bucket, 0) {
                    counts[bucket as usize] = per_cpu.iter().sum();
                }
            }
            let report = format_latency_histogram(&counts);
            print!("{report}");
            if let Some(parent) = opt.latency_out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&opt.latency_out, report)?;
            println!("Latency histogram saved to {}", opt.latency_out.display());
        }
    }

    Ok(())
}

fn read_class_counts(ebpf: &aya::Ebpf) -> anyhow::Result<[u64; 3]> {
    let map_ref = ebpf
        .map("CLASS_COUNTS")
        .ok_or_else(|| anyhow::anyhow!("map CLASS_COUNTS not found"))?;
    let counts_map = PerCpuArray::<_, u64>::try_from(map_ref)?;
    let mut counts = [0u64; 3];
    for class in 0..3u32 {
        counts[class as usize] = counts_map.get(&class, 0)?.iter().sum();
    }
    Ok(counts)
}

fn print_class_summary(elapsed_seconds: u64, previous: [u64; 3], current: [u64; 3]) {
    println!("{}", format_class_summary(elapsed_seconds, previous, current));
}

fn format_class_summary(
    elapsed_seconds: u64,
    previous: [u64; 3],
    current: [u64; 3],
) -> String {
    let normal = current[0].saturating_sub(previous[0]);
    let suspect = current[1].saturating_sub(previous[1]);
    let attack = current[2].saturating_sub(previous[2]);
    format!(
        "[live {:02}s] total={} | NORMAL={} | SUSPECT={} | ATTACK={}",
        elapsed_seconds,
        normal + suspect + attack,
        normal,
        suspect,
        attack,
    )
}

fn format_latency_histogram(counts: &[u64; 32]) -> String {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return "[latency] no samples collected\n".to_string();
    }

    let mut report = String::new();
    report.push_str(&format!("\n[latency histogram] {} samples\n", total));
    report.push_str(&format!(
        "{:>12}  {:>10}  {:>10}\n",
        "bucket_ns", "count", "cumul%"
    ));

    let mut cumul = 0u64;
    for (i, &c) in counts.iter().enumerate() {
        cumul += c;
        let bucket_label = if i < 31 {
            format!("{:>6}", i * 100)
        } else {
            ">=3100".to_string()
        };
        report.push_str(&format!(
            "{:>12}  {:>10}  {:>9.2}%\n",
            bucket_label,
            c,
            cumul as f64 / total as f64 * 100.0
        ));
    }

    for (label, threshold) in [("p50", 0.50f64), ("p99", 0.99), ("p999", 0.999)] {
        let target = (total as f64 * threshold).ceil() as u64;
        let mut acc = 0u64;
        for (i, &c) in counts.iter().enumerate() {
            acc += c;
            if acc >= target {
                let lo = i * 100;
                if i < 31 {
                    report.push_str(&format!("{}: {}-{} ns\n", label, lo, (i + 1) * 100));
                } else {
                    report.push_str(&format!("{}: {}-inf ns\n", label, lo));
                }
                break;
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::format_class_summary;

    #[test]
    fn class_summary_reports_counter_delta() {
        assert_eq!(
            format_class_summary(12, [10, 20, 30], [15, 22, 33]),
            "[live 12s] total=10 | NORMAL=5 | SUSPECT=2 | ATTACK=3",
        );
    }
}
