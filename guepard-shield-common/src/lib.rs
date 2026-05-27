#![no_std]

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TransitionRow {
    pub dst: [i32; 102],
}

#[cfg(feature = "user")]
unsafe impl aya::Pod for TransitionRow {}

/// Event emitted to userspace on a notable DFA transition.
/// kind: 1 = SUSPECT (next state has low training frequency),
///       2 = ATTACK  (no transition exists for this (state, token) pair).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct DfaEvent {
    pub tgid: u32,
    pub state: u32,
    pub token: u32,
    pub next_state: u32,
    pub kind: u8,
    pub _pad: [u8; 3],
}

#[cfg(feature = "user")]
unsafe impl aya::Pod for DfaEvent {}
