#![no_std]

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TransitionRow {
    pub dst: [i32; 102],
}

#[cfg(feature = "user")]
unsafe impl aya::Pod for TransitionRow {}
