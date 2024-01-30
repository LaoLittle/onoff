use crate::kernel::process::KernelProcess;
use sharded_slab::Slab;
use std::ops::Deref;

pub struct KernelContext {
    pub processes: Slab<KernelProcess>,
    current_process: usize,
}

impl KernelContext {
    // Initial kip id
    pub const INIT_KIP_ID: usize = 0x1;
    // Initial process id
    pub const INIT_PROCESS_ID: usize = 0x51;
    /// Supervisor call count
    pub const SVC_COUNT: usize = 0xC0;
    /// Memory block allocator size
    pub const MEM_BLOCK_ALLOCATOR_SIZE: usize = 0x2710;
    pub const USER_SLAB_HEAP_SIZE: usize = 0x3de000;
    pub const COUNTER_FREQUENCY: usize = 19200000;

    pub fn new() -> Self {
        let processes = Slab::new();

        Self {
            processes,
            current_process: Self::INIT_PROCESS_ID,
        }
    }

    pub fn insert_process(&self, proc: KernelProcess) -> Option<usize> {
        self.processes
            .insert(proc)?
            .checked_sub(Self::INIT_PROCESS_ID)
    }

    pub fn scoped_process<T>(&self, id: usize, f: impl FnOnce(&KernelProcess) -> T) -> Option<T> {
        let mut proc = self.processes.get(id.wrapping_sub(Self::INIT_PROCESS_ID))?;

        Some(f(&proc))
    }

    pub fn run(&self) {}
}
