use crate::inst::{Inst, InstDecoder};
use crate::mem::{Memory, PageMemory};
use crate::translate::{BlockAbi, ExecutionReturn, InterruptType, Translator};
use lru::LruCache;
use onoff_core::error::Result;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};
use std::num::{NonZeroU16, NonZeroUsize};

pub struct Cpu {
    context: CpuContext,
    memory: PageMemory,
    translator: Translator,
    code_cache: BTreeMap<u64, BlockAbi>,

    lru: LruCache<u64, BlockAbi>,
}

impl Cpu {
    #[inline]
    pub fn new() -> Self {
        Self {
            context: CpuContext::new(),
            memory: PageMemory::new(),
            translator: Translator::new().unwrap(),
            code_cache: BTreeMap::new(),
            lru: LruCache::new(NonZeroUsize::new(512).unwrap()),
        }
    }

    /// S: Steps
    pub fn execute<const S: usize>(&mut self) -> Result<CpuStatus> {
        assert!(S <= 64, "too many steps!");

        let pc = self.context.pc();

        if let Some(&block) = self.lru.get(&pc) {
            let status = unsafe { block(&mut self.context) };

            return Ok(CpuStatus::from_u32(status));
        }

        if let Some(&block) = self.code_cache.get(&pc) {
            let status = unsafe { block(&mut self.context) };

            return Ok(CpuStatus::from_u32(status));
        }

        // no compiled code found, let's compile.
        let mut v = SmallVec::<Inst, S>::new();
        for i in 0..S as u64 {
            let Ok(mem) = self.memory.read32(pc + i * 4) else {
                break;
            };

            let mem = mem.to_le_bytes();

            let mut decoder = InstDecoder::new(mem.as_slice());
            let inst = decoder.decode_inst()?;

            v.push(inst);
        }

        let block = self.translator.translate(&v);

        let status = unsafe { block(&mut self.context) };

        if let Some((k, v)) = self.lru.push(pc, block) {
            self.code_cache.insert(k, v);
        }

        Ok(CpuStatus::from_u32(status))
    }

    #[inline]
    pub fn memory_mut(&mut self) -> &mut PageMemory {
        &mut self.memory
    }

    #[inline]
    pub fn set_pc(&mut self, base: u64) {
        self.context.pc = base;
    }

    #[inline]
    pub fn set_sp(&mut self, top: u64) {
        *self.context.gpr_mut(REG_SP) = top;
    }

    #[inline]
    pub fn context(&self) -> &CpuContext {
        &self.context
    }
}

impl Debug for Cpu {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cpu")
            .field("context", &self.context)
            .finish()
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct CpuContext {
    // general purpose registers
    pub gprs: [u64; 32],
    // program counter
    pub pc: u64,

    // process state
    pub pstate: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interrupt {
    Svc,
    Udf,
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuStatus {
    pub break_type: Option<Interrupt>,
    pub val: u16,
}

impl CpuStatus {
    pub fn normal() -> Self {
        Self {
            break_type: None,
            val: 0,
        }
    }

    pub fn from_u32(ret: u32) -> Self {
        let exec = ExecutionReturn::from_u32(ret);

        Self::from(exec)
    }
}

impl From<ExecutionReturn> for CpuStatus {
    fn from(ret: ExecutionReturn) -> Self {
        Self {
            break_type: match ret.ty() {
                InterruptType::None => None,
                InterruptType::Udf => Some(Interrupt::Udf),
                InterruptType::Svc => Some(Interrupt::Svc),
            },
            val: ret.val(),
        }
    }
}

// special registers' index
pub const REG_FP: u8 = 29;
pub const REG_LR: u8 = 30;
pub const REG_SP: u8 = 31;

pub const REG_ZR: u8 = 31;

impl CpuContext {
    #[inline]
    pub fn new() -> Self {
        Self {
            gprs: [0; 32],
            pc: 0,
            pstate: 0,
        }
    }

    #[inline]
    pub fn gpr_mut(&mut self, rd: u8) -> &mut u64 {
        assert!(rd <= 31, "destination register must less than 31");
        &mut self.gprs[rd as usize]
    }

    #[inline]
    pub fn fp(&self) -> u64 {
        self.gprs[29]
    }

    #[inline]
    pub fn lr(&self) -> u64 {
        self.gprs[30]
    }

    #[inline]
    pub fn lr_mut(&mut self) -> &mut u64 {
        &mut self.gprs[30]
    }

    #[inline]
    pub fn sp(&self) -> u64 {
        self.gprs[31]
    }

    #[inline]
    pub fn pc(&self) -> u64 {
        self.pc
    }

    #[inline]
    pub fn pc_mut(&mut self) -> &mut u64 {
        &mut self.pc
    }
}

#[cfg(test)]
mod tests {
    use crate::context::Cpu;

    #[test]
    fn execute() {
        // Adr { rd: 0, label: 12 }
        // B { label: 4 }
        let inst = [0x60, 0x00, 0x00, 0x10, 0x01, 0x00, 0x00, 0x14];

        let mut cpu = Cpu::new();

        cpu.memory_mut().write_exact(&inst, 0x10000).unwrap();

        cpu.set_pc(0x10000);
        let status = cpu.execute::<2>().unwrap();

        dbg!(status);

        println!("{:?}", cpu);

        assert_eq!(cpu.context().gprs[0], 0x10000 + 12);
        assert_eq!(cpu.context().pc(), 0x10000 + 4 + 4);

        cpu.set_pc(0x10000);
        let status = cpu.execute::<2>().unwrap();

        dbg!(status);

        println!("{:?}", cpu);

        assert_eq!(cpu.context().gprs[0], 0x10000 + 12);
        assert_eq!(cpu.context().pc(), 0x10000 + 4 + 4);
    }
}
