use crate::inst::{Inst, InstDecoder};
use crate::mem::{Memory, PageMemory};
use crate::translate::{BlockAbi, Translator};
use lru::LruCache;
use onoff_core::error::Result;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroUsize;

pub struct Cpu {
    context: CpuContext,
    memory: PageMemory,
    translator: Translator,
    code_cache: BTreeMap<u64, BlockAbi>,

    lru: LruCache<u64, BlockAbi>,
}

pub enum Break {
    Svc,
    Eof,
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
    pub fn execute<const S: usize>(&mut self) -> Result<Option<Break>> {
        assert!(S <= 64, "too many steps!");

        let pc = self.context.pc();

        if let Some(&block) = self.lru.get(&pc) {
            unsafe {
                block(&mut self.context);
            }
            return Ok(None);
        }

        if let Some(&block) = self.code_cache.get(&pc) {
            unsafe {
                block(&mut self.context);
            }
            return Ok(None);
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

        unsafe {
            block(&mut self.context);
        }

        if let Some((k, v)) = self.lru.push(pc, block) {
            self.code_cache.insert(k, v);
        }

        Ok(None)
    }

    /*
    pub fn execute(&mut self, steps: u32) -> Result<Option<Break>> {
        for _ in 0..steps {
            let pc = self.context.pc;
            let inst = self.memory.read32(pc)?;
            let bytes = inst.to_le_bytes();
            let mut decoder = InstDecoder::new(bytes.as_slice());

            match self.execute_single_inst(decoder.decode_inst()?) {
                None => {}
                Some(b) => return Ok(Some(b)),
            }
        }

        Ok(None)
    }

     */

    /*pub fn execute_single_inst(&mut self, inst: Inst) -> Option<Break> {
        match inst {
            Inst::B { label } => {
                let pc = self.context.pc_mut();
                *pc = pc.wrapping_add_signed(label).wrapping_sub(4);
            }
            Inst::Bl { label } => {
                *self.context.lr_mut() = self.context.pc();
                let pc = self.context.pc_mut();
                *pc = pc.wrapping_add_signed(label).wrapping_sub(4);
            }
            Inst::Adr { rd, label } => {
                let pc = self.context.pc();
                let rd = self.context.gpr_mut(rd);
                *rd = pc.wrapping_add_signed(label);
            }
            Inst::Adrp { rd, label } => {
                let pc = self.context.pc();
                let rd = self.context.gpr_mut(rd);
                *rd = (pc & !4095).wrapping_add_signed(label);
            }
            _ => {},
        }

        self.context.pc += 4;

        None
    }

     */

    #[inline]
    pub fn memory_mut(&mut self) -> &mut PageMemory {
        &mut self.memory
    }

    pub fn check_cache(&mut self, addr: u64) {
        let mut addrs = SmallVec::<u64, 16>::new();
        for (&base, _) in self.code_cache.iter() {
            if (base..base + 64 * 4).contains(&addr) {
                addrs.push(base);
            }
        }

        for addr in addrs {
            self.code_cache.remove(&addr);
        }
    }

    #[inline]
    pub fn set_pc(&mut self, base: u64) {
        self.context.pc = base;
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
#[repr(C)] // for jit purpose
pub struct CpuContext {
    pub gprs: [u64; 32],
    pub pc: u64,
}

pub const REG_FP: u8 = 29;
pub const REG_LR: u8 = 30;
pub const REG_SP: u8 = 31;

impl CpuContext {
    #[inline]
    pub fn new() -> Self {
        Self {
            gprs: [0; 32],
            pc: 0,
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
        cpu.execute::<2>().unwrap();

        println!("{:?}", cpu);

        assert_eq!(cpu.context().gprs[0], 0x10000 + 12);
        assert_eq!(cpu.context().pc(), 0x10000 + 4 + 4);

        cpu.set_pc(0x10000);
        cpu.execute::<2>().unwrap();

        println!("{:?}", cpu);

        assert_eq!(cpu.context().gprs[0], 0x10000 + 12);
        assert_eq!(cpu.context().pc(), 0x10000 + 4 + 4);
    }
}
