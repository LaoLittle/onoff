use crate::inst::{Inst, InstDecoder};
use crate::mem::{Memory, PageMemory};
use onoff_core::error::Result;

#[derive(Debug)]
pub struct Cpu {
    context: CpuContext,
    memory: PageMemory,
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
        }
    }

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

    pub fn execute_single_inst(&mut self, inst: Inst) -> Option<Break> {
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
            _ => todo!(),
        }

        self.context.pc += 4;

        None
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
    pub fn context(&self) -> &CpuContext {
        &self.context
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
        let inst = [0x01, 0x00, 0x00, 0x14, 0x60, 0x00, 0x00, 0x10];

        let mut cpu = Cpu::new();

        cpu.memory_mut().write_exact(&inst, 0).unwrap();

        cpu.set_pc(0);
        cpu.execute(2).unwrap();

        println!("{:#?}", cpu);
    }
}
