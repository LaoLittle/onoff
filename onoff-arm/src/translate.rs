use crate::context::{CpuContext, REG_LR};
use crate::inst::Inst as ArmInst;
use cranelift::codegen::ir::UserFuncName;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use onoff_core::error::{Error, Result};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Translator {
    pub module: JITModule,
    pub signature: Signature,
    pub counter: AtomicU32,
    debug: bool,
}

pub type BlockAbi = unsafe extern "C" fn(*mut CpuContext) -> u64;

impl Translator {
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("use_colocated_libcalls", "false");
        let _ = flag_builder.set("is_pic", "false");
        let _ = flag_builder.set("opt_level", "speed");
        let isa_builder = cranelift_native::builder().map_err(|_| Error::NotSupported)?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|_| Error::NotSupported)?;

        let module = JITModule::new(JITBuilder::with_isa(isa.clone(), default_libcall_names()));

        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(types::R64));
        signature.returns.push(AbiParam::new(types::I64));

        Ok(Self {
            module,
            signature,
            counter: AtomicU32::new(0),
            debug: false,
        })
    }

    #[inline]
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Translate a set of arm instructions into a basic block targeting the host architecture.
    ///
    /// The block represents a standard C function, and the definition is below:
    ///
    /// ```c
    /// void bb(context*);
    /// ```
    pub fn translate(&mut self, insts: &[ArmInst]) -> BlockAbi {
        let name = self.update_counter();
        let sig = self.signature.clone();
        let block = self
            .module
            .declare_function(&name.to_string(), Linkage::Local, &sig)
            .unwrap();

        let mut mcx = self.module.make_context();
        mcx.set_disasm(self.debug);
        mcx.func.signature = sig;
        mcx.func.name = UserFuncName::user(0, block.as_u32());
        let mut fcx = FunctionBuilderContext::new();

        // perform codegen
        {
            let mut bcx = FunctionBuilder::new(&mut mcx.func, &mut fcx);
            let block = bcx.create_block();
            bcx.switch_to_block(block);
            bcx.append_block_params_for_function_params(block);

            // ref mut CpuContext
            let ctx = bcx.block_params(block)[0];

            let mut saver = RegisterSaver::new(ctx);

            let mut save_pc = true;
            let mut pc_add = 0;
            for &inst in insts {
                match inst {
                    ArmInst::Adr { rd, label } => {
                        let pc = saver.load_pc(&mut bcx, pc_add);
                        let val = bcx.ins().iadd_imm(pc, label);
                        saver.store_register(rd, &mut bcx, val);
                    }
                    ArmInst::Adrp { rd, label } => {
                        let pc = saver.load_pc(&mut bcx, pc_add);
                        let hi = bcx.ins().band_imm(pc, !4095);
                        let val = bcx.ins().iadd_imm(hi, label);
                        saver.store_register(rd, &mut bcx, val);
                    }
                    ArmInst::Add { rd, rn, imm, sf } => {
                        let val = if sf {
                            let rn = saver.load_register(rn, &mut bcx);
                            let add = bcx.ins().iadd_imm(rn, imm as i64);
                            add
                        } else {
                            let rn = saver.load_register32(rn, &mut bcx);
                            let add = bcx.ins().iadd_imm(rn, imm as i64);
                            bcx.ins().uextend(types::I64, add)
                        };

                        saver.store_register(rd, &mut bcx, val);
                    }
                    ArmInst::Sub { rd, rn, imm, sf } => {
                        let val = if sf {
                            let rn = saver.load_register(rn, &mut bcx);
                            let add = bcx.ins().iadd_imm(rn, -(imm as i64));
                            add
                        } else {
                            let rn = saver.load_register32(rn, &mut bcx);
                            let add = bcx.ins().iadd_imm(rn, -(imm as i64));
                            bcx.ins().uextend(types::I64, add)
                        };

                        saver.store_register(rd, &mut bcx, val);
                    }
                    ArmInst::Ret { rn } => {
                        let dest = saver.load_register(rn, &mut bcx);

                        saver.store_pc(&mut bcx, dest);

                        save_pc = false;

                        break;
                    }
                    ArmInst::B { label } => {
                        let pc = saver.load_pc(&mut bcx, pc_add);
                        let dest = bcx.ins().iadd_imm(pc, label);

                        saver.store_pc(&mut bcx, dest);

                        save_pc = false;

                        break;
                    }
                    ArmInst::Bl { label } => {
                        let pc = saver.load_pc(&mut bcx, pc_add);
                        saver.store_register(REG_LR, &mut bcx, pc);

                        let dest = bcx.ins().iadd_imm(pc, label);
                        saver.store_pc(&mut bcx, dest);

                        save_pc = false;

                        break;
                    }
                }

                pc_add += 4;
            }

            if save_pc {
                let val = saver.load_pc(&mut bcx, pc_add);
                saver.store_pc(&mut bcx, val);
            }

            saver.save_registers(&mut bcx);

            let ret = bcx.ins().iconst(types::I64, 0);
            bcx.ins().return_(&[ret]);
            bcx.seal_all_blocks();
            bcx.finalize();
        }

        self.module.define_function(block, &mut mcx).unwrap();
        self.module.finalize_definitions().unwrap();

        if self.debug {
            println!(
                "--------------\n{}",
                mcx.compiled_code().unwrap().vcode.as_ref().unwrap()
            );
        }

        let ptr = self.module.get_finalized_function(block);

        // Safety: we do know the function is well-defined
        unsafe { core::mem::transmute::<_, BlockAbi>(ptr) }
    }

    #[inline]
    pub fn update_counter(&self) -> u32 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }
}

struct RegisterSaver {
    gprs: [bool; 32], // pointers to the registers
    pc: bool,
    cpu_ctx: Value,
}

const PC_VAR: u32 = 100;

impl RegisterSaver {
    pub fn new(cpu_ctx: Value) -> Self {
        Self {
            gprs: [false; 32],
            pc: false,
            cpu_ctx,
        }
    }

    pub fn load_register(&mut self, rn: u8, bcx: &mut FunctionBuilder) -> Value {
        let rds = rn as usize;

        let var = Variable::from_u32(rn as u32);
        if self.gprs[rds] {
            bcx.use_var(var)
        } else {
            let val = get_register(bcx, self.cpu_ctx, rn);
            bcx.declare_var(var, types::I64);
            bcx.def_var(var, val);
            self.gprs[rds] = true;
            val
        }
    }

    pub fn load_register32(&mut self, rn: u8, bcx: &mut FunctionBuilder) -> Value {
        let rds = rn as usize;

        let var = Variable::from_u32(rn as u32);
        if self.gprs[rds] {
            let r = bcx.use_var(var);
            bcx.ins().ireduce(types::I32, r)
        } else {
            let val = get_register32(bcx, self.cpu_ctx, rn);
            bcx.declare_var(var, types::I64);
            let extended = bcx.ins().uextend(types::I64, val);
            bcx.def_var(var, extended);
            self.gprs[rds] = true;
            val
        }
    }

    pub fn store_register(&mut self, rn: u8, bcx: &mut FunctionBuilder, val: Value) {
        let var = Variable::from_u32(rn as u32);
        if !self.gprs[rn as usize] {
            bcx.declare_var(var, types::I64);
        }

        bcx.def_var(var, val);
        self.gprs[rn as usize] = true;
    }

    pub fn store_register32(&mut self, rn: u8, bcx: &mut FunctionBuilder, val: Value) {
        let var = Variable::from_u32(rn as u32);
        if !self.gprs[rn as usize] {
            bcx.declare_var(var, types::I64);
        }

        let extended = bcx.ins().uextend(types::I64, val);
        bcx.def_var(var, extended);
        self.gprs[rn as usize] = true;
    }

    pub fn load_pc(&mut self, bcx: &mut FunctionBuilder, add: i64) -> Value {
        let var = Variable::from_u32(PC_VAR);
        if !self.pc {
            bcx.declare_var(var, types::I64);
            let pc = get_pc(bcx, self.cpu_ctx);
            bcx.def_var(var, pc);
        }

        self.pc = true;

        let rel = bcx.use_var(var);
        bcx.ins().iadd_imm(rel, add)
    }

    pub fn store_pc(&mut self, bcx: &mut FunctionBuilder, val: Value) {
        let var = Variable::from_u32(PC_VAR);
        if !self.pc {
            bcx.declare_var(var, types::I64);
        }

        self.pc = true;

        bcx.def_var(var, val);
    }

    pub fn save_registers(self, bcx: &mut FunctionBuilder) {
        for (rd, gpr) in self.gprs.into_iter().enumerate() {
            if !gpr {
                continue;
            }

            let val = bcx.use_var(Variable::from_u32(rd as u32));
            set_register(bcx, self.cpu_ctx, val, rd as u8);
        }

        if self.pc {
            let pc = bcx.use_var(Variable::from_u32(PC_VAR));
            set_pc(bcx, self.cpu_ctx, pc);
        }
    }
}

fn get_register(bcx: &mut FunctionBuilder, cpu_context: Value, rn: u8) -> Value {
    let rd = rn as usize;
    let offset = memoffset::offset_of!(CpuContext, gprs) + (core::mem::size_of::<u64>() * rd);

    bcx.ins().load(
        types::I64,
        MemFlags::trusted(),
        cpu_context,
        i32::try_from(offset).unwrap(),
    )
}

fn get_register32(bcx: &mut FunctionBuilder, cpu_context: Value, rn: u8) -> Value {
    let rd = rn as usize;
    let offset = memoffset::offset_of!(CpuContext, gprs)
        + (core::mem::size_of::<u64>() * rd)
        + core::mem::size_of::<u32>();

    bcx.ins().load(
        types::I32,
        MemFlags::trusted(),
        cpu_context,
        i32::try_from(offset).unwrap(),
    )
}

fn set_register(bcx: &mut FunctionBuilder, cpu_context: Value, val: Value, rn: u8) {
    let rd = rn as usize;
    let offset = memoffset::offset_of!(CpuContext, gprs) + (core::mem::size_of::<u64>() * rd);

    bcx.ins().store(
        MemFlags::trusted(),
        val,
        cpu_context,
        i32::try_from(offset).unwrap(),
    );
}

fn set_register32(bcx: &mut FunctionBuilder, cpu_context: Value, val: Value, rn: u8) {
    let rd = rn as usize;
    let offset = memoffset::offset_of!(CpuContext, gprs)
        + (core::mem::size_of::<u64>() * rd)
        + core::mem::size_of::<u32>();

    bcx.ins().store(
        MemFlags::trusted(),
        val,
        cpu_context,
        i32::try_from(offset).unwrap(),
    );
}

fn get_pc(bcx: &mut FunctionBuilder, cpu_context: Value) -> Value {
    let offset = memoffset::offset_of!(CpuContext, pc);

    let pc = bcx.ins().load(
        types::I64,
        MemFlags::trusted(),
        cpu_context,
        i32::try_from(offset).unwrap(),
    );

    pc
}

fn set_pc(bcx: &mut FunctionBuilder, cpu_context: Value, pc: Value) {
    let offset = memoffset::offset_of!(CpuContext, pc);

    bcx.ins().store(
        MemFlags::trusted(),
        pc,
        cpu_context,
        i32::try_from(offset).unwrap(),
    );
}

#[cfg(test)]
mod tests {
    use crate::context::CpuContext;
    use crate::inst::Inst;
    use crate::translate::Translator;

    #[test]
    fn gen() {
        let mut trans = Translator::new().unwrap();
        trans.set_debug(true);
        let fptr = trans.translate(&[
            Inst::Adr { rd: 2, label: 8 },
            Inst::Adr { rd: 0, label: 0 },
            Inst::B { label: 8 },
        ]);

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 16;

        unsafe {
            fptr(&mut cpu_ctx);
        }

        println!("{:?}", cpu_ctx);
        assert_eq!(cpu_ctx.pc, 16 + 4 + 4 + 8);
        assert_eq!(cpu_ctx.gprs[0], 16 + 4 + 0);
        assert_eq!(cpu_ctx.gprs[2], 16 + 8);

        let fptr = trans.translate(&[Inst::Adr { rd: 30, label: 32 }, Inst::Ret { rn: 30 }]);

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 32;

        unsafe {
            fptr(&mut cpu_ctx);
        }

        println!("{:?}", cpu_ctx);
        assert_eq!(cpu_ctx.pc, 32 + 32);
        assert_eq!(cpu_ctx.gprs[30], 32 + 32);

        let fptr = trans.translate(&[
            Inst::Add {
                rd: 0,
                rn: 0,
                imm: 12,
                sf: true,
            },
            Inst::Add {
                rd: 8,
                rn: 0,
                imm: 16,
                sf: false,
            },
            Inst::Sub {
                rd: 4,
                rn: 8,
                imm: 14,
                sf: true,
            },
            Inst::Sub {
                rd: 16,
                rn: 4,
                imm: 6,
                sf: false,
            },
        ]);

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 32;

        unsafe {
            fptr(&mut cpu_ctx);
        }

        println!("{:?}", cpu_ctx);
        assert_eq!(cpu_ctx.pc, 32 + 4 * 4);
        assert_eq!(cpu_ctx.gprs[0], 12);
        assert_eq!(cpu_ctx.gprs[4], 28 - 14);
        assert_eq!(cpu_ctx.gprs[8], 12 + 16);
        assert_eq!(cpu_ctx.gprs[16], 14 - 6);
    }
}
