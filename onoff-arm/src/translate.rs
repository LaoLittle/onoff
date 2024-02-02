use crate::context::{CpuContext, REG_LR};
use crate::inst::{Inst as ArmInst, Operand, ShiftType};
use cranelift::codegen::ir::UserFuncName;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use modular_bitfield::{bitfield, BitfieldSpecifier};
use onoff_core::error::{Error, Result};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Translator {
    pub module: JITModule,
    pub signature: Signature,
    pub counter: AtomicU32,
    debug: bool,
}

pub type BlockAbi = unsafe extern "C" fn(*mut CpuContext) -> u32;

#[derive(BitfieldSpecifier, Debug, PartialEq, Eq)]
#[bits = 16]
pub enum InterruptType {
    None = 0,
    Udf = 1,
    Svc = 2,
}

#[bitfield]
pub struct ExecutionReturn {
    pub ty: InterruptType,
    pub val: u16,
}

impl ExecutionReturn {
    #[inline]
    pub const fn into_u32(self) -> u32 {
        u32::from_ne_bytes(self.into_bytes())
    }

    #[inline]
    pub const fn from_u32(i: u32) -> Self {
        Self::from_bytes(i.to_ne_bytes())
    }
}

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
        signature.returns.push(AbiParam::new(types::I32));

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
            let mut ret = ExecutionReturn::new()
                .with_ty(InterruptType::None)
                .with_val(0);

            for &inst in insts {
                match inst {
                    ArmInst::Udf { imm } => {
                        ret.set_ty(InterruptType::Udf);
                        ret.set_val(imm);

                        break;
                    }
                    ArmInst::Adr { rd, label } => {
                        let pc = saver.load_pc(&mut bcx);
                        let val = bcx.ins().iadd_imm(pc, label);
                        saver.store_register(rd, &mut bcx, val);
                    }
                    ArmInst::Adrp { rd, label } => {
                        let pc = saver.load_pc(&mut bcx);
                        let hi = bcx.ins().band_imm(pc, !4095);
                        let val = bcx.ins().iadd_imm(hi, label);
                        saver.store_register(rd, &mut bcx, val);
                    }
                    ArmInst::Add { rd, rn, op2, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let op2 = saver.load_operand(&mut bcx, op2, sf);
                        let val = bcx.ins().iadd(rn, op2);

                        saver.store_register_with_sf(rd, &mut bcx, val, sf);
                    }
                    ArmInst::Adds { rd, rn, op2, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let op2 = saver.load_operand(&mut bcx, op2, sf);

                        let val = bcx.ins().iadd(rn, op2);

                        saver.store_register_with_sf(rd, &mut bcx, val, sf);

                        saver.check_alu_pstate(&mut bcx, val, rn, op2, sf, |bcx, us, x, y| {
                            if us {
                                bcx.ins().uadd_overflow(x, y).1
                            } else {
                                bcx.ins().sadd_overflow(x, y).1
                            }
                        });
                    }
                    ArmInst::Sub { rd, rn, op2, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let op2 = saver.load_operand(&mut bcx, op2, sf);
                        let neg = bcx.ins().ineg(op2);

                        let val = bcx.ins().iadd(rn, neg);

                        saver.store_register_with_sf(rd, &mut bcx, val, sf);
                    }
                    ArmInst::Subs { rd, rn, op2, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let op2 = saver.load_operand(&mut bcx, op2, sf);
                        let neg = bcx.ins().ineg(op2);

                        let val = bcx.ins().iadd(rn, neg);

                        saver.store_register_with_sf(rd, &mut bcx, val, sf);

                        saver.check_alu_pstate(&mut bcx, val, rn, op2, sf, |bcx, us, x, y| {
                            if us {
                                bcx.ins().usub_overflow(x, y).1
                            } else {
                                bcx.ins().ssub_overflow(x, y).1
                            }
                        });
                    }
                    ArmInst::Svc { imm } => {
                        ret.set_ty(InterruptType::Svc);
                        ret.set_val(imm);

                        break;
                    }
                    ArmInst::Nop => {
                        let i = bcx.ins().iconst(types::I64, 0);
                        let i2 = bcx.ins().iconst(types::I64, 1);
                        let cin = bcx.ins().iconst(types::I8, 2);
                        bcx.ins().iadd_carry(i, i2, cin);

                        bcx.ins().nop();
                    }
                    // ret will give a hint to cpu
                    ArmInst::Br { rn } | ArmInst::Ret { rn } => {
                        let dest = saver.load_register(rn, &mut bcx);
                        saver.store_pc(&mut bcx, dest);
                        save_pc = false;

                        break;
                    }
                    ArmInst::Blr { rn } => {
                        let pc = saver.load_pc(&mut bcx);
                        saver.store_register(REG_LR, &mut bcx, pc);

                        let dest = saver.load_register(rn, &mut bcx);
                        saver.store_pc(&mut bcx, dest);
                        save_pc = false;

                        break;
                    }
                    ArmInst::B { label } => {
                        let pc = saver.load_pc(&mut bcx);
                        let dest = bcx.ins().iadd_imm(pc, label);
                        saver.store_pc(&mut bcx, dest);
                        save_pc = false;

                        break;
                    }
                    ArmInst::Bl { label } => {
                        let pc = saver.load_pc(&mut bcx);
                        saver.store_register(REG_LR, &mut bcx, pc);

                        let dest = bcx.ins().iadd_imm(pc, label);
                        saver.store_pc(&mut bcx, dest);

                        save_pc = false;

                        break;
                    }
                }

                saver.update_pc();
            }

            if save_pc {
                let val = saver.load_pc(&mut bcx);
                saver.store_pc(&mut bcx, val);
            }

            saver.save_registers(&mut bcx);

            let ret = bcx.ins().iconst(types::I32, ret.into_u32() as i64);
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

const N_MASK: i64 = 0b10000000;
const Z_MASK: i64 = 0b01000000;
const C_MASK: i64 = 0b00100000;
const V_MASK: i64 = 0b00010000;

struct PState {
    n: bool,
    z: bool,
    c: bool,
    v: bool,
}

impl PState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            n: false,
            z: false,
            c: false,
            v: false,
        }
    }
}

struct RegisterSaver {
    gprs: [bool; 32], // pointers to the registers
    pc: bool,
    pstate: Option<PState>,
    cpu_ctx: Value,
    pc_offset: i64,
}

const PC_VAR: u32 = 100;
const PSTATE_VAR: u32 = 101;
const N_VAR: u32 = 110;
const Z_VAR: u32 = 111;
const C_VAR: u32 = 112;
const V_VAR: u32 = 113;

impl RegisterSaver {
    pub fn new(cpu_ctx: Value) -> Self {
        Self {
            gprs: [false; 32],
            pc: false,
            pstate: None,
            cpu_ctx,
            pc_offset: 0,
        }
    }

    #[inline]
    pub fn load_register_with_sf(&mut self, rn: u8, bcx: &mut FunctionBuilder, sf: bool) -> Value {
        if sf {
            self.load_register(rn, bcx)
        } else {
            self.load_register32(rn, bcx)
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

    pub fn load_operand(&mut self, bcx: &mut FunctionBuilder, operand: Operand, sf: bool) -> Value {
        let ty = if sf { types::I64 } else { types::I32 };

        match operand {
            Operand::Imm(imm) => {
                let imm = imm as i64;
                bcx.ins().iconst(ty, imm)
            }
            Operand::ShiftedReg {
                rm,
                shift_type,
                amount,
            } => {
                let amount = amount as i64;
                let reg = if sf {
                    self.load_register(rm, bcx)
                } else {
                    self.load_register32(rm, bcx)
                };

                match shift_type {
                    ShiftType::Lsl => bcx.ins().ishl_imm(reg, amount),
                    ShiftType::Lsr => bcx.ins().ushr_imm(reg, amount),
                    ShiftType::Asr => bcx.ins().sshr_imm(reg, amount),
                    ShiftType::Ror => bcx.ins().rotr_imm(reg, amount),
                }
            }
        }
    }

    pub fn store_register_with_sf(
        &mut self,
        rn: u8,
        bcx: &mut FunctionBuilder,
        mut val: Value,
        sf: bool,
    ) {
        if sf {
            self.store_register(rn, bcx, val);
        } else {
            self.store_register32(rn, bcx, val);
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

    pub fn load_pc(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(PC_VAR);
        if !self.pc {
            bcx.declare_var(var, types::I64);
            let pc = get_pc(bcx, self.cpu_ctx);
            bcx.def_var(var, pc);
            self.pc = true;
        }

        let rel = bcx.use_var(var);
        bcx.ins().iadd_imm(rel, self.pc_offset)
    }

    pub fn store_pc(&mut self, bcx: &mut FunctionBuilder, val: Value) {
        let var = Variable::from_u32(PC_VAR);
        if !self.pc {
            bcx.declare_var(var, types::I64);
            self.pc = true;
        }

        bcx.def_var(var, val);
    }

    #[inline]
    pub fn update_pc(&mut self) {
        self.pc_offset += 4;
    }

    /// load the whole pstate
    pub fn load_pstate(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(PSTATE_VAR);
        if self.pstate.is_none() {
            bcx.declare_var(var, types::I8);
            bcx.declare_var(Variable::from_u32(N_VAR), types::I8);
            bcx.declare_var(Variable::from_u32(Z_VAR), types::I8);
            bcx.declare_var(Variable::from_u32(C_VAR), types::I8);
            bcx.declare_var(Variable::from_u32(V_VAR), types::I8);
            let pstate = get_pstate(bcx, self.cpu_ctx);
            bcx.def_var(var, pstate);
            self.pstate = Some(PState::new());
        }

        bcx.use_var(var)
    }

    /// store the whole pstate
    pub fn store_pstate(&mut self, bcx: &mut FunctionBuilder, pstate: Value) {
        let var = Variable::from_u32(PSTATE_VAR);
        if self.pstate.is_none() {
            bcx.declare_var(var, types::I8);
            bcx.declare_var(Variable::from_u32(N_VAR), types::I8);
            bcx.declare_var(Variable::from_u32(Z_VAR), types::I8);
            bcx.declare_var(Variable::from_u32(C_VAR), types::I8);
            bcx.declare_var(Variable::from_u32(V_VAR), types::I8);
            self.pstate = Some(PState::new());
        }

        bcx.def_var(var, pstate);
    }

    pub fn load_n(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(N_VAR);

        let pval = self.load_pstate(bcx);

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        if pstate.n {
            bcx.use_var(var)
        } else {
            let bit = bcx.ins().band_imm(pval, N_MASK);

            bcx.def_var(var, bit);
            pstate.n = true;
            bit
        }
    }

    pub fn store_n(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        let var = Variable::from_u32(N_VAR);

        if self.pstate.is_none() {
            self.load_pstate(bcx);
        }

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        pstate.n = true;

        bcx.def_var(var, n);
    }

    pub fn load_z(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(Z_VAR);

        let pval = self.load_pstate(bcx);

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        if pstate.z {
            bcx.use_var(var)
        } else {
            let bit = bcx.ins().band_imm(pval, Z_MASK);

            bcx.def_var(var, bit);
            pstate.z = true;
            bit
        }
    }

    pub fn store_z(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        let var = Variable::from_u32(Z_VAR);

        if self.pstate.is_none() {
            self.load_pstate(bcx);
        }

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        pstate.z = true;

        bcx.def_var(var, n);
    }

    pub fn load_c(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(C_VAR);

        let pval = self.load_pstate(bcx);

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        if pstate.c {
            bcx.use_var(var)
        } else {
            let bit = bcx.ins().band_imm(pval, C_MASK);

            bcx.def_var(var, bit);
            pstate.c = true;
            bit
        }
    }

    pub fn store_c(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        let var = Variable::from_u32(C_VAR);

        if self.pstate.is_none() {
            self.load_pstate(bcx);
        }

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        pstate.c = true;

        bcx.def_var(var, n);
    }

    pub fn load_v(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(V_VAR);

        let pval = self.load_pstate(bcx);

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        if pstate.v {
            bcx.use_var(var)
        } else {
            let bit = bcx.ins().band_imm(pval, V_MASK);

            bcx.def_var(var, bit);
            pstate.v = true;
            bit
        }
    }

    pub fn store_v(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        let var = Variable::from_u32(V_VAR);

        if self.pstate.is_none() {
            self.load_pstate(bcx);
        }

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        pstate.v = true;

        bcx.def_var(var, n);
    }

    pub fn check_alu_pstate(
        &mut self,
        bcx: &mut FunctionBuilder,
        val: Value,
        a: Value,
        b: Value,
        sf: bool,
        mut checker: impl FnMut(&mut FunctionBuilder, bool, Value, Value) -> Value,
    ) {
        // todo: optimize it

        let f = bcx.ins().iconst(types::I8, 0);
        let t = bcx.ins().iconst(types::I8, 1);

        // true if the value is negative
        let is_neg = if sf {
            bcx.ins().ushr_imm(val, 63)
        } else {
            bcx.ins().ushr_imm(val, 31)
        };

        let n = bcx.ins().ireduce(types::I8, is_neg);
        self.store_n(bcx, n);

        let z = bcx.ins().select(val, f, t);
        self.store_z(bcx, z);

        // todo: optimize it
        let (c, v) = if sf {
            let c = checker(bcx, true, a, b);
            let v = checker(bcx, false, a, b);

            (c, v)
        } else {
            let c = checker(bcx, true, a, b);
            let v = checker(bcx, false, a, b);

            (c, v)
        };

        self.store_c(bcx, c);
        self.store_v(bcx, v);
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

        if let Some(pstate) = self.pstate {
            let mut var = bcx.use_var(Variable::from_u32(PSTATE_VAR));
            let f = bcx.ins().iconst(types::I8, 0);

            if pstate.n {
                let n = bcx.use_var(Variable::from_u32(N_VAR));
                let mask = bcx.ins().iconst(types::I8, N_MASK);
                let n = bcx.ins().select(n, mask, f);
                var = bcx.ins().bor(var, n);
            }

            if pstate.z {
                let n = bcx.use_var(Variable::from_u32(Z_VAR));
                let mask = bcx.ins().iconst(types::I8, Z_MASK);
                let n = bcx.ins().select(n, mask, f);
                var = bcx.ins().bor(var, n);
            }

            if pstate.c {
                let n = bcx.use_var(Variable::from_u32(C_VAR));
                let mask = bcx.ins().iconst(types::I8, C_MASK);
                let n = bcx.ins().select(n, mask, f);
                var = bcx.ins().bor(var, n);
            }

            if pstate.v {
                let n = bcx.use_var(Variable::from_u32(V_VAR));
                let mask = bcx.ins().iconst(types::I8, V_MASK);
                let n = bcx.ins().select(n, mask, f);
                var = bcx.ins().bor(var, n);
            }

            set_pstate(bcx, self.cpu_ctx, var);
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

fn get_pstate(bcx: &mut FunctionBuilder, cpu_context: Value) -> Value {
    let offset = memoffset::offset_of!(CpuContext, pstate);

    bcx.ins().load(
        types::I8,
        MemFlags::trusted(),
        cpu_context,
        i32::try_from(offset).unwrap(),
    )
}

fn set_pstate(bcx: &mut FunctionBuilder, cpu_context: Value, pstate: Value) {
    let offset = memoffset::offset_of!(CpuContext, pstate);

    bcx.ins().store(
        MemFlags::trusted(),
        pstate,
        cpu_context,
        i32::try_from(offset).unwrap(),
    );
}

#[cfg(test)]
mod tests {
    use crate::context::CpuContext;
    use crate::inst::{Inst, Operand};
    use crate::translate::{ExecutionReturn, InterruptType, Translator};

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
                op2: Operand::imm(12),
                sf: true,
            },
            Inst::Add {
                rd: 8,
                rn: 0,
                op2: Operand::imm(16),
                sf: false,
            },
            Inst::Sub {
                rd: 4,
                rn: 8,
                op2: Operand::imm(14),
                sf: true,
            },
            Inst::Sub {
                rd: 16,
                rn: 4,
                op2: Operand::imm(6),
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

        let fptr = trans.translate(&[Inst::Udf { imm: 114 }]);

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 32;

        let status = unsafe { fptr(&mut cpu_ctx) };

        let ret = ExecutionReturn::from_u32(status);

        assert_eq!(ret.ty(), InterruptType::Udf);
        assert_eq!(ret.val(), 114);

        let fptr = trans.translate(&[
            Inst::Add {
                rd: 0,
                rn: 0,
                op2: Operand::imm(u32::MAX as u32),
                sf: false,
            },
            Inst::Adds {
                rd: 0,
                rn: 0,
                op2: Operand::imm(1),
                sf: false,
            },
            Inst::Nop,
            Inst::Svc { imm: 2 },
        ]);

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 32;

        let status = unsafe { fptr(&mut cpu_ctx) };
        let exec = ExecutionReturn::from_u32(status);

        println!("{:08b}", cpu_ctx.pstate);
        println!("{:?}", cpu_ctx);
        dbg!(exec.ty(), exec.val());
    }
}
