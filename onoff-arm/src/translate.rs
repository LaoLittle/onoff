use crate::context::{CpuContext, REG_LR};
use crate::inst::{Condition, Inst as ArmInst, Inst, Operand, ShiftType};
use cranelift::codegen::ir::UserFuncName;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
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

        let jb = JITBuilder::with_isa(isa.clone(), default_libcall_names());

        let mut module = JITModule::new(jb);
        let ptr = module.target_config().pointer_type();
        let abi_ptr = AbiParam::new(ptr);

        let mut signature = module.make_signature();
        signature.params.push(abi_ptr);
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

                        let val = bcx.ins().isub(rn, op2);

                        saver.store_register_with_sf(rd, &mut bcx, val, sf);

                        let nop2 = bcx.ins().bnot(op2);
                        let ty = RegisterSaver::type_with_sf(sf);
                        let carry = bcx.ins().iconst(ty, 1);
                        saver.check_alu_pstate_with_carry(
                            &mut bcx,
                            val,
                            rn,
                            nop2,
                            carry,
                            sf,
                            |bcx, us, x, y| {
                                if us {
                                    bcx.ins().uadd_overflow(x, y).1
                                } else {
                                    bcx.ins().sadd_overflow(x, y).1
                                }
                            },
                        );
                    }
                    ArmInst::Adc { rd, rn, rm, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let rm = saver.load_register_with_sf(rm, &mut bcx, sf);

                        let c = saver.load_c(&mut bcx);
                        let ty = RegisterSaver::type_with_sf(sf);
                        let c = bcx.ins().uextend(ty, c);

                        let add = bcx.ins().iadd(rn, rm);
                        let adc = bcx.ins().iadd(add, c);

                        saver.store_register_with_sf(rd, &mut bcx, adc, sf);
                    }
                    ArmInst::Adcs { rd, rn, rm, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let rm = saver.load_register_with_sf(rm, &mut bcx, sf);

                        let c = saver.load_c(&mut bcx);
                        let ty = RegisterSaver::type_with_sf(sf);
                        let c = bcx.ins().uextend(ty, c);

                        let add = bcx.ins().iadd(rn, rm);
                        let adc = bcx.ins().iadd(add, c);

                        saver.store_register_with_sf(rd, &mut bcx, adc, sf);

                        saver.check_alu_pstate_with_carry(
                            &mut bcx,
                            adc,
                            rn,
                            rm,
                            c,
                            sf,
                            |bcx, us, x, y| {
                                if us {
                                    bcx.ins().uadd_overflow(x, y).1
                                } else {
                                    bcx.ins().sadd_overflow(x, y).1
                                }
                            },
                        );
                    }
                    ArmInst::Sbc { rd, rn, rm, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let rm = saver.load_register_with_sf(rm, &mut bcx, sf);
                        let rm = bcx.ins().bnot(rm);

                        let c = saver.load_c(&mut bcx);
                        let ty = RegisterSaver::type_with_sf(sf);
                        let c = bcx.ins().uextend(ty, c);

                        let add = bcx.ins().iadd(rn, rm);
                        let adc = bcx.ins().iadd(add, c);

                        saver.store_register_with_sf(rd, &mut bcx, adc, sf);
                    }
                    ArmInst::Sbcs { rd, rn, rm, sf } => {
                        let rn = saver.load_register_with_sf(rn, &mut bcx, sf);
                        let rm = saver.load_register_with_sf(rm, &mut bcx, sf);
                        let rm = bcx.ins().bnot(rm);

                        let c = saver.load_c(&mut bcx);
                        let ty = RegisterSaver::type_with_sf(sf);
                        let c = bcx.ins().uextend(ty, c);

                        let add = bcx.ins().iadd(rn, rm);
                        let adc = bcx.ins().iadd(add, c);

                        saver.store_register_with_sf(rd, &mut bcx, adc, sf);

                        saver.check_alu_pstate_with_carry(
                            &mut bcx,
                            adc,
                            rn,
                            rm,
                            c,
                            sf,
                            |bcx, us, x, y| {
                                if us {
                                    bcx.ins().uadd_overflow(x, y).1
                                } else {
                                    bcx.ins().sadd_overflow(x, y).1
                                }
                            },
                        );
                    }
                    ArmInst::Movz { rd, imm, shift, sf } => {
                        let result = (imm as u64) << shift;
                        let result = bcx
                            .ins()
                            .iconst(RegisterSaver::type_with_sf(sf), result as i64);

                        saver.store_register_with_sf(rd, &mut bcx, result, sf);
                    }
                    ArmInst::Svc { imm } => {
                        ret.set_ty(InterruptType::Svc);
                        ret.set_val(imm);

                        break;
                    }
                    ArmInst::Nop => {
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
                    Inst::Strb {
                        rt,
                        rn,
                        offset,
                        wback,
                        postindex,
                    } => {
                        todo!()
                    }
                    Inst::Strh { .. } => {
                        todo!()
                    }
                    Inst::Str {
                        rt,
                        rn,
                        offset,
                        wback,
                        postindex,
                        sf,
                    } => {
                        let mut addr = saver.load_register(rn, &mut bcx);
                        let mut acc_off = 0;

                        if !postindex {
                            acc_off = offset as i32;
                        }

                        let data = if rt == 31 {
                            saver.load_zr(&mut bcx, sf)
                        } else {
                            saver.load_register_with_sf(rt, &mut bcx, sf)
                        };

                        bcx.ins().store(MemFlags::trusted(), data, addr, acc_off);

                        if wback {
                            if postindex {
                                addr = bcx.ins().iadd_imm(addr, offset);
                            }

                            saver.store_register(rn, &mut bcx, addr);
                        }
                    }
                    Inst::Ldrb { .. } => todo!(),
                    Inst::Ldrh { .. } => todo!(),
                    Inst::Ldr {
                        rt,
                        rn,
                        offset,
                        wback,
                        postindex,
                        sf,
                    } => {
                        let mut addr = saver.load_register(rn, &mut bcx);
                        let mut acc_off = 0;

                        if !postindex {
                            acc_off = offset as i32;
                        }

                        let data = bcx.ins().load(
                            RegisterSaver::type_with_sf(sf),
                            MemFlags::trusted(),
                            addr,
                            acc_off,
                        );
                        saver.store_register_with_sf(rt, &mut bcx, data, sf);

                        if wback {
                            if postindex {
                                addr = bcx.ins().iadd_imm(addr, offset);
                            }

                            saver.store_register(rn, &mut bcx, addr);
                        }
                    }
                    Inst::Csinc {
                        rd,
                        rn,
                        rm,
                        cond,
                        sf,
                    } => {
                        let op1 = if rn == 31 {
                            saver.load_zr(&mut bcx, sf)
                        } else {
                            saver.load_register_with_sf(rn, &mut bcx, sf)
                        };
                        let op2 = if rm == 31 {
                            saver.load_zr(&mut bcx, sf)
                        } else {
                            saver.load_register_with_sf(rm, &mut bcx, sf)
                        };
                        let op2 = bcx.ins().iadd_imm(op2, 1);

                        let cond = saver.load_cond(&mut bcx, cond);

                        let result = bcx.ins().select(cond, op1, op2);

                        saver.store_register_with_sf(rd, &mut bcx, result, sf);
                    }
                    Inst::Tbz {
                        rt,
                        bit_pos,
                        offset,
                        sf,
                    } => {
                        let mask = 0b1 << bit_pos;
                        let rt = saver.load_register_with_sf(rt, &mut bcx, sf);

                        let pc = saver.load_pc(&mut bcx);
                        let t = bcx.ins().iadd_imm(pc, 4);
                        let f = bcx.ins().iadd_imm(pc, offset);

                        let bit = bcx.ins().band_imm(rt, mask);

                        let target = bcx.ins().select(bit, t, f);
                        saver.store_pc(&mut bcx, target);

                        save_pc = false;

                        break;
                    }
                    Inst::Tbnz {
                        rt,
                        bit_pos,
                        offset,
                        sf,
                    } => {
                        let mask = 0b1 << bit_pos;
                        let rt = saver.load_register_with_sf(rt, &mut bcx, sf);

                        let pc = saver.load_pc(&mut bcx);
                        let t = bcx.ins().iadd_imm(pc, offset);
                        let f = bcx.ins().iadd_imm(pc, 4);

                        let bit = bcx.ins().band_imm(rt, mask);

                        let target = bcx.ins().select(bit, t, f);
                        saver.store_pc(&mut bcx, target);

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

            let ret = bcx.ins().iconst(types::I32, ret.into_u32() as u64 as i64);
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

#[derive(Debug, Clone, Copy)]
struct LoadState {
    pub loaded: bool,
    pub stored: bool,
}

impl LoadState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            loaded: false,
            stored: false,
        }
    }

    #[inline]
    pub fn set_loaded(&mut self) {
        self.loaded = true
    }

    #[inline]
    pub fn set_stored(&mut self) {
        self.loaded = true;
        self.stored = true;
    }
}

const N_IDX: usize = 0;
const Z_IDX: usize = 1;
const C_IDX: usize = 2;
const V_IDX: usize = 3;

struct PState {
    pub nzcv: [LoadState; 4],
}

impl PState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            nzcv: [LoadState::new(); 4],
        }
    }
}

struct RegisterSaver {
    gprs: [LoadState; 32], // pointers to the registers
    pc: LoadState,
    pstate: Option<PState>,
    cpu_ctx: Value,
    pc_offset: i64,
}

const PC_VAR: u32 = 100;
const PSTATE_VAR: u32 = 101;

const NZCV_BASE_VAR: u32 = 110;

impl RegisterSaver {
    pub fn new(cpu_ctx: Value) -> Self {
        Self {
            gprs: [LoadState::new(); 32],
            pc: LoadState::new(),
            pstate: None,
            cpu_ctx,
            pc_offset: 0,
        }
    }

    #[inline]
    pub const fn cpu_ctx(&self) -> Value {
        self.cpu_ctx
    }

    pub fn type_with_sf(sf: bool) -> types::Type {
        if sf {
            types::I64
        } else {
            types::I32
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
        if self.gprs[rds].loaded {
            bcx.use_var(var)
        } else {
            let val = get_register(bcx, self.cpu_ctx, rn);
            bcx.declare_var(var, types::I64);
            bcx.def_var(var, val);
            self.gprs[rds].set_loaded();
            val
        }
    }

    pub fn load_register32(&mut self, rn: u8, bcx: &mut FunctionBuilder) -> Value {
        let rds = rn as usize;

        let var = Variable::from_u32(rn as u32);
        if self.gprs[rds].loaded {
            let r = bcx.use_var(var);
            bcx.ins().ireduce(types::I32, r)
        } else {
            let val = get_register32(bcx, self.cpu_ctx, rn);
            bcx.declare_var(var, types::I64);
            let extended = bcx.ins().uextend(types::I64, val);
            bcx.def_var(var, extended);
            self.gprs[rds].set_loaded();
            val
        }
    }

    pub fn load_zr(&mut self, bcx: &mut FunctionBuilder, sf: bool) -> Value {
        let ty = Self::type_with_sf(sf);

        bcx.ins().iconst(ty, 0)
    }

    pub fn load_operand(&mut self, bcx: &mut FunctionBuilder, operand: Operand, sf: bool) -> Value {
        let ty = if sf { types::I64 } else { types::I32 };

        match operand {
            Operand::Imm(imm) => {
                let imm = imm as u64 as i64;
                bcx.ins().iconst(ty, imm)
            }
            Operand::ShiftedReg {
                rm,
                shift_type,
                amount,
            } => {
                let amount = amount as i64;
                let reg = self.load_register_with_sf(rm, bcx, sf);

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
        val: Value,
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
        let state = &mut self.gprs[rn as usize];
        if !state.loaded {
            bcx.declare_var(var, types::I64);
        }

        bcx.def_var(var, val);
        state.set_stored();
    }

    pub fn store_register32(&mut self, rn: u8, bcx: &mut FunctionBuilder, val: Value) {
        let var = Variable::from_u32(rn as u32);
        let state = &mut self.gprs[rn as usize];
        if !state.loaded {
            bcx.declare_var(var, types::I64);
        }

        let extended = bcx.ins().uextend(types::I64, val);
        bcx.def_var(var, extended);
        state.set_stored();
    }

    pub fn load_pc(&mut self, bcx: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(PC_VAR);
        if !self.pc.loaded {
            bcx.declare_var(var, types::I64);
            let pc = get_pc(bcx, self.cpu_ctx);
            bcx.def_var(var, pc);
            self.pc.set_loaded();
        }

        let rel = bcx.use_var(var);
        bcx.ins().iadd_imm(rel, self.pc_offset)
    }

    pub fn store_pc(&mut self, bcx: &mut FunctionBuilder, val: Value) {
        let var = Variable::from_u32(PC_VAR);
        if !self.pc.loaded {
            bcx.declare_var(var, types::I64);
        }

        bcx.def_var(var, val);
        self.pc.set_stored();
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
            Self::declare_nzcv(bcx);
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
            Self::declare_nzcv(bcx);
            self.pstate = Some(PState::new());
        }

        bcx.def_var(var, pstate);
    }

    pub fn clear_pstate(&mut self) {
        self.pstate = None;
    }

    #[inline]
    pub fn load_n(&mut self, bcx: &mut FunctionBuilder) -> Value {
        self.load_nzcv(bcx, N_IDX)
    }

    #[inline]
    pub fn store_n(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        self.store_nzcv(bcx, n, N_IDX);
    }

    #[inline]
    pub fn load_z(&mut self, bcx: &mut FunctionBuilder) -> Value {
        self.load_nzcv(bcx, Z_IDX)
    }

    #[inline]
    pub fn store_z(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        self.store_nzcv(bcx, n, Z_IDX);
    }

    #[inline]
    pub fn load_c(&mut self, bcx: &mut FunctionBuilder) -> Value {
        self.load_nzcv(bcx, C_IDX)
    }

    #[inline]
    pub fn store_c(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        self.store_nzcv(bcx, n, C_IDX);
    }

    #[inline]
    pub fn load_v(&mut self, bcx: &mut FunctionBuilder) -> Value {
        self.load_nzcv(bcx, V_IDX)
    }

    #[inline]
    pub fn store_v(&mut self, bcx: &mut FunctionBuilder, n: Value) {
        self.store_nzcv(bcx, n, V_IDX);
    }

    pub fn load_nzcv(&mut self, bcx: &mut FunctionBuilder, idx: usize) -> Value {
        let var = Variable::from_u32(NZCV_BASE_VAR + idx as u32);

        let pval = self.load_pstate(bcx);

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        let mask = match idx {
            N_IDX => N_MASK,
            Z_IDX => Z_MASK,
            C_IDX => C_MASK,
            V_IDX => V_MASK,
            _ => panic!("unknown nczv index: {idx}"),
        };

        if pstate.nzcv[idx].loaded {
            bcx.use_var(var)
        } else {
            let bit = bcx.ins().band_imm(pval, mask);
            let is = utils::i2bool(bcx, bit);

            bcx.def_var(var, is);
            pstate.nzcv[idx].set_loaded();
            is
        }
    }

    pub fn store_nzcv(&mut self, bcx: &mut FunctionBuilder, n: Value, idx: usize) {
        let var = Variable::from_u32(NZCV_BASE_VAR + idx as u32);

        if self.pstate.is_none() {
            self.load_pstate(bcx);
        }

        let Some(pstate) = &mut self.pstate else {
            unreachable!("pstate should be defined in the context!");
        };

        bcx.def_var(var, n);
        pstate.nzcv[idx].set_stored();
    }

    pub fn declare_nzcv(bcx: &mut FunctionBuilder) {
        for i in [N_IDX, Z_IDX, C_IDX, V_IDX] {
            bcx.declare_var(Variable::from_u32(NZCV_BASE_VAR + i as u32), types::I8);
        }
    }

    pub fn load_cond(&mut self, bcx: &mut FunctionBuilder, cond: Condition) -> Value {
        use Condition::*;

        match cond {
            Eq => self.load_z(bcx),
            Ne => {
                let cond = self.load_z(bcx);
                utils::is_zero(bcx, cond)
            }
            Cs => self.load_c(bcx),
            Cc => {
                let cond = self.load_c(bcx);
                utils::is_zero(bcx, cond)
            }
            Mi => self.load_n(bcx),
            Pl => {
                let cond = self.load_n(bcx);
                utils::is_zero(bcx, cond)
            }
            Vs => self.load_v(bcx),
            Vc => {
                let cond = self.load_v(bcx);
                utils::is_zero(bcx, cond)
            }
            Hi => {
                let c = self.load_c(bcx);
                let z = self.load_z(bcx);
                let z = utils::is_zero(bcx, z);
                bcx.ins().band(c, z)
            }
            Ls => {
                let cond = {
                    let c = self.load_c(bcx);
                    let z = self.load_z(bcx);
                    let z = utils::is_zero(bcx, z);
                    bcx.ins().band(c, z)
                };

                utils::is_zero(bcx, cond)
            }
            Ge => {
                let n = self.load_n(bcx);
                let v = self.load_v(bcx);

                utils::is_eq(bcx, n, v)
            }
            Lt => {
                let n = self.load_n(bcx);
                let v = self.load_v(bcx);

                utils::is_neq(bcx, n, v)
            }
            Gt => {
                let c0 = {
                    let n = self.load_n(bcx);
                    let v = self.load_v(bcx);

                    utils::is_eq(bcx, n, v)
                };

                let z = self.load_z(bcx);
                let z = utils::is_zero(bcx, z);

                bcx.ins().band(c0, z)
            }
            Le => {
                let cond = {
                    let c0 = {
                        let n = self.load_n(bcx);
                        let v = self.load_v(bcx);

                        utils::is_eq(bcx, n, v)
                    };

                    let z = self.load_z(bcx);
                    let z = utils::is_zero(bcx, z);

                    bcx.ins().band(c0, z)
                };

                utils::is_zero(bcx, cond)
            }
            Al => bcx.ins().iconst(types::I8, 1),
        }
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
        // true if the value is negative
        let is_neg = if sf {
            bcx.ins().ushr_imm(val, 63)
        } else {
            bcx.ins().ushr_imm(val, 31)
        };

        let n = bcx.ins().ireduce(types::I8, is_neg);
        self.store_n(bcx, n);

        let z = utils::is_zero(bcx, val);
        self.store_z(bcx, z);

        let c = checker(bcx, true, a, b);
        let v = checker(bcx, false, a, b);

        self.store_c(bcx, c);
        self.store_v(bcx, v);
    }

    pub fn check_alu_pstate_with_carry(
        &mut self,
        bcx: &mut FunctionBuilder,
        val: Value,
        a: Value,
        b: Value,
        cin: Value,
        sf: bool,
        mut checker: impl FnMut(&mut FunctionBuilder, bool, Value, Value) -> Value,
    ) {
        // todo: optimize it
        // true if the value is negative
        let is_neg = if sf {
            bcx.ins().ushr_imm(val, 63)
        } else {
            bcx.ins().ushr_imm(val, 31)
        };

        let n = bcx.ins().ireduce(types::I8, is_neg);
        self.store_n(bcx, n);

        let z = utils::is_zero(bcx, val);
        self.store_z(bcx, z);

        let ab = bcx.ins().iadd(a, b);
        let c0 = checker(bcx, true, a, b);
        let v0 = checker(bcx, false, a, b);

        let c1 = checker(bcx, true, ab, cin);
        let v1 = checker(bcx, false, ab, cin);

        let c = bcx.ins().bor(c0, c1);
        let v = bcx.ins().bor(v0, v1);

        self.store_c(bcx, c);
        self.store_v(bcx, v);
    }

    pub fn save_registers(self, bcx: &mut FunctionBuilder) {
        for (rd, gpr) in self.gprs.into_iter().enumerate() {
            if !gpr.stored {
                continue;
            }

            let val = bcx.use_var(Variable::from_u32(rd as u32));
            set_register(bcx, self.cpu_ctx, val, rd as u8);
        }

        if self.pc.stored {
            let pc = bcx.use_var(Variable::from_u32(PC_VAR));
            set_pc(bcx, self.cpu_ctx, pc);
        }

        if let Some(pstate) = self.pstate {
            let mut var = bcx.use_var(Variable::from_u32(PSTATE_VAR));
            let zr = bcx.ins().iconst(types::I8, 0);

            for s in &pstate.nzcv {
                if s.stored {
                    // clear pstate if we set any flag
                    var = zr;
                    break;
                }
            }

            for (i, mask) in [
                (N_IDX, N_MASK),
                (Z_IDX, Z_MASK),
                (C_IDX, C_MASK),
                (V_IDX, V_MASK),
            ] {
                if pstate.nzcv[i].stored {
                    let s = bcx.use_var(Variable::from_u32(NZCV_BASE_VAR + i as u32));
                    let mask = bcx.ins().iconst(types::I8, mask);
                    let s = bcx.ins().select(s, mask, zr);
                    var = bcx.ins().bor(var, s);
                }
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
    let offset = memoffset::offset_of!(CpuContext, gprs) + (core::mem::size_of::<u64>() * rd);

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
    let offset = memoffset::offset_of!(CpuContext, gprs) + (core::mem::size_of::<u64>() * rd);

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

mod utils {
    use cranelift::prelude::*;

    /// cast any integer type into a bool value
    pub fn i2bool(bcx: &mut FunctionBuilder, val: Value) -> Value {
        // val != 0
        bcx.ins().icmp_imm(IntCC::NotEqual, val, 0)
    }

    pub fn is_zero(bcx: &mut FunctionBuilder, val: Value) -> Value {
        // val == 0
        bcx.ins().icmp_imm(IntCC::Equal, val, 0)
    }

    pub fn is_eq(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        bcx.ins().icmp(IntCC::Equal, a, b)
    }

    pub fn is_neq(bcx: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        bcx.ins().icmp(IntCC::NotEqual, a, b)
    }
}

mod external {
    use crate::context::CpuContext;

    const N_MASK: u8 = 0b10000000;
    const Z_MASK: u8 = 0b01000000;
    const C_MASK: u8 = 0b00100000;
    const V_MASK: u8 = 0b00010000;

    pub unsafe extern "C" fn adds_with_carry(a: u64, b: u64, ctx: *mut CpuContext) -> u64 {
        let carry = (((*ctx).pstate & C_MASK) != 0) as u64;

        let mut pstate = 0;
        let val = u64::wrapping_add(u64::wrapping_add(a, b), carry);

        if (val as i64).is_negative() {
            pstate |= N_MASK;
        }

        if val == 0 {
            pstate |= Z_MASK;
        }

        let (v, c0) = u64::overflowing_add(a, b);
        let (_, c1) = u64::overflowing_add(v, carry);

        if c0 || c1 {
            pstate |= C_MASK;
        }

        let (v, v0) = i64::overflowing_add(a as i64, b as i64);
        let (_, v1) = i64::overflowing_add(v, carry as i64);

        if v0 || v1 {
            pstate |= V_MASK;
        }

        (*ctx).pstate = pstate;

        val
    }

    pub unsafe extern "C" fn adds_with_carry32(a: u32, b: u32, ctx: *mut CpuContext) -> u32 {
        let carry = (((*ctx).pstate & C_MASK) != 0) as u32;

        let mut pstate = 0;
        let val = u32::wrapping_add(u32::wrapping_add(a, b), carry);

        if (val as i32).is_negative() {
            pstate |= N_MASK;
        }

        if val == 0 {
            pstate |= Z_MASK;
        }

        let (v, c0) = u32::overflowing_add(a, b);
        let (_, c1) = u32::overflowing_add(v, carry);

        if c0 || c1 {
            pstate |= C_MASK;
        }

        let (v, v0) = i32::overflowing_add(a as i32, b as i32);
        let (_, v1) = i32::overflowing_add(v, carry as i32);

        if v0 || v1 {
            pstate |= V_MASK;
        }

        (*ctx).pstate = pstate;

        val
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{CpuContext, REG_SP};
    use crate::inst::Operand::Imm;
    use crate::inst::ShiftType;
    use crate::inst::{
        Inst::{self, *},
        Operand,
    };
    use crate::translate::{ExecutionReturn, InterruptType, Translator};
    use core::slice;

    #[test]
    fn test_general() {
        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 16;

        run_codegen_test(
            &[
                Adr { rd: 2, label: 8 },
                Adr { rd: 0, label: 0 },
                B { label: 8 },
            ],
            cpu_ctx,
            |ctx, ret| {
                assert_eq!(ctx.pc, 16 + 4 + 4 + 8);
                assert_eq!(ctx.gprs[0], 16 + 4 + 0);
                assert_eq!(ctx.gprs[2], 16 + 8);

                assert_eq!(ret.ty(), InterruptType::None);
                assert_eq!(ret.val(), 0);
            },
        );

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 32;

        run_codegen_test(
            &[
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
            ],
            cpu_ctx,
            |ctx, ret| {
                assert_eq!(ctx.pc, 32 + 4 * 4);
                assert_eq!(ctx.gprs[0], 12);
                assert_eq!(ctx.gprs[4], 28 - 14);
                assert_eq!(ctx.gprs[8], 12 + 16);
                assert_eq!(ctx.gprs[16], 14 - 6);

                assert_eq!(ret.ty(), InterruptType::None);
                assert_eq!(ret.val(), 0);
            },
        );

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 16;

        run_codegen_test(
            &[
                Inst::Adr { rd: 2, label: 8 },
                Inst::Adr { rd: 0, label: 0 },
                Inst::B { label: 8 },
            ],
            cpu_ctx,
            |ctx, ret| {
                assert_eq!(ctx.pc, 16 + 16);
                assert_eq!(ctx.gprs[0], 16 + 4);
                assert_eq!(ctx.gprs[2], 16 + 8);
            },
        );

        let mut cpu_ctx = CpuContext::new();
        cpu_ctx.pc = 16;

        run_codegen_test(
            &[Inst::Adr { rd: 30, label: 32 }, Inst::Ret { rn: 30 }],
            cpu_ctx,
            |ctx, ret| {
                assert_eq!(ctx.pc, 16 + 32);
                assert_eq!(ctx.gprs[30], 16 + 32);
            },
        );

        run_codegen_test(
            &[
                Add {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(114),
                    sf: true,
                },
                Add {
                    rd: 1,
                    rn: 1,
                    op2: Operand::imm(514),
                    sf: true,
                },
                Add {
                    rd: 2,
                    rn: 0,
                    op2: Operand::shifted_reg(1, ShiftType::Lsl, 1, true),
                    sf: true,
                },
                Add {
                    rd: 3,
                    rn: 0,
                    op2: Operand::shifted_reg(1, ShiftType::Ror, 3, true),
                    sf: true,
                },
                Add {
                    rd: 4,
                    rn: 0,
                    op2: Operand::shifted_reg(1, ShiftType::Lsr, 1, true),
                    sf: true,
                },
                Add {
                    rd: 5,
                    rn: 5,
                    op2: Operand::imm(i32::MIN as u32),
                    sf: true,
                },
                Add {
                    rd: 6,
                    rn: 0,
                    op2: Operand::shifted_reg(5, ShiftType::Asr, 3, false),
                    sf: false,
                },
            ],
            CpuContext::new(),
            |ctx, _| {
                assert_eq!(ctx.gprs[2], 114 + (514 << 1));
                assert_eq!(ctx.gprs[3], 114 + (u64::rotate_right(514, 3)));
                assert_eq!(ctx.gprs[4], 114 + (514 >> 1));

                let i = (i32::MIN >> 3) as u32;

                assert_eq!(ctx.gprs[6], 114 + i as u64);
            },
        );
    }

    #[test]
    fn test_interrupt() {
        run_codegen_test(&[Udf { imm: 114 }], CpuContext::new(), |ctx, ret| {
            assert_eq!(ret.ty(), InterruptType::Udf);
            assert_eq!(ret.val(), 114);
        });

        run_codegen_test(&[Nop, Svc { imm: 514 }], CpuContext::new(), |ctx, ret| {
            assert_eq!(ret.ty(), InterruptType::Svc);
            assert_eq!(ret.val(), 514);
        });
    }

    #[test]
    fn test_pstate() {
        run_codegen_test(
            &[
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
            ],
            CpuContext::new(),
            |ctx, ret| {
                assert_eq!(ctx.pstate, 0b01100000); // zero and carry (unsigned overflow)
                assert_eq!(ret.ty(), InterruptType::None);
            },
        );

        run_codegen_test(
            &[
                Inst::Add {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(i32::MAX as u32),
                    sf: false,
                },
                Inst::Adds {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(1),
                    sf: false,
                },
            ],
            CpuContext::new(),
            |ctx, ret| {
                assert_eq!(ctx.pstate, 0b10010000); // negative and overflow (signed overflow)
                assert_eq!(ret.ty(), InterruptType::None);
            },
        );

        run_codegen_test(
            &[
                Inst::Add {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(2),
                    sf: false,
                },
                Inst::Subs {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(1),
                    sf: false,
                },
            ],
            CpuContext::new(),
            |ctx, ret| {
                assert_eq!(ctx.gprs[0], 1);
                assert_eq!(ctx.pstate, 0b00100000);
                assert_eq!(ret.ty(), InterruptType::None);
            },
        );

        run_codegen_test(
            &[
                Inst::Add {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(i32::MIN as u32),
                    sf: false,
                },
                Inst::Subs {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(1),
                    sf: false,
                },
            ],
            CpuContext::new(),
            |ctx, ret| {
                assert_eq!(ctx.pstate, 0b00110000);
                assert_eq!(ret.ty(), InterruptType::None);
            },
        );

        let c = CpuContext::new();

        run_codegen_test(
            &[
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
                Add {
                    rd: 1,
                    rn: 1,
                    op2: Operand::imm(10),
                    sf: true,
                },
                Adc {
                    rd: 1,
                    rn: 1,
                    rm: 1,
                    sf: true,
                },
            ],
            c,
            |cx, _| {
                assert_eq!(cx.gprs[1], 10 + 10 + 1);
            },
        );

        let mut c = CpuContext::new();
        c.pstate = 0b11110000;

        run_codegen_test(
            &[
                Add {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(u32::MAX - 1),
                    sf: false,
                },
                Adds {
                    rd: 1,
                    rn: 0,
                    op2: Operand::imm(2),
                    sf: false,
                },
                Add {
                    rd: 1,
                    rn: 1,
                    op2: Operand::imm(1),
                    sf: false,
                },
                Adcs {
                    rd: 2,
                    rn: 0,
                    rm: 1,
                    sf: false,
                },
            ],
            c,
            |cx, _| {
                assert_eq!(cx.gprs[1], 1);
                assert_eq!(cx.gprs[2], 0); // overflowed: u32::MAX - 1 + 1 + C(1)
                assert_eq!(cx.pstate, 0b01100000);
            },
        );

        let mut c = CpuContext::new();
        c.pstate = 0b00000000;

        run_codegen_test(
            &[
                Add {
                    rd: 0,
                    rn: 0,
                    op2: Operand::imm(114),
                    sf: false,
                },
                Add {
                    rd: 1,
                    rn: 1,
                    op2: Operand::imm(2),
                    sf: false,
                },
                Inst::Sbc {
                    rd: 0,
                    rn: 0,
                    rm: 1,
                    sf: true,
                },
            ],
            c,
            |cx, _| {
                assert_eq!(cx.gprs[0], 114 - 2 - 1);
            },
        );
    }

    #[test]
    fn test_mem() {
        let mut stack = Box::new([1u8; 16]);

        let mut ctx = CpuContext::new();
        *ctx.gpr_mut(8) = 114;
        *ctx.gpr_mut(REG_SP) = stack.as_ptr() as u64;

        run_codegen_test(
            &[Str {
                rt: 8,
                rn: 31,
                offset: 4,
                wback: false,
                postindex: false,
                sf: false,
            }],
            ctx,
            |cx, _| {
                println!("{:?}", stack);
            },
        );

        drop(stack);
    }

    fn run_codegen_test(
        insts: &[Inst],
        mut ctx: CpuContext,
        f: impl FnOnce(CpuContext, ExecutionReturn),
    ) {
        eprintln!("start test");

        let mut trans = Translator::new().unwrap();
        trans.set_debug(true);
        let fptr = trans.translate(insts);

        let ret = unsafe { fptr(&mut ctx) };

        let exec = ExecutionReturn::from_u32(ret);

        eprintln!("Context: {ctx:?}");
        eprintln!("Pstate: {:08b}", ctx.pstate);
        dbg!(exec.ty(), exec.val());

        f(ctx, exec);
    }
}
