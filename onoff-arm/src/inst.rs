use onoff_core::error::{Error, Result};
use std::io::Read;

#[derive(Debug, Copy, Clone)]
pub enum Inst {
    /// `UDF #<imm>`
    Udf { imm: u16 },
    /// `SVC #<imm>`
    Svc { imm: u16 },
    /// `NOP`
    Nop,
    /// `BR <Xn>`
    Br { rn: u8 },
    /// `BLR <Xn>`
    Blr { rn: u8 },
    /// `RET <Rn>`
    Ret { rn: u8 },
    /// `B <label>`
    B { label: i64 },
    /// `BL <label>`
    Bl { label: i64 },
    /// `ADR <Xd>, <label>`
    Adr { rd: u8, label: i64 },
    /// `ADRP <Xd>, <label>`
    Adrp { rd: u8, label: i64 },
    /// `ADD <Rd>, <Rn|RSP>, #<imm>{, <shift>}`
    Add {
        rd: u8,
        rn: u8,
        op2: Operand,
        sf: bool,
    },
    /// `ADDS <Rd>, <Rn|RSP>, #<imm>{, <shift>}`
    /// `ADDS <Rd>, <Rn>, <Rm>{, <shift> #<amount>}`
    Adds {
        rd: u8,
        rn: u8,
        op2: Operand,
        sf: bool,
    },
    /// `SUB <Rd>, <Rn|RSP>, #<imm>{, <shift>}`
    Sub {
        rd: u8,
        rn: u8,
        op2: Operand,
        sf: bool,
    },
    /// `SUBS <Rd>, <Rn|RSP>, #<imm>{, <shift>}`
    Subs {
        rd: u8,
        rn: u8,
        op2: Operand,
        sf: bool,
    },
    /// `ADC <Rd>, <Rn>, <Rm>`
    Adc { rd: u8, rn: u8, rm: u8, sf: bool },
    /// `ADCS <Rd>, <Rn>, <Rm>`
    Adcs { rd: u8, rn: u8, rm: u8, sf: bool },
    /// `SBC <Rd>, <Rn>, <Rm>`
    Sbc { rd: u8, rn: u8, rm: u8, sf: bool },
    /// `SBCS <Rd>, <Rn>, <Rm>`
    Sbcs { rd: u8, rn: u8, rm: u8, sf: bool },
}

#[derive(Debug, Clone, Copy)]
pub enum Operand {
    Imm(u32),
    ShiftedReg {
        rm: u8,
        shift_type: ShiftType,
        amount: u8,
    },
}

impl Operand {
    #[inline]
    pub const fn imm(imm: u32) -> Self {
        Self::Imm(imm)
    }

    #[inline]
    pub const fn reg(rm: u8) -> Self {
        Self::shifted_reg(rm, ShiftType::Lsl, 0, true)
    }

    pub const fn shifted_reg(rm: u8, shift_type: ShiftType, amount: u8, sf: bool) -> Self {
        assert!(rm < 32, "register number should be in [0, 32)");

        if sf {
            assert!(amount < 64, "shift amount should be less than 64!");
        } else {
            assert!(amount < 32, "shift amount should be less than 32!");
        }

        Self::ShiftedReg {
            rm,
            shift_type,
            amount,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ShiftType {
    /// Logical shift left
    Lsl = 0b00,
    /// Logical shift right
    Lsr = 0b01,
    /// Arithmetic shift right
    Asr = 0b10,
    /// Rotate right
    Ror = 0b11,
}

impl ShiftType {
    #[inline]
    pub const fn from_u32(value: u32) -> Option<Self> {
        use ShiftType::*;

        Some(match value {
            0b00 => Lsl,
            0b01 => Lsr,
            0b10 => Asr,
            0b11 => Ror,
            _ => return None,
        })
    }
}

pub struct InstDecoder<R> {
    reader: R,
}

impl<R: Read> InstDecoder<R> {
    #[inline]
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    #[inline]
    pub fn decode_inst(&mut self) -> Result<Inst> {
        let inst = self.next_u32()?;
        decode_inst_u32(inst)
    }

    fn next_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];

        self.reader.read_exact(&mut buf)?;

        Ok(u32::from_le_bytes(buf))
    }
}

impl<R: Read> IntoIterator for InstDecoder<R> {
    type Item = Inst;
    type IntoIter = IntoIter<R>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

pub struct IntoIter<R> {
    decoder: InstDecoder<R>,
}

impl<R> IntoIter<R> {
    #[inline]
    pub fn new(decoder: InstDecoder<R>) -> Self {
        Self { decoder }
    }
}

impl<R: Read> Iterator for IntoIter<R> {
    type Item = Inst;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.decoder.decode_inst().ok()
    }
}

#[inline]
const fn unspported() -> Result<Inst> {
    Err(Error::NotSupported)
}

// The main decoding entry
fn decode_inst_u32(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<29, 24>(inst);

    match op0 {
        // Reserved
        0b00000 | 0b00001 => decode_reserved(inst),
        // Data Processing -- Immediate
        0b10000 | 0b10001 | 0b10010 | 0b10011 => decode_dp_imm(inst),
        // Branches, Exception Generating and System instructions
        0b10100 | 0b10101 | 0b10110 | 0b10111 => decode_bes(inst),
        // Data Processing -- Register
        0b01010 | 0b01011 | 0b11010 | 0b11011 => decode_dp_reg(inst),
        _ => unspported(),
    }
}

const fn decode_reserved(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<32, 29>(inst);
    let op1 = extract_field::<25, 16>(inst);

    match (op0, op1) {
        (0b000, 0b000000000) => decode_udf(inst),
        _ => unspported(),
    }
}

const fn decode_udf(inst: u32) -> Result<Inst> {
    let imm = extract_field::<16, 0>(inst);

    Ok(Inst::Udf { imm: imm as u16 })
}

// Data Processing -- Immediate
const fn decode_dp_imm(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<26, 23>(inst);

    match op0 {
        0b000 | 0b001 => decode_pc_rel(inst),
        0b010 => decode_addsub_imm(inst),
        _ => unspported(),
    }
}

const fn decode_pc_rel(inst: u32) -> Result<Inst> {
    let op = extract_field::<32, 31>(inst);
    let immlo = extract_field::<31, 29>(inst);
    let immhi = extract_field::<24, 5>(inst);
    let rd = extract_field::<5, 0>(inst) as u8;

    let imm = sign_extend_64::<21>((immhi << 2) | immlo);

    let inst = match op {
        0b0 => Inst::Adr { rd, label: imm },
        0b1 => Inst::Adrp {
            rd,
            label: imm * 4096,
        },
        _ => unreachable!(),
    };

    Ok(inst)
}

const fn decode_addsub_imm(inst: u32) -> Result<Inst> {
    let sf = extract_field::<32, 31>(inst);
    let op = extract_field::<31, 30>(inst);
    let s = extract_field::<30, 29>(inst);
    // shift
    let sh = extract_field::<23, 22>(inst);
    let imm12 = extract_field::<22, 10>(inst);
    let rn = extract_field::<10, 5>(inst);
    let rd = extract_field::<5, 0>(inst);

    let sf = sf == 1;
    let sh = sh == 1;

    match (op, s) {
        (0b0, 0b0) => decode_add_imm(sf, sh, imm12, rn, rd),
        (0b1, 0b0) => decode_sub_imm(sf, sh, imm12, rn, rd),
        (0b0, 0b1) => decode_adds_imm(sf, sh, imm12, rn, rd),
        (0b1, 0b1) => decode_subs_imm(sf, sh, imm12, rn, rd),
        _ => unreachable!(),
    }
}

const fn decode_add_imm(sf: bool, sh: bool, imm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Add {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::imm(if sh { imm << 12 } else { imm }),
        sf,
    })
}

const fn decode_sub_imm(sf: bool, sh: bool, imm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Sub {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::imm(if sh { imm << 12 } else { imm }),
        sf,
    })
}

const fn decode_adds_imm(sf: bool, sh: bool, imm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Adds {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::imm(if sh { imm << 12 } else { imm }),
        sf,
    })
}

const fn decode_subs_imm(sf: bool, sh: bool, imm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Subs {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::imm(if sh { imm << 12 } else { imm }),
        sf,
    })
}

// Branch, Exception Generating and System
const fn decode_bes(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<32, 29>(inst);
    let op1 = extract_field::<26, 12>(inst);
    let op2 = extract_field::<5, 0>(inst);

    let op1_f = extract_field::<26, 25>(inst);
    let op1_f2 = extract_field::<26, 24>(inst);

    match (op0, op1) {
        (0b010, _) if op1_f == 0b0 => todo!(), // B.cond
        (0b110, _) if op1_f2 == 0b00 => decode_except_gen(inst),
        (0b110, 0b01000000110010) if op2 == 0b11111 => decode_hints(inst), // hints
        (0b110, _) if op1_f == 0b1 => decode_ubr_reg(inst),
        (0b000 | 0b100, _) => decode_ubr_imm(inst),
        _ => Err(Error::NotSupported),
    }
}

// Exception generation
const fn decode_except_gen(inst: u32) -> Result<Inst> {
    let opc = extract_field::<24, 21>(inst);
    let imm16 = extract_field::<21, 5>(inst);
    let op2 = extract_field::<5, 2>(inst);
    let ll = extract_field::<2, 0>(inst);

    let imm = imm16 as u16;

    let inst = match (opc, op2, ll) {
        (0b000, 0b000, 0b01) => Inst::Svc { imm },
        _ => return Err(Error::NotSupported),
    };

    Ok(inst)
}

// Hints
const fn decode_hints(inst: u32) -> Result<Inst> {
    let crm = extract_field::<12, 8>(inst);
    let op2 = extract_field::<8, 5>(inst);

    match (crm, op2) {
        (0b0000, 0b000) => Ok(Inst::Nop),
        _ => Err(Error::NotSupported),
    }
}

// Unconditioned branch, register
const fn decode_ubr_reg(inst: u32) -> Result<Inst> {
    let opc = extract_field::<25, 21>(inst);
    let op2 = extract_field::<21, 16>(inst);
    let op3 = extract_field::<16, 10>(inst);
    let rn = extract_field::<10, 5>(inst);
    let op4 = extract_field::<5, 0>(inst);

    match (opc, op3) {
        _ if op2 != 0b11111 => Err(Error::NotSupported),
        (0b0000, 0b000000) if op4 == 0b00000 => decode_br(rn),
        (0b0001, 0b000000) if op4 == 0b00000 => decode_blr(rn),
        (0b0010, 0b000000) if op4 == 0b00000 => decode_ret(rn),
        _ => Err(Error::NotSupported),
    }
}

const fn decode_br(rn: u32) -> Result<Inst> {
    Ok(Inst::Br { rn: rn as u8 })
}

const fn decode_blr(rn: u32) -> Result<Inst> {
    Ok(Inst::Blr { rn: rn as u8 })
}

const fn decode_ret(rn: u32) -> Result<Inst> {
    Ok(Inst::Ret { rn: rn as u8 })
}

// Unconditioned branch, immediate
const fn decode_ubr_imm(inst: u32) -> Result<Inst> {
    let op = extract_field::<32, 31>(inst);
    let imm26 = extract_field::<26, 0>(inst);
    let imm = sign_extend_64::<26>(imm26);
    let label = imm * 4;

    Ok(match op {
        0b0 => Inst::B { label },
        0b1 => Inst::Bl { label },
        _ => unreachable!(),
    })
}

// Data Processing -- Register
const fn decode_dp_reg(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<31, 30>(inst);
    let op1 = extract_field::<29, 28>(inst);
    let op2 = extract_field::<25, 21>(inst);
    let op3 = extract_field::<16, 10>(inst);

    match (op1, op2) {
        (0b0, 0b1000 | 0b1010 | 0b1100 | 0b1110) => decode_addsub_sreg(inst),
        (0b1, 0b0000) if op3 == 0b000000 => decode_addsub_carry(inst),
        _ => Err(Error::NotSupported),
    }
}

const fn decode_addsub_sreg(inst: u32) -> Result<Inst> {
    let sf = extract_field::<32, 31>(inst);
    let op = extract_field::<31, 30>(inst);
    let s = extract_field::<30, 29>(inst);
    // shift
    let sh = extract_field::<24, 22>(inst);
    let rm = extract_field::<21, 16>(inst);
    let imm6 = extract_field::<16, 10>(inst);
    let rn = extract_field::<10, 5>(inst);
    let rd = extract_field::<5, 0>(inst);

    let sf = sf == 1;
    let Some(sh) = ShiftType::from_u32(sh) else {
        return Err(Error::NotSupported);
    };

    match (op, s) {
        (0b0, 0b0) => decode_add_sreg(sf, sh, rm, imm6, rn, rd),
        (0b1, 0b0) => decode_sub_sreg(sf, sh, rm, imm6, rn, rd),
        (0b0, 0b1) => decode_adds_sreg(sf, sh, rm, imm6, rn, rd),
        (0b1, 0b1) => decode_subs_sreg(sf, sh, rm, imm6, rn, rd),
        _ => unreachable!(),
    }
}

const fn decode_addsub_carry(inst: u32) -> Result<Inst> {
    let sf = extract_field::<32, 31>(inst);
    let op = extract_field::<31, 30>(inst);
    let s = extract_field::<30, 29>(inst);
    let rm = extract_field::<21, 16>(inst);
    let rn = extract_field::<10, 9>(inst);
    let rd = extract_field::<5, 0>(inst);

    let sf = sf == 1;

    match (op, s) {
        (0b0, 0b0) => decode_adc(sf, rm, rn, rd),
        (0b1, 0b0) => decode_sbc(sf, rm, rn, rd),
        (0b0, 0b1) => decode_adcs(sf, rm, rn, rd),
        (0b1, 0b1) => decode_sbcs(sf, rm, rn, rd),
        _ => unreachable!(),
    }
}

const fn decode_adc(sf: bool, rm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Adc {
        rd: rd as u8,
        rn: rn as u8,
        rm: rm as u8,
        sf,
    })
}

const fn decode_sbc(sf: bool, rm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Sbc {
        rd: rd as u8,
        rn: rn as u8,
        rm: rm as u8,
        sf,
    })
}

const fn decode_adcs(sf: bool, rm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Adcs {
        rd: rd as u8,
        rn: rn as u8,
        rm: rm as u8,
        sf,
    })
}

const fn decode_sbcs(sf: bool, rm: u32, rn: u32, rd: u32) -> Result<Inst> {
    Ok(Inst::Sbcs {
        rd: rd as u8,
        rn: rn as u8,
        rm: rm as u8,
        sf,
    })
}

const fn decode_add_sreg(
    sf: bool,
    sh: ShiftType,
    rm: u32,
    imm: u32,
    rn: u32,
    rd: u32,
) -> Result<Inst> {
    Ok(Inst::Add {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::shifted_reg(rm as u8, sh, imm as u8, sf),
        sf,
    })
}

const fn decode_sub_sreg(
    sf: bool,
    sh: ShiftType,
    rm: u32,
    imm: u32,
    rn: u32,
    rd: u32,
) -> Result<Inst> {
    Ok(Inst::Sub {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::shifted_reg(rm as u8, sh, imm as u8, sf),
        sf,
    })
}

const fn decode_adds_sreg(
    sf: bool,
    sh: ShiftType,
    rm: u32,
    imm: u32,
    rn: u32,
    rd: u32,
) -> Result<Inst> {
    Ok(Inst::Adds {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::shifted_reg(rm as u8, sh, imm as u8, sf),
        sf,
    })
}

const fn decode_subs_sreg(
    sf: bool,
    sh: ShiftType,
    rm: u32,
    imm: u32,
    rn: u32,
    rd: u32,
) -> Result<Inst> {
    Ok(Inst::Subs {
        rd: rd as u8,
        rn: rn as u8,
        op2: Operand::shifted_reg(rm as u8, sh, imm as u8, sf),
        sf,
    })
}

const fn extract_field<const S: u32, const E: u32>(inst: u32) -> u32 {
    assert!(
        S > E,
        "the start offset must be greater than the end offset"
    );
    assert!(
        S <= u32::BITS,
        "the start offset must be less than or equals the bit size of u32"
    );
    let pad = u32::BITS - S;
    let mask = (((u32::MAX << pad) >> pad) >> E) << E;

    (inst & mask) >> E
}

const fn sign_extend_64<const I: u32>(imm: u32) -> i64 {
    let pad = (u64::BITS - I) as u64;
    (((imm as u64) << pad) as i64) >> pad
}

#[cfg(test)]
mod tests {
    use crate::inst::InstDecoder;

    #[test]
    fn decode() {
        let inst = [
            0x7e, 0x05, 0x00, 0x14, 0x46, 0x14, 0x00, 0xf0, 0xc0, 0x03, 0x5f, 0xd6, 0xff, 0x83,
            0x01, 0xd1,
        ];

        let mut decoder = InstDecoder::new(inst.as_slice());

        while let Ok(inst) = decoder.decode_inst() {
            println!("{:?}", inst);
        }
    }
}
