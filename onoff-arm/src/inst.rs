use onoff_core::error::{Error, Result};
use std::io::Read;

#[derive(Debug, Copy, Clone)]
pub enum Inst {
    B { label: i64 },
    Bl { label: i64 },
    Adr { rd: u8, label: i64 },
    Adrp { rd: u8, label: i64 },
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

fn decode_inst_u32(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<29, 24>(inst);

    let inst = match op0 {
        // Data Processing -- Immediate
        0b10000 | 0b10001 | 0b10010 | 0b10011 => decode_dp_imm(inst)?,
        // Branches, Exception Generating and System instructions
        0b10100 | 0b10101 | 0b10110 | 0b10111 => decode_bes(inst)?,
        _ => return Err(Error::NotSupported),
    };

    Ok(inst)
}

fn decode_dp_imm(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<26, 23>(inst);

    match op0 {
        0b000 | 0b001 => decode_pc_rel(inst),

        _ => Err(Error::NotSupported),
    }
}

fn decode_pc_rel(inst: u32) -> Result<Inst> {
    let op = extract_field::<32, 31>(inst);
    let immlo = extract_field::<31, 29>(inst);
    let immhi = extract_field::<24, 5>(inst);
    let rd = extract_field::<5, 0>(inst) as u8;

    let imm = sign_extend_64::<21>((immhi << 2) | immlo);

    let inst = match op {
        0 => Inst::Adr { rd, label: imm },
        1 => Inst::Adrp {
            rd,
            label: imm * 4096,
        },
        _ => unreachable!(),
    };

    Ok(inst)
}

fn decode_bes(inst: u32) -> Result<Inst> {
    let op0 = extract_field::<31, 29>(inst);

    match op0 {
        0b000 | 0b100 => decode_ubr_imm(inst),
        _ => Err(Error::NotSupported),
    }
}

fn decode_ubr_imm(inst: u32) -> Result<Inst> {
    let op = extract_field::<32, 31>(inst);
    let imm26 = extract_field::<26, 0>(inst);
    let imm = sign_extend_64::<26>(imm26);
    let label = imm * 4;

    Ok(match op {
        0 => Inst::B { label },
        1 => Inst::Bl { label },
        _ => unreachable!(),
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
            0x7e, 0x05, 0x00, 0x14, 0x05, 0x03, 0x00, 0xd0, 0x46, 0x14, 0x00, 0xf0, 0x00, 0x00,
            0x00, 0x90, 0x00, 0x00, 0x00, 0xb0,
        ];

        let mut decoder = InstDecoder::new(inst.as_slice());

        while let Ok(inst) = decoder.decode_inst() {
            println!("{:?}", inst);
        }
    }
}
