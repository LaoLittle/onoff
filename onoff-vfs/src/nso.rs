use bitflags::bitflags;
use onoff_core::error::{Error, Result};
use std::io::Read;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct NsoFlags: u32 {
        const TextCompressed = 0b00000001;
        const RodataCompressed = 0b00000010;
        const DataCompressed = 0b00000100;
        const TextCheckHash = 0b000001000;
        const RodataCheckHash = 0b000010000;
        const DataCheckHash = 0b000100000;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct NsoSegmentHeader {
    pub file_offset: u32,
    pub memory_offset: u32,
    pub section_size: u32,
}
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct NsoRodataRelativeSegmentHeader {
    pub offset: u32,
    pub size: u32,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct NsoHeader {
    pub magic: u32,
    pub version: u32,
    pub reserved_1: [u8; 4],
    pub flags: NsoFlags,
    pub text_segment: NsoSegmentHeader,
    pub module_name_offset: u32,
    pub rodata_segment: NsoSegmentHeader,
    pub module_name_size: u32,
    pub data_segment: NsoSegmentHeader,
    pub bss_size: u32,
    pub module_id: [u8; 0x20],
    pub text_file_size: u32,
    pub rodata_file_size: u32,
    pub data_file_size: u32,
    pub reserved_2: [u8; 0x1C],
    pub rodata_api_info_segment: NsoRodataRelativeSegmentHeader,
    pub rodata_dynstr_segment: NsoRodataRelativeSegmentHeader,
    pub rodata_dynsym_segment: NsoRodataRelativeSegmentHeader,
    pub text_hash: [u8; 0x20],
    pub rodata_hash: [u8; 0x20],
    pub data_hash: [u8; 0x20],
}

const NSO_HEADER_SIZE: usize = core::mem::size_of::<NsoHeader>();

#[derive(Debug, Clone)]
pub struct Nso {
    header: NsoHeader,
    text_section: Vec<u8>,
    rodata_section: Vec<u8>,
    data_section: Vec<u8>,
}

impl Nso {
    pub const MAGIC: u32 = u32::from_le_bytes(*b"NSO0");

    pub fn new(mut reader: impl Read) -> Result<Self> {
        fn extract_section(
            reader: &mut impl Read,
            file_size: u32,
            offset: u32,
            compressed: bool,
            section_size: u32,
            decomp_buf: &mut Vec<u8>,
            pos: &mut usize,
        ) -> Result<Vec<u8>> {
            let mut section = vec![0u8; file_size as usize];
            let next = (offset as usize)
                .checked_sub(*pos)
                .ok_or(Error::InvalidNsoHeader)?;

            reader.read_exact(&mut section[..next])?;
            reader.read_exact(&mut section)?;

            *pos += file_size as usize;

            if compressed {
                decomp_buf.resize(section_size as usize, 0);
                lz4_flex::decompress_into(&section, decomp_buf)
                    .map_err(|_| Error::InvalidNsoHeader)?;
                core::mem::swap(decomp_buf, &mut section);
            }

            Ok(section)
        }

        let mut buf = [0u8; NSO_HEADER_SIZE];
        reader.read_exact(&mut buf)?;

        let header: NsoHeader = unsafe { core::mem::transmute(buf) };

        if header.magic != Self::MAGIC {
            return Err(Error::InvalidNsoHeader);
        }

        let mut curr = NSO_HEADER_SIZE;

        let mut decomp_buf: Vec<u8> = Vec::new();

        let text_section = extract_section(
            &mut reader,
            header.text_file_size,
            header.text_segment.file_offset,
            header.flags.contains(NsoFlags::TextCompressed),
            header.text_segment.section_size,
            &mut decomp_buf,
            &mut curr,
        )?;
        let rodata_section = extract_section(
            &mut reader,
            header.rodata_file_size,
            header.rodata_segment.file_offset,
            header.flags.contains(NsoFlags::RodataCompressed),
            header.rodata_segment.section_size,
            &mut decomp_buf,
            &mut curr,
        )?;
        let data_section = extract_section(
            &mut reader,
            header.data_file_size,
            header.data_segment.file_offset,
            header.flags.contains(NsoFlags::DataCompressed),
            header.data_segment.section_size,
            &mut decomp_buf,
            &mut curr,
        )?;

        Ok(Self {
            header,
            text_section,
            rodata_section,
            data_section,
        })
    }

    pub fn bss(&self) {}
}

#[cfg(test)]
mod tests {
    use crate::nso::Nso;
    use std::fs::File;

    #[test]
    fn parse() {
        let nso = Nso::new(File::open("../test-resources/test_helloworld.nso").unwrap()).unwrap();

        assert!(!nso.text_section.is_empty());
    }
}
