use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

use onoff_core::error::{Error, Result};

pub trait Memory {
    type Addr;

    fn read8(&self, addr: Self::Addr) -> Result<u8>;

    fn write8(&mut self, addr: Self::Addr, src: u8) -> Result<()>;

    fn read16(&self, addr: Self::Addr) -> Result<u16>;

    fn write16(&mut self, addr: Self::Addr, src: u16) -> Result<()>;

    fn read32(&self, addr: Self::Addr) -> Result<u32>;

    fn write32(&mut self, addr: Self::Addr, src: u32) -> Result<()>;

    fn read64(&self, addr: Self::Addr) -> Result<u64>;

    fn write64(&mut self, addr: Self::Addr, src: u64) -> Result<()>;
}

pub const PAGE_SIZE: usize = 4096;

pub struct Page(pub [u8; PAGE_SIZE]);

impl Page {
    #[inline]
    pub const fn new() -> Self {
        Self([0u8; PAGE_SIZE])
    }

    pub fn read_exact(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        let offset = offset as usize;
        let Some(mem) = self.0.get(offset..offset + buf.len()) else {
            return Err(Error::InvalidMemoryAddress);
        };

        buf.copy_from_slice(mem);

        Ok(())
    }

    pub fn write_exact(&mut self, buf: &[u8], offset: u64) -> Result<()> {
        let offset = offset as usize;
        let Some(mem) = self.0.get_mut(offset..offset + buf.len()) else {
            return Err(Error::InvalidMemoryAddress);
        };

        mem.copy_from_slice(buf);

        Ok(())
    }
}

impl Debug for Page {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("<Page>")
    }
}

#[derive(Debug)]
pub struct PageMemory {
    pages: HashMap<u64, Page>
}

impl PageMemory {
    #[inline]
    pub fn new() -> Self {
        Self {
            pages: HashMap::new()
        }
    }

    pub fn alloc(&mut self, addr: u64) -> &mut Page {
        let (base, _) = align_4k(addr);

        self.pages.entry(base).or_insert(Page::new())
    }

    // write bytes into certain page(s) provided by addr argument
    // alloc if there were no available pages
    pub fn write_exact(&mut self, buf: &[u8], addr: u64) -> Result<()> {
        let (mut hi, lo) = (addr & !4095, addr & 4095);

        let remain = PAGE_SIZE - lo as usize;

        // |Page .. ++| |Page ++ ..|
        for chunk in buf.chunks(PAGE_SIZE) {
            let (f, b) = if remain > chunk.len() {
                (chunk, [].as_slice())
            } else {
                chunk.split_at(remain)
            };

            self.alloc(hi).write_exact(f, lo)?;
            hi += 4096;

            if !b.is_empty() {
                self.alloc(hi).write_exact(b, 0)?;
            }
        }

        Ok(())
    }

    #[inline]
    pub fn get_page(&self, addr: u64) -> Result<&Page> {
        self.pages.get(&addr).ok_or(Error::InvalidMemoryAddress)
    }

    #[inline]
    pub fn get_page_mut(&mut self, addr: u64) -> Result<&mut Page> {
        self.pages.get_mut(&addr).ok_or(Error::InvalidMemoryAddress)
    }
}

impl Memory for PageMemory {
    type Addr = u64;

    fn read8(&self, addr: Self::Addr) -> Result<u8> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 1];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u8::from_le_bytes(buf))
    }

    fn write8(&mut self, addr: Self::Addr, src: u8) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }

    fn read16(&self, addr: Self::Addr) -> Result<u16> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 2];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u16::from_le_bytes(buf))
    }

    fn write16(&mut self, addr: Self::Addr, src: u16) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }

    fn read32(&self, addr: Self::Addr) -> Result<u32> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 4];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u32::from_le_bytes(buf))
    }

    fn write32(&mut self, addr: Self::Addr, src: u32) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }

    fn read64(&self, addr: Self::Addr) -> Result<u64> {
        let (hi, lo) = align_4k(addr);

        let mut buf = [0u8; 8];
        self.get_page(hi)?.read_exact(&mut buf, lo)?;

        Ok(u64::from_le_bytes(buf))
    }

    fn write64(&mut self, addr: Self::Addr, src: u64) -> Result<()> {
        let (hi, lo) = align_4k(addr);

        self.get_page_mut(hi)?.write_exact(&src.to_le_bytes(), lo)?;

        Ok(())
    }
}

// returns (hi, lo)
const fn align_4k(addr: u64) -> (u64, u64) {
    (addr & !4095, addr & 4095)
}

#[cfg(test)]
mod tests {
    use crate::mem::{Memory, PageMemory};
    use onoff_core::error::Result;

    #[test]
    fn page_mem() {
        let mut page_mem = PageMemory::new();


        for addr in [0, 4099, 10023] {
            page_mem.alloc(addr);

            mem_validation(&mut page_mem, addr).unwrap();
        }
    }

    fn mem_validation<M: Memory>(mem: &mut M, addr: M::Addr) -> Result<()>
    where M::Addr: Copy
    {
        let a = 114;
        mem.write8(addr, a)?;
        let b = mem.read8(addr)?;
        assert_eq!(a,b);

        let a = 514;
        mem.write16(addr, a)?;
        let b = mem.read16(addr)?;
        assert_eq!(a,b);

        let a = 1919810;
        mem.write32(addr, a)?;
        let b = mem.read32(addr)?;
        assert_eq!(a,b);

        let a = 1145141919810;
        mem.write64(addr, a)?;
        let b = mem.read64(addr)?;
        assert_eq!(a,b);

        Ok(())
    }
}