use aes::cipher::generic_array::GenericArray;
use aes::cipher::KeyInit;
use aes::Aes128;
use std::io;
use std::io::Read;
use xts_mode::Xts128;

pub struct XtsReader<R> {
    reader: R,
    xts: Xts128<Aes128>,
}

impl<R: Read> XtsReader<R> {
    pub fn new(reader: R, key: &[u8; 32]) -> Self {
        let (k1, k2) = key.split_at(16);
        let xts = Xts128::new(
            Aes128::new(GenericArray::from_slice(k1)),
            Aes128::new(GenericArray::from_slice(k2)),
        );

        Self { reader, xts }
    }

    pub fn decrypt(
        &mut self,
        buf: &mut [u8],
        sector_size: usize,
        index: u128,
    ) -> Result<(), io::Error> {
        self.reader.read_exact(buf)?;

        self.xts
            .decrypt_area(buf, sector_size, index, Self::tweak_from_index);

        Ok(())
    }

    #[inline]
    fn tweak_from_index(index: u128) -> [u8; 16] {
        index.to_be_bytes()
    }
}
