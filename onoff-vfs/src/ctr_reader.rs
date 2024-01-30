use aes::cipher::KeyIvInit;
use aes::Aes128;

pub struct CtrReader {
    ctr: u64,
}

impl CtrReader {
    fn decrypt(&mut self, buf: &mut [u8], key: &[u8]) {
        let ctr = ctr::Ctr128LE::<Aes128>::new_from_slices(key, &[]).unwrap();
    }

    #[inline]
    fn tweak_from_index(index: u128) -> [u8; 16] {
        index.to_be_bytes()
    }
}
