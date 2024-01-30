use crate::xts_reader::XtsReader;
use onoff_core::error::{Error, Result};
use smallvec::SmallVec;
use std::io::Read;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum DistributionType {
    System = 0,
    Gamecard = 1,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ContentType {
    Program = 0,
    Meta = 1,
    Control = 2,
    Manual = 3,
    Data = 4,
    PublicData = 5,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct RSASignature {
    part_1: [u8; 0x20],
    part_2: [u8; 0x20],
    part_3: [u8; 0x20],
    part_4: [u8; 0x20],
    part_5: [u8; 0x20],
    part_6: [u8; 0x20],
    part_7: [u8; 0x20],
    part_8: [u8; 0x20],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct SdkAddonVersion {
    unk: u8,
    micro: u8,
    minor: u8,
    major: u8,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct FileSystemEntry {
    start_offset: u32,
    end_offset: u32,
    hash_offsets: u32,
    reserved: u32,
}

impl FileSystemEntry {
    pub fn has_info(&self) -> bool {
        self.start_offset != 0 || self.end_offset != 0
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct Sha256Hash {
    hash: [u8; 0x20],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum KeyAreaEncryptionKeyIndex {
    Application = 0,
    Ocean = 1,
    System = 2,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(C)]
pub struct KeyArea {
    aes_xts_key: [u8; 0x20],
    aes_ctr_key: [u8; 0x10],
    unk_key: [u8; 0x10],
}

impl KeyArea {
    pub fn empty() -> Self {
        Self {
            aes_xts_key: [0; 0x20],
            aes_ctr_key: [0; 0x10],
            unk_key: [0; 0x10],
        }
    }

    pub fn from_slice(slice: &[u8]) -> Self {
        let (xts, r) = slice.split_at(0x20);
        let (ctr, unk) = r.split_at(0x10);

        Self {
            aes_xts_key: xts.try_into().unwrap(),
            aes_ctr_key: ctr.try_into().unwrap(),
            unk_key: unk.try_into().unwrap(),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const _ as *const u8, std::mem::size_of::<Self>())
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self as *mut _ as *mut u8, std::mem::size_of::<Self>())
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct RightsID(pub [u8; 0x10]);

impl RightsID {
    pub const ZERO: Self = Self([0; 0x10]);

    pub fn is_zero(&self) -> bool {
        self.0 == Self::ZERO.0
    }
}

pub const MAX_FILESYSTEM_COUNT: usize = 4;
pub const SECTOR_SIZE: usize = 0x200;
pub const MEDIA_UNIT_SIZE: usize = 0x200;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct NcaHeader {
    pub header_rsa_sig_1: RSASignature,
    pub header_rsa_sig_2: RSASignature,
    pub magic: u32,
    pub dist_type: DistributionType,
    pub content_type: ContentType,
    pub key_generation_old: u8,
    pub key_area_encryption_key_index: KeyAreaEncryptionKeyIndex,
    pub content_size: usize,
    pub program_id: u64,
    pub content_idx: u32,
    pub sdk_addon_ver: SdkAddonVersion,
    pub key_generation: u8,
    pub header_1_signature_key_generation: u8,
    pub reserved: [u8; 0xE],
    pub rights_id: RightsID,
    pub fs_entries: [FileSystemEntry; MAX_FILESYSTEM_COUNT],
    pub fs_header_hashes: [Sha256Hash; MAX_FILESYSTEM_COUNT],
    pub encrypted_key_area: KeyArea,
    pub reserved_1: [u8; 0x20],
    pub reserved_2: [u8; 0x20],
    pub reserved_3: [u8; 0x20],
    pub reserved_4: [u8; 0x20],
    pub reserved_5: [u8; 0x20],
    pub reserved_6: [u8; 0x20],
}

impl NcaHeader {
    pub const MAGIC: u32 = u32::from_le_bytes(*b"NCA3");

    #[inline]
    pub fn get_key_generation(self) -> u8 {
        let base_key_gen = {
            if self.key_generation_old < self.key_generation {
                self.key_generation
            } else {
                self.key_generation_old
            }
        };

        if base_key_gen > 0 {
            // Both 0 and 1 are master key 0...
            base_key_gen - 1
        } else {
            base_key_gen
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum FileSystemType {
    RomFs = 0,
    PartitionFs = 1,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum HashType {
    Auto = 0,
    HierarchicalSha256 = 2,
    HierarchicalIntegrity = 3,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum EncryptionType {
    Auto = 0,
    None = 1,
    AesCtrOld = 2,
    AesCtr = 3,
    AesCtrEx = 4,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct HierarchicalSha256 {
    hash_table_hash: Sha256Hash,
    block_size: u32,
    unk_2: u32,
    hash_table_offset: u64,
    hash_table_size: usize,
    pfs0_offset: u64,
    pfs0_size: usize,
    reserved_1: [u8; 0x20],
    reserved_2: [u8; 0x20],
    reserved_3: [u8; 0x20],
    reserved_4: [u8; 0x20],
    reserved_5: [u8; 0x20],
    reserved_6: [u8; 0x10],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct HierarchicalIntegrityLevelInfo {
    offset: u64,
    size: usize,
    block_size_log2: u32,
    reserved: [u8; 0x4],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct HierarchicalIntegrity {
    magic: u32,
    magic_num: u32,
    maybe_master_hash_size: u32,
    unk_7: u32,
    levels: [HierarchicalIntegrityLevelInfo; 6],
    reserved: [u8; 0x20],
    hash: Sha256Hash,
}

impl HierarchicalIntegrity {
    pub const MAGIC: u32 = u32::from_le_bytes(*b"IVFC");
}

#[derive(Copy, Clone)]
#[repr(C)]
pub union HashInfo {
    hierarchical_sha256: HierarchicalSha256,
    hierarchical_integrity: HierarchicalIntegrity,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct BucketRelocationInfo {
    offset: u64,
    size: usize,
    magic: u32,
    unk_1: u32,
    unk_2: i32,
    unk_3: u32,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct PatchInfo {
    info: BucketRelocationInfo,
    info_2: BucketRelocationInfo,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct BucketInfo {
    offset: u64,
    size: usize,
    header: [u8; 0x10],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct SparseInfo {
    pub bucket: BucketInfo,
    pub physical_offset: u64,
    pub generation: u16,
    pub reserved: [u8; 6],
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct FileSystemHeader {
    version: u16,
    fs_type: FileSystemType,
    hash_type: HashType,
    encryption_type: EncryptionType,
    pad: [u8; 0x3],
    hash_info: HashInfo,
    patch_info: PatchInfo,
    ctr: u64,
    sparse_info: SparseInfo,
    reserved_1: [u8; 0x20],
    reserved_2: [u8; 0x20],
    reserved_3: [u8; 0x20],
    reserved_4: [u8; 0x20],
    reserved_5: [u8; 0x8],
}

const NCA_HEADER_SIZE: usize = core::mem::size_of::<NcaHeader>();
const FS_HEADER_SIZE: usize = core::mem::size_of::<FileSystemHeader>();

pub struct NcaReader<R> {
    read: XtsReader<R>,
    header: NcaHeader,
    fs_headers: SmallVec<FileSystemHeader, 4>,
}

impl<R: Read> NcaReader<R> {
    pub const MAGIC0: u32 = u32::from_le_bytes(*b"NCA0");
    pub const MAGIC1: u32 = u32::from_le_bytes(*b"NCA1");
    pub const MAGIC2: u32 = u32::from_le_bytes(*b"NCA2");
    pub const MAGIC3: u32 = u32::from_le_bytes(*b"NCA3");

    pub const BUFFER_SIZE: usize = 0xC00;

    pub fn new(read: R) -> Result<Self> {
        let s = "aeaab1ca08adf9bef12991f369e3c567d6881e4e4a6a47a51f6e4877062d542d";
        let mut hkey = [0; 32];
        for i in 0..32 {
            let i2 = i * 2;
            hkey[i] = u8::from_str_radix(&s[i2..=i2 + 1], 16).unwrap();
        }

        let mut xts = XtsReader::new(read, &hkey);
        let mut header_buf = [0u8; NCA_HEADER_SIZE];

        xts.decrypt(&mut header_buf, SECTOR_SIZE, 0)?;

        let header: NcaHeader = unsafe { core::mem::transmute(header_buf) };

        if header.magic != Self::MAGIC3 {
            return Err(Error::InvalidNcaHeader);
        }

        let mut fs_buf = [0u8; FS_HEADER_SIZE * MAX_FILESYSTEM_COUNT];

        xts.decrypt(&mut fs_buf, SECTOR_SIZE, 2)?;

        let fs_headers: [FileSystemHeader; MAX_FILESYSTEM_COUNT] =
            unsafe { core::mem::transmute(fs_buf) };

        let mut fs_vec = SmallVec::new();
        for i in 0..MAX_FILESYSTEM_COUNT {
            if header.fs_entries[i].has_info() {
                fs_vec.push(fs_headers[i]);
            }
        }

        Ok(Self {
            read: xts,
            header,
            fs_headers: fs_vec,
        })
    }

    pub fn read(&mut self) {}
}

#[cfg(test)]
mod tests {
    use crate::nca::NcaReader;
    use std::fs::File;

    #[test]
    fn test_read() {
        let reader = NcaReader::new(File::open("../test-resources/test.nca").unwrap()).unwrap();
    }
}
