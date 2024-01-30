use std::io::Write;
use vfs::error::VfsErrorKind;
use vfs::{FileSystem, SeekAndRead, VfsError, VfsMetadata, VfsResult};

#[derive(Debug)]
pub struct PartitionFileSystem {}

impl FileSystem for PartitionFileSystem {
    fn read_dir(&self, path: &str) -> VfsResult<Box<dyn Iterator<Item = String> + Send>> {
        todo!()
    }

    fn create_dir(&self, path: &str) -> VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn open_file(&self, path: &str) -> VfsResult<Box<dyn SeekAndRead + Send>> {
        todo!()
    }

    fn create_file(&self, path: &str) -> VfsResult<Box<dyn Write + Send>> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn append_file(&self, path: &str) -> VfsResult<Box<dyn Write + Send>> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn metadata(&self, path: &str) -> VfsResult<VfsMetadata> {
        todo!()
    }

    fn exists(&self, path: &str) -> VfsResult<bool> {
        todo!()
    }

    fn remove_file(&self, path: &str) -> VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }

    fn remove_dir(&self, path: &str) -> VfsResult<()> {
        Err(VfsErrorKind::NotSupported.into())
    }
}
