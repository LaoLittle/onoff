use std::fs::{write, File};
use std::io::Read;
use zip::ZipArchive;

fn main() {
    let file = File::open("test-resources/Firmware_17.0.1.zip").unwrap();

    let mut archive = ZipArchive::new(file).unwrap();

    for i in 0..archive.len() {
        let f = archive.by_index(i).unwrap();
    }
}
