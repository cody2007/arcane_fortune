// Code from: https://github.com/BurntSushi/rust-snappy/ Accessed July 20, 2020

//pub use crate::error::{Error, Result};
//pub mod error; pub use error::*;

/// We don't permit compressing a block bigger than what can fit in a u32.
pub const MAX_INPUT_SIZE: u64 = std::u32::MAX as u64;

/// The maximum number of bytes that we process at once. A block is the unit
/// at which we scan for candidates for compression.
pub const MAX_BLOCK_SIZE: usize = 1 << 16;

pub mod bytes; //pub use bytes::*;
pub mod compress; //pub use compress::*;
pub mod crc32; //pub use crc32::*;
pub mod crc32_table; //pub use crc32_table::*;
pub mod decompress; //pub use decompress::*;
pub mod error; //pub use error::*;
pub mod frame; //pub use frame::*;
pub mod raw; //pub use raw::*;
pub mod read; //pub use read::*;
pub mod tag; //pub use tag::*;
pub mod write; //pub use write::*;

