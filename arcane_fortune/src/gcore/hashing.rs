use std::hash::{Hasher, BuildHasherDefault};
use std::collections::{HashMap, HashSet};

use crate::map::{Map, MapEx};
use crate::zones::{ZoneEx, FogVars};

pub struct HashStruct64 {
	hash: u64,
}

impl Default for HashStruct64 {
	fn default() -> Self {
		HashStruct64 {hash: 0}
	}
}

impl Hasher for HashStruct64 {
	#[inline]
	fn write(&mut self, msg: &[u8]) {
		//let mut vals = [0; 4];
		//vals.copy_from_slice(&msg[0..4]);
		//self.hash = u32::from_le_bytes(vals);
		for (i,m) in msg.iter().enumerate() {
			self.hash += (*m as u64) << (i*8);
		}
	}
	
	#[inline]
	fn finish(&self) -> u64 {
		self.hash
	}
}

pub type HashedMap<'rt> = HashMap<u64, Map<'rt>, BuildHasherDefault<HashStruct64>>;
pub type HashedMapEx<'bt,'ut,'rt,'dt> = HashMap<u64, MapEx<'bt,'ut,'rt,'dt>, BuildHasherDefault<HashStruct64>>;
pub type HashedMapZoneEx = HashMap<u64, ZoneEx, BuildHasherDefault<HashStruct64>>;
pub type HashedFogVars<'bt,'ut,'rt,'dt> = HashMap<u64, FogVars<'bt,'ut,'rt,'dt>, BuildHasherDefault<HashStruct64>>;
pub type HashedCoords = HashSet<u64, BuildHasherDefault<HashStruct64>>;

