use std::io::prelude::*;
use std::fs::{File, create_dir_all};
use std::path::Path;
#[cfg(not(feature="cpu_only"))]
use crate::data_wrappers::TensorShape;
use crate::data_wrappers::FilterShape;
use crate::layers::*;

macro_rules! fn_headers{() => (
	fn sv_buf(&self, res: &mut Vec<u8>);
	fn type_str() -> &'static str;
)}

pub trait Sv {fn_headers!();}
pub trait SvO<T> {fn_headers!();}
pub trait SvStruct {
	fn sv(&self, save_dir: &str, file_nm: &str);
}

macro_rules! impl_ident{($type:ty, $type_str: expr) => (
	impl Sv for $type {
		fn sv_buf(&self, res: &mut Vec<u8>){ res.extend_from_slice(&self.to_le_bytes()); }
		fn type_str() -> &'static str {$type_str}
	}
	
	impl SvStruct for Vec<$type> {
		fn sv(&self, save_dir: &str, file_nm: &str) {
			sv_w_shape::<FilterShape, $type>(self, save_dir, file_nm, None);
		}
	}
)}

impl_ident!(u8, "<u1");
impl_ident!(u16, "<u2");
impl_ident!(f32, "<f4");
impl_ident!(u64, "<u8");

impl Sv for String {
	fn sv_buf(&self, res: &mut Vec<u8>){
		let s = self.clone();
		s.into_bytes().sv_buf(res);
	}
	
	fn type_str() -> &'static str {"<u1"}
}

impl <T: Sv> SvO <T> for Vec<T> {
	fn sv_buf(&self, res: &mut Vec<u8>){
		for d in self.iter() { // save vals
			d.sv_buf(res);
		}
	}
	fn type_str() -> &'static str {panic!("type error saving -- primitives should only use this")}
}

// writes numpy file
//	T: type, ex. f32 
pub fn sv_w_shape<T: ToString, D: Sv>(data: &Vec<D>, save_dir: &str, file_nm: &str, shape: Option<T>) {
	let mut buf: Vec<u8> = Vec::new();
	
	let magic_txt = String::from("NUMPY"); 
	0x93_u8.sv_buf(&mut buf);
	magic_txt.sv_buf(&mut buf);
	0x01_u8.sv_buf(&mut buf);
	0x00_u8.sv_buf(&mut buf);
	
	let shape_txt = if let Some(shape) = shape {
		shape.to_string()
	}else{
		format!("({},)", data.len())
	};
	
	let mut dict_str = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}}}",
			<D>::type_str(), shape_txt);
	
	// pad dict_str to ALIGNMENT
	{
		let total_len = 1 + magic_txt.len() + 4 + dict_str.len() + 1;
		
		const ALIGNMENT: usize = 16;
		let remainder = total_len % ALIGNMENT;
		let pad = if remainder != 0 {
			ALIGNMENT - remainder
		}else{
			0
		};
		
		for _ in 0..pad {dict_str.push(' ');}
	}
	dict_str.push('\n');
	
	(dict_str.len() as u16).sv_buf(&mut buf); // length of dict string
	dict_str.sv_buf(&mut buf);
	
	data.sv_buf(&mut buf);
	
	save_file(save_dir, &format!("{}.npy", file_nm), &buf);
}

// writes buf to file, no pre-processing added
pub fn save_file(save_dir: &str, file_nm: &str, buf: &Vec<u8>) {
	let file_nm = format!("{}/{}", save_dir, file_nm);
	let path = Path::new(&file_nm).as_os_str();
	if let Result::Ok(_) = create_dir_all(save_dir) {
		if let Result::Ok(ref mut file) = File::create(path) {
			if let Result::Err(_) = file.write_all(buf) {
				panic!("failed writing file: {}", file_nm);
			}
		}else {panic!("failed opening file for writing: {}", file_nm);}
	}else {panic!("failed creating directory for file: {}", file_nm);}
}

#[cfg(not(feature="cpu_only"))]
macro_rules! impl_filter_ld_sv{($filter: ty) => {
	impl $filter {
		pub fn ld(&self, save_dir: &str, file_nm: &str) {
			let vals = load_numpy::<f32>(&format!("{}/{}.npy", save_dir, file_nm));
			self.mem.set(&vals);
		}
	}
	
	impl SvStruct for $filter {
		fn sv(&self, save_dir: &str, file_nm: &str) {
			sv_w_shape(&self.ret(), save_dir, file_nm, Some(self.shape));
		}
	}

}}

#[cfg(not(feature="cpu_only"))]
impl_filter_ld_sv!(Filter);
#[cfg(not(feature="cpu_only"))]
impl_filter_ld_sv!(Filter3);

macro_rules! impl_filterCPU_ld{($filter: ty) => {
	impl $filter {
		pub fn ld(&mut self, save_dir: &str, file_nm: &str) {
			self.mem = load_numpy::<f32>(&format!("{}/{}.npy", save_dir, file_nm));
		}
	}
}}

impl_filterCPU_ld!(FilterCPU);
impl_filterCPU_ld!(Filter3CPU);

#[cfg(not(feature="cpu_only"))]
impl SvStruct for Tensor {
	fn sv(&self, save_dir: &str, file_nm: &str) {
		sv_w_shape(&self.ret(), save_dir, file_nm, Some(self.shape));
	}
}

#[cfg(not(feature="cpu_only"))]
impl SvStruct for gpuMem {
	fn sv(&self, save_dir: &str, file_nm: &str) {
		sv_w_shape::<f32, f32>(&self.ret(self.n_elements), save_dir, file_nm, None);
	}
}

#[cfg(not(feature="cpu_only"))]
impl SvStruct for RNNData {
	fn sv(&self, save_dir: &str, file_nm: &str) {
		sv_w_shape(&self.ret(), save_dir, file_nm, Some(TensorShape::from(self)));
	}
}

#[cfg(not(feature="cpu_only"))]
impl SvStruct for Output {
	fn sv(&self, save_dir: &str, file_nm: &str) {
		run_output!(self => sv(save_dir, file_nm));
	}
}

// return buffer of file "nm"
fn read_file(nm: &str) -> Vec<u8> {
	let mut buf = Vec::new();
	
	let nm = Path::new(nm).as_os_str().to_str().unwrap();
	
	if let Result::Ok(ref mut file) = File::open(nm) {
		if let Result::Err(_) = file.read_to_end(&mut buf) {
			panic!("Failed reading configuration file: {}", nm);
		}
	}else {
		panic!("Failed opening file for reading: {}", nm);
	}
	buf
}

#[cfg(not(feature="cpu_only"))]
pub fn ld_gpumem_if_exists(save_dir: &str, file_nm: &str) -> Option<gpuMem> {
	let path_nm = format!("{}/{}.npy", save_dir, file_nm);
	if Path::new(&path_nm).exists() {
		let vals = load_numpy::<f32>(&path_nm);
		Some( gpuMem::init(cudnnDataType_t::CUDNN_DATA_FLOAT, &vals) )
	}else{ None }
}

use std::mem::size_of;
pub trait FromBytes {
	fn from_bytes(buf: &Vec<u8>, start_pos: &mut usize) -> Self;
}

macro_rules! impl_from_bytes{($type: ty) => {
	impl FromBytes for $type {
		fn from_bytes(buf: &Vec<u8>, start_pos: &mut usize) -> Self {
			const SZ: usize = size_of::<$type>();
			let mut cropped: [u8; SZ] = Default::default();
			cropped.copy_from_slice(&buf[*start_pos..*start_pos + SZ]); *start_pos += SZ;
			Self::from_le_bytes(cropped)
		}
	}
}}

impl_from_bytes!(u8);
impl_from_bytes!(f32);
impl_from_bytes!(u64);
impl_from_bytes!(u32);

pub fn load_numpy<T: FromBytes>(file_nm: &str) -> Vec<T> {
	let buf = read_file(&file_nm);
	
	let mut start_pos = None;
	
	for (i, v) in buf.iter().enumerate() {
		if *v == '\n' as u8 {
			start_pos = Some(i+1);
			break;
		}
	}
	let mut start_pos = start_pos.unwrap();
	let mut vals = Vec::with_capacity(buf.len() - start_pos);
	
	while start_pos < buf.len() {
		vals.push(<T>::from_bytes(&buf, &mut start_pos));
	}
	vals
}

use std::str::FromStr;
impl Filter3CPU {
	pub fn load_numpy(file_nm: &str) -> Self {
		let buf = read_file(&file_nm);
		
		let mut data_start_pos = {
			let mut start_pos = None;
			
			for (i, v) in buf.iter().enumerate() {
				if *v == '\n' as u8 {
					start_pos = Some(i+1);
					break;
				}
			}
			start_pos.unwrap()
		};
		
		// will contain something like:
		//	{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1024, 1024)}   
		let descr_txt = {
			let mut descr_start_pos = None;
			for (i, v) in buf.iter().take(data_start_pos).enumerate() {
				if *v == '{' as u8 {
					descr_start_pos = Some(i+1);
					break;
				}
			}
			let descr_start_pos = descr_start_pos.unwrap();
			assert!(descr_start_pos < data_start_pos,
					"{} {}", descr_start_pos, data_start_pos);
			
			String::from_utf8(buf[descr_start_pos..data_start_pos].to_vec()).unwrap()
		};
		
		assert!(descr_txt.contains("'descr': '<f4'")); // check data type correct
		
		let shape = {
			let shape_txt = descr_txt.split("'shape':").collect::<Vec<&str>>()[1];
			if let Ok(shape) = Filter3Shape::from_str(shape_txt) {
				shape
			}else{
				panic!("could not parse shape: {}", shape_txt);
			}
		};
		
		////////// load values
		let mut vals = Vec::with_capacity(buf.len() - data_start_pos);
		
		while data_start_pos < buf.len() {
			vals.push(f32::from_bytes(&buf, &mut data_start_pos));
		}
		
		Self {
			mem: vals,
			shape: FilterShape {
					k: shape.dim1,
					c: shape.dim2,
					h: shape.dim3,
					w: 1
			}
		}
	}
}

