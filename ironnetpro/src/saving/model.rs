use std::fs::File;
use std::path::Path;
use std::io::prelude::*;

#[derive(Copy, Clone, PartialEq)]
pub struct ScreenSz {
	pub h: usize,
	pub w: usize,
	pub sz: usize,
}

pub struct KeyPair {
	key: String,
	val: String,
	sz: Option<ScreenSz>
}

// return buffer of file "nm"
pub fn read_file(nm: &str) -> Vec<u8> {
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

// parse buffer into KeyPairs 
pub fn config_parse(buf: Vec<u8>) -> Vec<Vec<KeyPair>> {
	let mut key_sets: Vec<Vec<KeyPair>> = Vec::new();
	let mut block = false;
	enum BlockType {
		None, // standard format key of variable_name: value
		KeepSpaces, // delims: ()
		TrimSpaces // delims: []
	}
	let mut sub_block = BlockType::None;
	
	for (line_num, line) in String::from_utf8(buf).unwrap().lines().enumerate() {
		// remove whitepsace
		let line_trim = line.trim();
		
		// start block
		if !block && line_trim == "{" {
			block = true;
			key_sets.push(Vec::new());
			continue;
		}
		
		// not in a block, ignore text until new block starts
		if !block {continue;}
		
		// end block
		if line_trim == "}" {
			match sub_block {
				BlockType::KeepSpaces => {panic!("Config file parsing at line: {}. There may be a missing \")\".", line_num+1);}
				BlockType::TrimSpaces => {panic!("Config file parsing at line: {}. There may be a missing \"]\".", line_num+1);}
				BlockType::None => {
					block = false;
					continue;
				}
			}
		}
		
		let ind = key_sets.len()-1;
		
		// store the sub-block text in a flattened string (in the `val` field):
		//    $line: either takes the trimmed or non-trimmed txt
		//	$delim_close: to determine of sub-block ends
		//	$delim: printable text
		macro_rules! store_sub_block_txt{($line: ident, $delim_close: expr, $delim_print: expr) => {
			assert!(key_sets.len() > 0, format!("Config file parsing at line: {}. Error with {} block. Try putting the closing \"{}\" on a new line?", line_num+1, $delim_print, $delim_close));
			assert!(key_sets[ind].len() > 0, format!("Config file parsing at line: {}. Error with {} block. Try putting the closing \"{}\" on a new line?", line_num+1, $delim_print, $delim_close));
			
			let sub_ind = key_sets[ind].len() - 1;
			let entry = &mut key_sets[ind][sub_ind];
			let sz = entry.sz.as_mut().unwrap();
			
			// close block
			if line_trim == $delim_close {
				sz.sz = sz.w*sz.h;
				sub_block = BlockType::None;
			// append text to the `val` field
			}else{
				// update entry sz (ex. number of lines)
				{
					let line_len = $line.chars().count();
					assert!(sz.w == line_len || sz.w == 0, format!("Config file parsing at line: {}. Each line in sub block must be the same width (given: {}, expected: {}). Line with error: \n{}", 
							line_num + 1, line_len, sz.w, $line));
					
					if sz.w == 0 {
						entry.sz = Some(ScreenSz {w: line_len, h: 1, sz: 0});
					}else{
						sz.h += 1;
					}
				}
				entry.val.push_str($line);
			}
			continue;
		};};	
		
		// sub-block action (store text or close block and continue)
		match sub_block {
			BlockType::KeepSpaces => {store_sub_block_txt!(line, ")", "()");}
			BlockType::TrimSpaces => {store_sub_block_txt!(line_trim, "]", "[]");}
			BlockType::None => {}
		}
		
		// add key: value pair
		let pair: Vec<&str> = line_trim.split(":").collect();
		
		assert!(pair.len() == 2, format!("Config file parsing at line: {}. Expected format: \"Key: value\". Do not use more than one colon (\":\") per line. Line with error: \n{}", line_num+1, line_trim));
		
		let key = pair[0].to_string().trim().to_string();
		let val = pair[1].to_string().trim().to_string();
		
		// start multi-line sub-block for value
		if val == "(" || val == "[" {
			sub_block = if val == "(" {BlockType::KeepSpaces} else {BlockType::TrimSpaces};
			key_sets[ind].push(KeyPair {
					key, 
					val: String::new(),
					sz: Some(ScreenSz {w: 0, h: 0, sz: 0})
				});
		
		// normal string value
		}else{
			key_sets[ind].push(KeyPair {key, val, sz: None});
		}
	}
	
	// check for unclosed delims
	match sub_block {
		BlockType::KeepSpaces => {panic!("Missing \")\". Note: it must be on a new-line.");}
		BlockType::TrimSpaces => {panic!("Missing \"]\". Note: it must be on a new-line.");}
		BlockType::None => {}
	}
	
	key_sets
}

// require key to be found. return string
pub fn find_req_key(nm: &str, keys: &Vec<KeyPair>) -> String {
	for k in keys {
		if k.key == nm {
			return k.val.clone();
		}
	}
	panic!("Configuration file entry does not have required entry: \"{}\"", nm);
}

// require key be found. parse into type T
pub fn find_req_key_parse<T: std::str::FromStr>(nm: &str, keys: &Vec<KeyPair>) -> T {
	let string_result = find_req_key(nm, keys);
	if let Result::Ok(val) = string_result.parse() {
		val
	}else{panic!("Cannot parse \"{}\" = \"{}\"", nm, string_result);}
}

// return None if key not found, else return string
pub fn find_key(nm: &str, keys: &Vec<KeyPair>) -> Option<String> {
	for k in keys {
		if k.key == nm {
			return Some(k.val.clone());
		}
	}
	None
}

// if key is found return Vec of T
pub fn find_key_vec<T: std::str::FromStr>(nm: &str, keys: &Vec<KeyPair>) -> Vec<T> {
	let mut vals = Vec::new();
	if let Some(string_result) = find_key(nm, keys) {
		let txt_strs: Vec<&str> = string_result.split(",").collect();
		for txt_str in txt_strs {
			vals.push(if let Result::Ok(val) = txt_str.trim().to_string().parse() {
					val
				}else{panic!("Cannot parse \"{}\" = \"{}\"", nm, string_result);}
			);
		}
	}
	vals
}

