use std::io;
use std::fs::{self, File};
use std::io::prelude::*;
use std::process::exit;
use std::path::Path;
use std::time::SystemTime;

use crate::gcore::XorState;
use crate::player::{PersonName, Nms};
use crate::disp_lib::endwin;
use crate::disp::{ScreenSz, DispChars};
use crate::units::*;
use crate::tech::{TechTemplate};
use crate::doctrine::DoctrineTemplate;
use crate::saving::save_game::SAVE_DIR;
use crate::resources::ResourceTemplate;
use crate::saving::snappy;

mod date;
use date::*;

#[derive(Debug, Clone)]
pub struct KeyPair {
	pub key: String,
	pub val: String,
	sz: Option<ScreenSz>
}

macro_rules! q{($txt:expr) => {
	endwin();
	println!("{}", $txt);
	exit(0);
};}

macro_rules! qa{($cond:expr, $txt:expr) => {
	if !$cond {q!($txt);}
};}

// return buffer of file "nm"
pub fn read_file(nm: &str) -> Vec<u8> {
	let mut buf = Vec::new();
	
	let nm = Path::new(nm).as_os_str().to_str().unwrap();
	
	if let Result::Ok(ref mut file) = File::open(nm) {
		if let Result::Err(_) = file.read_to_end(&mut buf) {
			q!(format!("Failed reading configuration file: {}", nm));
		}
	}else {
		panicq!("Failed opening file for reading: {}", nm);
	}

	buf
}

// return buffer of file "nm" (decompress first)
pub fn read_file_decompress(nm: &str) -> Vec<u8> {
	let mut buf = Vec::new();
	
	let nm = Path::new(nm).as_os_str().to_str().unwrap();
	
	if let Result::Ok(ref mut file) = File::open(nm) {
		let mut rdr = snappy::read::FrameDecoder::new(file);
		io::copy(&mut rdr, &mut buf).expect("Failed to read compressed file");
	}else {
		panicq!("Failed opening file for reading: {}", nm);
	}

	buf
}

#[derive(PartialEq)]
pub struct SaveFile {
	pub nm: String,
	pub modified: String,
	elapsed: u64
}

// for open game window
pub fn return_save_files() -> Vec<SaveFile> {
	let sv_gms_dir = Path::new(SAVE_DIR);
	
	let mut save_files = Vec::new();
	
	// loop over files in the directory, each file contains a seprate them of city names
	if let Result::Ok(dir_entries) = fs::read_dir(sv_gms_dir) {
		for entry in dir_entries {
			if let Result::Ok(e) = entry {
			if let Result::Ok(m) = e.metadata() {
				if !m.is_file() {continue;}
					
				if let Some(fname) = e.path().file_name() {
				if let Some(fname_str) = fname.to_str() {
				if let Result::Ok(modified) = m.modified() {
				if let Result::Ok(dur) = modified.duration_since(SystemTime::UNIX_EPOCH) {
					save_files.push(SaveFile {
							nm: fname_str.to_string(),
							modified: HttpDate::from(modified).string(),
							elapsed: dur.as_secs()
					});
				}}}}}}}
	} else {q!(format!("failed to open {}", SAVE_DIR));}
	
	save_files.sort_unstable_by(|a, b| b.elapsed.partial_cmp(&a.elapsed).unwrap());
	
	save_files
}

pub fn return_names_list(buf: Vec<u8>) -> Vec<String> {
	let mut names = Vec::new();
	
	for line in String::from_utf8(buf).unwrap().lines() {
		if line.starts_with("//") {continue;}
		
		// capitilize first letter of each word
		let mut start_space = true;
		let mut capitalized = String::new();
		for c in line.trim().to_string().as_bytes() {
			if start_space {
				capitalized.push((*c as char).to_uppercase().next().unwrap());
				start_space = false;
			}else{
				capitalized.push(*c as char);
			}
			
			if (*c as char) == ' ' {
				start_space = true;
			}
		}
		names.push(capitalized)
	}
	names
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
				BlockType::KeepSpaces => {q!(format!("Config file parsing at line: {}. There may be a missing \")\".", line_num+1));}
				BlockType::TrimSpaces => {q!(format!("Config file parsing at line: {}. There may be a missing \"]\".", line_num+1));}
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
			qa!(key_sets.len() > 0, format!("Config file parsing at line: {}. Error with {} block. Try putting the closing \"{}\" on a new line?", line_num+1, $delim_print, $delim_close));
			qa!(key_sets[ind].len() > 0, format!("Config file parsing at line: {}. Error with {} block. Try putting the closing \"{}\" on a new line?", line_num+1, $delim_print, $delim_close));
			
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
					qa!(sz.w == line_len || sz.w == 0, format!("Config file parsing at line: {}. Each line in sub block must be the same width (given: {}, expected: {}). Line with error: \n{}", 
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
		const DELIM_STR: &str = ":";
		let pair: Vec<&str> = line_trim.split(DELIM_STR).collect();
		
		qa!(pair.len() >= 2, format!("Config file parsing at line: {}. Expected format: \"Key: value\". Line with error: \n{}", line_num+1, line_trim));
		
		let key = pair[0].trim().to_string();
		let val = {
			// re-combine if there were more than one ":"
			let mut recombined = String::new();
			for (part_ind, part) in pair.iter().enumerate().skip(1) {
				recombined.push_str(part);
				if part_ind != (pair.len()-1) {
					recombined.push_str(DELIM_STR);
				}
			}
			
			// remove comments at end of the line
			let comment_split: Vec<&str> = recombined.split("//").collect();
			comment_split[0].trim().to_string()
		};
		
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
		BlockType::KeepSpaces => {q!(format!("Missing \")\". Note: it must be on a new-line."));}
		BlockType::TrimSpaces => {q!(format!("Missing \"]\". Note: it must be on a new-line."));}
		BlockType::None => {}
	}
	
	key_sets
}

// require each Vec<KeyPair>.nm be unique
pub fn chk_key_unique(nm: &str, key_sets: &Vec<Vec<KeyPair>>) {
	for (ind_i, i) in key_sets.iter().enumerate() {
		let val = find_req_key(nm, i);
		
		for (ind_j, j) in key_sets.iter().enumerate() {
			if ind_i == ind_j {continue;}
			
			if val == find_req_key(nm, j) {
				q!(format!("Configuration error. All entries for \"{}\" must be unique. Duplicates found for \"{}\"", nm, val));
			}
		}
	}
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

// require key to be found. return string
pub fn find_req_key(nm: &str, keys: &Vec<KeyPair>) -> String {
	for k in keys {
		if k.key == nm {
			return k.val.clone();
		}
	}
	panicq!("Configuration file entry does not have required entry: \"{}\"", nm);
}

// require key be found. parse into type T
pub fn find_req_key_parse<T: std::str::FromStr>(nm: &str, keys: &Vec<KeyPair>) -> T {
	let string_result = find_req_key(nm, keys);
	if let Result::Ok(val) = string_result.parse() {
		val
	}else{q!(format!("Cannot parse \"{}\" = \"{}\"", nm, string_result));}
}

// use default value if key not found
pub fn find_key_parse<T: std::str::FromStr>(nm: &str, def: T, keys: &Vec<KeyPair>) -> T {
	if let Some(string_result) = find_key(nm, keys) {
		if let Result::Ok(val) = string_result.parse() {
			val
		}else{panicq!("Cannot parse \"{}\" = \"{}\"", nm, string_result);}
	}else{
		def
	}
}

// if key is not found, return none
pub fn find_opt_key_parse<T: std::str::FromStr>(nm: &str, keys: &Vec<KeyPair>) -> Option<T> {
	if let Some(string_result) = find_key(nm, keys) {
		if let Result::Ok(val) = string_result.parse() {
			Some(val)
		}else{q!(format!("Cannot parse \"{}\" = \"{}\"", nm, string_result));}
	}else{
		None
	}
}

// if key is found return Vec of UnitTemplates, else return None
// used for the units for buildings which produce units
pub fn find_opt_key_units_producable<'ut,'rt>(nm: &str, keys: &Vec<KeyPair>, 
		unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> Option<Vec<&'ut UnitTemplate<'rt>>>{
	
	if let Some(string_result) = find_key(nm, keys) {
		let units_producable_strs: Vec<&str> = string_result.split(",").collect();
		
		let mut units_producable = Vec::new();
		
		for unit_producable_str in units_producable_strs {
			units_producable.push(UnitTemplate::frm_str(unit_producable_str.trim(), unit_templates));
		}
		
		Some(units_producable)
	}else{
		None
	}
}

// if key is found return Vec of ResourceTemplates, else return None
// used for the units for buildings which produce units
pub fn find_opt_key_resources<'ut,'rt>(nm: &str, keys: &Vec<KeyPair>, 
		resource_templates: &'rt Vec<ResourceTemplate>) -> Option<Vec<&'rt ResourceTemplate>>{
	
	if let Some(string_result) = find_key(nm, keys) {
		let resource_strs: Vec<&str> = string_result.split(",").collect();
		
		let mut resources = Vec::new();
		
		for resource_str in resource_strs {
			resources.push(ResourceTemplate::frm_str(resource_str.trim(), resource_templates));
		}
		
		Some(resources)
	}else{
		None
	}
}

// if key is found return Vec of &strs, else return None
// used for the menu entries for buildings which produce units
pub fn find_opt_key_vec_string(nm: &str, keys: &Vec<KeyPair>) -> Option<Vec<String>> {
	if let Some(string_result) = find_key(nm, keys) {
		let txt_strs: Vec<&str> = string_result.split(",").collect();
		
		let mut txt = Vec::with_capacity(txt_strs.len());
		
		for txt_str in txt_strs {
			txt.push(txt_str.trim().to_string());
		}
		
		Some(txt)
	}else{
		None
	}
}

pub fn parse_vec_quoted_string(string: String) -> Vec<String> {
	let txt_strs: Vec<&str> = string.split("\",").collect();
	
	let mut txt = Vec::with_capacity(txt_strs.len());
	
	for txt_str in txt_strs {
		txt.push(txt_str.trim().to_string().replace("\"", ""));
	}
	
	txt
}

pub fn find_req_key_vec_quoted_string(nm: &str, keys: &Vec<KeyPair>) -> Vec<String> {
	if let Some(string_result) = find_key(nm, keys) {
		parse_vec_quoted_string(string_result)
	}else{
		panicq!("could not find required entry: {}", nm);
	}
}

// convert unicode lines to display characters
// or create box from dimensions
pub fn find_req_key_print_str(nm: &str, keys: &Vec<KeyPair>, disp_chars: &DispChars) -> String {
	let val = find_req_key(nm, keys);
		
	// create box from dimensions.
	// (ScreenSz in the key pair will be set if we've previously parsed
	// something of the format "[ some text ]")
	
	let mut screen_sz_not_set = false;
	for kp in keys.iter() {
		if kp.key== nm {
			screen_sz_not_set = kp.sz.is_none();
			break;
		}
	};
	
	if screen_sz_not_set {
		let sz = parse_box_width_height(&val);
		
		let mut print_str = String::new();
		for _ in 0..sz.sz {print_str.push(' ');}
		
		let mut print_str = print_str.into_bytes();
		
		// l/r sides
		for i in 1..(sz.h-1) {
			print_str[i*sz.w] = disp_chars.vline_char as u8;
			print_str[i*sz.w + sz.w-1] = disp_chars.vline_char as u8;
		}
		
		// top/bottom
		for j in 1..(sz.w-1) {
			print_str[j] = disp_chars.hline_char as u8;
			print_str[(sz.h-1)*sz.w + j] = disp_chars.hline_char as u8;
		}
		
		// corners
		print_str[0] = disp_chars.ulcorner_char as u8;
		print_str[sz.w-1] = disp_chars.urcorner_char as u8;
		print_str[(sz.h-1)*sz.w] = disp_chars.llcorner_char as u8;
		print_str[sz.w*sz.h - 1] = disp_chars.lrcorner_char as u8;
		
		String::from_utf8(print_str).unwrap()

	// create print_str from what's in the configuration file:
	}else{
		let val_bytes = val.into_bytes();
		let mut print_str = Vec::new();
		
		let mut skip = None; // skip 3 chars for the unicode box chars
		
		for i in 0..val_bytes.len() {
			
			// start of unicode box chars
			if skip == None && val_bytes[i] == 0xE2 {
				skip = Some(0);
				continue;
			}
			
			// unicode box chars
			if let Some(ref mut skip_num) = &mut skip {
				*skip_num += 1;
				qa!(*skip_num != 1 || val_bytes[i] == 0x94, format!("Unknown unicode symbol when parsing key: \"{}\"", nm));
				
				// last char of the unicode box char
				// the last char is the informative one, parse into DispChars:
				if *skip_num == 2 {
					skip = None;
					print_str.push(match val_bytes[i] {
						0x98 => disp_chars.lrcorner_char,
						0x94 => disp_chars.llcorner_char,
						0x82 => disp_chars.vline_char,
						0x90 => disp_chars.urcorner_char,
						0x80 => disp_chars.hline_char,
						0x8C => disp_chars.ulcorner_char,
						_ => {q!(format!("Unknown unicode symbol when parsing key: \"{}\"", nm));}
					} as u8);
				}
				continue;
			}
			
			// regular ascii character
			print_str.push(val_bytes[i]);
		}
		
		String::from_utf8(print_str).unwrap()
	}
}

pub fn parse_box_width_height(txt: &String) -> ScreenSz {
	const BOX_STR: &str = "Box(";
	qa!(txt.starts_with(BOX_STR), format!("Expected format \"Box(width, height)\". Found: \"{}\"", txt));
	qa!(txt.ends_with(")"), format!("Expected format \"Box(width, height)\". Found: \"{}\"", txt));
	
	let txt = txt.trim_start_matches(BOX_STR).to_string();
	let txt = txt.trim_end_matches(")").to_string();
	
	let pair: Vec<&str> = txt.split(",").collect();
	
	qa!(pair.len() == 2, format!("Expected format \"Box(width, height)\". Found: \"{}\"", txt));
	
	let w: usize = if let Result::Ok(w) = pair[0].to_string().trim().parse() {
			w
		}else{
			q!(format!("Cannot parse into integer. Failed with: \"{}\"", txt));
		};
	
	let h: usize = if let Result::Ok(h) = pair[0].to_string().trim().parse() {
			h
		}else{
			q!(format!("Cannot parse into integer. Failed with: \"{}\"", txt));
		};

	ScreenSz {w, h, sz: w*h}
}

// sz (height, width) for characters to display on screen
pub fn find_req_key_print_sz(nm: &str, keys: &Vec<KeyPair>) -> ScreenSz {
	for k in keys {
		if k.key == nm {
			return if let Some(sz) = k.sz {
				sz
			}else{
				parse_box_width_height(&k.val)
			}
		}
	}
	q!(format!("Configuration file entry does not have required entry: \"{}\"", nm));
}

pub fn find_tech_req(nm: &String, keys: &Vec<KeyPair>, tech_templates: &Vec<TechTemplate>) -> Option<Vec<usize>> {
	if let Some(tech_strs) = find_key("tech_req", &keys) {
		let tech_strs: Vec<&str> = tech_strs.split(",").collect();
		let mut tech_reqs = Vec::with_capacity(tech_strs.len());
		
		for (tech_str_i, tech_str) in tech_strs.iter().enumerate() {
			for (i, t) in tech_templates.iter().enumerate() {
				if t.nm[0] == *tech_str.trim() {
					tech_reqs.push(i);
					break;
				}
			}
			
			if (tech_reqs.len()-1) != tech_str_i {
				q!(format!("Could not find tech requirement \"{}\" when adding unit \"{}\". Check technology configuration file.", tech_str, nm));
			}
		}
		Some(tech_reqs)
	}else {None}
}

pub fn find_doctrine_req<'dt>(nm: &String, keys: &Vec<KeyPair>, doctrine_templates: &'dt Vec<DoctrineTemplate>) -> Option<&'dt DoctrineTemplate> {
	if let Some(doc_str) = find_key("doctrine_req", &keys) {
		let doc_str = doc_str.trim();
		for dt in doctrine_templates.iter() {
			if dt.nm[0] == doc_str {
				return Some(dt);
			}
		}
		
		panicq!("Could not find doctrine requirement \"{}\" when adding unit \"{}\". Check doctrine or building configuration file.", doc_str, nm);
	}else {None}
}

pub fn find_resources_req<'rt>(nm: &String, keys: &Vec<KeyPair>, resource_templates: &'rt Vec<ResourceTemplate>) -> Vec<&'rt ResourceTemplate> {
	if let Some(resource_strs) = find_key("resource_req", &keys) {
		let resource_strs: Vec<&str> = resource_strs.split(",").collect();
		let mut resource_reqs = Vec::with_capacity(resource_strs.len());
		
		for (resource_str_i, resource_str) in resource_strs.iter().enumerate() {
			for r in resource_templates.iter() {
				if r.nm[0] == *resource_str.trim() {
					resource_reqs.push(r);
					break;
				}
			}
			
			if (resource_reqs.len()-1) != resource_str_i {
				q!(format!("Could not find resource requirement \"{}\" when adding unit \"{}\". Check resource configuration file.", resource_str, nm));
			}
		}
		resource_reqs
	}else {Vec::new()}
}

use crate::map::ZoneType;

// used to load zone-specific bonuses for buildings and resources
pub fn load_zone_bonuses(keys: &Vec<KeyPair>) -> Vec<Option<isize>> {
	let mut bonuses = vec!{None; ZoneType::N as usize};
	
	bonuses[ZoneType::Agricultural as usize] = find_opt_key_parse("agricultural_bonus", keys);
	bonuses[ZoneType::Residential as usize] = find_opt_key_parse("residential_bonus", keys);
	bonuses[ZoneType::Business as usize] = find_opt_key_parse("business_bonus", keys);
	bonuses[ZoneType::Industrial as usize] = find_opt_key_parse("industrial_bonus", keys);
	
	bonuses
}

pub fn get_usize_map_config(buffer_key: &str) -> usize {
	const MAP_CONFIG: &str = "config/map.txt";
	
	for keys in config_parse(read_file(MAP_CONFIG)).iter() {
		if let Some(max_zoom_in_buffer_sz) = find_opt_key_parse(buffer_key, &keys) {
			return max_zoom_in_buffer_sz;
		}
	}
	panicq!("could not find `{}` in {}", buffer_key, MAP_CONFIG);
}

use crate::disp_lib::KEY_ESC;
pub const UNSET_KEY: i32 = -2;
pub fn find_kbd_key(nm: &str, key_sets: &Vec<Vec<KeyPair>>) -> i32 {
	for keys in key_sets.iter() {
		for k in keys.iter().filter(|k| k.key == nm) {
			return if k.val.len() == 1 {
				if let Some(kbd_char) = k.val.chars().nth(0) {
					kbd_char as i32
				}else {panicq!("could not load keyboard setting {} for `{}`", k.val, nm);}
			}else if k.val == "<space>" {
				' ' as i32
			}else if k.val == "<tab>" {
				'\t' as i32
			}else if k.val == "<enter>" {
				'\n' as i32
			}else if k.val == "<esc>" {
				KEY_ESC
			}else {panicq!("could not interpret keyboard configuration '{}' for setting `{}`", k.val, nm);};
		}
	}
	UNSET_KEY // should hopefully never occur... somewhat of a hack
}

impl PersonName {
	// first output is `gender_female`, second is the name
	pub fn new(nms: &Nms, rng: &mut XorState) -> (bool, Self) {
		///// select gender
		let (gender_female, nms) = match rng.usize_range(0,2) {
			0 => {(true, &nms.females)}
			1 => {(false, &nms.males)}
			_ => {panicq!("invalid random number");}
		};
		
		// select name
		let nm = &nms[rng.usize_range(0, nms.len())];
		
		// split into first & last name
		let nm_pair: Vec<&str> = nm.split(" ").collect();
		
		(gender_female, PersonName {
			first: nm_pair[0].to_string().clone(),
		 	last: nm_pair[1].to_string().clone()
		 })
	}
	
	pub fn new_w_gender(gender_female: bool, nms: &Nms, rng: &mut XorState) -> Self {
		let nms = if gender_female {&nms.females} else {&nms.males};
		
		// select name
		let nm = &nms[rng.usize_range(0, nms.len())];
		
		// split into first & last name
		let nm_pair: Vec<&str> = nm.split(" ").collect();
		
		Self {
			first: nm_pair[0].to_string().clone(),
		 	last: nm_pair[1].to_string().clone()
		 }
	}
}

