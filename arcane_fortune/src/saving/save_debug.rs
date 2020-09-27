/*
//use std::hash::{Hasher, BuildHasherDefault};
//use std::collections::HashMap;
use crate::gcore::hashing::HashedMapEx;

use crate::map::*;
use crate::units::*;
//use crate::buildings::*;
//use crate::disp::*;
//use crate::units::*;
//use crate::gcore::hashing::HashStruct64;
use crate::disp_lib::endwin;
use std::fs::File;
use std::io::prelude::*;
use std::process::exit;
use std::path::Path;

pub fn save_exf_debug(exs: &HashedMapEx) {
	const DBG_SAVE_FILE: &str = "saves/exf_dbg.txt";

	let mut buf = Vec::new();

	let mut n_demand_raws_set = 0;
	
	/*for (coord, ex) in exs.iter() {
		if let Some(zone) = &ex.zone {
			for zdr in zone.demand_raw.iter() {
				if let Some(demand_raw) = &zdr {
					buf.append(&mut format!("coord: {}, zone.demand_raw[i].turn_computed = {}\n", coord, 
								demand_raw.turn_computed).into_bytes());
					n_demand_raws_set += 1;
				}else{
					buf.append(&mut format!("coord: {}, zone.demand_raw = None\n", coord).into_bytes());
				} // ZoneDemandRaw
			} // Box<[Option<ZoneDemandRaw>]>
		} // zone
	} // exs*/
	
	'ex_loop: for (coord, ex) in exs.iter() {
		if let Some(zone) = &ex.zone {
			if ex.actual.owner_id.unwrap() != 0 {continue;}
			for zdr in zone.demand_raw.iter() {
				if let Some(demand_raw) = &zdr {
					buf.append(&mut format!("coord: {}, owner: {}, zone.demand_raw[i].turn_computed = {}\n", coord, ex.actual.owner_id.unwrap(),
								demand_raw.turn_computed).into_bytes());
					n_demand_raws_set += 1;
					continue 'ex_loop;
				} // ZoneDemandRaw
			} // Box<[Option<ZoneDemandRaw>]>
			//buf.append(&mut format!("coord: {}, zone.demand_raw = None\n", coord).into_bytes());
		} // zone
	} // exs
	buf.append(&mut format!("n_demand_raws_set: {}\n", n_demand_raws_set).into_bytes());
	
	if let Result::Ok(ref mut file) = File::create(Path::new(DBG_SAVE_FILE).as_os_str()) {
		if let Result::Err(_) = file.write_all(&buf) {
			q!(format!("failed writing file"));
		}
	}else{
		q!(format!("failed opening file for writing"));
	}
}

pub fn save_unit_exs_debug(exs: &Vec<HashedMapEx>, map_data: &MapData, units: &Vec<Unit>) {
	const DBG_SAVE_FILE: &str = "saves/unit_exs_dbg.txt";

	let mut buf = Vec::new();
	
	for zoom_ind in 0..=map_data.max_zoom_ind() {
		for (coord, ex) in exs[zoom_ind].iter() {
			if let Some(unit_inds) = &ex.unit_inds {
				//if let Some(owner) = ex.actual.owner_id {
					buf.append(&mut format!("zoom {} coord {} owner {} unit_ind {}\n", zoom_ind, coord, 
								units[unit_inds[0]].owner_id, unit_inds.len()).into_bytes());
				//}
			} // unit inds
		} // exs
	} // zoom_ind
	
	if let Result::Ok(ref mut file) = File::create(Path::new(DBG_SAVE_FILE).as_os_str()) {
		if let Result::Err(_) = file.write_all(&buf) {
			q!(format!("failed writing file"));
		}
	}else{
		q!(format!("failed opening file for writing"));
	}
}
*/

//use std::fs::File;
/*use std::io::prelude::*;
use std::process::exit;
use std::path::Path;
use std::fs::OpenOptions;
use crate::disp_lib::endwin;

pub fn save_txt(msg: String) {
	const DBG_SAVE_FILE: &str = "/tmp/tmp_output.txt";
	
	if let Result::Ok(ref mut file) = OpenOptions::new().append(true).open(Path::new(DBG_SAVE_FILE).as_os_str()) {
		if let Result::Err(_) = file.write_all(&msg.into_bytes()) {
			q!(format!("failed writing file"));
		}
	}else{
		q!(format!("failed opening file for writing"));
	}
}
*/
