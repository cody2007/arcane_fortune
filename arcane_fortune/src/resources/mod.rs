use crate::saving::*;
use crate::config_load::{config_parse, read_file, chk_key_unique,
	find_req_key, load_zone_bonuses, find_opt_key_parse, find_tech_req,
	find_req_key_print_sz, find_req_key_parse, find_req_key_print_str};
use crate::tech::TechTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::map::{ArabilityType, MapData, ZoneType, TechProg};
use crate::player::Stats;
use crate::disp::{ScreenSz, DispChars};
use crate::localization::Localization;
//use crate::disp_lib::endwin;

pub const N_RESOURCES_DISCOV_LOG: usize = 20; // # of locations to log for each resource type

#[derive(PartialEq, Clone)]
pub struct ResourceTemplate {
	pub id: SmSvType,
	pub nm: Vec<String>,
	
	pub tech_req: Vec<usize>,
	
	pub sz: ScreenSz,
	pub print_str: String,
	pub plot_zoomed: char, // what to show when zoomed out
	
	pub zone: ZoneType,
	pub zone_bonuses: Vec<Option<isize>>, // zone bonuses given to neighboring regions indexed by ZoneType
	pub arability_probs: Vec<Option<f32>> // probability of resource occuring at each ArabilityType (which it is indexed by)
}

impl_saving_template!{ResourceTemplate {id, nm, tech_req, sz, print_str,
	plot_zoomed, zone, zone_bonuses, arability_probs}}

impl ResourceTemplate {
	pub fn frm_str<'rt>(txt: &str, resource_templates: &'rt Vec<ResourceTemplate>) -> &'rt ResourceTemplate {
		for rt in resource_templates.iter() {
			if txt == rt.nm[0] {
				return rt;
			}
		}
		panicq!("Could not find resource \"{}\"", txt);
	}
}

pub fn init_resource_templates(tech_templates: &Vec<TechTemplate>,
		disp_chars: &DispChars,
		l: &Localization) -> Vec<ResourceTemplate> {
	let key_sets = config_parse(read_file("config/resources.txt"));
	chk_key_unique("nm", &key_sets);
	
	let mut resource_templates = Vec::new();
	
	for (id, keys) in key_sets.iter().enumerate() {
		let eng_nm = find_req_key("nm", keys);
		let nm = if let Some(nm) = l.resource_nms.iter().find(|nms| nms[0] == eng_nm) {
			nm.clone()
		}else{panicq!("could not find translations of resource `{}`. the localization file may need to be updated", eng_nm);};
		
		// get indices for tech_reqs
		let tech_req = if let Some(tech_reqs) = find_tech_req(&eng_nm, keys, tech_templates)
					{tech_reqs} else {Vec::new()};
		
		// load probability of reouce occuring in each arability type
		let n_arability_types = ArabilityType::N as usize;
		let mut arability_probs = Vec::with_capacity(n_arability_types);
		for arability_ind in 0..n_arability_types {
			let arability_type = ArabilityType::from(arability_ind);
			arability_probs.push(find_opt_key_parse(&arability_type.to_config_str(), &keys));
		}
		
		// create template
		resource_templates.push( ResourceTemplate {
			id: id as SmSvType,
			nm,
			tech_req,
			
			sz: find_req_key_print_sz("print_str", keys),
			print_str: find_req_key_print_str("print_str", keys, disp_chars),
			plot_zoomed: find_req_key_parse("plot_zoomed", keys),
			
			zone: find_req_key_parse("zone", keys),
			zone_bonuses: load_zone_bonuses(keys),
			arability_probs
		} );
	}
	
	// check ordering is correct
	#[cfg(any(feature="opt_debug", debug_assertions))]
	for (i, r) in resource_templates.iter().enumerate() {
		debug_assertq!(r.id == i as SmSvType);
	}
	
	resource_templates
}

impl Stats<'_,'_,'_,'_> {
	// in the technological sense
	pub fn resource_discov(&self, resource: &ResourceTemplate) -> bool {
		resource.tech_req.iter().all(|&tech_req|
				self.techs_progress[tech_req] == TechProg::Finished)
	}
	
	pub fn resources_met(&self, resources_chk: &Vec<&ResourceTemplate>) -> bool {
		resources_chk.iter().
			all(|resource_chk| (self.resources_avail[resource_chk.id as usize] > 0 &&
						self.resource_discov(resource_chk)))
	}
	
	pub fn unit_producable(&self, ut: &UnitTemplate) -> bool {
		self.resources_met(&ut.resources_req) && self.tech_met(&ut.tech_req)
	}
}

use crate::disp::{Coord};
use crate::disp_lib::{endwin, chtype};
use crate::map::{ZoomInd, MapSz, Map};

impl <'rt>Map<'rt> {
	pub fn get_resource(&self, coord: u64, map_data: &mut MapData<'rt>, map_sz: MapSz) -> Option<&'rt ResourceTemplate> {
		// upper left
		if let Some(_) = self.resource {
			return self.resource;
		}
		
		// not at upper left of resource
		if let Some(resource_cont) = self.resource_cont {
			// offsets into the resource
			let i = resource_cont.offset_i;
			let j = resource_cont.offset_j;
			
			// use offsets to get coordinate of resource
			let c = Coord::frm_ind(coord, map_sz);
			if let Some(coord_resource) = map_sz.coord_wrap(c.y - i as isize, c.x - j as isize) {
				// get resource at the coordinates specified by the offsets
				let mfc_resource = map_data.get(ZoomInd::Full, coord_resource).resource.unwrap();
				return Some(mfc_resource);
			}else{panicq!("resource offset invalid");}
		}
		None
	}
	
	// resources can span multiple tiles, so this returns the upper left coordinate of the resource
	pub fn get_resource_and_coord(&self, coord: u64, map_data: &mut MapData<'rt>, map_sz: MapSz) -> Option<(&'rt ResourceTemplate, u64)> {
		// upper left
		if let Some(resource) = self.resource {
			return Some((resource, coord));
		}
		
		// not at upper left of resource
		if let Some(resource_cont) = self.resource_cont {
			// offsets into the resource
			let i = resource_cont.offset_i;
			let j = resource_cont.offset_j;
			
			// use offsets to get coordinate of resource
			let c = Coord::frm_ind(coord, map_sz);
			if let Some(coord_resource) = map_sz.coord_wrap(c.y - i as isize, c.x - j as isize) {
				// get resource at the coordinates specified by the offsets
				let mfc_resource = map_data.get(ZoomInd::Full, coord_resource).resource.unwrap();
				return Some((mfc_resource, coord_resource));
			}else{panicq!("resource offset invalid");}
		}
		None
	}

	
	pub fn resource_char(&self, coord: u64, map_data: &mut MapData, map_sz: MapSz, 
				disp_chars: &DispChars) -> Option<chtype> {
		// upper left
		if let Some(r) = self.resource {
			return Some(disp_chars.convert_to_line(
					r.print_str.chars().nth(0).unwrap()));
		}
		
		// not at upper left of resource
		if let Some(resource_cont) = self.resource_cont {
			// offsets into the resource
			let i = resource_cont.offset_i;
			let j = resource_cont.offset_j;
			
			// use offsets to get coordinate of resource
			let c = Coord::frm_ind(coord, map_sz);
			if let Some(coord_resource) = map_sz.coord_wrap(c.y - i as isize, c.x - j as isize) {
				// get resource at the coordinates specified by the offsets
				let mfc_resource = map_data.get(ZoomInd::Full, coord_resource).resource.unwrap();
				
				// get the index into the string based on the print dimensions
				let str_ind = (i as usize)*mfc_resource.sz.w + j as usize;
				
				return Some(disp_chars.convert_to_line(
							mfc_resource.print_str.chars().nth(str_ind).unwrap()));
			}else{panicq!("resource offset invalid");}
		}
		None
	}
}

