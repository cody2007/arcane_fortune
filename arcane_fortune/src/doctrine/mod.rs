use crate::saving::*;
use crate::config_load::*;
use crate::player::Stats;
use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::localization::Localization;
use crate::renderer::{endwin};//, CInt};

pub mod disp; pub use disp::*;

#[derive(PartialEq, Clone)]
pub struct DoctrineTemplate {
	pub id: usize,
	pub nm: Vec<String>,
	
	pub pre_req_ind: Option<usize>, // index into doctrine_templates
	pub bldg_req: f32, // doctrinality points required
	
	pub health_bonus: f32,
	pub crime_bonus: f32,
	pub pacifism_bonus: f32,
	pub happiness_bonus: f32,
	pub tax_aversion: f32
}

impl_saving_template!{DoctrineTemplate {id, nm, pre_req_ind,
	bldg_req, health_bonus, crime_bonus, pacifism_bonus,
	happiness_bonus, tax_aversion
} }

pub fn init_doctrine_templates(l: &Localization) -> Vec<DoctrineTemplate> {
	const DOCTRINE_CONFIG: &str = "config/doctrine.txt";
	let key_sets = config_parse(read_file(DOCTRINE_CONFIG));
	chk_key_unique("nm", &key_sets);
	
	let mut doctrine_templates = Vec::new();
	
	let mut nms = Vec::with_capacity(key_sets.len());
	for keys in key_sets.iter() {nms.push(find_req_key("nm", keys))}
	
	for (id, keys) in key_sets.iter().enumerate() {
		let find_ind = || {
			if let Some(pre_req_nm) = find_key("pre_req_nm", keys) {
				for (doctrinality_ind, nm) in nms.iter().enumerate() {
					if *nm == pre_req_nm {
						return Some(doctrinality_ind);
					}
				}
				panicq!("could not find doctrinality pre-requisate {} in {}", pre_req_nm, DOCTRINE_CONFIG);
			}
			None
		};
		
		let eng_nm = find_req_key("nm", keys);
		let nm = if let Some(nm) = l.doctrine_nms.iter().find(|nms| nms[0] == eng_nm) {
			nm.clone()
		}else{panicq!("could not find translations of doctrine `{}`. the localization file may need to be updated.", eng_nm);};
		
		doctrine_templates.push(DoctrineTemplate {
			id, nm,
			
			pre_req_ind: find_ind(),
			bldg_req: find_key_parse("bldg_req", 0., keys),
			
			health_bonus: find_key_parse("health_bonus", 0., keys),
			crime_bonus: find_key_parse("crime_bonus", 0., keys),
			pacifism_bonus: find_key_parse("pacifism_bonus", 0., keys),
			happiness_bonus: find_key_parse("happiness_bonus", 0., keys),
			tax_aversion: find_key_parse("tax_aversion", 0., keys)
		});
	}
	
	doctrine_templates
}

impl DoctrineTemplate {
	pub fn is_available(&self, pstats: &Stats, doctrine_templates: &Vec<DoctrineTemplate>) -> bool {
		if self.id == 0 {return false;} // undefined doctrinality
		
		if let Some(pre_req_ind) = self.pre_req_ind {
			pre_req_ind == 0 || doctrine_templates[pre_req_ind].bldg_req <= pstats.locally_logged.doctrinality_sum[pre_req_ind]
			// ^ either the pre-req is the undefined doctrinality or the building requirements for the pre-req have been met
		}else{true}
	}
	
	pub fn bldg_reqs_met(&self, pstats: &Stats) -> bool {
		self.bldg_req <= pstats.locally_logged.doctrinality_sum[self.id]
	}
	
	pub fn closest_available<'dt>(&'dt self, pstats: &Stats, doctrine_templates: &'dt Vec<DoctrineTemplate>
			) -> &'dt DoctrineTemplate {
		if self.is_available(pstats, doctrine_templates) || self.id == 0 { // id = 0 is the undefined doctrine, it is the closesest available to itself
			return self;
		}
		
		if let Some(pre_req_ind) = self.pre_req_ind {
			return doctrine_templates[pre_req_ind].closest_available(pstats, doctrine_templates);
		}
		panicq!("pre-req not found, id: {}", self.id);
	}
	
	pub fn bldgs_unlocks<'bt,'ut,'rt,'dt>(&self, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>) -> Vec<&'bt BldgTemplate<'ut,'rt,'dt>> {
		let mut unlocked = Vec::with_capacity(bldg_templates.len());
		for bt in bldg_templates.iter() {
			if bt.doctrine_req == Some(&self) {
				unlocked.push(bt);
			}
		}
		unlocked
	}
}

pub fn available_doctrines<'dt>(pstats: &Stats, doctrine_templates: &'dt Vec<DoctrineTemplate>) -> Vec<&'dt DoctrineTemplate> {
	let mut avail = Vec::with_capacity(doctrine_templates.len());
	for candidate in doctrine_templates.iter()
		.filter(|candidate| candidate.is_available(pstats, doctrine_templates)) {
			avail.push(candidate);
	}
	avail
}

