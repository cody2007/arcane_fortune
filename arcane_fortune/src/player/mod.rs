use std::collections::HashMap;
use core::hash::BuildHasherDefault;

use crate::disp_lib::endwin;
use crate::disp::Coord;
use crate::saving::*;
use crate::gcore::{HashedMapZoneEx, Bonuses, Relations};
use crate::tech::TechTemplate;
use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::units::UnitTemplate;
use crate::map::{MapData, MapSz};
use crate::ai::{AIState, BarbarianState};
use crate::gcore::hashing::{HashedMap, HashStruct64, HashedFogVars};

pub mod player_type; pub use player_type::*;
pub mod personalization; pub use personalization::*;
pub mod stats; pub use stats::*;

// container
#[derive(Clone, PartialEq)]
pub struct Player<'bt,'ut,'rt,'dt> {
	pub id: SmSvType,
	pub ptype: PlayerType<'bt,'ut,'rt,'dt>,
	pub personalization: Personalization,
	pub stats: Stats<'bt,'ut,'rt,'dt>,
	pub zone_exs: HashedMapZoneEx
}

// cannot use macro because stats has a reference and can't have a Default
//impl_saving!{Player<'bt,'ut,'rt,'dt> {id, ptype, personalization, stats, zone_exs}}

// unit management functions
impl <'bt,'ut,'rt,'dt>Player<'bt,'ut,'rt,'dt> {
	pub fn add_unit(&mut self, unit_ind: usize, unit_coord: u64, unit_template: &UnitTemplate,
			unit_templates: &Vec<UnitTemplate>, map_sz: MapSz) {
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} => {
				ai_state.add_unit(unit_ind, unit_coord, unit_template, &self.stats, unit_templates, map_sz);
			} PlayerType::Barbarian {barbarian_state} => {
				barbarian_state.add_unit(unit_ind, unit_template);
			} PlayerType::Nobility {house} => {
				house.add_unit(unit_ind, unit_coord, unit_template, &self.stats, unit_templates, map_sz);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn rm_unit(&mut self, unit_ind: usize, unit_template: &UnitTemplate) {
		self.stats.rm_unit_frm_brigade(unit_ind);
		
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} => {
				ai_state.rm_unit(unit_ind, unit_template);
			} PlayerType::Barbarian {barbarian_state} => {
				barbarian_state.rm_unit(unit_ind);
			} PlayerType::Nobility {house} => {
				house.rm_unit(unit_ind, unit_template);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn chg_unit_ind(&mut self, unit_ind_frm: usize, unit_ind_to: usize, ut: &UnitTemplate) {
		self.stats.chg_brigade_unit_ind(unit_ind_frm, unit_ind_to);
		
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} => {
				ai_state.chg_unit_ind(unit_ind_frm, unit_ind_to, ut);
			} PlayerType::Barbarian {barbarian_state} => {
				barbarian_state.chg_unit_ind(unit_ind_frm, unit_ind_to);
			} PlayerType::Nobility {house} => {
				house.chg_unit_ind(unit_ind_frm, unit_ind_to, ut);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn log_damaged_wall(&mut self, coord: Coord) {
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} | PlayerType::Human {ai_state} => {
				ai_state.log_damaged_wall(coord);
			} PlayerType::Nobility {house} => {
				house.city_state.log_damaged_wall(coord);
			} PlayerType::Barbarian {..} => {}
		}
	}
	
	pub fn log_repaired_wall(&mut self, coord: Coord) {
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} | PlayerType::Human {ai_state} => {
				ai_state.log_repaired_wall(coord);
			} PlayerType::Nobility {house} => {
				house.city_state.log_repaired_wall(coord);
			} PlayerType::Barbarian {..} => {}
		}
	}
	
	pub fn chg_bldg_ind(&mut self, bldg_ind_frm: usize, bldg_ind_to: usize, bt: &BldgTemplate) {
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} => {
				ai_state.chg_bldg_ind(bldg_ind_frm, bldg_ind_to, bt);
			} PlayerType::Barbarian {barbarian_state} => {
				debug_assert!(barbarian_state.camp_ind == bldg_ind_frm);
				barbarian_state.camp_ind = bldg_ind_to;
			} PlayerType::Nobility {house} => {
				house.city_state.chg_bldg_ind(bldg_ind_frm, bldg_ind_to, bt);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn add_bldg(&mut self, bldg_ind: usize, coord: u64, bt: &BldgTemplate, map_sz: MapSz) {
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} => {
				ai_state.add_bldg(bldg_ind, coord, bt, map_sz);
			} PlayerType::Barbarian {..} => { // not logged
			} PlayerType::Nobility {house} => {
				house.city_state.register_bldg(bldg_ind, bt);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn rm_bldg(&mut self, bldg_ind: usize, bt: &BldgTemplate, disband_unit_inds: &mut Vec<usize>) {
		match &mut self.ptype {
			PlayerType::AI {ai_state, ..} => {
				ai_state.rm_bldg(bldg_ind, bt, disband_unit_inds);
			} PlayerType::Barbarian {..} => { // not logged
			} PlayerType::Nobility {house} => {
				house.unregister_bldg(bldg_ind, bt, disband_unit_inds);
			} PlayerType::Human {..} => {}
		}
	}
}

use crate::gcore::rand::XorState;
use crate::nn;
impl <'bt,'ut,'rt,'dt>Player<'bt,'ut,'rt,'dt> {
	pub fn new(id: SmSvType, ptype: PlayerType<'bt,'ut,'rt,'dt>, personality: AIPersonality,
			nm: String, ruler_nm: PersonName, gender_female: bool, bonuses: &Bonuses, color: i32, 
			txt_gen: &mut nn::TxtGenerator, relations: &mut Relations, nms: &Nms,
			tech_templates: &Vec<TechTemplate>, resource_templates: &Vec<ResourceTemplate>,
			doctrine_templates: &'dt Vec<DoctrineTemplate>, map_data: &MapData,
			rng: &mut XorState) -> Self {
		
		*relations = Relations::new(id as usize+1);
		
		let personalization = {
			let motto = txt_gen.gen_str(nn::TxtCategory::from(&personality));
			
			Personalization {
				color,
				nm, // of country
				gender_female,
				ruler_nm,
				doctrine_advisor_nm: PersonName::new(nms, rng).1,
				crime_advisor_nm: PersonName::new(nms, rng).1,
				pacifism_advisor_nm: PersonName::new(nms, rng).1,
				health_advisor_nm: PersonName::new(nms, rng).1,
				unemployment_advisor_nm: PersonName::new(nms, rng).1,
				city_nm_theme: rng.usize_range(0, nms.cities.len()),
				motto,
			}
		};
		
		Self {
			id,
			ptype,
			personalization,
			stats: Stats::default_init(id, bonuses, tech_templates, resource_templates, doctrine_templates, map_data),
			zone_exs: HashMap::with_hasher(Default::default()),
		}
	}
}

