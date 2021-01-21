use std::collections::HashMap;
use crate::disp::Coord;
use crate::saving::*;
use crate::gcore::*;
use crate::buildings::*;
use crate::units::UnitTemplate;
use crate::map::{MapData, MapSz};
use crate::containers::*;
use crate::ai::*;
use crate::nn;

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

// unit & bldg management functions
impl <'bt,'ut,'rt,'dt>Player<'bt,'ut,'rt,'dt> {
	pub fn add_unit(&mut self, unit_ind: usize, unit_coord: u64, unit_template: &UnitTemplate,
			unit_templates: &Vec<UnitTemplate>, map_sz: MapSz) {
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) => {
				ai_state.add_unit(unit_ind, unit_coord, unit_template, &self.stats, unit_templates, map_sz);
			} PlayerType::Barbarian(barbarian_state) => {
				barbarian_state.add_unit(unit_ind, unit_template);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn rm_unit(&mut self, unit_ind: usize, unit_template: &UnitTemplate) {
		self.stats.rm_unit_frm_brigade(unit_ind);
		
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) => {
				ai_state.rm_unit(unit_ind, unit_template);
			} PlayerType::Barbarian(barbarian_state) => {
				barbarian_state.rm_unit(unit_ind);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn chg_unit_ind(&mut self, unit_ind_frm: usize, unit_ind_to: usize, ut: &UnitTemplate) {
		self.stats.chg_brigade_unit_ind(unit_ind_frm, unit_ind_to);
		
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) => {
				ai_state.chg_unit_ind(unit_ind_frm, unit_ind_to, ut);
			} PlayerType::Barbarian(barbarian_state) => {
				barbarian_state.chg_unit_ind(unit_ind_frm, unit_ind_to);
			} PlayerType::Human {..} => {}
		}
	}
	
	pub fn log_damaged_wall(&mut self, coord: Coord) {
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) |
			PlayerType::Human(ai_state) => {
				ai_state.log_damaged_wall(coord);
			} PlayerType::Barbarian {..} => {}
		}
	}
	
	pub fn log_repaired_wall(&mut self, coord: Coord) {
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) |
			PlayerType::Human(ai_state) => {
				ai_state.log_repaired_wall(coord);
			} PlayerType::Barbarian(_) => {}
		}
	}
	
	pub fn chg_bldg_ind(&mut self, bldg_ind_frm: usize, bldg_ind_to: usize, b: &Bldg) {
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) => {
				ai_state.chg_bldg_ind(bldg_ind_frm, bldg_ind_to, b);
			} PlayerType::Barbarian(barbarian_state) => {
				debug_assert!(barbarian_state.camp_ind == bldg_ind_frm);
				barbarian_state.camp_ind = bldg_ind_to;
			} PlayerType::Human(_) => {}
		}
	}
	
	pub fn add_bldg(&mut self, bldg_ind: usize, coord: u64, b: &Bldg, map_sz: MapSz) {
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) => {
				ai_state.add_bldg(bldg_ind, coord, b, map_sz);
			} PlayerType::Barbarian(_) => { // not logged
			} PlayerType::Human(_) => {}
		}
	}
	
	pub fn rm_bldg(&mut self, bldg_ind: usize, b: &Bldg, disband_unit_inds: &mut Vec<usize>) {
		match &mut self.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) => {
				ai_state.rm_bldg(bldg_ind, b, disband_unit_inds);
			} PlayerType::Barbarian(_) => { // not logged
			} PlayerType::Human(_) => {}
		}
	}
}

// combat bonus
impl <'bt,'ut,'rt,'dt>Player<'bt,'ut,'rt,'dt> {
	pub fn combat_bonus_factor(&self) -> f32 {
		let factor = if let Some(personality) = self.ptype.personality() {
			if personality.diplomatic_friendliness_bonus() {0.75
			}else if personality.diplomatic_friendliness_penalty() {1.5
			}else{1.}
		}else{1.};
		
		self.stats.bonuses.combat_factor * factor
	}
	
	pub fn tech_bonus_factor(&self) -> f32 {
		if let Some(personality) = self.ptype.personality() {
			if personality.scientific() {
				return 1.75;
			}
		}
		1.
	}
}

impl <'bt,'ut,'rt,'dt>Player<'bt,'ut,'rt,'dt> {
	pub fn new(id: SmSvType, ptype: PlayerType<'bt,'ut,'rt,'dt>, personality: AIPersonality,
			nm: String, ruler_nm: PersonName, gender_female: bool, bonuses: &Bonuses, color: i32, 
			txt_gen: &mut nn::TxtGenerator, gstate: &mut GameState, n_log_entries: usize,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &MapData) -> Self {
		
		gstate.relations.add_player();
		
		let personalization = Personalization {
			color,
			nm, // of country
			gender_female,
			ruler_nm,
			doctrine_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			crime_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			pacifism_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			health_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			unemployment_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			city_nm_theme: gstate.rng.usize_range(0, temps.nms.cities.len()),
			motto: txt_gen.gen_str(nn::TxtCategory::from(&personality)),
			founded: gstate.turn
		};
		
		Self {
			id,
			ptype,
			personalization,
			stats: Stats::default_init(id, bonuses, temps, map_data, n_log_entries),
			zone_exs: HashMap::with_hasher(Default::default()),
		}
	}
}

