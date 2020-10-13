use crate::map::*;
use crate::units::*;
use crate::buildings::*;
use crate::movement::*;
use crate::disp::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::zones::{HappinessCategory, PacifismMilitarism};
use crate::disp_lib::DispState;
use crate::saving::*;
use crate::disp_lib::endwin;
use crate::ai::*;
use crate::localization::Localization;
use crate::containers::Templates;
use crate::player::{Stats, Player, PlayerType};

pub const HUMAN_PLAYER_IND: usize = 0; // assumed to be one because new_game() adds barbarians interspersed with AI/Human players
pub const HUMAN_PLAYER_ID: u32 = HUMAN_PLAYER_IND as u32;

pub mod profiling;
pub mod hashing;
pub mod rand;
pub mod sectors; pub use sectors::*;
pub mod brigades; pub use brigades::*;
pub mod log; pub use log::*;
mod end_turn;
mod non_menu_keys;
mod relations;
mod economy;
mod difficulty;
mod testing;

pub use profiling::*;
pub use hashing::*;
pub use rand::*;
pub use end_turn::*;
pub use non_menu_keys::*;
pub use relations::*;
pub use economy::*;
pub use difficulty::*;
pub use testing::*;

/*#[inline]
pub fn approx_eq(a: f32, b: f32) -> bool {
	((a + std::f32::EPSILON) >= b) && ((a - std::f32::EPSILON) <= b)
}*/

#[inline]
pub fn approx_eq_tol(a: f32, b: f32, tol: f32) -> bool {
	((a + tol) >= b) && ((a - tol) <= b)
}

fn in_debt(pstats: &Stats) -> bool {
	let assets = pstats.gold + pstats.net_income();
	assets < 0. && !approx_eq_tol(assets, 0., 0.001)
}

impl Stats<'_,'_,'_,'_> {
	pub fn net_income(&self) -> f32 {
		self.bonuses.gold_per_day + self.tax_income - self.unit_expenses - self.bldg_expenses
	}
}

// either from collapse or take-over
pub fn civ_destroyed(player: &mut Player, relations: &mut Relations, iface_settings: &mut IfaceSettings, turn: usize, d: &mut DispState) {
	player.stats.alive = false;
	player.stats.gold = 0.;
	
	for war_enemy in relations.at_war_with(player.id as usize) {
		relations.declare_peace_wo_logging(war_enemy, player.id as usize, turn);
	}
	
	// human player end game
	if player.id == iface_settings.cur_player {
		iface_settings.player_end_game(relations, d);
	
	// ai end game -- clear attack fronts
	}else{
		match &mut player.ptype {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) |
			PlayerType::Human(ai_state) => {
				//debug_assertq!(ai_state.city_states.len() == 0, "n_cities: {}", ai_state.city_states.len());
				// ^ a city_state could exist, but \/ none should have a city hall
				debug_assertq!(!ai_state.city_states.iter().any(|cs| !cs.population_center_ind.is_none()));
				
				ai_state.city_states = Vec::new();
				ai_state.attack_fronts.vals = Vec::new();
			} PlayerType::Barbarian(_) => {}
		}
	}
}

// for destroying a civ
pub fn rm_player_zones<'bt,'ut,'rt,'dt>(owner_id: usize, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>, 
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, map_sz: MapSz) {
	// gather ex coords
	let exf = exs.last().unwrap();
	let mut coords = Vec::with_capacity(exf.len());
	for coord in exf.keys() {
		coords.push(*coord);
	}
	
	// rm zones
	let mut coords_rmd = Vec::with_capacity(exf.len());
	for coord in coords {
		let ex = exs.last_mut().unwrap().get_mut(&coord).unwrap();
		if ex.actual.owner_id == Some(owner_id as SmSvType) {
			ex.actual.rm_zone(coord, players, temps.doctrines, map_sz);
			coords_rmd.push(coord);
		}
	}
	
	// update map
	for coord in coords_rmd {
		compute_zooms_coord(coord, bldgs, temps.bldgs, map_data, exs, players);
	}
}

pub fn worker_inds(unit_inds: &Vec<usize>, units: &Vec<Unit>) -> Vec<usize> {
	let mut unit_inds_keep = Vec::with_capacity(unit_inds.len());
	for unit_ind in unit_inds {
		if units[*unit_ind].template.nm[0] == WORKER_NM {
			unit_inds_keep.push(*unit_ind);
		}
	}
	unit_inds_keep
}

