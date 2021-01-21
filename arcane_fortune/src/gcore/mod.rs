use crate::map::*;
use crate::units::*;
use crate::buildings::*;
use crate::movement::*;
use crate::disp::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::zones::{HappinessCategory, PacifismMilitarism};
use crate::renderer::*;
use crate::saving::*;
use crate::ai::*;
use crate::containers::*;
use crate::player::{Stats, Player, PlayerType};

pub const HUMAN_PLAYER_IND: usize = 0; // assumed to be one because new_game() adds barbarians interspersed with AI/Human players
pub const HUMAN_PLAYER_ID: u32 = HUMAN_PLAYER_IND as u32;

pub const TURNS_PER_YEAR: usize = 360;
pub const GAME_START_TURN: usize = 100*TURNS_PER_YEAR;

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

fn in_debt(pstats: &Stats, players: &Vec<Player>, relations: &Relations) -> bool {
	let assets = pstats.gold + pstats.net_income(players, relations);
	assets < 0. && !approx_eq_tol(assets, 0., 0.001)
}

impl Stats<'_,'_,'_,'_> {
	// fiefdoms have this value taxed
	pub fn tax_income(&self, players: &Vec<Player>, relations: &Relations) -> f32 {
		let cur_player = self.id as usize;
		
		let pre_tax_tax_income = |player: &Stats| {
			player.tax_income + player.bonuses.gold_per_day
		};
		
		let mut tax_income = pre_tax_tax_income(self);
		
		match players[cur_player].ptype {
			// sum contributions from noble houses
			PlayerType::Human(_) | PlayerType::Empire(_) => {
				for player_ind in (0..players.len())
						.filter(|&player_ind| player_ind != cur_player) {
					if let Some(tax_rate) = relations.tax_rate(player_ind, cur_player) {
						tax_income += 
							(tax_rate as f32 / 100.) * 
							pre_tax_tax_income(&players[player_ind].stats);
					}
				}
			}
			
			// subtract income if part of an empire
			PlayerType::Nobility(_) => {
				for player_ind in (0..players.len())
						.filter(|&player_ind| player_ind != cur_player) {
					if let Some(tax_rate) = relations.tax_rate(player_ind, cur_player) {
						tax_income *= 1. - (tax_rate as f32 / 100.);
					}
				}
			}
			
			PlayerType::Barbarian(_) => {}
		}
		tax_income
	}
	
	pub fn net_income(&self, players: &Vec<Player>, relations: &Relations) -> f32 {
		self.tax_income(players, relations)
		-self.unit_expenses 
		-self.bldg_expenses
		+relations.trade_gold_per_turn(self.id as usize)
	}
}

// budget issues or no succession for noble house
// if disband_unit_inds is supplied no units are removed, but they are instead stored in that variable
// calls civ_destroyed()
pub fn civ_collapsed<'bt,'ut,'rt,'dt>(owner_ind: usize, disband_unit_inds: &mut Option<&mut Vec<usize>>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, 
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
		map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		gstate: &mut GameState, map_sz: MapSz, temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp) {
	let owner_id = owner_ind as SmSvType;
	
	// store units to be removed in disband_unit_inds
	if let Some(disband_unit_inds) = disband_unit_inds {
		for (unit_ind, _) in units.iter().enumerate().filter(|(_, u)| owner_id == u.owner_id) {
			disband_unit_inds.push(unit_ind);
		}
	// rm units in reverse order to avoid index issues
	}else{
		for unit_ind in (0..units.len()).rev() {
			if (owner_id as SmSvType) == units[unit_ind].owner_id {
				disband_unit(unit_ind, owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, players, gstate, map_sz);
			}
		}
	}
	
	// rm bldgs in reverse order to avoid index issues (the tax paying bldgs weren't removed above)
	for bldg_ind in (0..bldgs.len()).rev() {
		if (owner_id as SmSvType) == bldgs[bldg_ind].owner_id {
			rm_bldg(bldg_ind, owner_id == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
		}
	}
	
	rm_player_zones(owner_ind, bldgs, temps, players, exs, map_data, map_sz);
	civ_destroyed(players.len(), &mut players[owner_ind], gstate, disp);
}

// either from collapse or take-over
// (called in all conditions)
pub fn civ_destroyed(n_players: usize, player: &mut Player, gstate: &mut GameState, disp: &mut Disp) {
	player.stats.alive = false;
	player.stats.gold = 0.;
	
	// declare peace w/ everyone
	for owner_id in (0..n_players).filter(|&owner_id| owner_id != player.id as usize) {
		gstate.relations.declare_peace_wo_logging(owner_id, player.id as usize, gstate.turn);
	}
	
	// human player end game
	if player.id == disp.state.iface_settings.cur_player {
		disp.player_end_game(&mut gstate.relations);
	
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

