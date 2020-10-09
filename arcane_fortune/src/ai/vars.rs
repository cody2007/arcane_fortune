use crate::map::{MapSz};
use crate::units::*;
use crate::buildings::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::saving::*;
use crate::disp_lib::endwin;
use crate::disp::Coord;
use crate::movement::{manhattan_dist};//, manhattan_dist_inds};
use crate::containers::Templates;
use crate::player::Player;
use super::*;

pub const ATTACK_SEARCH_TIMEOUT: usize = 90; // presently, for barbarians
pub const MAX_FAILED_MV_ATTEMPTS: usize = 3;// presently, for AI defenders

enum_From!{ Neighbors {Possible, NotPossible, NotKnown}}

#[derive(Clone, PartialEq)]
pub struct Defender {
	pub unit_ind: usize,
	pub failed_mv_attempts: usize // unit will be disbanded if this exceeds MAX_FAILED_MV_ATTEMPTS
}

impl_saving!{Defender {unit_ind, failed_mv_attempts}}

#[derive(Clone, PartialEq)]
pub struct CityState<'bt,'ut,'rt,'dt> {
	pub coord: u64, // city hall location
	pub gate_loc: u64, // location of gate in wall around city
	pub wall_coords: Vec<Coord>, // location of walls
	pub damaged_wall_coords: Vec<Coord>, // should be a subset of `wall_coords`
	
	pub city_ul: Coord, // upper left boundary of city
	pub city_lr: Coord, // lower right boundary of city
	// ^ used for determining defense area for city
	
	// update:
	//		ai_state.build_offensive_unit(),
	//		ai_state.add_bldg(), rm_bldg(), chg_bldg_ind(),
	//		city_state.register_bldg(), city_state.unregister_bldg()
	//		transfer_city_ai_state() ...
	// if more entries added here:
		pub ch_ind: Option<usize>, // city hall once it's created
		pub boot_camp_ind: Option<usize>, // boot camp once it's created
		pub academy_ind: Option<usize>, // academy once it's created
		pub bonus_bldg_inds: Vec<usize>,
	
	pub worker_actions: Vec<ActionMeta<'bt,'ut,'rt,'dt>>,
	
	// update:
	//		ai_state.add_unit(), rm_unit(), chng_unit_ind(),
	//		transfer_city_ai_state() ...
	// if more entries added here:
		pub worker_inds: Vec<usize>, // workers improving this city
		
		pub explorer_inds: Vec<usize>,
		
		pub defenders: Vec<Defender>, // defending units of the city
		pub defense_positions: Vec<u64>, // locations to put defenders
		// ^ ordered by priority, with first entry being where preference should be to place units
	
	pub neighbors_possible: Neighbors // possible to build neighboring cities?
}

impl_saving!{CityState<'bt,'ut,'rt,'dt> {coord, gate_loc, wall_coords,
	damaged_wall_coords, city_ul, city_lr,
	ch_ind, boot_camp_ind,
	academy_ind, bonus_bldg_inds,
	worker_actions, worker_inds, explorer_inds, defenders,
	defense_positions, neighbors_possible}}

#[derive(Clone, PartialEq)]
pub struct AIState<'bt,'ut,'rt,'dt> {
	pub city_states: Vec<CityState<'bt,'ut,'rt,'dt>>,
	pub attack_fronts: AttackFronts,
	pub icbm_inds: Vec<usize>,
	
	pub damaged_wall_coords: Vec<Coord>, // for logging of walls even if they are not contained in a city state
							 // (used w/ brigade automation actions, especially for the human player(s))
	
	pub next_bonus_bldg: Option<&'bt BldgTemplate<'ut,'rt,'dt>>,
	// ^ when ai_state.build_offensive_unit() = False, we build `next_bonus_bldg` (or
	// 	wait until we can afford to)
	//	ai_state.set_bonus_bldg() sets the new bonus
	
	pub goal_doctrine: Option<&'dt DoctrineTemplate>,
	
	pub paused: bool // to temporarily stop the AI from moving
}

impl_saving!{AIState<'bt,'ut,'rt,'dt> {city_states, attack_fronts, icbm_inds, damaged_wall_coords,
	next_bonus_bldg, goal_doctrine, paused}}

// unit record keeping, which city are the workers in
impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	pub fn add_unit(&mut self, unit_ind: usize, unit_coord: u64, unit_template: &UnitTemplate, pstats: &Stats,
			unit_templates: &Vec<UnitTemplate>, map_sz: MapSz) {
		// check to make sure not already added
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			for city_state in self.city_states.iter() {
				debug_assertq!(!city_state.worker_inds.contains(&unit_ind),
					"unit {} worker already added to {:?}", unit_ind, city_state.worker_inds);
				debug_assertq!(!city_state.explorer_inds.contains(&unit_ind),
					"unit {} explorer already added to {:?}", unit_ind, city_state.explorer_inds);
				debug_assertq!(!city_state.defenders.iter().any(|d| d.unit_ind == unit_ind),
					"unit {} defender already added to", unit_ind);
				//debug_assertq!(!city_state.attacker_inds.contains(&unit_ind),
				//	"unit {} attacker already added to owner {}", unit_ind, owner.nm);
			}
			debug_assertq!(!self.icbm_inds.contains(&unit_ind));
		}
		
		let unit_coord = Coord::frm_ind(unit_coord, map_sz);
		
		// find closest city and log unit as being part of it
		if let Some(min_city) = self.city_states.iter_mut().min_by_key(|c| 
					manhattan_dist(Coord::frm_ind(c.coord, map_sz), unit_coord, map_sz)) {
			
			match unit_template.nm[0].as_str() {
				WORKER_NM => {min_city.worker_inds.push(unit_ind);}
				EXPLORER_NM => {min_city.explorer_inds.push(unit_ind);}
				ICBM_NM => {self.icbm_inds.push(unit_ind);}
				RIOTER_NM => {}
				// attacker or defender
				_ => {
					// add as defender
					if min_city.defenders.len() < min_city.max_defenders() {
						min_city.defenders.push(Defender {unit_ind, failed_mv_attempts: 0});
					}else{
						self.attack_fronts.add_unit(unit_ind, unit_template, pstats, unit_templates);
					}
				}
			}
		}else{panicq!("no minimum city found, unit {}", unit_ind);}
	}
	
	pub fn rm_unit(&mut self, unit_ind: usize, unit_template: &UnitTemplate) {
		if self.city_states.len() == 0 {return;} // empire is destroyed
		
		match unit_template.nm[0].as_str() {
			WORKER_NM => {
				for city_state in self.city_states.iter_mut() {
					if let Some(pos) = city_state.worker_inds.iter().position(|&ind| ind == unit_ind) {
						city_state.worker_inds.swap_remove(pos);
						return;
					}
				}
			} EXPLORER_NM => {
				for city_state in self.city_states.iter_mut() {
					if let Some(pos) = city_state.explorer_inds.iter().position(|&ind| ind == unit_ind) {
						city_state.explorer_inds.swap_remove(pos);
						return;
					}
				}
			} ICBM_NM => {
				if let Some(pos) = self.icbm_inds.iter().position(|&icbm_ind| icbm_ind == unit_ind) {
					self.icbm_inds.swap_remove(pos);
					return;
				}
			} RIOTER_NM => {
			// defender or attacker
			} _ => {
				// if defender, remove
				for city_state in self.city_states.iter_mut() {
					if let Some(pos) = city_state.defenders.iter().position(|d| d.unit_ind == unit_ind) {
						city_state.defenders.swap_remove(pos);
						return;
					}
				}
				
				if self.attack_fronts.rm_unit(unit_ind) {return;}
			}
		}
		
		/*panicq!("{} id ({}) not contained in any city_state, owner: {} ({})",
				unit_template.nm,
				unit_ind, owner.nm, owner.id);*/
	}
	
	pub fn chg_unit_ind(&mut self, frm_ind: usize, to_ind: usize, unit_template: &UnitTemplate) {
		match unit_template.nm[0].as_str() {
			WORKER_NM => {
				//printlnq!("changing worker {} -> {} (owner {})", frm_ind, to_ind, owner.id);
				
				for city_state in self.city_states.iter_mut() {
					if city_state.worker_inds.iter_mut().any(|ind| {
						if *ind == frm_ind {
							*ind = to_ind;
							true
						}else{
							false
						}
					}){return;}
				}
			} EXPLORER_NM => {
				for city_state in self.city_states.iter_mut() {
					if city_state.explorer_inds.iter_mut().any(|ind| {
						if *ind == frm_ind {
							*ind = to_ind;
							true
						}else {false}
					}){return;}
				}
			} ICBM_NM => {
				if self.icbm_inds.iter_mut().any(|ind| {
					if *ind == frm_ind {
						*ind = to_ind;
						true
					}else {false}
				}){return;}
			} RIOTER_NM => {return;
			// defender or attacker
			} _ => {
				//printlnq!("changing soldier {} -> {} (owner {})", frm_ind, to_ind, owner.id);
				
				// if defender, change
				for city_state in self.city_states.iter_mut() {
					if city_state.defenders.iter_mut().any(|d| {
						if d.unit_ind == frm_ind {
							d.unit_ind = to_ind;
							true
						}else{
							false
						}
					}){return;}
				}
				
				if self.attack_fronts.chg_unit_ind(frm_ind, to_ind) {return;}
			}
		}
		
		// debug
		{
			#[cfg(any(feature="opt_debug", debug_assertions))]
			{
				for city_state in self.city_states.iter() {
					printlnq!("workers: {:?}", city_state.worker_inds);
					//printlnq!("defenders: ");
					//for ind in city_state.defender_inds.iter() {println!("{}", ind);}
				}
			}
			// if all cities are destroyed, the later cleanup involves removing all units
			// and in the process some unit indices may shift around such that the code
			// will change unit indices of civs that are in the process of being destroyed
			// the only condition where a unit isn't logged should be when there are no cities remaining
			assertq!(self.city_states.len() == 0,
				"could not change {} -> {} for {} in any city_state, n_cities: {}",
				frm_ind, to_ind, unit_template.nm[0], self.city_states.len());
		}
	}
}

// bldg record keeping
impl CityState<'_,'_,'_,'_> {
	pub fn max_defenders(&self) -> usize {
		self.defense_positions.len()*2
	}
	
	pub fn within_city_defense_area(&self, test_coord: Coord) -> bool {
		debug_assertq!(self.city_ul.y < self.city_lr.y);
		
		if self.city_ul.x < self.city_lr.x {
			test_coord.x >= self.city_ul.x &&
			test_coord.y >= self.city_ul.y &&
			test_coord.x <= self.city_lr.x &&
			test_coord.y <= self.city_lr.y
		}else{
			test_coord.y >= self.city_ul.y &&
			test_coord.y <= self.city_lr.y &&
			((test_coord.x >= self.city_ul.x) ||
			 (test_coord.x <= self.city_lr.x))
		}
	}
	
	// use ai_state.add_bldg() if the exact city is not known
	pub fn register_bldg(&mut self, bldg_ind: usize, bldg_template: &BldgTemplate) {
		if let BldgType::Taxable(_) = bldg_template.bldg_type {return;}
		
		match bldg_template.nm[0].as_str() {
			BOOT_CAMP_NM => {
				if self.boot_camp_ind.is_none() {
					self.boot_camp_ind = Some(bldg_ind);
				}
			}
			ACADEMY_NM => {
				if self.academy_ind.is_none() {
					self.academy_ind = Some(bldg_ind);
				}
			}
			_ => {
				// can already be added in the case of transfer_city_ai_state() which re-uses the
				// city_state of the attackee and does not clear the bldg inds
				if !self.bonus_bldg_inds.contains(&bldg_ind) {
					self.bonus_bldg_inds.push(bldg_ind);
				}
			}
		}
	}
	
	// returns true if found and set
	pub fn chg_bldg_ind(&mut self, frm_ind: usize, to_ind: usize, bldg_template: &BldgTemplate) -> bool {
		if let BldgType::Taxable(_) = bldg_template.bldg_type {return false;}
		
		match bldg_template.nm[0].as_str() {
			CITY_HALL_NM => {
				if self.ch_ind == Some(frm_ind) {
					self.ch_ind = Some(to_ind);
					return true;
				}
			} BOOT_CAMP_NM => {
				if self.boot_camp_ind == Some(frm_ind) {
					self.boot_camp_ind = Some(to_ind);
					return true;
				}
			} ACADEMY_NM => {
				if self.academy_ind == Some(frm_ind) {
					self.academy_ind = Some(to_ind);
					return true;
				}
			} _ => {
				for bonus_bldg_ind in self.bonus_bldg_inds.iter_mut() {
					if *bonus_bldg_ind == frm_ind {
						*bonus_bldg_ind = to_ind;
						return true;
					}
				}
			}
		}
		false
	}
	
	// add to `damaged_wall_coords`
	// returns true either if already added or now added
	pub fn log_damaged_wall(&mut self, wall_coord: Coord) -> bool {
		// already logged
		if self.damaged_wall_coords.contains(&wall_coord) {return true;}
		
		if self.wall_coords.contains(&wall_coord) {
			self.damaged_wall_coords.push(wall_coord);
			return true;
		}
		false
	}
	
	// remove from `damaged_wall_coords`
	// returns true if removed
	pub fn log_repaired_wall(&mut self, wall_coord: Coord) -> bool {
		for (list_ind, damaged_wall_coord) in self.damaged_wall_coords.iter().enumerate() {
			if *damaged_wall_coord == wall_coord {
				self.damaged_wall_coords.swap_remove(list_ind);
				return true;
			}
		}
		false
	}

}

// bldg record keeping -- keep track of city halls and boot camps
impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	pub fn add_bldg(&mut self, bldg_ind: usize, bldg_coord: u64, bldg_template: &BldgTemplate, map_sz: MapSz) {
		if let BldgType::Taxable(_) = bldg_template.bldg_type {return;}
		
		// check to make sure not already added
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			for city_state in self.city_states.iter() {
				if bldg_template.nm[0] == CITY_HALL_NM {
					debug_assertq!(city_state.ch_ind != Some(bldg_ind), "bldg {} already added", bldg_ind);
				}else if bldg_template.nm[0] == BOOT_CAMP_NM {
					debug_assertq!(city_state.boot_camp_ind != Some(bldg_ind), "bldg {} already added", bldg_ind);
				}else if bldg_template.nm[0] == ACADEMY_NM {
					debug_assertq!(city_state.academy_ind != Some(bldg_ind), "bldg {} already added", bldg_ind);
				}else{
					debug_assertq!(!city_state.bonus_bldg_inds.contains(&bldg_ind));
				}
				//}else{panicq!("invalid bldg type");}
			}
		}
		
		match bldg_template.nm[0].as_str() {
			CITY_HALL_NM => {
				for city in self.city_states.iter_mut() {
					if city.coord == bldg_coord {
						city.ch_ind = Some(bldg_ind);
						return;
					}
				}
			} BOOT_CAMP_NM => {
				if let Some(city) = self.find_closest_city(Coord::frm_ind(bldg_coord, map_sz), map_sz) {
					debug_assertq!(city.boot_camp_ind == None);
					city.boot_camp_ind = Some(bldg_ind);
					return;
				}
			} ACADEMY_NM => {
				if let Some(city) = self.find_closest_city(Coord::frm_ind(bldg_coord, map_sz), map_sz) {
					debug_assertq!(city.academy_ind == None);
					city.academy_ind = Some(bldg_ind);
					return;
				}
			} _ => {
				if let Some(city) = self.find_closest_city(Coord::frm_ind(bldg_coord, map_sz), map_sz) {
					city.bonus_bldg_inds.push(bldg_ind);
					return;
				}
			}
		}
		
		////////////// debug
		// can occur if human player builds a building far from a city
		/*{
			for city in self.city_states.iter() {
				printlnq!("city coord {} dist {} ch_ind none {}", city.coord, manhattan_dist_inds(city.coord, bldg_coord, map_sz),
						city.ch_ind.is_none());
			}
			panicq!("no matching city found, bldg {}, owner: {} {}, bldg_coord {}, city_states.len() {}, bldg nm {}",
					bldg_ind, owner.id, owner.nm, bldg_coord, self.city_states.len(), bldg_template.nm);
		}*/
	}
	
	pub fn rm_bldg(&mut self, bldg_ind: usize, bldg_template: &BldgTemplate, disband_unit_inds: &mut Vec<usize>) {
		if let BldgType::Taxable(_) = bldg_template.bldg_type {return;}
		
		match bldg_template.nm[0].as_str() {
			CITY_HALL_NM => {
				for (city_ind, city_state) in self.city_states.iter_mut().enumerate() {
					// remove entire city and all its units
					if city_state.ch_ind == Some(bldg_ind) {
						// rm units
						disband_unit_inds.append(&mut city_state.worker_inds);
						disband_unit_inds.append(&mut city_state.explorer_inds);
						for defender in city_state.defenders.iter() {
							disband_unit_inds.push(defender.unit_ind);
						}
						
						self.city_states.swap_remove(city_ind);
						
						return;
					}
				}
			} BOOT_CAMP_NM => {
				for city_state in self.city_states.iter_mut() {
					if city_state.boot_camp_ind == Some(bldg_ind) {
						city_state.boot_camp_ind = None;
						return;
					}
				}
			} ACADEMY_NM => {
				for city_state in self.city_states.iter_mut() {
					if city_state.academy_ind == Some(bldg_ind) {
						city_state.academy_ind = None;
						return;
					}
				}
			} _ => {
				for city_state in self.city_states.iter_mut() {
					for (list_ind, bonus_bldg_ind) in city_state.bonus_bldg_inds.iter().enumerate() {
						if *bonus_bldg_ind == bldg_ind {
							city_state.bonus_bldg_inds.swap_remove(list_ind);
							return;
						}
					}
				}
			}
		}
	}
	
	pub fn chg_bldg_ind(&mut self, frm_ind: usize, to_ind: usize, bldg_template: &BldgTemplate) {
		if let BldgType::Taxable(_) = bldg_template.bldg_type {return;}
		
		self.city_states.iter_mut().any(|cs| cs.chg_bldg_ind(frm_ind, to_ind, bldg_template));
	}
	
	// add to `damaged_wall_coords`
	pub fn log_damaged_wall(&mut self, wall_coord: Coord) {
		// add to city list
		self.city_states.iter_mut().any(|c| c.log_damaged_wall(wall_coord));
		
		// add to global list
		if !self.damaged_wall_coords.contains(&wall_coord) {
			self.damaged_wall_coords.push(wall_coord);
		}
	}
	
	// remove from `damaged_wall_coords`
	pub fn log_repaired_wall(&mut self, wall_coord: Coord) {
		// add to city list
		self.city_states.iter_mut().any(|c| c.log_repaired_wall(wall_coord));
		
		// remove from the global list
		for (list_ind, damaged_wall_coord) in self.damaged_wall_coords.iter().enumerate() {
			if *damaged_wall_coord == wall_coord {
				self.damaged_wall_coords.swap_remove(list_ind);
				return;
			}
		}
	}
}

use crate::zones::{set_owner, set_all_adj_owner};
use crate::gcore::Log;
use crate::gcore::hashing::HashedMapZoneEx;

// `ch_ind`: city hall bldg ind to transfer (previously owned by `frm_owner_id`
// returns the transfered CityState set to be owned by `to_owner_id`
pub fn transfer_city_ai_state<'bt,'ut,'rt,'dt>(ch_ind: usize, frm_owner_id: usize, to_owner_id: usize, cur_player: usize,
		units: &Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
		logs: &mut Vec<Log>, map_sz: MapSz, turn: usize) -> Option<CityState<'bt,'ut,'rt,'dt>> {
	// remove entry from sender
	let player_frm = &mut players[frm_owner_id];
	let pstats_frm = &mut player_frm.stats;
	if let Some(frm_ai_state) = player_frm.ptype.ai_state_mut() {
		if let Some(city_state_ind) = frm_ai_state.city_states.iter().position(|cs| cs.ch_ind == Some(ch_ind)) {
			let mut city_state = frm_ai_state.city_states.swap_remove(city_state_ind);
			
			macro_rules! re_add_unit{($unit_ind: expr) => {
				let u = &units[$unit_ind];
				frm_ai_state.add_unit($unit_ind, u.return_coord(), u.template, &pstats_frm, temps.units, map_sz);
			}};
			
			// remove & re-add units so they are registered in another city_state
			if frm_ai_state.city_states.len() != 0 {
				for unit_ind in city_state.worker_inds.drain(..) {re_add_unit!(unit_ind);}
				for unit_ind in city_state.explorer_inds.drain(..) {re_add_unit!(unit_ind);}
				for d in city_state.defenders.drain(..) {re_add_unit!(d.unit_ind);}
			// no cities left; remove units from city_state
			}else{
				city_state.worker_inds.clear();
				city_state.explorer_inds.clear();
				city_state.defenders.clear();
			}
			
			// transfer wall ownerships
			for wall_coord in city_state.wall_coords.iter() {
				//printlnq!("transfering wall {}", wall_coord);
				set_owner(wall_coord.to_ind(map_sz) as u64, to_owner_id, frm_owner_id, cur_player, &mut None, units, bldgs, temps, exs, players, map_data, relations, logs, map_sz, turn);
			}
			
			//printlnq!("transfering city from {} to {}", frm_owner_id, to_owner_id);
			
			// add owner to receiver
			if let Some(_to_ai_state) = players[to_owner_id].ptype.ai_state() {
				//printlnq!("transfering city hall from owner {} to {}", frm_owner_id, to_owner_id);
				
				//city_state.worker_actions.clear();
				
				// transfer bldg ownerships
				{
					if let Some(boot_camp_ind) = city_state.boot_camp_ind {
						let b = &bldgs[boot_camp_ind];
						set_all_adj_owner(vec![b.coord], to_owner_id, frm_owner_id, cur_player, &mut Some(&mut city_state), units, bldgs, temps, exs, players, map_data, relations, logs, map_sz, turn);
					}
					
					if let Some(academy_ind) = city_state.academy_ind {
						let b = &bldgs[academy_ind];
						set_all_adj_owner(vec![b.coord], to_owner_id, frm_owner_id, cur_player, &mut Some(&mut city_state), units, bldgs, temps, exs, players, map_data, relations, logs, map_sz, turn);
					}
					
					let mut bonus_coords = Vec::with_capacity(city_state.bonus_bldg_inds.len());
					for bonus_bldg_ind in city_state.bonus_bldg_inds.iter() {
						bonus_coords.push(bldgs[*bonus_bldg_ind].coord);
					}
					set_all_adj_owner(bonus_coords, to_owner_id, frm_owner_id, cur_player, &mut Some(&mut city_state), units, bldgs, temps, exs, players, map_data, relations, logs, map_sz, turn);
				}
				
				return Some(city_state);
			}
		}else{
			for city in frm_ai_state.city_states.iter() {
				printlnq!("city coord {} ch_ind is none {}", Coord::frm_ind(city.coord, map_sz), city.ch_ind.is_none());
			}
			
			///////// dbg
			for (bldg_ind, b) in bldgs.iter().enumerate() {
				if b.owner_id == to_owner_id as SmSvType {
					if let BldgArgs::CityHall {..} = b.args {
						printlnq!("city hall coord {} {}", Coord::frm_ind(b.coord, map_sz), bldg_ind);
					}
				}
			}
			
			panicq!("could not find city hall in ai states ch_ind {} frm_owner_id {} to_owner_id {}",
						ch_ind, frm_owner_id, to_owner_id);
		}
	
	// add new entry to receiver (sender is human)
	}else if let Some(to_ai_state) = players[to_owner_id].ptype.ai_state_mut() {
		let b = &bldgs[ch_ind];
		
		let mut gate_loc = Coord::frm_ind(b.coord, map_sz);
		gate_loc.y += (b.template.sz.h + 2) as isize;
		
		// city boundaries
		let (city_ul, city_lr) = {
			let b_coord = Coord::frm_ind(b.coord, map_sz);
			
			let hheight = (CITY_HEIGHT/2) as isize;
			let hwidth = (CITY_WIDTH/2) as isize;
			
			let city_ul = if let Some(city_ul_ind) = map_sz.coord_wrap(b_coord.y - hheight,
												     b_coord.x - hwidth) {
				Coord::frm_ind(city_ul_ind, map_sz)
			
			// y must've been less than zero
			}else{
				Coord {y: 0, x: b_coord.x - hwidth }.wrap(map_sz)
			};
			
			let city_lr = if let Some(city_lr_ind) = map_sz.coord_wrap(b_coord.y + hheight,
												     b_coord.x + hwidth) {
				Coord::frm_ind(city_lr_ind, map_sz)
			
			// y must've extended off the map
			}else{
				Coord {y: map_sz.h as isize - 1, x: b_coord.x + hwidth }.wrap(map_sz)
			};

			(city_ul, city_lr)
		};
		
		return Some(CityState {
				coord: b.coord,
				gate_loc: gate_loc.to_ind(map_sz) as u64,
				wall_coords: Vec::new(),
				damaged_wall_coords: Vec::new(),
				
				city_ul, city_lr,
				
				ch_ind: Some(ch_ind),
				boot_camp_ind: None,
				academy_ind: None,
				bonus_bldg_inds: Vec::new(),
				
				worker_actions: Vec::new(),
				worker_inds: Vec::new(),
				explorer_inds: Vec::new(),
				
				defenders: Vec::new(),
				defense_positions: Vec::new(),
				
				neighbors_possible: Neighbors::NotKnown
		});
	}
	
	None
}

