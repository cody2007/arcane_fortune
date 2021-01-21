use std::cmp::max;
use crate::saving::*;
use crate::units::*;
use crate::buildings::*;
use crate::resources::*;
use crate::map::{MapData, MapSz, StructureType};
use crate::player::Stats;
use crate::gcore::Relations;
use crate::doctrine::DoctrineTemplate;
use crate::gcore::hashing::HashedMapEx;
use crate::disp::{Coord, ScreenSz, ActionInterfaceMeta};
use crate::renderer::endwin;
use crate::ai::{CITY_HEIGHT, CITY_WIDTH, AI_MAX_SEARCH_DEPTH, set_target_attackable};
use crate::movement::{MvVars, manhattan_dist_inds, manhattan_dist, movable_to};
use crate::containers::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;
use crate::gcore::{dbg_log, Log};
//use crate::disp::IfaceSettings;

const MAX_ATTACK_FRONTS_PER_CITY: usize = 10;

const UNITS_PER_POSITION: usize = 2;
const POSITIONS_PER_LINE: usize = 3;
const LINES_PER_FRONT: usize = 2;
const UNITS_PER_FRONT: usize = UNITS_PER_POSITION * POSITIONS_PER_LINE * LINES_PER_FRONT;

const SIEGERS_PER_FRONT: usize = UNITS_PER_POSITION * POSITIONS_PER_LINE;
const ATTACKERS_PER_FRONT: usize = UNITS_PER_FRONT - SIEGERS_PER_FRONT;

#[derive(Clone, PartialEq)]
pub enum AttackFrontState {
	Recruitment {
		unreachable_city_coords: Vec<u64>
		// ^ if the units cannot access the assemble_location when the state is
		//   `AssembleToLocation`, then we move the state back to Recruitment and add
		//	the coordinate to this vector
	},
	AssembleToLocation {
		assemble_location: u64, // meeting point, lower right square was chosen to have nothing on it
		target_city_coord: u64,
		unreachable_city_coords: Vec<u64> // initialized from the Recruitment values
							    // and used to reset to Recruitment if assembly fails
	},
	WallAttack {
		target_city_coord: u64,
		wall_coord: u64,
		attacks_initiated: bool,
		// ^ set to true once the siege units are sent
		//	in to attack. state is progressed via progress_state_to_attack_wall_or_city()
		//	once all units are idle again
	},
	CityAttack {
		target_city_coord: u64,
		attacks_initiated: bool,
		// ^ set to true once all attackers have attacked the city hall
		//    once they have, the state is set back to recruitment for staging a new attack
	}
}

#[derive(Clone, PartialEq)]
pub struct AttackFront {
	// update ai_state.build_offensive_unit() if this changes
	
	pub siegers: Vec<usize>, // unit inds
	pub attackers: Vec<usize>, // unit inds
	pub state: AttackFrontState,
}

impl_saving!{AttackFront {siegers, attackers, state}}

impl AttackFront {
	// unfilled unit positions for the attack front
	fn n_empty_slots(&self) -> usize {
		debug_assertq!((self.siegers.len() + self.attackers.len()) <= UNITS_PER_FRONT, "siegers {} attackers {} units_per_front {}", self.siegers.len(), self.attackers.len(), UNITS_PER_FRONT);
		UNITS_PER_FRONT - self.siegers.len() - self.attackers.len()
	}
	
	// based on where the units are located, and skipping over
	// any city in AttackFrontState {unreachable_city_coords}
	fn find_nearest_war_city_loc(&self, ai_ind: usize, unreachable_city_coords: &Vec<u64>, 
			units: &Vec<Unit>, bldgs: &Vec<Bldg>, relations: &Relations, map_sz: MapSz) -> Option<u64> {
		let civs_at_war_with = relations.at_war_with(ai_ind);
		debug_assertq!(civs_at_war_with.len() != 0);
		let mut min_dist = None;
		let mut min_city_coord = None;
		for b in bldgs.iter().filter(|b| {
			if b.template.nm[0] == BARBARIAN_CAMP_NM {true} else
			if let BldgArgs::PopulationCenter {..} = &b.args {true} else {false}
		}) {
			// if we've already failed to find a route, or are not at war with this civ, then skip
			if unreachable_city_coords.contains(&b.coord) || !civs_at_war_with.contains(&(b.owner_id as usize)) {continue;}
			
			let mut dist = 0;
			for sieger in self.siegers.iter() {
				dist += manhattan_dist_inds(units[*sieger].return_coord(), b.coord, map_sz);
			}
			
			if let Some(ref mut min_dist) = &mut min_dist {
				if *min_dist > dist {
					*min_dist = dist;
					min_city_coord = Some(b.coord);
				}
			}else{
				min_dist = Some(dist);
				min_city_coord = Some(b.coord);
			}
		}
		
		min_city_coord
	}
	
	// checks if recruitment is done, then progresses to assembly mode where units gather near the city to be attacked
	fn progress_state_to_assembly(&mut self, ai_ind: usize, units: &Vec<Unit>,
			target_city_coord_opt: Option<u64>, bldgs: &Vec<Bldg>, 
			map_data: &mut MapData, exf: &HashedMapEx, gstate: &mut GameState, map_sz: MapSz) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attack_fronts.progress_state_to_assembly");
		
		const N_ATTEMPTS: usize = 200;
		
		if let AttackFrontState::Recruitment {unreachable_city_coords} = &self.state {
			if self.n_empty_slots() != 0 {return;} // chk that all units have been added to the attack front
			
			// if a target city is not already supplied, find one
			let target_city_coord_opt = if target_city_coord_opt.is_none() {
				self.find_nearest_war_city_loc(ai_ind, unreachable_city_coords, units, bldgs, &gstate.relations, map_sz)
			}else{target_city_coord_opt};
			
			if let Some(target_city_coord) = target_city_coord_opt {
				/////////////////////////////// determine assembly point
				let loc_dist = max(CITY_HEIGHT, CITY_WIDTH) as isize;
				let blank_spot = {
					let length = max(LINES_PER_FRONT, POSITIONS_PER_LINE);
					ScreenSz {h: length, w: length, sz: 0}
				};
				
				let city_coord = Coord::frm_ind(target_city_coord, map_sz);
				for _attempt in 0..N_ATTEMPTS {
					let (y_off, x_off) = {
						let sign = if gstate.rng.usize_range(0, 2) == 0 {1} else {-1};
						
						if gstate.rng.usize_range(0, 2) == 0 {
							(sign*loc_dist, gstate.rng.isize_range(0, loc_dist))
						}else{
							(gstate.rng.isize_range(0, loc_dist), sign*loc_dist)
						}
					};
					
					if let Some(assemble_location) = map_sz.coord_wrap(city_coord.y + y_off, city_coord.x + x_off) {
						if let Some(_) = square_clear(assemble_location, blank_spot, Quad::Lr, map_data, exf) {
							self.state = AttackFrontState::AssembleToLocation {
										assemble_location, target_city_coord,
										unreachable_city_coords: unreachable_city_coords.clone()
							};
							return;
						}
					}
				}
				//printlnq!("could not find assembly location");
			}//else{printlnq!("could not find target city (attacker: {}, unreachable_city_coords: {:?}, wars: {:?})", ai_ind, unreachable_city_coords, relations.at_war_with(ai_ind));}
		}else{panicq!("state should be recruitment");}
	}
	
	// move units to assembly location, near the city where they will attack but not too close
	// after assembly, they will attack
	fn assemble_to_location(&mut self, units: &mut Vec<Unit>, bldgs: &Vec<Bldg>, 
			map_data: &mut MapData, exs: &Vec<HashedMapEx>, map_sz: MapSz, logs: &mut Vec<Log>) {//, iface_settings: &mut IfaceSettings) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attack_fronts.assemble_to_location");
		
		const DIST_ACCEPTED: usize = POSITIONS_PER_LINE*2; // acceptable distance for unit to be away from assembly location and to still consider it assembled
		if let AttackFrontState::AssembleToLocation {assemble_location, unreachable_city_coords, target_city_coord} = &self.state {
			let loc = Coord::frm_ind(*assemble_location, map_sz);
			
			let mut all_assembled = true;
			
			// for each unit, check if assembled to location, if not move to location
			macro_rules! assemble{($stack: ident, $row_offset: expr) => {
				for (pos, unit_ind) in self.$stack.iter().enumerate() {
					let u = &units[*unit_ind];
					if u.action.len() != 0 {
						all_assembled = false;
						continue;
					}
					
					let unit_coord = Coord::frm_ind(u.return_coord(), map_sz);
					if manhattan_dist(unit_coord, loc, map_sz) > DIST_ACCEPTED {
						all_assembled = false;
						
						let mut action_iface = ActionInterfaceMeta {
							action: ActionMeta::new(ActionType::MvIgnoreOwnWalls),
							unit_ind: Some(*unit_ind),
							max_search_depth: AI_MAX_SEARCH_DEPTH,
							start_coord: unit_coord,
							movement_type: u.template.movement_type,
							movable_to: &movable_to
						};
						
						let dest_coord = Coord {y: loc.y + $row_offset, x: loc.x + (pos/2) as isize};
						
						//printlnq!("pre-assembly move start");
						action_iface.update_move_search(dest_coord, map_data, exs, MvVars::NonCivil{units, start_owner: units[*unit_ind].owner_id, blind_undiscov: None}, bldgs);
						//printlnq!("path coords len {}", action_iface.action.path_coords.len());
						
						// cannot move to target, try another
						if action_iface.action.path_coords.len() == 0 {
							/*printlnq!("could not assemble to location dest_coord {} start_coord {} assemble location {} (attacker {} attackee {}) unreachable_coords: {:?}",
									dest_coord, unit_coord, loc,
									units[*unit_ind].owner_id, exs.last().unwrap().get(target_city_coord).unwrap().actual.owner_id.unwrap(),
									unreachable_city_coords);*/
							
							let mut unreachable_city_coords = unreachable_city_coords.clone();
							unreachable_city_coords.push(*target_city_coord);
							
							self.state = AttackFrontState::Recruitment {unreachable_city_coords};
							return;
						}
						
						units[*unit_ind].action.push(action_iface.action);
					}
				}
			};};
			
			//printlnq!("{:?}", unreachable_city_coords);
			
			assemble!(siegers, 0);
			assemble!(attackers, 1);
			
			//printlnq!("{}", all_assembled);
			
			if all_assembled {self.progress_state_to_attack_wall_or_city(units, bldgs, map_data, exs, map_sz, logs);}//, iface_settings);}
		}else{panicq!("attempted to assemble to location without being in correct mode");}
	}
	
	fn progress_state_to_attack_wall_or_city(&mut self, units: &mut Vec<Unit>, bldgs: &Vec<Bldg>,
			map_data: &mut MapData, exs: &Vec<HashedMapEx>, map_sz: MapSz, logs: &mut Vec<Log>) {//, iface_settings: &mut IfaceSettings) {
		if let AttackFrontState::AssembleToLocation {target_city_coord, ..} | AttackFrontState::WallAttack {target_city_coord, ..} = &self.state {
			if self.n_empty_slots() != 0 {
				self.state = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
				return;
			}
			
			////////////
			// check if a wall is in the way, traveling from the assembly point to the city
			let path_coords = {
				let unit_ind = self.attackers[0];
				let u = &units[unit_ind];
				let ai_ind = u.owner_id;
				let mut action_iface = ActionInterfaceMeta {
					action: ActionMeta::new(ActionType::MvIgnoreWallsAndOntoPopulationCenters),
					unit_ind: Some(unit_ind),
					max_search_depth: AI_MAX_SEARCH_DEPTH,
					start_coord: Coord::frm_ind(u.return_coord(), map_sz), //*assemble_location, map_sz),
					movement_type: u.template.movement_type,
					movable_to: &movable_to
				};
				
				action_iface.update_move_search(Coord::frm_ind(*target_city_coord, map_sz), map_data, exs, MvVars::NonCivil{units, start_owner: ai_ind, blind_undiscov: None}, bldgs);
				
				if action_iface.action.path_coords.len() == 0 {
					/*panicq!("could not move to city {} -> {} (attacker: {}, current loc: {}, attackee: {})", action_iface.start_coord, Coord::frm_ind(*target_city_coord, map_sz),
							units[unit_ind].owner_id, Coord::frm_ind(units[unit_ind].return_coord(), map_sz),
							exs.last().unwrap().get(target_city_coord).unwrap().actual.owner_id.unwrap());*/
					let owner_id = units[unit_ind].owner_id;
					dbg_log(&format!("could not move to city {} -> {} (attacker: {}, current loc: {}, attackee: {})", action_iface.start_coord, Coord::frm_ind(*target_city_coord, map_sz),
							owner_id, Coord::frm_ind(units[unit_ind].return_coord(), map_sz),
							exs.last().unwrap().get(target_city_coord).unwrap().actual.owner_id.unwrap()),
						owner_id, logs, 0);
					//iface_settings.set_auto_turn(false);
				}
				action_iface.action.path_coords
			};
			
			// try to find a wall between the city and the assembly point
			let exf = exs.last().unwrap();
			for path_coord in path_coords.iter().rev() {
				if let Some(ex) = exf.get(&path_coord) {
					if let Some(structure) = &ex.actual.structure {
						if structure.structure_type == StructureType::Wall {
							self.state = AttackFrontState::WallAttack {
									target_city_coord: *target_city_coord,
									wall_coord: *path_coord,
									attacks_initiated: false
							};
							return;
						}
					}
				}
			}
			
			// no wall found, go to CityAttack
			/*printlnq!("city attack (attacker: {}, current_loc: {}, city: {} attackee: {})",
						units[self.attackers[0]].owner_id, Coord::frm_ind(units[self.attackers[0]].return_coord(), map_sz),
						Coord::frm_ind(*target_city_coord, map_sz),
						exs.last().unwrap().get(target_city_coord).unwrap().actual.owner_id.unwrap());
			*/
			self.state = AttackFrontState::CityAttack {
					target_city_coord: *target_city_coord,
					attacks_initiated: false
			};
		}else{panicq!("attempted to progress state to attack wall or city, but we are not in the correct state");}
	}
	
	fn wall_attack<'bt,'ut,'rt,'dt>(&mut self, cur_player: usize, gstate: &mut GameState, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, map_sz: MapSz) {//, iface_settings: &mut IfaceSettings) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attack_fronts.wall_attack");

		if let AttackFrontState::WallAttack {target_city_coord, wall_coord, ref mut attacks_initiated} = self.state {
			// progress state if all units are idle (attacks have already been initiated)
			if *attacks_initiated {
				// some units are still performing actions:
				if self.siegers.iter().any(|&s| units[s].action.len() != 0) || self.attackers.iter().any(|&a| units[a].action.len() != 0) {
					return;
				}
				
				self.progress_state_to_attack_wall_or_city(units, bldgs, map_data, exs, map_sz, &mut gstate.logs);//, iface_settings);
			// initiate attacks
			}else{				
				*attacks_initiated = true;
				
				if self.n_empty_slots() != 0 {
					self.state = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
					return;
				}
				
				macro_rules! abort{() => {
					self.state = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
					return;
				};};
				
				// check that city and wall owner match and we are actually at war with the owner
				let city_owner = {
					let exf = exs.last().unwrap();
					
					let city_owner = if let Some(ex) = exf.get(&target_city_coord) {
						if let Some(owner_id) = ex.actual.owner_id {
							owner_id
						}else{
							//printlnq!("could not get city owner");
							abort!();
						}
					}else{
						//printlnq!("could not get city coords");
						abort!();
					};
					
					let wall_owner = if let Some(ex) = exf.get(&wall_coord) {
						if let Some(owner_id) = ex.actual.owner_id {
							owner_id
						}else{
							//printlnq!("could not get wall owner");
							abort!();
						}
					}else{
						//printlnq!("could not get wall coords");
						abort!();
					};
					
					// not owned by same player
					if city_owner != wall_owner {
						//printlnq!("wall and city owners are different");
						abort!();
					}
					
					// not at war with
					if !gstate.relations.at_war(cur_player, city_owner as usize) {
						//printlnq!("not at war with city");
						abort!();
					}
					city_owner
				};
								
				///////////////////////////////////////////////////
				let mut action_type = ActionType::Attack {
					attack_coord: Some(wall_coord),
					attackee:  Some(city_owner),
					ignore_own_walls: false
				};
				
				macro_rules! attack_wall{($sieger: expr) => {
					if !set_target_attackable(&action_type, *$sieger, true, AI_MAX_SEARCH_DEPTH, units, bldgs, exs, map_data, map_sz) {
						//let u = &units[*$sieger];
						//printlnq!("could not set siege attack {} -> {} (attacker {}, attackee {})", Coord::frm_ind(u.return_coord(), map_sz),
						//		Coord::frm_ind(wall_coord, map_sz), u.owner_id,
						//		exs.last().unwrap().get(&target_city_coord).unwrap().actual.owner_id.unwrap());
						abort!();
					}
				}};
				
				let mut ind = 0;
				let wall_c = Coord::frm_ind(wall_coord, map_sz);
				
				const OFFSETS: &[isize] = &[-1, 0, 1];
				macro_rules! set_action_type_frm_next_wall_coord{() => {
					let exf = exs.last().unwrap();
					loop {
						if ind >= (OFFSETS.len()*OFFSETS.len()) {
							//printlnq!("couldn't find wall to attack. wall {}, city {}, attacker {} attackee {}",
							//		Coord::frm_ind(*wall_coord, map_sz), Coord::frm_ind(*target_city_coord, map_sz),
							//		units[self.siegers[0]].owner_id, exs.last().unwrap().get(target_city_coord).unwrap().actual.owner_id.unwrap());
							abort!();
						}
						let (i,j) = (OFFSETS[ind / 3], OFFSETS[ind % 3]);
						ind += 1;
						if i == 0 && j == 0 { // position 1 covers this, aka wall_coord
							continue;
						}
						if let Some(c) = map_sz.coord_wrap(wall_c.y + i, wall_c.x + j) {
							if let Some(ex) = exf.get(&c) {
								if let Some(structure) = &ex.actual.structure {
									if structure.structure_type == StructureType::Wall {
										action_type = ActionType::Attack {
											attack_coord: Some(c),
											attackee: Some(city_owner),
											ignore_own_walls: false
										};
										break;
									}
								}
							}
						}
					}
				};};
				
				// set attack position to be the same path as the sieger, 
				// but terminating in the position right before the end (right before the wall)
				macro_rules! set_attack_positions{($skip: expr) => {
					//printlnq!("siegers {}", self.siegers.len());
					let mut action = units[self.siegers[$skip]].action.last().unwrap().clone();
					debug_assertq!(action.action_meta_cont.is_none(), "attacker is trying to follower the sieger, but the path is too long");
					//printlnq!("action path_coords len {} nm: {} {} -> {}",
					//		action.path_coords.len(), units[self.siegers[$skip]].template.nm,
					//		Coord::frm_ind(units[self.siegers[$skip]].return_coord(), map_sz), wall_c);
					if action.path_coords.len() >= 2 {
						action.path_coords = action.path_coords[1..].to_vec();
					}
					for attacker in self.attackers.iter().skip($skip).take(UNITS_PER_POSITION) {
						units[*attacker].action.push(action.clone());
					}
				};};
				
				////////////////// position 1
				for sieger in self.siegers.iter().take(UNITS_PER_POSITION) {attack_wall!(sieger);}
				set_attack_positions!(0);
				
				////////////////// position 2
				set_action_type_frm_next_wall_coord!();
				for sieger in self.siegers.iter().skip(UNITS_PER_POSITION).take(UNITS_PER_POSITION) {attack_wall!(sieger);}
				set_attack_positions!(UNITS_PER_POSITION);
				
				////////////////// position 3
				set_action_type_frm_next_wall_coord!();
				for sieger in self.siegers.iter().skip(2*UNITS_PER_POSITION).take(UNITS_PER_POSITION) {attack_wall!(sieger);}
				set_attack_positions!(2*UNITS_PER_POSITION);
			}
		}else{panicq!("attempted to attack wall with incorrect state");}
	}
	
	fn city_attack<'bt,'ut,'rt,'dt>(&mut self, cur_player: usize, relations: &Relations, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, map_sz: MapSz) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attack_fronts.city_attack");

		if let AttackFrontState::CityAttack {target_city_coord, ref mut attacks_initiated} = &mut self.state {
			macro_rules! abort{() => {
				self.state = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
				return;
			};};
			
			// reset to recruitment, if all units are idle (attacks have already been initiated)
			if *attacks_initiated {
				// some units are still performing actions:
				if self.siegers.iter().any(|&s| units[s].action.len() != 0) || self.attackers.iter().any(|&a| units[a].action.len() != 0) {
					return;
				}
				
				abort!();
			// initiate attacks
			}else{
				*attacks_initiated = true;
				//printlnq!("attacking city (attacker {}, attackee {}) city: {} map_sz: {}",
				//		cur_player, exs.last().unwrap().get(target_city_coord).unwrap().actual.owner_id.unwrap(), 
				//		Coord::frm_ind(*target_city_coord, map_sz), map_sz);
				
				// check that we are actually at war with the city owner
				let city_owner = {
					let exf = exs.last().unwrap();
					
					let city_owner = if let Some(ex) = exf.get(&target_city_coord) {
						if let Some(owner_id) = ex.actual.owner_id {
							owner_id
						}else{
							//printlnq!("could not get city owner");
							abort!();
						}
					}else{
						//printlnq!("could not get city coords");
						abort!();
					};
					
					// not at war with
					if !relations.at_war(cur_player, city_owner as usize) {
						//printlnq!("not at war with city");
						abort!();
					}
					city_owner
				};
				
				let action_type = ActionType::Attack {
					attack_coord: Some(*target_city_coord),
					attackee:  Some(city_owner),
					ignore_own_walls: false
				};
				
				// debug
				/*for (attacker_ind, attacker) in self.attackers.iter().enumerate() {
					println!("\t{}: unit_ind {} actions len {} owner id {}", attacker_ind, 
							*attacker,
							units[*attacker].action.len(),
							units[*attacker].owner_id);
					
					if let Some(action) = &units[*attacker].action.last() {
						println!("\t\t{} path coords len {} action_meta_cont {}",
								action.action_type, action.path_coords.len(),
								action.action_meta_cont.is_none());
					}
				}*/
				
				// initiate attacks on city hall
				for attacker in self.attackers.iter() {
					if !set_target_attackable(&action_type, *attacker, true, AI_MAX_SEARCH_DEPTH, units, bldgs, exs, map_data, map_sz) {
						//printlnq!("could not attack city");
						abort!();
					}
				}
			}
		}else{panicq!("attempted to attack city with incorrect state");}
	}
}

#[derive(Clone, PartialEq)]
pub struct AttackFronts {
	// update ai_state.build_offensive_unit() if this changes
	pub vals: Vec::<AttackFront>
}

impl_saving!{AttackFronts {vals}}

impl AttackFronts {
	// returns next unit required for filling up the most completed attack front
	// in the recruitment state
	pub fn next_req_unit<'bt,'ut,'rt,'dt>(&mut self, n_cities: usize, pstats: &Stats<'bt,'ut,'rt,'dt>, 
			unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> Option<&'ut UnitTemplate<'rt>> {
		// find most completed attack front (that isn't fully completed) in the recruitment state
		if let Some(attack_front) = self.vals.iter()
					.filter(|front| 
							if let AttackFrontState::Recruitment {..} = &front.state {
								front.n_empty_slots() != 0 // not fully completed
							} else {false})
					.min_by_key(|front| front.n_empty_slots()) {
			
			// return an attacker template
			Some(if attack_front.attackers.len() < ATTACKERS_PER_FRONT {
				pstats.max_attack_unit(unit_templates)
			// return a sieger template
			}else{
				debug_assertq!(attack_front.siegers.len() < SIEGERS_PER_FRONT);
				pstats.max_siege_unit(unit_templates)
			})
		// create new front if we have not already exceeded the max
		}else if self.vals.len() < (MAX_ATTACK_FRONTS_PER_CITY * n_cities) {
			self.vals.push(AttackFront::default());
			
			// return attacker template
			Some(pstats.max_attack_unit(unit_templates))
		
		// no unfilled front found & no need to create new one
		}else {None}
	}
	
	// add unit to the front where it is needed -- if multiple fronts are available for it,
	// choose the one which is closest to being filled
	pub fn add_unit(&mut self, unit_ind: usize, unit_template: &UnitTemplate, pstats: &Stats,
			unit_templates: &Vec<UnitTemplate>) {
		let max_attack_unit = pstats.max_attack_unit(unit_templates);
		let max_siege_unit = pstats.max_siege_unit(unit_templates);
		debug_assertq!(max_attack_unit != max_siege_unit);
		
		enum UnitFrontType {Sieger, Attacker}
		let unit_front_type = if max_siege_unit == unit_template {UnitFrontType::Sieger} else {UnitFrontType::Attacker};
		
		struct AttackFrontSlots {
			n_slots: usize,
			attack_front_ind: usize
		}
		
		// get number of remaining slots for each front
		let mut attack_front_n_slots: Vec<AttackFrontSlots> = self.vals.iter().enumerate()
			.filter(|(_, front)| if let AttackFrontState::Recruitment {..} = &front.state {true} else {false})
			.map(|(attack_front_ind, front)| { 
				debug_assertq!(!front.siegers.contains(&unit_ind) && !front.attackers.contains(&unit_ind));
				
				AttackFrontSlots {
					n_slots: front.n_empty_slots(),
					attack_front_ind
				}
			}).collect();
				
		// sort from least to greatest (first entry being least)
		attack_front_n_slots.sort_unstable_by(|a, b| a.n_slots.partial_cmp(&b.n_slots).unwrap());
		
		// start from front with least remaining slots, and add unit, also
		// move state of front to `Transit` if it has been filled
		for af_slots in attack_front_n_slots.iter() {
			let af = &mut self.vals[af_slots.attack_front_ind];
			match unit_front_type {
				UnitFrontType::Sieger => {
					if af.siegers.len() < SIEGERS_PER_FRONT {
						af.siegers.push(unit_ind);
						return;
					}
				}
				UnitFrontType::Attacker => {
					if af.attackers.len() < ATTACKERS_PER_FRONT {
						af.attackers.push(unit_ind);
						return;
					}
				}
			}
		}
		
		// if the function hasn't returned yet, then create a new front and add the unit to it, update state
		let mut af = AttackFront::default();
		match unit_front_type {
			UnitFrontType::Sieger => {
				if af.siegers.len() < SIEGERS_PER_FRONT {
					af.siegers.push(unit_ind);
				}
			}
			UnitFrontType::Attacker => {
				if af.attackers.len() < ATTACKERS_PER_FRONT {
					af.attackers.push(unit_ind);
				}
			}
		}
		self.vals.push(af);
	}
	
	// returns true if removed
	pub fn rm_unit(&mut self, unit_ind: usize) -> bool {
		macro_rules! rm_frm_stack{($stack: ident) => {
			for (af_ind, af) in self.vals.iter_mut().enumerate() {
				if let Some(pos) = af.$stack.iter().position(|&ind| ind == unit_ind) {
					// remove unit only
					if (af.siegers.len() + af.attackers.len()) != 1 {
						af.$stack.swap_remove(pos);
					// remove entire front which is now empty
					}else{
						self.vals.swap_remove(af_ind);
					}
					return true;
				}
			}
		};};
		
		rm_frm_stack!(siegers);
		rm_frm_stack!(attackers);
		
		false
	}
	
	// returns true if changed
	pub fn chg_unit_ind(&mut self, frm_ind: usize, to_ind: usize) -> bool {
		macro_rules! chg_in_stack{($stack: ident) => {
			for af in self.vals.iter_mut() {
				if af.$stack.iter_mut().any(|ind| {
					if *ind == frm_ind {
						*ind = to_ind;
						true
					}else{
						false
					}
				}){return true;}
			}
		};};
		
		chg_in_stack!(siegers);
		chg_in_stack!(attackers);
		false
	}
	
	// if target_city_coord_opt is some value, the AI will target that city exclusively
	pub fn execute_actions<'bt,'ut,'rt,'dt>(&mut self, ai_ind: usize, target_city_coord_opt: Option<u64>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, map_data: &mut MapData<'rt>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz) {//, iface_settings: &mut IfaceSettings) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attack_fronts.execute_actions");
		
		let war_enemies = gstate.relations.at_war_with(ai_ind);
		
		// not at war, set everything to the recruitment state
		if war_enemies.len() == 0 {
			for af in self.vals.iter_mut() {
				af.state = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
			}
			return;
		}
		
		// check if target cities are still owned by a war enemy
		let exf = exs.last().unwrap();
		'chk_loop: for af in self.vals.iter_mut() {
			match &af.state {
				AttackFrontState::Recruitment {..} => {}
				
				AttackFrontState::AssembleToLocation {target_city_coord, ..} |
				AttackFrontState::WallAttack {target_city_coord, ..} |
				AttackFrontState::CityAttack {target_city_coord, ..} => {
					if let Some(ex) = exf.get(target_city_coord) {
						if let Some(bldg_ind) = ex.bldg_ind {
							if war_enemies.contains(&(bldgs[bldg_ind].owner_id as usize)) {
								continue 'chk_loop;
							}
						}
					}
					af.state = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
					
					//printlnq!("mving back to recruitment");
				}
			}
		}
		
		let n_attack_fronts = self.vals.len();
		if n_attack_fronts == 0 {return;}
		let af = &mut self.vals[gstate.rng.usize_range(0, n_attack_fronts)];
		
		//const CONT_CHANCE: f32 = 0.025;
		
		match &af.state {
			AttackFrontState::Recruitment {..} => {
				af.progress_state_to_assembly(ai_ind, units, target_city_coord_opt, bldgs, map_data, exs.last().unwrap(), gstate, map_sz);
			} AttackFrontState::AssembleToLocation {..} => {
				//if gstate.rng.gen_f32b() < CONT_CHANCE {
					//printlnq!("assemble ai {}", ai_ind);
					af.assemble_to_location(units, bldgs, map_data, exs, map_sz, &mut gstate.logs); // then progresses state
				//}
			} AttackFrontState::WallAttack {..} => {
				//if gstate.rng.gen_f32b() < CONT_CHANCE {
					af.wall_attack(ai_ind, gstate, units, bldgs, exs, map_data, map_sz); // then progresses state
				//}
			} AttackFrontState::CityAttack {..} => {
				//if gstate.rng.gen_f32b() < CONT_CHANCE {
					af.city_attack(ai_ind, &mut gstate.relations, units, bldgs, exs, map_data, map_sz); // then sets state to recruitment once finished
				//}
			}
		}
	}
}

