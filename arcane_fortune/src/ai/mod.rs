use std::cmp::{min, max, Ordering};
use crate::gcore::rand::XorState;
use crate::units::{ActionMeta, ActionType};
use crate::map::{StructureType, MapSz, ZoneType, ZoomInd, MapType};
use crate::player::Stats;
use crate::disp::{Coord, ActionInterfaceMeta, ScreenSz};
use crate::units::*;
use crate::buildings::*;
use crate::disp_lib::*;
use crate::map::MapData;
use crate::gcore::hashing::HashedMapEx;
use crate::gcore::Relations;
use crate::player::Player;
use crate::movement::{manhattan_dist, manhattan_dist_components, MvVars, movable_to};
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

pub const AI_MAX_SEARCH_DEPTH: usize = 300*4*2; // 300*4 was not enough to move to the gate

pub mod city_planning; pub use city_planning::*;
pub mod config; pub use config::*;
pub mod barbarians; pub use barbarians::*;
pub mod vars; pub use vars::*;
pub mod ai_actions; pub use ai_actions::*;
pub mod attack_fronts; pub use attack_fronts::*;
pub mod economy_planning; pub use economy_planning::*;

pub const CITY_GRID_HEIGHT: usize = 10;
pub const CITY_GRID_WIDTH: usize = 2*CITY_GRID_HEIGHT; // should always be twice the height (assumed by the construction code in new_city_plan() & create_city_grid_actions())
// ^ if the height to width relationship is changed for some reason, so should the FIEFDOM_ variables

pub const MIN_DIST_FRM_CITY_CENTER: usize = 3; // min radius of city

pub const GRID_SZ: usize = 20;

pub const CITY_HEIGHT: usize = CITY_GRID_HEIGHT * GRID_SZ;
pub const CITY_WIDTH: usize = CITY_GRID_WIDTH * GRID_SZ;

//pub const CITY_HALL_OFFSET_HEIGHT: isize = (GRID_SZ*CITY_GRID_HEIGHT/2) as isize;
//pub const CITY_HALL_OFFSET_WIDTH: isize = (GRID_SZ*CITY_GRID_WIDTH/2) as isize;

const BUFFER_AROUND_CITY: isize = 5; // additional distance outside of city walls to protect (`city_ul`, `city_lr`)

impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	// adds the city and worker actions into `ai_state.city_states`
	// `loc` is the upper left corner of the city boundary
	pub fn create_city_plan(&mut self, loc: Coord, rng: &mut XorState, map_data: &mut MapData,
			map_sz: MapSz, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>) {
		let bldg_template = BldgTemplate::frm_str(CITY_HALL_NM, bldg_templates);
		self.city_states.push(CityState::new_city_plan(loc, CITY_GRID_HEIGHT, MIN_DIST_FRM_CITY_CENTER, bldg_template, rng, map_data, map_sz));
	}
	
	// find the closest city that the unit likely is in
	pub fn find_closest_city(&mut self, loc: Coord, map_sz: MapSz) -> Option<&mut CityState<'bt,'ut,'rt,'dt>> {
		const MAX_DIST_ALLOWED: usize = CITY_WIDTH + CITY_HEIGHT;
		if let Some(min_city) = self.city_states.iter_mut().min_by_key(|c| 
				manhattan_dist(Coord::frm_ind(c.coord, map_sz), loc, map_sz)) {
			if manhattan_dist(Coord::frm_ind(min_city.coord, map_sz), loc, map_sz) < MAX_DIST_ALLOWED {
				return Some(min_city);
			}
		}
		None
	}
	
	pub fn current_war_advantage(&self, owner_id: usize, players: &Vec<Player>, relations: &Relations) -> Option<isize> {
		if let Some(&offense_power) = players[owner_id].stats.offense_power_log.last() {
			Some((offense_power as isize) - relations.at_war_with(owner_id).iter()
						.map(|&enemy| *players[enemy].stats.defense_power_log.last().unwrap() as isize)
						.sum::<isize>())
		}else{
			None
		}
	}
} // impl AIState

impl <'bt,'ut,'rt,'dt>CityState<'bt,'ut,'rt,'dt> {
	// find new city location near an existant city
	// todo: priortize options
	pub fn find_new_city_loc(&self, ai_states: &AIState, pstats: &Stats, map_data: &mut MapData, exf: &HashedMapEx, ai_config: &AIConfig, rng: &mut XorState, map_sz: MapSz) -> Option<Coord> {
		#[cfg(feature="profile")]
		let _g = Guard::new("find_new_city_loc");
		
		let valid_proposed_coord = |coord: Coord| {
			coord.y < (map_sz.h - CITY_HEIGHT - 3) as isize && coord.y >= 0 &&
			coord.x < (map_sz.w - CITY_WIDTH  - 3) as isize && coord.x >= 0
		};
		
		//////////// strategic resources
		// select location with no minimum distance between it and originating city
		// preference given to closer cities
		
		for strategic_resource in ai_config.strategic_resources.iter() {
			//if !pstats.resource_discov(strategic_resource) {continue;} // in the technological sense
			let resource_id = strategic_resource.id as usize;
			if pstats.resources_avail[resource_id] != 0 {continue;} // we already have this resource
			
			// loop over instances of `strategic_resource` that we have discovered on the map
			'resource_exemplars: for resource_coord in pstats.resources_discov_coords[resource_id].iter() {
				let mut resource_coord = Coord::frm_ind(*resource_coord, map_sz);
				if !valid_proposed_coord(resource_coord) {continue;}
				
				// check to make sure city not already founded here
				for city_state in ai_states.city_states.iter() {
					if manhattan_dist(resource_coord, Coord::frm_ind(city_state.coord, map_sz),
							map_sz) < CITY_WIDTH {
						continue 'resource_exemplars;
					}
				}
				
				// coordinate of proposed city will be near this resource
				resource_coord.y -= ((CITY_HEIGHT/2) + 10) as isize;
				resource_coord.x -= ((CITY_WIDTH/2) + 10) as isize;
				
				if valid_proposed_coord(resource_coord) {
					return Some(resource_coord);
				}
			}
		}
		
		//////////// find best nearby candidate location
		// (1) based on sampling known nearby resources
		// (2) random samples
		{
			let neighbor_c = Coord::frm_ind(self.coord, map_sz); // neighboring city the new city will be close to
			
			const N_ATTEMPTS: usize = 50;
			
			// in increments of the city dimension
			const MIN_DIST: usize = 1;
			const MAX_DIST: usize = 3;
			const MAX_RESOURCE_DIST: usize = 5*CITY_WIDTH;
			
			const ARABILITY_SZ: usize = 8;
			let arability_sz = ScreenSz {h: ARABILITY_SZ, w: ARABILITY_SZ*2, sz: 0};
			
			struct CandidateLocation {coord: u64, score: f32};
			
			let mut candidate_locations = Vec::with_capacity(N_ATTEMPTS);
			
			////////////////////////////////////////// discovered resources loop
			// (note: uses all resources even if they haven't technologically been discovered)
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("find_new_city_loc resource loop");
				
				debug_assertq!(pstats.resources_discov_coords.len() == ai_config.city_creation_resource_bonus.len());
				for (resource_coords, bonus) in pstats.resources_discov_coords.iter()
						.zip(ai_config.city_creation_resource_bonus.iter()) {
					if *bonus == 0 {continue;}
					
					// loop over exemplars for the given resource
					for resource_coord in resource_coords.iter() {
						let mut resource_coord = Coord::frm_ind(*resource_coord, map_sz);
						
						// resource is not too close or far away
						let dist = manhattan_dist_components(resource_coord, neighbor_c, map_sz);
						if dist.h > (MIN_DIST*CITY_HEIGHT) && dist.w > (MIN_DIST*CITY_HEIGHT) &&
								(dist.h + dist.w) < MAX_RESOURCE_DIST {
							resource_coord.y -= ((CITY_HEIGHT/2) + 10) as isize;
							resource_coord.x -= ((CITY_WIDTH/2) + 10) as isize;
							
							// chk if location is valid
							if let Some(map_coord) = map_sz.coord_wrap(resource_coord.y, resource_coord.x) {
								if square_clear(map_coord, ScreenSz{h: CITY_HEIGHT, w: CITY_WIDTH, sz: 0}, Quad::Lr, map_data, exf) == None ||
								   !valid_proposed_coord(resource_coord) {
									continue;
								}
								
								let score = (*bonus as f32) + arability_mean(resource_coord, arability_sz, map_data, map_sz);
								
								candidate_locations.push(CandidateLocation {coord: map_coord, score});
							}
						}
					}
				}
			}
			
			////////////////////////////////////////////// arability loop
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("find_new_city_loc arability loop");
				
				//let mut max_arability_score = 0.;
				for _ in 0..N_ATTEMPTS {
					// find offset from neighbor_c (use equation for ellipse); store offsets as `y`, `x`
					
					// elipse: y = (y_lim/x_lim) * sqrt(x_lim^2 - x^2)
					let x_lim = rng.isize_range((MIN_DIST*CITY_WIDTH) as isize, (MAX_DIST*CITY_WIDTH) as isize);
					let y_lim = rng.isize_range((MIN_DIST*CITY_HEIGHT) as isize, (MAX_DIST*CITY_HEIGHT) as isize);
					
					let x = rng.isize_range(-x_lim, x_lim);
					let y = ((y_lim as f32 / x_lim as f32) * ((x_lim*x_lim - x*x) as f32).sqrt()).round() as isize;
					
					// add offsets to get `c_new` -- the upper left boundary of the city
					let c_new = if rng.usize_range(0,2) < 1 {
						Coord {y: neighbor_c.y + y, x: neighbor_c.x + x}
					}else{
						Coord {y: neighbor_c.y - y, x: neighbor_c.x + x}
					};
					
					// check that the city does not wrap around map -- can cause problems with
					// the city planning of worker actions
					if (c_new.x < 0) || ((c_new.x as usize + CITY_WIDTH) >= map_sz.w) {continue;}
					
					// check if coord on map & square is clear
					if let Some(map_coord) = map_sz.coord_wrap(c_new.y, c_new.x) {
						if square_clear(map_coord, ScreenSz{h: CITY_HEIGHT, w: CITY_WIDTH, sz: 0}, Quad::Lr, map_data, exf) == None || 
						   !valid_proposed_coord(c_new) {
							continue;
						}
						
						let score = arability_mean(c_new, arability_sz, map_data, map_sz);
						//if score > max_arability_score {max_arability_score = score;}
						
						candidate_locations.push(CandidateLocation {coord: map_coord, score});
					}
				}
			}
			
			// sort from greatest to least
			candidate_locations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Less));
			
			if let Some(candidate_location) = candidate_locations.first() {
				//printlnq!("max arability {} {}", max_arability_score, candidate_location.score);
				Some(Coord::frm_ind(candidate_location.coord, map_sz))
			}else {None}
		}
	}
	
	#[inline]
	pub fn current_defense_pos_ind(&self, coord: u64) -> Option<usize> {
		self.defense_positions.iter().position(|&pos| pos == coord)
	}
	
	#[inline]
	pub fn next_unfilled_defense_pos_ind(&self, units: &Vec<Unit>) -> Option<usize> {
		// find first position that is unoccupied
		self.defense_positions.iter().position(|&defense_pos| {
			// determine if max defenders are here
			let mut count = 0;
			for defender in self.defenders.iter() {
				if units[defender.unit_ind].return_coord() == defense_pos {
					count += 1;
					if count == 2 {return false;}
				}
			}
			true
		})
	}
}

// returns true on success
pub fn set_target_attackable<'bt,'ut,'rt,'dt>(target: &ActionType<'bt,'ut,'rt,'dt>, attacker_ind: usize,
		clear_action_que: bool, max_search_depth: usize,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
			map_data: &mut MapData, map_sz: MapSz) -> bool {
	#[cfg(feature="profile")]
	let _g = Guard::new("set_target_attackable");
	
	if let ActionType::Attack {attack_coord: Some(coord), attackee: Some(attackee_unw), ..} = &target {
		let u = &units[attacker_ind];
		debug_assertq!(u.owner_id != *attackee_unw);
		debug_assertq!(u.action.len() == 0, "action len {} nm {} action type {}",
				u.action.len(), u.template.nm[0], u.action.last().unwrap().action_type);
		
		let coord = Coord::frm_ind(*coord, map_sz);
		
		let u = &units[attacker_ind];
		let mut action_iface = ActionInterfaceMeta {
			action: ActionMeta::new(target.clone()),
			unit_ind: Some(attacker_ind),
			max_search_depth,
			start_coord: Coord::frm_ind(u.return_coord(), map_sz),
			movement_type: u.template.movement_type,
			movable_to: &movable_to
		};
		
		action_iface.update_move_search(coord, map_data, exs, MvVars::NonCivil{units, start_owner: units[attacker_ind].owner_id, blind_undiscov: None}, bldgs);
		
		// move possible, send unit on their way
		return if action_iface.action.path_coords.len() > 0 {
			let u = &mut units[attacker_ind];
			if clear_action_que {u.action.clear();}
			u.action.push(action_iface.action);
			////////// dbg
			/*{
				let c = Coord::frm_ind(u.return_coord(), map_sz);
				let cf = Coord::frm_ind(u.action.last().unwrap().path_coords[0], map_sz);
				printlnq!("start coord {} {} path_coords.len {}  path_coords last {} {}", c.y, c.x, u.action.last().unwrap().path_coords.len(), cf.y, cf.x);
				printlnq!("actions_req {}", u.action.last().unwrap().actions_req);
			}*/
			////////
			u.set_attack_range(map_data, exs.last().unwrap(), map_sz);
			//////////////
			/*if let ActionType::Attack {attack_coord, attackee, ..} = u.action.last().unwrap().action_type {
				printlnq!("attack_coord {}, attackee {}", attack_coord.unwrap(), attackee.unwrap());
			}
			printlnq!("ret true");*/
			true
		}else {false}; // <- move not possible
	}else{
		panicq!("invalid input to is_target_attackable()");
	}
}

// `coord` specifies upper left coorner
pub fn arability_mean(coord: Coord, blank_spot: ScreenSz, map_data: &mut MapData, map_sz: MapSz) -> f32 {
	#[cfg(feature="profile")]
	let _g = Guard::new("arability_mean");
	
	let mut mean = 0.;
	
	for i_off in 0..blank_spot.h as isize {
	for j_off in 0..blank_spot.w as isize {
		let coord_chk = map_sz.coord_wrap(coord.y + i_off, coord.x + j_off).unwrap();
		let mfc = map_data.get(ZoomInd::Full, coord_chk);
		debug_assertq!(mfc.map_type == MapType::Land);
		mean += mfc.arability;
	}}
	
	mean / (blank_spot.h * blank_spot.w) as f32
}

