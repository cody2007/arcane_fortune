use super::*;
use crate::movement::manhattan_dist_components;
use crate::map::{MapSz, Stats};
use crate::ai::{CITY_WIDTH, AIState};

const NOBILITY_TURN_DELAY: usize = TURNS_PER_YEAR * 5;
const NEW_NOBILITY_PROB: f32 = 1./(2. * TURNS_PER_YEAR as f32);
const MAX_NOBILITY_PER_CITY: usize = 3;

const MIN_NOBILITY_CITY_DIST: usize = 2*CITY_WIDTH;
const MAX_NOBILITY_CITY_DIST: usize = 5*CITY_WIDTH;

pub fn new_unaffiliated_houses<'bt,'ut,'rt,'dt>(unaffiliated_houses: &mut Vec<House>, stats: &mut Vec<Stats<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, doctrine_templates: &'dt Vec<DoctrineTemplate>, map_data: &mut MapData<'rt>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, zone_exs_owners: &mut Vec<HashedMapZoneEx>,
		owners: &Vec<Owner>, nms: &Nms, logs: &mut Vec<Log>,
		ai_states: &mut Vec<Option<AIState<'bt,'ut,'rt,'dt>>>, map_sz: MapSz, rng: &mut XorState, turn: usize) {
	// don't add nobility yet
	if turn < (GAME_START_TURN + NOBILITY_TURN_DELAY) {return;}
	
	for owner_ind in 0..owners.len() {
		if let Some(ai_state) = &ai_states[owner_ind] {
			let n_cities = ai_state.city_states.len();
			for city_ind in 0..n_cities {
				let city_state = &ai_states[owner_ind].as_ref().unwrap().city_states[city_ind];
				
				// probabilistically add nobility
				if rng.gen_f32b() > NEW_NOBILITY_PROB {continue;}
				
				let city_coord = Coord::frm_ind(city_state.coord, map_sz);
				
				// limit number of nobility per city
				if n_nobility_near_coord(city_coord, unaffiliated_houses, stats, map_sz) >= MAX_NOBILITY_PER_CITY {continue;}
				
				let mut offset = || {
					rng.isize_range(MIN_NOBILITY_CITY_DIST as isize, MAX_NOBILITY_CITY_DIST as isize)
				};
				
				let house_coord = Coord {
					y: city_coord.y + offset(),
					x: city_coord.x + offset()
				};
				
				let ai_state_opt = &mut ai_states[NOBILITY_OWNER_IND];
				if let Some(house) = House::new(house_coord, stats, bldgs, bldg_templates, doctrine_templates, map_data, exs, zone_exs_owners, ai_state_opt, owners, nms, logs, map_sz, rng, turn) {
					unaffiliated_houses.push(house);
				}
			}
		}
	}
}

pub fn nobility_near_coord(coord: Coord, house_coord: Coord, map_sz: MapSz) -> bool {
	let dist = manhattan_dist_components(coord, house_coord, map_sz);
		
	dist.h <= MAX_NOBILITY_CITY_DIST || dist.w <= MAX_NOBILITY_CITY_DIST
}

pub fn n_nobility_near_coord(coord: Coord, unaffiliated_houses: &Vec<House>, stats: &Vec<Stats>,
		map_sz: MapSz) -> usize {
	let mut n = 0;
	
	let mut inc_counter = |house_coord| {
		if nobility_near_coord(coord, house_coord, map_sz) {
			n += 1;
		}
	};
	
	for house in unaffiliated_houses.iter() {
		inc_counter(house.coord);
	}
	
	for pstats in stats.iter() {
		for house in pstats.houses.houses.iter() {
			inc_counter(house.coord);
		}
	}
	
	n
}

impl House {
	pub fn nearby_empire_ind(&self, ai_states: &Vec<Option<AIState>>, map_sz: MapSz) -> Option<usize> {
		for (owner_ind, ai_state) in ai_states.iter().enumerate() {
			if let Some(ai_state) = ai_state {
				for city_state in ai_state.city_states.iter() {
					let city_coord = Coord::frm_ind(city_state.coord, map_sz);
					if nobility_near_coord(city_coord, self.coord, map_sz) {
						return Some(owner_ind);
					}
				}
			}
		}
		None
	}
}

