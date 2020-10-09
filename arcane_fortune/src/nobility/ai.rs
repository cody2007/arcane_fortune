use super::*;
use crate::disp::NOBILITY_COLOR;
use crate::movement::manhattan_dist_components;
use crate::map::{MapSz};
use crate::gcore::Relations;
use crate::player::{Stats, Player};
use crate::ai::{CITY_WIDTH, AIState};
use crate::nn;

const NOBILITY_TURN_DELAY: usize = TURNS_PER_YEAR * 5;
const NEW_NOBILITY_PROB: f32 = 1./(2. * TURNS_PER_YEAR as f32);
const MAX_NOBILITY_PER_CITY: usize = 3;

const MIN_NOBILITY_CITY_DIST: usize = 2*CITY_WIDTH;
const MAX_NOBILITY_CITY_DIST: usize = 5*CITY_WIDTH;

pub fn new_unaffiliated_houses<'bt,'ut,'rt,'dt>(players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, exf: &HashedMapEx, map_data: &mut MapData<'rt>,
		relations: &mut Relations, map_sz: MapSz, rng: &mut XorState, turn: usize) {
	// don't add nobility yet
	if turn < (GAME_START_TURN + NOBILITY_TURN_DELAY) {return;}
	
	let mut new_house_coords = Vec::with_capacity(players.len()*2);
	let mut ai_bonus = None;
	
	// determine if/where to add noble houses (immutably access players)
	for player in players.iter() {
		if let Some(ai_state) = player.ptype.ai_state() {
			for city_state in ai_state.city_states.iter() {
				// probabilistically add nobility
				if rng.gen_f32b() > NEW_NOBILITY_PROB {continue;}
				
				let city_coord = Coord::frm_ind(city_state.coord, map_sz);
				
				// limit number of nobility per city
				if n_nobility_near_coord(city_coord, players, map_sz) >= MAX_NOBILITY_PER_CITY {continue;}
				
				let mut offset = || {
					rng.isize_range(MIN_NOBILITY_CITY_DIST as isize, MAX_NOBILITY_CITY_DIST as isize)
				};
				
				new_house_coords.push(Coord {
					y: city_coord.y + offset(),
					x: city_coord.x + offset()
				});
				
				if ai_bonus.is_none() {
					ai_bonus = Some(player.stats.bonuses.clone());
				}
			}
		}
	}
	
	// add noble houses (mutably access players)
	if new_house_coords.len() != 0 {
		let mut txt_gen = nn::TxtGenerator::new(rng.gen());
		for house_coord in new_house_coords {
			if let Some(house) = House::new(house_coord, temps, exf, map_data, map_sz, rng, turn) {
				let name = temps.nms.noble_houses[rng.usize_range(0, temps.nms.noble_houses.len())].clone();
				let personality = house.personality.clone();
				let head_noble = &house.noble_pairs[house.head_noble_pair_ind].noble;
				let ruler_nm = head_noble.name.clone();
				let gender_female = head_noble.gender_female;
				let ptype = PlayerType::Nobility {house};
				
				players.push(Player::new(players.len() as SmSvType, ptype, personality, name, ruler_nm,
					gender_female, &ai_bonus.unwrap(), NOBILITY_COLOR, &mut txt_gen, relations,
					&temps.nms, temps.techs, temps.resources, temps.doctrines, map_data, rng));
			}
		}
	}
}

pub fn nobility_near_coord(coord: Coord, house_coord: Coord, map_sz: MapSz) -> bool {
	let dist = manhattan_dist_components(coord, house_coord, map_sz);
		
	dist.h <= MAX_NOBILITY_CITY_DIST || dist.w <= MAX_NOBILITY_CITY_DIST
}

pub fn n_nobility_near_coord(coord: Coord, players: &Vec<Player>,	map_sz: MapSz) -> usize {
	players.iter().filter(|player| 
			if let Some(house) = player.ptype.house() {
				nobility_near_coord(coord, Coord::frm_ind(house.city_state.coord, map_sz), map_sz)
			}else {false}
		).count()
}

impl House<'_,'_,'_,'_> {
	pub fn nearby_empire_ind(&self, players: &Vec<Player>, map_sz: MapSz) -> Option<usize> {
		for player in players.iter() {
			if let Some(ai_state) = player.ptype.ai_state() {
				for city_state in ai_state.city_states.iter() {
					let city_coord = Coord::frm_ind(city_state.coord, map_sz);
					if nobility_near_coord(city_coord, Coord::frm_ind(self.city_state.coord, map_sz), map_sz) {
						return Some(player.id as usize);
					}
				}
			}
		}
		None
	}
}

