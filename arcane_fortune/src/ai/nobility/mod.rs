use super::*;
use std::cmp::min;
use crate::map::{MapData, MapSz};
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::containers::Templates;
use crate::rand::XorState;
use crate::player::*;
use crate::disp_lib::endwin;

pub mod vars; pub use vars::*;
pub mod actions; pub use actions::*;
pub mod disp; pub use disp::*;

const MAX_HEAD_AGE: usize = 65 * TURNS_PER_YEAR;
const MIN_HEAD_AGE: usize = 25 * TURNS_PER_YEAR;
const ADULTHOOD_AGE: usize = 16 * TURNS_PER_YEAR; // ADULTHOOD_AGE should be < MIN_HEAD_AGE
const MAX_PARTNER_AGE_DIFF: usize = 20 * TURNS_PER_YEAR;
const SAME_SEX_PARTNER_PROB: f32 = 0.03; // https://en.wikipedia.org/w/index.php?title=Demographics_of_sexual_orientation&oldid=973439591#General_findings

const NOBILITY_TURN_DELAY: usize = TURNS_PER_YEAR * 5;
const NEW_NOBILITY_PROB: f32 = 1./(2. * TURNS_PER_YEAR as f32);
const MAX_NOBILITY_PER_CITY: usize = 3;

const MIN_NOBILITY_CITY_DIST: usize = 2*CITY_WIDTH;
const MAX_NOBILITY_CITY_DIST: usize = 5*CITY_WIDTH;

const MIN_DIST_FRM_FIEFDOM_CENTER: usize = 2; // min radius

const FIEFDOM_GRID_HEIGHT: usize = 6;
//const FIEFDOM_GRID_WIDTH: usize = 2*FIEFDOM_GRID_HEIGHT;

//const FIEFDOM_HEIGHT: usize = FIEFDOM_GRID_HEIGHT * GRID_SZ;
//const FIEFDOM_WIDTH: usize = FIEFDOM_GRID_WIDTH * GRID_SZ;

////////////////// initialization and utility functions

impl <'bt,'ut,'rt,'dt>NobilityState<'bt,'ut,'rt,'dt> {
	pub fn new(coord: Coord, temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData<'rt>,
			exf: &HashedMapEx, map_sz: MapSz, rng: &mut XorState, turn: usize) -> Option<Self> {
		let house = House::new(temps, rng, turn);
		
		let manor = &BldgTemplate::frm_str(MANOR_NM, temps.bldgs);
		if let Some(ai_state) = AIState::new(coord, FIEFDOM_GRID_HEIGHT, MIN_DIST_FRM_FIEFDOM_CENTER, manor, temps, map_data, exf, map_sz, rng) {
			Some(Self {ai_state, house})
		}else{
			None
		}
	}
}

impl House {
	pub fn new(temps: &Templates, rng: &mut XorState, turn: usize) -> Self {
		debug_assertq!(ADULTHOOD_AGE < MIN_HEAD_AGE);
		
		let matriarch_age = rng.usize_range(MIN_HEAD_AGE, MAX_HEAD_AGE);
		let patriarch_age = rng.usize_range(MIN_HEAD_AGE, MAX_HEAD_AGE);
		
		let mut noble_pairs = vec![NoblePair {
			noble: Noble::new(&temps.nms, matriarch_age, Some(true), rng, turn),
			marriage: Some(Marriage {
				partner: Noble::new(&temps.nms, patriarch_age, Some(false), rng, turn),
				children: Vec::new()
			})
		}];
		
		new_noble_pair_children(0, &mut noble_pairs, &temps.nms, rng, turn);
		
		Self {
			head_noble_pair_ind: 0,
			noble_pairs,
			has_req_to_join: false
		}
	}
	
	pub fn head_personality(&self) -> AIPersonality {
		let head_noble_pair = &self.noble_pairs[self.head_noble_pair_ind];
		head_noble_pair.noble.personality
	}
}

// adds children into noble_pairs[parent_ind]
// also recursively adds children for the children
fn new_noble_pair_children(parent_ind: usize, noble_pairs: &mut Vec<NoblePair>, 
		nms: &Nms, rng: &mut XorState, turn: usize) {
	let parents = &mut noble_pairs[parent_ind];
	if let Some(marriage) = &parents.marriage {
		// check if the marriage can have children
		let min_parent_age = min(turn - marriage.partner.born_turn, turn - parents.noble.born_turn);
		if min_parent_age < ADULTHOOD_AGE {return;}
		if marriage.partner.gender_female == parents.noble.gender_female {return;}
		
		const MAX_CHILDREN: usize = 6;
		let n_children = rng.usize_range(2, MAX_CHILDREN);
		let mut children = Vec::with_capacity(n_children);
		
		for _ in 0..n_children {
			children.push(noble_pairs.len());
			
			let age = rng.usize_range(0, min_parent_age - ADULTHOOD_AGE);
			let gender_female = rng.usize_range(0, 2) == 0;
			let noble = Noble::new(nms, age, Some(gender_female), rng, turn);
			
			let marriage = if rng.gen_f32b() < 0.5 {
				Marriage::new(&noble, nms, rng, turn)
			}else{None};
			
			noble_pairs.push(NoblePair {noble, marriage});
			new_noble_pair_children(noble_pairs.len()-1, noble_pairs, nms, rng, turn);
		}
		
		noble_pairs[parent_ind].marriage.as_mut().unwrap().children = children;
	}
}

use crate::movement::manhattan_dist_components;
use crate::nn;

// as in noble houses, not individual nobility within a house
pub fn new_unaffiliated_nobility<'bt,'ut,'rt,'dt>(players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		logs: &mut Vec<Log>, temps: &Templates<'bt,'ut,'rt,'dt,'_>, n_log_entries: usize,
		relations: &mut Relations, map_sz: MapSz, rng: &mut XorState, turn: usize) {
	// don't add nobility yet
	if turn < (GAME_START_TURN + NOBILITY_TURN_DELAY) {return;}
	
	let mut new_house_coords = Vec::with_capacity(players.len()*2);
	let mut ai_bonus = None;
	
	// determine if/where to add noble houses (immutably access players)
	for player in players.iter() {
		match &player.ptype {
			PlayerType::Human(ai_state) |
			PlayerType::Empire(EmpireState {ai_state, ..}) => {
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
			} PlayerType::Barbarian(_) | PlayerType::Nobility(_) => {}
		}
	}
	
	// add noble houses (mutably access players)
	if new_house_coords.len() != 0 {
		let mut txt_gen = nn::TxtGenerator::new(rng.gen());
		let worker_t = &UnitTemplate::frm_str(WORKER_NM, temps.units);
		for house_coord in new_house_coords {
			if let Some(nobility_state) = NobilityState::new(house_coord, temps, map_data, exs.last().unwrap(), map_sz, rng, turn) {
				let name = temps.nms.noble_houses[rng.usize_range(0, temps.nms.noble_houses.len())].clone();
				let house = &nobility_state.house;
				let personality = house.head_personality();
				let head_noble = &house.noble_pairs[house.head_noble_pair_ind].noble;
				let ruler_nm = head_noble.name.clone();
				let gender_female = head_noble.gender_female;
				let ptype = PlayerType::Nobility(nobility_state);
				
				players.push(Player::new(players.len() as SmSvType, ptype, personality, name, ruler_nm,
					gender_female, &ai_bonus.unwrap(), NOBILITY_COLOR, &mut txt_gen, relations, n_log_entries, temps, map_data, turn, rng));
				
				for _ in 0..2 {
					add_unit(house_coord.to_ind(map_sz) as u64, false, worker_t, units, map_data, exs, bldgs, players.last_mut().unwrap(), relations,
						logs, temps, turn, rng);
				}
			}
		}
	}
}

fn n_nobility_near_coord<'bt,'ut,'rt,'dt>(coord: Coord, players: &Vec<Player<'bt,'ut,'rt,'dt>>,
		map_sz: MapSz) -> usize {
	players.iter().filter(|player| {
		if let Some(NobilityState {ai_state, ..}) = player.ptype.nobility_state() {
			if let Some(city_state) = ai_state.city_states.first() {
				return city_state.near_coord(coord, MAX_NOBILITY_CITY_DIST, map_sz);
			}
		}
		false
	}).count()
}

impl <'bt,'ut,'rt,'dt>CityState<'bt,'ut,'rt,'dt> {
	fn near_coord(&self, coord: Coord, max_dist: usize, map_sz: MapSz) -> bool {
		let dist = manhattan_dist_components(coord, Coord::frm_ind(self.coord, map_sz), map_sz);
			
		dist.h <= max_dist || dist.w <= max_dist
	}
	
	fn nearby_empire_ind(&self, max_dist: usize, players: &Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz) -> Option<usize> {
		for player in players.iter() {
			if let Some(ai_state) = player.ptype.empire_or_human_ai_state() {
				for city_state in ai_state.city_states.iter() {
					let city_coord = Coord::frm_ind(city_state.coord, map_sz);
					if self.near_coord(city_coord, max_dist, map_sz) {
						return Some(player.id as usize);
					}
				}
			}
		}
		None
	}
}

