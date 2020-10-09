use std::cmp::min;
use crate::map::{MapData, MapSz};
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::units::{UnitTemplate, Quad, square_clear, WORKER_NM, EXPLORER_NM, RIOTER_NM, Unit};
use crate::ai::{CityState, AIState, GRID_SZ, Defender, AttackFronts};
use crate::buildings::{Bldg, BldgTemplate, add_bldg, MANOR_NM, BldgType};
use crate::gcore::{HashedMapEx, HashedMapZoneEx, Relations};
use crate::containers::Templates;
use crate::gcore::Log;
use crate::rand::XorState;
use crate::player::{AIPersonality, PersonName, Stats, PlayerType, Nms};
use crate::saving::*;
use crate::disp_lib::endwin;
use crate::disp::{TURNS_PER_YEAR, ScreenCoord, Coord, ScreenSz};

pub mod disp; pub use disp::*;
pub mod ai; pub use ai::*;

#[derive(Clone, PartialEq)]
pub struct House<'bt,'ut,'rt,'dt> {
	pub personality: AIPersonality,
	
	pub head_noble_pair_ind: usize, // index into noble_pairs
	pub noble_pairs: Vec<NoblePair>,
	
	pub city_state: CityState<'bt,'ut,'rt,'dt>,
	pub attack_fronts: AttackFronts,
	
	pub has_req_to_join: bool // requested to join nearby empire
}

impl_saving!{House<'bt,'ut,'rt,'dt> {personality, head_noble_pair_ind, noble_pairs, city_state,
	attack_fronts, has_req_to_join}}

#[derive(Clone, PartialEq)]
pub struct Marriage {
	pub partner: Noble,
	pub children: Vec<usize> // index into House.noble_pairs
}

impl_saving!{Marriage {partner, children}}

impl Marriage {
	pub fn new(partner: &Noble, nms: &Nms, rng: &mut XorState, turn: usize) -> Option<Self> {
		let partner_age = turn - partner.born_turn;
		if partner_age >= ADULTHOOD_AGE {
			let new_gender_female = if rng.gen_f32b() <= SAME_SEX_PARTNER_PROB {
				partner.gender_female
			}else{!partner.gender_female};
			
			let new_age = rng.usize_range(ADULTHOOD_AGE, partner_age + MAX_PARTNER_AGE_DIFF);
			
			Some(Self {
				partner: Noble::new(nms, new_age, Some(new_gender_female), rng, turn),
				children: Vec::new()
			})
		}else{None}
	}
}

#[derive(Clone, PartialEq)]
pub struct NoblePair {
	pub noble: Noble,
	pub marriage: Option<Marriage>
}

impl_saving!{NoblePair {noble, marriage}}

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

#[derive(Clone, PartialEq)]
pub struct Noble {
	pub name: PersonName,
	pub personality: AIPersonality,
	pub born_turn: usize,
	pub gender_female: bool,
}

impl_saving!{Noble {name, personality, born_turn, gender_female}}

impl Noble {
	// `age` is in turns
	pub fn new(nms: &Nms, age: usize, gender_female_opt: Option<bool>,
			rng: &mut XorState, turn: usize) -> Self {
		let (gender_female, name) = if let Some(gender_female) = gender_female_opt {
			(gender_female, PersonName::new_w_gender(gender_female, nms, rng))
		}else{PersonName::new(nms, rng)};
		
		Noble {
			name,
			personality: AIPersonality::new(rng),
			born_turn: if turn >= age {turn - age} else {GAME_START_TURN},
			gender_female
		}
	}
}

const MAX_HEAD_AGE: usize = 65 * TURNS_PER_YEAR;
const MIN_HEAD_AGE: usize = 25 * TURNS_PER_YEAR;
const ADULTHOOD_AGE: usize = 16 * TURNS_PER_YEAR; // ADULTHOOD_AGE should be < MIN_HEAD_AGE
const MAX_PARTNER_AGE_DIFF: usize = 20 * TURNS_PER_YEAR;
const SAME_SEX_PARTNER_PROB: f32 = 0.03; // https://en.wikipedia.org/w/index.php?title=Demographics_of_sexual_orientation&oldid=973439591#General_findings

const MARRIAGE_PROB_PER_TURN: f32 = 1./(TURNS_PER_YEAR as f32*10.);
const CHILD_PROB_PER_TURN: f32 = 1./(TURNS_PER_YEAR as f32*10.);

const FIEFDOM_GRID_HEIGHT: usize = 6;
const FIEFDOM_GRID_WIDTH: usize = 2*FIEFDOM_GRID_HEIGHT;

const MIN_DIST_FRM_FIEFDOM_CENTER: usize = 2; // min radius

const FIEFDOM_HEIGHT: usize = FIEFDOM_GRID_HEIGHT * GRID_SZ;
const FIEFDOM_WIDTH: usize = FIEFDOM_GRID_WIDTH * GRID_SZ;

impl <'bt,'ut,'rt,'dt>House<'bt,'ut,'rt,'dt> {
	pub fn new(coord: Coord, temps: &Templates<'bt,'ut,'rt,'dt,'_>, exf: &HashedMapEx,
			map_data: &mut MapData<'rt>, map_sz: MapSz,
			rng: &mut XorState, turn: usize) -> Option<Self> {
		
		// check if location possible
		if coord.y >= (map_sz.h - FIEFDOM_HEIGHT - 3) as isize ||
		   coord.x >= (map_sz.w - FIEFDOM_WIDTH - 3) as isize ||
		   square_clear(coord.to_ind(map_sz) as u64, ScreenSz{h: FIEFDOM_HEIGHT, w: FIEFDOM_WIDTH, sz: 0}, Quad::Lr, map_data, exf) == None {
		   	   return None;
		   }

		
		let city_state = { // worker plans
			let bt = &BldgTemplate::frm_str(MANOR_NM, temps.bldgs);
			CityState::new_city_plan(coord, FIEFDOM_GRID_HEIGHT, MIN_DIST_FRM_FIEFDOM_CENTER, bt, rng, map_data, map_sz)
		};
		
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
		
		Some(House {
			personality: AIPersonality::new(rng),
			head_noble_pair_ind: 0, noble_pairs,
			city_state,
			attack_fronts: Default::default(),
			has_req_to_join: false
		})
	}
	
	pub fn plan_actions(&mut self, ai_ind: usize, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, map_data: &mut MapData<'rt>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, relations: &Relations, map_sz: MapSz, rng: &mut XorState, logs: &mut Vec<Log>,
			nms: &Nms, turn: usize) {
		// ask to join or join empire?
		if !self.has_req_to_join {
			
		}
		
		// add children and marriages
		let mut child_ind = self.noble_pairs.len()-1;
		for noble_pair in self.noble_pairs.iter_mut() {
			// add new child to marriage?
			if let Some(marriage) = &mut noble_pair.marriage {
				if rng.gen_f32b() > CHILD_PROB_PER_TURN {continue;}
				
				child_ind += 1;
				marriage.children.push(child_ind);
				
			// add marriage?
			}else{
				if rng.gen_f32b() > MARRIAGE_PROB_PER_TURN {continue;}
				noble_pair.marriage = Marriage::new(&noble_pair.noble, nms, rng, turn);
			}
		}
		
		// add children to noble_pairs
		for _ in (self.noble_pairs.len()-1)..child_ind {
			self.noble_pairs.push(NoblePair {
				noble: Noble::new(nms, 0, None, rng, turn),
				marriage: None
			});
		}
		
		self.attack_fronts.execute_actions(ai_ind, units, bldgs, map_data, exs, relations, map_sz, rng, logs);
	}
	
	pub fn add_unit(&mut self, unit_ind: usize, unit_coord: u64, unit_template: &UnitTemplate, pstats: &Stats,
			unit_templates: &Vec<UnitTemplate>, map_sz: MapSz) {
		// check to make sure not already added
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			debug_assertq!(!self.city_state.worker_inds.contains(&unit_ind),
				"unit {} worker already added; {:?}", unit_ind, self.city_state.worker_inds);
			debug_assertq!(!self.city_state.explorer_inds.contains(&unit_ind),
				"unit {} explorer already added; {:?}", unit_ind, self.city_state.explorer_inds);
			debug_assertq!(!self.city_state.defenders.iter().any(|d| d.unit_ind == unit_ind),
				"unit {} defender already added", unit_ind);
			//debug_assertq!(!city_state.attacker_inds.contains(&unit_ind),
			//	"unit {} attacker already added to owner {}", unit_ind, owner.nm);
		}
		
		match unit_template.nm[0].as_str() {
			WORKER_NM => {self.city_state.worker_inds.push(unit_ind);}
			EXPLORER_NM => {self.city_state.explorer_inds.push(unit_ind);}
			RIOTER_NM => {}
			// attacker or defender
			_ => {
				// add as defender
				if self.city_state.defenders.len() < self.city_state.max_defenders() {
					self.city_state.defenders.push(Defender {unit_ind, failed_mv_attempts: 0});
				}else{
					self.attack_fronts.add_unit(unit_ind, unit_template, pstats, unit_templates);
				}
			}
		}
	}
	
	pub fn rm_unit(&mut self, unit_ind: usize, unit_template: &UnitTemplate) {
		match unit_template.nm[0].as_str() {
			WORKER_NM => {
				if let Some(pos) = self.city_state.worker_inds.iter().position(|&ind| ind == unit_ind) {
					self.city_state.worker_inds.swap_remove(pos);
					return;
				}
			} EXPLORER_NM => {
				if let Some(pos) = self.city_state.explorer_inds.iter().position(|&ind| ind == unit_ind) {
					self.city_state.explorer_inds.swap_remove(pos);
					return;
				}
			} RIOTER_NM => {
			// defender or attacker
			} _ => {
				// if defender, remove
				if let Some(pos) = self.city_state.defenders.iter().position(|d| d.unit_ind == unit_ind) {
					self.city_state.defenders.swap_remove(pos);
					return;
				}
				
				if self.attack_fronts.rm_unit(unit_ind) {return;}
			}
		}
	}
	
	pub fn chg_unit_ind(&mut self, frm_ind: usize, to_ind: usize, unit_template: &UnitTemplate) {
		match unit_template.nm[0].as_str() {
			WORKER_NM => {
				if self.city_state.worker_inds.iter_mut().any(|ind| {
					if *ind == frm_ind {
						*ind = to_ind;
						true
					}else {false}
				}){return;}
			} EXPLORER_NM => {
				if self.city_state.explorer_inds.iter_mut().any(|ind| {
					if *ind == frm_ind {
						*ind = to_ind;
						true
					}else {false}
				}){return;}
			} RIOTER_NM => {return;
			// defender or attacker
			} _ => {
				// if defender, change
				if self.city_state.defenders.iter_mut().any(|d| {
					if d.unit_ind == frm_ind {
						d.unit_ind = to_ind;
						true
					}else {false}
				}){return;}
				
				if self.attack_fronts.chg_unit_ind(frm_ind, to_ind) {return;}
			}
		}
		
		// debug
		{
			#[cfg(any(feature="opt_debug", debug_assertions))]
			{
				printlnq!("workers: {:?}", self.city_state.worker_inds);
				//printlnq!("defenders: ");
				//for ind in city_state.defender_inds.iter() {println!("{}", ind);}
			}
			// if all cities are destroyed, the later cleanup involves removing all units
			// and in the process some unit indices may shift around such that the code
			// will change unit indices of civs that are in the process of being destroyed
			// the only condition where a unit isn't logged should be when there are no cities remaining
			//assertq!(self.city_states.len() == 0,
			//	"could not change {} -> {} for {} in any city_state, owner: {}, n_cities: {}",
			//	frm_ind, to_ind, unit_template.nm[0], owner.nm, self.city_states.len());
		}
	}
	
	pub fn unregister_bldg(&mut self, bldg_ind: usize, bldg_template: &BldgTemplate, disband_unit_inds: &mut Vec<usize>) {
		if let BldgType::Taxable(_) = bldg_template.bldg_type {return;}
		
		match bldg_template.nm[0].as_str() {
			BOOT_CAMP_NM => {
				if self.city_state.boot_camp_ind == Some(bldg_ind) {
					self.city_state.boot_camp_ind = None;
					return;
				}
			} ACADEMY_NM => {
				if self.city_state.academy_ind == Some(bldg_ind) {
					self.city_state.academy_ind = None;
					return;
				}
			} _ => {
				for (list_ind, bonus_bldg_ind) in self.city_state.bonus_bldg_inds.iter().enumerate() {
					if *bonus_bldg_ind == bldg_ind {
						self.city_state.bonus_bldg_inds.swap_remove(list_ind);
						return;
					}
				}
			}
		}
	}
}

