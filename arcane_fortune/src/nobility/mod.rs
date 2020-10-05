use std::cmp::min;
use crate::map::{AIPersonality, PersonName, MapData, Owner, Stats, MapSz};
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::units::UnitTemplate;
use crate::ai::AIState;
use crate::buildings::{Bldg, BldgTemplate, add_bldg, MANOR_NM};
use crate::gcore::{HashedMapEx, HashedMapZoneEx};
use crate::gcore::Log;
use crate::rand::XorState;
use crate::map::Nms;
use crate::saving::*;
use crate::disp_lib::endwin;
use crate::disp::{TURNS_PER_YEAR, ScreenCoord, Coord};

pub mod disp; pub use disp::*;
pub mod ai; pub use ai::*;

#[derive(Clone, PartialEq)]
pub struct Houses {
	pub houses: Vec<House>,
}

impl_saving!{Houses {houses}}

#[derive(Clone, PartialEq)]
pub struct House {
	pub name: String,
	
	pub head_noble_pair_ind: usize, // index into noble_pairs
	pub noble_pairs: Vec<NoblePair>,
	
	pub coord: Coord,
	
	pub has_req_to_join: bool // requested to join nearby empire
}

impl_saving!{House {name, head_noble_pair_ind, noble_pairs, coord, has_req_to_join}}

#[derive(Clone, PartialEq)]
pub struct Marriage {
	partner: Noble,
	children: Vec<usize> // index into House.noble_pairs
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
	noble: Noble,
	marriage: Option<Marriage>
}

impl_saving!{NoblePair {noble, marriage}}

// adds children into noble_pairs[parent_ind]
// also recursively adds children for the children
fn new_noble_pair_children(parent_ind: usize, noble_pairs: &mut Vec<NoblePair>, 
		nms: &Nms, rng: &mut XorState, turn: usize) {
	let new_ind = noble_pairs.len();
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
	name: PersonName,
	personality: AIPersonality,
	born_turn: usize,
	gender_female: bool,
}

impl_saving!{Noble {name, personality, born_turn, gender_female}}

impl Noble {
	// `age` is in turns
	pub fn new(nms: &Nms, mut age: usize, gender_female_opt: Option<bool>,
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

pub const NOBILITY_OWNER_IND: usize = 0; // index into owners among others
pub const NOBILITY_OWNER_ID: u32 = NOBILITY_OWNER_IND as u32;

impl House {
	pub fn new<'bt,'ut,'rt,'dt>(coord: Coord, stats: &mut Vec<Stats<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
			doctrine_templates: &'dt Vec<DoctrineTemplate>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			zone_exs_owners: &mut Vec<HashedMapZoneEx>, ai_state_opt: &mut Option<AIState<'bt,'ut,'rt,'dt>>, owners: &Vec<Owner>, nms: &Nms,
			logs: &mut Vec<Log>, map_sz: MapSz, rng: &mut XorState, turn: usize) -> Option<Self> {
		{ // add manor
			if let Some(coord_ind) = map_sz.coord_wrap(coord.y, coord.x) {
				let bt = &BldgTemplate::frm_str(MANOR_NM, bldg_templates);
				if !add_bldg(coord_ind, NOBILITY_OWNER_ID, bldgs, bt, None, bldg_templates, doctrine_templates, map_data, exs, zone_exs_owners, stats, ai_state_opt, owners, nms, turn, logs, rng) {
					return None;
				}
			}else{return None;}
		}
		
		let name = nms.noble_houses[rng.usize_range(0, nms.noble_houses.len())].clone();
		
		debug_assertq!(ADULTHOOD_AGE < MIN_HEAD_AGE);
		
		let matriarch_age = rng.usize_range(MIN_HEAD_AGE, MAX_HEAD_AGE);
		let patriarch_age = rng.usize_range(MIN_HEAD_AGE, MAX_HEAD_AGE);
		
		let mut noble_pairs = vec![NoblePair {
			noble: Noble::new(nms, matriarch_age, Some(true), rng, turn),
			marriage: Some(Marriage {
				partner: Noble::new(nms, patriarch_age, Some(false), rng, turn),
				children: Vec::new()
			})
		}];
		
		new_noble_pair_children(0, &mut noble_pairs, nms, rng, turn);
		
		Some(House {
			name, head_noble_pair_ind: 0, noble_pairs, coord,
			has_req_to_join: false
		})
	}
	
	pub fn end_turn(&mut self, nms: &Nms, rng: &mut XorState, turn: usize) {
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
	}
}

impl Houses {
	// adds children & new children
	// (eventually... deaths too)
	pub fn end_turn(&mut self, nms: &Nms, rng: &mut XorState, turn: usize) {
		for house in self.houses.iter_mut() {house.end_turn(nms, rng, turn);}
	}
}

//////////////////////////////////////////
// defaults

impl Default for Houses {
	fn default() -> Self {
		Self {houses: Vec::new()}
	}
}

impl Default for House {
	fn default() -> Self {
		Self {
			name: String::new(),
			
			head_noble_pair_ind: 0,
			noble_pairs: Vec::new(),
			
			coord: Default::default(),
			has_req_to_join: false
		}
	}
}

impl Default for Noble {
	fn default() -> Self {
		Self {
			name: Default::default(),
			personality: Default::default(),
			born_turn: 0,
			gender_female: false
		}
	}
}

impl Default for Marriage {
	fn default() -> Self {
		Self {
			partner: Default::default(),
			children: Vec::new()
		}
	}
}

impl Default for NoblePair {
	fn default() -> Self {
		Self {
			noble: Default::default(),
			marriage: None
		}
	}
}

