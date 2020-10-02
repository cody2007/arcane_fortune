use std::cmp::min;
use crate::map::{AIPersonality, PersonName};
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::units::UnitTemplate;
use crate::buildings::BldgTemplate;
use crate::rand::XorState;
use crate::map::Nms;
use crate::saving::*;
use crate::disp_lib::endwin;
use crate::disp::{TURNS_PER_YEAR, ScreenCoord};

pub mod disp; pub use disp::*;

#[derive(Clone, PartialEq)]
pub struct Houses {
	pub houses: Vec<House>,
}

impl_saving!{Houses {houses}}

#[derive(Clone, PartialEq)]
pub struct House {
	pub name: String,
	
	pub head_noble_pair_ind: usize, // index into noble_pairs
	pub noble_pairs: Vec<NoblePair>
}

impl_saving!{House {name, head_noble_pair_ind, noble_pairs}}

#[derive(Clone, PartialEq)]
pub struct Marriage {
	partner: Noble,
	children: Vec<usize> // index into House.noble_pairs
}

impl_saving!{Marriage {partner, children}}

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
			
			let marriage = if age >= ADULTHOOD_AGE && rng.gen_f32b() < 0.5 {
				let partner_gender_female = if rng.gen_f32b() <= SAME_SEX_PARTNER_PROB {
					gender_female
				}else{!gender_female};
				
				let partner_age = rng.usize_range(ADULTHOOD_AGE, age + MAX_PARTNER_AGE_DIFF);
				
				Some(Marriage {
					partner: Noble::new(nms, partner_age, Some(gender_female), rng, turn),
					children: Vec::new()
				})
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

impl House {
	pub fn new(nms: &Nms, rng: &mut XorState, turn: usize) -> Self {
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
		
		House {name, head_noble_pair_ind: 0, noble_pairs}
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
			noble_pairs: Vec::new()
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

