use super::*;

#[derive(Clone, PartialEq, Default)]
pub struct NobilityState<'bt,'ut,'rt,'dt> {
	pub ai_state: AIState<'bt,'ut,'rt,'dt>,
	pub house: House
}

impl_saving!{NobilityState<'bt,'ut,'rt,'dt> {ai_state, house}}

#[derive(Clone, PartialEq)]
pub struct House {
	pub head_noble_pair_ind: usize, // index into noble_pairs
	pub noble_pairs: Vec<NoblePair>,
	
	pub has_req_to_join: bool // requested to join nearby empire
}

impl_saving!{House {head_noble_pair_ind, noble_pairs}}

#[derive(Clone, PartialEq)]
pub struct NoblePair {
	pub noble: Noble,
	pub marriage: Option<Marriage>
}

impl_saving!{NoblePair {noble, marriage}}

#[derive(Clone, PartialEq)]
pub struct Marriage {
	pub partner: Noble,
	pub children: Vec<usize> // index into House.noble_pairs
}

impl_saving!{Marriage {partner, children}}

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
	pub fn new(nms: &Nms, age: usize, gender_female_opt: Option<bool>, rng: &mut XorState, turn: usize) -> Self {
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


