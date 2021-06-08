use crate::buildings::BldgTemplate;
use crate::player::*;
use super::*;

pub struct BonusBldg<'bt,'ut,'rt,'dt> {
	pub scientific_bldgs: Vec<&'bt BldgTemplate<'ut,'rt,'dt>>,
	pub doctrine_bldgs: Vec<&'bt BldgTemplate<'ut,'rt,'dt>>,
	other_bldgs: Vec<&'bt BldgTemplate<'ut,'rt,'dt>>
}

impl <'bt,'ut,'rt,'dt>BonusBldg<'bt,'ut,'rt,'dt> {
	pub fn new(bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, pstats: &Stats) -> Self {
		let mut bonus_bldgs = BonusBldg {
			scientific_bldgs: Vec::with_capacity(bldg_templates.len()),
			doctrine_bldgs: Vec::with_capacity(bldg_templates.len()),
			other_bldgs: Vec::with_capacity(bldg_templates.len())
		};
		
		for bt in bldg_templates.iter().filter(|bt| 
				bt.nm[0] == CITY_HALL_NM || bt.nm[0] == MANOR_NM || !bt.available(pstats)) {
			
			if let BldgType::Gov(_) = &bt.bldg_type {
				let mut added = false;
				if bt.research_prod != 0 {
					//printlnq!("science {}", bt.nm[0]);
					bonus_bldgs.scientific_bldgs.push(bt);
					added = true;
				}
				
				if bt.doctrinality_bonus > 0. {
					//printlnq!("doctrine {}", bt.nm[0]);
					bonus_bldgs.doctrine_bldgs.push(bt);
					added = true;
				}
				
				if !added {
					bonus_bldgs.other_bldgs.push(bt);
				}
			}
		}
		bonus_bldgs
	}
}

impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	// W = offensive_prob = (1-friendliness)/2
	// D = development_prob = (spirituality + 1).abs()
	// ^ each term is [0:1]
	
	// W / (W+D) proportion of non-city income spent on offensive units
	// D / (W+D) proportion of non-city income spent on building improvements
	
	// returns true when we have a strong enough offensive (and should therefore
	// use excess money to produce improvement bldgs instead)
	pub fn is_offense_strong_enough(&self, pstats: &Stats, personality: &AIPersonality, units: &Vec<Unit>, bldgs: &Vec<Bldg>) -> bool {
		let development_expenses = {
			let mut development_expenses = 0.;
			
			for city in self.city_states.iter() {
				for bonus_bldg_ind in city.bonus_bldg_inds.iter() {
					//printlnq!("bonus_bldg_ind {} bldgs len {}", *bonus_bldg_ind, bldgs.len());
					//printlnq!("owner id {}", bldgs[*bonus_bldg_ind].owner_id);
					let expense = bldgs[*bonus_bldg_ind].template.upkeep;
					debug_assertq!(expense >= 0.);
					development_expenses += expense;
				}
			}
			development_expenses
		};
		
		let offensive_expenses = {
			let mut offensive_expenses = 0.;
			
			for icbm_ind in self.icbm_inds.iter() {
				let expense = units[*icbm_ind].template.upkeep;
				debug_assertq!(expense >= 0.);
				offensive_expenses += expense;
			}
			
			for attack_front in self.attack_fronts.vals.iter() {
				for seiger in attack_front.siegers.iter() {
					let expense = units[*seiger].template.upkeep;
					debug_assertq!(expense >= 0.);
					offensive_expenses += expense;
				}
				
				for attacker in attack_front.attackers.iter() {
					let expense = units[*attacker].template.upkeep;
					debug_assertq!(expense >= 0.);
					offensive_expenses += expense;
				}
			}
			offensive_expenses
		};
		
		debug_assertq!(pstats.bldg_expenses >= development_expenses);
		debug_assertq!(pstats.unit_expenses >= offensive_expenses);
		
		debug_assertq!(personality.friendliness >= -1. && personality.friendliness <= 1.);
		debug_assertq!(personality.spirituality >= -1. && personality.spirituality <= 1.);
		
		let offensive_prob = (1.-personality.friendliness)/2.;
		let development_prob = personality.spirituality.abs() * 2.; // <-- not actually a probability range: 0:2
		
		if development_expenses == 0. && offensive_expenses == 0. {
			return offensive_prob > development_prob;
		}
		
		let goal_offensive_proportion = offensive_prob / (offensive_prob + development_prob);
		let actual_offensive_proportion = offensive_expenses / (offensive_expenses + development_expenses);
		
		goal_offensive_proportion < actual_offensive_proportion
	}
	
	pub fn set_next_bonus_bldg(&mut self, pstats: &Stats, personality: &AIPersonality,
			bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, rng: &mut XorState) {
		let bonus_bldgs = BonusBldg::new(bldg_templates, pstats);
		
		let doctrine_bldg_prob = (1. + personality.spirituality)/2.;
		let scientific_bldg_prob = (1. - personality.spirituality)/2.;
		// ^ between the ranges of 0 and 2 (basically amplify any inclinations the AI's personality has)
		
		//printlnq!("{} {}", doctrine_bldg_prob, scientific_bldg_prob);
		
		macro_rules! choose_rand_bldg{($list: ident, $remain1: ident, $remain2: ident) => {
			if bonus_bldgs.$list.len() != 0 {
				let ind = rng.usize_range(0, bonus_bldgs.$list.len());
				self.next_bonus_bldg = Some(bonus_bldgs.$list[ind]);
			}else{
				let ind = rng.usize_range(0,
						bonus_bldgs.$remain1.len() + bonus_bldgs.$remain2.len());
				self.next_bonus_bldg = Some(if ind < bonus_bldgs.$remain1.len() {
					bonus_bldgs.$remain1[ind]
				}else{
					bonus_bldgs.$remain2[ind - bonus_bldgs.$remain1.len()]
				});
			}
			return;
		};}
		
		//if personality.spirituality > 0. {
			if rng.gen_f32b() < doctrine_bldg_prob {
				choose_rand_bldg!(doctrine_bldgs, scientific_bldgs, other_bldgs);
			}else if rng.gen_f32b() < scientific_bldg_prob {
				choose_rand_bldg!(scientific_bldgs, doctrine_bldgs, other_bldgs);
			}
		/*}else{
			if rng.gen_f32b() < scientific_bldg_prob {
				choose_rand_bldg!(scientific_bldgs, doctrine_bldgs, other_bldgs);
			}else if rng.gen_f32b() < doctrine_bldg_prob {
				choose_rand_bldg!(doctrine_bldgs, scientific_bldgs, other_bldgs);
			}
		}*/
		
		let ind = rng.usize_range(0, bldg_templates.len());
		self.next_bonus_bldg = Some(&bldg_templates[ind]);
	}
}

