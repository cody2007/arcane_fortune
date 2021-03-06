use super::*;
use crate::saving::SmSvType;
use crate::gcore::*;
use crate::doctrine::DoctrineTemplate;
use crate::containers::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

const WAR_WEARINESS_FACTOR: f32 = 1./(10. * TURNS_PER_YEAR as f32);
const MAX_TIME_MASSACRE: usize = 10 * TURNS_PER_YEAR;
const MASSACRE_FACTOR: f32 = 10.;

fn nrelu(val: f32) -> f32 {
	if val < 0. {val} else {0.}
}

enum_From! {PacifismMilitarism {Pacifism, Militarism}}

#[derive(Clone, Copy, PartialEq)]
pub enum HappinessCategory {
	Doctrine,
	PacifismOrMilitarism(PacifismMilitarism),
	Health,
	Unemployment,
	Crime
}

struct NegativeSumPenalties {
	crime: f32,
	health: f32,
	unemployment: f32,
	sum: f32 // sum of them
}

// local statistics for a given zone_ex
impl ZoneAgnosticStats {
	// updates self.locally_logged
	// computation is similar to return_wealth()
	fn set_happiness(&mut self, cur_player: usize, war_time: usize, cur_doctrine: &DoctrineTemplate, gstate: &GameState) {
		let n_massacres = {
			let mut n_massacres = 0;
			for log in gstate.logs.iter().rev() {
				if (gstate.turn - log.turn) > MAX_TIME_MASSACRE {break;}
				if let LogType::RiotersAttacked {owner_id} = &log.val {
					if *owner_id == cur_player {
						n_massacres += 1;
					}
				}
			}
			n_massacres
		};
		
		let war_bonus = (war_time as f32*WAR_WEARINESS_FACTOR).min(25.);
		
		// 0:+
		let doctrine = self.locally_logged.doctrine_sum();
		
		let pacifism_signed = self.locally_logged.pacifism_sum + cur_doctrine.pacifism_bonus;
		let pacifism = pacifism_signed.abs();
		
		let pos_sum = doctrine + pacifism;
		
		// -:0
		let neg_sum = self.negative_sum_penalties(cur_doctrine);
		
		self.locally_logged.happiness_sum = 
			self.gov_bldg_happiness_sum +
			
			pos_sum + neg_sum.sum +
			
			if pacifism_signed < 0. {war_bonus} else {-war_bonus} +
			
			cur_doctrine.happiness_bonus - (n_massacres as f32)*MASSACRE_FACTOR;
		
		/*printlnq!("gov bldg {} pos {} (pacifism_signed {} war_bonus {}) neg {} (crime {} health {} unemployment {}) doc {}", self.gov_bldg_happiness_sum,
				pos_sum, pacifism_signed, war_bonus,
				neg_sum, crime, health, unemployment,
				cur_doctrine.happiness_bonus);*/
		
		let weighting = |val, sum, n_sum| {
			if sum != 0. {
				val / sum
			}else{
				1. / n_sum as f32
			}
		};
		
		self.locally_logged.contrib = ZoneAgnosticContribFracs {
			doctrine: weighting(doctrine, pos_sum, 2),
			pacifism: weighting(pacifism, pos_sum, 2),
			
			crime: weighting(neg_sum.crime, neg_sum.sum, 3),
			health: weighting(neg_sum.health, neg_sum.sum, 3),
			unemployment: weighting(neg_sum.unemployment, neg_sum.sum, 3),
			
			pos_sum,
			neg_sum: neg_sum.sum
		};
	}
	
	// similar computation to set_happiness()
	fn return_wealth(&self, cur_doctrine: &DoctrineTemplate, rng: &mut XorState) -> i32 {
		let doctrine = self.locally_logged.doctrine_sum();
		let neg_sum = self.negative_sum_penalties(cur_doctrine);
		
		let multiplier = ((rng.isize_range(-20, 20) as f32)/100.) + 1.; // 1 +/- 0.2
		//printlnq!("{}", ((doctrine + neg_sum.sum) * multiplier * 200.));
		
		((6.*doctrine + neg_sum.sum) * multiplier * 200.).round() as i32
	}
	
	fn negative_sum_penalties(&self, cur_doctrine: &DoctrineTemplate) -> NegativeSumPenalties {
		let crime = nrelu(-(self.crime_sum + cur_doctrine.crime_bonus));
		let health = nrelu(self.health_sum + cur_doctrine.health_bonus);
		let unemployment = -self.unemployment_sum;
		
		NegativeSumPenalties {
			crime, health, unemployment,
			sum: crime + health + unemployment
		}
	}
	
	// starts at map location coord, and then finds paths to the closest buildings
	// that contribute to agnostic zone stats
	fn new(coord: u64, owner_id: SmSvType, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
			bldgs: &Vec<Bldg>, doctrine: &DoctrineTemplate, temps: &Templates, gstate: &GameState, map_sz: MapSz) -> Self {
		const N_ZONE_AGNOSTIC_SAMPLES: usize = 9;
		
		let start_c = Coord::frm_ind(coord, map_sz);
		let exf = exs.last().unwrap();
		let mut zone_agnostic_stats = Self::default_init(gstate.turn, temps.doctrines);
		
		macro_rules! set_happiness {($zstats: expr) => {
			$zstats.set_happiness(owner_id as usize, gstate.relations.war_lengths(owner_id as usize, gstate.turn), doctrine, gstate);
			return $zstats;
		};}
		
		if let Some(mut action_iface) = start_civil_mv_mode(coord, map_data, exf, map_sz) {
			let mut n_conn_found = 0;
			
			macro_rules! add_bldg_stats{($b: expr) => {
				if $b.owner_id == owner_id && $b.construction_done == None && !action_iface.too_far(start_c, $b, bldgs, exf, exs, map_data, map_sz) {
					let dist = manhattan_dist(start_c, Coord::frm_ind($b.coord, map_sz), map_sz) as f32;
					
					// public event, get bonus from public event type
					if let BldgArgs::PublicEvent {public_event_type, ..} = $b.args {
						zone_agnostic_stats.gov_bldg_happiness_sum += public_event_type.happiness_bonus(&temps.bldg_config);
					// get bonuses from building template
					}else{
						let bt = &$b.template;
						zone_agnostic_stats.crime_sum += bt.crime_bonus / dist;
						zone_agnostic_stats.gov_bldg_happiness_sum += bt.happiness_bonus / dist;
						zone_agnostic_stats.locally_logged.doctrinality_sum[$b.doctrine_dedication.id] += bt.doctrinality_bonus / dist;
						zone_agnostic_stats.locally_logged.pacifism_sum += bt.pacifism_bonus / dist;
						zone_agnostic_stats.health_sum += bt.health_bonus / dist;
						
						// unemployment sum
						match bt.bldg_type {
							BldgType::Taxable(_) => {
								let frac = $b.operating_frac();
								debug_assertq!(frac >= 0. && frac <= 1.);
								zone_agnostic_stats.unemployment_sum += 1. - frac;
							}
							BldgType::Gov(_) => {}
						}
					}
					
					n_conn_found += 1;
					// found enough bldgs, return
					if n_conn_found > N_ZONE_AGNOSTIC_SAMPLES {
						set_happiness!(zone_agnostic_stats);
					}
				}
			};}
			
			///////////////////////////
			// loop over bldgs until we find enough or we run out 
			
			// first search for public events
			for b in bldgs {
				if let BldgArgs::PublicEvent {..} = b.args {
					add_bldg_stats!(b);
				}
			}
			
			// then search for gov bldgs
			for b in bldgs {
				if let BldgType::Gov(_) = b.template.bldg_type {
					add_bldg_stats!(b);
				}
			}
			
			// then search for residential bldgs
			for b in bldgs {
				if let BldgType::Taxable(_) = b.template.bldg_type {
					add_bldg_stats!(b);
				}
			}
		}
		set_happiness!(zone_agnostic_stats);
	}
}

impl ZoneAgnosticLocallyLogged {
	// negative values indicate scientific focus
	fn doctrine_sum(&self) -> f32 {
		self.doctrinality_sum.iter().sum::<f32>().abs()
	}
}

pub fn return_wealth(coord: u64, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>, bldgs: &Vec<Bldg>, player: &mut Player,
		temps: &Templates, gstate: &mut GameState, map_sz: MapSz) -> i32 {
	let doctrine = player.stats.doctrine_template;
	//let player_id = player.id;
	let zone_agnostic_stats = return_happiness(coord, map_data, exs, bldgs, player, temps, gstate, map_sz);
	
	//printlnq!("{} player: {}", zone_agnostic_stats.return_wealth(doctrine, &mut gstate.rng), player_id);
	
	zone_agnostic_stats.return_wealth(doctrine, &mut gstate.rng)
}

const WEALTH_RANGE: f32 = 2000.; // for printing & tax purposes. wealth levels can be outside of this range
pub enum WealthLevel {Low, Medium, High, N}

impl BldgArgs<'_,'_> {
	pub fn wealth_txt<'l>(&self, l: &'l Localization) -> &'l String {
		let wealth = self.wealth() as f32;
		const N_STEPS: f32 = 6.;
		const STEP: f32 = WEALTH_RANGE / N_STEPS;
		if wealth < (-WEALTH_RANGE/2.) {&l.Wealth_lowest
		}else if wealth < (-WEALTH_RANGE/2. + STEP) {&l.Wealth_lower
		}else if wealth < (-WEALTH_RANGE/2. + 2.*STEP) {&l.Wealth_low
		}else if wealth < (-WEALTH_RANGE/2. + 3.*STEP) {&l.Wealth_middle
		}else if wealth < (-WEALTH_RANGE/2. + 4.*STEP) {&l.Wealth_high
		}else if wealth < (-WEALTH_RANGE/2. + 5.*STEP) {&l.Wealth_higher
		}else{&l.Wealth_highest}
	}
	
	pub fn wealth_level(&self) -> WealthLevel {
		WealthLevel::from(self.wealth())
	}
}

impl WealthLevel {
	pub fn from(wealth: i32) -> Self {
		const N_STEPS: f32 = 3.;
		const STEP: f32 = WEALTH_RANGE / N_STEPS;
		let wealth = wealth as f32;
		if wealth < (-WEALTH_RANGE/2.) {WealthLevel::Low
		}else if wealth < (-WEALTH_RANGE/2. + STEP) {WealthLevel::Medium
		}else{WealthLevel::High}
	}
}

const N_TURNS_RECOMP_ZONE_AGNOSTIC_STATS: usize = 30*12;//*5; //30*12 * 1;//75;

// returns happiness, recomputes if needed
pub fn return_happiness<'z>(mut coord: u64, map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx>, bldgs: &Vec<Bldg>, player: &'z mut Player,
		temps: &Templates, gstate: &GameState, map_sz: MapSz) -> &'z ZoneAgnosticStats {
	
	//////////// compute zone demands on a spaced grid, unless zone doesn't match the grid
	coord = return_zone_coord(coord, map_sz);
	let zone_ex = player.zone_exs.get_mut(&coord).unwrap(); // should be created by add_zone() method in FogVars
	
	////// check if we re-compute or use old vals
	if (zone_ex.zone_agnostic_stats.turn_computed + N_TURNS_RECOMP_ZONE_AGNOSTIC_STATS) < gstate.turn || zone_ex.zone_agnostic_stats.turn_computed == 0 {
		let stats_new = ZoneAgnosticStats::new(coord, player.id, map_data, exs, bldgs, player.stats.doctrine_template, temps, gstate, map_sz);
		let stats_old = &mut zone_ex.zone_agnostic_stats;
		
		//////////////////////
		// update values stored in pstats
		player.stats.locally_logged = player.stats.locally_logged.clone() + stats_new.locally_logged.clone() - stats_old.locally_logged.clone();
		
		*stats_old = stats_new;
	}
	
	&zone_ex.zone_agnostic_stats
}

// randomly updates old happiness values
pub fn randomly_update_happiness<'bt,'ut,'rt,'dt>(map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx>, players: &mut Vec<Player>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, gstate: &mut GameState, map_sz: MapSz) {
	#[cfg(feature="profile")]
	let _g = Guard::new("randomly_update_happiness");
	
	let exf = exs.last().unwrap();
	if let Some(coord) = SampleType::ZoneAgnostic.coord_frm_turn_computed(players, exf, map_sz, gstate) {
		if let Some(ex) = exf.get(&coord) {
			if let Some(owner_id) = ex.actual.owner_id {
				let zone_agnostic_stats = return_happiness(coord, map_data, exs, bldgs, &mut players[owner_id as usize], temps, gstate, map_sz);
				
				// get max doctrinality of the zone
				let max_doc = {
					let mut max_doc_sum = zone_agnostic_stats.locally_logged.doctrinality_sum[0];
					let mut max_doc = &temps.doctrines[0];
					
					for (ds, doc) in zone_agnostic_stats.locally_logged.doctrinality_sum.iter()
							.zip(temps.doctrines.iter()).skip(1) {
						if *ds > max_doc_sum {
							max_doc_sum = *ds;
							max_doc = doc;
						}
					}
					max_doc
				};
				
				// update residential building dedications to match that doctrinality
				let zone_coord = Coord::frm_ind(return_zone_coord(coord, map_sz), map_sz);
				for b in bldgs.iter_mut().filter(|b| b.owner_id == owner_id && b.template.doctrinality_bonus != 0.) {
					match b.template.bldg_type {
						BldgType::Taxable(_) => {
							let b_coord = Coord::frm_ind(b.coord, map_sz);
							
							if b_coord.y >= zone_coord.y && b_coord.y <= (zone_coord.y + ZONE_SPACING) &&
							   b_coord.x >= zone_coord.x && b_coord.x <= (zone_coord.x + ZONE_SPACING) {
							   	  b.doctrine_dedication = max_doc; 
							}
						}
						BldgType::Gov(_) => {continue;}
					}
				}
			}
		}
	}
}

use std::ops::{Add, Sub};
impl Add for ZoneAgnosticContribFracs {
	type Output = ZoneAgnosticContribFracs;
	
	fn add(self, b: Self) -> ZoneAgnosticContribFracs {
		ZoneAgnosticContribFracs {
			doctrine: self.doctrine + b.doctrine,
			pacifism: self.pacifism + b.pacifism,
			
			health: self.health + b.health,
			unemployment: self.unemployment + b.unemployment,
			crime: self.crime + b.crime,
			
			pos_sum: self.pos_sum + b.pos_sum,
			neg_sum: self.neg_sum + b.neg_sum
		}
	}
}

/*impl Add for &ZoneAgnosticContribFracs {
	type Output = ZoneAgnosticContribFracs;
	
	fn add(self, b: &ZoneAgnosticContribFracs) -> ZoneAgnosticContribFracs {
		ZoneAgnosticContribFracs {
			doctrine: self.doctrine + b.doctrine,
			pacifism: self.pacifism + b.pacifism,
			
			health: self.health + b.health,
			unemployment: self.unemployment + b.unemployment,
			crime: self.crime + b.crime
		}
	}
}*/

impl Sub for ZoneAgnosticContribFracs {
	type Output = ZoneAgnosticContribFracs;
	
	fn sub(self, b: Self) -> ZoneAgnosticContribFracs {
		ZoneAgnosticContribFracs {
			doctrine: self.doctrine - b.doctrine,
			pacifism: self.pacifism - b.pacifism,
			
			health: self.health - b.health,
			unemployment: self.unemployment - b.unemployment,
			crime: self.crime - b.crime,
			
			pos_sum: self.pos_sum - b.pos_sum,
			neg_sum: self.neg_sum - b.neg_sum
		}
	}
}

impl Add for ZoneAgnosticLocallyLogged {
	type Output = ZoneAgnosticLocallyLogged;
	
	fn add(self, b: Self) -> ZoneAgnosticLocallyLogged {
		let doctrinality_sum = {
			debug_assert!(self.doctrinality_sum.len() == b.doctrinality_sum.len());
			let mut doctrinality_sum = Vec::with_capacity(b.doctrinality_sum.len());
			for (a,b) in self.doctrinality_sum.iter().zip(b.doctrinality_sum.iter()) {
				doctrinality_sum.push(a + b);
			}
			doctrinality_sum
		};
		
		ZoneAgnosticLocallyLogged {
			happiness_sum: self.happiness_sum + b.happiness_sum,
			doctrinality_sum,
			pacifism_sum: self.pacifism_sum + b.pacifism_sum,
			contrib: self.contrib + b.contrib
		}
	}
}

impl Sub for ZoneAgnosticLocallyLogged {
	type Output = ZoneAgnosticLocallyLogged;
	
	fn sub(self, b: Self) -> ZoneAgnosticLocallyLogged {
		let doctrinality_sum = {
			debug_assert!(self.doctrinality_sum.len() == b.doctrinality_sum.len());
			let mut doctrinality_sum = Vec::with_capacity(b.doctrinality_sum.len());
			for (a,b) in self.doctrinality_sum.iter().zip(b.doctrinality_sum.iter()) {
				doctrinality_sum.push(a - b);
			}
			doctrinality_sum
		};
		
		ZoneAgnosticLocallyLogged {
			happiness_sum: self.happiness_sum - b.happiness_sum,
			doctrinality_sum,
			pacifism_sum: self.pacifism_sum - b.pacifism_sum,
			contrib: self.contrib - b.contrib
		}
	}
}

