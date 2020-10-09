use super::*;
use crate::saving::SmSvType;
use crate::gcore::{XorState, Relations, Log, LogType};
use crate::doctrine::DoctrineTemplate;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

const WAR_WEARINESS_FACTOR: f32 = 1./3600.;
const MAX_TIME_MASSACRE: usize = 360*10;
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

// local statistics for a given zone_ex
impl ZoneAgnosticStats {
	fn set_happiness(&mut self, cur_player: usize, war_time: usize, cur_doctrine: &DoctrineTemplate, logs: &Vec<Log>, turn: usize) {
		let n_massacres = {
			let mut n_massacres = 0;
			for log in logs.iter().rev() {
				if (turn - log.turn) > MAX_TIME_MASSACRE {break;}
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
		let doctrine = self.locally_logged.doctrinality_sum.iter().sum::<f32>().abs();
		
		let pacifism_signed = self.locally_logged.pacifism_sum + cur_doctrine.pacifism_bonus;
		let pacifism = pacifism_signed.abs();
		
		let pos_sum = doctrine + pacifism;
		
		// -:0
		let crime = nrelu(-(self.crime_sum + cur_doctrine.crime_bonus));
		let health = nrelu(self.health_sum + cur_doctrine.health_bonus);
		let unemployment = -self.unemployment_sum;
		
		let neg_sum = crime + health + unemployment;
		
		self.locally_logged.happiness_sum = 
			self.gov_bldg_happiness_sum +
			
			pos_sum + neg_sum +
			
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
			
			crime: weighting(crime, neg_sum, 3),
			health: weighting(health, neg_sum, 3),
			unemployment: weighting(unemployment, neg_sum, 3),
			
			pos_sum,
			neg_sum
		};
	}
	
	// starts at map location coord, and then finds paths to the closest buildings
	// that contribute to agnostic zone stats
	fn new(coord: u64, owner_id: SmSvType, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
			bldgs: &Vec<Bldg>, doctrine: &DoctrineTemplate, doctrine_templates: &Vec<DoctrineTemplate>,
			relations: &Relations, logs: &Vec<Log>, map_sz: MapSz, turn: usize) -> Self {
		const N_ZONE_AGNOSTIC_SAMPLES: usize = 9;
		
		let start_c = Coord::frm_ind(coord, map_sz);
		let exf = exs.last().unwrap();
		let mut zone_agnostic_stats = Self::default_init(turn, doctrine_templates);
		
		macro_rules! set_happiness {($zstats: expr) => {
			$zstats.set_happiness(owner_id as usize, relations.war_lengths(owner_id as usize, turn), doctrine, logs, turn);
			return $zstats;
		};};
		
		if let Some(mut action_iface) = start_civil_mv_mode(coord, map_data, exf, map_sz) {
			let mut n_conn_found = 0;
			
			macro_rules! add_bldg_stats{($b: expr) => {
				if $b.owner_id == owner_id && $b.construction_done == None && !action_iface.too_far(start_c, $b, bldgs, exf, exs, map_data, map_sz) {
					let dist = manhattan_dist(start_c, Coord::frm_ind($b.coord, map_sz), map_sz) as f32;
					
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
					
					n_conn_found += 1;
					if n_conn_found > N_ZONE_AGNOSTIC_SAMPLES {
						set_happiness!(zone_agnostic_stats);
					}
				}
			};};
			
			///////////////////////////
			// loop over bldgs until we find enough or we run out (first search for gov bldgs)
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

const N_TURNS_RECOMP_ZONE_AGNOSTIC_STATS: usize = 30*12;//*5; //30*12 * 1;//75;

// returns happiness, recomputes if needed
pub fn return_happiness<'z>(mut coord: u64, map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx>, bldgs: &Vec<Bldg>, player: &'z mut Player,
		doctrine_templates: &Vec<DoctrineTemplate>, relations: &Relations, logs: &Vec<Log>,
		map_sz: MapSz, turn: usize) -> &'z ZoneAgnosticStats {
	
	//////////// compute zone demands on a spaced grid, unless zone doesn't match the grid
	coord = return_zone_coord(coord, map_sz);
	let zone_ex = player.zone_exs.get_mut(&coord).unwrap(); // should be created by add_zone() method in FogVars
	
	////// check if we re-compute or use old vals
	if (zone_ex.zone_agnostic_stats.turn_computed + N_TURNS_RECOMP_ZONE_AGNOSTIC_STATS) < turn || zone_ex.zone_agnostic_stats.turn_computed == 0 {
		let stats_new = ZoneAgnosticStats::new(coord, player.id, map_data, exs, bldgs, player.stats.doctrine_template, doctrine_templates, relations, logs, map_sz, turn);
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
		exs: &mut Vec<HashedMapEx>, players: &mut Vec<Player>,
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		doctrine_templates: &'dt Vec<DoctrineTemplate>, relations: &Relations,
		logs: &Vec<Log>, map_sz: MapSz, turn: usize, rng: &mut XorState) {
	#[cfg(feature="profile")]
	let _g = Guard::new("randomly_update_happiness");
	
	let exf = exs.last().unwrap();
	if let Some(coord) = SampleType::ZoneAgnostic.coord_frm_turn_computed(players, exf, map_sz, turn, rng) {
		if let Some(ex) = exf.get(&coord) {
			if let Some(owner_id) = ex.actual.owner_id {
				let zone_agnostic_stats = return_happiness(coord, map_data, exs, bldgs, &mut players[owner_id as usize], doctrine_templates, relations, logs, map_sz, turn);
				
				// get max doctrinality of the zone
				let max_doc = {
					let mut max_doc_sum = zone_agnostic_stats.locally_logged.doctrinality_sum[0];
					let mut max_doc = &doctrine_templates[0];
					
					for (ds, doc) in zone_agnostic_stats.locally_logged.doctrinality_sum.iter()
							.zip(doctrine_templates.iter()).skip(1) {
						if *ds > max_doc_sum {
							max_doc_sum = *ds;
							max_doc = doc;
						}
					}
					max_doc
				};
				
				// update residential building dedications to match that doctrinality
				let zone_coord = Coord::frm_ind(return_zone_coord(coord, map_sz), map_sz);
				for b in bldgs.iter_mut() {
					if b.owner_id != owner_id || b.template.doctrinality_bonus == 0. {continue;}
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

