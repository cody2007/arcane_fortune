use crate::disp::Coord;
use crate::map::MapSz;
use crate::gcore::rand::XorState;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::zones::ZONE_SPACING;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

pub enum SampleType {ZoneDemand, ZoneAgnostic}

impl SampleType {
	// prioritizes uncomputed coords & then ones which were computed long ago
	// output is used for recomputing zone demands and happiness
	pub fn coord_frm_turn_computed(&self, zone_exs_owners: &mut Vec<HashedMapZoneEx>,
			exf: &HashedMapEx, map_sz: MapSz, turn: usize,
			rng: &mut XorState) -> Option<u64> {
		let n_zone_coords = {
			let mut n_zone_coords = 0;
			for zone_exs in zone_exs_owners.iter() {
				n_zone_coords += zone_exs.len();
			}
			n_zone_coords
		};
		
		struct ZoneComputeTurn {t_computed_elapsed: usize, coord: u64};
		
		macro_rules! return_if_zoned{($coord: expr) => {
			if let Some(ex) = exf.get(&$coord) {
				if let Some(_zone_type) = ex.actual.ret_zone_type() {
					return Some($coord);
				}
			}
		};};
		
		// gather zoned coords
		let (zone_coords, n_not_computed, turn_sum) = {
			#[cfg(feature="profile")]
			let _g = Guard::new("sample_zone_coord_frm_turn_computed coords");
			
			let mut zone_coords = Vec::with_capacity(n_zone_coords);
			let mut n_not_computed = 0;
			let mut turn_sum = 0;
			for zone_exs in zone_exs_owners.iter() {
				for (coord, zone_ex) in zone_exs.iter() {
					let turn_computed = match self {
						SampleType::ZoneDemand => {
							let mut turn_computed = 0;
							for demand_raw_opt in zone_ex.demand_raw.iter() {
								if let Some(demand_raw) = demand_raw_opt {
									if demand_raw.turn_computed > turn_computed {
										turn_computed = demand_raw.turn_computed;
									}
								}
							}
							turn_computed
						} SampleType::ZoneAgnostic => {
							zone_ex.zone_agnostic_stats.turn_computed
						}
					};
					
					if turn_computed == 0 {n_not_computed += 1;}
					
					let t_computed_elapsed = turn - turn_computed;
					turn_sum += t_computed_elapsed;
					
					zone_coords.push(ZoneComputeTurn {t_computed_elapsed, coord: *coord});
				}
			}
			(zone_coords, n_not_computed, turn_sum)
		};
		
		// nothing computed -- equal weighting
		if n_not_computed == n_zone_coords {
			const N_TRIES: usize = 10;
			for _ in 0..N_TRIES {
				let (coord, _ex) = exf.iter().nth(rng.usize_range(0, exf.len())).unwrap();
				return_if_zoned!(*coord);
			}
		// some zone demands have been computed
		}else{
			#[cfg(feature="profile")]
			let _g = Guard::new("sample_zone_coord_frm_turn_computed select coord");
			
			let prob_val = rng.gen_f32b();
			let turn_sum = turn_sum as f32;
			let mut val_sum = 0.;
			
			for zone_coord in zone_coords.iter() {
				val_sum += zone_coord.t_computed_elapsed as f32 / turn_sum;
				if prob_val < val_sum {
					const N_TRIES: usize = 5;
					for _ in 0..N_TRIES {
						let mut coord = Coord::frm_ind(zone_coord.coord, map_sz);
						coord.y += rng.usize_range(0, ZONE_SPACING as usize) as isize;
						coord.x += rng.usize_range(0, ZONE_SPACING as usize) as isize;
						if let Some(coord) = map_sz.coord_wrap(coord.y, coord.x) {
							return_if_zoned!(coord);
						}
					}
				}
			}
		}
		
		//panicq!("could not find zone coord");
		return None;
	}
	
	// returns random zoned coordinate regardless of how long ago it was computed
	/*pub fn coord(&self, exf: &HashedMapEx, rng: &mut XorState) -> Option<u64> {
		if exf.len() == 0 {return None;}
		
		const N_TRIES: usize = 10;
		for _ in 0..N_TRIES {
			let (coord, ex) = exf.iter().nth(rng.usize_range(0, exf.len())).unwrap();
			if let Some(_zone_type) = ex.actual.ret_zone_type() {
				return Some(*coord);
			}
		}
		
		//panicq!("could not find zone coord");
		return None;
	}*/
}

// prioritizes coords with the lowest happiness values
// output is used for creating rioters and is checked to contain no buildings
pub fn sample_low_happiness_coords(zone_exs: &HashedMapZoneEx,
		exf: &HashedMapEx, map_sz: MapSz, rng: &mut XorState) -> Option<u64> {
	if zone_exs.len() == 0 {return None;}
	struct ZoneHappiness {unhappiness: f32, coord: u64};
	
	// gather zoned coords
	let (zone_coords, unhappiness_sum, min_unhappiness) = {
		let mut zone_coords = Vec::with_capacity(zone_exs.len());
		let mut unhappiness_sum = 0.;
		let mut min_unhappiness_opt = None; // added to unhappiness values to ensure min value >= 0
		for (coord, zone_ex) in zone_exs.iter() {
			let unhappiness = -zone_ex.zone_agnostic_stats.locally_logged.happiness_sum;
			unhappiness_sum += unhappiness;
			
			// update min unhappiness
			if let Some(min_unhappiness) = min_unhappiness_opt {
				if min_unhappiness > unhappiness {
					min_unhappiness_opt = Some(unhappiness);
				}
			}else {min_unhappiness_opt = Some(unhappiness);}
			
			zone_coords.push(ZoneHappiness {unhappiness, coord: *coord});
		}
		let min_unhappiness = min_unhappiness_opt.unwrap();
		(zone_coords, unhappiness_sum + min_unhappiness, min_unhappiness)
	};
	
	if unhappiness_sum == 0. {return None;}
	
	let prob_val = rng.gen_f32b();
	let mut val_sum = 0.;
	
	for zone_coord in zone_coords.iter() {
		val_sum += (zone_coord.unhappiness + min_unhappiness) / unhappiness_sum;
		if prob_val < val_sum {
			const N_TRIES: usize = 5;
			for _ in 0..N_TRIES {
				let mut coord = Coord::frm_ind(zone_coord.coord, map_sz);
				coord.y += rng.usize_range(0, ZONE_SPACING as usize) as isize;
				coord.x += rng.usize_range(0, ZONE_SPACING as usize) as isize;
				if let Some(coord) = map_sz.coord_wrap(coord.y, coord.x) {
					// return if no building found
					if let Some(ex) = exf.get(&coord) {
						if ex.bldg_ind.is_none() {
							return Some(coord);
						}
					}else{
						return Some(coord);
					}
				}
			}
		}
	}
	
	return None;
}
