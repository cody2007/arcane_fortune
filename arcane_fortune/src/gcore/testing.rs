#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::disp_lib::*;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::map::*;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::buildings::*;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::units::*;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::zones::ZONE_SPACING;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::disp::Coord;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::gcore::{approx_eq_tol, in_debt};
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::ai::{BarbarianState, AIState};
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::doctrine::DoctrineTemplate;
#[cfg(any(feature="opt_debug", debug_assertions))]
use crate::saving::SmSvType;

#[cfg(any(feature="opt_debug", debug_assertions))]
pub const TOL: f32 = 0.01;

#[cfg(any(feature="opt_debug", debug_assertions))]
pub fn chk_data(units: &Vec<Unit>, bldgs: &Vec<Bldg>, exs: &Vec<HashedMapEx>, zone_exs_owners: &Vec<HashedMapZoneEx>,
		stats: &Vec<Stats>, barbarian_states: &Vec<Option<BarbarianState>>, ai_states: &Vec<Option<AIState>>,
		doctrine_templates: &Vec<DoctrineTemplate>,
		map_data: &mut MapData, owners: &Vec<Owner>, map_sz: MapSz) {
	macro_rules! X {($test:expr, $($p:expr),*) => {
			if !$test {
				endwin();
				assert!($test, $($p),*);
			}};
			($test:expr) => {
			if !$test {
				endwin();
				assert!($test);
			}
			};};

	//endwin();
	
	// re-compute counters/sums. first initialize:
	let mut stats_new = vec!{Stats::default(doctrine_templates); owners.len()};
	let mut zone_demand_sum_map = Vec::new();
	let mut zone_demand_sum_map_n_summed = Vec::new();
	for _ in 0..owners.len() {
		zone_demand_sum_map.push(vec!{0.; ZoneType::N as usize});
		zone_demand_sum_map_n_summed.push(vec!{0; ZoneType::N as usize});
	}
	
	let exf = exs.last().unwrap();
	
	/////////////////// zone demands
	for owner_id in 0..owners.len() {
		for (coord, zone_ex) in zone_exs_owners[owner_id].iter() {
			let c = Coord::frm_ind(*coord, map_sz);
			
			debug_assertq!((c.y % ZONE_SPACING) == 0);
			debug_assertq!((c.x % ZONE_SPACING) == 0);
			
			for zt in 0..(ZoneType::N as usize) {
				if let Some(val) = zone_ex.demand_weighted_sum[zt] {
					// if it's not claimed to be on the map, don't count it
					if zone_ex.demand_weighted_sum_map_counter[zt] > 0 {
						zone_demand_sum_map[owner_id][zt] += val;
						zone_demand_sum_map_n_summed[owner_id][zt] += 1;
					}
				}
			}
		}
	}
	
	///////// todo: count zone_ex map utilization, compare to values on map
	
	
	// check that buildings stats match and resource properly registered
	for (bldg_ind, b) in bldgs.iter().enumerate() {
		// bldg should be pointed to on map
		let ex = exf.get(&b.coord).unwrap();
		X!(ex.bldg_ind == Some(bldg_ind));
		
		b.chk_connection_inds(bldgs.len()); // make sure all connections use valid bldg inds
		
		// stats:
		let pstats = &mut stats_new[b.owner_id as usize];
		
		if b.template.bldg_type == BldgType::Taxable(ZoneType::Residential) {
			pstats.population += b.n_residents();
			pstats.employed += b.n_sold();
		}
		
		if b.template.upkeep < 0. {
			pstats.tax_income += b.return_taxable_upkeep();
		}else{
			pstats.bldg_expenses += b.return_taxable_upkeep();
		}
		
		if b.construction_done == None {
			pstats.research_per_turn += b.template.research_prod;
		}
		
		// in correct zone w/ correct owner?
		X!(ex.actual.owner_id == Some(b.owner_id), "ex owner is {} but building owner is {} (bldg ind {} nm {})",
				ex.actual.owner_id.unwrap(), b.owner_id, bldg_ind, b.template.nm[0]);
		let ex_zone = ex.actual.ret_zone_type();
		match b.template.bldg_type {
			BldgType::Gov(_) => {X!(ex_zone == None);}
			BldgType::Taxable(zone) => {X!(ex_zone == Some(zone));}
		}
		
		// check that registered resource is correct
		if let BldgType::Taxable(_) = b.template.bldg_type {
			X!(bldg_resource(b.coord, b.template, map_data, map_sz) == b.resource);
		}
	}
	
	for (unit_ind, u) in units.iter().enumerate() {
		// unit should be on map
		let ex = exf.get(&u.return_coord()).unwrap();
		X!(ex.unit_inds.as_ref().unwrap().contains(&unit_ind));
		
		let owner_id = u.owner_id as usize;
		
		// stats:
		stats_new[owner_id].unit_expenses += u.template.upkeep;
		
		// contained in barbarian states
		if let Some(b_state) = &barbarian_states[owner_id] {
			debug_assertq!(owners[owner_id].player_type == PlayerType::Barbarian);
			
			// sum # times found across attackers and defenders
			let n_logged = b_state.defender_inds.iter().filter(|&&d_ind| d_ind == unit_ind).count() + 
					   b_state.attacker_inds.iter().filter(|&&a_ind| a_ind == unit_ind).count();
			
			if n_logged != 1 {
				endwin();
				println!("defenders");
				for defender in b_state.defender_inds.iter() {
					println!("{}", defender);
				}
				println!("attackers");
				for attacker in b_state.attacker_inds.iter() {
					println!("{}", attacker);
				}
				panicq!("{} owner id does not contain {} unit_ind only once. found {}", owner_id, unit_ind, n_logged);
			}
		}
		
		// contained in ai states
		if let Some(a_state) = &ai_states[owner_id] {
			//debug_assertq!(owners[owner_id].player_type.is_ai());
			match u.template.nm[0].as_str() {
				WORKER_NM => {
					// sum # times found across all cities and their worker_inds
					let mut n_logged = 0;
					for city_state in a_state.city_states.iter() {
						n_logged += city_state.worker_inds.iter().filter(|&&worker_ind| worker_ind == unit_ind).count();
					}
					
					if n_logged != 1 {
						for city_state in a_state.city_states.iter() {
							printlnq!("city workers: {}", city_state.worker_inds.len());
						}
						panicq!("{} owner id does not contain unit_ind {} only once. found {}", owner_id, unit_ind, n_logged);
					}
				}
				EXPLORER_NM => {
					// sum # times found across all cities and their worker_inds
					let mut n_logged = 0;
					for city_state in a_state.city_states.iter() {
						n_logged += city_state.explorer_inds.iter().filter(|&&explorer_ind| explorer_ind == unit_ind).count();
					}
					
					if n_logged != 1 {
						for city_state in a_state.city_states.iter() {
							printlnq!("city explorers: {}", city_state.explorer_inds.len());
						}
						panicq!("{} owner id does not contain unit_ind {} only once. found {}", owner_id, unit_ind, n_logged);
					}
				}
				_ => {
					// sum # times found across all cities (defenders)
					let mut n_logged = 0;
					for city_state in a_state.city_states.iter() {
						n_logged += city_state.defenders.iter().filter(|d| d.unit_ind == unit_ind).count();
					}
					
					// count attackers
					for af in a_state.attack_fronts.vals.iter() {
						n_logged += af.siegers.iter().filter(|&&ind| ind == unit_ind).count();
						n_logged += af.attackers.iter().filter(|&&ind| ind == unit_ind).count();
					}
					
					n_logged += a_state.icbm_inds.iter().filter(|&&ind| ind == unit_ind).count();
					
					if n_logged != 1 {
						panicq!("{} owner id does not contain unit_ind {} only once. found {}", owner_id, unit_ind, n_logged);
					}
				}
			}
		}
	}
	
	///////// check all unit inds in barbarian_states are owned by the barbarian and on the map
	for (owner_id, barbarian_state_opt) in barbarian_states.iter().enumerate() {
		if let Some(barbarian_state) = barbarian_state_opt {
			for defender in barbarian_state.defender_inds.iter() {
				if units[*defender].owner_id != owner_id as SmSvType {
					panicq!("{} barbarian owner's defenders list claims to own {} but owner is {}", owner_id, *defender, units[*defender].owner_id);
				}
			}
			
			for attacker in barbarian_state.attacker_inds.iter() {
				X!(units[*attacker].owner_id == owner_id as SmSvType);
			}
		}
	}
	
	///////////////// ai states
	// all unit inds in ai_states are owned by the ai and on the map
	// all bldgs are also owned by the ai
	// all alive AIs have at least one city, and all AIs that have no cities aren't alive
	for ((owner_id, ai_state_opt), pstats) in ai_states.iter().enumerate().zip(stats.iter()) {
		if let Some(ai_state) = ai_state_opt {
			// alive AIs must have a city
			debug_assertq!((ai_state.city_states.len() != 0 && pstats.alive) || (ai_state.city_states.len() == 0 && !pstats.alive),
					"alive AIs must have a city; owner {}, alive: {}, cities: {}", owner_id, pstats.alive, ai_state.city_states.len());
			
			for city in ai_state.city_states.iter() {
				// unit checks
				{
					for worker_ind in city.worker_inds.iter() {
						let u = &units[*worker_ind];
						
						debug_assertq!(u.template.nm[0] == WORKER_NM, "{} ai owner's worker list has a {} (unit_id {} owner {}) ", owner_id, u.template.nm[0], *worker_ind, u.owner_id);
						debug_assertq!(u.owner_id == owner_id as SmSvType, "{} ai owner's worker list claims to own {} but owner is {}", owner_id, *worker_ind, u.owner_id);
					}
					
					for explorer_ind in city.explorer_inds.iter() {
						let u = &units[*explorer_ind];
						
						debug_assertq!(u.template.nm[0] == EXPLORER_NM, "{} ai owner's explorer list has a {} (unit_id {} owner {}) ", owner_id, u.template.nm[0], *explorer_ind, u.owner_id);
						debug_assertq!(u.owner_id == owner_id as SmSvType, "{} ai owner's explorer list claims to own {} but owner is {}", owner_id, *explorer_ind, u.owner_id);
					}
					
					for defender in city.defenders.iter() {
						let u = &units[defender.unit_ind];
						debug_assertq!(u.owner_id == owner_id as SmSvType, "{} ai owners defender list claims to own {} but owner is {}", owner_id, defender.unit_ind, u.owner_id);
					}
				}
				
				// bldg checks
				{
					if let Some(ch_ind) = city.ch_ind {
						X!(bldgs[ch_ind].owner_id == owner_id as SmSvType);
					}
					if let Some(boot_camp_ind) = city.boot_camp_ind {
						X!(bldgs[boot_camp_ind].owner_id == owner_id as SmSvType, "owner {} claims to own boot camp {}, but it's owned by {}",
								owner_id, boot_camp_ind, bldgs[boot_camp_ind].owner_id);
					}
					if let Some(academy_ind) = city.academy_ind {
						X!(bldgs[academy_ind].owner_id == owner_id as SmSvType, "owner {} claims to own academy {}, but it's owned by {}",
								owner_id, academy_ind, bldgs[academy_ind].owner_id);
					}
					
					for bonus_bldg_ind in city.bonus_bldg_inds.iter() {
						X!(bldgs[*bonus_bldg_ind].owner_id == owner_id as SmSvType, "owner {} claims to own bonus bldg {}, but it's owned by {}",
								owner_id, *bonus_bldg_ind, bldgs[*bonus_bldg_ind].owner_id);
					}
				}
			}
			
			for icbm_ind in ai_state.icbm_inds.iter() {
				X!(units[*icbm_ind].owner_id == owner_id as SmSvType, "owner {} claims to own icbm {}, but it's owned by {}",
						owner_id, *icbm_ind, units[*icbm_ind].owner_id);
			}
			
			for attack_front in ai_state.attack_fronts.vals.iter() {
				for sieger in attack_front.siegers.iter() {
					X!(units[*sieger].owner_id == owner_id as SmSvType, "owner {} claims to own sieger {}, but it's owned by {}",
							owner_id, *sieger, units[*sieger].owner_id);
				}
				
				for attacker in attack_front.attackers.iter() {
					X!(units[*attacker].owner_id == owner_id as SmSvType, "owner {} claims to own attacker {}, but it's owned by {}",
							owner_id, *attacker, units[*attacker].owner_id);
				}
			}
		}
	}
	
	/////////////////
	// check that re-computed sums equal data in stats:
	for ((owner_id, s), pstats) in stats_new.iter().enumerate().zip(stats) {
		X!(s.population == pstats.population);
		X!(s.employed == pstats.employed);
		assertq!(s.research_per_turn == pstats.research_per_turn,
				"owner {}, recomputed research per turn {}, stored research per turn {}", owner_id, s.research_per_turn, pstats.research_per_turn);
		
		X!(approx_eq_tol(s.tax_income, pstats.tax_income, TOL),
			"owner {} recomputed tax income: {} old: {} ", owner_id, s.tax_income, pstats.tax_income);
		
		X!(approx_eq_tol(s.bldg_expenses, pstats.bldg_expenses, TOL),
			"recomputed bldg expenses: {} old: {}", s.bldg_expenses, pstats.bldg_expenses);
		
		for zt in 0..(ZoneType::N as usize) {
			let pstatsz = &pstats.zone_demand_sum_map[zt];
			
			//println!("recomputed n_summed zone demand: {} old: {}. {} {}. player {} zone type {}", zone_demand_sum_map_n_summed[owner_id][zt], sn.n_summed(),
			//	zone_demand_sum_map[owner_id][zt], pstatsz.demand_weighted_sum(), owner_id, zt);
			X!(zone_demand_sum_map_n_summed[owner_id][zt] == pstatsz.n_summed(), 
				"recomputed n_summed zone demand: {} old: {}. {} {}. player {} zone type {}", zone_demand_sum_map_n_summed[owner_id][zt], pstatsz.n_summed(),
				zone_demand_sum_map[owner_id][zt], pstatsz.demand_weighted_sum(), owner_id, zt);
			
			X!(approx_eq_tol(zone_demand_sum_map[owner_id][zt], pstatsz.demand_weighted_sum(), TOL),
				"recomputed zone demand: {} old: {}", zone_demand_sum_map[owner_id][zt], pstatsz.demand_weighted_sum());
		}
		
		let cond = !in_debt(pstats) || (s.alive && s.population != 0) || (!s.alive && s.population == 0);
		if !cond {
			endwin();	
			println!("owner_id {}", owner_id);
			println!("gold {}", pstats.gold);
			println!("employed {}", pstats.employed);
			println!("tax_income {}", pstats.tax_income);
			println!("unit_expenses {}", pstats.unit_expenses);
			println!("bldg_expenses {}", pstats.bldg_expenses);
			X!(cond);
		}
	}
}
