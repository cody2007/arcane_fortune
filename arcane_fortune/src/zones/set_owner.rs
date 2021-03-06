use crate::gcore::hashing::*;
use crate::buildings::*;
use crate::units::Unit;
use crate::map::*;
use crate::player::Player;
use crate::saving::SmSvType;
use super::{return_zone_coord, Zone};
use crate::map::utils::ZoneExFns;
use crate::ai::CityState;
use crate::containers::*;
use crate::disp::Coord;
//use crate::disp_lib::endwin;

pub fn set_owner<'bt,'ut,'rt,'dt>(coord: u64, to_owner: usize, frm_owner: usize, cur_player: usize, to_city_state_opt: &mut Option<&mut CityState>,
		units: &Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, gstate: &mut GameState, map_sz: MapSz) {
	
	if let Some(ref mut ex) = exs.last_mut().unwrap().get_mut(&coord) {
		if ex.actual.owner_id == Some(to_owner as SmSvType) {return;}
		
		// zone has bldg
		if let Some(bldg_ind_taxable) = ex.bldg_ind {
			let b_update = &mut bldgs[bldg_ind_taxable];
			
			// not already set from previous ex location (also, only set if at corner so it is only set once)
			if b_update.owner_id != (to_owner as SmSvType) && b_update.coord == coord {
				let bt = &b_update.template;
				
				// update population stats
				if let BldgType::Taxable(Zone {ztype: ZoneType::Residential, ..}) = bt.bldg_type {
					players[frm_owner].stats.population -= b_update.n_residents();
					players[to_owner].stats.population += b_update.n_residents();
					
				// update employment stats
				}else if let BldgType::Taxable(_) = bt.bldg_type {
					players[frm_owner].stats.employed -= b_update.n_residents();
					players[to_owner].stats.employed += b_update.n_residents();
				}
				
				// update tax income or expenses
				if bt.upkeep < 0. {
					players[frm_owner].stats.tax_income -= b_update.return_taxable_upkeep();
					players[to_owner].stats.tax_income += b_update.return_taxable_upkeep();
				}else{
					players[frm_owner].stats.bldg_expenses -= b_update.return_taxable_upkeep();
					players[to_owner].stats.bldg_expenses += b_update.return_taxable_upkeep();
				}
				
				//se also when altering: worker create building, building/add_bldg
				// research, crime, happiness, doctrinality, health, pacifism
				if b_update.construction_done == None {
					players[frm_owner].stats.bldg_stats(StatsAction::Rm, bt);
					players[to_owner].stats.bldg_stats(StatsAction::Add, bt);
				}
				
				// log resource
				if let Some(resource) = bldg_resource(coord, bt, map_data, map_sz) {
					players[frm_owner].stats.resources_avail[resource.id as usize] += 1;
					players[to_owner].stats.resources_avail[resource.id as usize] += 1;
				}

				b_update.owner_id = to_owner as SmSvType;
				
				// register bldg in ai state
				if let Some(to_city_state) = to_city_state_opt {
					//printlnq!("registering bldg {}; frm owner {} to {}", bldg_ind_taxable, frm_owner, to_owner);
					to_city_state.register_bldg(bldg_ind_taxable, b_update);
				}
			}
		}
		
		/////////////////////
		// merge zone ex for the zone we are moving over
		
		// zone agnostic stats:
		//	pstats counters of: happiness, doctrinality, and pacifism
		//			are updated in rm_zone(). see notes near the ZoneAgnosticStats definition
		let zone_wrapped = ex.actual.ret_zone();
		if let Some(Zone {ztype, ..}) = zone_wrapped {
			let zone_coord = return_zone_coord(coord, map_sz);
			if players[frm_owner].zone_exs.contains_key(&zone_coord) {
				let b_zone_ex = players[frm_owner].zone_exs.get(&zone_coord).unwrap().clone();
				players[to_owner].zone_exs.create_if_empty(zone_coord, temps.doctrines);
				let zone_ex = players[to_owner].zone_exs.get_mut(&zone_coord).unwrap();
				let zt = ztype as usize;
				
				// do not need to set demand_weighted_sum_map_counter because 
				// it will be set with calls to rm_zone and add_zone
				
				// merge demand_weighted_sum if not set
				if let Some(dws) = b_zone_ex.demand_weighted_sum[zt] {
					if zone_ex.demand_weighted_sum[zt].is_none() {
						zone_ex.demand_weighted_sum[zt] = Some(dws);
					}
				}
				
				// merge demand_raw if not set
				if let Some(dr) = &b_zone_ex.demand_raw[zt] {
					if zone_ex.demand_raw[zt].is_none() {
						zone_ex.demand_raw[zt] = Some(dr.clone());
					}
				}
				
				// basically, the receiver's city hall takes precidence and the sender's city hall assignments are updated
				// if the receiver already has values
				//	-if receiving owner's distance is not set, merge over from sender
				//	-if receiving owner's distance is set, set the sending owner's Dist city hall ind
				// 	 to the receiving owner's city hall
				//	 (The value stored in zone_exs_owners[frm_owner] is then cleared, but the
				// 	 use of set_city_hall_dist is needed to make sure the residents in the zone are added to the receiver's
				// 	 city hall population count)
				match zone_ex.city_hall_dist {
					Dist::NotInit | Dist::NotPossible {..} => {
						zone_ex.set_city_hall_dist(zone_coord, b_zone_ex.city_hall_dist, bldgs, map_sz);
					} Dist::Is {..} | Dist::ForceRecompute {..} => {
						let zone_ex = zone_ex.clone();
						let b_zone_ex = players[frm_owner].zone_exs.get_mut(&zone_coord).unwrap();
						b_zone_ex.set_city_hall_dist(zone_coord, zone_ex.city_hall_dist, bldgs, map_sz);
					}
				}
				
				// clear senders city hall dist
				match b_zone_ex.city_hall_dist {
					Dist::NotInit | Dist::NotPossible {..} => {}
					Dist::Is {..} | Dist::ForceRecompute {..} => {
						let b_zone_ex = players[frm_owner].zone_exs.get_mut(&zone_coord).unwrap();
						b_zone_ex.city_hall_removed();
					}
				}
			}
			ex.actual.rm_zone(coord, players, temps.doctrines, map_sz);
		}
		
		// don't do this until zone has been removed (so that stats and zone_ex counters are updated)
		ex.actual.owner_id = Some(to_owner as SmSvType);
		
		// add zone type with new owner
		if let Some(zone) = zone_wrapped {
			ex.actual.add_zone(coord, zone, &mut players[to_owner], temps.doctrines, map_sz);
		}
		
		compute_zooms_coord(coord, bldgs, temps.bldgs, map_data, exs, players);
		
		compute_active_window(coord, cur_player == to_owner, PresenceAction::SetPresentAndDiscover, map_data, exs, &mut players[to_owner].stats, map_sz, gstate, units);
		compute_active_window(coord, cur_player == frm_owner, PresenceAction::SetAbsent, map_data, exs, &mut players[frm_owner].stats, map_sz, gstate, units);
	}
}

// adj_stack: coords
// registers bldgs in to_city_state_opt, if provided
pub fn set_all_adj_owner<'bt,'ut,'rt,'dt>(mut adj_stack: Vec<u64>, to_owner: usize, frm_owner: usize, cur_player: usize, to_city_state_opt: &mut Option<&mut CityState>,
		units: &Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, gstate: &mut GameState, map_sz: MapSz) {
	if adj_stack.len() == 0 {return;}
	
	let mut adj_stack_ind = 0;
	loop {
		let coord = adj_stack[adj_stack_ind];
		let prev_coord = Coord::frm_ind(coord, map_sz);
		
		///////////// update owner
		set_owner(coord, to_owner, frm_owner, cur_player, to_city_state_opt, units, bldgs, temps, exs, players, map_data, gstate, map_sz);
		let exf = &exs.last().unwrap();
		
		//////////// add all neighbors to list to be updated
		for i_off in -1..=(1 as isize) {
		for j_off in -1..=(1 as isize) {
			if i_off == 0 && j_off == 0 {continue;} // same as prev_coord
			
			if let Some(coord) = map_sz.coord_wrap(prev_coord.y + i_off, prev_coord.x + j_off) {
				if let Some(ex) = exf.get(&coord) {
					// add to list only if not already on list, and not already set to the attacking owner
					if ex.actual.owner_id == Some(frm_owner as SmSvType) && !adj_stack.contains(&coord) {
						adj_stack.push(coord);
					}
				} 
			} // valid coord
		}} // i,j
		
		adj_stack_ind += 1;
		if adj_stack_ind >= adj_stack.len() {break;}
	} // loop through adj
}

