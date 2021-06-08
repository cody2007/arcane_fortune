use super::*;
use crate::gcore::{return_effective_tax_rate, MAX_TAXABLE_DIST};

const N_TURNS_RECOMP_CONNECTION: usize = 3*12; // number of turns until paths to water sources or city halls are rechecked if they haven't yet been found

// recomputes if it's been previously set to NotPossible
pub fn ret_water_dist_recomp(coord: u64, zone_exs: &mut HashedMapZoneEx, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
		bldgs: &Vec<Bldg>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz, turn: usize) {
	let zone_coord = return_zone_coord(coord, map_sz);
	zone_exs.create_if_empty(zone_coord, doctrine_templates);
	let zone_ex = zone_exs.get_mut(&zone_coord).unwrap();
	
	// compute distance to water source?
	match zone_ex.water_source_dist {
		Dist::NotInit | Dist::ForceRecompute {..} => {
			set_water_dist(coord, zone_exs, map_data, exs, bldgs, map_sz, turn);
		} Dist::NotPossible {turn_computed} => {
			if (turn_computed + N_TURNS_RECOMP_CONNECTION) < turn {
				set_water_dist(coord, zone_exs, map_data, exs, bldgs, map_sz, turn);
			}
		} Dist::Is {..} => {}
	}
}


impl Player<'_,'_,'_,'_> {
	// recomputes if it's been previously set to to NotPossible
	pub fn ret_city_hall_dist_recomp(&mut self, coord: u64, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
			bldgs: &mut Vec<Bldg>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz, turn: usize) -> Dist {
		let zone_coord = return_zone_coord(coord, map_sz);
		self.zone_exs.create_if_empty(zone_coord, doctrine_templates);
		let zone_ex = self.zone_exs.get(&zone_coord).unwrap();
		
		// compute dist to city_hall?
		match zone_ex.ret_city_hall_dist() {
			Dist::NotInit | Dist::ForceRecompute {..} => {
				self.set_city_hall_dist(coord, map_data, exs, bldgs, doctrine_templates, map_sz, turn);
			} Dist::NotPossible {turn_computed} => {
				if (turn_computed + N_TURNS_RECOMP_CONNECTION) < turn {
					self.set_city_hall_dist(coord, map_data, exs, bldgs, doctrine_templates, map_sz, turn);
				}
			} Dist::Is {..} => {}
		}
		
		self.zone_exs.get(&zone_coord).unwrap().ret_city_hall_dist()
	}
}

const MAX_WATER_SOURCE_DIST: usize = 200;

// sets players.zone_exs.water_source_dist
// (recomputes if already stored)
//
// actual bldg need not be present for calling this to make sense 
//

pub fn set_water_dist(start_coord: u64, zone_exs: &mut HashedMapZoneEx,
		map_data: &mut MapData, exs: &Vec<HashedMapEx>,
		bldgs: &Vec<Bldg>, map_sz: MapSz, turn: usize) {
	let exf = exs.last().unwrap();
	let ex = exf.get(&start_coord).unwrap();
	
	let closest_water_source = {
		let mut closest_water_source = Dist::NotPossible {turn_computed: turn};
		
		// nearby pipes
		if let Some(action_iface) = &mut start_water_mv_mode(start_coord, map_data, exf, map_sz) {
			let owner_id = ex.actual.owner_id.unwrap();
			
			///////// find closest water source
			for (bldg_ind_chk, b) in bldgs.iter().enumerate().filter(|(_, b)| b.owner_id == owner_id && b.construction_done == None && b.template.water_source) {
				if let Some(start_coord) = find_closest_pipe(b.coord, map_data, exs.last().unwrap(), map_sz) {
					let c = Coord::frm_ind(start_coord, map_sz);
					action_iface.update_move_search(c, map_data, exs, MvVars::None, bldgs);
					
					let path_len = action_iface.action.path_coords.len(); // ??
					if path_len == 0 || path_len > MAX_WATER_SOURCE_DIST {continue;} // no path found or too far
					
					// update
					if let Dist::Is {dist, bldg_ind} = &mut closest_water_source {
						if path_len < *dist {
							*dist = path_len;
							*bldg_ind = bldg_ind_chk;
						}
					}else{
						closest_water_source = Dist::Is {dist: path_len, bldg_ind: bldg_ind_chk};
					}
				}
			} // find closest
		}
		closest_water_source
	};
	
	let zone_coord = return_zone_coord(start_coord, map_sz);
	let zone_ex = zone_exs.get_mut(&zone_coord).unwrap();
	zone_ex.water_source_dist = closest_water_source;
}

// sets players.zone_exs.city_hall_dist
// (recomputes if already stored)
//
// actual bldg need not be present for calling this to make sense (called implicitly
// when determining whether to create a taxable bldg in this location)
//
// stats[owner.id].tax_income will be updated if a building exists at start_coord and
// it has b.template.bldg_type set to BldgType::Taxable(_)

impl Player<'_,'_,'_,'_> {
	pub fn set_city_hall_dist(&mut self, start_coord: u64, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
			bldgs: &mut Vec<Bldg>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz, turn: usize){
		let exf = exs.last().unwrap();
		let ex = exf.get(&start_coord).unwrap();
		
		// find closest city hall, if one is near
		let closest_city_hall = {
			let mut closest_city_hall = Dist::NotPossible {turn_computed: turn};
			
			// nearby roads
			if let Some(action_iface) = &mut start_civil_mv_mode(start_coord, map_data, exf, map_sz) {
				let owner_id = ex.actual.owner_id.unwrap();
		
				///////// find closest city hall
				for (bldg_ind_chk, b) in bldgs.iter().enumerate().filter(|(_, b)| b.owner_id == owner_id && b.construction_done == None) {
					if let BldgArgs::PopulationCenter {..} = &b.args {
						if let Some(start_coord) = find_closest_road(b.coord, map_data, exs.last().unwrap(), map_sz) {
							let c = Coord::frm_ind(start_coord, map_sz);
							action_iface.update_move_search(c, map_data, exs, MvVars::None, bldgs);
							
							let path_len = action_iface.action.path_coords.len(); // ??
							if path_len == 0 || path_len > MAX_TAXABLE_DIST {continue;} // no path found or too far
							
							// update
							if let Dist::Is {dist, bldg_ind} = &mut closest_city_hall {
								if path_len < *dist {
									*dist = path_len;
									*bldg_ind = bldg_ind_chk;
								}
							}else{
								closest_city_hall = Dist::Is {dist: path_len, bldg_ind: bldg_ind_chk};
							}
						}
					}
				} // bldg loop
			} // nearby roads
			closest_city_hall
		};
		
		{ // set city hall dist
			let zone_coord = return_zone_coord(start_coord, map_sz);
			let zone_ex = self.zone_exs.get_mut(&zone_coord).unwrap();
			zone_ex.set_city_hall_dist(zone_coord, closest_city_hall, bldgs, map_sz);
		}
		
		// update tax counter 
		if let Some(bldg_ind) = ex.bldg_ind {
			let b_start = &bldgs[bldg_ind];
			if b_start.coord == start_coord {
				let bt = b_start.template;
				if let BldgType::Taxable(_) = bt.bldg_type {
					debug_assertq!(bt.upkeep < 0.);
					let upkeep_new = if let Dist::Is {..} = &closest_city_hall {
						-bt.upkeep * return_effective_tax_rate(start_coord, map_data, exs, self, bldgs, doctrine_templates, map_sz, turn)
					}else{0.};
					
					bldgs[bldg_ind].set_taxable_upkeep(upkeep_new, &mut self.stats);
				}
			}
		}
	}
}

