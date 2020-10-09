use crate::map::*;
use crate::buildings::*;
use crate::movement::*;
use crate::disp::*;
use crate::zones::*;
use crate::saving::*;
use crate::gcore::rand::XorState;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::doctrine::DoctrineTemplate;
use crate::gcore::Log;
use crate::disp_lib::endwin;
use crate::ai::AIState;
use crate::containers::Templates;
use crate::player::{Player, Nms, Stats};
#[cfg(feature="profile")]
use crate::gcore::profiling::*;
//use std::cmp::min;

//const ZONE_DEMAND_SEARCH_DEPTH: usize = 1200;

pub fn sell_prod(bldg_ind_sell: usize, bldgs: &mut Vec<Bldg>, map_data: &mut MapData, exs: &Vec <HashedMapEx>, 
		map_sz: MapSz, pstats: &mut Stats, rng: &mut XorState){
	
	const CHANCE_SKIP_BLDG: f32 = 0.5;
	
	let b_sell = &bldgs[bldg_ind_sell];
	let zone_seller = if let BldgType::Taxable(zone) = b_sell.template.bldg_type {
		zone
	}else{panicq!("seller not taxable");};
	
	debug_assertq!( (zone_seller == ZoneType::Residential && b_sell.n_sold() < b_sell.n_residents()) || // everyone's employed
		       (zone_seller != ZoneType::Residential && b_sell.n_sold() < b_sell.prod_capac())); // bldg has production to sell
	
	let coord_sell = Coord::frm_ind(b_sell.coord, map_sz);
	let action_iface = start_civil_mv_mode(b_sell.coord, map_data, exs.last().unwrap(), map_sz); 
	if action_iface.is_none() {return;}
	let mut action_iface = action_iface.unwrap();
	
	pub enum DistSimp {
		NotInit,
		NotPossible,
		Is
	}
	
	// sort distances
	struct BldgDist {dist: usize, bldg_ind: usize, road_dist: DistSimp}
	let mut sorted_dists = Vec::with_capacity(bldgs.len());
	for (bldg_ind, b) in bldgs.iter().enumerate() {
		let coord_i = Coord::frm_ind(b.coord, map_sz);
		sorted_dists.push(BldgDist {dist: manhattan_dist(coord_sell, coord_i, map_sz), bldg_ind,
					    road_dist: DistSimp::NotInit});
	}
	sorted_dists.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
	
	// find buyer
	let bldg_ind_buy;
	let mut found_any_route = false;
	'outer: loop {
		for dist in sorted_dists[1..].iter_mut() {
			if let DistSimp::NotPossible = dist.road_dist {continue;}
			
			let b = &bldgs[dist.bldg_ind];
			// diff owner
			if b.owner_id != b_sell.owner_id {
				continue;
			}
			
			// zoned
			if let BldgType::Taxable(zone) = b.template.bldg_type {
				// not in correct zone
				if (zone_seller == ZoneType::Residential && zone == ZoneType::Residential) ||
				   (zone_seller == ZoneType::Business && zone != ZoneType::Residential) ||
				   (zone_seller == ZoneType::Agricultural && zone != ZoneType::Residential && zone != ZoneType::Business) ||
				   (zone_seller == ZoneType::Industrial && zone != ZoneType::Business && zone != ZoneType::Agricultural) {
					   continue;
				}
				
				// if commercial transaction, consider consumption capacity of purchaser
				if zone_seller != ZoneType::Residential {
					let cons_capac = b.cons_capac();
					
					// can't consume anything
					if cons_capac == 0 {continue;}
					
					let cons = b.cons();
					debug_assertq!(cons <= cons_capac); // can't buy more than capacity
					
					// can't consume any more
					if cons == cons_capac {continue;}
				
				// if resident trying to find a job, this bldg is not hiring
				}else if b.n_residents() == b.template.resident_max {
					continue;
				}
							
				// only recompute route if not already computed
				if let DistSimp::NotInit = dist.road_dist {
					// \/ none if not close enough to any road
					if let Some(coord_buy) = find_closest_road(b.coord, map_data, exs.last().unwrap(), map_sz) {
						let coord_buy = Coord::frm_ind(coord_buy, map_sz);
						
						// too far to consider, abort loop (all bldgs after this in the list will be larger)
						if manhattan_dist(coord_buy, coord_sell, map_sz) >= action_iface.max_search_depth {
							break;
						}
						
						// check if bldg connects to current position
						action_iface.update_move_search(coord_buy, map_data, exs, MvVars::None, bldgs);
						
						// no connection
						if action_iface.action.path_coords.len() == 0 {
							dist.road_dist = DistSimp::NotPossible; // skip this bldg next loop
							continue;
						}
						dist.road_dist = DistSimp::Is;
						found_any_route = true;
					}
				}
				
				if rng.gen_f32b() < CHANCE_SKIP_BLDG {continue;}
				
				bldg_ind_buy = dist.bldg_ind;
				break 'outer;
			}
		} // loop through bldgs
		
		// found nothing, abort
		if !found_any_route {return;}
	}
	
	debug_assertq!(bldg_ind_sell != bldg_ind_buy);
	
	add_commute_to(bldg_ind_sell, bldg_ind_buy, bldgs, pstats);
	debug_assertq!( (zone_seller == ZoneType::Residential && bldgs[bldg_ind_sell].n_sold() <= bldgs[bldg_ind_sell].n_residents()) || // everyone's employed
		       (zone_seller != ZoneType::Residential && bldgs[bldg_ind_sell].n_sold() <= bldgs[bldg_ind_sell].prod_capac())); // bldg has production to sell
}

const MAX_TAXABLE_DIST: usize = 200;

// uninit all of owner's city hall distances
// for example after road has been constructed
pub fn uninit_city_hall_dists(owner_id: SmSvType, zone_exs: &mut HashedMapZoneEx, bldgs: &mut Vec<Bldg>, map_sz: MapSz) {
	for bldg_ind in 0..bldgs.len() {
		let b = &bldgs[bldg_ind];
		if b.owner_id != owner_id {continue;}
		
		let zone_coord = return_zone_coord(b.coord, map_sz);
		if let Some(zone_ex) = zone_exs.get_mut(&zone_coord) {
			match zone_ex.ret_city_hall_dist() {
				Dist::NotInit | Dist::NotPossible {..} | Dist::ForceRecompute {..} => {}
				Dist::Is {dist, bldg_ind} => {
					zone_ex.set_city_hall_dist(zone_coord, Dist::ForceRecompute {bldg_ind, dist}, bldgs, map_sz);
				}
			}
		}
	}
}

// sets city_hall_dist and city_hall_bldg_ind in zones_info
// (recomputes both if already stored)
//
// actual bldg need not be present for calling this to make sense (called implicitly
// when determining whether to create a taxable bldg in this location)
//
// stats[owner.id].tax_income will be updated if a building exists at start_coord and
// it has b.template.bldg_type set to BldgType::Taxable(_)

pub fn set_city_hall_dist(start_coord: u64, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
		player: &mut Player, bldgs: &mut Vec<Bldg>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz, turn: usize){
	
	let exf = exs.last().unwrap();
	let ex = exf.get(&start_coord).unwrap();
	let owner_id = ex.actual.owner_id.unwrap();
	
	let mut action_iface = start_civil_mv_mode(start_coord, map_data, exf, map_sz); 
	
	// decrease counter, and increase at end of function
	// if function returns early, there's no distance possible, and therefore no tax
	macro_rules! set_taxable_upkeep{($val: expr) => (
		if let Some(bldg_ind) = exs.last().unwrap().get(&start_coord).unwrap().bldg_ind {
			let b_start = &bldgs[bldg_ind];
			if b_start.coord == start_coord {
				if let BldgType::Taxable(_) = b_start.template.bldg_type {
					bldgs[bldg_ind].set_taxable_upkeep($val, &mut player.stats);
				}
			}
		}
	);}
	
	let zone_coord = return_zone_coord(start_coord, map_sz);
	
	// roads around
	if let Some(action_iface) = &mut action_iface {
		///////// find closest city hall
		let mut closest_dist: Option<usize> = None;
		let mut closest_city_hall_ind: Option<usize> = None;
		
		for (bldg_ind, b) in bldgs.iter().enumerate() {
			// skip if it's not the current player's city hall
			if b.owner_id != owner_id || b.template.nm[0] != CITY_HALL_NM || !b.construction_done.is_none() {
				continue;
			}
			
			if let Some(start_coord) = find_closest_road(b.coord, map_data, exs.last().unwrap(), map_sz) {
				let c = Coord::frm_ind(start_coord, map_sz);
				action_iface.update_move_search(c, map_data, exs, MvVars::None, bldgs);
				
				let path_len = action_iface.action.path_coords.len(); // ??
				if path_len == 0 {continue;} // no path found
				
				// update
				if closest_dist.is_none() || path_len < closest_dist.unwrap() {
					closest_dist = Some(path_len);
					closest_city_hall_ind = Some(bldg_ind);
				}
			}
		} // find closest
		
		let zone_ex = player.zone_exs.get_mut(&zone_coord).unwrap();
		
		if closest_dist.is_none() || closest_dist.unwrap() > MAX_TAXABLE_DIST {
			zone_ex.set_city_hall_dist(zone_coord, Dist::NotPossible {turn_computed: turn}, bldgs, map_sz);
			set_taxable_upkeep!(0.);
			return;
		}
		
		zone_ex.set_city_hall_dist(zone_coord, Dist::Is {  dist: closest_dist.unwrap(), 
						bldg_ind: closest_city_hall_ind.unwrap() }, bldgs, map_sz);
		
		// update tax counter 
		if let Some(bldg_ind) = ex.bldg_ind {
			let b_start = &bldgs[bldg_ind];
			if b_start.coord == start_coord {
				let bt = b_start.template;
				if let BldgType::Taxable(_) = bt.bldg_type {
					debug_assertq!(bt.upkeep < 0.);
					let upkeep_new = -bt.upkeep * return_effective_tax_rate(start_coord, map_data, exs, player, bldgs, doctrine_templates, map_sz, turn);
					bldgs[bldg_ind].set_taxable_upkeep(upkeep_new, &mut player.stats);
				}
			}
		}
		
	// no roads around
	}else{
		let zone_ex = player.zone_exs.get_mut(&zone_coord).unwrap();
		
		zone_ex.set_city_hall_dist(zone_coord, Dist::NotPossible {turn_computed: turn}, bldgs, map_sz);
		set_taxable_upkeep!(0.);
	}
}

// return potential tax income for zone type
// does not recompute city_hall_dist if it's already initialized
pub fn return_effective_tax_rate(coord: u64, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
		player: &mut Player, bldgs: &mut Vec<Bldg>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz, turn: usize) -> f32 {
	let zone_coord = return_zone_coord(coord, map_sz);
	player.zone_exs.create_if_empty(zone_coord, doctrine_templates);
	let zone_ex = player.zone_exs.get_mut(&zone_coord).unwrap();
	
	// compute dist to city_hall?
	match zone_ex.ret_city_hall_dist() {
		Dist::NotInit | Dist::ForceRecompute {..} => {
			set_city_hall_dist(coord, map_data, exs, player, bldgs, doctrine_templates, map_sz, turn);
		} Dist::NotPossible{turn_computed} => {
			if (turn_computed + N_TURNS_RECOMP_ZONE_DEMAND) < turn {
				set_city_hall_dist(coord, map_data, exs, player, bldgs, doctrine_templates, map_sz, turn);
			}
		} Dist::Is {..} => {}
	}
	
	let zone_ex = player.zone_exs.get(&zone_coord).unwrap();
	
	// compute tax income
	match zone_ex.ret_city_hall_dist() {
		Dist::Is {dist, bldg_ind} => {
			let b = &bldgs[bldg_ind];
			debug_assertq!(b.template.nm[0] == CITY_HALL_NM);
			
			let dist_scale = if dist >= MAX_TAXABLE_DIST {
					0.
				}else{
					((MAX_TAXABLE_DIST - dist) as f32) / (MAX_TAXABLE_DIST as f32)
				};
			
			if let BldgArgs::CityHall{tax_rates, ..} = &b.args {
				let zt = exs.last().unwrap().get(&coord).unwrap().actual.ret_zone_type().unwrap() as usize;
				dist_scale * (tax_rates[zt] as f32 / 100.)
				
			}else {panicq!("no city hall tax arguments")}
	
		} Dist::NotPossible {..} => {0.
		} Dist::NotInit | Dist::ForceRecompute {..} => {panicq!("city hall dist left unitialized")}
	}
}

// run at end of turn for new taxable constructions moving in for each zone type
pub fn create_taxable_constructions<'bt,'ut,'rt,'dt>(map_data: &mut MapData<'rt>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, logs: &mut Vec<Log>, map_sz: MapSz, turn: usize, rng: &mut XorState){
	#[cfg(feature="profile")]
	let _g = Guard::new("create_taxable_constructions");
	
	if rng.gen_f32b() < 0.25 {return;}
	
	let exf = exs.last_mut().unwrap();
	
	if let Some(coord) = SampleType::ZoneDemand.coord_frm_turn_computed(players, exf, map_sz, turn, rng) {
		if let Some(ex) = exf.get(&coord) {
			if !land_clear_ign_zone_ign_owner(coord, &map_data.get(ZoomInd::Full, coord), exf) {
				return;
			}
			
			// continue if we don't reach the end of the map
			let c = Coord::frm_ind(coord, map_sz);
			if map_sz.coord_wrap(c.y + 1, c.x + 1).is_none() {return;}
			
			if let Some(zone_type) = ex.actual.ret_zone_type() {
				
				let owner_id = ex.actual.owner_id.unwrap();
				
				let resource_present = {
					let mfc = map_data.get(ZoomInd::Full, coord);
					!mfc.resource.is_none() || !mfc.resource_cont.is_none()
				};
				
				let mut zone_demand: Option<f32> = None;
				let bldg_template_inds = rng.inds(temps.bldgs.len());
				
				// loop over bldgs
				'bldg_loop: for bldg_template_ind in bldg_template_inds.iter() {
					let bt = &temps.bldgs[*bldg_template_ind];
					
					let h = bt.sz.h as isize;
					let w = bt.sz.w as isize;
					
					// not correct zone type
					if bt.bldg_type != BldgType::Taxable(zone_type) {continue;}
					
					// at edge of map
					if map_sz.coord_wrap(c.y + h, c.x + w) == None {continue;}
					
					// check if all plots in the to-be bldg are valid
					let exf = exs.last_mut().unwrap();
					for i2 in 0..h {
					for j2 in 0..w {
						let coord = map_sz.coord_wrap(c.y + i2, c.x + j2).unwrap();
						
						if !land_clear(coord, Some(MatchZoneOwner {zone: zone_type, owner_id}), &map_data.get(ZoomInd::Full, coord), exf) {
							continue 'bldg_loop;
						}
					}}
					
					///////////
					let player = &mut players[owner_id as usize];
					let effective_tax = return_effective_tax_rate(coord, map_data, exs, player, bldgs, temps.doctrines, map_sz, turn);
					assert!(effective_tax >= 0.);
					
					// prevent re-computing unnecessarily
					if zone_demand.is_none() {
						zone_demand = Some(return_potential_demand(coord, map_data, exs, player, bldgs, map_sz, turn));
					}
					
					// add bldg
					if ((rng.gen_f32b() < 2.*(4.*zone_demand.unwrap() - effective_tax)) || // add based on zone demand
						(resource_present && (rng.gen_f32b() < 1.))) && // add based on a resource being present
							add_bldg(coord, owner_id, bldgs, bt, None, temps.bldgs, temps.doctrines, map_data, exs, players, &temps.nms, turn, logs, rng) {
						
						let ind = bldgs.len()-1;
						bldgs[ind].construction_done = None;
						
						break;
					}
				} // bldg loop
			}
		}
	}
}

