use crate::map::*;
use crate::buildings::*;
use crate::movement::*;
use crate::disp::*;
use crate::zones::*;
use crate::saving::*;
use crate::gcore::rand::XorState;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::doctrine::DoctrineTemplate;
use crate::renderer::endwin;
use crate::containers::*;
use crate::player::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;
//use std::cmp::min;

//const ZONE_DEMAND_SEARCH_DEPTH: usize = 1200;

pub fn sell_prod(bldg_ind_sell: usize, bldgs: &mut Vec<Bldg>, map_data: &mut MapData, exs: &Vec <HashedMapEx>, 
		map_sz: MapSz, pstats: &mut Stats, rng: &mut XorState){
	
	const CHANCE_SKIP_BLDG: f32 = 0.5;
	
	let b_sell = &bldgs[bldg_ind_sell];
	let zone_seller = if let Some(zone_type) = b_sell.template.bldg_type.zone_type() {
		zone_type
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
			if let Some(zone_type) = b.template.bldg_type.zone_type() {
				// not in correct zone
				if (zone_seller == ZoneType::Residential && zone_type == ZoneType::Residential) ||
				   (zone_seller == ZoneType::Business && zone_type != ZoneType::Residential) ||
				   (zone_seller == ZoneType::Agricultural && zone_type != ZoneType::Residential && zone_type != ZoneType::Business) ||
				   (zone_seller == ZoneType::Industrial && zone_type != ZoneType::Business && zone_type != ZoneType::Agricultural) {
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

pub const MAX_TAXABLE_DIST: usize = 200;

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

// return potential tax income for zone type
// does not recompute city_hall_dist if it's already initialized
pub fn return_effective_tax_rate(coord: u64, map_data: &mut MapData, exs: &mut Vec<HashedMapEx>,
		player: &mut Player, bldgs: &mut Vec<Bldg>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz, turn: usize) -> f32 {
	// compute tax income
	match player.ret_city_hall_dist_recomp(coord, map_data, exs, bldgs, doctrine_templates, map_sz, turn) {
		Dist::Is {dist, bldg_ind} => {
			let b = &bldgs[bldg_ind];
			
			let dist_scale = if dist >= MAX_TAXABLE_DIST {
					0.
				}else{
					((MAX_TAXABLE_DIST - dist) as f32) / (MAX_TAXABLE_DIST as f32)
				};
			
			if let BldgArgs::PopulationCenter {tax_rates, ..} = &b.args {
				let zt = exs.last().unwrap().get(&coord).unwrap().actual.ret_zone_type().unwrap() as usize;
				dist_scale * (tax_rates[zt] as f32 / 100.)
				
			}else {panicq!("no population center tax arguments")}
	
		} Dist::NotPossible {..} => {0.
		} Dist::NotInit | Dist::ForceRecompute {..} => {panicq!("city hall dist left unitialized")}
	}
}

// run at end of turn for new taxable constructions moving in for each zone type
pub fn create_taxable_constructions<'bt,'ut,'rt,'dt>(map_data: &mut MapData<'rt>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, map_sz: MapSz, gstate: &mut GameState){
	#[cfg(feature="profile")]
	let _g = Guard::new("create_taxable_constructions");
	
	//if gstate.rng.gen_f32b() < 0.25 {return;}
	
	let exf = exs.last_mut().unwrap();
	
	if let Some(coord) = SampleType::ZoneDemand.coord_frm_turn_computed(players, exf, map_sz, gstate) {
		if let Some(ex) = exf.get(&coord) {
			if !land_clear_ign_zone_ign_owner(coord, &map_data.get(ZoomInd::Full, coord), exf) {
				return;
			}
			
			// continue if we don't reach the end of the map
			let c = Coord::frm_ind(coord, map_sz);
			if map_sz.coord_wrap(c.y + 1, c.x + 1).is_none() {return;}
			
			if let Some(zone) = ex.actual.ret_zone() {
				let owner_id = ex.actual.owner_id.unwrap();
				
				let resource_present = {
					let mfc = map_data.get(ZoomInd::Full, coord);
					!mfc.resource.is_none() || !mfc.resource_cont.is_none()
				};
				
				struct ZoneStats {
					demand: f32,
					wealth: i32
				}
				
				let mut zone_stats_cache: Option<ZoneStats> = None;
				let bldg_template_inds = gstate.rng.inds(temps.bldgs.len());
				
				// loop over bldgs
				'bldg_loop: for bldg_template_ind in bldg_template_inds.iter() {
					let bt = &temps.bldgs[*bldg_template_ind];
					
					let h = bt.sz.h as isize;
					let w = bt.sz.w as isize;
					
					// not correct zone type
					if bt.bldg_type != BldgType::Taxable(zone) {continue;}
					
					// at edge of map
					if map_sz.coord_wrap(c.y + h, c.x + w) == None {continue;}
					
					// check if all plots in the to-be bldg are valid
					let exf = exs.last_mut().unwrap();
					for i2 in 0..h {
					for j2 in 0..w {
						let coord = map_sz.coord_wrap(c.y + i2, c.x + j2).unwrap();
						
						if !land_clear(coord, Some(MatchZoneOwner {zone_type: zone.ztype, owner_id}), &map_data.get(ZoomInd::Full, coord), exf) {
							continue 'bldg_loop;
						}
					}}
					
					///////////
					let player = &mut players[owner_id as usize];
					let effective_tax = return_effective_tax_rate(coord, map_data, exs, player, bldgs, temps.doctrines, map_sz, gstate.turn);
					assert!(effective_tax >= 0.);
					
					macro_rules! add_bldg{($zone_stats: expr) => {
						if ((gstate.rng.gen_f32b() < 4.*(4.*$zone_stats.demand - effective_tax)) || // add based on zone demand
								(resource_present && (gstate.rng.gen_f32b() < 1.))) && // add based on a resource being present
								add_bldg(coord, owner_id, bldgs, bt, None, Some($zone_stats.wealth), temps, map_data, exs, players, gstate) {
							
							bldgs.last_mut().unwrap().construction_done = None;
							
							break;
						}
					};}
					
					if let Some(zone_stats) = &zone_stats_cache {
						add_bldg!(zone_stats);
					}else{
						let zone_stats = ZoneStats {
							demand: return_potential_demand(coord, map_data, exs, player, bldgs, map_sz, gstate.turn),
							wealth: return_wealth(coord, map_data, exs, bldgs, player, temps, gstate, map_sz)
						};
						
						add_bldg!(zone_stats);
						
						zone_stats_cache = Some(zone_stats);
					}
				} // bldg loop
			}
		}
	}
}

