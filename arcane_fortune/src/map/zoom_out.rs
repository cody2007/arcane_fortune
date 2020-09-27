use crate::disp::*;
use crate::disp_lib::endwin;
use crate::map::ZoomInd;
use crate::map::vars::*;
use crate::map::utils::*; // update_ex_data(), rm_ex_data()
use crate::buildings::*;
use crate::saving::SmSvType;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::zones::{StructureData, FOG_UNIT_DIST};
use crate::gcore::{Log, Relations};
use crate::units::Unit;
use crate::zones::return_zone_coord;
use crate::resources::N_RESOURCES_DISCOV_LOG;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

fn ret_max_ind(vec: Box<[isize]>) -> Option<usize> {
	let mut max_val = 0;
	let mut max_ind = None;
	for (ind, v) in vec.iter().enumerate(){
		if (*v) <= max_val {
			continue;
		}
		max_ind = Some(ind);
		max_val = *v;
	}
	max_ind
}

struct CityHallSum<'b,'bt,'ut,'rt,'dt> {
	sum: u32,
	bldg: &'b Bldg<'bt,'ut,'rt,'dt>
}

// used when the next lowest map down is the fully zoomed in map (where there are valid bldg inds and zone_ex information)
fn add_bldg_to_city_hall_sums<'b,'bt,'ut,'rt,'dt>(city_hall: &'b Bldg<'bt,'ut,'rt,'dt>, city_hall_sums: &mut Vec<CityHallSum<'b,'bt,'ut,'rt,'dt>>) {
	// already added?
	for city_hall_sum in city_hall_sums.iter_mut() {
		if city_hall_sum.bldg == city_hall {
			city_hall_sum.sum += 1;
			return;
		}
	}
	
	// new entry
	city_hall_sums.push(CityHallSum {
		sum: 1,
		bldg: city_hall
	});
}

// used on all zoom levels where there are not valid bldg inds and zone ex information stored (at the next zoom level down)
// find bldg by search for city matching `city_nm`
fn add_nm_to_city_hall_sums<'b,'bt,'ut,'rt,'dt>(city_nm: &str, city_hall_sums: &mut Vec<CityHallSum<'b,'bt,'ut,'rt,'dt>>, bldgs: &'b Vec<Bldg<'bt,'ut,'rt,'dt>>) {
	for b in bldgs.iter() {
		if let BldgArgs::CityHall {nm, ..} = &b.args {
			// found the city hall, now check if we've already logged it in `city_hall_sums`
			if nm == city_nm {
				// already added?
				for city_hall_sum in city_hall_sums.iter_mut() {
					if city_hall_sum.bldg == b {
						city_hall_sum.sum += 1;
						return;
					}
				}
				
				// new entry
				city_hall_sums.push(CityHallSum {
					sum: 1,
					bldg: b
				});
				
				return;
			}
		}
	}
	
	//panicq!("could not find city hall with name: {}", city_nm);
}

////////////////
// find the max bldg and owner on zoomed map at map[zoom_ind][i,j] using values on map[zoom_ind+1] (`...p1` variables)
fn compute_bldg_and_zoning_zoom_zcoord<'bt,'ut,'rt,'dt>(map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		zone_exs_owners: &Vec<HashedMapZoneEx>, bldgs: &Vec<Bldg>,
		bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, owners: &Vec<Owner>, 
		zoom_ind: usize, i: usize, j: usize) {
	
	//if zoom_ind == 3 && i == 99 && j == 451 {printlnq!("updating");}
	
	debug_assertq!(zoom_ind != map_data.max_zoom_ind());
	
	let map_sz = map_data.map_szs[zoom_ind];
	let map_sz_p1 = map_data.map_szs[zoom_ind+1];

	debug_assertq!(i < map_sz.h);
	debug_assertq!(j < map_sz.w);

	let (i_min, i_max, j_min, j_max) = {
		// compute the min and max values 
		// on the zoomed in map that result in (i,j)
		// on the zoomed out map
		// (i_min, i_max, ...) are on the zoomed in map
					
		let frac_i = (map_sz_p1.h as f32) / (map_sz.h as f32);
		let frac_j = (map_sz_p1.w as f32) / (map_sz.w as f32);

		let i_min = ((i as f32 - 0.5)*frac_i).ceil() as usize;
		let i_max = ((i as f32 + 0.5)*frac_i).floor() as usize;
		
		let j_min = ((j as f32 - 0.5)*frac_j).ceil() as usize;
		let j_max = ((j as f32 + 0.5)*frac_j).floor() as usize;
		
		(i_min, i_max, j_min, j_max)
	};
	
	/////////////
	// find all of the following variables on zoomed in map within defined box (derived above)
	
	let n_players = owners.len();
	
	let mut zone_type_sum = vec!(0; ZoneType::N as usize).into_boxed_slice();
	let mut owner_sum = vec!(0; n_players).into_boxed_slice();
	let mut bldg_template_sum = vec!(0; bldg_templates.len()).into_boxed_slice();
	let mut structure_type_sum = vec!(0; StructureType::N as usize).into_boxed_slice();
	let mut structure_orientation_sum = vec!(0; 4).into_boxed_slice();
	let mut city_hall_sums = Vec::new();
	
	let orientations = "/|\\-";
	{
		let exz = &mut exs[zoom_ind+1]; // zoomed in relative to point we're updating
		
		for i_p1 in i_min..=i_max {
		for j_p1 in j_min..=j_max {
			let j_p1_use = j_p1 % map_sz_p1.w;
			let map_coord = (i_p1*map_sz_p1.w + j_p1_use) as u64; // on zoom_ind + 1
			
			// only deal with locations that have zones, structures, etc
			if let Some(ex) = exz.get(&map_coord) {
				// zone
				if let Some(zone_type) = ex.actual.ret_zone_type() {
					zone_type_sum[zone_type as usize] += 1;
					
					// city hall
					if (zoom_ind+1) == map_data.max_zoom_ind() {
						// add based on bldg_ind
						let zone_coord = return_zone_coord(map_coord, map_sz_p1);
						let owner_id = ex.actual.owner_id.unwrap_or_else(|| panicq!("could not get owner")) as usize;
						let zone_ex = zone_exs_owners[owner_id].get(&zone_coord).unwrap_or_else(||
								panicq!("could not get zone. owner {} coord {}", owner_id, Coord::frm_ind(zone_coord, map_sz_p1)));
						match zone_ex.ret_city_hall_dist() {
							Dist::Is {bldg_ind, ..} | Dist::ForceRecompute {bldg_ind, ..} => {
								add_bldg_to_city_hall_sums(&bldgs[bldg_ind as usize], &mut city_hall_sums);
							}
							Dist::NotInit | Dist::NotPossible {..} => {}
						}
					}
				}
				
				// owner
				if let Some(owner_id) = ex.actual.owner_id {
					owner_sum[owner_id as usize] += 1;
				}
				
				// structure
				if let Some(structure) = ex.actual.structure {
					structure_type_sum[structure.structure_type as usize] += 1;
					
					let mut found = false;
					for o in 0..4 {
						if structure.orientation != orientations.chars().nth(o).unwrap() {
							continue;
						}
						structure_orientation_sum[o] += 1;
						found = true;
						break;
					}
					debug_assertq!(found);
				}
				
				// find max bldg type and owner for display
				if let Some(bldg_ind) = ex.bldg_ind {
					let b = &bldgs[bldg_ind as usize];
					bldg_template_sum[b.template.id as usize] += 1;
					
					if let BldgArgs::CityHall {..} = &b.args {
						add_bldg_to_city_hall_sums(b, &mut city_hall_sums);
					}
				}else if let Some(bt) = ex.actual.max_bldg_template {
					bldg_template_sum[bt.id as usize] += 1;
				}
				
				// city hall (for zoomed out levels where ex.bldg_ind & zone_ex_owners are not set)
				if let Some(max_city_nm) = &ex.actual.max_city_nm {
					add_nm_to_city_hall_sums(max_city_nm, &mut city_hall_sums, bldgs);
				}
			} // ex present
		}}
	}
	
	//////////////////////////
	// update zoom_ind_ex
	let map_coord = (i*map_sz.w + j) as u64;
	
	let max_zone_type_ind = ret_max_ind(zone_type_sum);
	let max_owner_ind = ret_max_ind(owner_sum);
	let max_bldg_template_ind = ret_max_ind(bldg_template_sum);
	let max_structure_type_ind = ret_max_ind(structure_type_sum);
	let max_structure_orientation_ind = ret_max_ind(structure_orientation_sum);
	let max_city_hall_sum = city_hall_sums.iter().max_by_key(|chs| chs.sum);
	
	// ex should be blank -- remove old ex?
	if max_zone_type_ind.is_none() && max_owner_ind.is_none() &&
			max_bldg_template_ind.is_none() && max_structure_type_ind.is_none() &&
			max_structure_orientation_ind.is_none() && max_city_hall_sum.is_none() {
		
		// remove ex if present
		if let Some(ex) = exs[zoom_ind].get_mut(&map_coord) {	
			ex.bldg_ind = None;
			ex.actual.max_bldg_template = None;
			ex.actual.structure = None;
			ex.actual.set_zone_direct(None);
			ex.actual.max_city_nm = None;
			
			// there are still units remaining, do not clear
			if let Some(unit_inds) = &ex.unit_inds {
				if unit_inds.len() != 0 {
					return;
				}
			}
			
			exs[zoom_ind].remove(&map_coord);
		}
		return;
	}
	
	// add ex
	exs[zoom_ind].create_if_empty(map_coord);
	let ex = exs[zoom_ind].get_mut(&map_coord).unwrap();
	
	ex.actual.owner_id = if let Some(ind) = max_owner_ind {
			Some(ind as SmSvType)
	}else {None};
	
	ex.actual.structure = if let Some(structure_ind) = max_structure_type_ind {
			Some(StructureData {
				structure_type: StructureType::from(structure_ind),
				health: std::u8::MAX,
				orientation: orientations.chars().nth(max_structure_orientation_ind.unwrap()).unwrap()
			})
		}else {None};
	
	ex.actual.set_zone_direct(if let Some(ind) = max_zone_type_ind {
			Some(ZoneType::from(ind))
		}else {None});
	
	ex.actual.max_bldg_template = if let Some(ind) = max_bldg_template_ind {
			Some(&bldg_templates[ind])
	}else {None};
	
	ex.actual.max_city_nm = if let Some(max_city_hall_sum) = max_city_hall_sum {
		if let BldgArgs::CityHall {nm, ..} = &max_city_hall_sum.bldg.args {
			Some(nm.clone())
		}else{panicq!("building type not a city hall");}
	}else {None};
}

///////////
// compute all zoom maps at all locations for:
// land type, arability, show_snow, elevation
// sets map_data.map_szs[0..N_EXPLICITLY_STORED_ZOOM_LVLS] (not setting ZOOM_IND_ROOT which is within this range)
impl <'rt>MapData<'rt> {
	pub fn compute_zoom_outs<'bt,'ut,'dt>(&mut self, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, zone_exs_owners: &Vec<HashedMapZoneEx>,
			bldgs: &Vec<Bldg>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, owners: &Vec<Owner>){
		
		let map_sz = self.map_szs[ZOOM_IND_ROOT];
		
		// start at most zoomed in level then zoom out (compute_bldg_and_zoning_zoom_zcoord() relies
		// on the next zoomed in level's values)
		for zoom_ind in (0..N_EXPLICITLY_STORED_ZOOM_LVLS).rev() {
			if zoom_ind == ZOOM_IND_ROOT {continue;}
			
			// map_szz: size of map for zoom_ind
			let map_szz = if zoom_ind != ZOOM_IND_SUBMAP {
					let z = MapData::zoom_spacing_explicitly_stored(zoom_ind);
					MapSz {h: map_sz.h/z, w: map_sz.w/z, sz: (map_sz.h/z)*(map_sz.w/z) }
				}else{
					MapSz {h: SUB_MAP_HEIGHT, w: SUB_MAP_WIDTH, sz: SUB_MAP_HEIGHT * SUB_MAP_WIDTH }
				};
			
			self.map_szs[zoom_ind] = map_szz;
			
			self.zoom_out[zoom_ind] = Vec::with_capacity(map_szz.sz);
			
			let n_avg_i = map_sz.h / map_szz.h;
			let n_avg_j = map_sz.w / map_szz.w;
			
			for i in 0..map_szz.h {
			for j in 0..map_szz.w {
				compute_bldg_and_zoning_zoom_zcoord(self, exs, zone_exs_owners, bldgs, bldg_templates, owners, zoom_ind, i, j);
				
				let mut val_type_sum = 0.;
				let mut val_arability_sum = 0.;
				let mut val_show_snow_sum = 0.;
				let mut val_elevation_sum = 0.;
				
				let mut n_summed: usize = 0;
				
				let mf = &self.zoom_out[ZOOM_IND_ROOT];

				for i_avg in 0..n_avg_i {
					let i_use = i*n_avg_i + i_avg; // for ZOOM_IND_ROOT
					if i_use >= map_sz.h { continue; }
					
					for j_avg in 0..n_avg_j {
						let j_use = (j*n_avg_j + j_avg) % map_sz.w;
						let map_coord = i_use*map_sz.w + j_use; // on zoomed out map
						
						let mfc = &mf[map_coord];
						
						val_type_sum += (mfc.map_type as usize) as f32;
						val_arability_sum += mfc.arability as f32;
						val_show_snow_sum += mfc.show_snow as usize as f32;
						val_elevation_sum += mfc.elevation as f32;
						
						n_summed += 1;
					} // j_avg
				} // i_avg
				
				debug_assertq!(n_summed > 0);
				let n_summed_f32 = n_summed as f32;
				self.zoom_out[zoom_ind].push(Map {
							map_type: MapType::from(val_type_sum / n_summed_f32),
							arability: val_arability_sum / n_summed_f32,
							elevation: val_elevation_sum / n_summed_f32,
							show_snow: (val_show_snow_sum / n_summed_f32).round() as i32 != 0,
							resource: None,
							resource_cont: None
						});
			}} // i,j
			
			let mz = &mut self.zoom_out[zoom_ind];
			
			// ensure deepwater is not adjacent to any land
			'map_type_loop: for i in 0..map_szz.sz {
				if mz[i].map_type != MapType::DeepWater {continue;}
				
				let c = Coord::frm_ind(i as u64, map_szz);
				for i_off in -1..=1 {
				for j_off in -1..=1 {
					if let Some(coord) = map_szz.coord_wrap(c.y + i_off, c.x + j_off) {
						if mz[coord as usize].map_type == MapType::Land {
							mz[i].map_type = MapType::ShallowWater;
							continue 'map_type_loop;
						}
					}
				}}
			}
		} // zoom loop
	}
}

#[derive(PartialEq)]
pub enum PresenceAction {
	SetPresentAndDiscover, // can discover land
	SetAbsent, // land should already be discovered
	DiscoverOnly // discover land but do not set as present
}

// add or remove fog of war window around unit, discover land if relevant
// 	only the active player has fog of war. others only have discovery
// 		^ (speed optimization--also, discoveries not computed on zoomed out maps for non-current players)
pub fn compute_active_window<'r,'bt,'ut,'rt,'dt>(coord_recompute: u64, player_ind: usize, is_cur_player: bool, action: PresenceAction,
		map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, pstats: &mut Stats<'bt,'ut,'rt,'dt>,
		owners: &Vec<Owner>, map_sz: MapSz, relations: &mut Relations, 
		units: &Vec<Unit>, logs: &mut Vec<Log>, turn: usize){
	#[cfg(feature="profile")]
	let _g = Guard::new("compute_active_window");
	
	assertq!(map_sz == *map_data.map_szs.last().unwrap());
	
	let i_pos = (coord_recompute as usize / map_sz.w) as isize;
	let j_pos = (coord_recompute as usize % map_sz.w) as isize;
	
	//let is_human = owners[player_ind].player_type == PlayerType::Human;
	
	let mut prev_land_discov_coord = ((map_sz.w * map_sz.h) + 3) as usize; // in the size of `pstats.land_discov` -- first set out of range
	
	for j in -(FOG_UNIT_DIST as isize)..=(FOG_UNIT_DIST as isize) {
		// ellipse: y = (h/w)*sqrt(w*w - x*x)
		let i_bound = (0.5*(4.*((FOG_UNIT_DIST*FOG_UNIT_DIST) as f32) - (j*j) as f32).sqrt()).round() as isize;
		for i in -i_bound..=i_bound {
			let coord_discover = Coord {y: (i_pos + i) as isize, x: (j + j_pos) as isize};
			
			if let Some(coord) = map_sz.coord_wrap(coord_discover.y, coord_discover.x) {
				// discover land, resources, and civs
				if let PresenceAction::SetPresentAndDiscover | PresenceAction::DiscoverOnly = action {
					let land_discov_coord = pstats.land_discov.last().unwrap().map_to_discov_coord(Coord::frm_ind(coord, map_sz));
					if prev_land_discov_coord != land_discov_coord {
						compute_zooms_coord(coord, RecompType::Discover {pstats, is_cur_player}, map_data, exs, owners);
						prev_land_discov_coord = land_discov_coord;
					}
					
					// log resource discovery
					if let Some((resource, coord)) = map_data.get(ZoomInd::Full, coord)
													.get_resource_and_coord(coord, map_data, map_sz) {
						let res_dis_coords = &mut pstats.resources_discov_coords[resource.id as usize];
						
						if res_dis_coords.len() < N_RESOURCES_DISCOV_LOG {
							if !res_dis_coords.contains(&coord) {
								res_dis_coords.push(coord);
							}
						}
					}
					
					// discover civs
					if let Some(ex) = exs.last().unwrap().get(&coord) {
						// land owned by undiscov civ?
						if let Some(owner_id) = ex.actual.owner_id {
							relations.discover_civ(player_ind, owner_id as usize, logs, turn);
						}
						
						// any undiscovered civs from units?
						if let Some(unit_inds) = &ex.unit_inds {
							for unit_ind in unit_inds.iter() {
								relations.discover_civ(player_ind, units[*unit_ind as usize].owner_id as usize, logs, turn);
							}
						} // units here
					} // discov units
				}
				
				// update fog of war
				if is_cur_player {
					if action == PresenceAction::SetPresentAndDiscover {
						compute_zooms_coord(coord, RecompType::Fog {pstats, present: true}, map_data, exs, owners);
					}else if action == PresenceAction::SetAbsent {
						compute_zooms_coord(coord, RecompType::Fog {pstats, present: false}, map_data, exs, owners);
					}
				}
			} // valid coord
		} // i
	} // j
}

// compute zoomed out discoveries when switching players because it has not been kept up-to-date for speed optimization
pub fn compute_zoomed_out_discoveries<'r,'bt,'ut,'rt,'dt>(map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		pstats: &mut Stats<'bt,'ut,'rt,'dt>, owners: &Vec<Owner>){
	let land_discov_full = pstats.land_discov.last().unwrap().clone();
	for coord in LandDiscovIter::from(&land_discov_full) {
		compute_zooms_coord(coord, RecompType::Discover {pstats, is_cur_player: true}, map_data, exs, owners);
	}
}

///////////////////
// recompute:
//
// fog for player (for when a part of the map is no longer visible)
// OR
// discover land (if is_cur_player = False, we compute only the first zoom level)
// OR
// unit maxes
// OR
// bldg maxes & zoning
//
// on all zoomed maps. specifically, only the points
// that represent coord_recompute (on the fully zoomed map)

pub enum RecompType<'r,'bt,'ut,'rt,'dt,'z> {
	Fog {pstats: &'r mut Stats<'bt,'ut,'rt,'dt>, present: bool},
	Discover {pstats: &'r mut Stats<'bt,'ut,'rt,'dt>, is_cur_player: bool},
	AddUnit(usize),
	RmUnit(usize),
	Bldgs(&'r Vec<Bldg<'bt,'ut,'rt,'dt>>, &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, &'z Vec<HashedMapZoneEx>)
}

pub fn compute_zooms_coord<'r,'bt,'ut,'rt,'z,'dt>(coord_recompute: u64, mut recomp_type: RecompType<'r,'bt,'ut,'rt,'dt,'z>,
		map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, owners: &Vec<Owner>) {
	#[cfg(feature="profile")]
	let _g = Guard::new(match recomp_type {
		RecompType::Fog {..} => {"compute_zooms_coord Fog"}
		RecompType::Discover {is_cur_player: true, ..} => {"compute_zooms_coord Discover ai=true"}
		RecompType::Discover {is_cur_player: false, ..} => {"compute_zooms_coord Discover ai=false"}
		RecompType::AddUnit(_) => {"compute_zooms_coord AddUnit"}
		RecompType::RmUnit(_) => {"compute_zooms_coord RmUnit"}
		RecompType::Bldgs(_,_,_) => {"compute_zooms_coord Bldgs"}
	});
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	// most zoomed in coordinates
	let mut coord_prev = Coord::frm_ind(coord_recompute, map_sz);
	
	/////////////////////////////////////////////////////////////
	// start at one step above fully zoomed in and then zoom out
	// (so that we can simply use ex values used from previous zoom levels instead of zooming
	//  all the way into ZoomInd::Full each time)
	
	let last_zoom_ind = match recomp_type {
		RecompType::Discover {..} | RecompType::Fog {..} | RecompType::AddUnit(_) | RecompType::RmUnit(_) => {
			map_data.max_zoom_ind()
		} RecompType::Bldgs(_,_,_) => {
			map_data.max_zoom_ind() - 1
		}
	};
	
	for zoom_ind in (0..=last_zoom_ind).rev() { // note RecompType::Discover {is_cur_player: False} assumes this reverse loop
		let map_sz = map_data.map_szs[zoom_ind];
		
		// convert coordinates to zoom_ind
		let coord = if zoom_ind != map_data.max_zoom_ind() {
			coord_prev.to_zoom(zoom_ind+1, zoom_ind, &map_data.map_szs)
		}else {coord_prev.clone()};
		
		let coord_ind = coord.to_ind(map_sz) as u64;
		
		match recomp_type {
			RecompType::Discover {ref mut pstats, is_cur_player} => {
				pstats.land_discov[zoom_ind].map_coord_discover(Coord::frm_ind(coord_ind, map_sz));
				if !is_cur_player {return;} // only full map is computed
			} RecompType::Fog {ref mut pstats, present} => {
				let fog = &mut pstats.fog[zoom_ind];
				
				// unit or bldg present, we just use the actual value of ex
				if present {
					fog.remove(&coord_ind);
					
				// unit or bldg not present, we copy the current ex value
				}else{
					if let Some(ex) = exs[zoom_ind].get(&coord_ind) {
						fog.insert(coord_ind, ex.actual.clone());
					}else{
						fog.remove(&coord_ind);
					}
				}
			} RecompType::AddUnit(unit_ind) => {
				// update previous ex
				if let Some(ref mut ex) = exs[zoom_ind].get_mut(&coord_ind) {
					if let Some(ref mut unit_inds) = ex.unit_inds {
						unit_inds.push(unit_ind);
					}else{
						ex.unit_inds = Some(vec![unit_ind]);
					}
					
				// add new ex
				}else{
					let mut ex = MapEx::default();
					ex.unit_inds = Some(vec![unit_ind]);
					exs[zoom_ind].update_or_insert(coord_ind, ex);
				}
			} RecompType::RmUnit(unit_ind) => {
				// check if we remove unit_ind from unit_inds or if we can remove the entire ex entry
				let mut rm_unit = || {
					// update previous ex
					if let Some(ref mut ex) = exs[zoom_ind].get_mut(&coord_ind) {
						if let Some(ref mut unit_inds) = &mut ex.unit_inds {
							// delete entire ex entry if empty
							if unit_inds.len() == 1 && ex.bldg_ind.is_none() && ex.actual.is_empty() {
								exs[zoom_ind].remove(&coord_ind);
							
							/////// remove only unit_ind from ex:
								
							// no unit_inds remaining
							}else if unit_inds.len() == 1 {
								ex.unit_inds = None;
							
							// remove only unit_ind from unit_inds
							}else{
								for (i, ui) in unit_inds.iter().enumerate() {
									if *ui == unit_ind {
										unit_inds.swap_remove(i);
										return;
									}
								}
								panicq!("could not find unit_ind {} at coord {}, {}  at zoom {}", unit_ind, coord.y, coord.x, zoom_ind);
							}
							
						/////// should never happen - ex.unit_inds = None
						}else {
							endwin();
							for (coord, ex) in exs[zoom_ind].iter() {
								if let Some(unit_inds) = &ex.unit_inds {
									
									for ind in unit_inds.iter() {
										println!("unit_ind {}", ind);
									}
									
									if unit_inds.contains(&unit_ind) {
										println!("found unit_ind at {}, {}", *coord / map_sz.w as u64, *coord % map_sz.w as u64);
										break;
									}
								}
							}
							panicq!("attempted to remove non-existant unit_ind {} from coord {}, {}  at zoom {}, max_zoom_ind {}, ex entries: {}", 
									unit_ind, coord.y, coord.x, zoom_ind, map_data.max_zoom_ind(), exs[zoom_ind].len());
						}
					}else {panicq!("attempted to remove non-existant unit_ind {} from coord {}, {}  at zoom {}", unit_ind, coord.y as usize, coord.x as usize, zoom_ind);}
				};
				rm_unit();
			} RecompType::Bldgs(b, bts, zone_exs_owners) => {
				compute_bldg_and_zoning_zoom_zcoord(map_data, exs, zone_exs_owners, b, bts, owners, zoom_ind, coord.y as usize, coord.x as usize);
			}
		}
		
		coord_prev = coord;
	} // zoom loop
}

