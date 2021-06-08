use crate::map::*;
use crate::disp::*;
use crate::gcore::hashing::HashedMapEx;
use crate::zones::StructureData;
use crate::renderer::endwin;

pub struct DistComponents {pub h: usize, pub w: usize}

pub fn manhattan_dist_components(c1: Coord, c2: Coord, map_sz: MapSz) -> DistComponents {
	debug_assertq!(c1.y >= 0 && c2.y >= 0);
	debug_assertq!(c1.x >= 0 && c2.y >= 0);
	debug_assertq!(c1.y < (map_sz.h as isize) && c2.y < (map_sz.h as isize));
	debug_assertq!(c1.x < (map_sz.w as isize) && c2.x < (map_sz.w as isize));
	
	let w = map_sz.w as isize;
	let d = c1.x - c2.x;
	
	let mut x_diff = d.abs();
	let x_diff2 = (d - w).abs();
	let x_diff3 = (d + w).abs();
	
	if x_diff > x_diff2 {x_diff = x_diff2;}
	if x_diff > x_diff3 {x_diff = x_diff3;}
	
	DistComponents {
		h: (c1.y - c2.y).abs() as usize,
		w: x_diff as usize
	}
}

pub fn manhattan_dist(c1: Coord, c2: Coord, map_sz: MapSz) -> usize {
	let c = manhattan_dist_components(c1, c2, map_sz);
	c.h + c.w
}

#[inline]
pub fn manhattan_dist_inds(c1: u64, c2: u64, map_sz: MapSz) -> usize {
	manhattan_dist(Coord::frm_ind(c1, map_sz), Coord::frm_ind(c2, map_sz), map_sz)
}

impl MapSz {
	pub fn average(&self, start: Coord, end: Coord) -> Coord {
		// note: doesn't properly handle wrapping around the map, should be fixed...
		//if let Some(c) = self.coord_wrap((start.y + end.y) / 2, (start.x + end.x) / 2) {
		//	c
		//}else{panicq!("could not compute average of {} and {}", start, end);}
		Coord {y: (start.y + end.y)/2, x: (start.x + end.x)/2}
	}
}


// over one step
pub fn mv_action_cost(from_coord: u64, to_coord: u64, use_roads: bool, _map_data: &mut MapData, 
		exf: &HashedMapEx, map_sz: MapSz) -> f32 {
	
	#[cfg(any(feature="opt_debug", debug_assertions))]
	{ // debug
		let w = map_sz.w;
		let i = (from_coord as usize / w) as isize;
		let i2 = (to_coord as usize / w) as isize;
		
		debug_assertq!(i < (map_sz.h as isize));
		debug_assertq!(i2 < (map_sz.h as isize));
		debug_assertq!((i-i2).abs() <= 1); // j could wrap
	}
		
	if !use_roads {return 1.}
	
	let road_present = |coord: u64| {
		if let Some(ex) = exf.get(&coord) {
			if let Some(StructureData {structure_type: StructureType::Road, ..}) = &ex.actual.structure {
				return 1;
			}else{ 
				return 0; 
			}
		}else{
			return 0;
		}
	};
	
	let n_roads = (road_present(from_coord) + road_present(to_coord)) as f32;
	return if n_roads == 0. {1.} else {1./n_roads}
}

// over entire path length
/*pub fn mv_path_cost(path_coords: &Vec<u64>, use_roads: bool, map_data: &mut MapData, 
		exf: &HashedMapEx, map_sz: MapSz) -> f32 {	
	let mut cost = 0.;
	for i in (1..path_coords.len()).rev() {
		cost += mv_action_cost(path_coords[i], path_coords[i-1], use_roads, map_data, exf, map_sz);
	}
	cost
}*/

pub fn est_mv_action_cost(from_coord: u64, to_coord: u64, map_sz: MapSz) -> f32 {
	let w = map_sz.w as isize;
	let c = Coord::frm_ind(from_coord, map_sz);
	let c2 = Coord::frm_ind(to_coord, map_sz);

	let i_diff = c.y - c2.y;
	
	// wrap around map?
	let mut j_diff = (c.x - c2.x).abs();
	let j_diff2 = (j_diff + w).abs();
	let j_diff3 = (j_diff - w).abs();
	if j_diff > j_diff2 {j_diff = j_diff2;}
	if j_diff > j_diff3 {j_diff = j_diff3;}
	
	((i_diff*i_diff + j_diff*j_diff) as f32).sqrt()
}

/////////////////
// note: doesn't actually check if road is accessible from map_coord_start
pub fn find_closest_road(coord_start: u64, map_data: &mut MapData, exf: &HashedMapEx, map_sz: MapSz) -> Option<u64> {	
	const MAX_ROAD_DIST: isize = 10; // in one dimension, ex. can be (i+MAX,j+MAX) away
	
	let c = Coord::frm_ind(coord_start, map_sz);
	debug_assertq!(map_data.get(ZoomInd::Full, coord_start).map_type == MapType::Land,
			"{} {:#?}", Coord::frm_ind(coord_start, map_sz), map_data.get(ZoomInd::Full, coord_start).map_type);
	
	for offset in 1..=MAX_ROAD_DIST {
		macro_rules! chk_road{($i: expr, $j: expr) => (
			if let Some(cur_coord) = map_sz.coord_wrap(c.y + $i, c.x + $j) {
				if let Some(ex) = exf.get(&cur_coord) {
					if let Some(StructureData {structure_type: StructureType::Road, ..}) = &ex.actual.structure {
						return Some(cur_coord);
					}
				} // ex
			} // valid coord
		);}

		for k in -offset..=offset {
			// row scan
			chk_road!(k, offset);
			chk_road!(k, -offset);
			
			// col scan
			chk_road!(offset, k);
			chk_road!(-offset, k);
		}
	}
	None
}

pub fn find_closest_pipe(coord_start: u64, map_data: &mut MapData, exf: &HashedMapEx, map_sz: MapSz) -> Option<u64> {	
	const MAX_PIPE_DIST: isize = 30; // in one dimension, ex. can be (i+MAX,j+MAX) away
	
	let c = Coord::frm_ind(coord_start, map_sz);
	debug_assertq!(map_data.get(ZoomInd::Full, coord_start).map_type == MapType::Land,
			"{} {:#?}", Coord::frm_ind(coord_start, map_sz), map_data.get(ZoomInd::Full, coord_start).map_type);
	
	for offset in 1..=MAX_PIPE_DIST {
		macro_rules! chk_pipe{($i: expr, $j: expr) => (
			if let Some(cur_coord) = map_sz.coord_wrap(c.y + $i, c.x + $j) {
				if let Some(ex) = exf.get(&cur_coord) {
					if !ex.actual.pipe_health.is_none() {
						return Some(cur_coord);
					}
				} // ex
			} // valid coord
		);}
		
		for k in -offset..=offset {
			// row scan
			chk_pipe!(k, offset);
			chk_pipe!(k, -offset);
			
			// col scan
			chk_pipe!(offset, k);
			chk_pipe!(-offset, k);
		}
	}
	None
}

