use crate::map::*;
use crate::movement::*;
//use crate::units::{Quad, square_clear};
use crate::gcore::hashing::HashedMapEx;
use crate::gcore::XorState;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

impl ExploreType {
	// `start_coord` is the original starting coord of the unit, not necessarily where it is now
	pub fn find_square_unexplored<'bt,'ut,'rt,'st>(&self, unit_ind: usize, start_coord: u64, map_data: &mut MapData,
			exs: &Vec<HashedMapEx>, units: &Vec<Unit>, bldgs: &Vec<Bldg>,
			land_discov: &LandDiscov, map_sz: MapSz, is_cur_player: bool,
			rng: &mut XorState) -> Option<ActionMeta<'bt,'ut,'rt,'st>> {
		#[cfg(feature="profile")]
		let _g = Guard::new("find_square_unexplored");
		
		let u = &units[unit_ind];
		
		let search_dist = if !is_cur_player {100} else {100};
		
		let mut action_iface = ActionInterfaceMeta {
			action: ActionMeta::new(ActionType::AutoExplore {start_coord, explore_type: *self}),
			unit_ind: Some(unit_ind),
			max_search_depth: 100,
			start_coord: Coord::frm_ind(u.return_coord(), map_sz),
			movement_type: u.template.movement_type,
			movable_to: &movable_to
		};
		
		match self {
			ExploreType::Random => {
				const STEP_SZ: usize = 50;
				
				macro_rules! chk_undiscov_and_movable_to{($i_off:expr, $j_off:expr) => {
					let coord_chk = Coord {y: action_iface.start_coord.y + $i_off,
								x: action_iface.start_coord.x + $j_off
					};
					
					// valid coord
					if let Some(coord) = map_sz.coord_wrap(coord_chk.y, coord_chk.x) {
						let coord = Coord::frm_ind(coord, map_sz);
						
						// undiscovered
						if !land_discov.map_coord_discovered(coord) {
							action_iface.update_move_search(coord, map_data, exs, 
									MvVars::NonCivil{units, start_owner: u.owner_id, blind_undiscov: None}, bldgs);
										//blind_undiscov: Some(land_discov)}, bldgs);
									
							if action_iface.action.path_coords.len() > 0 {
								return Some(action_iface.action);
							}
						}
					} // valid coord
				};}
				
				macro_rules! search_perimeter_of_offset{($k: expr, $offset:expr) => {
					for ordering in rng.inds(4).iter() {
						match ordering {
							// row scan
							0 => {chk_undiscov_and_movable_to!($k, $offset);}
							1 => {chk_undiscov_and_movable_to!($k, -$offset);}
							
							// col scan
							2 => {chk_undiscov_and_movable_to!($offset, $k);}
							3 => {chk_undiscov_and_movable_to!(-$offset, $k);}
							
							_ => {panicq!("invalid random number");}
						}
					}
				};}
				
				if rng.gen_f32b() < 0.5 {
					for offset in (1..search_dist).rev() {
						if rng.gen_f32b() < 0.5 {
							for k in (-offset..=offset).step_by(STEP_SZ) {
								search_perimeter_of_offset!(k, offset);
							}
						}else{
							for k in (-offset..(offset+1)).step_by(STEP_SZ).rev() {
								search_perimeter_of_offset!(k, offset);
							}
						}
					}
				}else{
					for offset in 1..search_dist {
						if rng.gen_f32b() < 0.5 {
							for k in (-offset..=offset).step_by(STEP_SZ) {
								search_perimeter_of_offset!(k, offset);
							}
						}else{
							for k in (-offset..(offset+1)).step_by(STEP_SZ).rev() {
								search_perimeter_of_offset!(k, offset);
							}
						}
					}
				}
			} ExploreType::SpiralOut => {
				// here we move counter-clockwise along rectangles with sides that are multiples of 
				// STEP_SZ_X & STEP_SZ_Y. if we cannot move along one of the perimeters, we move in a random
				// direction pointed away from the perimeter.
				// if we reach the upper left corner of one of the rectangles, we move to the next perimeter out
				
				const Y_FACTOR: isize = 1;
				
				const UNIT_STEP_SZ_Y: isize = 6;
				const UNIT_STEP_SZ_X: isize = 6;//UNIT_STEP_SZ_Y*Y_FACTOR;
				
				let cur = action_iface.start_coord; // current unit position
				let initial_start_coord = Coord::frm_ind(start_coord, map_sz);
				
				let diff = Coord {y: cur.y - initial_start_coord.y, x: cur.x - initial_start_coord.x};
				let dist_comps = manhattan_dist_components(initial_start_coord, cur, map_sz);
				
				let on_perim_row = (dist_comps.h*Y_FACTOR as usize) >= dist_comps.w;
				let on_perim_col = dist_comps.w >= (dist_comps.h*Y_FACTOR as usize);
				
				macro_rules! chk_movable_to{($i_off:expr, $j_off:expr) => {
					let coord_chk = Coord {y: action_iface.start_coord.y + $i_off,
								x: action_iface.start_coord.x + $j_off
					};
					
					// valid coord
					if let Some(coord) = map_sz.coord_wrap(coord_chk.y, coord_chk.x) {
						let coord = Coord::frm_ind(coord, map_sz);
						
						action_iface.update_move_search(coord, map_data, exs, 
								MvVars::NonCivil{units, start_owner: u.owner_id, blind_undiscov: None}, bldgs);
									//blind_undiscov: Some(land_discov)}, bldgs);
								
						if action_iface.action.path_coords.len() > 0 {
							return Some(action_iface.action);
						}
					}
				};}
				
				// attempt to move along the current perimeter
				if on_perim_col || on_perim_row {
					// on a corner
					if on_perim_col && on_perim_row {//|| (diff.y.abs()*Y_FACTOR) != diff.x.abs() {
						// top right corner
						if diff.y < 0 && diff.x > 0 {
							chk_movable_to!(UNIT_STEP_SZ_Y, 0); // down
						// bottom right corner
						}else if diff.y > 0 && diff.x > 0 {
							chk_movable_to!(0, -UNIT_STEP_SZ_X); // left
						// bottom left corner
						}else if diff.y > 0 && diff.x < 0 {
							chk_movable_to!(-UNIT_STEP_SZ_Y, 0); // up
						}
						
						// top left corner not covered, so a random direction will be chosen
						// (preventing the unit from moving in an identical counterclockwise loop forever)
					
					// left & right lines
					}else if on_perim_col {
						// left
						if diff.x < 0 {
							chk_movable_to!(-UNIT_STEP_SZ_Y, 0); // up
						// right
						}else{
							chk_movable_to!(UNIT_STEP_SZ_Y, 0); // down
						}
						
					// top & bottom lines
					}else if on_perim_row {
						// top
						if diff.y < 0 {
							chk_movable_to!(0, UNIT_STEP_SZ_X); // right
						// bottom
						}else{
							chk_movable_to!(0, -UNIT_STEP_SZ_X); // left
						}
					}
				}
				
				// if the function has not returned, either we failed to move along the perimeter line OR
				// we are not even on a perimeter line
				const N_TRIES: usize = 10;
				let current_dist = manhattan_dist(initial_start_coord, cur, map_sz);
				for _ in 0..N_TRIES {
					//let y_dist = ((cur.y / PERIM_STEP_SZ_Y) + 1)*PERIM_STEP_SZ_Y;
					//let 
					
					let i_off = rng.isize_range(-1,2);
					let j_off = rng.isize_range(-1,2);
					if let Some(coord) = map_sz.coord_wrap(cur.y + i_off, cur.x + j_off) {
						// only check coordinate if it actually moves the unit away from its initial starting position
						if manhattan_dist(initial_start_coord, Coord::frm_ind(coord, map_sz), map_sz) >= current_dist {
							chk_movable_to!(i_off, j_off);
						}
					}
				}
			} ExploreType::N => {panicq!("invalid explore type");}
		}
		None
	}
}

/// find square that a building can be build + a small border around it
///
/// ** search_start need not actually be valid (ex. slightly below or above map)
pub fn find_square_buildable<'bt,'ut,'rt>(search_start: Coord, bldg_template: &BldgTemplate, map_data: &mut MapData, 
		exf: &HashedMapEx, map_sz: MapSz) -> Option<u64> {
	const SEARCH_DIST: isize = 150;
	
	let mut blank_spot = bldg_template.sz;
	blank_spot.w += 1;
	blank_spot.h += 1;
	
	for offset in 1..SEARCH_DIST {
		macro_rules! ret_sq_clr{($i_off:expr, $j_off:expr) => {
			if let Some(start_coord) = map_sz.coord_wrap(search_start.y + $i_off, search_start.x + $j_off) {
				if let Some(_end_coord) = square_clear(start_coord, blank_spot, Quad::Lr, map_data, exf) {
					return Some(start_coord);
				}
			} // valid coord
		};}
		
		for k in -offset..=offset {
			// row scan
			ret_sq_clr!(k, offset);
			ret_sq_clr!(k, -offset);
			
			// col scan
			ret_sq_clr!(offset, k);
			ret_sq_clr!(-offset, k);
		}
	}
	None
}


/*pub struct SquareClear<'bt,'ut,'rt> {
	pub start_coord: u64,
	pub end_coord: u64,
	pub action_meta: ActionMeta<'bt,'ut,'rt,'st>
}

// first check if square is clear of units and everything, then check if the start is traversable to
pub fn find_square_clear_traversable<'bt,'ut,'rt>(search_start: Coord, u: &Unit, unit_ind: usize, blank_spot: ScreenSz, map_data: &mut MapData,
		exf: &HashedMapEx, units: &Vec<Unit>, bldgs: &Vec<Bldg>, owners: &Vec<Owner>) -> Option<SquareClear<'bt,'ut,'rt>> {
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	const SEARCH_DIST: isize = 150;
	
	let mut action_iface = ActionInterfaceMeta {
		action: ActionMeta {
				action_type: ActionType::Mv,
				actions_req: 0.,
				path_coords: Vec::new(),
				final_end_coord: None
		},
		
		unit_ind: Some(unit_ind),
		max_search_depth: 300,
		start_coord: Coord::frm_ind(u.return_coord(), map_sz),
		movement_type: u.template.movement_type,
		blind_undiscov: !owners[units[unit_ind].owner_id as usize].ai, // if not AI, pretend all undiscovered land is movable to
		movable_to: &movable_to
	};

	for offset in 1..SEARCH_DIST {
		macro_rules! chk_quad{($start_coord: expr, $quad:expr) => (
			if let Some(end_coord) = square_clear($start_coord, blank_spot, $quad, map_data, exf) {
				update_move_search(Coord::frm_ind($start_coord, map_sz), map_data, exf, 
						MvVars::NonCivil{units, start_owner: u.owner_id}, bldgs, &mut action_iface, map_sz);
				
				if action_iface.action.path_coords.len() > 0 {
					return Some(SquareClear {
							start_coord: $start_coord, end_coord,
							action_meta: action_iface.action
						});
				}
			}
		);};

		macro_rules! ret_sq_clr{($i_off:expr, $j_off:expr) => {
			if let Some(start_coord) = map_sz.coord_wrap(search_start.y + $i_off, search_start.x + $j_off) {
				chk_quad!(start_coord, Quad::Lr);
				chk_quad!(start_coord, Quad::Ll);
				chk_quad!(start_coord, Quad::Ur);
				chk_quad!(start_coord, Quad::Ul);
				
			} // valid coord
		};};
		
		for k in -offset..=offset {
			// row scan
			ret_sq_clr!(k, offset);
			ret_sq_clr!(k, -offset);
			
			// col scan
			ret_sq_clr!(offset, k);
			ret_sq_clr!(-offset, k);
		}
	}
	None
}*/

