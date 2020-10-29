use super::*;
use std::cmp::max;
use crate::units::{ActionMeta, ActionType};
use crate::zones::new_zone_w_roads;

impl ActionMeta<'_,'_,'_,'_> {
	// use `cur_mc` the cursor map coordinate
	pub fn draw_selection(&self, cur_mc: Coord, units: &Vec<Unit>, exf: &HashedMapEx, map_data: &mut MapData, map_sz: MapSz, dstate: &mut DispState) -> char {
		let mut crosshair = 'X';
		
		match &self.action_type {
			/////////////////////////////////// build mode, show bldg at cursor
			ActionType::WorkerBuildBldg {template: bt, valid_placement: true, ..} => {
				let bldg_coord = Coord {y: dstate.iface_settings.cur.y, x: dstate.iface_settings.cur.x + 1};
				
				// prevent plotting off map
				let stop_col = if (bldg_coord.x + bt.sz.w as isize) < dstate.iface_settings.map_screen_sz.w as isize {
						bldg_coord.x + bt.sz.w as isize
					}else{
						dstate.iface_settings.map_screen_sz.w as isize
					};
				
				//// print bldg
				for row in dstate.iface_settings.cur.y..(dstate.iface_settings.cur.y + bt.sz.h as isize) {
					dstate.mv(row as i32, bldg_coord.x as i32);
					for col in (dstate.iface_settings.cur.x + 1)..stop_col {
						
						print_bldg_char(Coord {y: row, x: col}, bldg_coord, bt, &None, map_sz, dstate);
						
					} // j
				} // i
			
			//////////////////////////// show zone
			} ActionType::WorkerZone {zone_type, start_coord: Some(start_coord_mi), valid_placement: true, ..} => {
				crosshair = 'x';
				
				// coords in map coordinates
				let start_coord_mc = Coord::frm_ind(*start_coord_mi, map_sz);
				
				let roads = new_zone_w_roads(*start_coord_mi, cur_mc.to_ind(map_sz) as u64, map_sz, &exf);
				
				let start_coord_sc = start_coord_mc.to_screen_coords_unchecked(dstate.iface_settings.map_loc);
				let (start_use, rect_sz) = start_coord_use(start_coord_sc, dstate.iface_settings.cur, map_sz); // in screen coordinates
				let h = rect_sz.h as isize;
				let w = rect_sz.w as isize;
				
				let row_offset = if start_use.y < MAP_ROW_START as isize {
					MAP_ROW_START as isize - start_use.y
				}else{0};
				
				let col_max = if (start_use.x + w) >= dstate.iface_settings.map_screen_sz.w as isize {
					dstate.iface_settings.map_screen_sz.w as isize - start_use.x
				}else{w};
				
				let col_offset = if start_use.x < 0 {
					-start_use.x
				}else{0};
				
				for row in row_offset..h {
					dstate.mv((start_use.y + row) as i32, max(0, start_use.x) as i32);
					for col in col_offset..col_max {
						
						// only show over land (show units as a priority)
						//let char_shown = inch() & A_CHARTEXT();
						//if char_shown == (disp_chars.land_char as chtype & A_CHARTEXT()) {
							if roads[(row*w + col) as usize] {
								dstate.addch(ROAD_CHAR);
							}else{
								dstate.plot_zone(*zone_type);
							}
						//}else{
						//	dstate.mv(row as i32, (col+1) as i32);
						//}
					} // col
				} // row
				
				// show start of zone at cursor
				if start_coord_sc.y >= MAP_ROW_START as isize && start_coord_sc.y < dstate.iface_settings.map_screen_sz.w as isize &&
				   start_coord_sc.x >= 0 && start_coord_sc.x < dstate.iface_settings.map_screen_sz.w as isize {
					dstate.mv(start_coord_sc.y as i32, start_coord_sc.x as i32);
					dstate.addch('X' as chtype | COLOR_PAIR(CRED));
				}
			
			////////////////////////////
			// show group rectangle if start coord chosen
			} ActionType::GroupMv {start_coord: Some(start_coord_mi), end_coord} |
			  ActionType::WorkerRmZonesAndBldgs {start_coord: Some(start_coord_mi), end_coord} |
			  ActionType::BrigadeCreation {start_coord: Some(start_coord_mi), end_coord, ..} |
			  ActionType::SectorCreation {start_coord: Some(start_coord_mi), end_coord, ..} => {
				crosshair = 'x';
				
				let start_coord_mc = Coord::frm_ind(*start_coord_mi, map_sz); // in map coords
				
				macro_rules! draw_rectangle{($start_use_sc: expr, $rect_sz: expr, $col_loop_nm: lifetime) => {
					let h = $rect_sz.h as isize;
					let w = $rect_sz.w as isize;
					
					let start_use_mc = dstate.iface_settings.screen_coord_to_map_coord($start_use_sc, map_data);
					
					for (row_sc, i_off) in ($start_use_sc.y..($start_use_sc.y + h)).zip(0..h) {
						dstate.mv(row_sc as i32, $start_use_sc.x as i32);
						$col_loop_nm: for j_off in 0..w {
							if let Some(map_coord) = map_sz.coord_wrap(start_use_mc.y + i_off, start_use_mc.x + j_off) {
								// if player has a unit on this tile, show the unit
								if let Some(ex) = exf.get(&map_coord) {
									if let Some(unit_inds) = &ex.unit_inds {
										for unit_ind in unit_inds.iter() {
											if units[*unit_ind].owner_id == dstate.iface_settings.cur_player {
												dstate.mv(row_sc as i32, ($start_use_sc.x + j_off + 1) as i32);
												continue $col_loop_nm;
											}
										}
									}
								}
								
								// shown if no unit present
								dstate.addch('*' as chtype);
							}
						} // col
					} // row
				};};
				
				if let Some(start_coord_sc) = start_coord_mc.to_screen_coords(dstate.iface_settings.map_loc, dstate.iface_settings.map_screen_sz) {
					//////// rectangle selection finished, use stored rectangle end coord
					if let Some(end_coord_mi) = end_coord {
						let end_coord_mc = Coord::frm_ind(*end_coord_mi, map_sz); // in map coords
						
						if let Some(end_coord_sc) = end_coord_mc.to_screen_coords(dstate.iface_settings.map_loc, dstate.iface_settings.map_screen_sz) {
							let (start_use_sc, rect_sz) = start_coord_use(start_coord_sc, end_coord_sc, map_sz); // in screen coordinates
							
							draw_rectangle!(start_use_sc, rect_sz, 'col_loop_finished);
						}
					/////// use cursor as rectangle end coord
					}else{
						let (start_use_sc, rect_sz) = start_coord_use(start_coord_sc, dstate.iface_settings.cur, map_sz); // in screen coordinates
						
						draw_rectangle!(start_use_sc, rect_sz, 'col_loop_cursor);
						
						dstate.mv(start_coord_sc.y as i32, start_coord_sc.x as i32);
						dstate.addch('X' as chtype | COLOR_PAIR(CRED));
					}
				}
			} _ => {}
		}
		crosshair
	}
	
	// for example, actions from the brigade build list
	pub fn draw_selection_finalized(&self, units: &Vec<Unit>, exf: &HashedMapEx, map_data: &mut MapData, map_sz: MapSz, dstate: &mut DispState) {
		match &self.action_type {
			/////////////////////////////////// build mode, show bldg at cursor
			ActionType::WorkerBuildBldg {template: bt, ..} => {
				let bldg_coord = {
					let mut bldg_coord = Coord::frm_ind(*self.path_coords.last().unwrap(), map_sz);
					bldg_coord.x += 1;
					bldg_coord.to_screen_coords_unchecked(dstate.iface_settings.map_loc)
				};
				
				// prevent plotting off map
				let stop_col = if (bldg_coord.x + bt.sz.w as isize) < dstate.iface_settings.map_screen_sz.w as isize {
						bldg_coord.x + bt.sz.w as isize
					}else{
						dstate.iface_settings.map_screen_sz.w as isize
					};
				
				//// print bldg
				for row in bldg_coord.y..(bldg_coord.y + bt.sz.h as isize) {
					if row < 0 || (bldg_coord.x as i32) < 0 {break;}
					
					dstate.mv(row as i32, bldg_coord.x as i32);
					for col in bldg_coord.x..stop_col {
						print_bldg_char(Coord {y: row, x: col}, bldg_coord, bt, &None, map_sz, dstate);
					} // j
				} // i
			
			//////////////////////////// show zone
			} ActionType::WorkerZone {zone_type, start_coord: Some(start_coord_mi), end_coord: Some(end_coord_mi), valid_placement: true, ..} => {
				// coords in map coordinates
				let start_coord_mc = Coord::frm_ind(*start_coord_mi, map_sz);
				let end_coord_mc = Coord::frm_ind(*end_coord_mi, map_sz);
				
				let roads = new_zone_w_roads(*start_coord_mi, *end_coord_mi as u64, map_sz, &exf);
				
				let start_coord_sc = start_coord_mc.to_screen_coords_unchecked(dstate.iface_settings.map_loc);
				let end_coord_sc = end_coord_mc.to_screen_coords_unchecked(dstate.iface_settings.map_loc);
				let (start_use, rect_sz) = start_coord_use(start_coord_sc, end_coord_sc, map_sz); // in screen coordinates
				let h = rect_sz.h as isize;
				let w = rect_sz.w as isize;
				
				let row_offset = if start_use.y < MAP_ROW_START as isize {
					MAP_ROW_START as isize - start_use.y
				}else{0};
				
				let col_max = if (start_use.x + w) >= dstate.iface_settings.map_screen_sz.w as isize {
					dstate.iface_settings.map_screen_sz.w as isize - start_use.x
				}else{w};
				
				let col_offset = if start_use.x < 0 {
					-start_use.x
				}else{0};
				
				for row in row_offset..h {
					dstate.mv((start_use.y + row) as i32, max(0, start_use.x) as i32);
					for col in col_offset..col_max {
						
						// only show over land (show units as a priority)
						//let char_shown = inch() & A_CHARTEXT();
						//if char_shown == (disp_chars.land_char as chtype & A_CHARTEXT()) {
							if roads[(row*w + col) as usize] {
								dstate.addch(ROAD_CHAR);
							}else{
								dstate.plot_zone(*zone_type); 
							}
						//}else{
						//	dstate.mv(row as i32, (col+1) as i32);
						//}
					} // col
				} // row
				
			////////////////////////////
			// show group rectangle if start coord chosen
			} ActionType::GroupMv {start_coord: Some(start_coord_mi), end_coord} |
			  ActionType::BrigadeCreation {start_coord: Some(start_coord_mi), end_coord, ..} |
			  ActionType::SectorCreation {start_coord: Some(start_coord_mi), end_coord, ..} => {
				let start_coord_mc = Coord::frm_ind(*start_coord_mi, map_sz); // in map coords
				
				macro_rules! draw_rectangle{($start_use_sc: expr, $rect_sz: expr, $col_loop_nm: lifetime) => {
					let h = $rect_sz.h as isize;
					let w = $rect_sz.w as isize;
					
					let start_use_mc = dstate.iface_settings.screen_coord_to_map_coord($start_use_sc, map_data);
					
					for (row_sc, i_off) in ($start_use_sc.y..($start_use_sc.y + h)).zip(0..h) {
						dstate.mv(row_sc as i32, $start_use_sc.x as i32);
						$col_loop_nm: for j_off in 0..w {
							if let Some(map_coord) = map_sz.coord_wrap(start_use_mc.y + i_off, start_use_mc.x + j_off) {
								// if player has a unit on this tile, show the unit
								if let Some(ex) = exf.get(&map_coord) {
									if let Some(unit_inds) = &ex.unit_inds {
										for unit_ind in unit_inds.iter() {
											if units[*unit_ind].owner_id == dstate.iface_settings.cur_player {
												dstate.mv(row_sc as i32, ($start_use_sc.x + j_off + 1) as i32);
												continue $col_loop_nm;
											}
										}
									}
								}
								
								// shown if no unit present
								dstate.addch('*' as chtype);
							}
						} // col
					} // row
				};};
				
				if let Some(start_coord_sc) = start_coord_mc.to_screen_coords(dstate.iface_settings.map_loc, dstate.iface_settings.map_screen_sz) {
					//////// rectangle selection finished, use stored rectangle end coord
					if let Some(end_coord_mi) = end_coord {
						let end_coord_mc = Coord::frm_ind(*end_coord_mi, map_sz); // in map coords
						
						if let Some(end_coord_sc) = end_coord_mc.to_screen_coords(dstate.iface_settings.map_loc, dstate.iface_settings.map_screen_sz) {
							let (start_use_sc, rect_sz) = start_coord_use(start_coord_sc, end_coord_sc, map_sz); // in screen coordinates
							
							draw_rectangle!(start_use_sc, rect_sz, 'col_loop_finished);
						}
					/////// use cursor as rectangle end coord
					}else{
						let (start_use_sc, rect_sz) = start_coord_use(start_coord_sc, dstate.iface_settings.cur, map_sz); // in screen coordinates
						
						draw_rectangle!(start_use_sc, rect_sz, 'col_loop_cursor);
					}
				}
			} _ => {}
		}
	}
}

