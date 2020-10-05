use crate::disp_lib::*;
use crate::map::*;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::keyboard::KeyboardMap;

use super::*;
use super::vars::*;
use super::color::*;

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn print_submap(&self, disp_chars: &DispChars, map_data: &mut MapData, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exs: &Vec<HashedMapEx>, zone_exs_owners: &Vec<HashedMapZoneEx>,
			pstats: &Stats, owners: &Vec<Owner>, alt_ind: usize, kbd: &KeyboardMap,
			l: &Localization, buttons: &mut Buttons, d: &mut DispState){
		////////// print bounding box
		macro_rules! bounding_box{($map_sz: expr) => {
			// -
			d.addch(disp_chars.ulcorner_char);
			for _ in 0..$map_sz.w {d.addch(disp_chars.hline_char as chtype);}
			d.addch(disp_chars.urcorner_char);
			let start_ln = cursor_pos(d).y;
			
			// |
			for i in 1..=$map_sz.h as isize {
				let r = (start_ln + i) as i32;
				d.mv(r, 0);
				d.addch(disp_chars.vline_char as chtype);
				d.mv(r, $map_sz.w as i32 + 1);
				d.addch(disp_chars.vline_char as chtype);
			}
			
			// -
			d.mv((start_ln + 1 + $map_sz.h as isize) as i32, 0);
			d.addch(disp_chars.llcorner_char);
			for _ in 0..$map_sz.w {d.addch(disp_chars.hline_char as chtype);}
			d.addch(disp_chars.lrcorner_char as chtype);
		};};
		
		macro_rules! print_submap_map{($submap_zoom_ind:expr) => {
			let (map_cursor_frac, map_view_start_frac, map_view_end_frac) = {
				let map_sz = map_data.map_szs[ZOOM_IND_ROOT];
				
				// loc on global map
				let z = map_data.zoom_spacing(self.zoom_ind);
				
				let div = ((map_sz.w as f32) / z) as isize;
				let d_f32 = div as f32;
				
				let map_cursor_frac = ScreenFrac {
					y: z * ((self.map_loc.y + self.cur.y - (MAP_ROW_START as isize)) as f32) / (map_sz.h as f32),
					x: (((self.map_loc.x + self.cur.x) % div) as f32) / d_f32
				};
				
				let mut map_view_start_frac = ScreenFrac {
					y: z * (self.map_loc.y as f32) / (map_sz.h as f32),
					x: ((self.map_loc.x % div) as f32) / d_f32
				};
				
				let mut map_view_end_frac = ScreenFrac {
					y: z * ((self.map_loc.y + ((self.screen_sz.h as isize) - (MAP_ROW_STOP_SZ as isize) - (MAP_ROW_START as isize))) as f32) / (map_sz.h as f32),
					x: (((self.map_loc.x + (self.screen_sz.w as isize) - (MAP_COL_STOP_SZ as isize)) % div) as f32) / d_f32
				};
				
				if ((self.screen_sz.w as isize) - (MAP_COL_STOP_SZ as isize)) >= div { // entire screen in view
					map_view_start_frac.x = 0.;
					map_view_end_frac.x = 1.;
				}
				(map_cursor_frac, map_view_start_frac, map_view_end_frac)
			};
			
			let submap_sz = map_data.map_szs[$submap_zoom_ind];
			
			///// convert fracs to offsets
			let (map_cursor_loc, map_view_start, map_view_end) = {
				let smh = (submap_sz.h - 1) as f32;
				let smw = (submap_sz.w - 1) as f32;
				
				let map_cursor_loc = Coord {
					y: (smh * map_cursor_frac.y).round() as isize,
					x: (smw * map_cursor_frac.x).round() as isize
				};
				
				let map_view_start = Coord {
					y: (smh * map_view_start_frac.y).round() as isize,
					x: (smw * map_view_start_frac.x).round() as isize
				};
				
				let map_view_end = Coord {
					y: (smh * map_view_end_frac.y).round() as isize,
					x: (smw * map_view_end_frac.x).round() as isize
				};
				
				(map_cursor_loc, map_view_start, map_view_end)
			};
			
			////// print submap
			let screen_start_sub_map = Coord {
				y: (self.screen_sz.h - submap_sz.h - 1) as isize,
				x: 1
			};
			
			for sub_map_y in 0..submap_sz.h as isize {
			for sub_map_x in 0..submap_sz.w as isize {
				d.mv((screen_start_sub_map.y + sub_map_y) as i32, (screen_start_sub_map.x + sub_map_x) as i32);
				
				// cursor not here
				if map_cursor_loc.y != sub_map_y || map_cursor_loc.x != sub_map_x {
					let sel = { // currently in view (show in red):
						let x_in_bounds = if map_view_start.x > map_view_end.x { // view wraps around sub map
							sub_map_x <= map_view_end.x || sub_map_x >= map_view_start.x
						}else{
							sub_map_x >= map_view_start.x && sub_map_x <= map_view_end.x
						};
						
						sub_map_y >= map_view_start.y && sub_map_y <= map_view_end.y && x_in_bounds
					};
					let map_coord = sub_map_y*submap_sz.w as isize + sub_map_x;
					self.plot_land($submap_zoom_ind, map_coord as u64, map_data, units, bldgs, exs, zone_exs_owners, pstats, owners, disp_chars, sel, alt_ind, d);
					
				// show cursor
				}else{d.addch((disp_chars.land_char as chtype) | COLOR_PAIR(CRED));}
			}}
		};};
		
		if !self.show_expanded_submap {
			///////// text line above sub-map
			{
				d.mv((self.screen_sz.h - MAP_ROW_STOP_SZ) as i32, 0);

				let coord_txt = format!("Z: {:.1}% (", 100.* map_data.map_szs[self.zoom_ind].w as f32 / map_data.map_szs.last().unwrap().w as f32);
							
				let mut tile_str = String::from(" = ");
				{
					let tile_len = METERS_PER_TILE*(map_data.map_szs.last().unwrap().w as f32) / (map_data.map_szs[self.zoom_ind].w as f32);
					
					let tile_str2 = if tile_len < 1000. {
						format!("{:.1} m", tile_len)
					}else{
						format!("{:.1} km", tile_len/1000.)
					};
					tile_str.push_str(&tile_str2);
				}
				
				let show_abrev_zoom_out = kbd.zoom_out == 'o' as i32 || kbd.zoom_out == 'O' as i32;
				let show_abrev_zoom_in = kbd.zoom_in == 'i' as i32 || kbd.zoom_in == 'I' as i32;
				
				let out_len = if show_abrev_zoom_out {
					"out".len()
				}else{
					key_txt(kbd.zoom_out, l).len()
				};
				
				let in_len = if show_abrev_zoom_in {
					"in".len()
				}else{
					key_txt(kbd.zoom_in, l).len()
				};
				
				let zoom_len = if self.zoom_ind == map_data.max_zoom_ind() {
					out_len + ")".len()
				}else if self.zoom_ind == 1 {
					in_len + ")".len()
				}else{
					in_len + " | ".len() + out_len + " )".len()
				};
				let txt_len = coord_txt.len() + zoom_len + tile_str.len() + 2;
				let pad = if txt_len < (SUB_MAP_WIDTH + 1) {
					(SUB_MAP_WIDTH + 1 - txt_len)/2
				}else{
					0
				};
				for _ in 0..pad { d.addch(' ' as chtype); }
				d.addstr(&coord_txt);
					
				if self.zoom_ind != map_data.max_zoom_ind() {
					if show_abrev_zoom_in {
						self.print_key(kbd.zoom_in, l, d);
						d.addch('n');
					}else{
						self.print_key(kbd.zoom_in, l, d)
					}
					d.addch(' ');
				}
				
				if self.zoom_ind != map_data.max_zoom_ind() && self.zoom_ind != 1 {
					d.addch(disp_chars.vline_char);
					d.addch(' ' as chtype);
				}
				if self.zoom_ind != 1 {
					if show_abrev_zoom_out {
						self.print_key(kbd.zoom_out, l, d);
						d.addstr("ut)");
					}else{
						self.print_key(kbd.zoom_out, l, d);
						d.addch(')');
					}
				}
				if self.zoom_ind == 1 { d.addstr(")"); }
				
				//// print scale of one tile
				d.addstr("  ");
				d.addch(disp_chars.land_char as chtype | COLOR_PAIR(CGREEN));
				d.addstr(&tile_str);
				d.mv((self.screen_sz.h - MAP_ROW_STOP_SZ) as i32 + 1, 0);
			}
			
			bounding_box!(ScreenSz {h: SUB_MAP_HEIGHT, w: SUB_MAP_WIDTH, sz: 0});
			print_submap_map!(ZOOM_IND_SUBMAP);
			
			// print button
			{
				let button = &mut buttons.show_expanded_submap;
				d.mv((self.screen_sz.h-1) as i32, 
					((SUB_MAP_WIDTH+2 - button.print_txt(l).len()-2)/2) as i32);
				d.addch(' ');
				button.print(Some(self), l, d);
				d.addch(' ');
			}
			
		}else if self.show_expanded_submap {
			let map_sz = map_data.map_szs[ZOOM_IND_EXPANDED_SUBMAP];
			let box_start_row = (self.screen_sz.h - map_sz.h-2) as i32;
			d.mv(box_start_row, 0);
			bounding_box!(ScreenSz {h: map_sz.h, w: map_sz.w, sz: 0});
			print_submap_map!(ZOOM_IND_EXPANDED_SUBMAP);
			
			// print button
			{
				let button = &mut buttons.hide_submap;
				d.mv(box_start_row, 
					((map_sz.w+2 - button.print_txt(l).len()-2)/2) as i32);
				d.addch(' ');
				button.print(Some(self), l, d);
				d.addch(' ');
			}

		}
	}
}

