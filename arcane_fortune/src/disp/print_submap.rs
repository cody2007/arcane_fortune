use crate::renderer::*;
use crate::map::*;
use crate::gcore::hashing::*;
use crate::player::*;

use super::*;
use super::vars::*;
use super::color::*;

impl Disp<'_,'_,'_,'_,'_,'_> {
	pub fn print_submap(&mut self, map_data: &mut MapData, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exs: &Vec<HashedMapEx>, players: &Vec<Player>, 
			gstate: &GameState, alt_ind: usize) {
		////////// print bounding box
		macro_rules! bounding_box{($map_sz: expr) => {
			// -
			self.addch(self.state.chars.ulcorner_char);
			for _ in 0..$map_sz.w {self.addch(self.state.chars.hline_char as chtype);}
			self.addch(self.state.chars.urcorner_char);
			let start_ln = cursor_pos(&mut self.state.renderer).y;
			
			// |
			for i in 1..=$map_sz.h as isize {
				let r = (start_ln + i) as i32;
				self.mv(r, 0);
				self.addch(self.state.chars.vline_char as chtype);
				self.mv(r, $map_sz.w as i32 + 1);
				self.addch(self.state.chars.vline_char as chtype);
			}
			
			// -
			self.mv((start_ln + 1 + $map_sz.h as isize) as i32, 0);
			self.addch(self.state.chars.llcorner_char);
			for _ in 0..$map_sz.w {self.addch(self.state.chars.hline_char as chtype);}
			self.addch(self.state.chars.lrcorner_char as chtype);
		};};
		
		macro_rules! print_submap_map{($submap_zoom_ind:expr) => {
			let (map_cursor_frac, map_view_start_frac, map_view_end_frac) = {
				let map_sz = map_data.map_szs[ZOOM_IND_ROOT];
				
				// loc on global map
				let z = map_data.zoom_spacing(self.state.iface_settings.zoom_ind);
				
				let div = ((map_sz.w as f32) / z) as isize;
				let d_f32 = div as f32;
				
				let map_cursor_frac = ScreenFrac {
					y: z * ((self.state.iface_settings.map_loc.y + self.state.iface_settings.cur.y - (MAP_ROW_START as isize)) as f32) / (map_sz.h as f32),
					x: (((self.state.iface_settings.map_loc.x + self.state.iface_settings.cur.x) % div) as f32) / d_f32
				};
				
				let mut map_view_start_frac = ScreenFrac {
					y: z * (self.state.iface_settings.map_loc.y as f32) / (map_sz.h as f32),
					x: ((self.state.iface_settings.map_loc.x % div) as f32) / d_f32
				};
				
				let mut map_view_end_frac = ScreenFrac {
					y: z * ((self.state.iface_settings.map_loc.y + ((self.state.iface_settings.screen_sz.h as isize) - (MAP_ROW_STOP_SZ as isize) - (MAP_ROW_START as isize))) as f32) / (map_sz.h as f32),
					x: (((self.state.iface_settings.map_loc.x + (self.state.iface_settings.screen_sz.w as isize) - (MAP_COL_STOP_SZ as isize)) % div) as f32) / d_f32
				};
				
				if ((self.state.iface_settings.screen_sz.w as isize) - (MAP_COL_STOP_SZ as isize)) >= div { // entire screen in view
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
				y: (self.state.iface_settings.screen_sz.h - submap_sz.h - 1) as isize,
				x: 1
			};
			
			for sub_map_y in 0..submap_sz.h as isize {
			for sub_map_x in 0..submap_sz.w as isize {
				self.mv((screen_start_sub_map.y + sub_map_y) as i32, (screen_start_sub_map.x + sub_map_x) as i32);
				
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
					self.plot_land($submap_zoom_ind, map_coord as u64, map_data, units, bldgs, exs, players, gstate, sel, alt_ind);
					
				// show cursor
				}else{self.addch((self.state.chars.land_char as chtype) | COLOR_PAIR(CRED));}
			}}
		};};
		
		if !self.state.iface_settings.show_expanded_submap.is_open() {
			///////// text line above sub-map
			{
				self.mv((self.state.iface_settings.screen_sz.h - MAP_ROW_STOP_SZ) as i32, 0);

				let coord_txt = format!("Z: {:.1}% (", 100.* map_data.map_szs[self.state.iface_settings.zoom_ind].w as f32 / map_data.map_szs.last().unwrap().w as f32);
							
				let mut tile_str = String::from(" = ");
				{
					let tile_len = METERS_PER_TILE*(map_data.map_szs.last().unwrap().w as f32) / (map_data.map_szs[self.state.iface_settings.zoom_ind].w as f32);
					
					let tile_str2 = if tile_len < 1000. {
						format!("{:.1} m", tile_len)
					}else{
						format!("{:.1} km", tile_len/1000.)
					};
					tile_str.push_str(&tile_str2);
				}
				
				let show_abrev_zoom_out = self.state.kbd.zoom_out == 'o' as i32 || self.state.kbd.zoom_out == 'O' as i32;
				let show_abrev_zoom_in = self.state.kbd.zoom_in == 'i' as i32 || self.state.kbd.zoom_in == 'I' as i32;
				
				let out_len = if show_abrev_zoom_out {
					"out".len()
				}else{
					key_txt(self.state.kbd.zoom_out, &self.state.local).len()
				};
				
				let in_len = if show_abrev_zoom_in {
					"in".len()
				}else{
					key_txt(self.state.kbd.zoom_in, &self.state.local).len()
				};
				
				let zoom_len = if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
					out_len + ")".len()
				}else if self.state.iface_settings.zoom_ind == 1 {
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
				for _ in 0..pad { self.addch(' ' as chtype); }
				self.addstr(&coord_txt);
					
				if self.state.iface_settings.zoom_ind != map_data.max_zoom_ind() {
					if show_abrev_zoom_in {
						self.print_key(self.state.kbd.zoom_in);
						self.addch('n');
					}else{
						self.print_key(self.state.kbd.zoom_in)
					}
					self.addch(' ');
				}
				
				if self.state.iface_settings.zoom_ind != map_data.max_zoom_ind() && self.state.iface_settings.zoom_ind != 1 {
					self.addch(self.state.chars.vline_char);
					self.addch(' ' as chtype);
				}
				if self.state.iface_settings.zoom_ind != 1 {
					if show_abrev_zoom_out {
						self.print_key(self.state.kbd.zoom_out);
						self.addstr("ut)");
					}else{
						self.print_key(self.state.kbd.zoom_out);
						self.addch(')');
					}
				}
				if self.state.iface_settings.zoom_ind == 1 { self.addstr(")"); }
				
				//// print scale of one tile
				self.addstr("  ");
				self.addch(self.state.chars.land_char as chtype | COLOR_PAIR(CGREEN));
				self.addstr(&tile_str);
				self.mv((self.state.iface_settings.screen_sz.h - MAP_ROW_STOP_SZ) as i32 + 1, 0);
			}
			
			bounding_box!(ScreenSz {h: SUB_MAP_HEIGHT, w: SUB_MAP_WIDTH, sz: 0});
			print_submap_map!(ZOOM_IND_SUBMAP);
			
			// print button
			{
				let button = &mut self.state.buttons.show_expanded_submap;
				self.state.renderer.mv((self.state.iface_settings.screen_sz.h-1) as i32, 
					((SUB_MAP_WIDTH+2 - button.print_txt(&self.state.local).len()-2)/2) as i32);
				self.state.renderer.addch(' ');
				button.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
				self.addch(' ');
			}
		
		// show expanded submap
		}else{
			let map_sz = map_data.map_szs[ZOOM_IND_EXPANDED_SUBMAP];
			let box_start_row = (self.state.iface_settings.screen_sz.h - map_sz.h-2) as i32;
			self.mv(box_start_row, 0);
			bounding_box!(ScreenSz {h: map_sz.h, w: map_sz.w, sz: 0});
			print_submap_map!(ZOOM_IND_EXPANDED_SUBMAP);
			
			// print button
			{
				let button = &mut self.state.buttons.hide_submap;
				self.state.renderer.mv(box_start_row, 
					((map_sz.w+2 - button.print_txt(&self.state.local).len()-2)/2) as i32);
				self.state.renderer.addch(' ');
				button.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
				self.addch(' ');
			}

		}
	}
}

