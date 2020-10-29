use super::*;
use crate::gcore::*;
use crate::units::ActionType;
use crate::disp::menus::ArgOptionUI;
use crate::player::Player;

impl <'f,'bt,'ut,'rt,'st>Disp<'f,'bt,'ut,'rt,'st> {
	pub fn print_map(&mut self, map_data: &mut MapData<'rt>, units: &Vec<Unit>, bldgs: &Vec<Bldg>,
			players: &Vec<Player>, temps: &Templates, exs: &Vec<HashedMapEx>,	gstate: &GameState,
			frame_stats: &mut FrameStats, alt_ind: usize) {
		let zoom_ind = self.state.iface_settings.zoom_ind;
		let map_sz = map_data.map_szs[zoom_ind];
		let map_screen_sz = self.state.iface_settings.map_screen_sz;
		let screen_sz = self.state.iface_settings.screen_sz;
		let player = &players[self.state.iface_settings.cur_player as usize];
		let exf = exs.last().unwrap();
		
		///////////
		// clear bottom part of map
		for row in (screen_sz.h - MAP_ROW_STOP_SZ)..screen_sz.h {
			self.mv(row as i32, 0);
			self.state.renderer.clrtoeol();
		}
		
		// plot map & planned actions
		//	(in screen reader mode the map may not be shown if a window is active)
		if !self.ui_mode.hide_map() {
			let cur_mc = self.state.iface_settings.cursor_to_map_coord(&map_data);
			let pstats = &player.stats;
			let map_loc = self.state.iface_settings.map_loc;
			let cursor_map_ind = self.state.iface_settings.cursor_to_map_ind(&map_data);
			
			// if the unit mv dist is far and the full path hasn't been computed, infer it from
			// the last full path coord computed
			let inferred_path = {
				let get_inferred_path = || {
					if let Some(action) = self.state.iface_settings.add_action_to.first_action() {
						if !action.action_meta_cont.is_none() {
							if let Some(last_path_coord) = action.path_coords.first() {
								return line_to(*last_path_coord, self.state.iface_settings.cursor_to_map_ind(&map_data), map_sz, self.state.iface_settings.screen_sz);
							}
						}
					}
					Vec::new()
				};
				get_inferred_path()
			};
			
			for screen_loc_y in MAP_ROW_START..(screen_sz.h - MAP_ROW_STOP_SZ) {
				self.mv(screen_loc_y as i32, 0);
				let i = (map_loc.y + (screen_loc_y as isize) - (MAP_ROW_START as isize)) as usize;
				
				// at edge of map, print black
				if i >= map_sz.h {
					for _screen_loc_x in 0..map_screen_sz.w { self.addch(' ' as chtype); };
					continue;
				}
				
				for screen_loc_x in 0..map_screen_sz.w {
					let j = ((map_loc.x + (screen_loc_x as isize)) as usize) % map_sz.w;
					let map_coord = (i*map_sz.w + j) as u64;
					
					let mut show_path = || {
						let actions = self.state.iface_settings.add_action_to.actions();
						
						// if in move mode, check if this should be a destination
						if actions.len() != 0 {
							for action in actions.iter() {
								// don't show any paths when the unit is moving with the cursor
								if let ActionType::MvWithCursor = action.action_type {
									return false;
								}
								
								if zoom_ind == map_data.max_zoom_ind() {
									if action.path_coords.contains(&map_coord) || inferred_path.contains(&map_coord) {return true;}
									
								}else if let Some(action_meta_cont) = &action.action_meta_cont {
									let map_coord_zoomed_in = Coord::frm_ind(map_coord, map_sz)
														  .to_zoom(zoom_ind, map_data.max_zoom_ind(), &map_data.map_szs)
														  .to_ind(*map_data.map_szs.last().unwrap()) as u64;
									
									//printlnq!("zoomed out path, {} {}", map_coord_zoomed_in, action_meta_cont.checkpoint_path_coords[0]);
									if action_meta_cont.checkpoint_path_coords.contains(&map_coord_zoomed_in) {return true;}
								}
							}
						
						// is the cursor on a unit?
						// and if so, is the unit the current player's and moving?
						}else if let Some(ex) = exs.last().unwrap().get(&cursor_map_ind) {
							if let Some(unit_inds) = &ex.unit_inds {
								if let Some(&unit_ind) = unit_inds.get(self.state.iface_settings.unit_subsel) {
									let u = &units[unit_ind];
									//let u = &units[unit_inds[self.state.iface_settings.unit_subsel]];
									if u.owner_id == self.state.iface_settings.cur_player || self.state.iface_settings.show_actions {
										if let Some(action) = u.action.last() {
											return action.path_coords.contains(&map_coord);
										} // unit is moving at contains cursor location
									} // owned by current player
								}else{
									self.state.iface_settings.unit_subsel = 0;
								}
							} // unit(s) exist at cursor
						} // ex exists at cursor
						
						false
					};
					
					if show_path() {
						self.addch('.' as chtype);
					}else{
						debug_assertq!(map_coord < (map_data.map_szs[zoom_ind].h*map_data.map_szs[zoom_ind].w) as u64, 
							"map_coord {} {}, {} {}, z: {} max: {}",
							map_coord / map_data.map_szs[zoom_ind].w as u64, map_coord % map_data.map_szs[zoom_ind].w as u64,
							map_data.map_szs[zoom_ind].h, map_data.map_szs[zoom_ind].w, zoom_ind, map_data.max_zoom_ind());
						
						self.plot_land(zoom_ind, map_coord, map_data, units, bldgs, exs, players, &gstate.relations, false, alt_ind);
					}
				
				} // screen_loc_x
			} // screen_loc_y	
			
			//////////////////////////// show planned actions, check if current action is valid and show it
			if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
				// draw build list actions
				if alt_ind % 3 == 0 {
					macro_rules! print_brigade_actions{($brigade_nm: expr) => {
						for action_meta in pstats.brigade_frm_nm($brigade_nm).build_list.iter() {
							action_meta.draw_selection_finalized(units, exf, map_data, map_sz, &mut self.state);
						}
					};};
					
					if let AddActionTo::BrigadeBuildList {brigade_nm, ..} |
						 AddActionTo::AllInBrigade {brigade_nm, ..} = &self.state.iface_settings.add_action_to {
						print_brigade_actions!(brigade_nm);
					}
					
					if let UIMode::BrigadeBuildList(BrigadeBuildListState {brigade_nm, ..}) = &self.ui_mode {
						print_brigade_actions!(brigade_nm);
					}
					
					if let UIMode::BrigadesWindow(BrigadesWindowState {mode, brigade_action}) = &self.ui_mode {
						match brigade_action {
							BrigadeAction::Join {..} | BrigadeAction::ViewBrigades => {
								let mut w = 0; let mut label_txt_opt = None;
								let entries = brigades_list(pstats, &mut w, &mut label_txt_opt, &self.state.local);
								if entries.options.len() != 0 {
									if let ArgOptionUI::BrigadeInd(brigade_ind) = &entries.options[*mode].arg {
										print_brigade_actions!(&pstats.brigades[*brigade_ind].nm);
									}else{panicq!("unexpected ArgOptionUI");}
								}
							} BrigadeAction::ViewBrigadeUnits {brigade_nm} => {
								print_brigade_actions!(brigade_nm);
							}
						}
					}
				}
				
				/////////////// check if zone placement is valid
				if let Some(ref mut action) = self.state.iface_settings.add_action_to.first_action_mut() {
					// check if placement valid
					match &mut action.action_type {
						//////////////////////////// not yet started drawing zone, check if initial location is valid
						ActionType::WorkerZone {start_coord: None, ref mut valid_placement, ..} => {
							let coord = cur_mc.to_ind(map_sz) as u64;
							*valid_placement = land_clear_ign_units_roads(coord, self.state.iface_settings.cur_player, &map_data.get(ZoomInd::Full, coord), &exf);
							
						//////////////////////////// check if we should show zone, then show it
						} ActionType::WorkerZone {start_coord: Some(start_coord_mi), ref mut valid_placement, ..} => {
							// coords in map coordinates
							let start_coord_mc = Coord::frm_ind(*start_coord_mi, map_sz);
							
							/////////////////////
							// check for valid zone placement
							*valid_placement = {
								let mut valid_placement = true;
								let (start_use, rect_sz) = start_coord_use(start_coord_mc, cur_mc, map_sz);
								
								'row_loop: for row in start_use.y..(start_use.y + rect_sz.h as isize) {
								for col in start_use.x..(start_use.x + rect_sz.w as isize) {
									if let Some(coord) = map_sz.coord_wrap(row, col) {
										if land_clear_ign_units_roads(coord, self.state.iface_settings.cur_player, &map_data.get(ZoomInd::Full, coord), &exf) {
											continue;
										}
									}
									
									valid_placement = false;
									break 'row_loop;
								}}
								valid_placement
							};
						} _ => {}
					}
				}
				
				//////////// draw building, zone drawing, group drawing and "X"
				if let Some(action) = self.state.iface_settings.add_action_to.first_action() {
					// show placement and crosshair
					let crosshair = action.clone().draw_selection(cur_mc, units, exf, map_data, map_sz, &mut self.state);
					self.mv(self.state.iface_settings.cur.y as i32, self.state.iface_settings.cur.x as i32);
					self.addch(crosshair as chtype | COLOR_PAIR(CRED));
				} // show bldg, zone, or group mv, and "X" at cursor for move modes
			}
		}
		
		if self.ui_mode.show_menu() {
			self.print_menus(!players[self.state.iface_settings.cur_player as usize].ptype.is_human());
		}
		
		///////////// show most recent log entry
		// (should come after print_menus() as that clears the first line
		//  and before rside_stats because this is added to the right side text list
		//  and should preferrably be at the of the list)
		{
			for log in gstate.logs.iter().rev() {
				// only show if somewhat recent
				if (log.turn + 30*12*5) <= gstate.turn {break;}
				
				// only show if civ discovered
				if !log.visible(self.state.iface_settings.cur_player as usize, &gstate.relations) {continue;}
				
				let date_txt = self.state.local.date_str(log.turn);
				const DELIM_TXT: &str = ": ";
				
				let log_len = print_log(&log.val, false, players, temps.doctrines, &mut self.state);
				let txt_len = date_txt.len() + log_len + DELIM_TXT.len();
				
				if txt_len < screen_sz.w { // only show if enough space
					let start_col = screen_sz.w - txt_len;
					
					self.mv(0, start_col as i32);
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.attron(COLOR_PAIR(CYELLOW));
					self.state.renderer.addstr(&date_txt);
					self.attroff(COLOR_PAIR(CYELLOW));
					self.state.renderer.addstr(DELIM_TXT);
					print_log(&log.val, true, players, temps.doctrines, &mut self.state);
				}
				break;
			}
		}
		
		//////////////
		// bottom stats, right side stats, sub map
		//	(in screen reader mode these may not be shown if a window is active)
		if !self.ui_mode.hide_map() {
			self.print_bottom_stats(map_data, exs, player, players, units, &temps.bldg_config, bldgs, gstate);
			if !screen_reader_mode() || self.ui_mode.right_side_tabbing() {
				self.print_rside_stats(frame_stats, gstate.turn, bldgs, players, temps, exf, map_data, map_sz);
			}
			self.print_submap(map_data, units, bldgs, exs, players, &gstate.relations, alt_ind);
		}
	}
}

