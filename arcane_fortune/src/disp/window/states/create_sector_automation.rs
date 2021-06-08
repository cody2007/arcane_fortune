use super::*;
//use crate::units::*;
use std::convert::TryFrom;

pub struct CreateSectorAutomationState {
	pub mode: usize,
	pub sector_nm: Option<String>,
	pub unit_enter_action: Option<SectorUnitEnterAction>,
	pub idle_action: Option<SectorIdleAction>, // if patrol, dist text is then asked for
		
	pub txt: String,
	pub curs_col: isize
}

impl CreateSectorAutomationState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, map_data: &mut MapData, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		///////////// sector automation: get sector name (step 1)
		if self.sector_nm.is_none() {
			// alert player that sector should be created
			if pstats.sectors.len() == 0 {
				let w = dstate.local.No_map_sectors_found.len() + 4;
				let w_pos = dstate.print_window(ScreenSz{w, h: 2+4, sz:0});
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let mut row = 0;
				macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;};
							     ($final: expr) => {dstate.mv(row + y, x);}}
				
				mvl!();
				dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
				
				mvl!(); mvl!(1);
				dstate.renderer.addstr(&dstate.local.No_map_sectors_found); // update width calculation if this changes
				
			// ask player which sector to use
			}else{
				let mut w = 0;
				let mut label_txt_opt = None;
				let map_sz = *map_data.map_szs.last().unwrap();
				let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
				
				let sectors = sector_list(pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
				
				let txt = dstate.local.In_which_sector_do_you_want_to_automate.clone();
				w = txt.len() + 4;
				let list_pos = dstate.print_list_window(self.mode, txt, sectors, Some(w), label_txt_opt, 0, None);
				dstate.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			}
		
		//////////// sector automation: get action to perform when unit enters sector (step 2)
		}else if self.unit_enter_action.is_none() {
			let l = &dstate.local;
			let unit_enter_actions = [l.Assault_desc.as_str(), l.Defense_desc.as_str(), l.Report_desc.as_str()];
			let category_options = OptionsUI::new(&unit_enter_actions);
			let txt = l.Select_unit_enter_action.clone();
			let len = Some(txt.len()+4);
			let list_pos = dstate.print_list_window(self.mode, txt, category_options, len, None, 0, None);
			dstate.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			
		//////////// sector automation: what to do when the unit is idle (step 3)
		}else if self.idle_action.is_none() {
			let l = &dstate.local;
			let idle_actions = [l.Sentry_desc.as_str(), l.Patrol_desc.as_str()];
			let idle_options = OptionsUI::new(&idle_actions);
			let txt = l.When_not_engaged_what_action.clone();
			let len = Some(txt.len()+4);
			let list_pos = dstate.print_list_window(self.mode, txt, idle_options, len, None, 0, None);
			dstate.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			
		//////////////////// sector automation: get distance to respond to threats (step 4)
		}else{
			let title_txt = dstate.local.At_what_distance.clone();
			let w = min(title_txt.len() + 4, dstate.iface_settings.screen_sz.w);
			let h = 7;
			
			let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			dstate.mv(y,x);
			center_txt(&title_txt, w, Some(COLOR_PAIR(TITLE_COLOR)), &mut dstate.renderer);
			
			// print entered txt
			dstate.mv(y+2,x+1);
			dstate.addstr(&self.txt);
			
			// instructions
			{
				let instructions_w = format!("{}  {}", dstate.buttons.Esc_to_close.print_txt(&dstate.local), dstate.buttons.Confirm.print_txt(&dstate.local)).len() as i32;
				let gap = ((w - instructions_w)/2) as i32;
				dstate.mv(y + 4, x - 1 + gap);
				dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer); dstate.addstr("  ");
				dstate.buttons.Confirm.print(None, &dstate.local, &mut dstate.renderer);
			}
			
			// mv to cursor location
			dstate.mv(y + 2, x + 1 + self.curs_col as i32);
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, pstats: &Stats, units: &mut Vec<Unit>, map_data: &mut MapData,
			exf: &HashedMapEx, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(unit_inds) = dstate.iface_settings.unit_inds_frm_sel(pstats, units, map_data, exf) {
			/////////// sector automation, select sector (step 1)
			if self.sector_nm.is_none() {
				let cursor_coord = dstate.iface_settings.cursor_to_map_coord(map_data); // immutable borrow
				let mut w = 0;
				let mut label_txt_opt = None;
				let map_sz = *map_data.map_szs.last().unwrap();
				let entries = sector_list(pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
				let entries_present = entries.options.len() > 0;
				
				macro_rules! enter_action{($mode: expr) => {
					self.sector_nm = Some(pstats.sectors[$mode].nm.clone());
					self.mode = 0;
				};}
				if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	enter_action!(ind);}
				
				match dstate.key_pressed {
					// down
					k if entries_present && (dstate.kbd.down(k)) => {
						if (self.mode + 1) <= (entries.options.len()-1) {
							self.mode += 1;
						}else{
							self.mode = 0;
						}
					
					// up
					} k if entries_present && (dstate.kbd.up(k)) => {
						if self.mode > 0 {
							self.mode -= 1;
						}else{
							self.mode = entries.options.len() - 1;
						}
						
					// enter
					} k if entries_present && k == dstate.kbd.enter => {
						enter_action!(self.mode);
					} _ => {}
				}
			/////////// sector automation, select unit entry action (step 2) (i.e., when a foreign unit enters the sector)
			}else if !self.sector_nm.is_none() && self.unit_enter_action.is_none() {
				let l = &dstate.local;
				let unit_enter_actions = [l.Assault_desc.as_str(), l.Defense_desc.as_str(), l.Report_desc.as_str()];
				let options = OptionsUI::new(&unit_enter_actions);
				
				let n_options = SectorUnitEnterAction::N as usize;
				
				macro_rules! progress_ui_state {() => {
					self.unit_enter_action = Some(SectorUnitEnterAction::from(self.mode));
					self.mode = 0;
					return UIModeControl::UnChgd;
				};}
				
				// shortcut key pressed
				for (option_ind, option) in options.options.iter().enumerate() {
					// match found
					if option.key == Some(dstate.key_pressed as u8 as char) {
						self.mode = option_ind;
						progress_ui_state!();
					} // match found
				} // loop over shortcuts
				
				if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {
					self.mode = ind;
					progress_ui_state!();
				}
				
				match dstate.key_pressed {
					// down
					k if dstate.kbd.down(k) => {
						if (self.mode + 1) <= (n_options-1) {
							self.mode += 1;
						}else{
							self.mode = 0;
						}
					
					// up
					} k if dstate.kbd.up(k) => {
						if self.mode > 0 {
							self.mode -= 1;
						}else{
							self.mode = n_options - 1;
						}
						
					// enter
					} k if k == dstate.kbd.enter => {
						progress_ui_state!();
					} _ => {}
				}
			/////////// sector automation: what to do when the unit is idle (step 3)
			}else if !self.sector_nm.is_none() && !self.unit_enter_action.is_none() && self.idle_action.is_none() {
				let l = &dstate.local;
				let idle_actions = [l.Sentry_desc.as_str(), l.Patrol_desc.as_str()];
				let idle_options = OptionsUI::new(&idle_actions);
				let n_options = idle_actions.len();
				
				macro_rules! progress_ui_state {($mode:expr) => {
					if $mode == 0 {
						for unit_ind in unit_inds {
							let u = &mut units[unit_ind];
							u.actions_used = None;
							u.action.push(ActionMeta::new(ActionType::SectorAutomation {
								unit_enter_action: self.unit_enter_action.unwrap(),
								idle_action: SectorIdleAction::Sentry,
								sector_nm: self.sector_nm.as_ref().unwrap().clone()
							}));
						}
						dstate.iface_settings.add_action_to = AddActionTo::None;
						dstate.iface_settings.update_all_player_pieces_mvd_flag(units);
						return UIModeControl::Closed;
					
					// progress UI state to ask for the player for `dist_monitor`
					}else{
						self.idle_action = Some(SectorIdleAction::Patrol {
							dist_monitor: 0,
							perim_coord_ind: 0,
							perim_coord_turn_computed: 0
						});
						dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
					}
					return UIModeControl::UnChgd;
				};}
				
				// shortcut key pressed
				for (option_ind, option) in idle_options.options.iter().enumerate() {
					// match found
					if option.key == Some(dstate.key_pressed as u8 as char) {
						progress_ui_state!(option_ind);
					}
				}
				if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	progress_ui_state!(ind);}
				
				match dstate.key_pressed {
					// down
					k if dstate.kbd.down(k) => {
						if (self.mode + 1) <= (n_options-1) {
							self.mode += 1;
						}else{
							self.mode = 0;
						}
					
					// up
					} k if dstate.kbd.up(k) => {
						if self.mode > 0 {
							self.mode -= 1;
						}else{
							self.mode = n_options - 1;
						}
						
					// enter
					} k if k == dstate.kbd.enter => {
						let mode = self.mode;
						progress_ui_state!(mode);
					} _ => {}
				}
			/////////// sector automation: get distance away from sector unit should respond to a threat (step 4)
			}else if !self.sector_nm.is_none() && !self.unit_enter_action.is_none() {
				if let Some(SectorIdleAction::Patrol {..}) = self.idle_action {
					if dstate.buttons.Confirm.activated(dstate.key_pressed, &dstate.mouse_event) {
						if self.txt.len() > 0 {
							if let Result::Ok(dist_monitor) = self.txt.parse() {
								for unit_ind in unit_inds {
									units[unit_ind].action.push(ActionMeta::new(ActionType::SectorAutomation {
										unit_enter_action: self.unit_enter_action.unwrap(),
										idle_action: SectorIdleAction::Patrol {
											dist_monitor,
											perim_coord_ind: 0,
											perim_coord_turn_computed: 0
										},
										sector_nm: self.sector_nm.as_ref().unwrap().clone()
									}));
								}
								dstate.iface_settings.add_action_to = AddActionTo::None;
								dstate.iface_settings.update_all_player_pieces_mvd_flag(units);
							}
							return UIModeControl::Closed;
						}
					}
					
					match dstate.key_pressed {
						KEY_LEFT => {if self.curs_col != 0 {self.curs_col -= 1;}}
						KEY_RIGHT => {
							if self.curs_col < (self.txt.len() as isize) {
								self.curs_col += 1;
							}
						}
						
						KEY_HOME | KEY_UP => {self.curs_col = 0;}
						
						// end key
						KEY_DOWN | 0x166 | 0602 => {self.curs_col = self.txt.len() as isize;}
						
						// backspace
						KEY_BACKSPACE | 127 | 0x8  => {
							if self.curs_col != 0 {
								self.curs_col -= 1;
								self.txt.remove(self.curs_col as usize);
							}
						}
						
						// delete
						KEY_DC => {
							if self.curs_col != self.txt.len() as isize {
								self.txt.remove(self.curs_col as usize);
							}
						}
						_ => { // insert character
							if self.txt.len() < (min(MAX_SAVE_AS_W, dstate.iface_settings.screen_sz.w)-5) {
								if let Result::Ok(c) = u8::try_from(dstate.key_pressed) {
									if let Result::Ok(ch) = char::try_from(c) {
										if "0123456789".contains(ch) {
											self.txt.insert(self.curs_col as usize, ch);
											self.curs_col += 1;
										}
									}
								}
							}
						}
					}
				}
			}
			return UIModeControl::UnChgd;
		}
		UIModeControl::Closed
	}
}
