use super::*;
// view or add unit to brigade
pub struct BrigadesWindowState {
	pub mode: usize,
	pub brigade_action: BrigadeAction
}

pub enum BrigadeAction {
	Join {unit_ind: usize},
	ViewBrigades,
	ViewBrigadeUnits {brigade_nm: String}
}

impl BrigadesWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, units: &Vec<Unit>, map_data: &mut MapData,
			temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		
		let pstats = &players[dstate.iface_settings.cur_player as usize].stats;
		
		let (entries, txt, n_gap_lines, has_buildable_actions) = match &self.brigade_action {
			BrigadeAction::Join {..} | BrigadeAction::ViewBrigades => {
				(brigades_list(pstats, &mut w, &mut label_txt_opt, &dstate.local),
				 dstate.local.Select_brigade.clone(), 0, false)
			} BrigadeAction::ViewBrigadeUnits {brigade_nm} => {
				let map_sz = *map_data.map_szs.last().unwrap();
				let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
				
				let has_buildable_actions = pstats.brigade_frm_nm(&brigade_nm).has_buildable_actions(units);
				
				let n_gap_lines = 4 + if has_buildable_actions {4} else {0};
				
				(brigade_unit_list(&brigade_nm, pstats, players, units, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local),
				 String::new(), n_gap_lines, has_buildable_actions)
			}
		};
		
		let list_pos = dstate.print_list_window(self.mode, txt, entries.clone(), Some(w), label_txt_opt, n_gap_lines, None);
		
		// print instructions & show info box
		if entries.options.len() > 0 {
			match entries.options[self.mode].arg {
				////////// case where brigade_action = Join or ViewBrigades
				ArgOptionUI::BrigadeInd(brigade_ind) => {
					if let Some(brigade) = pstats.brigades.get(brigade_ind) {
						let top_right = list_pos.top_right;
						dstate.show_brigade_units(brigade, Coord {x: top_right.x - 1, y: top_right.y + self.mode as isize + 5}, units);
					}
				///////// case where brigade_action = ViewBrigadeUnits
				} ArgOptionUI::UnitInd(unit_ind) => {
					const OPTION_STR: &str = "   * ";
					
					let top_left = list_pos.top_left;
					let mut roff = top_left.y as i32 + 2;
					macro_rules! mvl{() => {dstate.mv(roff, top_left.x as i32 + 2); roff += 1;};
						($fin: expr) => {dstate.mv(roff, top_left.x as i32 + 2);};};
					
					mvl!();
					
					let brigade_nm = if let BrigadeAction::ViewBrigadeUnits {brigade_nm} = &self.brigade_action {brigade_nm} else {panicq!("invalid brigade action");};
					
					// title
					{
						let title = dstate.local.The_NM_Brigade.replace("[]", &brigade_nm);
						for _ in 0..((w-title.len() - 4)/2) {dstate.addch(' ');}
						
						addstr_c(&title, TITLE_COLOR, &mut dstate.renderer);
						mvl!();
					}
					
					// Build list:   X actions (/: view)
					// Automatic repair behavior: Repair walls in Sector ABC
					if has_buildable_actions {
						mvl!();
						dstate.renderer.addstr(&dstate.local.Build_list);
						
						let brigade = pstats.brigade_frm_nm(&brigade_nm);
						
						let n_brigade_units = brigade.build_list.len();
						
						let action_txt = if n_brigade_units != 1 {&dstate.local.actions} else {&dstate.local.action};
						let txt = format!("{} {} ", n_brigade_units, action_txt);
						
						let txt_added = dstate.local.Build_list.len() + txt.len() + dstate.buttons.view_brigade.print_txt(&dstate.local).len() + 4 + 2;
						if txt_added <= w {
							let gap = w - txt_added;
							for _ in 0..gap {dstate.addch(' ');}
						}
						dstate.renderer.addstr(&txt); dstate.addch('(');
						dstate.buttons.view_brigade.print(None, &dstate.local, &mut dstate.renderer);
						dstate.addch(')'); mvl!();
						
						// auto-repair behavior
						dstate.renderer.addstr(&dstate.local.Automatic_repair_behavior);
						let (txt, button) = if let Some(sector_nm) = &brigade.repair_sector_walls {
							(dstate.local.Repair_damaged_walls.replace("[]", sector_nm), &mut dstate.buttons.clear_brigade_repair)
						}else{(dstate.local.None.clone(), &mut dstate.buttons.change_brigade_repair)};
						
						let txt_added = dstate.local.Automatic_repair_behavior.len() + txt.len() + button.print_txt(&dstate.local).len() + 5;
						if txt_added <= w {
							let gap = w - txt_added;
							for _ in 0..gap{dstate.renderer.addch(' ');}
						}
						dstate.renderer.addstr(&txt); dstate.renderer.addch(' ');
						button.print(None, &dstate.local, &mut dstate.renderer);
						mvl!();
					}
					
					{ // Assign an action to: 
						mvl!(); dstate.renderer.addstr(&dstate.local.Assign_an_action_to);
						
						mvl!(); dstate.renderer.addstr(OPTION_STR);
						dstate.buttons.assign_action_to_all_in_brigade.print(None, &dstate.local, &mut dstate.renderer);
						
						if has_buildable_actions {
							mvl!(); dstate.renderer.addstr(OPTION_STR);
							dstate.buttons.add_action_to_brigade_build_list.print(None, &dstate.local, &mut dstate.renderer);
						}
						
						mvl!(1); dstate.renderer.addstr(OPTION_STR);
						dstate.renderer.addstr(&dstate.local.an_individual_battalion);
					}
					
					// print infobox
					let top_right = list_pos.top_right;
					dstate.show_exemplar_info(units[unit_ind].template.id as usize, EncyclopediaCategory::Unit, OffsetOrPos::Pos(top_right.x as usize - 1), None, OffsetOrPos::Pos(n_gap_lines + top_right.y as usize + self.mode + 4), InfoLevel::AbbrevNoCostNoProdTime, temps, pstats);
				} _ => {panicq!("invalid UI setting");}
			}
		}
		
		dstate.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player>, units: &Vec<Unit>, bldgs: &Vec<Bldg>, map_data: &mut MapData,
			gstate: &GameState, exs: &Vec<HashedMapEx>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		let cur_player = dstate.iface_settings.cur_player as usize;
		let pstats = &players[cur_player].stats;
		let key_pressed = dstate.key_pressed;
		
		let entries = match &self.brigade_action {
			BrigadeAction::Join {..} | BrigadeAction::ViewBrigades => {
				brigades_list(&pstats, &mut w, &mut label_txt_opt, &dstate.local)
			} BrigadeAction::ViewBrigadeUnits {brigade_nm} => {
				brigade_unit_list(&brigade_nm, &pstats, players, units, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local)
			}
		};
		let entries_present = entries.options.len() > 0;
		
		// handle buttons
		if entries_present {
			if let BrigadeAction::ViewBrigadeUnits {brigade_nm} = &self.brigade_action {
				// change brigade repair wall behavior (get the sector to repair walls in)
				if dstate.buttons.change_brigade_repair.activated(key_pressed, &dstate.mouse_event) {
					return UIModeControl::New(UIMode::SectorsWindow(SectorsWindowState {
						mode: 0,
						sector_action: SectorAction::SetBrigadeToRepairWalls(brigade_nm.to_string())
					}));
					
				// clear brigade repair wall behavior
				}else if dstate.buttons.clear_brigade_repair.activated(key_pressed, &dstate.mouse_event) {
					players[cur_player].stats.brigade_frm_nm_mut(brigade_nm).repair_sector_walls = None;
				
				// assign action to all units in brigade
				}else if dstate.buttons.assign_action_to_all_in_brigade.activated(key_pressed, &dstate.mouse_event) {
					dstate.iface_settings.add_action_to = AddActionTo::AllInBrigade {
						brigade_nm: brigade_nm.clone(),
						action_ifaces: None
					};
					return UIModeControl::Closed;
				
				// add action to brigade build list
				}else if dstate.buttons.add_action_to_brigade_build_list.activated(key_pressed, &dstate.mouse_event) && 
					   players[cur_player].stats.brigade_frm_nm(brigade_nm).has_buildable_actions(units) {
					dstate.iface_settings.add_action_to = AddActionTo::BrigadeBuildList {
						brigade_nm: brigade_nm.clone(),
						action: None
					};
					return UIModeControl::Closed;
					
				// view brigade build list
				}else if dstate.buttons.view_brigade.activated(key_pressed, &dstate.mouse_event) {
					return UIModeControl::New(UIMode::BrigadeBuildList(BrigadeBuildListState {
						mode: self.mode,
						brigade_nm: brigade_nm.to_string()
					}));
				}
			}
		}
		
		//////////////////////// handle keys
		macro_rules! enter_action {($mode: expr) => {
			// move cursor to entry
			let coord = match entries.options[$mode].arg {
				ArgOptionUI::BrigadeInd(brigade_ind) => {
					// make sure the brigade exists and is non-empty
					if let Some(brigade) = players[cur_player].stats.brigades.get_mut(brigade_ind) {
						self.mode = 0;
						match self.brigade_action {
							BrigadeAction::ViewBrigades => {
								self.brigade_action = BrigadeAction::ViewBrigadeUnits{brigade_nm: brigade.nm.clone()};
								return UIModeControl::UnChgd;
							} BrigadeAction::Join {unit_ind} => {
								debug_assertq!(!brigade.unit_inds.contains(&unit_ind));
								brigade.unit_inds.push(unit_ind);
								return UIModeControl::Closed;
							} BrigadeAction::ViewBrigadeUnits {..} => {panicq!("list should supply unit inds, not brigade inds");}
						}
					}else{return UIModeControl::Closed;}
				}
				_ => {panicq!("unit inventory list argument option not properly set");}
			};
			
			return UIModeControl::CloseAndGoTo(coord);
		};};
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	enter_action!(ind);}
		
		let kbd = &dstate.kbd;
		match dstate.key_pressed {
			k if entries_present && (kbd.down(k)) => {
				if (self.mode + 1) <= (entries.options.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if entries_present && (kbd.up(k)) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = entries.options.len() - 1;
				}
				
			// enter
			} k if entries_present && (k == kbd.enter) => {
				enter_action!(self.mode);
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}
