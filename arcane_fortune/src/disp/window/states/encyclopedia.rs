use super::*;

#[derive(PartialEq, Clone, Copy)]
pub enum EncyclopediaCategory {Unit, Bldg, Tech, Doctrine, Resource}

#[derive(PartialEq)]
pub enum EncyclopediaWindowState {
	CategorySelection {mode: usize}, // main screen. selection between Unit, Bldg, or Tech info (`mode` controls this selection)
	ExemplarSelection { // ex. Unit selection screen (when `selection_mode` = true), or info page (when `selection_mode` = false)
		selection_mode: bool,
		category: EncyclopediaCategory, // should not be changed from initial value
		mode: usize, // specific unit (for example)
     }
}

impl EncyclopediaWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		match self {
			///////////////////////////////////////// first page shown when window is first created, prompting for category type
			Self::CategorySelection {mode} => {
				let category_options = OptionsUI::new(ENCYCLOPEDIA_CATEGORY_NMS);
				let list_pos = dstate.print_list_window(*mode, dstate.local.What_would_you_like_to_learn_about.clone(), category_options, Some(dstate.local.What_would_you_like_to_learn_about.len()+4), None, 0, None);
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			
			} Self::ExemplarSelection {selection_mode, category, mode} => {
				//////////////////////////////////////////////////// show names of unit templates etc
				if *selection_mode {
					// set exemplar names
					macro_rules! set_exemplar_nms{($templates: expr, $txt: expr) => {
						let mut exemplar_nms = Vec::with_capacity($templates.len());
						for template in $templates.iter() {exemplar_nms.push(template.nm[dstate.local.lang_ind].as_str());}
						
						let exemplar_options = OptionsUI::new(exemplar_nms.as_slice());
						let list_pos = dstate.print_list_window(*mode, format!("Select a {}:", $txt), exemplar_options, Some(30), None, 0, None);
						dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
					};}
					
					match category {
						EncyclopediaCategory::Unit => {set_exemplar_nms!(temps.units, "unit");}
						EncyclopediaCategory::Bldg => {
							let exemplar_options = encyclopedia_bldg_list(temps.bldgs, &dstate.local);
							let list_pos = dstate.print_list_window(*mode, dstate.local.Select_a_building.clone(), exemplar_options, Some(30), None, 0, None);
							dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
						}
						EncyclopediaCategory::Tech => {set_exemplar_nms!(temps.techs, "technology");}
						EncyclopediaCategory::Doctrine => {set_exemplar_nms!(temps.doctrines, "doctrine");}
						EncyclopediaCategory::Resource => {set_exemplar_nms!(temps.resources, "resource");}
					}
					
				///////////////////////////////////// show info for the selected unit
				}else{
					let exemplar_ind = if *category == EncyclopediaCategory::Bldg {
						let exemplar_options = encyclopedia_bldg_list(temps.bldgs, &dstate.local);
						if let ArgOptionUI::Ind(Some(ind)) = exemplar_options.options[*mode].arg {
							ind
						}else{panicq!("invalid option argument");}
					}else{
						*mode
					};
					dstate.show_exemplar_info(exemplar_ind, *category, OffsetOrPos::Offset(0), None, OffsetOrPos::Offset(0), InfoLevel::Full, temps, &players[dstate.iface_settings.cur_player as usize].stats);
				}
			} // either showing exemplar selection (unit, bldg, tech) or showing specific unit information
		} // match state (EncyclopediaState)
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, temps: &Templates, dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const N_CATEGORIES: usize = 5; // unit, bldg, tech, doctrine, resource
		
		let n_options = |category| {
			match category {
				EncyclopediaCategory::Unit => {temps.units.len()}
				EncyclopediaCategory::Bldg => {temps.bldgs.len()}
				EncyclopediaCategory::Tech => {temps.techs.len()}
				EncyclopediaCategory::Doctrine => {temps.doctrines.len()}
				EncyclopediaCategory::Resource => {temps.resources.len()}
			}
		};
		
		let mode2category = |mode| {
			match mode {
				0 => {EncyclopediaCategory::Unit}
				1 => {EncyclopediaCategory::Bldg}
				2 => {EncyclopediaCategory::Tech}
				3 => {EncyclopediaCategory::Doctrine}
				4 => {EncyclopediaCategory::Resource}
				_ => {panicq!("unknown encyclopedia category index");}
			}
		};
		
		// shortcut key pressed? (they are only on the main category selection screen)
		if let Self::CategorySelection {..} = self {
			let category_options = OptionsUI::new(ENCYCLOPEDIA_CATEGORY_NMS);
			
			for (new_menu_ind, option) in category_options.options.iter().enumerate() {
				// match found
				if option.key == Some(dstate.key_pressed as u8 as char) {
					*self = Self::ExemplarSelection {
							selection_mode: true,
							category: mode2category(new_menu_ind),
							mode: 0
					};
					return UIModeControl::UnChgd;
				} // match found
			} // loop over shortcuts
		}
		
		macro_rules! enter_action {() => {
			match self {
				// progress to exemplar selection (ex. unit templates) from main menu
				Self::CategorySelection {mode} => {
					let category = mode2category(*mode);
					*self = Self::ExemplarSelection {
							selection_mode: true,
							category,
							// skip empty first entry for bldgs:
							mode: if category == EncyclopediaCategory::Bldg {1} else {0}
					};
				// progress to showing info screen for exemplar (ex. specific unit template)
				} Self::ExemplarSelection {ref mut selection_mode, ..} => {
					*selection_mode = false;
				}
			}
			return UIModeControl::UnChgd;
		};}
		
		// list item clicked
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {
			// make sure an inactive menu item wasn't clicked
			if let Self::ExemplarSelection {selection_mode: true, 
					category: EncyclopediaCategory::Bldg, ..} = &self {
				let options = encyclopedia_bldg_list(temps.bldgs, &dstate.local);
				if let ArgOptionUI::Ind(None) = options.options[ind].arg {
					return UIModeControl::UnChgd;
				}
			}
			
			match self {
				Self::CategorySelection {ref mut mode} => {*mode = ind;}
				Self::ExemplarSelection {..} => {}
			}
			enter_action!();
		// go back to previous menu
		}else if dstate.buttons.to_go_back.activated(dstate.key_pressed, &dstate.mouse_event) {
			match self {
				Self::CategorySelection {..} => {}
				Self::ExemplarSelection {ref mut selection_mode, ..} => {
					if *selection_mode == false {
						*selection_mode = true;
					}else{
						*self = Self::CategorySelection {mode: 0};
					}
				}
			}
		}
		
		// non-shortcut keys
		match dstate.key_pressed {
			// down
			k if dstate.kbd.down(k) => {
				match self {
					Self::CategorySelection {ref mut mode} => {
						*mode += 1;
						if *mode >= N_CATEGORIES {*mode = 0;}
					} Self::ExemplarSelection {selection_mode: true, category, ref mut mode} => {
						// skip empty entries
						if *category == EncyclopediaCategory::Bldg {
							let options = encyclopedia_bldg_list(temps.bldgs, &dstate.local);
							*mode += 1;
							// wrap
							if *mode >= options.options.len() {
								*mode = 1;
							// skip empty entry
							}else	if let ArgOptionUI::Ind(None) = options.options[*mode].arg {
								*mode += 1;
							}
							
						// all entries valid
						}else{
							*mode += 1;
							if *mode >= n_options(*category) {*mode = 0;}
						}
					} Self::ExemplarSelection {selection_mode: false, ..} => {}
				}
			
			// up
			} k if dstate.kbd.up(k) => {
				match self {
					Self::CategorySelection {ref mut mode} => {
						if *mode > 0 {*mode -= 1;} else {*mode = N_CATEGORIES-1;}
					} Self::ExemplarSelection {selection_mode: true, category, ref mut mode} => {
						// skip empty entries
						if *category == EncyclopediaCategory::Bldg {
							let options = encyclopedia_bldg_list(temps.bldgs, &dstate.local);
							if *mode > 0 {
								*mode -= 1;
								// skip empty entry
								if let ArgOptionUI::Ind(None) = options.options[*mode].arg {
									if *mode > 0 {
										*mode -= 1
									// wrap
									}else{
										*mode = options.options.len() - 1;
									}
								}
							// wrap
							}else{
								*mode = options.options.len() - 1;
							}
						
						// all entries valid
						}else{
							if *mode > 0 {*mode -= 1;} else {*mode = n_options(*category) - 1;}
						}
					} Self::ExemplarSelection {selection_mode: false, ..} => {}
				}
			// enter (progress forward to next menu)
			} k if k == dstate.kbd.enter => {
				enter_action!();
			} _ => {}
		} // end non-shortcut keys
		
		UIModeControl::UnChgd
	}
}
