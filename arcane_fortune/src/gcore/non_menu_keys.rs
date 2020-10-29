use std::collections::VecDeque;
use crate::renderer::*;
use crate::disp::*;
use crate::units::*;
use crate::map::*;
use crate::gcore::*;
use crate::zones::return_zone_coord;
use crate::disp::menus::{ArgOptionUI, FindType};

// check if unit present, it is owned by current player, has actions available, and no other action mode is active
impl <'f,'bt,'ut,'rt,'dt>IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	fn sel_units_owned(&self, pstats: &Stats, units: &Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, exf: &HashedMapEx<'bt,'ut,'rt,'dt>) -> Vec<usize> {
		if !self.add_action_to.first_action().is_none() {return Vec::new();} // areadly performing some other action
		
		// brigade
		if let Some(brigade_nm) = &self.add_action_to.brigade_sel_nm() {
			let brigade = pstats.brigade_frm_nm(brigade_nm);
			return brigade.unit_inds.clone();
		// individual unit
		}else if let Some(unit_ind) = self.unit_ind_frm_cursor(units, map_data, exf) { // checks cur_player owns it
			let u = &units[unit_ind];
			if u.template.nm[0] != RIOTER_NM || self.show_actions {
				if let Some(_) = u.actions_used { // make sure not all actions are used
					return vec![unit_ind];
				}
			}
		}
		Vec::new()
	}
}

// start zoning mode for worker
fn start_zoning<'bt,'ut,'rt,'dt>(unit_inds: &Vec<usize>, zone_type: ZoneType, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>,
		map_data: &mut MapData<'rt>, exf: &HashedMapEx, disp: &mut Disp) {
	// zone if in build list mode or there is a worker selected
	if disp.state.iface_settings.add_action_to.is_build_list() || unit_inds.iter().any(|&ind| units[ind].template.nm[0] == WORKER_NM) {
		let act = ActionType::WorkerZone { valid_placement: false, zone_type, start_coord: None, end_coord: None };
		disp.state.iface_settings.start_build_mv_mode(act, unit_inds, units, map_data);
	
	// set taxes
	}else if let Some(bldg_ind) = disp.state.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf){
		if bldgs[bldg_ind].template.nm[0] == CITY_HALL_NM {
			disp.ui_mode = UIMode::SetTaxes(zone_type);
		}
	}
}

pub fn non_menu_keys<'bt,'ut,'rt,'dt,'f>(map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, frame_stats: &mut FrameStats, disp: &mut Disp<'f,'bt,'ut,'rt,'dt>) {
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	// cursor and view movement
	macro_rules! lupdate{($coord_set: expr, $sign: expr)=> {
		match disp.state.iface_settings.view_mv_mode {
			ViewMvMode::Cursor => disp.linear_update($coord_set, $sign, map_data, exs, units, bldgs, gstate, players, map_sz),
			ViewMvMode::Screen => disp.linear_update_screen($coord_set, $sign, map_data, exs, units, bldgs, gstate, players, map_sz),
			ViewMvMode::N => {panicq!("invalid view setting");}
		}
	}};
	macro_rules! aupdate{($coord_set: expr, $sign: expr)=> {
		match disp.state.iface_settings.view_mv_mode {
			ViewMvMode::Cursor => disp.accel_update($coord_set, ($sign) as f32, map_data, exs, units, bldgs, gstate, players, map_sz),
			ViewMvMode::Screen => disp.accel_update_screen($coord_set, ($sign) as f32, map_data, exs, units, bldgs, gstate, players, map_sz),
			ViewMvMode::N => {panicq!("invalid view setting");}
		}
	}};
	
	macro_rules! end_turn_c{()=>(end_turn(gstate, units, bldgs, temps, disp, map_data, exs, players, frame_stats););};
		
	macro_rules! set_taxes{($inc: expr)=>{
		if let UIMode::SetTaxes(zone_type) = disp.ui_mode {
			if let Some(city_hall_ind_set) = disp.state.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exs.last().unwrap()) {
				if let BldgArgs::PopulationCenter {ref mut tax_rates, ..} = bldgs[city_hall_ind_set].args {
					let t = &mut tax_rates[zone_type as usize];
					let n = (*t as isize) + $inc;
					if n >= 0 && n <= 100 {
						*t = n as u8;
						
						// update taxable income on all bldgs connected to this city hall
						let owner_id = bldgs[city_hall_ind_set].owner_id;
						
						for bldg_ind in 0..bldgs.len() {
							let b = &bldgs[bldg_ind];
							if owner_id != b.owner_id {continue;}
							if let BldgType::Taxable(_) = b.template.bldg_type { // req. that the bldg actually be taxable
								let zone_exs = &mut players[owner_id as usize].zone_exs;
								if let Some(zone_ex) = zone_exs.get(&return_zone_coord(b.coord, map_sz)) {
									if let Dist::Is {bldg_ind: city_hall_ind, ..} = zone_ex.ret_city_hall_dist() {
										if city_hall_ind == city_hall_ind_set {
											let player = &mut players[b.owner_id as usize];
											let new_income = -bldgs[bldg_ind].template.upkeep * 
												return_effective_tax_rate(b.coord, map_data, exs, player, bldgs, temps.doctrines, map_sz, gstate.turn);
											
											bldgs[bldg_ind].set_taxable_upkeep(new_income, &mut player.stats);
										} // city hall used is the one set
									} // has city hall dist set
								}else{ panicq!("taxable bldg should be in taxable zone"); } // bldg is in a zone
							} // bldg is taxable
						} // bldg loop
					} // update taxes
				}else{panicq!("could not get population center bldg args");}
			}else{panicq!("tax ui mode set but no bldg selected");}
		} // in tax-setting UI mode
	};};
	
	let k = disp.state.key_pressed;
	let mut cursor_moved = false;
	
	//// expand/minimize large submap (should be before the section below or else it'll never be called when the submap is expanded)
	if disp.state.buttons.show_expanded_submap.activated(k, &disp.state.mouse_event) || disp.state.buttons.hide_submap.activated(k, &disp.state.mouse_event) {disp.state.iface_settings.show_expanded_submap ^= true;}
	
	let exf = exs.last().unwrap();
	let unit_inds = disp.state.iface_settings.sel_units_owned(&players[disp.state.iface_settings.cur_player as usize].stats, units, map_data, exf);
	
	// mouse
	// 	-dragging (both submap and normal map)
	//	-cursor location updating / move search updating
	if let Some(m_event) = &disp.state.mouse_event {
		let submap_start_row = disp.state.iface_settings.screen_sz.h as i32 - MAP_ROW_STOP_SZ as i32;
		
		// expanded submap (clicking and dragging)
		if disp.state.iface_settings.show_expanded_submap {
			let expanded_submap = map_data.map_szs[ZOOM_IND_EXPANDED_SUBMAP];
			let within_submap = m_event.y >= (disp.state.iface_settings.screen_sz.h - (expanded_submap.h+2)) as i32 && 
						  m_event.x < (expanded_submap.w+2) as i32;
			// go to location clicked in submap
			if within_submap && disp.state.kbd.map_drag.released_clicked_or_dragging(&disp.state.mouse_event) { //(lbutton_released(&disp.state.mouse_event) || lbutton_clicked(&disp.state.mouse_event)) || ldragging(&disp.state.mouse_event) {
				let screen_sz = disp.state.iface_settings.screen_sz;
				
				let sub_map_frac = {
					// click location offset on the screen
					let sub_map_loc = Coord {
						y: m_event.y as isize - (screen_sz.h - expanded_submap.h - 1) as isize,
						x: m_event.x as isize - 1
					};
					
					if sub_map_loc.x < 0 || sub_map_loc.x >= expanded_submap.w as isize ||
					   sub_map_loc.y < 0 || sub_map_loc.y >= expanded_submap.h as isize {return;}
					
					// rel location on submap
					ScreenFrac {
						y: sub_map_loc.y as f32 / expanded_submap.h as f32,
						x: sub_map_loc.x as f32 / expanded_submap.w as f32
					}
				};
				
				// # tiles shown on the main map screen
				//	(this is used to center the view on the plot that was clicked on)
				let d = ScreenFrac {
					y: (screen_sz.h - MAP_ROW_START - MAP_ROW_STOP_SZ) as f32/2.,
					x: (screen_sz.w - MAP_COL_STOP_SZ) as f32/2.
				};
				
				let cur_zoom = map_data.map_szs[disp.state.iface_settings.zoom_ind];
				
				disp.state.iface_settings.map_loc = Coord {
					y: (sub_map_frac.y*(cur_zoom.h as f32) - d.y).round() as isize,
					x: (sub_map_frac.x*(cur_zoom.w as f32) - d.x).round() as isize,
				};
				
				disp.state.iface_settings.chk_cursor_bounds(map_data);
				
				//center_cursor();
				return;
				
			// no longer in submap
			}else if !within_submap {
				disp.state.iface_settings.show_expanded_submap = false;
			}
		
		// move text cursor on main map to current mouse location (clicking and dragging)
		}else if m_event.y >= MAP_ROW_START as i32 &&
		   m_event.y < submap_start_row && m_event.x < disp.state.iface_settings.map_screen_sz.w as i32 {
			if disp.state.kbd.map_drag.released_clicked_pressed_or_dragging(&disp.state.mouse_event) || disp.state.kbd.action_drag.released_clicked_pressed_or_dragging(&disp.state.mouse_event) {  //lbutton_clicked(&disp.state.mouse_event) || lbutton_released(&disp.state.mouse_event) || lbutton_pressed(&disp.state.mouse_event) || ldragging(&disp.state.mouse_event) {
				let screen_coord = Coord {y: m_event.y as isize, x: m_event.x as isize};
				let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
				disp.set_text_coord(screen_coord, units, bldgs, exs, map_data, players, map_sz, gstate);
				
				if disp.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
					if disp.state.kbd.action_drag.released_clicked_pressed_or_dragging(&disp.state.mouse_event) {
						disp.state.iface_settings.start_individual_mv_mode(ActionType::Mv, &unit_inds, units, map_data);
					}
					
					let cur_mi = disp.state.iface_settings.cursor_to_map_ind(map_data);
					disp.state.iface_settings.set_action_start_coord_if_not_set(cur_mi, units, exs.last().unwrap(), map_data);
				}
			}
		// submap clicked or hovered in -> expand it
		}else if m_event.y >= (submap_start_row + 2) && m_event.x < (map_data.map_szs[ZOOM_IND_SUBMAP].w+1) as i32 && 
				disp.state.iface_settings.start_map_drag == None {
			disp.state.iface_settings.show_expanded_submap = true;
		}
		
		// drag map (update location of map shown on the screen or start move mode)
		if disp.state.kbd.map_drag.dragging(&disp.state.mouse_event) &&
				(!TEXT_MODE || (disp.state.iface_settings.zoom_ind != map_data.max_zoom_ind() || disp.state.iface_settings.add_action_to.actions().len() == 0)) {
				 // ^ if in text mode, only drag if we are not moving a unit -- ncurses doesn't seem to reliably send which mouse button is being held when dragging
			// continue drag
			if let Some(start_map_drag) = disp.state.iface_settings.start_map_drag {
				let m_event = disp.state.mouse_event.as_ref().unwrap();
				let cur = disp.state.iface_settings.screen_coord_to_map_coord(Coord {y: m_event.y as isize, x: m_event.x as isize}, map_data);
				disp.state.iface_settings.map_loc.y -= cur.y - start_map_drag.y;
				disp.state.iface_settings.map_loc.x -= cur.x - start_map_drag.x;
				disp.state.iface_settings.chk_cursor_bounds(map_data);
			
			// attempt to start moving unit, otherwise drag map
			}else if !TEXT_MODE || !disp.state.iface_settings.start_individual_mv_mode(ActionType::Mv, &unit_inds, units, map_data) {
				disp.state.iface_settings.start_map_drag = Some(disp.state.iface_settings.cursor_to_map_coord(map_data));
			}
			return;
		}else{disp.state.iface_settings.start_map_drag = None;}
	}
	
	///// menu
	if disp.state.kbd.open_top_menu == k {disp.start_menu();}
	
	///// zoom
	{
		macro_rules! set_text_coord{() => {
			if let Some(screen_coord) = disp.state.renderer.mouse_pos() {
				let screen_coord = Coord {y: screen_coord.0 as isize, x: screen_coord.1 as isize};
				disp.set_text_coord(screen_coord, units, bldgs, exs, map_data, players, map_sz, gstate);
			}
		};};
		
		macro_rules! chg_zoom{($dir:expr) => {disp.chg_zoom($dir, map_data, exs, units, bldgs, gstate, players, map_sz);};};
		
		if disp.state.kbd.zoom_in == k {chg_zoom!(1);}
		if disp.state.kbd.zoom_out == k {chg_zoom!(-1);}
		
		if scroll_up(&disp.state.mouse_event) {set_text_coord!(); chg_zoom!(1);}
		if scroll_down(&disp.state.mouse_event) {set_text_coord!(); chg_zoom!(-1);}
	}
	
	if disp.state.kbd.toggle_cursor_mode == k {
		disp.state.iface_settings.view_mv_mode = match disp.state.iface_settings.view_mv_mode {
			ViewMvMode::Cursor => ViewMvMode::Screen,
			ViewMvMode::Screen => ViewMvMode::Cursor,
			ViewMvMode::N => {panicq!("invalid view setting")}
		};
	}
		
	////////// cursor OR view straight
	if disp.state.kbd.up_normal(k) {lupdate!(CoordSet::Y, -1); cursor_moved = true;}
	if disp.state.kbd.down_normal(k) {lupdate!(CoordSet::Y, 1); cursor_moved = true;}
	if disp.state.kbd.left == k || KEY_LEFT == k {lupdate!(CoordSet::X, -1); cursor_moved = true;}
	if disp.state.kbd.right == k || KEY_RIGHT == k {lupdate!(CoordSet::X, 1); cursor_moved = true;}
	
	if disp.state.kbd.fast_up == k {aupdate!(CoordSet::Y, -1); cursor_moved = true;}
	if disp.state.kbd.fast_down == k {aupdate!(CoordSet::Y, 1); cursor_moved = true;}
	if disp.state.kbd.fast_left == k {aupdate!(CoordSet::X, -1); cursor_moved = true;}
	if disp.state.kbd.fast_right == k {aupdate!(CoordSet::X, 1); cursor_moved = true;}
	
	/////////// screen reader text tabbing
	if screen_reader_mode() {
		if k == disp.state.kbd.start_tabbing_through_bottom_screen_mode {
			disp.ui_mode = UIMode::TextTab {
				mode: 0,
				loc: TextTabLoc::BottomStats
			};
			disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		}
		
		if k == disp.state.kbd.start_tabbing_through_right_screen_mode {
			disp.ui_mode = UIMode::TextTab {
				mode: 0,
				loc: TextTabLoc::RightSide
			};
			disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		}
		
		if k == disp.state.kbd.forward_tab {
			if let UIMode::TextTab {ref mut mode, loc} = &mut disp.ui_mode {
				*mode += 1;
				// check if we wrap
				match loc {
					TextTabLoc::BottomStats => {
						if disp.state.txt_list.bottom.len() <= *mode {*mode = 0;}
					}
					TextTabLoc::RightSide => {
						if disp.state.txt_list.right.len() <= *mode {*mode = 0;}
					}
				}
			}
		}
		
		if k == disp.state.kbd.backward_tab {
			if let UIMode::TextTab {ref mut mode, loc} = &mut disp.ui_mode {
				if *mode >= 1 {
					*mode -= 1;
				// wrap
				}else{
					match loc {
						TextTabLoc::BottomStats => {
							*mode = disp.state.txt_list.bottom.len()-1;
						}
						TextTabLoc::RightSide => {
							*mode = disp.state.txt_list.right.len()-1;
						}
					}
				}
			}
		}
	}
	
	/////////// cursor OR view diagonol
	{
		// upper right
		if disp.state.kbd.diag_up_right == k {
			lupdate!(CoordSet::X, 2);
			lupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		if disp.state.kbd.fast_diag_up_right == k {
			aupdate!(CoordSet::X, 2);
			aupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		// upper left
		if disp.state.kbd.diag_up_left == k {
			lupdate!(CoordSet::X, -2);
			lupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		if disp.state.kbd.fast_diag_up_left == k {
			aupdate!(CoordSet::X, -2);
			aupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		// lower right
		if disp.state.kbd.diag_down_right == k {
			lupdate!(CoordSet::X, 2);
			lupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
		
		if disp.state.kbd.fast_diag_down_right == k {
			aupdate!(CoordSet::X, 2);
			aupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
		
		// lower left
		if disp.state.kbd.diag_down_left == k {
			lupdate!(CoordSet::X, -2);
			lupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
		
		if disp.state.kbd.fast_diag_down_left == k {
			aupdate!(CoordSet::X, -2);
			aupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
			
		// center on cursor
		if disp.state.kbd.center_on_cursor == k {disp.state.iface_settings.ctr_on_cur(map_data);}
	}
	
	if disp.state.kbd.center_on_next_unmoved_unit == k {
		disp.center_on_next_unmoved_menu_item(true, FindType::Units, map_data, exs, units, bldgs, gstate, players);
	}
	
	////////// end turn
	if disp.state.buttons.progress_day.activated(k, &disp.state.mouse_event) && disp.state.iface_settings.all_player_pieces_mvd {end_turn_c!();}
	if disp.state.buttons.progress_day_ign_unmoved_units.activated(k, &disp.state.mouse_event) { end_turn_c!();}
	if disp.state.buttons.progress_month.activated(k, &disp.state.mouse_event) { for _i in 0..FAST_TURN_INC {end_turn_c!();}}
	if disp.state.buttons.finish_all_unit_actions.activated(k, &disp.state.mouse_event) || disp.state.buttons.stop_fin_all_unit_actions.activated(k, &disp.state.mouse_event) {
		// stop finishing all actions
		if disp.state.iface_settings.auto_turn == AutoTurn::FinishAllActions {
			disp.state.set_auto_turn(AutoTurn::Off);
		// start finishing all actions
		}else	if disp.state.iface_settings.all_player_pieces_mvd {
			disp.state.set_auto_turn(AutoTurn::FinishAllActions);
		// alert that there are unmoved units
		}else if disp.state.iface_settings.auto_turn == AutoTurn::Off {
			disp.center_on_next_unmoved_menu_item(true, FindType::Units, map_data, exs, units, bldgs, gstate, players);
			disp.create_interrupt_window(UIMode::UnmovedUnitsNotification(UnmovedUnitsNotificationState {}));
		}
	}
	
	let pstats = &players[disp.state.iface_settings.cur_player as usize].stats;
	let exf = exs.last().unwrap();
	
	if let AddActionTo::None = disp.state.iface_settings.add_action_to {
		// mode change: assign action to everyone in brigade
		/*if disp.state.buttons.assign_action_to_all_in_brigade.activated(k, &disp.state.mouse_event) {
			if let Some(unit_ind) = iface_settings.unit_ind_frm_cursor(units, map_data, exf) {
				if let Some(brigade_nm) = pstats.unit_brigade_nm(unit_ind) {
					iface_settings.add_action_to = AddActionTo::AllInBrigade {
						brigade_nm: brigade_nm.to_string(),
						action_ifaces: None
					};
					return;
				}
			}
		}*/
		
		// view brigade
		if disp.state.buttons.view_brigade.activated(k, &disp.state.mouse_event) {
			if let Some(unit_ind) = disp.state.iface_settings.unit_ind_frm_cursor(units, map_data, exf) {
				if let Some(brigade_nm) = pstats.unit_brigade_nm(unit_ind) {
					disp.create_window(UIMode::BrigadesWindow(BrigadesWindowState {
						mode: 0,
						brigade_action: BrigadeAction::ViewBrigadeUnits {
							brigade_nm: brigade_nm.to_string()
						}
					}));
					return;
				}
			}
		}
		
		///// building production
		if disp.state.buttons.change_bldg_production.activated(k, &disp.state.mouse_event) {
			if let Some(bldg_ind) = disp.state.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) { // checks cur_player owns it
				let b = &bldgs[bldg_ind];
				if let Some(_) = &b.template.units_producable {
					let production_opt = match &b.args {
						BldgArgs::PopulationCenter {production, ..} | 
						BldgArgs::GenericProducable {production, ..} => production,
						BldgArgs::None => {panicq!("bldg arguments do not store production");}};
					
					// convert &UnitTemplate (production) into an index, to use for the window selection
					let mode = if let Some(production) = production_opt.last() {
						disp.state.production_options.bldgs[b.template.id as usize].as_ref().unwrap().options.iter().position(|o| {
							if let ArgOptionUI::UnitTemplate(Some(ut)) = o.arg {
								return ut == production.production;
							}
							false
						}).unwrap()
					}else{0};
					
					disp.create_window(UIMode::ProdListWindow(ProdListWindowState {mode}));
					return;
				}
			}
		}
		
		//// building production list
		if disp.state.buttons.view_production.activated(k, &disp.state.mouse_event) {
			if let Some(bldg_ind) = disp.state.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) { // checks cur_player owns it
				let b = &bldgs[bldg_ind];
				if let Some(_) = &b.template.units_producable {
					disp.create_window(UIMode::CurrentBldgProd(CurrentBldgProdState {mode: 0}));
					return;
				}
			}
		}
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// unit actions
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	let worker_unit_inds: Vec<usize> = unit_inds.iter().cloned().filter(|&ind| units[ind].template.nm[0] == WORKER_NM).collect();
	
	// individual actions:
	//	should be assigned only to one unit at a time (no applying these to every unit in a brigade)
	if let AddActionTo::None = disp.state.iface_settings.add_action_to {
		//// move with cursor
		if disp.state.buttons.move_with_cursor.activated(k, &disp.state.mouse_event) {
			disp.state.iface_settings.start_individual_mv_mode(ActionType::MvWithCursor, &unit_inds, units, map_data);
			return;
		}
		
		// group move (every unit in selected rectangle is assigned the movement action when enter is pressed)
		if disp.state.buttons.group_move.activated_ign_not_being_on_screen(k, &disp.state.mouse_event) ||
		   (shift_pressed() && lbutton_pressed(&disp.state.mouse_event)) {
			// 0. start rectangle corner at cursor location
			// 1. select second corner bounding rectangle
			// 2. select destination
			
			// Because group movement can start when the cursor is anywhere,
			// we do not initially have a unit to associate the action w/.
			// After step 1 is completed, we need to supply a unit_ind before
			// move paths can be computed in step 2.
			disp.state.iface_settings.add_action_to = AddActionTo::NoUnit {
				action: ActionMeta::new(ActionType::GroupMv {
					start_coord: Some(disp.state.iface_settings.cursor_to_map_ind(map_data)),
					end_coord: None
				})
			};
			return;
		}
		
		// join or leave brigade
		if disp.state.buttons.join_brigade.activated(k, &disp.state.mouse_event) || disp.state.buttons.leave_brigade.activated(k, &disp.state.mouse_event) {
			if let Some(unit_ind) = unit_inds.first() {
				let pstats = &mut players[disp.state.iface_settings.cur_player as usize].stats;
				
				// leave brigade
				if let Some(_) = pstats.unit_brigade_nm(*unit_ind) {
					pstats.rm_unit_frm_brigade(*unit_ind);
				// join brigade
				}else if pstats.brigades.len() != 0 {
					disp.create_window(UIMode::BrigadesWindow(BrigadesWindowState {
						mode: 0,
						brigade_action: BrigadeAction::Join {unit_ind: *unit_ind}
					}));
				}
				return;
			}
		}
	}
	
	// broadcastable actions:
	// 	can be assigned to all units in brigade at once (also can be assigned to individual units)
	if let AddActionTo::AllInBrigade {..} | AddActionTo::None = disp.state.iface_settings.add_action_to {
		//// movement
		if disp.state.buttons.move_unit.activated(k, &disp.state.mouse_event) {
			disp.state.iface_settings.start_broadcastable_mv_mode(ActionType::Mv, &unit_inds, units, map_data);
			return;
		}
		
		// fortify
		if disp.state.buttons.fortify.activated(k, &disp.state.mouse_event) {
			for unit_ind in unit_inds.iter() {
				let u = &mut units[*unit_ind];
				
				// prevent turn timestamp from being needlessly reset:
				if let Some(action) = &u.action.last() {
					if let ActionType::Fortify {..} = action.action_type {
						continue;
					}
				}
				
				u.action.clear();
				u.action.push(ActionMeta::new(ActionType::Fortify { turn: gstate.turn }));
			}
			disp.state.iface_settings.add_action_to = AddActionTo::None;
			disp.state.iface_settings.update_all_player_pieces_mvd_flag(units);
			return;
		}
		
		// pass move
		if disp.state.buttons.pass_move.activated(k, &disp.state.mouse_event) {
			////// pass move
			for unit_ind in unit_inds.iter() {
				units[*unit_ind].actions_used = None;
			}
			
			if unit_inds.len() != 0 {
				disp.state.iface_settings.add_action_to = AddActionTo::None;
				disp.state.iface_settings.update_all_player_pieces_mvd_flag(units);
				return;
			}
		}
		
		// auto-explore
		if disp.state.buttons.auto_explore.activated(k, &disp.state.mouse_event) {
			if unit_inds.len() != 0 {
				disp.create_window(UIMode::SelectExploreType(SelectExploreTypeState {mode: 0}));
				return;
			}
		}
		
		// disband
		if disp.state.buttons.disband.activated(k, &disp.state.mouse_event) {
			for unit_ind in unit_inds.iter() {
				disband_unit(*unit_ind, true, units, map_data, exs, players, gstate, map_sz);
			}
			
			if unit_inds.len() != 0 {
				disp.state.iface_settings.add_action_to = AddActionTo::None;
				disp.state.iface_settings.update_all_player_pieces_mvd_flag(units);
				
				// prevent unit_subsel out-of-bounds error
				if let Some(ex) = exs.last().unwrap().get(&disp.state.iface_settings.cursor_to_map_ind(map_data)) {
					if let Some(unit_inds) = &ex.unit_inds {
						if disp.state.iface_settings.unit_subsel >= unit_inds.len() {
							disp.state.iface_settings.unit_subsel = 0;
						}
					}else{
						disp.state.iface_settings.unit_subsel = 0;
					}
				}else{
					disp.state.iface_settings.unit_subsel = 0;
				}
				return;
			}
		}
		
		// attack
		if disp.state.buttons.attack.activated(k, &disp.state.mouse_event) {
			let unit_inds_filtered = unit_inds.iter().cloned().filter(|&ind| {
				let ut = units[ind].template;
				WORKER_NM != ut.nm[0] && !ut.attack_per_turn.is_none()
			}).collect();
			
			let act = ActionType::Attack {
				attack_coord: None,
				attackee: None,
				ignore_own_walls: false
			};
			
			disp.state.iface_settings.start_broadcastable_mv_mode(act, &unit_inds_filtered, units, map_data);
			return;
		}
		
		// sector automation
		if disp.state.buttons.soldier_automate.activated(k, &disp.state.mouse_event) {
			if unit_inds.iter().any(|&ind| !units[ind].template.attack_per_turn.is_none()) {
				disp.create_window(UIMode::CreateSectorAutomation(CreateSectorAutomationState {
					mode: 0,
					sector_nm: None,
					unit_enter_action: None,
					idle_action: None,
					curs_col: 1,
					txt: String::from("0")
				}));
				return;
			}
		}
		
		// automate zone creation
		if disp.state.buttons.automate_zone_creation.activated(k, &disp.state.mouse_event) {
			for unit_ind in worker_unit_inds.iter() {
				let u = &mut units[*unit_ind];
				disp.state.iface_settings.add_action_to = AddActionTo::None;
				
				// create sector around city, then automate
				if disp.state.iface_settings.workers_create_city_sectors {
					if let Some(ai_state) = &players[u.owner_id as usize].ptype.any_ai_state() {
						let u_coord = Coord::frm_ind(u.return_coord(), map_sz);
						if let Some(min_city) = ai_state.city_states.iter().min_by_key(|c|
								manhattan_dist(Coord::frm_ind(c.coord, map_sz), u_coord, map_sz)) {
							// get upper left and lower right coordinates
							if let Some(first_coord) = min_city.wall_coords.first() {
								const OFFSET: isize = 1;
								let (ul, lr) = {
									let mut ul = Coord {y: first_coord.y - 1, x: first_coord.x - 1};
									let mut lr = Coord {y: first_coord.y + 1, x: first_coord.x + 1};
									
									for wall_coord in min_city.wall_coords.iter().skip(1) {
										if ul.y > wall_coord.y {ul.y = wall_coord.y - OFFSET;}
										if ul.x > wall_coord.x {ul.x = wall_coord.x - OFFSET;}
										if lr.y < wall_coord.y {lr.y = wall_coord.y + OFFSET;}
										if lr.x < wall_coord.x {lr.x = wall_coord.x + OFFSET;}
									}
									(ul, lr)
								};
								
								let segments = vec![Rectangle {
									start: ul,
									end: lr
								}];
								
								if let Some(MapEx {bldg_ind: Some(bldg_ind), ..}) = exs.last().unwrap().get(&min_city.coord) {
									if let BldgArgs::PopulationCenter {nm, ..} = &bldgs[*bldg_ind as usize].args {
										let sectors = &mut players[u.owner_id as usize].stats.sectors;
										if !sectors.iter().any(|s| s.nm == *nm) {
											sectors.push(Sector {
												nm: nm.clone(),
												perim_coords: PerimCoords::new(gstate.turn, &segments, map_sz),
												segments
											});
										}
									}
								}
							}
							
							u.action.clear();
							u.action.push(ActionMeta::new(ActionType::UIWorkerAutomateCity));
							
						// no city found
						}else{
							disp.create_window(UIMode::GenericAlert(GenericAlertState {
								txt: disp.state.local.No_city_halls_found.clone()
							}));
						}
					}
				}
			}
			disp.state.iface_settings.update_all_player_pieces_mvd_flag(units);
			return;
		}
		
		// continue construction of bldg
		if disp.state.buttons.continue_bldg_construction.activated(k, &disp.state.mouse_event) {
			disp.state.iface_settings.start_broadcastable_mv_mode(ActionType::WorkerContinueBuildBldg, &worker_unit_inds, units, map_data);
			return;
		}
		
		// repair wall
		if disp.state.buttons.repair_wall.activated(k, &disp.state.mouse_event) {
			let unit_inds_filtered = unit_inds.iter().cloned().filter(|&ind| units[ind].template.repair_wall_per_turn != None).collect();
			disp.state.iface_settings.start_broadcastable_mv_mode(ActionType::WorkerRepairWall {wall_coord: None, turns_expended: 0}, &unit_inds_filtered, units, map_data);
			return;
		}
		
		// unload boat
		if disp.state.buttons.unload_boat.activated(k, &disp.state.mouse_event) {
			for unit_ind in unit_inds.iter() {
				//// unload units from boat
				let u = &units[*unit_ind];
				if let Some(units_carried) = &u.units_carried {
					debug_assertq!(u.template.carry_capac >= units_carried.len());
					
					while let Unboard::Loc {coord, carried_ind} = unboard_land_adj(*unit_ind, units, bldgs, map_data, exs.last().unwrap()) {
						if let Some(ref mut units_carried) = &mut units[*unit_ind].units_carried {
							let ur = units_carried.swap_remove(carried_ind);
							let owner_id = ur.owner_id;
							debug_assertq!(owner_id == units[*unit_ind].owner_id);
							unboard_unit(coord, ur, units, map_data, exs);
						}else{panicq!("carried unit should be available");}
					}
					
					disp.state.iface_settings.add_action_to = AddActionTo::None;
				}
			}
			return;
		}
	}
	
	let exf = exs.last().unwrap();
	
	// actions that can be performed by individuals, everyone in a brigade or added to the build list:
	{
		// build bldg
		if disp.state.buttons.build_bldg.activated(k, &disp.state.mouse_event) {
			if worker_unit_inds.len() != 0 {
				disp.create_window(UIMode::ProdListWindow(ProdListWindowState {mode: 0}));
				return;
			}
		}
	}
	
	// build list actions:
	//	can be added to a brigade's build list but not be assigned simultanously to multiple units at once (also can be assigned to individual units)
	if let AddActionTo::BrigadeBuildList {..} | AddActionTo::None = disp.state.iface_settings.add_action_to {
		// road
		if disp.state.buttons.build_road.activated(k, &disp.state.mouse_event) {
			let act = ActionType::WorkerBuildStructure {structure_type: StructureType::Road, turns_expended: 0};
			disp.state.iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		// wall
		if disp.state.buttons.build_wall.activated(k, &disp.state.mouse_event) {
			// check that no other units are here... prevent building wall on them
			if let Some(ex) = exs.last().unwrap().get(&disp.state.iface_settings.cursor_to_map_ind(map_data)) {
				if let Some(unit_inds) = &ex.unit_inds {
					if unit_inds.len() > 1 {return;}
				}
			}
			
			let act = ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, turns_expended: 0};
			disp.state.iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		// gate
		if disp.state.buttons.build_gate.activated(k, &disp.state.mouse_event) {
			let act = ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, turns_expended: 0};
			disp.state.iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		// rm zones & bldgs
		if disp.state.buttons.rm_bldgs_and_zones.activated(k, &disp.state.mouse_event) {
			let act = ActionType::WorkerRmZonesAndBldgs {start_coord: None, end_coord: None};
			disp.state.iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		//////////// zoning (creation and setting tax rates)
		if disp.state.buttons.zone_agricultural.activated(k, &disp.state.mouse_event) || disp.state.buttons.tax_agricultural.activated(k, &disp.state.mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Agricultural, units, bldgs, map_data, exf, disp); return;}
		if disp.state.buttons.zone_residential.activated(k, &disp.state.mouse_event) || disp.state.buttons.tax_residential.activated(k, &disp.state.mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Residential, units, bldgs, map_data, exf, disp); return;}
		if disp.state.buttons.zone_business.activated(k, &disp.state.mouse_event) || disp.state.buttons.tax_business.activated(k, &disp.state.mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Business, units, bldgs, map_data, exf, disp); return;}
		if disp.state.buttons.zone_industrial.activated(k, &disp.state.mouse_event) || disp.state.buttons.tax_industrial.activated(k, &disp.state.mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Industrial, units, bldgs, map_data, exf, disp); return;}
	}
	
	// tab -- unit subselection when multiple units on single plot of land
	// increment index, and wrap if needed
	if disp.state.buttons.tab.activated(k, &disp.state.mouse_event) {
		if !disp.state.iface_settings.add_action_to.is_none() {return;}
		let map_coord = disp.state.iface_settings.cursor_to_map_ind(map_data);
		if let Some(ex) = exs.last().unwrap().get(&map_coord) {
			if let Some(unit_inds) = &ex.unit_inds {
				let n_inds = unit_inds.len();
				debug_assertq!(n_inds > 0);
				debug_assertq!(n_inds <= MAX_UNITS_PER_PLOT);
				
				disp.state.iface_settings.unit_subsel += 1;
				if disp.state.iface_settings.unit_subsel >= n_inds { // wrap
					disp.state.iface_settings.unit_subsel = 0;
				}
			}
		}
		return;
	}
	
	if disp.state.buttons.increase_tax.activated(k, &disp.state.mouse_event) || disp.state.buttons.increase_tax_alt.activated(k, &disp.state.mouse_event) {
		set_taxes!(2);
		return;
	}else if disp.state.buttons.decrease_tax.activated(k, &disp.state.mouse_event) {
		set_taxes!(-2);
		return;
	}
	
	// remaining keys are not currently configurable...
	if k == KEY_ESC || disp.state.kbd.action_cancel.released(&disp.state.mouse_event) || disp.state.kbd.action_cancel.pressed(&disp.state.mouse_event) || disp.state.kbd.action_cancel.clicked(&disp.state.mouse_event) ||
			disp.state.buttons.Cancel.activated(k, &disp.state.mouse_event) {
		
		// clear the screen if we've exited screen reader text tabbing mode
		//	otherwise text on the right side of the screen will remain even though it's not needed
		if let UIMode::TextTab {..} = disp.ui_mode {
			disp.state.renderer.clear();
		}
		disp.ui_mode = UIMode::None;
		disp.state.iface_settings.show_expanded_submap = false;
		
		// clear action from unit if we were moving with cursor
		if let AddActionTo::IndividualUnit {
			action_iface: ActionInterfaceMeta {
				action: ActionMeta {
					action_type: ActionType::MvWithCursor, ..
				},
				unit_ind, ..
			}
		} = &disp.state.iface_settings.add_action_to {
			units[unit_ind.unwrap()].action.pop();
		}
		
		disp.state.iface_settings.add_action_to = AddActionTo::None;
		
	////////////////////////////////
	// enter
	} else if k == disp.state.kbd.enter || disp.state.kbd.action_drag.released(&disp.state.mouse_event) || disp.state.kbd.action_drag.clicked(&disp.state.mouse_event) { //lbutton_released(&disp.state.mouse_event) || lbutton_clicked(&disp.state.mouse_event) {
		if disp.state.iface_settings.zoom_ind != map_data.max_zoom_ind() {return;}
		let cur_mi = disp.state.iface_settings.cursor_to_map_ind(map_data);
		
		match &mut disp.state.iface_settings.add_action_to {
			AddActionTo::NoUnit {..} => {
				/////////////////////
				// drawing rectangle generally, set start coord to cursor
				if disp.state.iface_settings.set_action_start_coord_if_not_set(cur_mi, units, exf, map_data) {
					if let AddActionTo::NoUnit {action} = &mut disp.state.iface_settings.add_action_to {
						match action.action_type {
							ActionType::GroupMv {ref mut start_coord, ref mut end_coord} => {
								debug_assertq!(!start_coord.is_none());
								
								// finished drawing rectangle, now set path (if any units were selected)
								if end_coord.is_none() {
									*end_coord = Some(cur_mi);
									
									////////
									// update action_iface to have a valid unit_ind
									// (find some unit in the rectangle that was drawn, if none present, cancel selection)
									
									// rectangle selecting group
									let (rect_start_c, rect_sz) = {
										let rect_start_c = Coord::frm_ind(start_coord.unwrap(), map_sz);
										let rect_end_c = Coord::frm_ind(cur_mi, map_sz);
										
										start_coord_use(rect_start_c, rect_end_c, map_sz)
									};
									
									let exf = exs.last().unwrap();
									
									// for the first unit we find, save its unit_ind in action_iface
									// resave add_action_to as IndividualUnit (... even though multiple units will be moved...)
									for i_off in 0..rect_sz.h as isize {
									for j_off in 0..rect_sz.w as isize {
										if let Some(coord) = map_sz.coord_wrap(rect_start_c.y + i_off, rect_start_c.x + j_off) {
											if let Some(ex) = exf.get(&coord) {
											if let Some(unit_inds) = &ex.unit_inds {
												for unit_ind in unit_inds.iter() {
													if units[*unit_ind].owner_id != disp.state.iface_settings.cur_player {continue;}
													
													disp.state.iface_settings.add_action_to = AddActionTo::IndividualUnit {
														action_iface: unit_action_iface(Coord::frm_ind(coord, map_sz), action.action_type.clone(), *unit_ind, units)
													};
													return;
												}
											}}
										}
									}}
									
									// no unit found, cancel action
									disp.state.iface_settings.add_action_to = AddActionTo::None;
									return;
								}
								
								// if start_coord & end_coord are both Some value, the actions are assigned below	
							} ActionType::BrigadeCreation {ref mut start_coord, ref nm, ..} => {
								// finished drawing rectangle, create brigade
								if let Some(start_coord) = start_coord {
									// rectangle selecting group
									let (rect_start_c, rect_sz) = {
										let rect_start_c = Coord::frm_ind(*start_coord, map_sz);
										let rect_end_c = Coord::frm_ind(cur_mi, map_sz);
										
										start_coord_use(rect_start_c, rect_end_c, map_sz)
									};
									
									// get units
									let mut brigade_unit_inds = Vec::new();
									for i_off in 0..rect_sz.h as isize {
									for j_off in 0..rect_sz.w as isize {
										if let Some(coord) = map_sz.coord_wrap(rect_start_c.y + i_off, rect_start_c.x + j_off) {
											if let Some(ex) = exf.get(&coord) {
											if let Some(unit_inds) = &ex.unit_inds {
												for unit_ind in unit_inds.iter() {
													let u = &units[*unit_ind];
													if u.owner_id != disp.state.iface_settings.cur_player || u.template.nm[0] == RIOTER_NM {continue;}
													
													brigade_unit_inds.push(*unit_ind);
												}
											}}
										}
									}}
									
									// create brigade
									if brigade_unit_inds.len() != 0 {
										let pstats = &mut players[disp.state.iface_settings.cur_player as usize].stats;
										
										// if any unit already belongs to a brigade, remove it from the existing brigade
										for brigade in pstats.brigades.iter_mut() {
											for unit_ind in brigade_unit_inds.iter() {
												if let Some(pos) = brigade.unit_inds.iter().position(
														|&unit_ind_chk| unit_ind_chk == *unit_ind) {
													brigade.unit_inds.swap_remove(pos);
												}
											}
										}
										
										pstats.rm_empty_brigades();
										
										pstats.brigades.push(Brigade {
											nm: nm.clone(),
											unit_inds: brigade_unit_inds,
											build_list: VecDeque::new(),
											repair_sector_walls: None
										});
									}
									
									// view brigade
									disp.ui_mode = UIMode::BrigadesWindow(BrigadesWindowState {
										mode: 0,
										brigade_action: BrigadeAction::ViewBrigadeUnits {
											brigade_nm: nm.clone()
										}
									});
									disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
									
									// close action
									disp.state.iface_settings.add_action_to = AddActionTo::None;
									return;
									
								}else{panicq!("start_coord should've already been set");}
							} ActionType::SectorCreation {ref mut start_coord, creation_type, ref nm, ..} => {
								// finished drawing rectangle, create or add to sector
								if let Some(start_coord) = start_coord {
									let pstats = &mut players[disp.state.iface_settings.cur_player as usize].stats;
									
									let segment = Rectangle {
										start: Coord::frm_ind(*start_coord, map_sz),
										end: Coord::frm_ind(cur_mi, map_sz)
									};
									match creation_type {
										SectorCreationType::New => {
											let segments = vec![segment];
											pstats.sectors.push(Sector {
												nm: nm.clone(),
												perim_coords: PerimCoords::new(gstate.turn, &segments, map_sz),
												segments
											});
										} SectorCreationType::AddTo => {
											pstats.sector_frm_nm_mut(nm).add(gstate.turn, segment, map_sz);
										} SectorCreationType::N => {panicq!("invalid creation type");}
									}
									
									disp.state.iface_settings.add_action_to = AddActionTo::None;
									return;
								// set start coord, now wait for end coord
								}else{panicq!("start_coord should've already been set");}
							} _ => {panicq!("unhandled action");}
						}
					}else{panicq!("invalid add_action_to state");}
				}
			} AddActionTo::BrigadeBuildList {action: Some(_action), ..} => {
				// chk if zone & bldg placement valid:
				if disp.state.iface_settings.pre_process_action_chk_valid(cur_mi, units, bldgs, exf, map_data) {
					if let AddActionTo::BrigadeBuildList {action: Some(action), brigade_nm} = &disp.state.iface_settings.add_action_to {
						let pstats = &mut players[disp.state.iface_settings.cur_player as usize].stats;
						let brigade_nm = brigade_nm.clone();
						let mut action = action.clone();
						
						// first entry of path_coords, depending on action_type, later used to move unit to the start
						// (this happens in action_meta.set_action_meta()
						match action.action_type {
							ActionType::WorkerZone {..} => {}
							_ => {action.path_coords = vec![cur_mi];}
						}
						
						pstats.brigade_frm_nm_mut(&brigade_nm).build_list.push_back(action.clone());
						
						disp.state.iface_settings.add_action_to = AddActionTo::BrigadeBuildList {
							action: None,
							brigade_nm
						};
					}else{panicq!("invalid add_action_to state");}
				}
			} AddActionTo::IndividualUnit {action_iface} => {
				// do nothing if moving w/ cursor
				if let ActionType::MvWithCursor = &action_iface.action.action_type {return;}
				
				if disp.state.iface_settings.pre_process_action_chk_valid(cur_mi, units, bldgs, exf, map_data) {
					if let AddActionTo::IndividualUnit {action_iface} = &disp.state.iface_settings.add_action_to {
						let action_iface = action_iface.clone();
						if disp.assign_action_iface_to_unit(action_iface, cur_mi, units, bldgs, exs, gstate, players, temps, map_data, map_sz, frame_stats) {
							disp.state.iface_settings.add_action_to = AddActionTo::None;
						}
					}else{panicq!("invalid add_action_to state");}
				}
				
			} AddActionTo::AllInBrigade {action_ifaces: Some(action_ifaces), ..} => {
				let mut any_success = false;
				let mv_action = if let Some(ActionInterfaceMeta {
					action: ActionMeta {
						action_type: ActionType::Mv, ..
					}, ..
				}) = action_ifaces.last() {true} else {false};
				
				// add action to all units in brigade
				for action_iface in action_ifaces.clone() {
					if disp.assign_action_iface_to_unit(action_iface, cur_mi, units, bldgs, exs, gstate, players, temps, map_data, map_sz, frame_stats) {
						any_success = true;
					}
				}
				
				if any_success {
					if !mv_action {
						disp.state.iface_settings.add_action_to = AddActionTo::None;
					}else if let AddActionTo::AllInBrigade {ref mut action_ifaces, ..} = disp.state.iface_settings.add_action_to {
						*action_ifaces = None;
					}
				}
			
			} AddActionTo:: None |
			  AddActionTo::BrigadeBuildList {action: None, ..} | // <-- & below, these conditions are when the brigade is selected but no action has been chosen yet
			  AddActionTo::AllInBrigade {action_ifaces: None, ..} => {}
		}
		
		disp.reset_unit_subsel();
		disp.state.iface_settings.update_all_player_pieces_mvd_flag(units);
	}
	
	// no key pressed, reset cursor velocity
	if !cursor_moved {
		disp.state.iface_settings.cur_v = VelocCoord {x: SCROLL_ACCEL_INIT, y: SCROLL_ACCEL_INIT};
		disp.state.iface_settings.map_loc_v = VelocCoord {x: SCROLL_ACCEL_INIT, y: SCROLL_ACCEL_INIT};
	}
}

