use std::collections::VecDeque;
use crate::disp_lib::*;
use crate::disp::*;
use crate::disp::window::ProdOptions;
use crate::units::*;
use crate::map::*;
use crate::gcore::*;
use crate::tech::TechTemplate;
use crate::ai::{AIState, BarbarianState, AIConfig};
use crate::zones::return_zone_coord;
use crate::disp::menus::{OptionsUI, ArgOptionUI, FindType};
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;

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
		map_data: &mut MapData<'rt>, exf: &HashedMapEx, iface_settings: &mut IfaceSettings) {
	// zone if in build list mode or there is a worker selected
	if iface_settings.add_action_to.is_build_list() || unit_inds.iter().any(|&ind| units[ind].template.nm[0] == WORKER_NM) {
		let act = ActionType::WorkerZone { valid_placement: false, zone_type, start_coord: None, end_coord: None };
		iface_settings.start_build_mv_mode(act, unit_inds, units, map_data);
	
	// set taxes
	}else if let Some(bldg_ind) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf){
		if bldgs[bldg_ind].template.nm[0] == CITY_HALL_NM {
			iface_settings.ui_mode = UIMode::SetTaxes(zone_type);
		}
	}
}

pub fn non_menu_keys<'bt,'ut,'rt,'dt,'f>(key_pressed: i32, mouse_event: &Option<MEVENT>, turn: &mut usize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
		zone_exs_owners: &mut Vec<HashedMapZoneEx>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		bldg_config: &BldgConfig, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, doctrine_templates: &'dt Vec<DoctrineTemplate>,
		unit_templates: &'ut Vec<UnitTemplate<'rt>>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, 
		tech_templates: &Vec<TechTemplate>, resource_templates: &'rt Vec<ResourceTemplate>, stats: &mut Vec<Stats<'bt,'ut,'rt,'dt>>,
		relations: &mut Relations, owners: &Vec<Owner>, nms: &Nms,
		iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>, production_options: &mut ProdOptions<'bt,'ut,'rt,'dt>,
		ai_states: &mut Vec<Option<AIState<'bt,'ut,'rt,'dt>>>, ai_config: &AIConfig,
		barbarian_states: &mut Vec<Option<BarbarianState>>,
		logs: &mut Vec<Log>, disp_settings: &DispSettings, disp_chars: &DispChars, menu_options: &mut OptionsUI, frame_stats: &mut FrameStats,
		kbd: &KeyboardMap, buttons: &mut Buttons, l: &Localization, rng: &mut XorState, d: &mut DispState){
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	// cursor and view movement
	macro_rules! lupdate{($coord_set: expr, $sign: expr)=> {
		match iface_settings.view_mv_mode {
			ViewMvMode::Cursor => iface_settings.linear_update($coord_set, $sign, map_data, exs, units, bldgs, relations, owners, barbarian_states, ai_states, stats, map_sz, logs, *turn, d),
			ViewMvMode::Screen => iface_settings.linear_update_screen($coord_set, $sign, map_data, exs, units, bldgs, relations, owners, barbarian_states, ai_states, stats, map_sz, logs, *turn, d),
			ViewMvMode::N => {panicq!("invalid view setting");}
		}
	}};
	macro_rules! aupdate{($coord_set: expr, $sign: expr)=> {
		match iface_settings.view_mv_mode {
			ViewMvMode::Cursor => iface_settings.accel_update($coord_set, ($sign) as f32, map_data, exs, units, bldgs, relations, owners, barbarian_states, ai_states, stats, map_sz, logs, *turn, d),
			ViewMvMode::Screen => iface_settings.accel_update_screen($coord_set, ($sign) as f32, map_data, exs, units, bldgs, relations, owners, barbarian_states, ai_states, stats, map_sz, logs, *turn, d),
			ViewMvMode::N => {panicq!("invalid view setting");}
		}
	}};
	
	macro_rules! end_turn_c{()=>(end_turn(turn, units, bldg_config, bldgs, doctrine_templates, unit_templates, resource_templates, bldg_templates, tech_templates, 
		map_data, exs, zone_exs_owners, stats, relations, owners, nms, iface_settings, production_options, 
		ai_states, ai_config, barbarian_states, logs, disp_settings, disp_chars, menu_options, frame_stats, rng, kbd, l, buttons, d););};
		
	macro_rules! set_taxes{($inc: expr)=>{
		if let UIMode::SetTaxes(zone_type) = iface_settings.ui_mode {
			if let Some(city_hall_ind_set) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exs.last().unwrap()) {
				if let BldgArgs::CityHall {ref mut tax_rates, ..} = bldgs[city_hall_ind_set].args {
					let t = &mut tax_rates[zone_type as usize];
					let n = (*t as isize) + $inc;
					if n >= 0 && n <= 100 {
						*t = n as u8;
						
						// update taxable income on all bldgs connected to this city hall
						let owner_id = bldgs[city_hall_ind_set].owner_id;
						let zone_exs = &mut zone_exs_owners[owner_id as usize];
						
						for bldg_ind in 0..bldgs.len() {
							let b = &bldgs[bldg_ind];
							if owner_id != b.owner_id {continue;}
							if let BldgType::Taxable(_) = b.template.bldg_type { // req. that the bldg actually be taxable
								if let Some(zone_ex) = zone_exs.get(&return_zone_coord(b.coord, map_sz)) {
									if let Dist::Is {bldg_ind: city_hall_ind, ..} = zone_ex.ret_city_hall_dist() {
										if city_hall_ind == city_hall_ind_set {
											let new_income = -bldgs[bldg_ind].template.upkeep * 
												return_effective_tax_rate(b.coord, map_data, exs, zone_exs, bldgs, stats, doctrine_templates, map_sz, *turn);
											
											bldgs[bldg_ind].set_taxable_upkeep(new_income, stats);
										} // city hall used is the one set
									} // has city hall dist set
								}else{ panicq!("taxable bldg should be in taxable zone"); } // bldg is in a zone
							} // bldg is taxable
						} // bldg loop
					} // update taxes
				}else{panicq!("could not get city hall bldg args");}
			}else{panicq!("tax ui mode set but no bldg selected");}
		} // in tax-setting UI mode
	};};
	
	let k = key_pressed;
	let mut cursor_moved = false;
	
	//// expand/minimize large submap (should be before the section below or else it'll never be called when the submap is expanded)
	if buttons.show_expanded_submap.activated(k, mouse_event) || buttons.hide_submap.activated(k, mouse_event) {iface_settings.show_expanded_submap ^= true;}
	
	let exf = exs.last().unwrap();
	let unit_inds = iface_settings.sel_units_owned(&stats[iface_settings.cur_player as usize], units, map_data, exf);
	
	// mouse
	// 	-dragging (both submap and normal map)
	//	-cursor location updating / move search updating
	if let Some(m_event) = &mouse_event {
		let submap_start_row = iface_settings.screen_sz.h as i32 - MAP_ROW_STOP_SZ as i32;
		
		// expanded submap (clicking and dragging)
		if iface_settings.show_expanded_submap {
			let expanded_submap = map_data.map_szs[ZOOM_IND_EXPANDED_SUBMAP];
			let within_submap = m_event.y >= (iface_settings.screen_sz.h - (expanded_submap.h+2)) as i32 && 
						  m_event.x < (expanded_submap.w+2) as i32;
			// go to location clicked in submap
			if within_submap && (lbutton_released(mouse_event) || lbutton_clicked(mouse_event)) || dragging(mouse_event) {
				let screen_sz = iface_settings.screen_sz;
				
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
				
				let cur_zoom = map_data.map_szs[iface_settings.zoom_ind];
				
				iface_settings.map_loc = Coord {
					y: (sub_map_frac.y*(cur_zoom.h as f32) - d.y).round() as isize,
					x: (sub_map_frac.x*(cur_zoom.w as f32) - d.x).round() as isize,
				};
				
				iface_settings.chk_cursor_bounds(map_data);
				
				//center_cursor();
				return;
				
			// no longer in submap
			}else if !within_submap {
				iface_settings.show_expanded_submap = false;
			}
		
		// move text cursor on main map to current mouse location (clicking and dragging)
		}else if m_event.y >= MAP_ROW_START as i32 &&
		   m_event.y < submap_start_row && m_event.x < iface_settings.map_screen_sz.w as i32 {			
			if lbutton_clicked(mouse_event) || lbutton_released(mouse_event) || lbutton_pressed(mouse_event) || dragging(mouse_event) {
				let screen_coord = Coord {y: m_event.y as isize, x: m_event.x as isize};
				let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
				iface_settings.set_text_coord(screen_coord, units, bldgs, exs, relations, map_data, barbarian_states, ai_states, stats, owners, map_sz, logs, *turn, d);
				
				if iface_settings.zoom_ind == map_data.max_zoom_ind() {
					let cur_mi = iface_settings.cursor_to_map_ind(map_data);
					iface_settings.set_action_start_coord_if_not_set(cur_mi, units, exs.last().unwrap(), map_data);
				}
			}
		// submap clicked or hovered in -> expand it
		}else if m_event.y >= (submap_start_row + 2) && m_event.x < (map_data.map_szs[ZOOM_IND_SUBMAP].w+1) as i32 && 
				iface_settings.start_map_drag == None {
			iface_settings.show_expanded_submap = true;
		}
		
		// drag map (update location of map shown on the screen or start move mode)
		if (iface_settings.zoom_ind != map_data.max_zoom_ind() || iface_settings.add_action_to.actions().len() == 0) && dragging(mouse_event) {
			// continue drag
			if let Some(start_map_drag) = iface_settings.start_map_drag {
				let cur = iface_settings.screen_coord_to_map_coord(Coord {y: m_event.y as isize, x: m_event.x as isize}, map_data);
				iface_settings.map_loc.y -= cur.y - start_map_drag.y;
				iface_settings.map_loc.x -= cur.x - start_map_drag.x;
				iface_settings.chk_cursor_bounds(map_data);
			
			// attempt to start moving unit, otherwise drag map
			}else if !iface_settings.start_individual_mv_mode(ActionType::Mv, &unit_inds, units, map_data) {
				iface_settings.start_map_drag = Some(iface_settings.cursor_to_map_coord(map_data));
			}
			return;
		}else{iface_settings.start_map_drag = None;}
	}
	
	///// menu
	if kbd.open_top_menu == k {iface_settings.start_menu(d);}
	
	///// zoom
	{
		macro_rules! set_text_coord{() => {
			if let Some(screen_coord) = d.mouse_pos() {
				let screen_coord = Coord {y: screen_coord.0 as isize, x: screen_coord.1 as isize};
				iface_settings.set_text_coord(screen_coord, units, bldgs, exs, relations, map_data, barbarian_states, ai_states, stats, owners, map_sz, logs, *turn, d);
			}
		};};
		
		macro_rules! chg_zoom{($dir:expr) => {iface_settings.chg_zoom($dir, map_data, exs, units, bldgs, relations, owners, barbarian_states, ai_states, stats, map_sz, logs, *turn, d);};};
		
		if kbd.zoom_in == k {chg_zoom!(1);}
		if kbd.zoom_out == k {chg_zoom!(-1);}
		
		if scroll_up(mouse_event) {set_text_coord!(); chg_zoom!(1);}
		if scroll_down(mouse_event) {set_text_coord!(); chg_zoom!(-1);}
	}
	
	if kbd.toggle_cursor_mode == k {
		iface_settings.view_mv_mode = match iface_settings.view_mv_mode {
			ViewMvMode::Cursor => ViewMvMode::Screen,
			ViewMvMode::Screen => ViewMvMode::Cursor,
			ViewMvMode::N => {panicq!("invalid view setting")}
		};
	}
		
	////////// cursor OR view straight
	if kbd.up == k || KEY_UP == k {lupdate!(CoordSet::Y, -1); cursor_moved = true;}
	if kbd.down == k || KEY_DOWN == k {lupdate!(CoordSet::Y, 1); cursor_moved = true;}
	if kbd.left == k || KEY_LEFT == k {lupdate!(CoordSet::X, -1); cursor_moved = true;}
	if kbd.right == k || KEY_RIGHT == k {lupdate!(CoordSet::X, 1); cursor_moved = true;}
	
	if kbd.fast_up == k {aupdate!(CoordSet::Y, -1); cursor_moved = true;}
	if kbd.fast_down == k {aupdate!(CoordSet::Y, 1); cursor_moved = true;}
	if kbd.fast_left == k {aupdate!(CoordSet::X, -1); cursor_moved = true;}
	if kbd.fast_right == k {aupdate!(CoordSet::X, 1); cursor_moved = true;}
	
	/////////// cursor OR view diagonol
	{
		// upper right
		if kbd.diag_up_right == k {
			lupdate!(CoordSet::X, 2);
			lupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		if kbd.fast_diag_up_right == k {
			aupdate!(CoordSet::X, 2);
			aupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		// upper left
		if kbd.diag_up_left == k {
			lupdate!(CoordSet::X, -2);
			lupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		if kbd.fast_diag_up_left == k {
			aupdate!(CoordSet::X, -2);
			aupdate!(CoordSet::Y, -1);
			cursor_moved = true;
		}
		
		// lower right
		if kbd.diag_down_right == k {
			lupdate!(CoordSet::X, 2);
			lupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
		
		if kbd.fast_diag_down_right == k {
			aupdate!(CoordSet::X, 2);
			aupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
		
		// lower left
		if kbd.diag_down_left == k {
			lupdate!(CoordSet::X, -2);
			lupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
		
		if kbd.fast_diag_down_left == k {
			aupdate!(CoordSet::X, -2);
			aupdate!(CoordSet::Y, 1);
			cursor_moved = true;
		}
			
		// center on cursor
		if kbd.center_on_cursor == k {iface_settings.ctr_on_cur(map_data);}
	}
	
	////////// end turn
	if buttons.progress_day.activated(k, mouse_event) && iface_settings.all_player_pieces_mvd {end_turn_c!();}
	if buttons.progress_day_ign_unmoved_units.activated(k, mouse_event) { end_turn_c!();}
	if buttons.progress_month.activated(k, mouse_event) { for _i in 0..FAST_TURN_INC {end_turn_c!();}}
	if buttons.finish_all_unit_actions.activated(k, mouse_event) || buttons.stop_fin_all_unit_actions.activated(k, mouse_event) {
		// stop finishing all actions
		if iface_settings.auto_turn == AutoTurn::FinishAllActions {
			iface_settings.set_auto_turn(AutoTurn::Off, d);
		// start finishing all actions
		}else	if iface_settings.all_player_pieces_mvd {
			iface_settings.set_auto_turn(AutoTurn::FinishAllActions, d);
		// alert that there are unmoved units
		}else if iface_settings.auto_turn == AutoTurn::Off {
			iface_settings.center_on_next_unmoved_menu_item(true, FindType::Units, map_data, exs, units, bldgs, relations, owners, barbarian_states, ai_states, stats, logs, *turn, d);
			d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
			iface_settings.ui_mode = UIMode::UnmovedUnitsNotification;
		}
	}
	
	let pstats = &stats[iface_settings.cur_player as usize];
	let exf = exs.last().unwrap();
	
	if let AddActionTo::None = iface_settings.add_action_to {
		// mode change: assign action to everyone in brigade
		/*if buttons.assign_action_to_all_in_brigade.activated(k, mouse_event) {
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
		if buttons.view_brigade.activated(k, mouse_event) {
			if let Some(unit_ind) = iface_settings.unit_ind_frm_cursor(units, map_data, exf) {
				if let Some(brigade_nm) = pstats.unit_brigade_nm(unit_ind) {
					iface_settings.ui_mode = UIMode::BrigadesWindow {
						mode: 0,
						brigade_action: BrigadeAction::ViewBrigadeUnits {
							brigade_nm: brigade_nm.to_string()
						}
					};
					d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
					return;
				}
			}
		}
		
		///// building production
		if buttons.change_bldg_production.activated(k, mouse_event) {
			if let Some(bldg_ind) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) { // checks cur_player owns it
				let b = &bldgs[bldg_ind];
				if let Some(_) = &b.template.units_producable {
					let production_opt = match &b.args {
						BldgArgs::CityHall {production, ..} | 
						BldgArgs::GenericProducable {production, ..} => production,
						BldgArgs::None => {panicq!("bldg arguments do not store production");}};
					
					// convert &UnitTemplate (production) into an index, to use for the window selection
					let mode = if let Some(production) = production_opt.last() {
						production_options.bldgs[b.template.id as usize].as_ref().unwrap().options.iter().position(|o| {
							if let ArgOptionUI::UnitTemplate(Some(ut)) = o.arg {
								return ut == production.production;
							}
							false
						}).unwrap()
					}else{0};
					
					iface_settings.ui_mode = UIMode::ProdListWindow {mode};
					d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
					return;
				}
			}
		}
		
		//// building production list
		if buttons.view_production.activated(k, mouse_event) {
			if let Some(bldg_ind) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) { // checks cur_player owns it
				let b = &bldgs[bldg_ind];
				if let Some(_) = &b.template.units_producable {
					iface_settings.ui_mode = UIMode::CurrentBldgProd {mode: 0};
					d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
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
	if let AddActionTo::None = iface_settings.add_action_to {
		//// move with cursor
		if buttons.move_with_cursor.activated(k, mouse_event) {
			iface_settings.start_individual_mv_mode(ActionType::MvWithCursor, &unit_inds, units, map_data);
			return;
		}
		
		// group move (every unit in selected rectangle is assigned the movement action when enter is pressed)
		if buttons.group_move.activated_ign_not_being_on_screen(k, mouse_event) ||
		   (shift_pressed() && lbutton_pressed(mouse_event)) {
			// 0. start rectangle corner at cursor location
			// 1. select second corner bounding rectangle
			// 2. select destination
			
			// Because group movement can start when the cursor is anywhere,
			// we do not initially have a unit to associate the action w/.
			// After step 1 is completed, we need to supply a unit_ind before
			// move paths can be computed in step 2.
			iface_settings.add_action_to = AddActionTo::NoUnit {
				action: ActionMeta::new(ActionType::GroupMv {
					start_coord: Some(iface_settings.cursor_to_map_ind(map_data)),
					end_coord: None
				})
			};
			return;
		}
		
		// join or leave brigade
		if buttons.join_brigade.activated(k, mouse_event) || buttons.leave_brigade.activated(k, mouse_event) {
			if let Some(unit_ind) = unit_inds.first() {
				let pstats = &mut stats[iface_settings.cur_player as usize];
				
				// leave brigade
				if let Some(_) = pstats.unit_brigade_nm(*unit_ind) {
					pstats.rm_unit_frm_brigade(*unit_ind);
				// join brigade
				}else if pstats.brigades.len() != 0 {
					iface_settings.ui_mode = UIMode::BrigadesWindow {
						mode: 0,
						brigade_action: BrigadeAction::Join {unit_ind: *unit_ind}
					};
					d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				}
				return;
			}
		}
	}
	
	// broadcastable actions:
	// 	can be assigned to all units in brigade at once (also can be assigned to individual units)
	if let AddActionTo::AllInBrigade {..} | AddActionTo::None = iface_settings.add_action_to {
		//// movement
		if buttons.move_unit.activated(k, mouse_event) {
			iface_settings.start_broadcastable_mv_mode(ActionType::Mv, &unit_inds, units, map_data);
			return;
		}
		
		// fortify
		if buttons.fortify.activated(k, mouse_event) {
			for unit_ind in unit_inds.iter() {
				let u = &mut units[*unit_ind];
				
				// prevent turn timestamp from being needlessly reset:
				if let Some(action) = &u.action.last() {
					if let ActionType::Fortify {..} = action.action_type {
						continue;
					}
				}
				
				u.action.clear();
				u.action.push(ActionMeta::new(ActionType::Fortify { turn: *turn }));
			}
			iface_settings.add_action_to = AddActionTo::None;
			iface_settings.update_all_player_pieces_mvd_flag(units);
			return;
		}
		
		// pass move
		if buttons.pass_move.activated(k, mouse_event) {
			////// pass move
			for unit_ind in unit_inds.iter() {
				units[*unit_ind].actions_used = None;
			}
			
			if unit_inds.len() != 0 {
				iface_settings.add_action_to = AddActionTo::None;
				iface_settings.update_all_player_pieces_mvd_flag(units);
				return;
			}
		}
		
		// auto-explore
		if buttons.auto_explore.activated(k, mouse_event) {
			if unit_inds.len() != 0 {
				iface_settings.ui_mode = UIMode::SelectExploreType {mode: 0};
				d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				return;
			}
		}
		
		// disband
		if buttons.disband.activated(k, mouse_event) {
			for unit_ind in unit_inds.iter() {
				disband_unit(*unit_ind, true, units, map_data, exs, stats, relations, barbarian_states, ai_states, owners, map_sz, logs, *turn);
			}
			
			if unit_inds.len() != 0 {
				iface_settings.add_action_to = AddActionTo::None;
				iface_settings.update_all_player_pieces_mvd_flag(units);
				
				// prevent unit_subsel out-of-bounds error
				if let Some(ex) = exs.last().unwrap().get(&iface_settings.cursor_to_map_ind(map_data)) {
					if let Some(unit_inds) = &ex.unit_inds {
						if iface_settings.unit_subsel >= unit_inds.len() {
							iface_settings.unit_subsel = 0;
						}
					}else{
						iface_settings.unit_subsel = 0;
					}
				}else{
					iface_settings.unit_subsel = 0;
				}
				return;
			}
		}
		
		// attack
		if buttons.attack.activated(k, mouse_event) {
			let unit_inds_filtered = unit_inds.iter().cloned().filter(|&ind| {
				let ut = units[ind].template;
				WORKER_NM != ut.nm[0] && !ut.attack_per_turn.is_none()
			}).collect();
			
			let act = ActionType::Attack {
				attack_coord: None,
				attackee: None,
				ignore_own_walls: false
			};
			
			iface_settings.start_broadcastable_mv_mode(act, &unit_inds_filtered, units, map_data);
			return;
		}
		
		// sector automation
		if buttons.soldier_automate.activated(k, mouse_event) {
			if unit_inds.iter().any(|&ind| !units[ind].template.attack_per_turn.is_none()) {
				d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				iface_settings.ui_mode = UIMode::CreateSectorAutomation {
					mode: 0,
					sector_nm: None,
					unit_enter_action: None,
					idle_action: None,
					curs_col: 1,
					txt: String::from("0")
				};
				return;
			}
		}
		
		// automate zone creation
		if buttons.automate_zone_creation.activated(k, mouse_event) {
			for unit_ind in worker_unit_inds.iter() {
				let u = &mut units[*unit_ind];
				iface_settings.add_action_to = AddActionTo::None;
				
				// create sector around city, then automate
				if iface_settings.workers_create_city_sectors {
					if let Some(ai_state) = &ai_states[u.owner_id as usize] {
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
									if let BldgArgs::CityHall {nm, ..} = &bldgs[*bldg_ind as usize].args {
										let sectors = &mut stats[u.owner_id as usize].sectors;
										if !sectors.iter().any(|s| s.nm == *nm) {
											sectors.push(Sector {
												nm: nm.clone(),
												perim_coords: PerimCoords::new(*turn, &segments, map_sz),
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
							d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
							iface_settings.ui_mode = UIMode::GenericAlert {
								txt: l.No_city_halls_found.clone()
							};
						}
					}
				}
			}
			iface_settings.update_all_player_pieces_mvd_flag(units);
			return;
		}
		
		// continue construction of bldg
		if buttons.continue_bldg_construction.activated(k, mouse_event) {
			iface_settings.start_broadcastable_mv_mode(ActionType::WorkerContinueBuildBldg, &worker_unit_inds, units, map_data);
			return;
		}
		
		// repair wall
		if buttons.repair_wall.activated(k, mouse_event) {
			let unit_inds_filtered = unit_inds.iter().cloned().filter(|&ind| units[ind].template.repair_wall_per_turn != None).collect();
			iface_settings.start_broadcastable_mv_mode(ActionType::WorkerRepairWall {wall_coord: None, turns_expended: 0}, &unit_inds_filtered, units, map_data);
			return;
		}
		
		// unload boat
		if buttons.unload_boat.activated(k, mouse_event) {
			for unit_ind in unit_inds.iter() {
				//// unload units from boat
				let u = &units[*unit_ind];
				if let Some(units_carried) = &u.units_carried {
					debug_assertq!(u.template.carry_capac >= units_carried.len());
					
					while let Unboard::Loc {coord, carried_ind} = unboard_land_adj(*unit_ind, units, bldgs, map_data, exs.last().unwrap()) {
						if let Some(ref mut units_carried) = &mut units[*unit_ind].units_carried {
							let ur = units_carried.swap_remove(carried_ind);
							debug_assertq!(ur.owner_id == units[*unit_ind].owner_id);
							unboard_unit(coord, ur, units, map_data, exs, owners);
						}else{panicq!("carried unit should be available");}
					}
					
					iface_settings.add_action_to = AddActionTo::None;
				}
			}
			return;
		}
	}
	
	let exf = exs.last().unwrap();
	
	// actions that can be performed by individuals, everyone in a brigade or added to the build list:
	{
		// build bldg
		if buttons.build_bldg.activated(k, mouse_event) {
			if worker_unit_inds.len() != 0 {
				iface_settings.ui_mode = UIMode::ProdListWindow {mode: 0};
				d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				return;
			}
		}
	}
	
	// build list actions:
	//	can be added to a brigade's build list but not be assigned simultanously to multiple units at once (also can be assigned to individual units)
	if let AddActionTo::BrigadeBuildList {..} | AddActionTo::None = iface_settings.add_action_to {
		// road
		if buttons.build_road.activated(k, mouse_event) {
			let act = ActionType::WorkerBuildStructure {structure_type: StructureType::Road, turns_expended: 0};
			iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		// wall
		if buttons.build_wall.activated(k, mouse_event) {
			// check that no other units are here... prevent building wall on them
			if let Some(ex) = exs.last().unwrap().get(&iface_settings.cursor_to_map_ind(map_data)) {
				if let Some(unit_inds) = &ex.unit_inds {
					if unit_inds.len() > 1 {return;}
				}
			}
			
			let act = ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, turns_expended: 0};
			iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		// gate
		if buttons.build_gate.activated(k, mouse_event) {
			let act = ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, turns_expended: 0};
			iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		// rm zones & bldgs
		if buttons.rm_bldgs_and_zones.activated(k, mouse_event) {
			let act = ActionType::WorkerRmZonesAndBldgs {start_coord: None, end_coord: None};
			iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
			return;
		}
		
		//////////// zoning (creation and setting tax rates)
		if buttons.zone_agricultural.activated(k, mouse_event) || buttons.tax_agricultural.activated(k, mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Agricultural, units, bldgs, map_data, exf, iface_settings); return;}
		if buttons.zone_residential.activated(k, mouse_event) || buttons.tax_residential.activated(k, mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Residential, units, bldgs, map_data, exf, iface_settings); return;}
		if buttons.zone_business.activated(k, mouse_event) || buttons.tax_business.activated(k, mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Business, units, bldgs, map_data, exf, iface_settings); return;}
		if buttons.zone_industrial.activated(k, mouse_event) || buttons.tax_industrial.activated(k, mouse_event) {start_zoning(&worker_unit_inds, ZoneType::Industrial, units, bldgs, map_data, exf, iface_settings); return;}
	}
	
	// tab -- unit subselection when multiple units on single plot of land
	// increment index, and wrap if needed
	if buttons.tab.activated(k, mouse_event) {
		if !iface_settings.add_action_to.is_none() {return;}
		let map_coord = iface_settings.cursor_to_map_ind(map_data);
		if let Some(ex) = exs.last().unwrap().get(&map_coord) {
			if let Some(unit_inds) = &ex.unit_inds {
				let n_inds = unit_inds.len();
				debug_assertq!(n_inds > 0);
				debug_assertq!(n_inds <= MAX_UNITS_PER_PLOT);
				
				iface_settings.unit_subsel += 1;
				if iface_settings.unit_subsel >= n_inds { // wrap
					iface_settings.unit_subsel = 0;
				}
			}
		}
		return;
	}
	
	if buttons.increase_tax.activated(k, mouse_event) || buttons.increase_tax_alt.activated(k, mouse_event) {
		set_taxes!(2);
		return;
	}else if buttons.decrease_tax.activated(k, mouse_event) {
		set_taxes!(-2);
		return;
	}
	
	// remaining keys are not currently configurable...
	if k == KEY_ESC || rbutton_pressed(mouse_event) || rbutton_released(mouse_event) || rbutton_clicked(mouse_event) ||
			buttons.Cancel.activated(k, mouse_event) {
		iface_settings.ui_mode = UIMode::None;
		iface_settings.show_expanded_submap = false;
		
		// clear action from unit if we were moving with cursor
		if let AddActionTo::IndividualUnit {
			action_iface: ActionInterfaceMeta {
				action: ActionMeta {
					action_type: ActionType::MvWithCursor, ..
				},
				unit_ind, ..
			}
		} = &iface_settings.add_action_to {
			units[unit_ind.unwrap()].action.pop();
		}
		
		iface_settings.add_action_to = AddActionTo::None;
		
	////////////////////////////////
	// enter
	} else if k == kbd.enter || lbutton_released(mouse_event) || lbutton_clicked(mouse_event) {
		if iface_settings.zoom_ind != map_data.max_zoom_ind() {return;}
		let cur_mi = iface_settings.cursor_to_map_ind(map_data);
		
		match &mut iface_settings.add_action_to {
			AddActionTo::NoUnit {..} => {
				/////////////////////
				// drawing rectangle generally, set start coord to cursor
				if iface_settings.set_action_start_coord_if_not_set(cur_mi, units, exf, map_data) {
					if let AddActionTo::NoUnit {action} = &mut iface_settings.add_action_to {
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
													if units[*unit_ind].owner_id != iface_settings.cur_player {continue;}
													
													iface_settings.add_action_to = AddActionTo::IndividualUnit {
														action_iface: unit_action_iface(Coord::frm_ind(coord, map_sz), action.action_type.clone(), *unit_ind, units)
													};
													return;
												}
											}}
										}
									}}
									
									// no unit found, cancel action
									iface_settings.add_action_to = AddActionTo::None;
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
													if u.owner_id != iface_settings.cur_player || u.template.nm[0] == RIOTER_NM {continue;}
													
													brigade_unit_inds.push(*unit_ind);
												}
											}}
										}
									}}
									
									// create brigade
									if brigade_unit_inds.len() != 0 {
										let pstats = &mut stats[iface_settings.cur_player as usize];
										
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
									iface_settings.ui_mode = UIMode::BrigadesWindow {
										mode: 0,
										brigade_action: BrigadeAction::ViewBrigadeUnits {
											brigade_nm: nm.clone()
										}
									};
									d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
									
									// close action
									iface_settings.add_action_to = AddActionTo::None;
									return;
									
								}else{panicq!("start_coord should've already been set");}
							} ActionType::SectorCreation {ref mut start_coord, creation_type, ref nm, ..} => {
								// finished drawing rectangle, create or add to sector
								if let Some(start_coord) = start_coord {
									let pstats = &mut stats[iface_settings.cur_player as usize];
									
									let segment = Rectangle {
										start: Coord::frm_ind(*start_coord, map_sz),
										end: Coord::frm_ind(cur_mi, map_sz)
									};
									match creation_type {
										SectorCreationType::New => {
											let segments = vec![segment];
											pstats.sectors.push(Sector {
												nm: nm.clone(),
												perim_coords: PerimCoords::new(*turn, &segments, map_sz),
												segments
											});
										} SectorCreationType::AddTo => {
											pstats.sector_frm_nm_mut(nm).add(*turn, segment, map_sz);
										} SectorCreationType::N => {panicq!("invalid creation type");}
									}
									
									iface_settings.add_action_to = AddActionTo::None;
									return;
								// set start coord, now wait for end coord
								}else{panicq!("start_coord should've already been set");}
							} _ => {panicq!("unhandled action");}
						}
					}else{panicq!("invalid add_action_to state");}
				}
			} AddActionTo::BrigadeBuildList {action: Some(_action), ..} => {
				// chk if zone & bldg placement valid:
				if iface_settings.pre_process_action_chk_valid(cur_mi, units, bldgs, exf, map_data) {
					if let AddActionTo::BrigadeBuildList {action: Some(action), brigade_nm} = &iface_settings.add_action_to {
						let pstats = &mut stats[iface_settings.cur_player as usize];
						let brigade_nm = brigade_nm.clone();
						let mut action = action.clone();
						
						// first entry of path_coords, depending on action_type, later used to move unit to the start
						// (this happens in action_meta.set_action_meta()
						match action.action_type {
							ActionType::WorkerZone {..} => {}
							_ => {action.path_coords = vec![cur_mi];}
						}
						
						pstats.brigade_frm_nm_mut(&brigade_nm).build_list.push_back(action.clone());
						
						iface_settings.add_action_to = AddActionTo::BrigadeBuildList {
							action: None,
							brigade_nm
						};
					}else{panicq!("invalid add_action_to state");}
				}
			} AddActionTo::IndividualUnit {action_iface} => {
				// do nothing if moving w/ cursor
				if let ActionType::MvWithCursor = &action_iface.action.action_type {return;}
				
				if iface_settings.pre_process_action_chk_valid(cur_mi, units, bldgs, exf, map_data) {
					if let AddActionTo::IndividualUnit {action_iface} = &iface_settings.add_action_to {
						let action_iface = action_iface.clone();
						if iface_settings.assign_action_iface_to_unit(action_iface, cur_mi, units, bldgs, stats, exs, ai_states, barbarian_states, relations, owners, bldg_config, unit_templates, bldg_templates, tech_templates, doctrine_templates, zone_exs_owners, map_data, map_sz, disp_settings, disp_chars, menu_options, logs, *turn, rng, frame_stats, kbd, l, buttons, d) {
							iface_settings.add_action_to = AddActionTo::None;
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
					if iface_settings.assign_action_iface_to_unit(action_iface, cur_mi, units, bldgs, stats, exs, ai_states, barbarian_states, relations, owners, bldg_config, unit_templates, bldg_templates, tech_templates, doctrine_templates, zone_exs_owners, map_data, map_sz, disp_settings, disp_chars, menu_options, logs, *turn, rng, frame_stats, kbd, l, buttons, d) {
						any_success = true;
					}
				}
				
				if any_success {
					if !mv_action {
						iface_settings.add_action_to = AddActionTo::None;
					}else if let AddActionTo::AllInBrigade {ref mut action_ifaces, ..} = iface_settings.add_action_to {
						*action_ifaces = None;
					}
				}
			
			} AddActionTo:: None |
			  AddActionTo::BrigadeBuildList {action: None, ..} | // <-- & below, these conditions are when the brigade is selected but no action has been chosen yet
			  AddActionTo::AllInBrigade {action_ifaces: None, ..} => {}
		}
		
		iface_settings.reset_unit_subsel();
		iface_settings.update_all_player_pieces_mvd_flag(units);
	}
	
	// no key pressed, reset cursor velocity
	if !cursor_moved {
		iface_settings.cur_v = VelocCoord {x: SCROLL_ACCEL_INIT, y: SCROLL_ACCEL_INIT};
		iface_settings.map_loc_v = VelocCoord {x: SCROLL_ACCEL_INIT, y: SCROLL_ACCEL_INIT};
	}
}

