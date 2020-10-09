use std::cmp::{min};
use std::convert::TryFrom;

use crate::disp_lib::*;
use crate::disp::*;
use crate::units::{ActionType, add_unit, ActionMeta, DelAction, mv_unit, ExploreType,
	SectorUnitEnterAction, SectorIdleAction, SectorCreationType};
use crate::tech::{TechTemplate, TECH_SZ_PRINT};
use crate::saving::{GameState};
use crate::gcore::{Log, Relations, XorState, worker_inds};
use crate::saving::save_game::{SaveType, SAVE_DIR};//, save_game};
use crate::ai::{AIState, BarbarianState, AIConfig};
use crate::resources::ResourceTemplate;
use crate::doctrine::{DoctrineTemplate, DOCTRINE_SZ_PRINT};
use crate::nn::{TxtPrinter, TxtCategory};
use crate::keyboard::KeyboardMap;
use crate::player::Player;
use crate::localization::Localization;
use crate::containers::Templates;
use crate::nobility::House;

use super::*;

pub fn end_window(iface_settings: &mut IfaceSettings, d: &mut DispState){
	iface_settings.reset_auto_turn(d);
	iface_settings.ui_mode = UIMode::None;
	d.clear();
	d.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
}

fn tree_window_movement(k: i32, sel_mv: &mut TreeSelMv, tree_offsets: &mut Option<TreeOffsets>,
		entry_sz_print: &ScreenSz, kbd: &KeyboardMap) {
	// entry selection (tech or spirituality)
	if k == KEY_RIGHT {*sel_mv = TreeSelMv::Right;}
	if k == KEY_LEFT {*sel_mv = TreeSelMv::Left;}
	if k == KEY_UP {*sel_mv = TreeSelMv::Up;}
	if k == KEY_DOWN {*sel_mv = TreeSelMv::Down;}
	////////////// diagonol scrolling
	// upper right
	if k == kbd.diag_up_right || k == kbd.fast_diag_up_right {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.row += entry_sz_print.h as i32 / 2;
			toffs.col -= entry_sz_print.w as i32 / 2;
		}
	}
	// upper left
	if k == kbd.diag_up_left || k == kbd.fast_diag_up_left {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.row += entry_sz_print.h as i32 / 2;
			toffs.col += entry_sz_print.w as i32 / 2;
		}
	}
	
	// lower left
	if k == kbd.diag_down_left || k == kbd.fast_diag_down_left {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.row -= entry_sz_print.h as i32 / 2;
			toffs.col += entry_sz_print.w as i32 / 2;
		}
	}
	
	// lower right
	if k == kbd.diag_down_right || k == kbd.fast_diag_down_right {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.row -= entry_sz_print.h as i32 / 2;
			toffs.col -= entry_sz_print.w as i32 / 2;
		}
	}
	
	////////////// straight scrolling
	// scroll tree up
	if k == kbd.up || k == kbd.fast_up {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.row += entry_sz_print.h as i32 / 2;
		}
	}
	
	// scroll tree down
	if k == kbd.down || k == kbd.fast_down {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.row -= entry_sz_print.h as i32 / 2;
		}
	}
	
	// scroll tree left
	if k == kbd.left || k == kbd.fast_left {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.col += entry_sz_print.w as i32 / 2;
		}
	}
		
	// scroll tree right
	if k == kbd.right || k == kbd.fast_right {
		if let Some(ref mut toffs) = tree_offsets {
			toffs.col -= entry_sz_print.w as i32 / 2;
		}
	}
}

// returns true if window active, false if not active or no longer active
pub fn do_window_keys<'f,'bt,'ut,'rt,'dt>(key_pressed: i32, mouse_event: &Option<MEVENT>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		production_options: &mut ProdOptions<'bt,'ut,'rt,'dt>, iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, relations: &mut Relations, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		logs: &mut Vec<Log>, disp_settings: &mut DispSettings, disp_chars: &DispChars,
		menu_options: &mut OptionsUI, frame_stats: &FrameStats,
		turn: usize, game_state: &mut GameState, game_difficulties: &GameDifficulties,
		rng: &mut XorState, kbd: &KeyboardMap, l: &Localization, 
		buttons: &mut Buttons, d: &mut DispState) -> UIRet {
	
	if buttons.Esc_to_close.activated(key_pressed, mouse_event) {
		end_window(iface_settings, d);
		return UIRet::Active;
	}
	
	// mouse hovering
	if let Some(ind) = buttons.list_item_hovered(mouse_event) {
		match iface_settings.ui_mode {
			// window w/ inactive entries -- prevent inactive items from being selected
			UIMode::EncyclopediaWindow {state:
				EncyclopediaState::ExemplarSelection {
					category: EncyclopediaCategory::Bldg,
					ref mut mode, ..}} => {
				let options = encyclopedia_bldg_list(temps.bldgs, l);
				if let ArgOptionUI::Ind(None) = options.options[ind].arg {
					return UIRet::Active;
				}
				*mode = ind;
				return UIRet::Active;
			}
			
			// windows with lists
			UIMode::CivilizationIntelWindow {ref mut mode, ..} |
			UIMode::DiscoverTechWindow {ref mut mode} |
			UIMode::ObtainResourceWindow {ref mut mode} |
			UIMode::UnitsWindow {ref mut mode} |
			UIMode::CitiesWindow {ref mut mode} |
			UIMode::ImprovementBldgsWindow {ref mut mode} |
			UIMode::MilitaryBldgsWindow {ref mut mode} |
			UIMode::BrigadesWindow {ref mut mode, ..} |
			UIMode::SectorsWindow {ref mut mode, ..} |
			UIMode::SetDifficultyWindow {ref mut mode} |
			UIMode::PlaceUnitWindow {ref mut mode} |
			UIMode::OpenWindow {ref mut mode, ..} |
			UIMode::ResourcesDiscoveredWindow {ref mut mode} |
			UIMode::CreateSectorAutomation {ref mut mode, ..} |
			UIMode::ProdListWindow {ref mut mode} |
			UIMode::CurrentBldgProd {ref mut mode} |
			UIMode::NobilityReqToJoin {ref mut mode, ..} |
			UIMode::SelectBldgDoctrine {ref mut mode, ..} |
			UIMode::SelectExploreType {ref mut mode} |
			UIMode::NoblePedigree {ref mut mode, ..} |
			UIMode::EncyclopediaWindow {state: 
				EncyclopediaState::CategorySelection {ref mut mode}} |
			UIMode::EncyclopediaWindow {state:
				EncyclopediaState::ExemplarSelection {ref mut mode, ..}} |
			UIMode::BrigadeBuildList {ref mut mode, ..} |
			UIMode::ContactEmbassyWindow {state:
				EmbassyState::CivSelection {ref mut mode}} |
			UIMode::SwitchToPlayerWindow {ref mut mode} => {
				*mode = ind;
				return UIRet::Active;
			}
			
			// windows that do not use lists
			UIMode::None |
			UIMode::TextTab {..} |
			UIMode::SetTaxes(_) |
			UIMode::Menu {..} |
			UIMode::GenericAlert {..} |
			UIMode::PublicPollingWindow |
			UIMode::CitizenDemandAlert {..} |
			UIMode::SaveAsWindow {..} |
			UIMode::SaveAutoFreqWindow {..} |
			UIMode::TechWindow {..} |
			UIMode::TechDiscoveredWindow {..} |
			UIMode::GetTextWindow {..} |
			UIMode::DoctrineWindow {..} |
			UIMode::PlotWindow {..} |
			UIMode::ContactEmbassyWindow {..} |
			UIMode::ResourcesAvailableWindow |
			UIMode::WorldHistoryWindow {..} |
			UIMode::BattleHistoryWindow {..} |
			UIMode::EconomicHistoryWindow {..} |
			UIMode::WarStatusWindow |
			UIMode::GoToCoordinateWindow {..} |
			UIMode::MvWithCursorNoActionsRemainAlert {..} |
			UIMode::PrevailingDoctrineChangedWindow |
			UIMode::RiotingAlert {..} |
			UIMode::CivicAdvisorsWindow | UIMode::InitialGameWindow |
			UIMode::EndGameWindow | UIMode::AboutWindow |
			UIMode::UnmovedUnitsNotification |
			UIMode::ForeignUnitInSectorAlert {..} => {}
		}
	}
	
	let exf = exs.last().unwrap();
	let pstats = &mut players[iface_settings.cur_player as usize].stats;
	
	/////////////// tech window
	// (should be checked first as the remaining if statements are first checking the cursor location, then
	//  are checking iface_settings.ui_mode -- if a unit/bldg is selected this function would terminate if
	//  those conditions are partially matched)
	if let UIMode::TechWindow {sel, ref mut sel_mv, ref mut tree_offsets, ..} = iface_settings.ui_mode {
		*sel_mv = TreeSelMv::None;
		
		tree_window_movement(key_pressed, sel_mv, tree_offsets, &TECH_SZ_PRINT, kbd);
		
		// start researching tech
		if key_pressed == '\n' as i32 {
			if let Some(sel) = sel {
				// not already scheduled
				if !pstats.techs_scheduled.contains(&sel) {
					pstats.start_researching(sel, temps.techs);
					
				// unschedule
				}else{
					pstats.stop_researching(sel, temps.techs);
				}
			}
		}
		
		return UIRet::Active;
	
	}else if let UIMode::NoblePedigree {ref mut mode, ref mut house_nm, ..} = iface_settings.ui_mode {
		/*let list = noble_houses_list(&pstats.houses);
		
		macro_rules! enter_action{($mode: expr) => {
			if let Some(house) = pstats.houses.houses.get(*mode) {
				*house_nm = Some(house.name.clone());
			}else{
				end_window(iface_settings, d);
			}
			return UIRet::Active;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == '\n' as i32 => {
				enter_action!(*mode);
			} _ => {}
		}*/
		return UIRet::Active;
		
	}else if let UIMode::DoctrineWindow {ref mut sel_mv, ref mut tree_offsets, ..} = iface_settings.ui_mode {
		*sel_mv = TreeSelMv::None;
		
		tree_window_movement(key_pressed, sel_mv, tree_offsets, &DOCTRINE_SZ_PRINT, kbd);
		
		return UIRet::Active;
	
	//////////////////// open window
	}else if let UIMode::OpenWindow {ref mut mode, save_files, ..} = &mut iface_settings.ui_mode {
		macro_rules! enter_action {($mode: expr) => {
			*game_state = GameState::Load(format!("{}/{}", SAVE_DIR, save_files[$mode].nm));
			return UIRet::ChgGameState;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		else if buttons.Open.activated(key_pressed, mouse_event) && save_files.len() != 0 {
			enter_action!(*mode);
		}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (save_files.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = save_files.len() - 1;
				}
			} _ => {}
		}
		
		return UIRet::Active;
	
	//////////////////// auto-save window
	}else if let UIMode::SaveAutoFreqWindow {ref mut curs_col, ref mut freq, ..} = &mut iface_settings.ui_mode {
		if buttons.Confirm.activated(key_pressed, mouse_event) && freq.len() > 0 {
			if let Result::Ok(val) = freq.parse() {
				iface_settings.checkpoint_freq = val;
			}
			end_window(iface_settings, d);
			return UIRet::Active;
		}

		match key_pressed {
			KEY_LEFT => {if *curs_col != 0 {*curs_col -= 1;}}
			KEY_RIGHT => {
				if *curs_col < (freq.len() as isize) {
					*curs_col += 1;
				}
			}
			
			KEY_HOME | KEY_UP => {*curs_col = 0;}
			
			// end key
			KEY_DOWN | 0x166 | 0602 => {*curs_col = freq.len() as isize;}
			
			// backspace
			KEY_BACKSPACE | 127 | 0x8  => {
				if *curs_col != 0 {
					*curs_col -= 1;
					freq.remove(*curs_col as usize);
				}
			}
			
			// delete
			KEY_DC => {
				if *curs_col != freq.len() as isize {
					freq.remove(*curs_col as usize);
				}
			}
			_ => { // insert character
				if freq.len() < (min(MAX_SAVE_AS_W, iface_settings.screen_sz.w)-5) {
					if let Result::Ok(c) = u8::try_from(key_pressed) {
						if let Result::Ok(ch) = char::try_from(c) {
							if "012345679".contains(ch) {
								freq.insert(*curs_col as usize, ch);
								*curs_col += 1;
							}
						}
					}
				}
			}
		}
		return UIRet::Active;
	
	//////////////////// get text window
	}else if let UIMode::GetTextWindow {ref mut curs_col, ref mut txt, txt_type} = &mut iface_settings.ui_mode {
		if buttons.Confirm.activated(key_pressed, mouse_event) {
			if txt.len() > 0 {
				let action_type = match txt_type {
					TxtType::BrigadeNm => {
						ActionType::BrigadeCreation {
							nm: txt.clone(),
							start_coord: None,
							end_coord: None
						}
					} TxtType::SectorNm => {
						ActionType::SectorCreation {
							nm: txt.clone(),
							creation_type: SectorCreationType::New,
							start_coord: None,
							end_coord: None
						}
					}
				};
				
				iface_settings.add_action_to = AddActionTo::NoUnit {
					action: ActionMeta::new(action_type),
				};
				
				end_window(iface_settings, d);
				return UIRet::Active;
			}
		}
		
		match key_pressed {
			KEY_LEFT => {if *curs_col != 0 {*curs_col -= 1;}}
			KEY_RIGHT => {
				if *curs_col < (txt.len() as isize) {
					*curs_col += 1;
				}
			}
			
			KEY_HOME | KEY_UP => {*curs_col = 0;}
			
			// end key
			KEY_DOWN | 0x166 | 0602 => {*curs_col = txt.len() as isize;}
			
			// backspace
			KEY_BACKSPACE | 127 | 0x8  => {
				if *curs_col != 0 {
					*curs_col -= 1;
					txt.remove(*curs_col as usize);
				}
			}
			
			// delete
			KEY_DC => {
				if *curs_col != txt.len() as isize {
					txt.remove(*curs_col as usize);
				}
			}
			_ => { // insert character
				if txt.len() < (min(MAX_SAVE_AS_W, iface_settings.screen_sz.w)-5) {
					if let Result::Ok(c) = u8::try_from(key_pressed) {
						if let Result::Ok(ch) = char::try_from(c) {
							txt.insert(*curs_col as usize, ch);
							*curs_col += 1;
						}
					}
				}
			}
		}
		return UIRet::Active;
	
	//////////////////// save window
	}else if let UIMode::SaveAsWindow {ref mut curs_col, ref mut save_nm, ..} = &mut iface_settings.ui_mode {
		if buttons.Save.activated(key_pressed, mouse_event) && save_nm.len() > 0 {
			iface_settings.save_nm = save_nm.clone();
			iface_settings.reset_auto_turn(d); // save_game will clear iface_settings.ui_mode which contains the prior value of the auto turn setting
			//save_game(SaveType::Manual, turn, map_data, exs, zone_exs_owners, bldg_config, bldgs, units, stats, unaffiliated_houses, relations, iface_settings, temps.doctrines, temps.bldgs, temps.units, temps.resources, owners, nms, disp_settings, disp_chars, temps.techs, ai_states, ai_config, barbarian_states, logs, l, frame_stats, rng, d);
			end_window(iface_settings, d);
		}else{
			do_txt_entry_keys!(key_pressed, curs_col, save_nm, Printable::FileNm, iface_settings, d);
		}
		return UIRet::Active;
	
	//////////////////// go to coordinate
	}else if let UIMode::GoToCoordinateWindow {ref mut curs_col, ref mut coordinate, ..} = &mut iface_settings.ui_mode {
		// enter pressed
		if key_pressed == '\n' as i32 && coordinate.len() > 0 {
			let coordinates: Vec<&str> = coordinate.split(",").collect();
			if let Result::Ok(y) = coordinates[0].trim().parse() {
				if let Result::Ok(x) = coordinates[1].trim().parse() {
					let map_sz = map_data.map_szs[iface_settings.zoom_ind];
					if y < map_sz.h as isize {
						iface_settings.center_on_next_unmoved_menu_item(true, FindType::Coord(Coord {y, x}.to_ind(map_sz) as u64), map_data, exs, units, bldgs, relations, players, logs, turn, d);
						iface_settings.reset_auto_turn(d);
						end_window(iface_settings, d);
					}
				}
			}
		// any key except enter was pressed
		}else{
			do_txt_entry_keys!(key_pressed, curs_col, coordinate, Printable::Coordinate, iface_settings, d);
		}
		return UIRet::Active;

	////////////////// plotting
	}else if let UIMode::PlotWindow {ref mut data} = iface_settings.ui_mode {
		match key_pressed {
			k if k == kbd.right as i32 || k == KEY_RIGHT => {
				data.next();
			} k if k == kbd.left as i32 || k == KEY_LEFT => {
				data.prev();
			} _ => {}
		}
		
		if buttons.Esc_to_close.activated(key_pressed, mouse_event) {
			end_window(iface_settings, d);
		}
		
		return UIRet::Active;
	
	////////// world & battle history
	}else if let UIMode::WorldHistoryWindow {..} |
			UIMode::BattleHistoryWindow {..} |
			UIMode::EconomicHistoryWindow {..} = &iface_settings.ui_mode {
		const SCROLL_FASTER_SPEED: usize = 10;
		
		let events = match &iface_settings.ui_mode {
			UIMode::WorldHistoryWindow {..} => {
				world_history_events(iface_settings.cur_player as usize, relations, logs)
			} UIMode::BattleHistoryWindow {..} => {
				battle_history_events(iface_settings.cur_player as usize, relations, logs)
			} UIMode::EconomicHistoryWindow {..} => {
				economic_history_events(iface_settings.cur_player as usize, relations, logs)
			} _ => {panicq!("unhandled UI mode");}
		};
		
		if let UIMode::WorldHistoryWindow {ref mut scroll_first_line} | 
			 UIMode::BattleHistoryWindow {ref mut scroll_first_line} |
			 UIMode::EconomicHistoryWindow {ref mut scroll_first_line} = iface_settings.ui_mode {
			match key_pressed {
				// scroll down
				k if k == kbd.down as i32 || k == KEY_DOWN => {
					let h = iface_settings.screen_sz.h;
					if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
						let n_rows_plot = h - (LOG_START_ROW + LOG_STOP_ROW) as usize;
						
						// check to make sure the log is long enough to keep scrolling
						if (events.len() - *scroll_first_line) > n_rows_plot {
							*scroll_first_line += 1;
						}
					}
				
				// scroll down (faster)
				} k if k == kbd.fast_down as i32 => {
					let h = iface_settings.screen_sz.h;
					if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
						let n_rows_plot = h - (LOG_START_ROW + LOG_STOP_ROW) as usize;
						
						// check to make sure the log is long enough to keep scrolling
						if (events.len() - *scroll_first_line) > n_rows_plot {
							*scroll_first_line += SCROLL_FASTER_SPEED;
							if (events.len() - *scroll_first_line) <= n_rows_plot {
								*scroll_first_line = events.len() - n_rows_plot;
							}
						}
					}
				
				// scroll up
				} k if k == kbd.up as i32 || k == KEY_UP => {
					let h = iface_settings.screen_sz.h;
					if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
						// check to make sure the log is long enough to keep scrolling
						if *scroll_first_line > 0 {
							*scroll_first_line -= 1;
						}
					}
				
				// scroll up (faster)
				} k if k == kbd.fast_up as i32 => {
					let h = iface_settings.screen_sz.h;
					if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
						// check to make sure the log is long enough to keep scrolling
						if *scroll_first_line > 0 {
							if *scroll_first_line >= SCROLL_FASTER_SPEED {
								*scroll_first_line -= SCROLL_FASTER_SPEED;
							}else{
								*scroll_first_line = 0;
							}
						}
					}	
				} _ => {}
			}
		}else{panicq!("unhandled UI mode");}
		
		return UIRet::Active;
	
	////////////
	}else if let UIMode::BrigadeBuildList {ref mut mode, brigade_nm} = &mut iface_settings.ui_mode {
		let list = brigade_build_list(brigade_nm, pstats, l);
		
		// add aciton to brigade build list
		if buttons.Press_to_add_action_to_brigade_build_list.activated(key_pressed, mouse_event) &&
				pstats.brigade_frm_nm(brigade_nm).has_buildable_actions(units) {
			iface_settings.add_action_to = AddActionTo::BrigadeBuildList {
				brigade_nm: brigade_nm.clone(),
				action: None
			};
			
			end_window(iface_settings, d);
			return UIRet::Active;
		}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == KEY_DC as i32 => {
				let brigade = pstats.brigade_frm_nm_mut(brigade_nm);
				if brigade.build_list.len() != 0 {
					brigade.build_list.remove(*mode);
					if *mode >= brigade.build_list.len() {
						*mode = brigade.build_list.len() - 1;
					}
				}				
			} _ => {}
		}
		return UIRet::Active;
		
	///////////////// go to unit, brigade, or building selected
	}else if let UIMode::UnitsWindow {..} |
			UIMode::ImprovementBldgsWindow {..} |
			UIMode::MilitaryBldgsWindow {..} |
			UIMode::BrigadesWindow {..} |
			UIMode::CitiesWindow {..} |
			UIMode::SectorsWindow {..} = iface_settings.ui_mode {
		let cursor_coord = iface_settings.cursor_to_map_coord(map_data); // immutable borrow
		
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		
		let entries = match &iface_settings.ui_mode {
			UIMode::UnitsWindow {..} => {
				owned_unit_list(units, iface_settings.cur_player, cursor_coord, pstats, &mut w, &mut label_txt_opt, map_sz, l)
			} UIMode::CitiesWindow {..} => {
				owned_city_list(bldgs, iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, logs, l)
			} UIMode::ImprovementBldgsWindow {..} => {
				owned_improvement_bldgs_list(bldgs, &temps.doctrines, iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l)
			} UIMode::MilitaryBldgsWindow {..} => {
				owned_military_bldgs_list(bldgs, iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l)
			} UIMode::BrigadesWindow {brigade_action, ..} => {
				match &brigade_action {
					BrigadeAction::Join {..} | BrigadeAction::ViewBrigades => {
						brigades_list(pstats, &mut w, &mut label_txt_opt, l)
					} BrigadeAction::ViewBrigadeUnits {brigade_nm} => {
						brigade_unit_list(brigade_nm, pstats, units, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l)
					}
				}
			} UIMode::SectorsWindow {..} => {
				sector_list(pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l)
			} _ => {panicq!("condition unmatched");}
		};
		let entries_present = entries.options.len() > 0;
		
		macro_rules! close_and_return {() => {
			end_window(iface_settings, d);
			return UIRet::Active;
		};};
		
		// handle buttons
		if entries_present {
			if let UIMode::BrigadesWindow {brigade_action: BrigadeAction::ViewBrigadeUnits {brigade_nm}, mode} = &iface_settings.ui_mode {
				// change brigade repair wall behavior (get the sector to repair walls in)
				if buttons.change_brigade_repair.activated(key_pressed, mouse_event) {
					iface_settings.ui_mode = UIMode::SectorsWindow {
						mode: 0,
						sector_action: SectorAction::SetBrigadeToRepairWalls(brigade_nm.to_string())
					};
					return UIRet::Active;
					
				// clear brigade repair wall behavior
				}else if buttons.clear_brigade_repair.activated(key_pressed, mouse_event) {
					pstats.brigade_frm_nm_mut(brigade_nm).repair_sector_walls = None;

				// assign action to all units in brigade
				}else if buttons.assign_action_to_all_in_brigade.activated(key_pressed, mouse_event) {
					iface_settings.add_action_to = AddActionTo::AllInBrigade {
						brigade_nm: brigade_nm.clone(),
						action_ifaces: None
					};
					close_and_return!();
				
				// add action to brigade build list
				}else if buttons.add_action_to_brigade_build_list.activated(key_pressed, mouse_event) && 
					   pstats.brigade_frm_nm(brigade_nm).has_buildable_actions(units) {
					iface_settings.add_action_to = AddActionTo::BrigadeBuildList {
						brigade_nm: brigade_nm.clone(),
						action: None
					};
					close_and_return!();
					
				// view brigade build list
				}else if buttons.view_brigade.activated(key_pressed, mouse_event) {
					iface_settings.ui_mode = UIMode::BrigadeBuildList {
						mode: *mode,
						brigade_nm: brigade_nm.to_string()
					};
					return UIRet::Active;
				}
			}
		}

		//////////////////////// handle keys
		if let UIMode::UnitsWindow {ref mut mode} |
			UIMode::CitiesWindow {ref mut mode} |
			UIMode::ImprovementBldgsWindow {ref mut mode} |
			UIMode::MilitaryBldgsWindow {ref mut mode} |
			UIMode::BrigadesWindow {ref mut mode, ..} |
			UIMode::SectorsWindow {ref mut mode, ..} = iface_settings.ui_mode {
				
			macro_rules! enter_action {($mode: expr) => {
				// move cursor to entry
				let coord = match entries.options[$mode].arg {
					ArgOptionUI::UnitInd(unit_ind) => {units[unit_ind].return_coord()}
					ArgOptionUI::BldgInd(bldg_ind) => {bldgs[bldg_ind].coord}
					ArgOptionUI::CityInd(city_ind) => {bldgs[city_ind].coord}
					ArgOptionUI::SectorInd(sector_ind) => {
						let pstats = &mut players[iface_settings.cur_player as usize].stats;
						let mode = $mode;
						if let Some(sector) = pstats.sectors.get_mut(sector_ind) {
							if let UIMode::SectorsWindow {sector_action, ..} = &iface_settings.ui_mode {
								match sector_action {
									SectorAction::GoTo => {
										sector.average_coord(map_sz)
									} SectorAction::AddTo => {
										iface_settings.add_action_to = AddActionTo::NoUnit {
											action: ActionMeta::new(
												ActionType::SectorCreation {
													nm: sector.nm.clone(),
													creation_type: SectorCreationType::AddTo,
													start_coord: None,
													end_coord: None
												}),
										};
										
										sector.average_coord(map_sz) // where the cursor / view will move to
									} SectorAction::Delete => {
										pstats.sectors.swap_remove(mode);
										close_and_return!();
									} SectorAction::SetBrigadeToRepairWalls(brigade_nm) => {
										let sector_nm = sector.nm.clone();
										let brigade = pstats.brigade_frm_nm_mut(brigade_nm);
										brigade.repair_sector_walls = Some(sector_nm);
										iface_settings.ui_mode = UIMode::BrigadesWindow {
											mode: 0,
											brigade_action: BrigadeAction::ViewBrigadeUnits {
												brigade_nm: brigade_nm.clone()
											}
										};
										return UIRet::Active;
									}
								}
							}else{panicq!("unknown ui mode");}
						}else{close_and_return!();}
					}
					ArgOptionUI::BrigadeInd(brigade_ind) => {
						// make sure the brigade exists and is non-empty
						if let Some(brigade) = players[iface_settings.cur_player as usize].stats.brigades.get_mut(brigade_ind) {
							*mode = 0;
							if let UIMode::BrigadesWindow {ref mut brigade_action, ..} = iface_settings.ui_mode {
								match brigade_action {
									BrigadeAction::ViewBrigades => {
										*brigade_action = BrigadeAction::ViewBrigadeUnits{brigade_nm: brigade.nm.clone()};
										return UIRet::Active;
									} BrigadeAction::Join {unit_ind} => {
										debug_assertq!(!brigade.unit_inds.contains(unit_ind));
										brigade.unit_inds.push(*unit_ind);
										close_and_return!();
									} BrigadeAction::ViewBrigadeUnits {..} => {panicq!("list should supply unit inds, not brigade inds");}
								}
							}else{panicq!("unknown ui mode");}
						}else{close_and_return!();}
					}
					_ => {panicq!("unit inventory list argument option not properly set");}
				};
				
				iface_settings.center_on_next_unmoved_menu_item(true, FindType::Coord(coord), map_data, exs, units, bldgs, relations, players, logs, turn, d);
				end_window(iface_settings, d);
				
				return UIRet::Active;
			};};
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
			
			match key_pressed {
				k if entries_present && (k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN) => {
					if (*mode + 1) <= (entries.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if entries_present && (k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP) => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = entries.options.len() - 1;
					}
					
				// enter
				} k if entries_present && (k == '\n' as i32) => {
					enter_action!(*mode);
				} _ => {}
			}
			return UIRet::Active;
		}else{panicq!("invalid UI mode");}
	
	///////////////// contact embassy
	}else if let UIMode::ContactEmbassyWindow {ref mut state} = iface_settings.ui_mode {
		match state {
			EmbassyState::CivSelection {ref mut mode} => {
				let list = contacted_civilizations_list(relations, players, iface_settings.cur_player, turn);
				
				macro_rules! enter_action{($mode: expr) => {
					let owner_id = if let ArgOptionUI::OwnerInd(owner_ind) = list.options[$mode].arg {
						owner_ind
					}else{panicq!("list argument option not properly set");};
					
					let quote_category = TxtCategory::from_relations(relations, owner_id, iface_settings.cur_player as usize, players);
					
					*state = EmbassyState::DialogSelection {
						mode: 0,
						owner_id,
						quote_printer: TxtPrinter::new(quote_category, rng.gen())
					};
					return UIRet::Active;
				};};
				if let Some(ind) = buttons.list_item_clicked(mouse_event) {enter_action!(ind);}
				
				match key_pressed {
					// down
					k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
						if (*mode + 1) <= (list.options.len()-1) {
							*mode += 1;
						}else{
							*mode = 0;
						}
					
					// up
					} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
						if *mode > 0 {
							*mode -= 1;
						}else{
							*mode = list.options.len() - 1;
						}
						
					// enter
					} k if k == '\n' as i32 => {
						enter_action!(*mode);
					} _ => {}
				}
			} EmbassyState::DialogSelection {ref mut mode, owner_id, ..} => {
				let cur_player = iface_settings.cur_player as usize;
				let n_options = if relations.at_war(cur_player, *owner_id) {
						2 // threaten, declare peace
					}else if None == relations.peace_treaty_turns_remaining(cur_player, *owner_id, turn) {
						2 // threaten, declare war
					}else {1 // threaten (active peace treaty prevents war)
				};
				
				match key_pressed {
					KEY_UP => {if *mode > 0 {*mode -= 1;} else {*mode = n_options-1;}
					} KEY_DOWN => {if *mode < (n_options-1) {*mode += 1;} else {*mode = 0;}
					// enter
					} k if k == '\n' as i32 => {
						match *mode {
							// threaten
							0 => {
								relations.threaten(cur_player, *owner_id);
								let quote_category = TxtCategory::from_relations(relations, *owner_id, cur_player, players);
								
								*state = EmbassyState::Threaten {
											owner_id: *owner_id,
											quote_printer: TxtPrinter::new(quote_category, rng.gen())
								};
							// declare peace or war
							} 1 => {
								// declare peace (move to treaty proposal page)
								if relations.at_war(*owner_id, cur_player) {
									let quote_category = TxtCategory::from_relations(relations, *owner_id, cur_player, players);
									d.curs_set(CURSOR_VISIBILITY::CURSOR_VISIBLE);
									
									const DEFAULT_GOLD: &str = "0";
									
									*state = EmbassyState::DeclarePeaceTreaty {
												owner_id: *owner_id,
												quote_printer: TxtPrinter::new(quote_category, rng.gen()),
												curs_col: DEFAULT_GOLD.len() as isize,
												gold_offering: String::from(DEFAULT_GOLD),
												treaty_rejected: false
									};
								// declare war
								}else{
									let owner_id = *owner_id;
									relations.declare_war(cur_player, owner_id, logs, players, turn, iface_settings, iface_settings.cur_player_paused(players),
											disp_settings, menu_options, rng, d);
									let quote_category = TxtCategory::from_relations(relations, owner_id, cur_player, players);
									
									// if let statement needed to satisfy the borrow checker
									if let UIMode::ContactEmbassyWindow {ref mut state} = iface_settings.ui_mode {
										*state = EmbassyState::DeclareWar {
												owner_id,
												quote_printer: TxtPrinter::new(quote_category, rng.gen())
										};
									}else{panicq!("ui mode invalid");}
								}
							} _ => {panicq!("unknown menu selection for embassy dialog selection ({})", *mode);}
						}
					} KEY_LEFT => {
						*state = EmbassyState::CivSelection {mode: *owner_id};
					} _ => {}
				}
			} EmbassyState::Threaten {..} | EmbassyState::DeclareWar {..} | EmbassyState::DeclarePeace {..} | EmbassyState::DeclaredWarOn {..} => {}
			
			
			EmbassyState::DeclarePeaceTreaty {owner_id, ref mut gold_offering, ref mut curs_col,
					ref mut treaty_rejected, ref mut quote_printer} => {
				// enter key pressed
				if key_pressed == '\n' as i32 && gold_offering.len() > 0 {
					if let Result::Ok(gold_offering) = gold_offering.parse() {
						if let Some(a_state) = players[*owner_id].ptype.ai_state() {
							// the relevant player has sufficient gold for the treaty
							let gold_sufficient = (gold_offering >= 0. && players[iface_settings.cur_player as usize].stats.gold >= gold_offering) ||
										    (gold_offering < 0. && players[*owner_id].stats.gold >= (-gold_offering));
							
							// ai accepts the treaty
							if gold_sufficient && a_state.accept_peace_treaty(*owner_id, iface_settings.cur_player as usize, gold_offering, relations, players, turn) {
								relations.declare_peace(iface_settings.cur_player as usize, *owner_id, logs, turn);
								
								players[iface_settings.cur_player as usize].stats.gold -= gold_offering;
								players[*owner_id].stats.gold += gold_offering;
								
								let quote_category = TxtCategory::from_relations(relations, *owner_id, iface_settings.cur_player as usize, players);
								
								*state = EmbassyState::DeclarePeace {
											owner_id: *owner_id,
											quote_printer: TxtPrinter::new(quote_category, rng.gen())
								};
								d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
							}else{
								*treaty_rejected = true;
								quote_printer.new_seq();
							}
						}
					}
					
				// enter key not pressed or no gold offered
				}else{
					do_txt_entry_keys!(key_pressed, curs_col, gold_offering, Printable::Numeric, iface_settings, d);
				}	
			}
		}
		return UIRet::Active;
	
	//////////////////// civilization intel
	}else if let UIMode::CivilizationIntelWindow {ref mut mode, ref mut selection_phase} = iface_settings.ui_mode {
		// civilization selection window
		if *selection_phase {
			let list = contacted_civilizations_list(relations, players, iface_settings.cur_player, turn);
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {	
				*mode = ind;
				*selection_phase = false;
				return UIRet::Active;
			}
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (list.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = list.options.len() - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					*selection_phase = false;
					
				} _ => {}
			}
		// display civ info
		}else{
			if key_pressed == KEY_LEFT {
				*selection_phase = true; // go back to selection page
			}
		}
		return UIRet::Active;

	//////////////////// switch to player
	}else if let UIMode::SwitchToPlayerWindow {ref mut mode} = iface_settings.ui_mode {
		let list = all_civilizations_list(players);
		macro_rules! enter_action{($mode: expr) => {
			if let ArgOptionUI::OwnerInd(owner_ind) = list.options[$mode].arg {
				iface_settings.cur_player = owner_ind as SmSvType;
				
			}else{panicq!("invalid UI setting");}
			
			iface_settings.unit_subsel = 0;
			iface_settings.add_action_to = AddActionTo::None;
			
			let pstats = &mut players[iface_settings.cur_player as usize].stats;
			*production_options = init_bldg_prod_windows(temps.bldgs, pstats, l);
			update_menu_indicators(menu_options, iface_settings, iface_settings.cur_player_paused(players), disp_settings);
			compute_zoomed_out_discoveries(map_data, exs, &mut players[iface_settings.cur_player as usize].stats);
			
			end_window(iface_settings, d);
			return UIRet::Active;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == '\n' as i32 => {
				enter_action!(*mode);
			} _ => {}
		}
		return UIRet::Active;
	
	//////////////////// discover technology
	}else if let UIMode::DiscoverTechWindow {ref mut mode} = iface_settings.ui_mode {
		let list = undiscovered_tech_list(&pstats, temps.techs, l);
		macro_rules! enter_action{($mode: expr) => {
			if let ArgOptionUI::TechInd(tech_ind) = list.options[$mode].arg {
				pstats.force_discover_undiscov_tech(tech_ind as SmSvType, temps, production_options, l);
			}else{panicq!("invalid UI setting");}
			
			end_window(iface_settings, d);
			return UIRet::Active;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == '\n' as i32 => {
				enter_action!(*mode);
			} _ => {}
		}
		return UIRet::Active;
	
	//////////////////// obtain resource
	}else if let UIMode::ObtainResourceWindow {ref mut mode} = iface_settings.ui_mode {
		let list = all_resources_list(temps.resources, l);
		macro_rules! enter_action{($mode:expr) => {
			if let ArgOptionUI::ResourceInd(resource_ind) = list.options[$mode].arg {
				for tech_req in temps.resources[resource_ind].tech_req.iter() {
					pstats.force_discover_undiscov_tech((*tech_req) as SmSvType, temps, production_options, l);
				}
				
				pstats.resources_avail[resource_ind] += 1;
				*production_options = init_bldg_prod_windows(temps.bldgs, pstats, l);
			}else{panicq!("invalid UI setting");}
			
			end_window(iface_settings, d);
			return UIRet::Active;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == '\n' as i32 => {
				enter_action!(*mode);
			} _ => {}
		}
		return UIRet::Active;
	
	//////////////////// set difficulty
	}else if let UIMode::SetDifficultyWindow {ref mut mode} = iface_settings.ui_mode {
		let list = game_difficulty_list(game_difficulties);
		macro_rules! enter_action{($mode:expr) => {
			let new_difficulty = &game_difficulties.difficulties[$mode];
			for player in players.iter_mut() {
				if player.ptype.is_human() {continue;}
				
				player.stats.bonuses = new_difficulty.ai_bonuses.clone();
			}
			
			end_window(iface_settings, d);
			return UIRet::Active;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == '\n' as i32 => {
				enter_action!(*mode);
			} _ => {}
		}
		return UIRet::Active;

	//////////////////// place unit
	}else if let UIMode::PlaceUnitWindow {ref mut mode} = iface_settings.ui_mode {
		let list = discovered_units_list(&pstats, temps.units, l);
		macro_rules! enter_action{($mode:expr) => {
			if let ArgOptionUI::UnitTemplate(Some(ut)) = list.options[$mode].arg {
				if iface_settings.zoom_ind == map_data.max_zoom_ind() {
					let c = iface_settings.cursor_to_map_ind(map_data);
					let owner_id = iface_settings.cur_player as usize;
					add_unit(c, true, ut, units, map_data, exs, bldgs, &mut players[owner_id], relations, logs, temps.units, &temps.nms, turn, rng);
				}
			}else{panicq!("invalid UI setting");}
			
			end_window(iface_settings, d);
			return UIRet::Active;
		};};
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
		
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				if (*mode + 1) <= (list.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == '\n' as i32 => {
				enter_action!(*mode);
			} _ => {}
		}
		return UIRet::Active;

	///////////////				            ********* (just accepts exit key) :
	//////////////// show resources available OR
	//////////////// tech discovered, OR about OR ...
	}else if let UIMode::ResourcesAvailableWindow | UIMode::AboutWindow | UIMode::TechDiscoveredWindow {..} |
			UIMode::WarStatusWindow | UIMode::InitialGameWindow | UIMode::EndGameWindow | UIMode::CivicAdvisorsWindow |
			UIMode::UnmovedUnitsNotification | UIMode::RiotingAlert {..} | UIMode::MvWithCursorNoActionsRemainAlert {..} |
			UIMode::CitizenDemandAlert {..} | UIMode::PublicPollingWindow | UIMode::GenericAlert {..} |
			UIMode::PrevailingDoctrineChangedWindow | UIMode::ForeignUnitInSectorAlert {..} = iface_settings.ui_mode {
		
		if buttons.Esc_to_close.activated(key_pressed, mouse_event) {
			end_window(iface_settings, d);
		}
		return UIRet::Active;
		
	//////////////////// resources discovered
	}else if let UIMode::ResourcesDiscoveredWindow {..} = iface_settings.ui_mode {
		let cursor_coord = iface_settings.cursor_to_map_coord(map_data);
		
		if let UIMode::ResourcesDiscoveredWindow {ref mut mode} = iface_settings.ui_mode {
			let list = discovered_resources_list(&pstats, cursor_coord,	temps.resources, *map_data.map_szs.last().unwrap());
			
			macro_rules! enter_action{($mode: expr) => {
				if let ArgOptionUI::ResourceWCoord {coord, ..} = list.options[$mode].arg {
					iface_settings.center_on_next_unmoved_menu_item(true, FindType::Coord(coord), map_data, exs, units, bldgs, relations, players, logs, turn, d);
				}else{panicq!("invalid UI setting");}
				
				end_window(iface_settings, d);
				return UIRet::Active;
			};};
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
			
			const SCROLL_FASTER_SPEED: usize = 3;
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (list.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = list.options.len() - 1;
					}
				
				// down
				} k if k == kbd.fast_down as i32 => {
					if (*mode + SCROLL_FASTER_SPEED) <= (list.options.len()-1) {
						*mode += SCROLL_FASTER_SPEED;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.fast_up as i32 => {
					if *mode >= SCROLL_FASTER_SPEED {
						*mode -= SCROLL_FASTER_SPEED;
					}else{
						*mode = list.options.len() - 1;
					}
				
				// enter
				} k if k == '\n' as i32 => {
					enter_action!(*mode);
				} _ => {}
			}
		}
		
		return UIRet::Active;
		
	//////////////// encyclopedia
	}else if let UIMode::EncyclopediaWindow {ref mut state} = &mut iface_settings.ui_mode {
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
		if let EncyclopediaState::CategorySelection {..} = state {
			let mut category_options = OptionsUI {options: Vec::with_capacity(ENCYCLOPEDIA_CATEGORY_NMS.len()), max_strlen: 0};
			register_shortcuts(ENCYCLOPEDIA_CATEGORY_NMS, &mut category_options);
			
			for (new_menu_ind, option) in category_options.options.iter().enumerate() {
				// match found
				if option.key == Some(key_pressed as u8 as char) {
					*state = EncyclopediaState::ExemplarSelection {
							selection_mode: true,
							category: mode2category(new_menu_ind),
							mode: 0
					};
					return UIRet::Active;
				} // match found
			} // loop over shortcuts
		}
		
		macro_rules! enter_action {() => {
			match state {
				// progress to exemplar selection (ex. unit templates) from main menu
				EncyclopediaState::CategorySelection {mode} => {
					let category = mode2category(*mode);
					*state = EncyclopediaState::ExemplarSelection {
							selection_mode: true,
							category,
							// skip empty first entry for bldgs:
							mode: if category == EncyclopediaCategory::Bldg {1} else {0}
					};
				// progress to showing info screen for exemplar (ex. specific unit template)
				} EncyclopediaState::ExemplarSelection {ref mut selection_mode, ..} => {
					*selection_mode = false;
				}
			}
			return UIRet::Active;
		};};
		
		// list item clicked
		if let Some(ind) = buttons.list_item_clicked(mouse_event) {
			// make sure an inactive menu item wasn't clicked
			if let EncyclopediaState::ExemplarSelection {selection_mode: true, 
					category: EncyclopediaCategory::Bldg, ..} = &state {
				let options = encyclopedia_bldg_list(temps.bldgs, l);
				if let ArgOptionUI::Ind(None) = options.options[ind].arg {
					return UIRet::Active;
				}
			}
			
			match state {
				EncyclopediaState::CategorySelection {ref mut mode} => {*mode = ind;}
				EncyclopediaState::ExemplarSelection {..} => {}
			}
			enter_action!();
		// go back to previous menu
		}else if buttons.to_go_back.activated(key_pressed, mouse_event) {
			match state {
				EncyclopediaState::CategorySelection {..} => {}
				EncyclopediaState::ExemplarSelection {ref mut selection_mode, ..} => {
					if *selection_mode == false {
						*selection_mode = true;
					}else{
						*state = EncyclopediaState::CategorySelection {mode: 0};
					}
				}
			}
		}
		
		// non-shortcut keys
		match key_pressed {
			// down
			k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
				match state {
					EncyclopediaState::CategorySelection {ref mut mode} => {
						*mode += 1;
						if *mode >= N_CATEGORIES {*mode = 0;}
					} EncyclopediaState::ExemplarSelection {selection_mode: true, category, ref mut mode} => {
						// skip empty entries
						if *category == EncyclopediaCategory::Bldg {
							let options = encyclopedia_bldg_list(temps.bldgs, l);
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
					} EncyclopediaState::ExemplarSelection {selection_mode: false, ..} => {}
				}
			
			// up
			} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
				match state {
					EncyclopediaState::CategorySelection {ref mut mode} => {
						if *mode > 0 {*mode -= 1;} else {*mode = N_CATEGORIES-1;}
					} EncyclopediaState::ExemplarSelection {selection_mode: true, category, ref mut mode} => {
						// skip empty entries
						if *category == EncyclopediaCategory::Bldg {
							let options = encyclopedia_bldg_list(temps.bldgs, l);
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
					} EncyclopediaState::ExemplarSelection {selection_mode: false, ..} => {}
				}
			// enter (progress forward to next menu)
			} k if k == '\n' as i32 => {
				enter_action!();
			} _ => {}
		} // end non-shortcut keys
		
		return UIRet::Active;
	
	//////////// unit selected actions
	// - create building
	// - select doctrine for building to build
	}else if let Some(unit_inds) = iface_settings.unit_inds_frm_sel(pstats, units, map_data, exf) {
		if let UIMode::ProdListWindow {ref mut mode} = iface_settings.ui_mode {
			debug_assertq!(iface_settings.zoom_ind == map_data.max_zoom_ind());
			//debug_assertq!(units[unit_ind].template.nm[0] == WORKER_NM);
			
			*production_options = init_bldg_prod_windows(&temps.bldgs, pstats, l);
			
			let opt = &production_options.worker; // get production options for worker
			
			macro_rules! set_production {($ind: expr) => (
				// start production
				if $ind != 0 {
					if let ArgOptionUI::BldgTemplate(Some(bt)) = &opt.options[$ind].arg {
						// choose doctrine to dedicate building?
						if bt.doctrinality_bonus > 0. {
							iface_settings.ui_mode = UIMode::SelectBldgDoctrine {
								mode: 0,
								bldg_template: bt
							};
							return UIRet::Active;
						}else{
							let act = ActionType::WorkerBuildBldg {
									valid_placement: false,
									doctrine_dedication: None,
									template: bt,
									bldg_coord: None 
							};
							iface_settings.start_build_mv_mode(act, &worker_inds(&unit_inds, units), units, map_data);
						}
					}else{panicq!("Option argument not set");}
				}
				end_window(iface_settings, d);
				return UIRet::Active;
			);};
			
			// shortcut key pressed?
			for (new_menu_ind, option) in opt.options.iter().enumerate() {
				// match found
				if option.key == Some(key_pressed as u8 as char) {
					set_production!(new_menu_ind);
				}
			}
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {set_production!(ind);}

			// generic keys
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (opt.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
					
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = opt.options.len() - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					set_production!(*mode);
				} _ => {}
			} // end key match
			
			return UIRet::Active;
		////////////////// select doctrine of building to be built
		}else if let UIMode::SelectBldgDoctrine {ref mut mode, bldg_template} = iface_settings.ui_mode {
			let list = doctrines_available_list(pstats, temps.doctrines, l);
			macro_rules! enter_action{($mode:expr) => {
				if let ArgOptionUI::DoctrineTemplate(Some(doc)) = list.options[$mode].arg {
					let act = ActionType::WorkerBuildBldg {
							valid_placement: false,
							doctrine_dedication: Some(doc),
							template: bldg_template,
							bldg_coord: None 
					};
					iface_settings.start_build_mv_mode(act, &worker_inds(&unit_inds, units), units, map_data);
					
					end_window(iface_settings, d);
					return UIRet::Active;
				}else{panicq!("invalid UI option argument");}
			};};
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {enter_action!(ind);}
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (list.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = list.options.len() - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					enter_action!(*mode);
				} _ => {}
			}
			return UIRet::Active;
		////////////////// select auto-exploration type
		}else if let UIMode::SelectExploreType {ref mut mode} = iface_settings.ui_mode {
			let list = explore_types_list(l);
			
			macro_rules! enter_action{($mode: expr) => {
				let map_sz = *map_data.map_szs.last().unwrap();
				let explore_type = ExploreType::from($mode);
				
				for unit_ind in unit_inds {
					let u = &mut units[unit_ind];
					u.action.pop();
					let land_discov = &players[iface_settings.cur_player as usize].stats.land_discov.last().unwrap();
					
					if let Some(new_action) = explore_type.find_square_unexplored(unit_ind, u.return_coord(), map_data, exs, units, bldgs, land_discov, map_sz, true, rng) {
						units[unit_ind].action.push(new_action);
						mv_unit(unit_ind, true, units, map_data, exs, bldgs, players, relations, map_sz, DelAction::Delete, logs, turn);
						iface_settings.reset_unit_subsel();
						iface_settings.update_all_player_pieces_mvd_flag(units);
					}
				}
				iface_settings.add_action_to = AddActionTo::None;
				end_window(iface_settings, d);
				return UIRet::Active;
			};};
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {enter_action!(ind);}
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (list.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = list.options.len() - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					enter_action!(*mode);
				} _ => {}
			}
			return UIRet::Active;
		/////////// sector automation, select sector (step 1)
		}else if let UIMode::CreateSectorAutomation {sector_nm: None, ..} = iface_settings.ui_mode {
			let cursor_coord = iface_settings.cursor_to_map_coord(map_data); // immutable borrow
			let mut w = 0;
			let mut label_txt_opt = None;
			let map_sz = *map_data.map_szs.last().unwrap();
			let entries = sector_list(pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l);
			let entries_present = entries.options.len() > 0;
			
			if let UIMode::CreateSectorAutomation {ref mut mode, ref mut sector_nm, ..} = iface_settings.ui_mode {
				macro_rules! enter_action{($mode: expr) => {
					*sector_nm = Some(pstats.sectors[$mode].nm.clone());
					*mode = 0;
					return UIRet::Active;
				};};
				if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
				
				match key_pressed {
					// down
					k if entries_present && (k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN) => {
						if (*mode + 1) <= (entries.options.len()-1) {
							*mode += 1;
						}else{
							*mode = 0;
						}
					
					// up
					} k if entries_present && (k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP) => {
						if *mode > 0 {
							*mode -= 1;
						}else{
							*mode = entries.options.len() - 1;
						}
						
					// enter
					} k if entries_present && (k == '\n' as i32) => {
						enter_action!(*mode);
					} _ => {}
				}
				return UIRet::Active;
			}else{panicq!("invalid UI mode");}
		/////////// sector automation, select unit entry action (step 2) (i.e., when a foreign unit enters the sector)
		}else if let UIMode::CreateSectorAutomation {sector_nm: Some(_), unit_enter_action: None, ref mut mode, ..} = iface_settings.ui_mode {
			let unit_enter_actions = [l.Assault_desc.as_str(), l.Defense_desc.as_str(), l.Report_desc.as_str()];
			let mut options = OptionsUI {options: Vec::with_capacity(unit_enter_actions.len()), max_strlen: 0};
			
			register_shortcuts(&unit_enter_actions, &mut options);
			
			let n_options = SectorUnitEnterAction::N as usize;
			
			macro_rules! progress_ui_state {() => {
				// \/ I don't know if it's possible to both match a None value and get a mutable reference of it, hence the following `if` statement
				if let UIMode::CreateSectorAutomation {ref mut unit_enter_action, ref mut mode, ..} = iface_settings.ui_mode {
					*unit_enter_action = Some(SectorUnitEnterAction::from(*mode));
					*mode = 0;
					return UIRet::Active;
				}else{panicq!("invalid UI mode");}
			};};
			
			// shortcut key pressed
			for (option_ind, option) in options.options.iter().enumerate() {
				// match found
				if option.key == Some(key_pressed as u8 as char) {
					*mode = option_ind;
					progress_ui_state!();
				} // match found
			} // loop over shortcuts
			
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {
				*mode = ind;
				progress_ui_state!();
			}
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (n_options-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = n_options - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					progress_ui_state!();
				} _ => {}
			}
			return UIRet::Active;
		/////////// sector automation: what to do when the unit is idle (step 3)
		}else if let UIMode::CreateSectorAutomation {sector_nm: Some(_), unit_enter_action: Some(_), idle_action: None, ref mut mode, ..} = iface_settings.ui_mode {
			let idle_actions = [l.Sentry_desc.as_str(), l.Patrol_desc.as_str()];
			let mut idle_options = OptionsUI {options: Vec::with_capacity(idle_actions.len()), max_strlen: 0};
			
			register_shortcuts(&idle_actions, &mut idle_options);
			let n_options = idle_actions.len();
			
			macro_rules! progress_ui_state {($mode:expr) => {
				// \/ I don't know if it's possible to both match a None value and get a mutable reference of it, hence the following `if` statement
				if let UIMode::CreateSectorAutomation {ref mut idle_action,
						sector_nm: Some(sector_nm), unit_enter_action: Some(unit_enter_action), ..} = &mut iface_settings.ui_mode {
					if $mode == 0 {
						for unit_ind in unit_inds {
							let u = &mut units[unit_ind];
							u.actions_used = None;
							u.action.push(ActionMeta::new(ActionType::SectorAutomation {
								unit_enter_action: *unit_enter_action,
								idle_action: SectorIdleAction::Sentry,
								sector_nm: sector_nm.clone()
							}));
						}
						iface_settings.add_action_to = AddActionTo::None;
						iface_settings.update_all_player_pieces_mvd_flag(units);
						end_window(iface_settings, d);
					
					// progress UI state to ask for the player for `dist_monitor`
					}else{
						*idle_action = Some(SectorIdleAction::Patrol {
							dist_monitor: 0,
							perim_coord_ind: 0,
							perim_coord_turn_computed: 0
						});
						d.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
					}
					return UIRet::Active;
				}else{panicq!("invalid UI mode");}
			};};
			
			// shortcut key pressed
			for (option_ind, option) in idle_options.options.iter().enumerate() {
				// match found
				if option.key == Some(key_pressed as u8 as char) {
					progress_ui_state!(option_ind);
				}
			}
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {	progress_ui_state!(ind);}
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (n_options-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = n_options - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					let mode = *mode;
					progress_ui_state!(mode);
				} _ => {}
			}
			return UIRet::Active;
		/////////// sector automation: get distance away from sector unit should respond to a threat (step 4)
		}else if let UIMode::CreateSectorAutomation {sector_nm: Some(sector_nm), unit_enter_action: Some(unit_enter_action),
				idle_action: Some(SectorIdleAction::Patrol {..}), ref mut curs_col, ref mut txt, ..} = &mut iface_settings.ui_mode {
			
			if buttons.Confirm.activated(key_pressed, mouse_event) {
				if txt.len() > 0 {
					if let Result::Ok(dist_monitor) = txt.parse() {
						for unit_ind in unit_inds {
							units[unit_ind].action.push(ActionMeta::new(ActionType::SectorAutomation {
								unit_enter_action: *unit_enter_action,
								idle_action: SectorIdleAction::Patrol {
									dist_monitor,
									perim_coord_ind: 0,
									perim_coord_turn_computed: 0
								},
								sector_nm: sector_nm.clone()
							}));
						}
						iface_settings.add_action_to = AddActionTo::None;
						iface_settings.update_all_player_pieces_mvd_flag(units);
					}
					end_window(iface_settings, d);
					return UIRet::Active;
				}
			}
			
			match key_pressed {
				KEY_LEFT => {if *curs_col != 0 {*curs_col -= 1;}}
				KEY_RIGHT => {
					if *curs_col < (txt.len() as isize) {
						*curs_col += 1;
					}
				}
				
				KEY_HOME | KEY_UP => {*curs_col = 0;}
				
				// end key
				KEY_DOWN | 0x166 | 0602 => {*curs_col = txt.len() as isize;}
				
				// backspace
				KEY_BACKSPACE | 127 | 0x8  => {
					if *curs_col != 0 {
						*curs_col -= 1;
						txt.remove(*curs_col as usize);
					}
				}
				
				// delete
				KEY_DC => {
					if *curs_col != txt.len() as isize {
						txt.remove(*curs_col as usize);
					}
				}
				_ => { // insert character
					if txt.len() < (min(MAX_SAVE_AS_W, iface_settings.screen_sz.w)-5) {
						if let Result::Ok(c) = u8::try_from(key_pressed) {
							if let Result::Ok(ch) = char::try_from(c) {
								if "0123456789".contains(ch) {
									txt.insert(*curs_col as usize, ch);
									*curs_col += 1;
								}
							}
						}
					}
				}
			}
			return UIRet::Active;
		}
	
	//////////////////// building selected
	////////// add to or remove from production list
	}else if let Some(bldg_ind) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
		let b = &mut bldgs[bldg_ind];
		if let UIMode::ProdListWindow {ref mut mode} = iface_settings.ui_mode {
			debug_assertq!(iface_settings.zoom_ind == map_data.max_zoom_ind());
			
			*production_options = init_bldg_prod_windows(&temps.bldgs, pstats, l);
			
			if let Some(opt) = &production_options.bldgs[b.template.id as usize] { // get production options for current bldg
				macro_rules! set_production {($ind: expr) => (
					match b.args { // unwrap bldg arguments:
						BldgArgs::CityHall {ref mut production, ..} |
						BldgArgs::GenericProducable {ref mut production} => {	
							// start production
							if $ind != 0 {
								if let ArgOptionUI::UnitTemplate(Some(ut)) = &opt.options[$ind].arg {
									production.push(ProductionEntry {
										production: ut,
										progress: 0
									});
								}	
							}/*else{
								*production_progress = None;
								*production = None;
							}*/
							end_window(iface_settings, d);
						} BldgArgs::None => {panicq!("bldg arguments do not store production");}	
					}
					return UIRet::Active;
				);};
				if let Some(ind) = buttons.list_item_clicked(mouse_event) {	set_production!(ind);}
				
				// shortcut key pressed?
				for (new_menu_ind, option) in opt.options.iter().enumerate() {
					// match found
					if option.key == Some(key_pressed as u8 as char) {
						set_production!(new_menu_ind);
					}
				}
				
				// generic keys
				match key_pressed {
					// down
					k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
						if (*mode + 1) <= (opt.options.len()-1) {
							*mode += 1;
						}else{
							*mode = 0;
						}
						
					// up
					} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
						if *mode > 0 {
							*mode -= 1;
						}else{
							*mode = opt.options.len() - 1;
						}
						
					// enter
					} k if k == '\n' as i32 => {
						set_production!(*mode);
					} _ => {}
				}
				
				return UIRet::Active;
			} // unwrapping of production_options for selected bldg
		
			end_window(iface_settings, d);
		}else if let UIMode::CurrentBldgProd {ref mut mode} = iface_settings.ui_mode {
			debug_assertq!(iface_settings.zoom_ind == map_data.max_zoom_ind());
			
			let list = bldg_prod_list(&b, l); 
			macro_rules! enter_action{($mode: expr) => {
				if let BldgArgs::CityHall {ref mut production, ..} |
					 	BldgArgs::GenericProducable {ref mut production, ..} = b.args {
					production.swap_remove($mode);
				}
				return UIRet::Active;
			};};
			if let Some(ind) = buttons.list_item_clicked(mouse_event) {	enter_action!(ind);}
			
			match key_pressed {
				// down
				k if k == kbd.down as i32 || k == kbd.fast_down as i32 || k == KEY_DOWN => {
					if (*mode + 1) <= (list.options.len()-1) {
						*mode += 1;
					}else{
						*mode = 0;
					}
				
				// up
				} k if k == kbd.up as i32 || k == kbd.fast_up as i32 || k == KEY_UP => {
					if *mode > 0 {
						*mode -= 1;
					}else{
						*mode = list.options.len() - 1;
					}
					
				// enter
				} k if k == '\n' as i32 => {
					enter_action!(*mode);
				} _ => {}
			}
			return UIRet::Active;
		}
	}
	return UIRet::Inactive;
}

