use crate::renderer::*;
use crate::disp::*;
use crate::saving::{GameControl};
use crate::gcore::*;
use crate::keyboard::KeyboardMap;
use crate::player::Player;

use super::*;

impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn end_window(&mut self){
		self.reset_auto_turn();
		self.ui_mode = UIMode::None;
		self.state.renderer.clear();
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
	}
}

pub fn tree_window_movement(k: i32, sel_mv: &mut TreeSelMv, tree_offsets: &mut Option<TreeOffsets>,
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

// called in individual window/states/*.rs for list mode updating
// returns true if action should be performed
impl <'bt,'ut,'rt,'dt>OptionsUI<'bt,'ut,'rt,'dt> {
	pub fn list_mode_update_and_action(&self, mode: &mut usize, dstate: &DispState) -> bool {
		if self.options.len() == 0 {return false;}
		
		// mouse click
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {
			*mode = ind;
			return true;
		}
		
		// keyboard
		match dstate.key_pressed {
			k if dstate.kbd.down(k) => {
				if (*mode + 1) <= (self.options.len()-1) {
					*mode += 1;
				}else{
					*mode = 0;
				}
			
			// up
			} k if dstate.kbd.up(k) => {
				if *mode > 0 {
					*mode -= 1;
				}else{
					*mode = self.options.len() - 1;
				}
				
			// enter
			} k if k == dstate.kbd.enter => {
				return true;
				
			// keyboard shortcut?
			} _ => {
				for (mode_val, _) in self.options.iter().enumerate()
						.filter(|(_, option)| option.key == Some(dstate.key_pressed as u8 as char)) {
					*mode = mode_val;
					return true;
				}
			}
		}
		false
	}
}

// called in individual window/states/*.rs for list mode updating
// returns true if action should be performed
pub fn button_mode_update_and_action(mode: &mut usize, buttons: Vec<&mut Button>, key_pressed: i32,
		mouse_event: &Option<MEVENT>, kbd: &KeyboardMap) -> Option<usize> {
	// button is activated
	if let Some(button_ind) = buttons.iter().position(|button| button.activated(0, mouse_event)) {
		return Some(button_ind);
	}
	
	// update hovered
	if let Some(button_ind) = buttons.iter().position(|button| button.hovered(mouse_event)) {
		*mode = button_ind;
	}
	
	// key pressed
	match key_pressed {
		k if kbd.up(k) => {if *mode > 0 {*mode -= 1;} else {*mode = buttons.len()-1;}
		} k if kbd.down(k) => {if *mode < (buttons.len()-1) {*mode += 1;} else {*mode = 0;}
		// enter
		} k if k == kbd.enter => {
			return Some(*mode);
		} _ => {}
	}
	
	None
}

// returns true if window active, false if not active or no longer active
pub fn do_window_keys<'f,'bt,'ut,'rt,'dt>(map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, disp: &mut Disp<'f,'_,'bt,'ut,'rt,'dt>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		game_control: &mut GameControl, frame_stats: &FrameStats, game_difficulties: &GameDifficulties) -> UIRet {
	
	let dstate = &mut disp.state;
	
	if let UIMode::Trade(_) = &disp.ui_mode {} else // <- ignore this window because it will show sub-windows which use Esc_to_close, and we do not want to close the main trade screen when they close
	if dstate.buttons.Esc_to_close.activated(dstate.key_pressed, &dstate.mouse_event) {
		disp.end_window();
		return UIRet::Active;
	}
	
	// mouse hovering over list
	if let Some(ind) = dstate.buttons.list_item_hovered(&dstate.mouse_event) {
		match disp.ui_mode {
			// window w/ inactive entries -- prevent inactive items from being selected
			UIMode::EncyclopediaWindow(
				EncyclopediaWindowState::ExemplarSelection {
					category: EncyclopediaCategory::Bldg,
					ref mut mode, ..
				}
			) => {
				let options = encyclopedia_bldg_list(temps.bldgs, &disp.state.local);
				if let ArgOptionUI::Ind(None) = options.options[ind].arg {
					return UIRet::Active;
				}
				*mode = ind;
				return UIRet::Active;
			}
			
			// windows with lists
			UIMode::ZoneLand(ZoneLandState::GetZoneType {ref mut mode}) |
			UIMode::ZoneLand(ZoneLandState::GetZoneDensity {ref mut mode, ..}) |
			UIMode::IntroNobilityJoinOptions(IntroNobilityJoinOptionsState {ref mut mode, ..}) |
			UIMode::NobilityDeclaresIndependenceWindow(NobilityDeclaresIndependenceWindowState {ref mut mode, ..}) |
			UIMode::CivilizationIntelWindow(CivilizationIntelWindowState {ref mut mode, ..}) |
			UIMode::DiscoverTechWindow(DiscoverTechWindowState {ref mut mode}) |
			UIMode::ObtainResourceWindow(ObtainResourceWindowState {ref mut mode}) |
			UIMode::UnitsWindow(UnitsWindowState {ref mut mode}) |
			UIMode::NobleUnitsWindow(NobleUnitsWindowState {ref mut mode, ..}) |
			UIMode::CitiesWindow(CitiesWindowState {ref mut mode}) |
			UIMode::ManorsWindow(ManorsWindowState {ref mut mode}) |
			UIMode::BldgsWindow(BldgsWindowState {ref mut mode, ..}) |
			UIMode::BrigadesWindow(BrigadesWindowState {ref mut mode, ..}) |
			UIMode::SectorsWindow(SectorsWindowState {ref mut mode, ..}) |
			UIMode::SetDifficultyWindow(SetDifficultyWindowState {ref mut mode}) |
			UIMode::PlaceUnitWindow(PlaceUnitWindowState {ref mut mode}) |
			UIMode::OpenWindow(OpenWindowState {ref mut mode, ..}) |
			UIMode::ResourcesDiscoveredWindow(ResourcesDiscoveredWindowState {ref mut mode}) |
			UIMode::CreateSectorAutomation(CreateSectorAutomationState {ref mut mode, ..}) |
			UIMode::ProdListWindow(ProdListWindowState {ref mut mode}) |
			UIMode::CurrentBldgProd(CurrentBldgProdState {ref mut mode}) |
			UIMode::NobilityRequestWindow(NobilityRequestWindowState {ref mut mode, ..}) |
			UIMode::ViewTrade(ViewTradeState {ref mut mode, ..}) |
			UIMode::ContactNobilityWindow(ContactNobilityState::NobilitySelection {ref mut mode}) |
			UIMode::ContactNobilityWindow(ContactNobilityState::PopulationTargetSelection {ref mut mode, ..}) |
			UIMode::SelectBldgDoctrine(SelectBldgDoctrineState {ref mut mode, ..}) |
			UIMode::Trade( // player is selecting an item to trade
				TradeState {
					add_trade_item: Some(
						AddTradeItemUI {
							state: 
								AddTradeItemStateUI::SelItemType {ref mut mode},
							..
						}
					),
					..
				}
			) |
			UIMode::SelectExploreType(SelectExploreTypeState {ref mut mode}) |
			UIMode::NoblePedigree(NoblePedigreeState {ref mut mode, ..}) |
			UIMode::EncyclopediaWindow(EncyclopediaWindowState::CategorySelection {ref mut mode}) |
			UIMode::EncyclopediaWindow(EncyclopediaWindowState::ExemplarSelection {ref mut mode, ..}) |
			UIMode::BrigadeBuildList(BrigadeBuildListState {ref mut mode, ..}) |
			UIMode::ContactEmbassyWindow(ContactEmbassyWindowState::CivSelection {ref mut mode}) |
			UIMode::SwitchToPlayerWindow(SwitchToPlayerWindowState {ref mut mode}) => {
				*mode = ind;
				return UIRet::Active;
			}
			
			// windows that do not use lists
			UIMode::None |
			UIMode::Trade(_) |
			UIMode::SetNobleTax(_) |
			UIMode::TextTab {..} |
			UIMode::SetTaxes(_) |
			UIMode::Menu {..} |
			UIMode::GenericAlert(_) |
			UIMode::PublicPollingWindow(_) |
			UIMode::CitizenDemandAlert(_) |
			UIMode::SaveAsWindow(_) |
			UIMode::SaveAutoFreqWindow(_) |
			UIMode::TechWindow(_) |
			UIMode::TechDiscoveredWindow(_) |
			UIMode::GetTextWindow(_) |
			UIMode::DoctrineWindow(_) |
			UIMode::PlotWindow(_) |
			UIMode::ContactEmbassyWindow(_) |
			UIMode::ResourcesAvailableWindow(_) |
			UIMode::HistoryWindow(_) |
			UIMode::ContactNobilityWindow(_) |
			UIMode::WarStatusWindow(_) | UIMode::FriendsAndFoesWindow(_) |
			UIMode::GoToCoordinateWindow(_) |
			UIMode::MvWithCursorNoActionsRemainAlert(_) |
			UIMode::PrevailingDoctrineChangedWindow(_) |
			UIMode::RiotingAlert(_) |
			UIMode::CivicAdvisorsWindow(_) | UIMode::InitialGameWindow(_) |
			UIMode::EndGameWindow(_) | UIMode::AboutWindow(_) |
			UIMode::UnmovedUnitsNotification(_) |
			UIMode::AcceptNobilityIntoEmpire(_) |
			UIMode::Budget(_) |
			UIMode::ForeignUnitInSectorAlert(_) => {}
		}
	}
	
	let map_sz = map_data.map_szs.last().unwrap().clone();
	let exf = exs.last().unwrap();
	let cur_player = dstate.iface_settings.cur_player as usize;
	let pstats = &mut players[cur_player].stats;
	
	let ret = match &mut disp.ui_mode {
		UIMode::None | UIMode::SetTaxes(_) | UIMode::TextTab {..} | UIMode::Menu {..} => {return UIRet::Inactive;}
		UIMode::IntroNobilityJoinOptions(state) => {state.keys(players, units, bldgs, map_data, exs, gstate, temps, dstate)}
		UIMode::Trade(state) => {state.keys(players, gstate, temps, map_data, dstate)}
		UIMode::ViewTrade(state) => {state.keys(gstate, players, dstate)}
		UIMode::SelectExploreType(state) => {state.keys(players, units, bldgs, gstate, map_data, exs, dstate)}
		UIMode::ContactNobilityWindow(state) => {state.keys(players, gstate, exs.last().unwrap(), bldgs, dstate)}
		UIMode::SwitchToPlayerWindow(state) => {state.keys(players, temps, map_data, dstate)}
		UIMode::ResourcesDiscoveredWindow(state) => {state.keys(map_data, temps, &players[cur_player].stats, dstate)}
		UIMode::CivilizationIntelWindow(state) => {state.keys(gstate, players, dstate)}
		UIMode::DoctrineWindow(state) => {state.keys(dstate)}
		UIMode::GoToCoordinateWindow(state) => {state.keys(map_data, dstate)}
		UIMode::AcceptNobilityIntoEmpire(state) => {state.keys(gstate, players, dstate)}
		UIMode::NoblePedigree(state) => {state.keys(gstate, players, dstate)}
		UIMode::TechWindow(state) => {state.keys(pstats, temps, dstate)}
		UIMode::ContactEmbassyWindow(state) => {state.keys(players, gstate, temps, dstate)}
		UIMode::OpenWindow(state) => {state.keys(game_control, dstate)}
		UIMode::NobleUnitsWindow(state) => {state.keys(players, gstate, units, map_data, dstate)}
		UIMode::GetTextWindow(state) => {state.keys(dstate)}
		UIMode::EncyclopediaWindow(state) => {state.keys(temps, dstate)}
		UIMode::SaveAsWindow(state) => {state.keys(gstate, map_data, exs, temps, bldgs, units, players, frame_stats, dstate)}
		UIMode::SaveAutoFreqWindow(state) => {state.keys(dstate)}
		UIMode::PlotWindow(state) => {state.keys(dstate)}
		UIMode::PlaceUnitWindow(state) => {state.keys(&mut players[cur_player], temps, map_data, exs, units, bldgs, gstate, dstate)}
		UIMode::CurrentBldgProd(state) => {state.keys(bldgs, map_data, exf, dstate)}
		UIMode::ProdListWindow(state) => {state.keys(units, bldgs, pstats, map_data, exf, temps, dstate)}
		UIMode::SelectBldgDoctrine(state) => {state.keys(units, pstats, map_data, exf, temps, dstate)}
		UIMode::HistoryWindow(state) => {state.keys(gstate, dstate)}
		UIMode::BrigadeBuildList(state) => {state.keys(&mut players[cur_player].stats, units, dstate)}
		UIMode::BrigadesWindow(state) => {state.keys(players, units, map_data, dstate)}
		UIMode::CitiesWindow(state) => {state.keys(bldgs, map_data, gstate, dstate)}
		UIMode::CreateSectorAutomation(state) => {state.keys(pstats, units, map_data, exf, dstate)}
		UIMode::BldgsWindow(state) => {state.keys(temps, bldgs, map_data, dstate)}
		UIMode::UnitsWindow(state) => {state.keys(players, units, map_data, dstate)}
		UIMode::ManorsWindow(state) => {state.keys(players, bldgs, gstate, map_data, dstate)}
		UIMode::SectorsWindow(state) => {state.keys(players, map_data, map_sz, dstate)}
		UIMode::DiscoverTechWindow(state) => {state.keys(pstats, temps, dstate)}
		UIMode::ObtainResourceWindow(state) => {state.keys(pstats, temps, dstate)}
		UIMode::SetDifficultyWindow(state) => {state.keys(players, game_difficulties, dstate)}
		UIMode::NobilityRequestWindow(state) => {state.keys(players, bldgs, gstate, temps, map_data, exs, dstate)}
		UIMode::InitialGameWindow(state) => {state.keys(players, temps, map_data, exf, map_sz, gstate, dstate)}
		UIMode::NobilityDeclaresIndependenceWindow(state) => {state.keys(players, gstate, dstate)}
		UIMode::ZoneLand(state) => state.keys(units, players, map_data, exs.last().unwrap(), dstate),
		UIMode::SetNobleTax(state) => {state.keys(&mut gstate.relations, dstate)}
		UIMode::Budget(state) => state.keys(),
		
		// no key actions
		UIMode::ResourcesAvailableWindow(_) | UIMode::AboutWindow(_) | UIMode::TechDiscoveredWindow(_) | UIMode::FriendsAndFoesWindow(_) |
		UIMode::WarStatusWindow(_) | UIMode::EndGameWindow(_) | UIMode::CivicAdvisorsWindow(_) |
		UIMode::UnmovedUnitsNotification(_) | UIMode::RiotingAlert(_) | UIMode::MvWithCursorNoActionsRemainAlert(_) |
		UIMode::CitizenDemandAlert(_) | UIMode::PublicPollingWindow(_) | UIMode::GenericAlert(_) |
		UIMode::PrevailingDoctrineChangedWindow(_) | UIMode::ForeignUnitInSectorAlert(_) => {
			UIModeControl::UnChgd
		}
	};
	
	match ret {
		UIModeControl::CloseAndGoTo(coord) => { // go to coordinate
			disp.reset_auto_turn();
			disp.end_window();
			disp.center_on_next_unmoved_menu_item(true, FindType::Coord(coord), map_data, exs, units, bldgs, gstate, players);
			UIRet::Inactive
		} UIModeControl::Closed => {
			disp.end_window();
			UIRet::Active
		} UIModeControl::New(new) => {
			disp.ui_mode = new;
			UIRet::Active
		} UIModeControl::ChgGameControl => UIRet::ChgGameControl,
		  UIModeControl::UnChgd => UIRet::Active
	}
}

