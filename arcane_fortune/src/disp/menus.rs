use std::process::exit;

use crate::renderer::*;
use crate::disp::*;
use crate::saving::*;
use crate::config_load::return_save_files;
use crate::resources::ResourceTemplate;
use crate::doctrine::*;
use crate::tech::*;
use crate::gcore::GameDifficulties;
use crate::player::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

use super::vars::IfaceSettings;
use super::color::DispChars;

const MENU_STR: &str = "Menu: ";
//const MENU_C1: char = 'M';
const MENU_1STR: &str = "enu: ";

const MENU_DELIMC: char = '|';

const MENU_INACTIVEC: char = '\\'; // if present as first char of entry indicates inactive item
macro_rules! inact {($s: tt) => (concat!("\\", $s))}

///// main menu
const MENU_NMS: &[&str] = &["F|ile", "A|c|counting", "Inte|l|", "V|iew", "G|o", "P|references", "H|elp"];

const SUB_MENU_NMS_INIT: &[&[&str]] = &[&[
	// file
	"O|pen", "N|ew", "S|ave", "Save |A|s", "E|x|it"],
	
	// accounting
	&[inact!("Military"),
		"  |B|attalions (units)", "  B|r|igades (groups of units)",
		"    Create |n|ew brigade",
	inact!("Cities"),
		"  |C|ity Halls", "  |M|ilitary buildings", "  |I|mprovement buildings",
	inact!("Noble houses"),
		"  Man|o|rs", "  Battalion|s| (units)",
	inact!("Resources"),
		"  |A|vailable", "  |D|iscovered"],
	
	// intel
	&[inact!("Plots"),
			"  |D|efensive power", "  |O|ffensive power", "  |P|opulation", 
			"  |R|esearch output", "  |T|ech development",
			inact!("  Economy"),
			"    |U|nemployed", "    |G|old", "    |N|et income", "    |Z|one demand", 
			inact!("  Life & Culture"),
			"    Happin|e|ss", "    Cri|m|e", "    Doctrinality - Methodicali|s|m", "    Your prevailing doctrines (|1|)",
			"    World prevailing doctrines (|2|)",
			"    Citizen p|a|cifism - militarism", "    Hea|l|th",
	inact!("Domestic intelligence"),
			"  Civic advisors (|3|)", "  Public polling (|4|)", "  Contact noble house (|5|)",
	inact!("International affairs"),
			"  |C|ontact embassy", "  Civilization |i|ntel", "  Active |w|ars & treaties", "  Friends & foes (|6|)"],
	
	// view
	&[inact!("Underlays (select one)"), "   |A|rability", "   |E|levation", "   Water & mountains |o|nly",
		inact!("Overlay Items"), "   |S|tructures", "   |U|nits", "   |B|uildings",  "   |Z|ones", "   |R|esources",
						"   Show un|c|onnected bldgs", "   Show unoccup|i|ed bldgs", "   Show sectors (|2|)",
		inact!("Civic Overlays"),
					"   Zone |d|emands",
					"   Happi|n|ess", "   Crime (|k|)",
		inact!("Trees"), "  |T|ech tree", "  Doctrine tree (|1|)", "  Noble pedigree (|3|)",
		inact!("History"), "  |W|orld", "  Battle (|h|)", "  Econo|m|ic"],
	
	// go	
	&["Ctr |o|n cursor (Space)", "N|ext unmoved unit (or press ';' when not in the menu)", "Next |C|ity Hall", "T|o coordinate",
	  "To map |s|ector",
		inact!("Sectors"),
		"    C|r|eate sector", "    |A|dd to sector", "    |D|elete sector"],
	
	// preferences
	&["S|ave auto-frequency", 
	inact!("Auto turn"), "   |A|uto turn increment", "   |I|nterrupt auto turn for important events",
	inact!("Terminal/Display"), "   |A|void special chars", "   |U|se only 8 colors",
	#[cfg(feature="sdl")]
	"   Toggle fullscreen m|o|de (F11)",
	#[cfg(feature="sdl")]
	"   Increase font (Ctrl +, Ctrl mouse wheel)",
	#[cfg(feature="sdl")]
	"   Decrease font (Ctrl -, Ctrl mouse wheel)",
	inact!("Misc"), "   Workers create city sectors",
	inact!("Omnipresence (cheating)"), "   |P|lace unit at cursor", "   Obtain |r|esource", "   Free mone|y|", 
	inact!("   Discovery"),"      |D|iscover all civs","      Disco|v|er specific technology", "      Discover al|l| technology",
	inact!("   Information"), "      Show full |m|ap", "      Show all unit ac|t|ions",        "      Show all zon|e| information" ,
	inact!("   Game state"),  "      S|w|itch to player", "      |C|hange game difficulty", "      Pause current AI's actio|n|s"],
	
	// help
	&["E|ncyclopedia", inact!("Performance"), "   |M|PD vs time", "A|bout"]];

#[derive(Clone)]
pub enum ArgOptionUI<'bt,'ut,'rt,'dt> {
	Blank, // sub-menus do not have further sub-menus ("sub_options")
	MainMenu {
	           col_start: usize, // ex: col to show "View" in top row of: Menu: | File | View | Go ..
		   sub_options: OptionsUI<'bt,'ut,'rt,'dt> // sub menu entries, ex: for file "Open, New, Save..."
	},
	Ind(Option<usize>),
	UnitTemplate(Option<&'ut UnitTemplate<'rt>>),
	BldgTemplate(Option<&'bt BldgTemplate<'ut,'rt,'dt>>),
	DoctrineTemplate(Option<&'dt DoctrineTemplate>),
	UnitInd(usize),
	BrigadeInd(usize),
	SectorInd(usize),
	BldgInd(usize),
	CityInd(usize),
	OwnerInd(usize),
	TechInd(usize),
	ResourceInd(usize),
	//Coord(u64),
	ResourceWCoord {rt: &'rt ResourceTemplate, coord: u64}
}

#[derive(Clone)]
pub struct OptionUI<'bt,'ut,'rt,'dt> {
	pub key: Option<char>, // shortcut key
	dyn_str: String, // may need to modify this for indicators
	strlen: usize, // sz of dyn_str w/o delims ("|")
	
	pub arg: ArgOptionUI<'bt,'ut,'rt,'dt> // embeds submenu when = ArgOptionUI::MainMenu {..}
}

// Series of menu items (or sub-menu items, or producable units for bldgs)
#[derive(Clone)]
pub struct OptionsUI<'bt,'ut,'rt,'dt> {
	pub options: Vec<OptionUI<'bt,'ut,'rt,'dt>>,
	pub max_strlen: usize // for display. max len across strlens[:]
}

////////////////////////////////////////////////////////////
// actions
//////////////////////////////////////////////////////////////
pub enum FindType {
	Units,
	CityHall,
	Coord(u64)
}

impl <'f,'bt,'ut,'rt,'dt>IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	fn find_next_unit(&mut self, map_data: &mut MapData<'rt>, exf: &HashedMapEx, units: &Vec<Unit>) -> Option<u64> {
		macro_rules! check_if_owned_avail{($u: expr) => (
			if $u.actions_used != None && $u.owner_id == self.cur_player && $u.action.len() == 0 {
				return Some($u.return_coord());
			});}
		
		// unit already selected at cursor?
		if self.zoom_ind == map_data.max_zoom_ind() {
			if let Some(unit_ind_cur) = self.unit_ind_frm_cursor(units, map_data, exf) { // checks cur_player owns it	
				// start from next unit on list and see if any are owned by cur_player
				for u in units[unit_ind_cur+1..].iter() {
					check_if_owned_avail!(u);
				}
			}
		}
				
		// when nothing is currently selected by cursor:
		for u in units {
			check_if_owned_avail!(u);
		} 
		
		None
	}

	fn find_next_city_hall(&mut self, map_data: &mut MapData<'rt>, exf: &HashedMapEx, bldgs: &Vec<Bldg>) -> Option<u64> {
		macro_rules! check_if_owned_avail{($b: expr) => (
			if $b.owner_id == self.cur_player && $b.template.nm[0] == CITY_HALL_NM {
				return Some($b.coord);
			});}
		

		// bldg already selected at cursor?
		if self.zoom_ind == map_data.max_zoom_ind() {
			if let Some(bldg_ind_cur) = self.bldg_ind_frm_cursor(bldgs, map_data, exf) { // checks cur_player owns it	
				if bldgs[bldg_ind_cur].template.nm[0] == CITY_HALL_NM {
					for b in bldgs[bldg_ind_cur+1..].iter() {
						check_if_owned_avail!(b);
					}
				}
			}
		}
				
		// when nothing is currently selected by cursor:
		for b in bldgs {
			check_if_owned_avail!(b);
		} 
		
		None
	}
}

pub fn start_menu_sep(ui_mode: &mut UIMode, iface_settings: &mut IfaceSettings, renderer: &mut Renderer) {
	*ui_mode = UIMode::Menu {
		mode: None,
		sub_mode: None,
		sel_loc: (0,0),
		prev_auto_turn: iface_settings.auto_turn
	};
	set_auto_turn_sep(AutoTurn::Off, iface_settings, renderer);
	renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
}

impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'_,'bt,'ut,'rt,'dt> {
	// reset auto turn to setting before the window was launched (stored in `prev_auto_turn`)
	pub fn reset_auto_turn(&mut self) {
		// set prior value of auto turn increment
		// (will already be set to None if we are changing the value of auto_turn_increment)
		match self.ui_mode {
			UIMode::Menu {prev_auto_turn, ..} |  
			UIMode::SaveAsWindow(SaveAsWindowState {prev_auto_turn, ..}) |
			UIMode::OpenWindow(OpenWindowState {prev_auto_turn, ..}) |
			UIMode::DoctrineWindow(DoctrineWindowState {prev_auto_turn, ..}) |
			UIMode::TechWindow(TechWindowState {prev_auto_turn, ..}) |
			UIMode::TechDiscoveredWindow(TechDiscoveredWindowState {prev_auto_turn, ..}) => {
				self.state.set_auto_turn(prev_auto_turn);
			} 
			UIMode::None |
			UIMode::TextTab {..} |
			UIMode::SetTaxes(_) |
			UIMode::Trade(_) |
			UIMode::ViewTrade(_) |
			UIMode::SetNobleTax(_) |
			UIMode::ProdListWindow(_) |
			UIMode::GenericAlert(_) |
			UIMode::PublicPollingWindow(_) |
			UIMode::CurrentBldgProd(_) |
			UIMode::ContactNobilityWindow(_) |
			UIMode::NobilityRequestWindow(_) |
			UIMode::SelectBldgDoctrine(_) |
			UIMode::SelectExploreType(_) |
			UIMode::SaveAutoFreqWindow(_) |
			UIMode::GetTextWindow(_) |
			UIMode::PlotWindow(_) |
			UIMode::CreateSectorAutomation(_) |
			UIMode::UnitsWindow(_) |
			UIMode::NobleUnitsWindow(_) |
			UIMode::SectorsWindow(_) |
			UIMode::CitizenDemandAlert(_) |
			UIMode::BrigadesWindow(_) |
			UIMode::BrigadeBuildList(_) |
			UIMode::BldgsWindow(_) |
			UIMode::CitiesWindow(_) |
			UIMode::ManorsWindow(_) |
			UIMode::ContactEmbassyWindow(_) |
			UIMode::CivilizationIntelWindow(_) |
			UIMode::MvWithCursorNoActionsRemainAlert(_) |
			UIMode::SwitchToPlayerWindow(_) |
			UIMode::SetDifficultyWindow(_) |
			UIMode::DiscoverTechWindow(_) |
			UIMode::ObtainResourceWindow(_) |
			UIMode::PlaceUnitWindow(_) |
			UIMode::ResourcesAvailableWindow(_) |
			UIMode::ResourcesDiscoveredWindow(_) |
			UIMode::HistoryWindow(_) |
			UIMode::WarStatusWindow(_) | UIMode::FriendsAndFoesWindow(_) |
			UIMode::EncyclopediaWindow(_) |
			UIMode::GoToCoordinateWindow(_) |
			UIMode::InitialGameWindow(_) | UIMode::IntroNobilityJoinOptions(_) |
			UIMode::EndGameWindow(_) |
			UIMode::NoblePedigree(_) |
			UIMode::UnmovedUnitsNotification(_) |
			UIMode::PrevailingDoctrineChangedWindow(_) |
			UIMode::CivicAdvisorsWindow(_) |
			UIMode::ForeignUnitInSectorAlert(_) |
			UIMode::AcceptNobilityIntoEmpire(_) | UIMode::NobilityDeclaresIndependenceWindow(_) |
			UIMode::RiotingAlert(_) |
			UIMode::AboutWindow(_) => {}
		}
	}

	pub fn center_on_next_unmoved_menu_item(&mut self, update_mv_search: bool, find_type: FindType, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		let exf = exs.last().unwrap();
		let iface_settings = &mut self.state.iface_settings;
		let found = match find_type {
			FindType::Units => {iface_settings.find_next_unit(map_data, exf, units)}
			FindType::CityHall => {iface_settings.find_next_city_hall(map_data, exf, bldgs)}
			FindType::Coord(coord) => {Some(coord)}
		};
		
		if let Some(coord) = found {
			iface_settings.zoom_ind = map_data.max_zoom_ind();
			let map_sz = map_data.map_szs[iface_settings.zoom_ind];
			
			let coord = Coord::frm_ind(coord, map_sz);
			
			// center screen on unit
			iface_settings.map_loc = Coord { y: coord.y - ((iface_settings.map_screen_sz.h/2) as isize),
							 x: coord.x - ((iface_settings.map_screen_sz.w/2) as isize) };
			
			// wrap screen horizontally
			if iface_settings.map_loc.x < 0 {
				let d = map_sz.w as isize;
				iface_settings.map_loc.x = d + (iface_settings.map_loc.x % d);
			}
			
			// if at bottom of map, bring back map_loc.y to not be out of bounds
			iface_settings.chk_cursor_bounds(map_data); 
			
			iface_settings.cur = coord.to_screen_coords(iface_settings.map_loc, iface_settings.map_screen_sz).unwrap();
			
			if update_mv_search {self.update_move_search_ui(map_data, exs, units, bldgs, gstate, players, map_sz);}
		}
	}
	
	pub fn start_menu(&mut self) {
		start_menu_sep(&mut self.ui_mode, &mut self.state.iface_settings, &mut self.state.renderer);
	}

	fn end_menu(&mut self) {
		self.reset_auto_turn();
		self.ui_mode = UIMode::None;
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
	}
}

fn execute_submenu<'f,'bt,'ut,'rt,'dt>(menu_mode: usize, sub_menu_mode: usize, disp: &mut Disp<'f,'_,'bt,'ut,'rt,'dt>, map_data: &mut MapData<'rt>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		gstate: &mut GameState,	frame_stats: &FrameStats, game_opts: &mut GameOptions, game_difficulties: &GameDifficulties) -> Option<GameControl> {
	
	// ret true/false depending on if two strings match main menu, then sub menu
	let m = |s1: &str, s2: &str| -> bool {
		if disp.state.menu_options.options[menu_mode].dyn_str.contains(s1) {
			if let ArgOptionUI::MainMenu {sub_options, ..} = &disp.state.menu_options.options[menu_mode].arg {
				if sub_options.options[sub_menu_mode].dyn_str.contains(s2) {return true;}
			}else{panicq!("main menu arguments not set");}
		}
		return false;
	};
	
	macro_rules! update_indicators{()=> (disp.state.update_menu_indicators(disp.state.iface_settings.cur_player_paused(players)););}
	
	if m("F|ile", "E|x|it") {
		endwin();
		#[cfg(feature="profile")]
		write_prof();
		exit(0);
	}else if m("F|ile", "S|ave") {
		disp.reset_auto_turn(); // disp.ui_mode is cleared by save_game(), so the tmp settings of auto turn are lost
		save_game(SaveType::Manual, gstate, map_data, exs, temps, bldgs, units, players, &mut disp.state, frame_stats);
	
	}else if m("F|ile", "Save |A|s") {
		if let UIMode::Menu {prev_auto_turn, ..} = disp.ui_mode {
			disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
			
			disp.ui_mode = UIMode::SaveAsWindow(SaveAsWindowState {
				prev_auto_turn,
				save_nm: disp.state.iface_settings.save_nm.clone(),
				curs_col: disp.state.iface_settings.save_nm.len() as isize
			});
		}else{panicq!("invalid UI mode setting, save as");}
		
		return None;
		
	}else if m("F|ile", "O|pen") {
		if let UIMode::Menu {prev_auto_turn, ..} = disp.ui_mode {
			disp.ui_mode = UIMode::OpenWindow(OpenWindowState {
				prev_auto_turn,
				save_files: return_save_files(),
				mode: 0
			});
		}else{panicq!("invalid UI mode setting");}
		
		return None;
		//return Some(GameControl::Load);
		
	}else if m("F|ile", "N|ew") {
		if disp.state.new_game_options(game_opts, game_difficulties) {
			return Some(GameControl::New);
		}
		
	}else if m("A|c|counting", "B|attalions (units)") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::UnitsWindow(UnitsWindowState {mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "B|r|igades (groups of units)") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::BrigadesWindow(BrigadesWindowState {mode: 0, brigade_action: BrigadeAction::ViewBrigades});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "Create |n|ew brigade") {
		let txt = players[disp.state.iface_settings.cur_player as usize].stats.new_brigade_nm(&temps.nms, &mut gstate.rng);
		disp.reset_auto_turn();
		disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		
		disp.ui_mode = UIMode::GetTextWindow(GetTextWindowState {
			curs_col: txt.len() as isize,
			txt_type: TxtType::BrigadeNm,
			txt
		});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "M|ilitary buildings") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::BldgsWindow(BldgsWindowState {mode: 0, bldgs_show: BldgsShow::Military});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "I|mprovement buildings") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::BldgsWindow(BldgsWindowState {mode: 0, bldgs_show: BldgsShow::Improvements});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "|C|ity Halls") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::CitiesWindow(CitiesWindowState{mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "Man|o|rs") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::ManorsWindow(ManorsWindowState {mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "Battalion|s| (units)") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::NobleUnitsWindow(NobleUnitsWindowState{mode: 0, house_nm: None});
		return None;
	
	}else if m("Inte|l|", "C|ontact embassy") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::ContactEmbassyWindow(ContactEmbassyWindowState::CivSelection {mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l|", "Civilization |i|ntel") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::CivilizationIntelWindow(CivilizationIntelWindowState {mode: 0, selection_phase: true});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l|", "Active |w|ars & treaties") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::WarStatusWindow(WarStatusWindowState {});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l|", "Friends & foes") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::FriendsAndFoesWindow(FriendsAndFoesWindowState {});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "|A|vailable") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::ResourcesAvailableWindow(ResourcesAvailableWindowState {});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("A|c|counting", "|D|iscovered") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::ResourcesDiscoveredWindow(ResourcesDiscoveredWindowState {mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "D|efensive power") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::DefensivePower});
		return None; // return because end_menu() will overwrite disp.ui_mode

	}else if m("Inte|l", "O|ffensive power") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::OffensivePower});
		return None; // return because end_menu() will overwrite disp.ui_mode

	}else if m("Inte|l", "P|opulation") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Population});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "U|nemployed") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Unemployed});
		return None; // return because end_menu() will overwrite disp.ui_mode

	}else if m("Inte|l", "G|old") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Gold});
		return None; // return because end_menu() will overwrite disp.ui_mode

	}else if m("Inte|l", "N|et income") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::NetIncome});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "R|esearch output") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::ResearchPerTurn});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "T|ech development") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::ResearchCompleted});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Z|one demand") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::ZoneDemands});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Happin|e|ss") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Happiness});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Cri|m|e") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Crime});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Doctrinality - Methodicali|s|m") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::DoctrineScienceAxis});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Your prevailing doctrines") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::YourPrevailingDoctrines});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "World prevailing doctrines") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::WorldPrevailingDoctrines});
		return None; // return because end_menu() will overwrite disp.ui_mode

	}else if m("Inte|l", "Citizen p|a|cifism") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Pacifism});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Hea|l|th") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::Health});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Civic advisors (|3|)") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::CivicAdvisorsWindow(CivicAdvisorsWindowState {});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Public polling (|4|)") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PublicPollingWindow(PublicPollingWindowState {});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("Inte|l", "Contact noble house") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::ContactNobilityWindow(ContactNobilityState::new());
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("V|iew", "|A|rability"){	
		disp.state.iface_settings.underlay = Underlay::Arability;  update_indicators!();
		
	}else if m("V|iew", "|E|levation"){
		disp.state.iface_settings.underlay = Underlay::Elevation;  update_indicators!();
		
	}else if m("V|iew", "Water & mountains |o|nly"){
		disp.state.iface_settings.underlay = Underlay::WaterMountains;  update_indicators!();
		
	}else if m("V|iew", "|S|tructures"){
		disp.state.iface_settings.show_structures ^= true;  update_indicators!();
	
	}else if m("V|iew", "|U|nits"){
		disp.state.iface_settings.show_units ^= true;  update_indicators!();
	
	}else if m("V|iew", "|B|uildings"){
		disp.state.iface_settings.show_bldgs ^= true;  update_indicators!();
	
	}else if m("V|iew", "|Z|ones"){
		disp.state.iface_settings.show_zones ^= true;  update_indicators!();
		
	}else if m("V|iew", "|R|esources"){
		disp.state.iface_settings.show_resources ^= true;  update_indicators!();
	
	}else if m("V|iew", "Zone |d|emands"){
		disp.state.iface_settings.zone_overlay_map = 
			if disp.state.iface_settings.zone_overlay_map == ZoneOverlayMap::ZoneDemands {
				ZoneOverlayMap::None
			}else{
				ZoneOverlayMap::ZoneDemands
			};
		update_indicators!();
	
	}else if m("V|iew", "Happi|n|ess"){
		disp.state.iface_settings.zone_overlay_map = 
			if disp.state.iface_settings.zone_overlay_map == ZoneOverlayMap::Happiness {
				ZoneOverlayMap::None
			}else{
				ZoneOverlayMap::Happiness
			};
		update_indicators!();
	
	}else if m("V|iew", "Crime (|k|)"){
		disp.state.iface_settings.zone_overlay_map = 
			if disp.state.iface_settings.zone_overlay_map == ZoneOverlayMap::Crime {
				ZoneOverlayMap::None
			}else{
				ZoneOverlayMap::Crime
			};
		update_indicators!();
	
	}else if m("V|iew", "Show un|c|onnected bldgs"){
		disp.state.iface_settings.show_unconnected_bldgs ^= true;  update_indicators!();
	
	}else if m("V|iew", "Show unoccup|i|ed bldgs"){
		disp.state.iface_settings.show_unoccupied_bldgs ^= true;  update_indicators!();
		
	}else if m("V|iew", "Show sectors"){
		disp.state.iface_settings.show_sectors ^= true;  update_indicators!();
		
	}else if m("V|iew", "|T|ech tree"){
		disp.create_tech_window(false);
		return None;
	
	}else if m("V|iew", "Doctrine tree (|1|)"){
		disp.create_spirituality_window();
		return None;
	}else if m("V|iew", "Noble pedigree") {
		disp.create_pedigree_window();
		return None;
	}else if m("V|iew", "W|orld") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::HistoryWindow(HistoryWindowState {scroll_first_line: 0, htype: HistoryType::World});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("V|iew", "Battle (|h|)") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::HistoryWindow(HistoryWindowState {scroll_first_line: 0, htype: HistoryType::Battle});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("V|iew", "Econo|m|ic") {
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::HistoryWindow(HistoryWindowState {scroll_first_line: 0, htype: HistoryType::Economic});
		return None; // return because end_menu() will overwrite disp.ui_mode

	}else if m("G|o", "Ctr |o|n cursor (Space)"){
		disp.state.iface_settings.ctr_on_cur(map_data);
		
	}else if m("G|o", "N|ext unmoved unit"){
		disp.center_on_next_unmoved_menu_item(true, FindType::Units, map_data, exs, units, bldgs, gstate, players);
	}else if m("G|o", "Next |C|ity Hall"){
		disp.center_on_next_unmoved_menu_item(true, FindType::CityHall, map_data, exs, units, bldgs, gstate, players);
	}else if m("G|o", "T|o coordinate"){
		disp.ui_mode = UIMode::GoToCoordinateWindow(GoToCoordinateWindowState::new(map_data, &mut disp.state));
		
		return None;
	}else if m("G|o", "To map |s|ector"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::SectorsWindow(SectorsWindowState {mode: 0, sector_action: SectorAction::GoTo});
		return None;
	}else if m("G|o", "C|r|eate sector"){
		let txt = players[disp.state.iface_settings.cur_player as usize].stats.new_sector_nm(&temps.nms);
		disp.reset_auto_turn();
		disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		
		disp.ui_mode = UIMode::GetTextWindow(GetTextWindowState {
			curs_col: txt.len() as isize,
			txt_type: TxtType::SectorNm,
			txt
		});
		return None;
	
	}else if m("G|o", "|A|dd to sector"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::SectorsWindow(SectorsWindowState {mode: 0, sector_action: SectorAction::AddTo});
		return None;

	}else if m("G|o", "|D|elete sector"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::SectorsWindow(SectorsWindowState {mode: 0, sector_action: SectorAction::Delete});
		return None;

	}else if m("P|references", "|A|void special chars"){
		disp.state.terminal.limit_schars ^= true;  update_indicators!();
		disp.state.chars = init_color_pairs(&disp.state.terminal, &mut disp.state.renderer);
	}else if m("P|references", "|U|se only 8 colors"){
		disp.state.terminal.limit_colors ^= true;  update_indicators!();
		disp.state.chars = init_color_pairs(&disp.state.terminal, &mut disp.state.renderer);
	}else if m("P|references", "Toggle fullscreen m|o|de (F11)") {
		disp.state.renderer.toggle_fullscreen();
	}else if m("P|references", "Increase font ") {
		disp.state.renderer.inc_font_sz();
	}else if m("P|references", "Decrease font ") {
		disp.state.renderer.dec_font_sz();
	}else if m("P|references", "Show full |m|ap"){
		disp.state.iface_settings.show_fog ^= true;  update_indicators!();
	}else if m("P|references", "Show all unit ac|t|ions"){
		disp.state.iface_settings.show_actions ^= true;  update_indicators!();
	}else if m("P|references", "|D|iscover all civs"){
		gstate.relations.discover_all_civs(disp.state.iface_settings.cur_player as usize);
	
	}else if m("P|references", "S|ave auto-frequency"){
		let freq = format!("{}", disp.state.iface_settings.checkpoint_freq);
		disp.reset_auto_turn();
		disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		
		disp.ui_mode = UIMode::SaveAutoFreqWindow(SaveAutoFreqWindowState {
			curs_col: freq.len() as isize,
			freq
		});
		return None;
	}else if m("P|references", "|A|uto turn increment"){	
		// toggle auto turn
		if let UIMode::Menu {prev_auto_turn, ..} = disp.ui_mode {
			disp.state.set_auto_turn(match prev_auto_turn {
				AutoTurn::On => AutoTurn::Off,
				AutoTurn::Off | AutoTurn::FinishAllActions => AutoTurn::On,
				AutoTurn::N => {panicq!("invalid auto turn");}
			});
		}else{panicq!("invalid UI mode setting");}
		
		// clear so that end_window() does not reset auto turn increment
		disp.ui_mode = UIMode::None;
		
		update_indicators!();
	}else if m("P|references", "Workers create city sectors"){
		disp.state.iface_settings.workers_create_city_sectors ^= true; update_indicators!();
	}else if m("P|references", "|I|nterrupt auto turn for important events"){
		disp.state.iface_settings.interrupt_auto_turn ^= true; update_indicators!();
	}else if m("P|references", "Show all zon|e| information"){
		disp.state.iface_settings.show_all_zone_information ^= true; update_indicators!();

	}else if m("P|references", "|P|lace unit at cursor"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlaceUnitWindow(PlaceUnitWindowState {mode: 0});
		return None;
	
	}else if m("P|references", "S|w|itch to player"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::SwitchToPlayerWindow(SwitchToPlayerWindowState {mode: disp.state.iface_settings.cur_player as usize});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("P|references", "Disco|v|er specific technology"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::DiscoverTechWindow(DiscoverTechWindowState {mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
		
	}else if m("P|references", "Discover al|l| technology"){
		let pstats = &mut players[disp.state.iface_settings.cur_player as usize].stats;
		for tech_ind in 0..temps.techs.len() {
			pstats.force_discover_undiscov_tech(tech_ind as SmSvType, temps, &mut disp.state);
		}
	
	}else if m("P|references", "Obtain |r|esource"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::ObtainResourceWindow(ObtainResourceWindowState {mode: 0});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("P|references", "Free mone|y|"){
		players[disp.state.iface_settings.cur_player as usize].stats.gold += 300_000.;
		
	}else if m("P|references", "|C|hange game difficulty"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::SetDifficultyWindow(SetDifficultyWindowState {
			mode: game_difficulties.cur_difficulty_ind(players)
		});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("P|references", "Pause current AI's actio|n|s"){
		match players[disp.state.iface_settings.cur_player as usize].ptype {
			PlayerType::Empire(EmpireState {ref mut ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ref mut ai_state, ..}) => {
				ai_state.paused ^= true; update_indicators!();
			}
			PlayerType::Barbarian(_) | PlayerType::Human(_) => {}
		}
	
	}else if m("H|elp", "E|ncyclopedia"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::EncyclopediaWindow(EncyclopediaWindowState::CategorySelection {mode: 0});
		return None;
	
	}else if m("H|elp", "M|PD vs time"){	
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::PlotWindow(PlotWindowState {data: PlotData::MPD});
		return None; // return because end_menu() will overwrite disp.ui_mode
	
	}else if m("H|elp", "A|bout"){
		disp.reset_auto_turn();
		disp.ui_mode = UIMode::AboutWindow(AboutWindowState {});
		return None;
	}

	disp.end_menu();
	None
}

#[derive(PartialEq)]
enum SubDirection {Down, Up}

fn sel_next_submenu_entry(direction: SubDirection, sub_options: &OptionsUI, ui_mode: &mut UIMode){
	let n_sub_items = sub_options.options.len();
	if n_sub_items == 0 {return;}
	
	// only one grayed out submenu, nothing to select
	if n_sub_items == 1 && sub_options.options[0].dyn_str.chars().nth(0) == Some(MENU_INACTIVEC) {return;}
	
	loop{
		if let UIMode::Menu {ref mut sub_mode, ..} = ui_mode {
			*sub_mode = Some(
				if sub_mode.is_none() {
					if direction == SubDirection::Down {0} else {n_sub_items - 1}
				}else if direction == SubDirection::Down && *sub_mode == Some(n_sub_items-1) {
					0
				}else if direction == SubDirection::Up && *sub_mode == Some(0) {
					n_sub_items - 1
				}else{
					((match direction {
						SubDirection::Down => {1}
						SubDirection::Up => {-1}
					}) + sub_mode.unwrap() as isize) as usize
				});
			
		}else {panicq!("UI mode not set to menu");}
		
		// only break when we are no longer on a grayed entry
		if let UIMode::Menu{sub_mode: Some(sub_mode), .. } = ui_mode {
			if sub_options.options[*sub_mode].dyn_str.chars().nth(0) != Some(MENU_INACTIVEC)
				{return;}
		}else{panicq!("sub_mode not set");}
	}
}

#[derive(PartialEq)]
pub enum UIRet {Active, Inactive, ChgGameControl}

// return true if menu remains active, false if not active any more or was not initially
pub fn do_menu_shortcut<'f,'bt,'ut,'rt,'dt>(disp: &mut Disp<'f,'_,'bt,'ut,'rt,'dt>, map_data: &mut MapData<'rt>, 
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, gstate: &mut GameState, frame_stats: &FrameStats,
		game_control: &mut GameControl, game_opts: &mut GameOptions, game_difficulties: &GameDifficulties) -> UIRet {
	
	macro_rules! exec_sub{($mode: expr, $sub:expr) => {
		return if let Some(g) = execute_submenu($mode, $sub, disp, map_data, exs, players, units, bldgs, temps, gstate, frame_stats, game_opts, game_difficulties) {
			*game_control = g;
			UIRet::ChgGameControl
		}else{
			UIRet::Active
		};
	};};
	
	// mouse selection (ignore scrolling)
	if !scroll_down(&disp.state.mouse_event) && !scroll_up(&disp.state.mouse_event) {
	if let Some(mouse_event) = &disp.state.mouse_event {
		let mouse_event = (*mouse_event).clone();
		// main menu
		if mouse_event.y == 0 {
			let mut prev_item_lcol = 1 + MENU_1STR.len() as i32;
			
			// left most part of menu clicked ("Menu: ") -- close if menu already active, open if it isn't
			if mouse_event.x < prev_item_lcol {
				if lbutton_clicked(&disp.state.mouse_event) || lbutton_pressed(&disp.state.mouse_event) {
					if let UIMode::Menu {mode: None, ..} = disp.ui_mode {
						disp.end_menu();
					}else{
						disp.start_menu();
					}
				}
				return UIRet::Active;
			}
			
			let prev_mode = if let UIMode::Menu {mode, ..} = disp.ui_mode {mode} else {None};
			
			// specific menu item clicked
			for (menu_ind, menu_item) in disp.state.menu_options.options.iter().enumerate() {
				if mouse_event.x >= prev_item_lcol &&
				   mouse_event.x < (3 + prev_item_lcol + menu_item.strlen as i32) {
					// open menu if not already opened
					if lbutton_clicked(&disp.state.mouse_event) || lbutton_pressed(&disp.state.mouse_event) {
						if let UIMode::Menu {..} = &disp.ui_mode {} else {start_menu_sep(&mut disp.ui_mode, &mut disp.state.iface_settings, &mut disp.state.renderer);}
					}
					
					// set menu item or close menu if it's already been clicked
					if let UIMode::Menu {ref mut mode, ref mut sub_mode, ..} = disp.ui_mode {
						// close menu because it was previously selected
						if Some(menu_ind) == prev_mode {
							if lbutton_clicked(&disp.state.mouse_event) || lbutton_pressed(&disp.state.mouse_event) {
								disp.end_menu();
								return UIRet::Inactive;
							}
						}
						*mode = Some(menu_ind);
						*sub_mode = None;
					}else{continue;}
					
					if let ArgOptionUI::MainMenu {sub_options, ..} = &menu_item.arg {
						sel_next_submenu_entry(SubDirection::Down, &sub_options, &mut disp.ui_mode);
					} else {panicq!("main menu had no arguments set");}
					
					return UIRet::Active;
				}
				prev_item_lcol += 3 + menu_item.strlen as i32;
			}
			
			disp.end_menu();
			return UIRet::Inactive;
			
		// sub menu (the one which has already been selected)
		}else if let UIMode::Menu {mode: Some(mode), sub_mode: Some(ref mut sub_mode), ..} = &mut disp.ui_mode {
			if let ArgOptionUI::MainMenu {sub_options, col_start} = &disp.state.menu_options.options[*mode].arg {
				if mouse_event.x >= (*col_start) as i32 && 
				   mouse_event.x < (*col_start + sub_options.max_strlen) as i32 &&
				   mouse_event.y >= 2 &&
				   mouse_event.y < (2 + sub_options.options.len()) as i32 {
					let sub_mode_candidate = (mouse_event.y - 2) as usize;
					
					// grayed out menu item, don't select it
					if sub_options.options[sub_mode_candidate].dyn_str.chars().nth(0) == Some(MENU_INACTIVEC) {
						return UIRet::Active;
					}
					
					*sub_mode = sub_mode_candidate;
					if lbutton_clicked(&disp.state.mouse_event) || lbutton_released(&disp.state.mouse_event) {
						exec_sub!(*mode, *sub_mode);
					}
				}
			}else{panicq!("main menu arguments not set");};
		}
		
		// clicked outside of menu, end menu mode (only if we're already in menu mode)
		if let UIMode::Menu {..} = disp.ui_mode {
			if lbutton_clicked(&disp.state.mouse_event) {
				disp.end_menu();
				return UIRet::Inactive;
			}
		}
	}}
	
	// return if not in menu mode
	match disp.ui_mode {
		UIMode::Menu {..} => {},
		_ => {return UIRet::Inactive;}
	}
	
	if disp.state.key_pressed == KEY_ESC { 
		disp.end_menu();
		return UIRet::Active;
	}
	
	// inc/dec submenu selection
	if disp.state.key_pressed == KEY_DOWN || disp.state.key_pressed == KEY_UP {
		if let UIMode::Menu {mode: Some(mode), ..} = disp.ui_mode {
			if let ArgOptionUI::MainMenu {sub_options, ..} = &disp.state.menu_options.options[mode].arg {
				
				let direction = if disp.state.key_pressed == KEY_DOWN {SubDirection::Down} else {SubDirection::Up};
				
				sel_next_submenu_entry(direction, &sub_options, &mut disp.ui_mode);
				
			} else {panicq!("main menu arguments not set");}
		} else {return UIRet::Active;} // not on a menu item
	}
	
	// inc/dec menu selection
	let n_menus = disp.state.menu_options.options.len();
	if disp.state.key_pressed == KEY_RIGHT || disp.state.key_pressed == KEY_LEFT {
		if let UIMode::Menu {ref mut mode, ref mut sub_mode, ..} = disp.ui_mode {
			*mode = Some(
				if mode.is_none() {
					if disp.state.key_pressed == KEY_RIGHT {0} else {n_menus - 1}
				}else if disp.state.key_pressed == KEY_LEFT && *mode == Some(0) {
					n_menus - 1
				}else if disp.state.key_pressed == KEY_RIGHT && *mode == Some(n_menus-1) {
					0
				}else{
					mode.unwrap() + ((disp.state.key_pressed == KEY_RIGHT) as usize)*2 - 1
				});
			
			*sub_mode = None;
			
			let option = &disp.state.menu_options.options[mode.unwrap()];
			if let ArgOptionUI::MainMenu {sub_options, ..} = &option.arg {
				sel_next_submenu_entry(SubDirection::Down, &sub_options, &mut disp.ui_mode);
			
			}else {panicq!("main menu arguments not set");}
		}else {panicq!("menu ui not set");}
		
		return UIRet::Active;
	}
	
	// return to center on cursor
	if disp.state.key_pressed == ' ' as i32 {
		disp.end_menu();
		return UIRet::Inactive;
	}
	
	// sub menu active -- execute command if enter or shortcut key pressed
	if let UIMode::Menu {mode: Some(mode), sub_mode: Some(sub_mode), ..} = disp.ui_mode {
		macro_rules! exec_sub{($mode: expr, $sub:expr) => {
			return if let Some(g) = execute_submenu($mode, $sub, disp, map_data, exs, players, units, bldgs, temps, gstate, frame_stats, game_opts, game_difficulties) {
				*game_control = g;
				UIRet::ChgGameControl
			}else{
				UIRet::Active
			};
		};};
		
		// pressed enter on selected item
		if disp.state.key_pressed == KEY_ENTER || disp.state.key_pressed == ('\n' as i32) {
			exec_sub!(mode, sub_mode);
		}
		
		// needed to get shortcut keys so the for loop below doesn't have borrow checking issues
		let sub_options = if let ArgOptionUI::MainMenu {sub_options, ..} = &disp.state.menu_options.options[mode].arg {
			sub_options.clone()
		}else{panicq!("main menu arguments not set")};
		
		// shortcut key pressed?
		for (sub_mode, sub_option) in sub_options.options.iter().enumerate() {
			if sub_option.key == Some(disp.state.key_pressed as u8 as char) {
				exec_sub!(mode, sub_mode);
			}
		}
	}else if let UIMode::Menu {mode: Some(_), sub_mode: None, ..} = disp.ui_mode {
		panicq!("menu mode set, but not sub_mode");
	}
	
	//////////// main menu active: select submenu based on shortcut key
	for (mode_val, option) in disp.state.menu_options.options.iter().enumerate() {
		if option.key == Some(disp.state.key_pressed as u8 as char) {
			if let UIMode::Menu {ref mut mode, ref mut sub_mode, ..} = &mut disp.ui_mode {
				*mode = Some(mode_val);
				*sub_mode = None;
			}else{panicq!("invalid setting for UI mode");}
			
			if let ArgOptionUI::MainMenu {sub_options, ..} = &option.arg {
				sel_next_submenu_entry(SubDirection::Down, &sub_options, &mut disp.ui_mode);
			} else {panicq!("main menu had no arguments set");}
			return UIRet::Active;
		}
	}
	
	UIRet::Active
}

////////////////////////////////////////////////////////////
// init
//////////////////////////////////////////////////////////////

// for each str in nms, append `options`, recording the shortcut key, indicated by delimitators in it
impl OptionsUI<'_,'_,'_,'_> {
	pub fn new(nms: &[&str]) -> Self {
		let mut opts = OptionsUI {
			options: Vec::with_capacity(nms.len()),
		   	max_strlen: 0
		};
		
		opts.register_shortcuts(nms);
		opts
	}
	
	fn register_shortcuts(&mut self, nms: &[&str]){
		for nm in nms.iter() {
			// count tokens
			let mut n_tokens = 1;
			for i in 0..nm.chars().count() {
				if nm.chars().nth(i).unwrap() == MENU_DELIMC {n_tokens += 1};	
			}
			
			if nm.chars().nth(0).unwrap() == MENU_INACTIVEC {n_tokens += 1}
			let nm_strlen = nm.chars().count();
			
			if nm_strlen > self.max_strlen { self.max_strlen = nm_strlen; }
			
			// print w/ key underlined
			let pats: Vec<&str> = nm.split(MENU_DELIMC).collect();
			let mut key = None;
			for (token_ind, pat) in pats.iter().enumerate() {
				if pat.chars().count() == 1 && (token_ind != 0 || n_tokens == 2) {
					key = Some(pat.chars().nth(0).unwrap().to_ascii_lowercase());
					break;
				}
			}
			
			self.options.push(OptionUI { key, 
						dyn_str: nm.to_string(),
						strlen: nm_strlen - n_tokens + 1,
						arg: ArgOptionUI::Blank });
		}
	}
}

pub fn init_menus<'f,'bt,'ut,'rt,'dt>(dstate: &mut DispState<'f,'_,'bt,'ut,'rt,'dt>, players: &Vec<Player<'bt,'ut,'rt,'dt>>) {
	let mut main_options = OptionsUI::new(MENU_NMS);
	
	// for each main menu (ex. "File", insert sub-menus ex. "Open", "Load")
	for i in 0..(MENU_NMS.len()) {
		
		// character start of each menu item on line
		let col_start = if i > 0 {
					let option_prev = &main_options.options[i-1];
					if let ArgOptionUI::MainMenu {col_start, ..} = option_prev.arg {
						col_start + option_prev.strlen + 3
					}else {panicq!("menu arguments not set");}
				}else{  MENU_STR.chars().count() - 1 };
		
		let sub_options = OptionsUI::new(SUB_MENU_NMS_INIT[i]);
		
		// set arguments of main menu
		main_options.options[i].arg = ArgOptionUI::MainMenu {col_start, sub_options};
	}
	dstate.menu_options = main_options;
	dstate.update_menu_indicators(dstate.iface_settings.cur_player_paused(players));
}

///////// update indicators, ex. overlays, "* Arability"
// if current player is not an AI, `cur_ai_player_is_paused` should be none
impl <'f,'bt,'ut,'rt,'dt>DispState<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn update_menu_indicators(&mut self, cur_ui_ai_player_is_paused: Option<bool>){
		let get_menu_ind = |nm, main_options: &OptionsUI| {
			for (i, option) in main_options.options.iter().enumerate() {
				if option.dyn_str.contains(nm) {
					return i;
				}
			}
			panicq!("menu {} not found in dyn_strs", nm);
		};
		
		macro_rules! set_indicator{($sub_match: expr, $cond: expr, $sub_option: expr) => {
			if $sub_option.dyn_str.contains($sub_match) {
				let mut r = $sub_option.dyn_str.clone().into_bytes();
				r[1] = if $cond {'*'} else {' '} as u8;
				$sub_option.dyn_str = String::from_utf8(r).unwrap();
			}}}
			
		let menu_ind = get_menu_ind("V|iew", &self.menu_options);
		if let ArgOptionUI::MainMenu { ref mut sub_options, .. } = &mut self.menu_options.options[menu_ind].arg {
			let iface_settings = &mut self.iface_settings;
			for sub_opt in sub_options.options.iter_mut() {
				set_indicator!("|A|rability", iface_settings.underlay == Underlay::Arability, sub_opt);
				set_indicator!("|E|levation", iface_settings.underlay == Underlay::Elevation, sub_opt);
				set_indicator!("Water & mountains |o|nly", iface_settings.underlay == Underlay::WaterMountains, sub_opt);
				
				set_indicator!("|S|tructures", iface_settings.show_structures, sub_opt);
				set_indicator!("|U|nits", iface_settings.show_units, sub_opt);
				set_indicator!("|B|uildings", iface_settings.show_bldgs, sub_opt);
				set_indicator!("|Z|ones", iface_settings.show_zones, sub_opt);
				set_indicator!("|R|esources", iface_settings.show_resources, sub_opt);
				set_indicator!("Zone |d|emands", iface_settings.zone_overlay_map == ZoneOverlayMap::ZoneDemands, sub_opt);
				set_indicator!("Happi|n|ess", iface_settings.zone_overlay_map == ZoneOverlayMap::Happiness, sub_opt);
				set_indicator!("Crime (|k|)", iface_settings.zone_overlay_map == ZoneOverlayMap::Crime, sub_opt);
				set_indicator!("Show un|c|onnected bldgs", iface_settings.show_unconnected_bldgs, sub_opt);
				set_indicator!("Show unoccup|i|ed bldgs", iface_settings.show_unoccupied_bldgs, sub_opt);
				set_indicator!("Show sectors (|2|)", iface_settings.show_sectors, sub_opt);
			}
		}else{panicq!("main menu arguments not set");}
		
		let menu_ind = get_menu_ind("P|references", &self.menu_options);
		if let ArgOptionUI::MainMenu { sub_options, .. } = &mut self.menu_options.options[menu_ind].arg {
			let iface_settings = &mut self.iface_settings;
			let terminal_settings = &mut self.terminal;
			for sub_opt in sub_options.options.iter_mut() {
				set_indicator!("|A|void special chars", terminal_settings.limit_schars, sub_opt);
				set_indicator!("Workers create city sectors", iface_settings.workers_create_city_sectors, sub_opt);
				set_indicator!("|U|se only 8 colors", terminal_settings.limit_colors, sub_opt);
				set_indicator!("Show full |m|ap", !iface_settings.show_fog, sub_opt);
				set_indicator!("Show all unit ac|t|ions", iface_settings.show_actions, sub_opt);
				set_indicator!("|A|uto turn increment", iface_settings.auto_turn == AutoTurn::On, sub_opt);
				set_indicator!("|I|nterrupt auto turn for important events", iface_settings.interrupt_auto_turn, sub_opt);
				set_indicator!("Show all zon|e| information", iface_settings.show_all_zone_information, sub_opt);
				
				if let Some(paused) = cur_ui_ai_player_is_paused {
					set_indicator!("Pause current AI's actio|n|s", paused, sub_opt);
				}
			}
		}else{panicq!("main menu arguments not set");}
	}
}

/////////////////////////////////////////////////////
// print menus
/////////////////////////////////////////////////////

// prints nm with one space before & after, will highlight if entry_active = True
// used for both top-menu, sub-menu, and building production displays
// returns the row and col the text starts at
fn print_menu_item(nm: &String, entry_active: bool, menu_active: bool, owner_opt: Option<&Player>, d: &mut Renderer) -> (i32, i32) {
	if nm.chars().nth(0).unwrap() == MENU_INACTIVEC {
		d.attron(COLOR_PAIR(CGRAY)); //A_DIM());
		d.addstr(format!(" {} ", &nm[1..]).as_ref());
		d.attroff(COLOR_PAIR(CGRAY)); //A_DIM());
		return (0,0);
	}
	
	// count tokens
	let mut n_tokens = 1;
	let strlen = nm.chars().count();
	for i in 0..strlen {
		if nm.chars().nth(i).unwrap() == MENU_DELIMC {n_tokens += 1};
	}
	
	if entry_active {
		d.attron(A_REVERSE());
	}else if let Some(owner) = owner_opt {
		set_player_color(owner, true, d);
	}
	
	d.addch(' ' as chtype);
	
	let color = if menu_active {shortcut_indicator()} else {COLOR_PAIR(CWHITE)};
	
	// for screen readers
	let mut first_non_whitespace_char_found = false; // attempt to skip over whitespace
	let mut sel_loc = {
		let curs = cursor_pos(d);
		(curs.y as i32, curs.x as i32)
	};
	
	// print w/ key underlined
	let pats: Vec<&str> = nm.split(MENU_DELIMC).collect(); 
	for (token_ind, pat) in pats.iter().enumerate() {
		// set cursor start loc for screen readers -- find first non whitespace
		if !first_non_whitespace_char_found {
			for (c_ind, _) in pat.chars().enumerate().filter(|(_, c)| *c != ' ') {
				let curs = cursor_pos(d);
				sel_loc = (curs.y as i32, curs.x as i32 + c_ind as i32);
				first_non_whitespace_char_found = true;
				break;
			}
		}
		if pat.chars().count() == 1 && (token_ind != 0 || n_tokens == 2) &&
			(token_ind != (n_tokens-1) || n_tokens == 2) &&
			(token_ind == 0 || nm.chars().count() != 3) { // < ex in the case of "G|o" choose the first
				if screen_reader_mode() {d.addch('[');}
				d.addch(pat.chars().nth(0).unwrap() as chtype | color);
				if screen_reader_mode() {d.addch(']');}
		}else{d.addstr(format!("{}", pat).as_ref()); }
	}
	
	// print padding
	d.addch(' ' as chtype);
	
	if entry_active {
		d.attroff(A_REVERSE());
	}else if let Some(owner) = owner_opt {
		set_player_color(owner, false, d);
	}
	sel_loc
}

// prints vertical list of menu items at row, col, with width w as:
// | Option1   |
// | Option2   |
// | Option345 |
// -------------
// setting entry_active = true, results in the option being highlighted
// `start_ind` indicates the first index included in the `sub_options` (ex. if
// only part of it is supplied here if more is in the list than is to be shown on the
// screen). `start_ind` is used to save values in `buttons`
// if owners_opt supplied, print owner colors
// sel_loc is set to the location of the selected text
pub fn print_menu_vstack(sub_options: &OptionsUI, row: i32, col: i32, w: usize, 
		ind_active: usize, show_ai_pause: bool, owners_opt: Option<&Vec<Player>>, start_ind: usize,
		sel_loc: &mut Option<&mut (i32, i32)>, chars: &DispChars, buttons: &mut Buttons,
		renderer: &mut Renderer){
	
	macro_rules! sub_ln_s{($row_add: expr) => {renderer.mv($row_add as i32 + row, col);};};
	let mut rows_added = 0;
	
	///////// sub menu entries
	for (i, sub_opt) in sub_options.options.iter().enumerate() {
		// if not on an AI player, don't show the `pause AI actions` menu item
		if !show_ai_pause && sub_opt.dyn_str == "   Pause current AI's actio|n|s" {
			continue;
		}
		
		sub_ln_s!(i);
		renderer.addch(chars.vline_char);
		
		let button_start = cursor_pos(renderer);
		let entry_active = i == ind_active;
		
		// print menu item
		let cur_sel_loc = (|| {
			if let Some(owners) = owners_opt {
				if let ArgOptionUI::OwnerInd(owner_ind) = sub_opt.arg {
					return print_menu_item(&sub_opt.dyn_str, entry_active, true, Some(&owners[owner_ind]), renderer);
				}
			}
			print_menu_item(&sub_opt.dyn_str, entry_active, true, None, renderer)
		})();
		
		// for screen readers
		if entry_active {
			if let Some(ref mut sel_loc) = sel_loc {
				**sel_loc = cur_sel_loc;
			}
		}
		
		// have menu end at same spot for all entries
		if entry_active { renderer.attron(A_REVERSE()); }
		if w > (sub_opt.strlen + 4) {
			let gap = w - sub_opt.strlen - 4;
			for _ in 0..gap { renderer.addch(' ' as chtype); }
		}
		if entry_active { renderer.attroff(A_REVERSE()); }
		
		buttons.add(button_start, i + start_ind, renderer);
		
		renderer.addch(chars.vline_char);
		rows_added += 1;
	}
	
	// last line
	sub_ln_s!(rows_added);
	renderer.addch(chars.llcorner_char);
	for _ in 0..(w-2) { renderer.addch(chars.hline_char); }
	renderer.addch(chars.lrcorner_char);
}

// prints top menu and expanded submenus
// inputs: iface_settings: menu_active, menu_mode, sub_menu_mode
impl Disp<'_,'_,'_,'_,'_,'_> {
	pub fn print_menus(&mut self, show_ai_pause: bool){
		let menu_active = match self.ui_mode {
			UIMode::Menu {ref mut sel_loc, ..} => {
				*sel_loc = (0,0); // cursor location for screen readers, this will be set below if a particular item is active
				true
			} _ => false
		};
		
		/////////////// top menu
		self.state.renderer.mv(0,0);
		if menu_active { self.state.renderer.attron(A_REVERSE()); }
		if self.state.kbd.open_top_menu as chtype == 'm' as chtype || self.state.kbd.open_top_menu as chtype == 'M' as chtype {
			self.state.renderer.addch(self.state.kbd.open_top_menu as chtype | shortcut_indicator());
			self.state.renderer.addstr(MENU_1STR);
		}else{
			self.state.renderer.addch(self.state.kbd.open_top_menu as chtype | shortcut_indicator());
			self.state.renderer.addch(':');
			for _ in 0..MENU_1STR.len()-1 {
				self.state.renderer.addch(' ');
			}
		}
		if menu_active { self.state.renderer.attroff(A_REVERSE()); }
		
		debug_assertq!(MENU_NMS.len() == self.state.menu_options.options.len());
		
		// print top-level menu items
		for (i, option) in self.state.menu_options.options.iter().enumerate() {
			let entry_active = if let UIMode::Menu {mode: Some(mode), sub_mode, ref mut sel_loc, ..} = self.ui_mode {
				if mode == i {
					// set cursor location for screen readers
					if sub_mode.is_none() {
						let cur = cursor_pos(&self.state.renderer);
						*sel_loc = (cur.y as i32, cur.x as i32);
					}
					true
				}else{false}
			}else {false};
			
			print_menu_item(&option.dyn_str, entry_active, menu_active, None, &mut self.state.renderer);
			
			if i != (MENU_NMS.len()-1) {self.state.renderer.addch(self.state.chars.vline_char);}
		}
		
		if menu_active {
			self.state.renderer.attron(COLOR_PAIR(ESC_COLOR));
			self.state.renderer.addstr("   ");
			self.state.renderer.addstr(&self.state.local.Esc_to_exit_menu);
			self.state.renderer.attroff(COLOR_PAIR(ESC_COLOR));
		}else{ 
			self.state.renderer.addch('\n' as chtype); }
		
		const SUB_MENU_LN_OFFSET: usize = 2;
		
		////// show submenu
		if let UIMode::Menu {mode: Some(mode), sub_mode: Some(sub_mode), ref mut sel_loc, ..} = self.ui_mode {
			let main_opt = &self.state.menu_options.options[mode];
			if let ArgOptionUI::MainMenu {col_start, sub_options} = &main_opt.arg {
				macro_rules! sub_ln_s{($row: expr) => {self.state.renderer.mv($row as i32, *col_start as i32);};};
				
				///// first line
				sub_ln_s!(1);
				self.state.renderer.addch(self.state.chars.vline_char);
				
				// spaces
				for _ in 0..(main_opt.strlen + 2) {
					self.state.renderer.addch(' ' as chtype); }
				
				self.state.renderer.addch(self.state.chars.llcorner_char);
				
				// top horizontal lines
				if sub_options.max_strlen > (main_opt.strlen + 3) {
					let gap = sub_options.max_strlen - main_opt.strlen - 2;
					for _ in 0..gap { self.state.renderer.addch(self.state.chars.hline_char); }
				}
				self.state.renderer.addch(self.state.chars.urcorner_char);
				
				// width to show submenu
				let w_use = if sub_options.max_strlen > main_opt.strlen {
					sub_options.max_strlen + 1
				}else{
					main_opt.strlen + 3
				};
				
				// print stack of submenu options
				print_menu_vstack(sub_options, SUB_MENU_LN_OFFSET as i32, *col_start as i32, 
						w_use + 2, sub_mode, show_ai_pause, None, 0, &mut Some(sel_loc),
						&self.state.chars, &mut self.state.buttons, &mut self.state.renderer);
			
			}else{panicq!("main menu arguments not set");}
		}else if let UIMode::Menu {mode: Some(_), sub_mode: None, ..} = self.ui_mode { 
			panicq!("menu mode set but not the sub_mode");
		}
	}
}

