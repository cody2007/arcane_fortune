use std::cmp::{min, max};
use super::vars::*;
use super::menus::*;
use super::color::*;

pub const TITLE_COLOR: i32 = CGREEN;

#[macro_use]
pub mod text_entry; pub use text_entry::*;
pub mod lists; pub use lists::*;
pub mod keys; pub use keys::*;
pub mod encyclopedia_info; pub use encyclopedia_info::*;
pub mod states; pub use states::*;

use std::cmp::Ordering;
use crate::renderer::*;
use crate::disp::*;
use crate::doctrine::*;
use crate::tech::{TechTemplate};
use crate::saving::{SmSvType};
use crate::gcore::*;
use crate::resources::ResourceTemplate;
//use crate::nn::{TxtPrinter, TxtCategory};
use crate::gcore::{GameDifficulties, LogType};
use crate::player::{Player, Stats, PlayerType, PersonName};
use crate::containers::Templates;
use crate::localization::Localization;

// row bounds to show log
const LOG_START_ROW: i32 = 2;
const LOG_STOP_ROW: i32 = 2;

const MAX_SAVE_AS_W: usize = 78;

const ENCYCLOPEDIA_CATEGORY_NMS: &[&str] = &["Military |u|nits", "B|uildings", "T|echnology", "D|octrines", "|R|esources"];

pub struct ProdOptions<'bt,'ut,'rt,'dt> {
	pub bldgs: Box<[Option<OptionsUI<'bt,'ut,'rt,'dt>>]>,
	pub worker: OptionsUI<'bt,'ut,'rt,'dt>
}

pub fn init_bldg_prod_windows<'bt,'ut,'rt,'dt>(bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, pstats: &Stats, l: &Localization) -> ProdOptions<'bt,'ut,'rt,'dt> {
	////////////////////////////////////// buildings workers produce
	let worker_options = {
		let mut txt_w_none = vec!{"N|one"};
		let mut bldg_template_inds = vec!{None};
		
		for bt in bldg_templates.iter() {
			if bt.barbarian_only {continue;}
			
			if let BldgType::Gov(_) = bt.bldg_type {
				if bt.available(pstats) {
					if let Some(menu_txt) = &bt.menu_txt {
						txt_w_none.push(menu_txt);
					}else{
						txt_w_none.push(&bt.nm[l.lang_ind]);
					}
					bldg_template_inds.push(Some(bt));
				}
			}
		}
		let mut worker_options = OptionsUI {options: Vec::with_capacity(txt_w_none.len()), max_strlen: 0};
		register_shortcuts(&txt_w_none, &mut worker_options);
		
		// set argument options to associate unit_template index with menu entry
		for (opt, bldg_template_ind) in worker_options.options.iter_mut().zip(bldg_template_inds.iter().cloned()) {
			opt.arg = ArgOptionUI::BldgTemplate(bldg_template_ind);
		}
		
		worker_options
	};
	
	///////////
	let mut production_options = ProdOptions {
			bldgs: vec!{None; bldg_templates.len()}.into_boxed_slice(),
			worker: worker_options
	};
	
	/////////////////////////////////// building unit productions
	for bt in bldg_templates.iter() {
		if let Some(units_producable_txt) = &bt.units_producable_txt {
			let units_producable = bt.units_producable.as_ref().unwrap();
			
			let mut txt_w_none = vec!{"N|one"};
			let mut unit_template_inds = vec!{None};
			for (txt, unit_producable) in units_producable_txt.iter().zip(units_producable.iter()) {
				if pstats.unit_producable(unit_producable) {
					txt_w_none.push(txt);
					unit_template_inds.push(Some(*unit_producable));
					//printlnq!("{} {}", bt.nm, unit_producable.nm);
				}
			}
			
			production_options.bldgs[bt.id as usize] = Some(OptionsUI {
								options: Vec::with_capacity(txt_w_none.len()),
								max_strlen: 0});
			
			register_shortcuts(&txt_w_none, production_options.bldgs[bt.id as usize].as_mut().unwrap());
			
			// set argument options to associate unit_template index with menu entry
			for (opt, unit_template_ind) in production_options.bldgs[bt.id as usize].as_mut().unwrap().options.iter_mut().
									zip(unit_template_inds.iter().cloned()) {
				opt.arg = ArgOptionUI::UnitTemplate(unit_template_ind);
			}
			
			assertq!(units_producable_txt.len() == bt.units_producable.as_ref().unwrap().len(), 
					"Building \"{}\"'s list of units_producable and units_producable_txt are not the same size ({}, and {}). The configuration file needs to be altered.",
					bt.nm[0], units_producable_txt.len(), bt.units_producable.as_ref().unwrap().len());
		}
	}
	
	production_options
}

pub fn center_txt(txt: &str, w: i32, color: Option<chtype>, renderer: &mut Renderer) {
	let g = (w as usize - txt.len())/2;
	let mut sp = String::new();
	for _ in 0..g {sp.push(' ');}
	
	renderer.addstr(&sp);
	
	if let Some(c) = color {renderer.attron(c);}
	renderer.addstr(&txt);
	if let Some(c) = color {renderer.attroff(c);}
}

// returns the upper left corner where the window was printed
impl <'f,'bt,'ut,'rt,'dt>DispState<'f,'bt,'ut,'rt,'dt> {
	fn print_window(&mut self, window_sz: ScreenSz) -> Coord {
		let screen_sz = self.iface_settings.screen_sz;
		let r = &mut self.renderer;
		debug_assertq!(screen_sz.h >= window_sz.h);
		debug_assertq!(screen_sz.w >= window_sz.w);
		
		let row_initial = ((screen_sz.h - window_sz.h)/2) as i32;
		let col = ((screen_sz.w - window_sz.w)/2) as i32;
		
		let h = window_sz.h as i32;
		let w = window_sz.w as i32;
		
		// print top line
		{
			r.mv(row_initial, col);
			r.addch(self.chars.ulcorner_char);
			for _ in 2..w {r.addch(self.chars.hline_char);}
			r.addch(self.chars.urcorner_char);
		}
		
		// print intermediate lines
		for row_off in 1..(h-1) {
			r.mv(row_initial + row_off, col);
			r.addch(self.chars.vline_char);
			for _ in 2..w {r.addch(' ');}
			r.addch(self.chars.vline_char);
		}
		
		// print bottom line
		{
			r.mv(row_initial + h-1, col);
			r.addch(self.chars.llcorner_char);
			for _ in 2..w {r.addch(self.chars.hline_char);}
			r.addch(self.chars.lrcorner_char);
		}
		
		Coord {y: row_initial as isize, x: col as isize}
	}

	pub fn print_window_at(&mut self, window_sz: ScreenSz, loc: ScreenCoord) {
		let row_initial = loc.y as i32;
		let col = loc.x as i32;
		let d = &mut self.renderer;
		
		// print top line
		{
			d.mv(row_initial, col);
			d.addch(self.chars.ulcorner_char);
			for _ in 2..window_sz.w {d.addch(self.chars.hline_char);}
			d.addch(self.chars.urcorner_char);
		}
		
		// print intermediate lines
		for row_off in 1..(window_sz.h as i32-1) {
			d.mv(row_initial + row_off, col);
			d.addch(self.chars.vline_char);
			for _ in 2..window_sz.w {d.addch(' ');}
			d.addch(self.chars.vline_char);
		}
		
		// print bottom line
		{
			d.mv(row_initial + window_sz.h as i32-1, col);
			d.addch(self.chars.llcorner_char);
			for _ in 2..window_sz.w {d.addch(self.chars.hline_char);}
			d.addch(self.chars.lrcorner_char);
		}
	}
}

// prints windows
impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'bt,'ut,'rt,'dt> {
	pub fn print_windows(&mut self, map_data: &mut MapData<'rt>, exf: &HashedMapEx<'bt,'ut,'rt,'dt>, units: &Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, players: &Vec<Player>, gstate: &GameState, game_difficulties: &GameDifficulties) {
		let h = self.state.iface_settings.screen_sz.h as i32;
		let w = self.state.iface_settings.screen_sz.w as i32;
		let cur_player = self.state.iface_settings.cur_player as usize;
		let dstate = &mut self.state;
		let pstats = &players[cur_player].stats;
		
		let ret = match &mut self.ui_mode {
			UIMode::None | UIMode::SetTaxes(_) | UIMode::TextTab {..} | UIMode::Menu {..} => {UIModeControl::UnChgd}
			UIMode::BldgsWindow(state) => {state.print(pstats, bldgs, temps, map_data, dstate)}
			UIMode::MvWithCursorNoActionsRemainAlert(state) => {state.print(units, dstate)}
			UIMode::CivilizationIntelWindow(state) => {state.print(players, gstate, dstate)}
			UIMode::ContactNobilityWindow(state) => {state.print(dstate)}
			UIMode::ResourcesAvailableWindow(state) => {state.print(pstats, temps, dstate)}
			UIMode::ResourcesDiscoveredWindow(state) => {state.print(pstats, map_data, temps, dstate)}
			UIMode::RiotingAlert(state) => {state.print(dstate)}
			UIMode::GenericAlert(state) => {state.print(dstate)}
			UIMode::CitizenDemandAlert(state) => {state.print(players, temps, dstate)}
			UIMode::CivicAdvisorsWindow(state) => {state.print(&players[cur_player], dstate)}
			UIMode::PublicPollingWindow(state) => {state.print(pstats, dstate)}
			UIMode::UnmovedUnitsNotification(state) => {state.print(dstate)}
			UIMode::ForeignUnitInSectorAlert(state) => {state.print(dstate)}
			UIMode::SelectBldgDoctrine(state) => {state.print(pstats, temps, dstate)}
			UIMode::CitiesWindow(state) => {state.print(map_data, bldgs, gstate, dstate)}
			UIMode::ManorsWindow(state) => {state.print(players, bldgs, map_data, gstate, dstate)}
			UIMode::PlaceUnitWindow(state) => {state.print(pstats, temps, dstate)}
			UIMode::ObtainResourceWindow(state) => {state.print(temps, dstate)}
			UIMode::DiscoverTechWindow(state) => {state.print(pstats, temps, dstate)}
			UIMode::ProdListWindow(state) => {state.print(units, bldgs, exf, pstats, temps, map_data, dstate)}
			UIMode::SetDifficultyWindow(state) => {state.print(game_difficulties, dstate)}
			UIMode::CurrentBldgProd(state) => {state.print(bldgs, players, map_data, exf, temps, dstate)}
			UIMode::AboutWindow(state) => {state.print(dstate)}
			UIMode::NoblePedigree(state) => {state.print(gstate, players, dstate)}
			UIMode::NobleUnitsWindow(state) => {state.print(players, units, map_data, temps, gstate, dstate)}
			UIMode::InitialGameWindow(state) => {state.print(players, dstate)}
			UIMode::EndGameWindow(state) => {state.print(players, dstate)}
			UIMode::SaveAutoFreqWindow(state) => {state.print(dstate)}
			UIMode::GoToCoordinateWindow(state) => {state.print(dstate)}
			UIMode::SaveAsWindow(state) => {state.print(dstate)}
			UIMode::TechWindow(state) => {state.print(temps, pstats, dstate)}
			UIMode::DoctrineWindow(state) => {state.print(temps, pstats, dstate)}
			UIMode::AcceptNobilityIntoEmpire(state) => {state.print(players, dstate)}
			UIMode::TechDiscoveredWindow(state) => {state.print(temps, dstate)}
			UIMode::PlotWindow(state) => {state.print(players, gstate, map_data, temps, dstate)}
			UIMode::GetTextWindow(state) => {state.print(dstate)}
			UIMode::OpenWindow(state) => {state.print(dstate)}
			UIMode::HistoryWindow(state) => {state.print(players, gstate, temps, dstate)}
			UIMode::EncyclopediaWindow(state) => {state.print(players, temps, dstate)}
			UIMode::SwitchToPlayerWindow(state) => {state.print(players, dstate)}
			UIMode::PrevailingDoctrineChangedWindow(state) => {state.print(players, dstate)}
			UIMode::BrigadesWindow(state) => {state.print(players, units, map_data, temps, dstate)}
			UIMode::BrigadeBuildList(state) => {state.print(pstats, dstate)}
			UIMode::SectorsWindow(state) => {state.print(pstats, map_data, dstate)}
			UIMode::CreateSectorAutomation(state) => {state.print(pstats, map_data, dstate)}
			UIMode::ContactEmbassyWindow(state) => {state.print(players, gstate, dstate)}
			UIMode::WarStatusWindow(state) => {state.print(&gstate.relations, players, dstate)}
			UIMode::UnitsWindow(state) => {state.print(players, units, map_data, temps, dstate)}
			UIMode::SelectExploreType(state) => {state.print(dstate)}
			UIMode::Trade(state) => {state.print(dstate)}
		};
	
		match ret {
			UIModeControl::Closed => {self.end_window()}
			UIModeControl::New(new) => {self.ui_mode = new}
			UIModeControl::UnChgd => {}
			UIModeControl::ChgGameControl | UIModeControl::CloseAndGoTo(_) => {panicq!("shouldn't occur here");}
		}
	}
}

