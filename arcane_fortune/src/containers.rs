use crate::buildings::{BldgConfig, BldgTemplate};
use crate::units::UnitTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::tech::TechTemplate;
use crate::player::{Nms};
use crate::ai::AIConfig;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;
use crate::disp::*;
use crate::disp::menus::*;
use crate::renderer::*;
use crate::gcore::*;
use crate::saving::*;

// disp_settings and disp_chars, while often passed to fns as immutable, are needed mutably in the menus for changing settings
// save_game() needs mutable refs to `owners` among others not normally passed as mutable. so fns that are called in
//	save_game() cant't reasonably take references to Templates -- for example, unfourtnately, add_bldg()
//	Templates could be created on demand in save_game() but it would be messy and error-prone, potentially
pub struct Templates<'bt,'ut,'rt,'dt,'tt> {
	pub bldgs: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
	pub units: &'ut Vec<UnitTemplate<'rt>>,
	pub doctrines: &'dt Vec<DoctrineTemplate>,
	pub resources: &'rt Vec<ResourceTemplate>,
	pub techs: &'tt Vec<TechTemplate>,
	pub ai_config: AIConfig<'rt>,
	pub bldg_config: BldgConfig,
	pub nms: Nms,
	
	// the following are loaded from config files and not saved in the save game file:
	//pub kbd: KeyboardMap,
	//pub l: Localization
}

pub struct Disp<'f,'r,'bt,'ut,'rt,'dt> {
	pub ui_mode: UIMode<'bt,'ut,'rt,'dt>, // active windows, menus, etc
	pub state: DispState<'f,'r,'bt,'ut,'rt,'dt>,
}

pub struct DispState<'f,'r,'bt,'ut,'rt,'dt> {
	pub iface_settings: IfaceSettings<'f,'bt,'ut,'rt,'dt>,
	pub terminal: TerminalSettings, // limit characters or colors
	pub chars: DispChars,
	pub menu_options: OptionsUI<'bt,'ut,'rt,'dt>, // top level menu
	pub production_options: ProdOptions<'bt,'ut,'rt,'dt>, // for bldgs, workers -- could be eventually removed and recomputed each frame when relevant
	pub txt_list: TxtList, // for screen readers
	pub buttons: Buttons,
	pub local: Localization,
	pub kbd: KeyboardMap,
	
	pub key_pressed: i32,
	pub mouse_event: Option<MEVENT>,
	
	pub renderer: &'r mut Renderer // sdl state variables
}

/*pub struct Map<'bt,'ut,'rt,'dt> {
	map_data: MapData<'rt>,
	exs: Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
	sz: MapSz
}*/

#[derive(Clone)]
pub struct GameState {
	pub relations: Relations,
	pub logs: Vec<Log>,
	pub rng: XorState,
	pub turn: usize
}

impl_saving!{GameState {relations, logs, rng, turn}}

/*
pub struct Objs {
	units: Vec<Unit<'bt,'ut,'rt,'dt>>,
	bldgs: Vec<Bldg<'bt,'ut,'rt,'dt>>
}
*/

/*
pub struct Game {
	objs: Objs,
	players: Vec<Player>,
	gstate: GameState,
	map: Map,
}
*/

///////////////
// pass-through functions
//	if a function already has Disp -- these can be used to shorten (code) line lengths
use std::convert::TryInto;
impl Disp<'_,'_,'_,'_,'_,'_> {
	pub fn addch<T: TryInto<chtype>>(&mut self, a: T) {self.state.renderer.addch(a);}
	pub fn attron<T: TryInto<chtype>>(&mut self, a: T) {self.state.renderer.attron(a);}
	pub fn attroff<T: TryInto<chtype>>(&mut self, a: T) {self.state.renderer.attroff(a);}
	pub fn mv<Y: TryInto<CInt>, X: TryInto<CInt>>(&mut self, y: Y, x: X) {self.state.renderer.mv(y,x);}
	pub fn addstr(&mut self, txt: &str) {self.state.renderer.addstr(txt);}
	pub fn inch(&self) -> chtype {self.state.renderer.inch()}
	//pub fn clear(&mut self) {self.state.renderer.clear()}
}

impl DispState<'_,'_,'_,'_,'_,'_> {
	pub fn addch<T: TryInto<chtype>>(&mut self, a: T) {self.renderer.addch(a);}
	pub fn attron<T: TryInto<chtype>>(&mut self, a: T) {self.renderer.attron(a);}
	pub fn attroff<T: TryInto<chtype>>(&mut self, a: T) {self.renderer.attroff(a);}
	pub fn mv<Y: TryInto<CInt>, X: TryInto<CInt>>(&mut self, y: Y, x: X) {self.renderer.mv(y,x);}
	pub fn addstr(&mut self, txt: &str) {self.renderer.addstr(txt);}
	pub fn inch(&self) -> chtype {self.renderer.inch()}
	pub fn clear(&mut self) {self.renderer.clear()}
}

