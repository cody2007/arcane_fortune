use crate::saving::*;
use crate::buildings::{BldgConfig, BldgTemplate};
use crate::units::UnitTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::tech::TechTemplate;
use crate::player::{Nms};
use crate::ai::AIConfig;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;

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
	pub kbd: KeyboardMap,
	pub l: Localization
}

/* todo?
pub struct Disp {
	disp_settings: DispSettings, // limit characters or colors
	disp_chars: DispChars,
	menu_options: OptionsUI, // top level menu
	production_options: ProdOptions<'bt,'ut,'rt,'dt>, // for bldgs, workers -- could be eventually removed and recomputed each frame when relevant
	txt_list: TxtList, // for screen readers
	buttons: Buttons,
	local: Localization,
	kbd: KeyboardMap,,
	disp_state: DispState // sdl state variables
}

pub struct Map<'bt,'ut,'rt,'dt> {
	map_data: MapData<'rt>,
	exs: Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
}

pub struct GameState {
	relations: Relations,
	logs: Vec<Log>,
	rng: XorState,
	turn: usize
}

pub struct Objs {
	units: Vec<Unit<'bt,'ut,'rt,'dt>>,
	bldgs: Vec<Bldg<'bt,'ut,'rt,'dt>>
}
*/

