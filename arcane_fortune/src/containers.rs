use crate::buildings::{BldgConfig, BldgTemplate};
use crate::units::UnitTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::tech::TechTemplate;
use crate::map::{Nms, Owner};
use crate::ai::AIConfig;
use crate::disp::{DispSettings, DispChars};
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;

// disp_settings and disp_chars, while often passed to fns as immutable, are needed mutably in the menus for changing settings
// save_game() needs mutable refs to `owners` among others not normally passed as mutable. so fns that are called in
//	save_game() cant't reasonably take references to Templates -- for example, unfourtnately, add_bldg()
//	Templates could be created on demand in save_game() but it would be messy and error-prone, potentially
pub struct Templates<'bt,'ut,'rt,'dt,'o,'tt> {
	pub bldgs: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
	pub units: &'ut Vec<UnitTemplate<'rt>>,
	pub doctrines: &'dt Vec<DoctrineTemplate>,
	pub resources: &'rt Vec<ResourceTemplate>,
	pub techs: &'tt Vec<TechTemplate>,
	pub ai_config: AIConfig<'rt>,
	pub bldg_config: BldgConfig,
	pub kbd: KeyboardMap,
	pub nms: Nms,
	pub l: Localization,
	pub owners: &'o Vec<Owner>
}

/*pub struct Map {
	zone_exs_owners
}*/
