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
	kbd: KeyboardMap,
	disp_state: DispState // sdl state variables
}
*/

/*pub struct Map<'bt,'ut,'rt,'dt> {
	map_data: MapData<'rt>,
	exs: Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
	sz: MapSz
}*/

/*pub struct GameState {
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

/*
pub struct Game {
	objs: Objs,
	players: Vec<Player>,
	gstate: GameState,
	map: Map,
}
*/

// new_unaffiliated_houses(players, objs, map, temps, n_log_entries, game_state)
// new_unaffiliated_houses(game)
// house.plan_actions(is_cur_player, pstats, objs, map_data, exs, relations, map_sz, rng, logs, nms, turn);
// house.plan_actions(is_cur_player, &mut pstats, &mut objs, &mut map, &mut game_state);

// disband_units(disband_unit_inds, cur_ui_player, units, map_data, exs, players, relations, map_sz, logs, turn);
// disband_units(disband_unit_inds, cur_ui_player, units, &mut map, &mut players, &mut gstate)

// mv_units(unit_ind, is_cur_player, units, map_data, exs, bldgs, players, relations, map_sz, del_action, logs, turn)
// mv_unit(unit_ind, del_action, is_cur_player, objs, map, players, gstate)
// mv_unit(unit_ind, del_action, is_cur_player, units, map, players, gstate) //
// mv_unit(unit_ind, del_action, is_cur_player, game)

// action_iface.update_move_search(end_coord, map, mv_vars, bldgs)

// plan_actions(ai_ind, players, units, bldgs, relations, map_data, exs, temps, disband_unit_inds, logs, rng, map_sz, turn, iface_settings, disp_settings, menu_options, cur_ui_ai_player_is_paused, d)
// plan_actions(ai_ind, players, objs, gstate, map, temps, disband_unit_inds, iface_settings, disp_settings, menu_options, cur_ui_ai_player_is_paused)

// add_bldg(coord, bldgs, bt, doctrine_dedication, temps, map_data, exs, players, turn, logs, rng)
// add_bldg(coord, bldgs, bt, doctrine_dedication, temps, map, players, gstate)
// add_bldg(coord, bt, doctrine_dedication, game)

// add_unit(coord, is_cur_player, unit_template, units, map_data, exs, bldgs, player, relations, logs, temps, turn, rng)
// add_unit(coord, is_cur_player, unit_template, player, objs, map, gstate)
