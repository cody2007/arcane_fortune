use std::time::*;
//use std::time::{SystemTime, UNIX_EPOCH};
//use std::process::exit;

#[macro_use]
mod debug;
mod disp_lib;
#[macro_use]
mod config_load;
#[macro_use]
mod saving;
#[macro_use]
mod map;
mod disp;
mod movement;
#[macro_use]
mod units;
mod gcore;
mod buildings;
mod zones;
mod tech;
mod resources;
mod doctrine;
mod ai;
mod nn;
mod keyboard;
mod localization;
mod nobility;

use disp_lib::*;
use map::*;
use disp::*;
use disp::menus::{init_menus, do_menu_shortcut, UIRet, FindType};
use disp::window::{init_bldg_prod_windows, do_window_keys};
use units::*;
use gcore::*;
use ai::AIConfig;
use buildings::*;
//use zones::*;
use saving::*;
use config_load::*;
use tech::init_tech_templates;
use resources::init_resource_templates;
use doctrine::init_doctrine_templates;
use keyboard::KeyboardMap;
use localization::Localization;
use nobility::*;

fn main(){
	disp::show_version_status_console();
	
	let mut lang = Localization::new();
	let (mut d, mut disp_settings) = init_display();
	let mut disp_chars = init_color_pairs(&disp_settings, &mut d);
	
	let mut game_state = GameState::TitleScreen;
	let mut game_opts = GameOptions {
			zoom_in_depth: 2,
			n_players: PLAYER_COLORS.len(),
			
			ai_bonuses: Default::default()
	};
	let game_difficulties = load_game_difficulties();
	let kbd = KeyboardMap::new();
	
	let doctrine_templates_junk = Vec::new();
	let resource_templates_junk = Vec::new();
	let unit_templates_junk = Vec::new();
	let bldg_templates_junk = Vec::new();
	
	///////////////
	// loop through (1) loading or creating a new game (2) playing the game
	'outer_loop: loop {
		///////////////////////
		//// placeholders
		let mut unit_templates = Vec::new();
		let mut bldg_templates = Vec::new();
		
		let mut iface_settings = IfaceSettings::default("".to_string(), 0);
		let mut menu_options = init_menus(&mut iface_settings, &vec![None; 1], &disp_settings);
		let mut map_data = MapData::default(Vec::new(), 0, 0, 7, 0, &resource_templates_junk);
		let mut owners = Vec::new();
		let mut relations = Relations::default();
		let mut nms = Nms::default(); // city names
		let mut exs = Vec::new(); // vec indexed by zoom
		let mut zone_exs_owners = Vec::new(); // vec indexed by owner
		let mut bldg_config = BldgConfig::default();
		let mut bldgs = Vec::new();
		let mut units = Vec::new();
		let mut stats = Vec::new();
		let mut doctrine_templates = Vec::new();
		let mut tech_templates = Vec::new();
		let mut resource_templates = Vec::new();
		let mut ai_states = Vec::new();
		let mut ai_config = AIConfig::default();
		let mut barbarian_states = Vec::new();
		let mut logs = Vec::new();
		let mut frame_stats = disp::FrameStats::init();
		let mut turn = 0;
		let mut rng = XorState::clock_init();
		let mut buttons = Buttons::new(&kbd, &lang);
		
		//////////////////////////
		// (1) create new game or load
		match &game_state {
			GameState::NewOptions => {
				game_state = if new_game_options(&mut game_opts, &game_difficulties, &lang, &disp_chars, &mut buttons, &mut d) {GameState::New} else {GameState::TitleScreen};
				continue 'outer_loop;
			} GameState::New => {
				doctrine_templates = init_doctrine_templates(&lang);
				tech_templates = init_tech_templates(&lang);
				resource_templates = init_resource_templates(&tech_templates, &disp_chars, &lang);
				unit_templates = init_unit_templates(&tech_templates, &resource_templates, &lang);
				bldg_templates = init_bldg_templates(&tech_templates, &unit_templates, &doctrine_templates, &disp_chars, &lang);
				
				new_game(&mut menu_options, &mut disp_settings, &mut turn, &mut map_data, &mut exs, &mut zone_exs_owners, &mut bldg_config, &mut bldgs, &mut units, &mut stats, &mut relations, &mut iface_settings, &doctrine_templates, &bldg_templates,
						&unit_templates, &resource_templates, &mut owners, &mut nms, &mut disp_chars, &tech_templates, &mut ai_states, &mut ai_config, &mut barbarian_states, &mut logs, &lang, &game_opts, &mut rng, &mut d);
			} GameState::Load(file_nm) => {
				print_clear_centered_logo_txt(&lang.Loading_game, &disp_chars, &mut d);
				
				let buf = read_file_decompress(&file_nm);
				
				let mut offset = 0;
				
				doctrine_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates_junk, &resource_templates_junk, &doctrine_templates_junk);
				resource_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates_junk, &resource_templates_junk, &doctrine_templates);
				unit_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates_junk, &resource_templates, &doctrine_templates);
				bldg_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates, &resource_templates, &doctrine_templates);
				
				load_game(buf, offset, &mut menu_options, &mut disp_settings, &mut turn, &mut map_data, &mut exs, &mut zone_exs_owners, &mut bldg_config, &mut bldgs, &mut units, &mut stats, &mut relations, &mut iface_settings, &doctrine_templates, &bldg_templates,
						&unit_templates, &resource_templates, &mut owners, &mut nms, &mut disp_chars, &mut tech_templates, &mut ai_states, &mut ai_config, &mut barbarian_states, &mut logs, &mut frame_stats, &mut rng, &mut d);
				
			} GameState::TitleScreen => {
				game_state = show_title_screen(&disp_chars, &kbd, &mut buttons, &mut lang, &mut d);
				continue 'outer_loop;
			}
		}
		
		let mut production_options = init_bldg_prod_windows(&bldg_templates, &stats[iface_settings.cur_player as usize], &lang);
		
		const KEY_PRESSED_PAUSE: u32 = 450;
		let mut t_last_key_pressed = Instant::now(); // temporarily pause game if key pressed within KEY_PRESS_PAUSE
		let mut key_pressed = 0_i32;
		iface_settings.set_screen_sz(&mut d);
		let mut screen_sz_prev = iface_settings.screen_sz.clone();
		
		const IDLE_TIMEOUT: usize = 1000; // n_frames when we stop refreshing the screen
		let mut n_idle = 0; // n_frames no keys were pressed
		
		const ALT_RESET: usize = 5000;
		let mut alt_ind = 0; // for multiple units on the same land plot
		let mut last_alt_time = Instant::now(); // when we last changed alt_ind
		
		match &iface_settings.ui_mode {
			UIMode::InitialGameWindow => {}
			_ => {d.curs_set(CURSOR_VISIBILITY::CURSOR_VISIBLE);}
		}
		
		//// tmp
		{
			stats[0].houses.houses.push(House::new(&nms, &mut rng, turn));
		}
		
		///////////////////
		// (2) play game after it's been loaded or a new one's been created
		loop {
			#[cfg(feature="profile")]
			let _g = Guard::new("main frame loop");
			
			for b in bldgs.iter_mut() {
				b.update_fire(&mut rng);
			}
			
			// clear screen if terminal size changes
			iface_settings.set_screen_sz(&mut d);
			if iface_settings.screen_sz != screen_sz_prev {
					d.clear();
					screen_sz_prev = iface_settings.screen_sz.clone();
					iface_settings.chk_cursor_bounds(&mut map_data);
			}
			
			// unit flashing indicator (for multiple units on the same land plot)
			// has enough time passed to show next set of units?
			let last_alt_time_updated = if last_alt_time.elapsed().as_millis() as u32 > ALT_DELAY {
				last_alt_time = Instant::now();
				
				if alt_ind < ALT_RESET {
					alt_ind += 1;
				}else{
					alt_ind = 0;
				}
				
				true
			}else{false};
			
			// no key was pressed
			let mouse_event = if key_pressed == ERR && iface_settings.auto_turn == AutoTurn::Off {
				macro_rules! get_another_key_and_skip_drawing{() => {
					key_pressed = d.getch();
					continue;
				};};
				
				// not yet at idle timeout, inc counter. and redraw screen if units need to be alternated
				if n_idle < IDLE_TIMEOUT {
					n_idle += 1;
					
					// units should be alternated if more than one per land plot
					if last_alt_time_updated {
						None // no mouse event
					}else{get_another_key_and_skip_drawing!();}
				// at idle timeout, don't do anything
				}else{get_another_key_and_skip_drawing!();}
			// a key was pressed
			}else{
				n_idle = 0;
				d.getmouse(key_pressed) // mouse event
			};
			
			// cursor
			if iface_settings.add_action_to.is_none() {
				d.set_mouse_to_arrow();
			}else{
				d.set_mouse_to_crosshair();
			}
			
			////////////////////////////
			// keys
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("main frame loop -> all key actions");
				
				match do_window_keys(key_pressed, &mouse_event, &mut map_data, &mut exs, &mut units, &bldg_config, &mut bldgs, &mut production_options, &mut iface_settings, &mut stats, &mut relations, &owners, &doctrine_templates, &unit_templates, &bldg_templates, &resource_templates, &tech_templates, &mut logs, &mut zone_exs_owners, &mut disp_settings, &disp_chars, &mut ai_states, &ai_config, &mut barbarian_states, &nms, &mut menu_options, &frame_stats, turn, &mut game_state, &game_difficulties, &mut rng, &kbd, &lang, &mut buttons, &mut d) {
					UIRet::Active => {}
					UIRet::ChgGameState => {break;}
					UIRet::Inactive => {
						match do_menu_shortcut(key_pressed, &mouse_event, &mut menu_options, &mut map_data, &mut exs, &mut zone_exs_owners, &mut iface_settings, &mut disp_settings, &lang, &mut disp_chars, &mut units, &bldg_config, &mut bldgs, &mut owners, &mut nms, &doctrine_templates, &bldg_templates, &unit_templates, &mut turn, &mut stats, &mut relations, &mut tech_templates, &resource_templates, &mut ai_states, &ai_config, &mut barbarian_states, &mut logs, &mut production_options, &frame_stats, &mut game_state, &mut game_opts, &game_difficulties, &lang, &mut buttons, &mut rng, &mut d) {
							UIRet::Active => {}
							UIRet::ChgGameState => {break;}
							UIRet::Inactive => {
								non_menu_keys(key_pressed, &mouse_event, &mut turn, &mut map_data, &mut exs, &mut zone_exs_owners, &mut units, &bldg_config, &mut bldgs, &doctrine_templates, &unit_templates, &bldg_templates, &tech_templates, &resource_templates, &mut stats, &mut relations, &owners, &nms, &mut iface_settings, &mut production_options, &mut ai_states, &ai_config, &mut barbarian_states, &mut logs, &disp_settings, &disp_chars, &mut menu_options, &mut frame_stats, &kbd, &mut buttons, &lang, &mut rng, &mut d);
							}
						}
					}
				}
			}
			buttons.clear_positions();
			
			// chk if bldg placement valid, if it isn't clear worker's computed path
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("main frame loop -> chk bldg placement valid");
				
				///// make sure building placement is valid (before printing any paths...)				
				if iface_settings.zoom_ind == map_data.max_zoom_ind() {
					let cur_mc = iface_settings.cursor_to_map_coord(&map_data);
					
					macro_rules! chk_if_bldg_is_in_valid_position{($valid_placement: expr, $template: expr,
							$path_coords: expr, $actions_req: expr, $chk_dock: lifetime, $adj_chk: lifetime,
							$outer: lifetime) => {
						*$valid_placement = true;
						let exf = exs.last().unwrap();
						
						let h = $template.sz.h as isize;
						let w = $template.sz.w as isize;
						
						let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
						
						// docks must touch water
						if $template.nm[0] == DOCK_NM {
							let mut touching_water = false;
							let mut touching_land = false;
							
							$chk_dock: for i_off in 0..h {
							for j_off in 0..w {
								if let Some(coord) = map_sz.coord_wrap(cur_mc.y + i_off, cur_mc.x + j_off + 1) {
									let mzo = if let BldgType::Taxable(zone) = $template.bldg_type {
										Some(MatchZoneOwner {zone, owner_id: iface_settings.cur_player})
									}else{
										None
									};
									
									let mfc = map_data.get(ZoomInd::Full, coord);
									if mfc.map_type == MapType::ShallowWater {
										touching_water = true;
									}else if mfc.map_type == MapType::Land {
										touching_land = true;
										
										if !land_clear(coord, mzo, &mfc, exf) {
											// clear path, if placement invalid
											$path_coords.clear();
											*$actions_req = 0.;
											
											*$valid_placement = false;
											break $chk_dock;
										}
										
										// any adjacent spot water?
										if !touching_water {
											$adj_chk: for i_adj_off in -1..=1 {
											for j_adj_off in -1..=1 {
												if let Some(coord) = map_sz.coord_wrap(cur_mc.y + i_off + i_adj_off, 
																cur_mc.x + j_off + j_adj_off) {
													if map_data.get(ZoomInd::Full, coord).map_type == MapType::ShallowWater {
														touching_water = true;
														break $adj_chk;
													}
												}
											}} // i/j
										} // adj chk
									} // land
								} // valid coord
							}} // i/j
							if !touching_water || !touching_land {
								*$valid_placement = false;
							}
							
						// all other bldgs must be entirely on land
						}else{
							$outer: for i_off in 0..h {
							for j_off in 0..w {
								if let Some(coord) = map_sz.coord_wrap(cur_mc.y + i_off, cur_mc.x + j_off + 1) {
									let mzo = if let BldgType::Taxable(zone) = $template.bldg_type {
										Some(MatchZoneOwner {zone, owner_id: iface_settings.cur_player})
									}else{
										None
									};
									if !land_clear(coord, mzo, &map_data.get(ZoomInd::Full, coord), exf) {
										// clear path, if placement invalid
										$path_coords.clear();
										*$actions_req = 0.;
										
										*$valid_placement = false;
										break $outer;
									} // land not clear
								} // valid coord
							}} // i/j
						}
					};};
					
					match &mut iface_settings.add_action_to {
						AddActionTo::BrigadeBuildList {
							action: Some(
								ActionMeta {
									action_type: ActionType::WorkerBuildBldg {
										template,
										ref mut valid_placement, ..
									},
									ref mut path_coords,
									ref mut actions_req, ..
								}
							), ..
						} => {
							chk_if_bldg_is_in_valid_position!(valid_placement, template, path_coords, actions_req, 'chk_dock, 'adj_loop, 'outer);
						}
						AddActionTo::IndividualUnit {
							action_iface: ActionInterfaceMeta {
								start_coord,
								action: ActionMeta {
									action_type: ActionType::WorkerBuildBldg {
										template,
										ref mut valid_placement, ..
									},
									ref mut path_coords,
									ref mut actions_req, ..
								}, ..
							}
						} => {
							// if the cursor is not at the start location and there is no path, loc is unreachable
							if *start_coord != cur_mc && path_coords.len() == 0 {
								*valid_placement = false;
							// chk if building can be constructed
							}else{
								chk_if_bldg_is_in_valid_position!(valid_placement, template, path_coords, actions_req, 'chk_dock2, 'adj_loop2, 'outer2);
							}
						}
						
						AddActionTo::BrigadeBuildList {..} | AddActionTo::IndividualUnit {..} |
						AddActionTo::None | AddActionTo::NoUnit {..} | AddActionTo::AllInBrigade {..} => {}
					}
				} // action present at full zoom
			}
			
			/////////////// show map, windows, update cursor
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("main frame loop -> map/screen printing");
				
				iface_settings.print_map(&mut menu_options, &disp_chars, &mut map_data, &units, &bldg_config, &bldgs, &owners, &ai_states, &stats, &tech_templates, &doctrine_templates, &zone_exs_owners, &exs, &relations, &logs, &mut frame_stats, alt_ind, turn, &kbd, &lang, &mut buttons, &mut d);
				iface_settings.print_windows(&mut map_data, exs.last().unwrap(), &units, &bldgs, &production_options, &disp_chars, &unit_templates, &bldg_templates, &tech_templates, &resource_templates, &doctrine_templates, &owners, &stats, &relations, &game_difficulties, &logs, turn, &kbd, &lang, &mut buttons, &mut d);
				
				// dbg, show cursor coordinates
				/*{
					d.mv(0, 85);
					//if let Some(mouse_event) = mouse_event {
					//	d.addstr(&format!("{} {}", mouse_event.y, mouse_event.x));//, mouse_event.bstate));
					//}
					d.addstr(&format!("{} zoom {} {}", iface_settings.cur_player, iface_settings.zoom_ind, iface_settings.cursor_to_map_coord(&map_data)));
				}*/
				
				// dbg show buffer utilization
				/*d.mv(30,0);
				addstr(&num_format(map_data.deque_zoom_in.len()));*/
				
				if !iface_settings.show_expanded_submap {
					buttons.print_tool_tip(&disp_chars, &mut d);
				}else{
					d.set_mouse_to_arrow();
				}
				
				iface_settings.update_cursor(&stats[iface_settings.cur_player as usize], &mut map_data, &disp_chars, &mut d);
			}
			
			/////////////// auto turn increment (if key hasn't been pressed)
			if t_last_key_pressed.elapsed().as_millis() as u32 > KEY_PRESSED_PAUSE {
				match iface_settings.auto_turn {
					AutoTurn::On | AutoTurn::FinishAllActions => {
						for _ in 0..frame_stats.days_per_frame() {
							end_turn(&mut turn, &mut units, &bldg_config, &mut bldgs, &doctrine_templates, &unit_templates, &resource_templates, &bldg_templates, &tech_templates, &mut map_data, &mut exs, &mut zone_exs_owners, &mut stats, &mut relations, &owners, &nms, &mut iface_settings, &mut production_options, &mut ai_states, &ai_config, &mut barbarian_states, &mut logs, &disp_settings, &disp_chars, &mut menu_options, &mut frame_stats, &mut rng, &kbd, &lang, &mut buttons, &mut d);
							
							// ex if the the game has ended or we now show the tech tree from discovering tech, then
							// stop progressing turns (also end_turn() will clear any open windows)
							if iface_settings.auto_turn == AutoTurn::Off {break;}
							
							// break if there are unmoved units and auto turn increment is on FinishAllActions
							if !iface_settings.all_player_pieces_mvd &&
							   iface_settings.auto_turn == AutoTurn::FinishAllActions {
								iface_settings.auto_turn = AutoTurn::Off;
								iface_settings.center_on_next_unmoved_menu_item(false, FindType::Units, &mut map_data, &mut exs, &mut units, &mut bldgs, &mut relations, &owners, &mut barbarian_states, &mut ai_states, &mut stats, &mut logs, turn, &mut d);
								break;
							}
						}
					} AutoTurn::Off => {
					} AutoTurn::N => {panicq!("invalid auto turn");}
				}
			}
			
			d.refresh();
			
			key_pressed = d.getch();
			if key_pressed != ERR {
				t_last_key_pressed = Instant::now();
			}
			flushinp();
		}
	}
}

