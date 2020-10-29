//#![allow(warnings)]
use std::time::*;
use std::path::Path;
use std::fs;

//use std::time::{SystemTime, UNIX_EPOCH};
//use std::process::exit;

#[macro_use]
mod debug;
mod renderer;
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
mod player;
mod containers;

use renderer::*;
use map::*;
use disp::*;
use disp::menus::{init_menus, do_menu_shortcut, UIRet, FindType};
use disp::window::{init_bldg_prod_windows, do_window_keys};
use units::*;
use gcore::*;
use ai::*;
use buildings::*;
//use zones::*;
use saving::*;
use config_load::*;
use tech::init_tech_templates;
use resources::init_resource_templates;
use doctrine::init_doctrine_templates;
use keyboard::KeyboardMap;
use localization::Localization;
use player::*;
use containers::*;

fn main(){
	disp::show_version_status_console();
	
	let mut game_control = GameControl::TitleScreen;
	let mut game_opts = GameOptions {
		zoom_in_depth: 2,
		n_players: PLAYER_COLORS.len(),
		
		ai_bonuses: Default::default()
	};
	let game_difficulties = load_game_difficulties();
	
	let doctrine_templates_junk = Vec::new();
	let resource_templates_junk = Vec::new();
	let unit_templates_junk = Vec::new();
	let bldg_templates_junk = Vec::new();
	
	///////////////
	// loop through (1) loading or creating a new game (2) playing the game
	'outer_loop: loop {
		///////////////////////
		//// placeholders
		
		// temps
		let mut bldg_templates = Vec::new();
		let mut unit_templates = Vec::new();
		let mut doctrine_templates = Vec::new();
		let mut resource_templates = Vec::new();
		let mut tech_templates = Vec::new();
		let mut ai_config = AIConfig::default();
		let mut bldg_config = BldgConfig::default();
		let mut nms = Nms::default(); // city names
		
		let mut disp = Disp::new();
		
		let mut gstate = GameState::default();
		let mut map_data = MapData::default(Vec::new(), 0, 0, 7, 0, &resource_templates_junk);
		let mut exs = Vec::new(); // vec indexed by zoom
		let mut bldgs = Vec::new();
		let mut units = Vec::new();
		let mut players = Vec::new();
		let mut frame_stats = disp::FrameStats::init();
		let temps;
		
		//////////////////////////
		// (1) create new game or load
		match &game_control {
			GameControl::NewOptions => {
				game_control = if disp.state.new_game_options(&mut game_opts, &game_difficulties) {GameControl::New} else {GameControl::TitleScreen};
				continue 'outer_loop;
			} GameControl::New => {
				{ // setup templates (static variables)
					doctrine_templates = init_doctrine_templates(&disp.state.local);
					tech_templates = init_tech_templates(&disp.state.local);
					resource_templates = init_resource_templates(&tech_templates, &disp.state.chars, &disp.state.local);
					unit_templates = init_unit_templates(&tech_templates, &resource_templates, &disp.state.local);
					bldg_templates = init_bldg_templates(&tech_templates, &unit_templates, &doctrine_templates, &disp.state.chars, &disp.state.local);
					
					bldg_config = BldgConfig::from_config_file();
					ai_config = init_ai_config(&resource_templates);
					
					///////// city names setup
					nms.cities.clear();
					{
						const CITY_NMS_DIR: &str = "config/names/cities/";
						let city_nms_dir = Path::new(CITY_NMS_DIR);
						
						// loop over files in the directory, each file contains a seprate them of city names
						if let Result::Ok(dir_entries) = fs::read_dir(city_nms_dir) {
							for entry in dir_entries {
								if let Result::Ok(e) = entry {
									nms.cities.push(return_names_list(read_file(
											e.path().as_os_str().to_str().unwrap())));
								}
							}
						} else {panicq!("failed to open {}", CITY_NMS_DIR);}
					}
					
					////////// unit names setup
					nms.units = return_names_list(read_file("config/names/battalion_names.txt"));
					nms.brigades = return_names_list(read_file("config/names/brigade_names.txt"));
					nms.sectors = return_names_list(read_file("config/names/sector_names.txt"));
					nms.noble_houses = return_names_list(read_file("config/names/noble_houses/english_names.txt"));
					nms.females = return_names_list(read_file("config/names/females.txt")); // names of rulers/nobility
					nms.males = return_names_list(read_file("config/names/males.txt")); // names of rulers/nobility
					
					temps = Templates {
						bldgs: &bldg_templates,
						units: &unit_templates,
						doctrines: &doctrine_templates,
						resources: &resource_templates,
						techs: &tech_templates,
						ai_config: ai_config.clone(),
						bldg_config: bldg_config.clone(),
						nms,
						//kbd: disp.state.kbd.clone(),
						//l: disp.state.lang.clone(),
					};
				}
				
				new_game(&mut gstate, &mut map_data, &mut exs, &mut players, &temps, &mut bldgs, &mut units, &game_opts, &mut disp);
			} GameControl::Load(file_nm) => {
				disp.state.print_clear_centered_logo_txt(disp.state.local.Loading_game.clone());
				
				let buf = read_file_decompress(&file_nm);
				
				let mut offset = 0;
				
				doctrine_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates_junk, &resource_templates_junk, &doctrine_templates_junk);
				resource_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates_junk, &resource_templates_junk, &doctrine_templates);
				unit_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates_junk, &resource_templates, &doctrine_templates);
				bldg_templates.ld(&buf, &mut offset, &bldg_templates_junk, &unit_templates, &resource_templates, &doctrine_templates);
				tech_templates.ld(&buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates);
				ai_config.ld(&buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates);
				bldg_config.ld(&buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates);
				nms.ld(&buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates);
				
				temps = Templates {
					bldgs: &bldg_templates,
					units: &unit_templates,
					doctrines: &doctrine_templates,
					resources: &resource_templates,
					techs: &tech_templates,
					ai_config: ai_config.clone(),
					bldg_config: bldg_config.clone(),
					nms: nms.clone(),
					//kbd: kbd.clone(),
					//l: lang.clone(),
				};
				
				load_game(buf, offset, &mut disp, &mut gstate, &mut map_data, &mut exs, &temps, &mut bldgs, &mut units, &mut players, &mut frame_stats);
				
			} GameControl::TitleScreen => {
				game_control = disp.state.show_title_screen();
				continue 'outer_loop;
			}
		}
		
		disp.state.production_options = init_bldg_prod_windows(&bldg_templates, &players[disp.state.iface_settings.cur_player as usize].stats, &disp.state.local);
		
		const KEY_PRESSED_PAUSE: u32 = 450;
		let mut t_last_key_pressed = Instant::now(); // temporarily pause game if key pressed within KEY_PRESS_PAUSE
		disp.state.set_screen_sz();
		let mut screen_sz_prev = disp.state.iface_settings.screen_sz.clone();
		
		const IDLE_TIMEOUT: usize = 1000; // n_frames when we stop refreshing the screen
		let mut n_idle = 0; // n_frames no keys were pressed
		
		const ALT_RESET: usize = 5000;
		let mut alt_ind = 0; // for multiple units on the same land plot
		let mut last_alt_time = Instant::now(); // when we last changed alt_ind
		
		match &disp.ui_mode {
			UIMode::InitialGameWindow(_) => {}
			_ => {disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VISIBLE);}
		}
		
		///////////////////
		// (2) play game after it's been loaded or a new one's been created
		loop {
			#[cfg(feature="profile")]
			let _g = Guard::new("main frame loop");
			
			for b in bldgs.iter_mut() {
				b.update_fire(&mut gstate.rng);
			}
			
			// clear screen if terminal size changes
			disp.state.set_screen_sz();
			if disp.state.iface_settings.screen_sz != screen_sz_prev {
					disp.state.renderer.clear();
					screen_sz_prev = disp.state.iface_settings.screen_sz.clone();
					disp.state.iface_settings.chk_cursor_bounds(&mut map_data);
			}
			
			// unit flashing indicator (for multiple units on the same land plot)
			// has enough time passed to show next set of units?
			let last_alt_time_updated = if last_alt_time.elapsed().as_millis() as u32 > ALT_DELAY {
				last_alt_time = Instant::now();
				
				if !screen_reader_mode() {
					if alt_ind < ALT_RESET {
						alt_ind += 1;
					}else{
						alt_ind = 0;
					}
				}
				
				true
			}else{false};
			
			// no key was pressed
			disp.state.mouse_event = if disp.state.key_pressed == ERR && disp.state.iface_settings.auto_turn == AutoTurn::Off {
				macro_rules! get_another_key_and_skip_drawing{() => {
					disp.state.key_pressed = disp.state.renderer.getch();
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
				disp.state.renderer.getmouse(disp.state.key_pressed) // mouse event
			};
			
			// cursor
			if disp.state.iface_settings.add_action_to.is_none() {
				disp.state.renderer.set_mouse_to_arrow();
			}else{
				disp.state.renderer.set_mouse_to_crosshair();
			}
			
			////////////////////////////
			// keys
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("main frame loop -> all key actions");
				
				match do_window_keys(&mut map_data, &mut exs, &mut units, &mut bldgs, &mut disp, &mut players, &mut gstate, &temps, &mut game_control, &frame_stats, &game_difficulties) {
					UIRet::Active => {}
					UIRet::ChgGameControl => {break;}
					UIRet::Inactive => {
						match do_menu_shortcut(&mut disp, &mut map_data, &mut exs, &mut players, &mut units, &mut bldgs, &temps, &mut gstate, &frame_stats, &mut game_control, &mut game_opts, &game_difficulties) {
							UIRet::Active => {}
							UIRet::ChgGameControl => {break;}
							UIRet::Inactive => {
								non_menu_keys(&mut map_data, &mut exs, &mut units, &mut bldgs, &temps, &mut players, &mut gstate, &mut frame_stats, &mut disp);
							}
						}
					}
				}
			}
			disp.state.buttons.clear_positions();
			disp.state.txt_list.clear();
			
			// chk if bldg placement valid, if it isn't clear worker's computed path
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("main frame loop -> chk bldg placement valid");
				
				///// make sure building placement is valid (before printing any paths...)				
				if disp.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
					let cur_mc = disp.state.iface_settings.cursor_to_map_coord(&map_data);
					
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
										Some(MatchZoneOwner {zone, owner_id: disp.state.iface_settings.cur_player})
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
										Some(MatchZoneOwner {zone, owner_id: disp.state.iface_settings.cur_player})
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
					
					match &mut disp.state.iface_settings.add_action_to {
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
			
			///////////////
			// screen reader cleanup display
			//	clear the screen if we are not supposed to show the map
			if disp.ui_mode.hide_map() {disp.state.renderer.clear();}
			
			/////////////// show map, windows, update cursor
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("main frame loop -> map/screen printing");
				
				disp.print_map(&mut map_data, &units, &bldgs, &players, &temps, &exs, &gstate, &mut frame_stats, alt_ind);
				disp.print_windows(&mut map_data, exs.last().unwrap(), &units, &bldgs, &temps, &players, &gstate, &game_difficulties);
				
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
				
				if !disp.state.iface_settings.show_expanded_submap {
					disp.state.buttons.print_tool_tip(&disp.state.chars, &mut disp.state.renderer);
				}else{
					disp.state.renderer.set_mouse_to_arrow();
				}
				
				disp.update_cursor(&players[disp.state.iface_settings.cur_player as usize].stats, &mut map_data);
			}
			
			/////////////// auto turn increment (if key hasn't been pressed)
			if t_last_key_pressed.elapsed().as_millis() as u32 > KEY_PRESSED_PAUSE {
				match disp.state.iface_settings.auto_turn {
					AutoTurn::On | AutoTurn::FinishAllActions => {
						for _ in 0..frame_stats.days_per_frame() {
							end_turn(&mut gstate, &mut units, &mut bldgs, &temps, &mut disp, &mut map_data, &mut exs, &mut players, &mut frame_stats);
							
							// ex if the the game has ended or we now show the tech tree from discovering tech, then
							// stop progressing turns (also end_turn() will clear any open windows)
							if disp.state.iface_settings.auto_turn == AutoTurn::Off {break;}
							
							// break if there are unmoved units and auto turn increment is on FinishAllActions
							if !disp.state.iface_settings.all_player_pieces_mvd &&
							   disp.state.iface_settings.auto_turn == AutoTurn::FinishAllActions {
								disp.state.iface_settings.auto_turn = AutoTurn::Off;
								disp.center_on_next_unmoved_menu_item(false, FindType::Units, &mut map_data, &mut exs, &mut units, &mut bldgs, &mut gstate, &mut players);
								break;
							}
						}
					} AutoTurn::Off => {
					} AutoTurn::N => {panicq!("invalid auto turn");}
				}
			}
			
			//d.addstr(&format!("{:#?}", txt_list));
			
			disp.state.renderer.refresh();
			
			disp.state.key_pressed = disp.state.renderer.getch();
			if disp.state.key_pressed != ERR {
				t_last_key_pressed = Instant::now();
			}
			flushinp();
		}
	}
}

