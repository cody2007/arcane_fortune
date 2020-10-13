use std::io;
use std::fs::{File};
//use std::io::prelude::*;
//use std::process::exit;
use std::path::Path;

use super::*;
use crate::disp_lib::*;
use crate::config_load::{return_names_list, read_file, get_usize_map_config};
use crate::gcore::rand::XorState;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx, HashStruct64};
use crate::gcore::{Log, Relations, HUMAN_PLAYER_IND, HUMAN_PLAYER_ID};
use crate::disp::menus::{update_menu_indicators, FindType, OptionsUI};
use crate::player::{PlayerType};
use crate::containers::Templates;
use crate::map::gen_utils::print_map_status;
use crate::localization::Localization;

pub const SAVE_DIR: &str = "saves/";

#[derive(PartialEq)]
pub enum SaveType {Auto, Manual} // if set to manual, name is pulled from iface_settings

pub fn save_game<'f,'bt,'ut,'rt,'dt>(save_type: SaveType, turn: usize, map_data: &MapData<'rt>,
		exs: &Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, units: &Vec<Unit<'bt,'ut,'rt,'dt>>,
		relations: &Relations, iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>,
		players: &Vec<Player>, disp_settings: &DispSettings,
		disp_chars: &DispChars, logs: &Vec<Log>, l: &Localization,
		frame_stats: &FrameStats, rng: &mut XorState, d: &mut DispState){
	let mut buf = Vec::new();
	
	// close any current windows or menus if user manually saved
	if SaveType::Manual == save_type {
		iface_settings.ui_mode = UIMode::None;
		d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
		print_clear_centered_logo_txt(&l.Saving_game, disp_chars, d);
	}
	
	//////////// save
	
	{ // templates
		// not all fields are saved -- some are loaded from config files each game load
		temps.doctrines.sv(&mut buf);
		temps.resources.sv(&mut buf);
		temps.units.sv(&mut buf);
		temps.bldgs.sv(&mut buf);
		temps.techs.sv(&mut buf);
		temps.ai_config.sv(&mut buf);
		temps.bldg_config.sv(&mut buf);
		temps.nms.sv(&mut buf);
	}
	
	{ // players
		// manually saved due to stats having a reference as a field and it not being possible to have a Default
		players.len().sv(&mut buf);
		for player in players.iter() {
			player.id.sv(&mut buf);
			player.ptype.sv(&mut buf);
			player.personalization.sv(&mut buf);
			player.stats.sv(&mut buf);
			player.zone_exs.sv(&mut buf);
		}
	}
	
	turn.sv(&mut buf);
	iface_settings.sv(&mut buf);
	map_data.sv(&mut buf);
	sv_exs(exs, &mut buf);
	bldgs.sv(&mut buf);
	units.sv(&mut buf);
	disp_settings.sv(&mut buf);
	relations.sv(&mut buf);
	logs.sv(&mut buf);
	frame_stats.sv(&mut buf);
	rng.sv(&mut buf);
	
	let file_nm = match save_type {
		SaveType::Auto => {save_nm_date(&players[iface_settings.cur_player as usize].personalization, turn, true, l)},
		SaveType::Manual => {iface_settings.save_nm.clone()}
	};
	
	if let Result::Ok(ref mut file) = File::create(Path::new(&format!("{}/{}", SAVE_DIR, file_nm)).as_os_str()) {
		let mut wtr = snappy::write::FrameEncoder::new(file);
		io::copy(&mut &buf[..], &mut wtr).expect("Saving compressed file failed");
		/*if let Result::Err(_) = file.write_all(&buf) {
			panicq!("failed writing file: {}/{}", SAVE_DIR, file_nm);
		}*/
	}else {panicq!("failed opening file for writing: {}/{}", SAVE_DIR, file_nm);}
	
	///////// test that loading works of saved data
	//#[cfg(any(feature="opt_debug", debug_assertions))]
	//test_save_load(turn, map_data, exs, temps, bldgs, units, relations, iface_settings, players, nms, disp_settings, logs, frame_stats, rng);
	
	if SaveType::Manual == save_type {
		d.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
	}
}

pub fn load_game<'f,'bt,'ut,'rt,'dt>(buf: Vec<u8>, mut offset: usize, menu_meta: &mut OptionsUI, disp_settings: &mut DispSettings, turn: &mut usize, map_data: &mut MapData<'rt>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		relations: &mut Relations, iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, disp_chars: &mut DispChars,
		logs: &mut Vec<Log>, frame_stats: &mut FrameStats,
		rng: &mut XorState, d: &mut DispState){
	
	macro_rules! ld_val{($val:ident) => ($val.ld(&buf, &mut offset, &temps.bldgs, &temps.units, &temps.resources, &temps.doctrines););};
	
	{ // players
		let mut sz: usize = 0;
		sz.ld(&buf, &mut offset, temps.bldgs, temps.units, temps.resources, temps.doctrines);
		*players = Vec::with_capacity(sz);
		for _ in 0..sz {
			let mut id: SmSvType = Default::default();
			let mut ptype = PlayerType::Human(Default::default());
			let mut personalization: Personalization = Default::default();
			let mut stats = Stats::default(temps.doctrines);
			let mut zone_exs: HashedMapZoneEx = Default::default();
			ld_val!(id);
			ld_val!(ptype);
			ld_val!(personalization);
			ld_val!(stats);
			ld_val!(zone_exs);
			players.push(Player {
				id, ptype, personalization, stats, zone_exs
			});
		}
	}
	
	ld_val!(turn);
	ld_val!(iface_settings);
	ld_val!(map_data);
	ld_exs(exs, &buf, &mut offset, &temps.bldgs, &temps.units, &temps.resources, &temps.doctrines);
	ld_val!(bldgs);
	ld_val!(units);
	ld_val!(disp_settings);
	ld_val!(relations);
	ld_val!(logs);
	ld_val!(frame_stats);
	ld_val!(rng);
	
	map_data.compute_zoom_outs(exs, players, bldgs, temps.bldgs);
	
	*disp_chars = init_color_pairs(disp_settings, d);
	update_menu_indicators(menu_meta, iface_settings, iface_settings.cur_player_paused(players), disp_settings);
	iface_settings.set_auto_turn(iface_settings.auto_turn, d); // set frame timeout
	
	//map_data.max_zoom_in_buffer_sz = 8_000_000;
}

/*#[cfg(any(feature="opt_debug", debug_assertions))]
pub fn test_save_load<'f,'bt,'ut,'rt,'dt>(mut turn: usize, map_data: &MapData<'rt>,
		exs: &Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt>,
		bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, units: &Vec<Unit<'bt,'ut,'rt,'dt>>,
		relations: &Relations, iface_settings: &IfaceSettings<'f,'bt,'ut,'rt,'dt>,
		players: &Vec<Player>, nms: &Nms, disp_settings: &DispSettings,
		logs: &Vec<Log>, frame_stats: &FrameStats,
		rng: &XorState){
	
	let mut buf = Vec::new();
	
	//////////// save
	temps.sv(&mut buf);
	turn.sv(&mut buf);
	iface_settings.sv(&mut buf);
	map_data.sv(&mut buf);
	sv_exs(exs, &mut buf);
	bldgs.sv(&mut buf);
	units.sv(&mut buf);
	disp_settings.sv(&mut buf);
	players.sv(&mut buf);
	nms.sv(&mut buf);
	relations.sv(&mut buf);
	logs.sv(&mut buf);
	//sv_zone_exs(zone_exs_owners, &mut buf);
	frame_stats.sv(&mut buf);
	rng.sv(&mut buf);
	
	///////// test that loading works of saved data
	let mut temps.resources2 = temps.resources.clone();
	let mut temps.units2 = temps.units.clone();
	let mut temps.bldgs2 = temps.bldgs.clone();
	let mut iface_settings2 = IfaceSettings::default("".to_string(), HUMAN_PLAYER_ID);
	let map_root = map_data.zoom_out[ZOOM_IND_ROOT].clone();
	let root_map_sz = map_data.map_szs[ZOOM_IND_ROOT];
	let zoom_in_depth = map_data.map_szs.len() - N_EXPLICITLY_STORED_ZOOM_LVLS;
	let mut map_data2 = MapData::default(map_root, root_map_sz.h, root_map_sz.w, zoom_in_depth, map_data.max_zoom_in_buffer_sz, &temps.resources);
	let mut exs2 = exs.clone();
	let mut bldg_config2 = bldg_config.clone();
	let mut bldgs2 = bldgs.clone();
	let mut units2 = units.clone();
	let mut stats2 = stats.clone();
	let mut unaffiliated_houses2 = unaffiliated_houses.clone();
	let mut relations2 = relations.clone();
	let mut disp_settings2 = disp_settings.clone();
	let mut owners2 = owners.clone();
	let mut nms2 = nms.clone();
	let mut temps.techs2 = temps.techs.clone();
	let mut ai_states2 = ai_states.clone();
	let mut ai_config2 = ai_config.clone();
	let mut barbarian_states2 = barbarian_states.clone();
	let mut logs2 = logs.clone();
	let mut zone_exs_owners2 = zone_exs_owners.clone();
	let mut frame_stats2 = frame_stats.clone();
	let mut rng2 = rng.clone();
	
	/////////// load
	let mut offset = 0;
	macro_rules! ld_val{($val:ident) => ($val.ld(&buf, &mut offset, &temps.bldgs, &temps.units, &temps.resources, &temps.doctrines););};
	
	ld_val!(temps.resources2);
	debug_assertq!(*temps.resources == temps.resources2);
	
	ld_val!(temps.units2);
	debug_assertq!(*temps.units == temps.units2);

	ld_val!(temps.bldgs2);
	debug_assertq!(*temps.bldgs == temps.bldgs2);

	ld_val!(turn);
	ld_val!(iface_settings2);

	ld_val!(map_data2);
	
	ld_exs(&mut exs2, &buf, &mut offset, &temps.bldgs, &temps.units, &temps.resources, &temps.doctrines);
	debug_assertq!(*exs == exs2);
	
	ld_val!(bldg_config2);
	debug_assertq!(*bldg_config == bldg_config2);
	
	ld_val!(bldgs2);
	debug_assertq!(*bldgs == bldgs2);
	
	ld_val!(units2);
	debug_assertq!(*units == units2);
	
	ld_val!(disp_settings2);
	debug_assertq!(*disp_settings == disp_settings2);
	
	ld_val!(owners2);
	debug_assertq!(*owners == owners2);
	
	ld_val!(nms2);
	debug_assertq!(*nms == nms2);

	ld_val!(temps.techs2);
	debug_assertq!(*temps.techs == temps.techs2);
	
	ld_val!(stats2);
	//debug_assertq!(*stats == stats2); // likely fails due to floating point comparisions?
	
	ld_val!(unaffiliated_houses2);
	debug_assertq!(*unaffiliated_houses == unaffiliated_houses2);
	
	ld_val!(relations2);
	debug_assertq!(*relations == relations2);
	
	ld_val!(ai_states2);
	debug_assertq!(*ai_states == ai_states2);
	
	ld_val!(ai_config2);
	debug_assertq!(*ai_config == ai_config2);

	ld_val!(barbarian_states2);
	debug_assertq!(*barbarian_states == barbarian_states2, "{:#?}\n\n\n{:#?}", barbarian_states2, *barbarian_states);
	
	ld_val!(logs2);
	debug_assertq!(*logs == logs2);
	
	ld_zone_exs(&mut zone_exs_owners2, &buf, &mut offset, &temps.bldgs, &temps.units, &temps.resources, &temps.doctrines);
	debug_assertq!(*zone_exs_owners == zone_exs_owners2);
	
	ld_val!(frame_stats2);
	debug_assertq!(*frame_stats == frame_stats2);
	
	ld_val!(rng2);
	debug_assertq!(*rng == rng2);
	
	map_data2.compute_zoom_outs(&mut exs2, &zone_exs_owners2, &bldgs2, temps.bldgs, &owners2);
	
	debug_assertq!(map_data.zoom_out[ZOOM_IND_ROOT] == map_data2.zoom_out[ZOOM_IND_ROOT]);
	if map_data.map_szs != map_data2.map_szs {
		endwin();
		for map_sz in map_data.map_szs.iter() {
			println!("{} {} {}", map_sz.h, map_sz.h, map_sz.sz);
		}
		for map_sz in map_data2.map_szs.iter() {
			println!("{} {} {}", map_sz.h, map_sz.h, map_sz.sz);
		}
        panicq!();

	}
	debug_assertq!(map_data.map_szs == map_data2.map_szs);
	debug_assertq!(map_data.max_zoom_in_buffer_sz == map_data2.max_zoom_in_buffer_sz);
}*/

use std::hash::{BuildHasherDefault};
use std::collections::{HashMap};

// returns "first_last_of_country"
impl Personalization {
	fn save_nm_first_part(&self) -> String {
		format!("{}_{}_of_{}", self.ruler_nm.first, self.ruler_nm.last, self.nm)
	}
	
	pub fn save_nm(&self) -> String {format!("{}.af_game", self.save_nm_first_part())}
}

pub fn save_nm_date(personalization: &Personalization, turn: usize, checkpoint: bool, l: &Localization) -> String {
	let mut nm = format!("{}_{}", personalization.save_nm_first_part(), l.date_str_underscores(turn));
	
	if checkpoint {nm.push_str("_autosave");}
	nm.push_str(".af_game");
	nm
}

pub const GAME_START_TURN: usize = 100*12*30;

pub const N_UNIT_PLACEMENT_ATTEMPTS: usize = 2000;
use crate::nn;

pub fn new_game<'f,'bt,'ut,'rt,'dt>(menu_meta: &mut OptionsUI, disp_settings: &DispSettings,
		turn: &mut usize, map_data: &mut MapData<'rt>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		relations: &mut Relations, iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>,
		disp_chars: &DispChars, logs: &mut Vec<Log>, l: &Localization,
		game_opts: &GameOptions, rng: &mut XorState, d: &mut DispState) {
	
	let mut txt_gen = nn::TxtGenerator::new(rng.gen());
	
	*turn = GAME_START_TURN;
	
	let country_names = return_names_list(read_file("config/names/countries.txt"));
	
	//////// owners setup
	//////////// loop until reasonable map and unit placement
	// each loop attempts N_PLACEMENT_ATTEMPTS before restarting the loop
	'map_attempt: loop {
		///// map setup
		let map_root = map_gen(MapSz {h: H_ROOT, w: W_ROOT, sz: H_ROOT*W_ROOT}, disp_chars, l, rng, d);
		*map_data = MapData::default(map_root, H_ROOT, W_ROOT, game_opts.zoom_in_depth, get_usize_map_config("max_zoom_in_buffer_sz"), temps.resources);
		
		//// exs
		*exs = Vec::with_capacity(map_data.max_zoom_ind() + 1);
		for _ex_ind in 0..=map_data.max_zoom_ind() {
			let s: BuildHasherDefault<HashStruct64> = Default::default();
			exs.push(HashMap::with_hasher(s));
		}
		
		map_data.compute_zoom_outs(exs, &Vec::new(), bldgs, temps.bldgs);
		
		bldgs.clear();
		units.clear();
		logs.clear();
		
		/////////////////// put units on map
		{
			let sz_prog = MapSz {h: 0, w: 0, sz: players.len()};
			let mut screen_sz = getmaxyxu(d);
			
			let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
			
			let city_h = CITY_HEIGHT as isize;
			let city_w = CITY_WIDTH as isize;
				
			macro_rules! add_u{($coord:expr, $player:expr, $type:expr) => (
				add_unit($coord, $player.id == HUMAN_PLAYER_IND as SmSvType, $type, units, map_data, exs, bldgs, $player, relations, logs, temps, *turn, rng););};
			
			
			//////////// ai, human, and nobility players
			for player_ind in 0..game_opts.n_players {
				let mut attempt = 0;
				
				print_map_status(Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(player_ind as usize), &mut screen_sz, sz_prog, 0, disp_chars, l, d);
				d.clrtoeol();
				
				'player_attempt: loop {
					// city location
					let map_coord_y = rng.usize_range(0, map_sz.h - CITY_HEIGHT - 1) as u64;
					let map_coord_x = rng.usize_range(0, map_sz.w - CITY_WIDTH - 1) as u64;
					let map_coord = map_coord_y * map_sz.w as u64 + map_coord_x;
					
					let exf = exs.last().unwrap();
					
					// not clear, try again
					if square_clear(map_coord, ScreenSz{h: CITY_HEIGHT, w: CITY_WIDTH, sz: 0}, Quad::Lr, map_data, exf) == None {
						if attempt > N_UNIT_PLACEMENT_ATTEMPTS { // try a new map
							continue 'map_attempt;
						}else{ // try another unit placement
							attempt += 1;
							print_attempt_update(attempt, player_ind as SmSvType, &sz_prog, &mut screen_sz, disp_chars, l, d);
							continue 'player_attempt;
						}
					}
					
					{ // add player
						let coord = Coord {y: map_coord_y as isize, x: map_coord_x as isize};
						
						let (ptype, bonuses, personality) = 
							if player_ind != HUMAN_PLAYER_IND {
								if let Some(empire_state) = EmpireState::new(coord, temps, map_data, exf, rng) {
									let personality = empire_state.personality;
									(PlayerType::Empire(empire_state), game_opts.ai_bonuses.clone(), personality)
								}else{continue 'player_attempt;}
							
							// human player
							}else{
								let city_hall = BldgTemplate::frm_str(CITY_HALL_NM, temps.bldgs);
								if let Some(ai_state) = AIState::new(coord, CITY_GRID_HEIGHT, MIN_DIST_FRM_CITY_CENTER, city_hall, temps, map_data, exf, map_sz, rng) {
									(PlayerType::Human(ai_state), Default::default(), Default::default())
								}else{continue 'player_attempt;}
							};
						
						// country nm -- prevent duplicates
						let (nm, gender_female, ruler_nm) = {
							let mut nm;
							let mut gender_female;
							let mut ruler_nm;
							'nm_loop: loop {
								nm = country_names[rng.usize_range(0, country_names.len())].clone();
								
								let ruler = PersonName::new(&temps.nms, rng);
								gender_female = ruler.0;
								ruler_nm = ruler.1;
								
								// prevent duplicates of names
								for player in players.iter() {
									let pers = &player.personalization;
									if ruler_nm == pers.ruler_nm || nm == pers.nm {
										continue 'nm_loop;
									}
								}
								break;
							} // end prevent duplicates
							(nm, gender_female, ruler_nm)
						};
						
						let player = Player::new(player_ind as SmSvType, ptype, personality, nm, ruler_nm, gender_female,
							&bonuses, PLAYER_COLORS[player_ind], &mut txt_gen, relations, 0, temps, map_data, *turn, rng);
						
						// ifacesettings
						if player_ind == HUMAN_PLAYER_IND {
							*iface_settings = IfaceSettings::default(player.personalization.save_nm(), HUMAN_PLAYER_ID);
							update_menu_indicators(menu_meta, iface_settings, iface_settings.cur_player_paused(players), disp_settings);
						}
						
						players.push(player);
					}
					
					// place initial units
					{
						let player = &mut players[player_ind];
						let u_coord = map_sz.coord_wrap(map_coord_y as isize + (CITY_HEIGHT/2) as isize,
											  map_coord_x as isize + (CITY_WIDTH/2) as isize).unwrap();
						
						let u_coord2 = map_sz.coord_wrap(map_coord_y as isize + (CITY_HEIGHT/2) as isize,
											  map_coord_x as isize + (CITY_WIDTH/2) as isize - 1).unwrap();
						
						//endwin();
						add_u!(u_coord, player, UnitTemplate::frm_str(WORKER_NM, temps.units));
						add_u!(u_coord, player, UnitTemplate::frm_str(WORKER_NM, temps.units));
						add_u!(u_coord2, player, UnitTemplate::frm_str(EXPLORER_NM, temps.units));
						
						// set explorer to auto-explore
						if player.ptype.is_empire() {
							let u = &mut units.last_mut().unwrap();
							assertq!(u.template.nm[0] == EXPLORER_NM);
							u.action.push(ActionMeta::new(ActionType::AutoExplore {
									start_coord: u.return_coord(),
									explore_type: ExploreType::Random
							}));
						}
					}
					
					//////////// place barbarians
					if !place_barbarians(&mut attempt, player_ind as SmSvType, city_h, city_w, map_coord_y, map_coord_x, &game_opts.ai_bonuses, players, units, bldgs,
							map_data, exs, temps, relations, logs, rng, map_sz, &sz_prog, &mut screen_sz, *turn, disp_chars, l, &mut txt_gen, d) {
						continue 'map_attempt;
					}
					
					///////// tech plan
					let player = &mut players[player_ind];
					if player.ptype.is_empire() {
						let mut tech_added = true;
						
						// loop until no techs added. only add techs that have had the reqs already added
						while tech_added {
							tech_added = false;
							'tech_loop: for &tech_ind in rng.inds(temps.techs.len()).iter() {
								// already scheduled
								if player.stats.techs_scheduled.contains(&(tech_ind as SmSvType)) {continue 'tech_loop;}
								
								let tech = &temps.techs[tech_ind];
								
								let schedule_tech = || {
									// check if tech reqs already added
									if let Some(tech_reqs) = &tech.tech_req {
										// true if all  tech reqs are already scheduled
										return tech_reqs.iter().all(|tech_req|
											player.stats.techs_scheduled.contains(tech_req));
									}
									// no tech reqs so it can be added immediately
									true
								};
								
								if schedule_tech() {
									player.stats.techs_scheduled.push(tech_ind as SmSvType);
									tech_added = true;
								}
							} // tech loop
						} // loop until no techs added
						
						// check that all techs added
						assertq!(temps.techs.len() == player.stats.techs_scheduled.len(), "not all techs added -- orphaned tech requirements?");
						
						// reverse order -- first added has no requirements, but techs are scheduled by popping off values
						let mut techs_scheduled_rev = Vec::with_capacity(player.stats.techs_scheduled.len());
						for ind in player.stats.techs_scheduled.iter().rev() {
							techs_scheduled_rev.push(*ind);
						}
						
						player.stats.techs_scheduled = techs_scheduled_rev;
					}
					break 'player_attempt;
				} // player placement loop
			} // place each player
			
			// print done
			print_map_status(Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), &mut screen_sz, sz_prog, 0, disp_chars, l, d);
		} // place players
		
		break 'map_attempt;
	}
	
	// center cursor on map
	iface_settings.set_screen_sz(d);
	iface_settings.center_on_next_unmoved_menu_item(true, FindType::Units, map_data, exs, units, bldgs, relations, players, logs, *turn, d);
	
	iface_settings.create_window(UIMode::InitialGameWindow, d);
}

pub fn print_attempt_update(attempt: usize, player: SmSvType, sz_prog: &MapSz, screen_sz: &mut ScreenSz, disp_chars: &DispChars, l: &Localization, d: &mut DispState) {
	if (attempt % 25) != 0 {return;}
	
	print_map_status(Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(player as usize), screen_sz, *sz_prog, 0, disp_chars, l, d);
	d.addstr(&format!(" ({}: {}/{})", l.attempt, attempt, N_UNIT_PLACEMENT_ATTEMPTS));
	d.clrtoeol();
}

