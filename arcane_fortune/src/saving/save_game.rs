use std::io;
use std::fs::{self, File};
//use std::io::prelude::*;
//use std::process::exit;
use std::path::Path;

use super::*;
use crate::disp_lib::*;
use crate::config_load::{return_names_list, read_file, get_usize_map_config};
use crate::tech::*;
use crate::gcore::rand::XorState;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx, HashStruct64};
use crate::gcore::{Log, Relations, Bonuses, HUMAN_PLAYER_IND, HUMAN_PLAYER_ID};
use crate::disp::menus::{update_menu_indicators, FindType, OptionsUI};
use crate::ai::{CITY_HEIGHT, CITY_WIDTH, AIState, BarbarianState, place_barbarians, AIConfig, init_ai_config};
use crate::nobility::{House};
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
			let mut ptype = PlayerType::Human {ai_state: Default::default()};
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
	let mut resource_templates2 = resource_templates.clone();
	let mut unit_templates2 = unit_templates.clone();
	let mut bldg_templates2 = bldg_templates.clone();
	let mut iface_settings2 = IfaceSettings::default("".to_string(), HUMAN_PLAYER_ID);
	let map_root = map_data.zoom_out[ZOOM_IND_ROOT].clone();
	let root_map_sz = map_data.map_szs[ZOOM_IND_ROOT];
	let zoom_in_depth = map_data.map_szs.len() - N_EXPLICITLY_STORED_ZOOM_LVLS;
	let mut map_data2 = MapData::default(map_root, root_map_sz.h, root_map_sz.w, zoom_in_depth, map_data.max_zoom_in_buffer_sz, &resource_templates);
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
	let mut tech_templates2 = tech_templates.clone();
	let mut ai_states2 = ai_states.clone();
	let mut ai_config2 = ai_config.clone();
	let mut barbarian_states2 = barbarian_states.clone();
	let mut logs2 = logs.clone();
	let mut zone_exs_owners2 = zone_exs_owners.clone();
	let mut frame_stats2 = frame_stats.clone();
	let mut rng2 = rng.clone();
	
	/////////// load
	let mut offset = 0;
	macro_rules! ld_val{($val:ident) => ($val.ld(&buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates););};
	
	ld_val!(resource_templates2);
	debug_assertq!(*resource_templates == resource_templates2);
	
	ld_val!(unit_templates2);
	debug_assertq!(*unit_templates == unit_templates2);

	ld_val!(bldg_templates2);
	debug_assertq!(*bldg_templates == bldg_templates2);

	ld_val!(turn);
	ld_val!(iface_settings2);

	ld_val!(map_data2);
	
	ld_exs(&mut exs2, &buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates);
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

	ld_val!(tech_templates2);
	debug_assertq!(*tech_templates == tech_templates2);
	
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
	
	ld_zone_exs(&mut zone_exs_owners2, &buf, &mut offset, &bldg_templates, &unit_templates, &resource_templates, &doctrine_templates);
	debug_assertq!(*zone_exs_owners == zone_exs_owners2);
	
	ld_val!(frame_stats2);
	debug_assertq!(*frame_stats == frame_stats2);
	
	ld_val!(rng2);
	debug_assertq!(*rng == rng2);
	
	map_data2.compute_zoom_outs(&mut exs2, &zone_exs_owners2, &bldgs2, bldg_templates, &owners2);
	
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
		bldg_config: &mut BldgConfig, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		relations: &mut Relations, iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>,
		doctrine_templates: &'dt Vec<DoctrineTemplate>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
		unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>,
		nms: &mut Nms, disp_chars: &DispChars, tech_templates: &Vec<TechTemplate>,
		ai_config: &mut AIConfig<'rt>, logs: &mut Vec<Log>, l: &Localization,
		game_opts: &GameOptions, rng: &mut XorState, d: &mut DispState) {
	
	let mut txt_gen = nn::TxtGenerator::new(rng.gen());
	
	*bldg_config = BldgConfig::from_config_file();
	
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
	let country_names = return_names_list(read_file("config/names/countries.txt"));
	
	*ai_config = init_ai_config(resource_templates);
	
	//////// owners setup
	//////////// loop until reasonable map and unit placement
	// each loop attempts N_PLACEMENT_ATTEMPTS before restarting the loop
	'map_attempt: loop {
		///// map setup
		let map_root = map_gen(MapSz {h: H_ROOT, w: W_ROOT, sz: H_ROOT*W_ROOT}, disp_chars, l, rng, d);
		*map_data = MapData::default(map_root, H_ROOT, W_ROOT, game_opts.zoom_in_depth, get_usize_map_config("max_zoom_in_buffer_sz"), resource_templates);
		
		//// exs
		*exs = Vec::with_capacity(map_data.max_zoom_ind() + 1);
		for _ex_ind in 0..=map_data.max_zoom_ind() {
			let s: BuildHasherDefault<HashStruct64> = Default::default();
			exs.push(HashMap::with_hasher(s));
		}
		
		map_data.compute_zoom_outs(exs, players, bldgs, bldg_templates);
		
		//// state and owner vars setup
		{
			*players = Vec::with_capacity(game_opts.n_players + 1);
			
			//////////// add human player and generic AI players			
			for id in 0..(game_opts.n_players) {
				let personality = AIPersonality::new(rng);
				
				let (player_type, bonuses) = 
					if id != HUMAN_PLAYER_IND {
						(PlayerType::AI {ai_state: Default::default(), personality}, game_opts.ai_bonuses.clone())
					}else{
						(PlayerType::Human {ai_state: Default::default()}, Default::default())
					};
				
				// country nm -- prevent duplicates
				let mut nm;
				let mut gender_female;
				let mut ruler_nm;
				'nm_loop: loop {
					nm = country_names[rng.usize_range(0, country_names.len())].clone();
					
					let ruler = PersonName::new(nms, rng);
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
				
				players.push(Player::new(id as SmSvType, player_type, personality, nm, ruler_nm, gender_female,
					&bonuses, PLAYER_COLORS[id], &mut txt_gen, relations, nms, tech_templates, resource_templates, doctrine_templates, map_data, rng));
			}
		}
		
		///////////////////
		
		map_data.compute_zoom_outs(exs, players, bldgs, bldg_templates);
		
		*iface_settings = IfaceSettings::default(players[HUMAN_PLAYER_IND].personalization.save_nm(), HUMAN_PLAYER_ID);
		
		update_menu_indicators(menu_meta, iface_settings, iface_settings.cur_player_paused(players), disp_settings);
				
		bldgs.clear();
		units.clear();
		logs.clear();
		
		*turn = GAME_START_TURN;
		
		/////////////////// put units on map
		{
			let sz_prog = MapSz {h: 0, w: 0, sz: players.len()};
			let mut screen_sz = getmaxyxu(d);
			
			let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
			
			let city_h = CITY_HEIGHT as isize;
			let city_w = CITY_WIDTH as isize;
				
			macro_rules! add_u{($coord:expr, $player:expr, $type:expr) => (
				add_unit($coord, $player.id == iface_settings.cur_player, $type, units, map_data, exs, bldgs, $player, relations, logs, unit_templates, nms, *turn, rng););};
			
			
			//////////// ai, human, and nobility players
			for player_ind in 0..players.len() {
				let player = &mut players[player_ind];
				let mut attempt = 0;
				
				print_map_status(Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(player.id as usize), &mut screen_sz, sz_prog, 0, disp_chars, l, d);
				d.clrtoeol();
				
				'player_attempt: loop{
					let map_coord_y = rng.usize_range(0, map_sz.h - CITY_HEIGHT - 1) as u64;
					let map_coord_x = rng.usize_range(0, map_sz.w - CITY_WIDTH - 1) as u64;
					let map_coord = map_coord_y * map_sz.w as u64 + map_coord_x;
					
					// not clear, try again
					if square_clear(map_coord, ScreenSz{h: CITY_HEIGHT, w: CITY_WIDTH, sz: 0}, Quad::Lr, map_data, exs.last().unwrap()) == None {
						if attempt > N_UNIT_PLACEMENT_ATTEMPTS { // try a new map
							continue 'map_attempt;
						}else{ // try another unit placement
							attempt += 1;
							print_attempt_update(attempt, player.id, &sz_prog, &mut screen_sz, disp_chars, l, d);
							continue 'player_attempt;
						}
					}
					
					match &mut player.ptype {
						PlayerType::AI {ref mut ai_state, personality} => {
							ai_state.set_next_bonus_bldg(&player.stats, personality, bldg_templates, rng);
							ai_state.create_city_plan(Coord {y: map_coord_y as isize, x: map_coord_x as isize}, rng, map_data, map_sz, bldg_templates);
							ai_state.goal_doctrine = Some(&doctrine_templates[rng.usize_range(1, doctrine_templates.len())]);
						} PlayerType::Human {ref mut ai_state} => {
							ai_state.set_next_bonus_bldg(&player.stats, &AIPersonality::default(), bldg_templates, rng);
						} PlayerType::Barbarian {..} | PlayerType::Nobility {..} => {panicq!("shouldn't be added yet");}
					}
					
					// place initial units
					{
						let u_coord = map_sz.coord_wrap(map_coord_y as isize + (CITY_HEIGHT/2) as isize,
											  map_coord_x as isize + (CITY_WIDTH/2) as isize).unwrap();
						
						let u_coord2 = map_sz.coord_wrap(map_coord_y as isize + (CITY_HEIGHT/2) as isize,
											  map_coord_x as isize + (CITY_WIDTH/2) as isize - 1).unwrap();
						
						//endwin();
						add_u!(u_coord, player, UnitTemplate::frm_str(WORKER_NM, unit_templates));
						add_u!(u_coord, player, UnitTemplate::frm_str(WORKER_NM, unit_templates));
						add_u!(u_coord2, player, UnitTemplate::frm_str(EXPLORER_NM, unit_templates));
						
						// set explorer to auto-explore
						if player.ptype.is_ai() {
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
							map_data, exs, doctrine_templates, unit_templates, bldg_templates, tech_templates, resource_templates, relations,
							logs, nms, rng, map_sz, &sz_prog, &mut screen_sz, *turn, disp_chars, l, &mut txt_gen, d) {
						continue 'map_attempt;
					}
					
					///////// tech plan
					let player = &mut players[player_ind];
					if let PlayerType::AI {..} = &player.ptype {
						let mut tech_added = true;
						
						// loop until no techs added. only add techs that have had the reqs already added
						while tech_added {
							tech_added = false;
							'tech_loop: for &tech_ind in rng.inds(tech_templates.len()).iter() {
								// already scheduled
								if player.stats.techs_scheduled.contains(&(tech_ind as SmSvType)) {continue 'tech_loop;}
								
								let tech = &tech_templates[tech_ind];
								
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
						assertq!(tech_templates.len() == player.stats.techs_scheduled.len(), "not all techs added -- orphaned tech requirements?");
						
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
	
	iface_settings.ui_mode = UIMode::InitialGameWindow;
	d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
}

/*pub fn add_owner_vars<'dt>(bonuses: &Bonuses, zone_exs_owners: &mut Vec<HashedMapZoneEx>,
			stats: &mut Vec<Stats<'_,'_,'_,'dt>>, relations: &mut Relations,
			barbarian_states: &mut Vec<Option<BarbarianState>>, ai_states: &mut Vec<Option<AIState>>,
			owners: &Vec<Owner>, tech_templates: &Vec<TechTemplate>, resource_templates: &Vec<ResourceTemplate>,
			doctrine_templates: &'dt Vec<DoctrineTemplate>, map_data: &MapData) {
	let s: BuildHasherDefault<HashStruct64> = Default::default();
	zone_exs_owners.push(HashMap::with_hasher(s));
	
	stats.push(Stats::default_init(bonuses, tech_templates, resource_templates, doctrine_templates, map_data));
	barbarian_states.push(None);
	ai_states.push(None);
	*relations = Relations::new(owners.len());
}*/

pub fn print_attempt_update(attempt: usize, player: SmSvType, sz_prog: &MapSz, screen_sz: &mut ScreenSz, disp_chars: &DispChars, l: &Localization, d: &mut DispState) {
	if (attempt % 25) != 0 {return;}
	
	print_map_status(Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(sz_prog.sz), Some(player as usize), screen_sz, *sz_prog, 0, disp_chars, l, d);
	d.addstr(&format!(" ({}: {}/{})", l.attempt, attempt, N_UNIT_PLACEMENT_ATTEMPTS));
	d.clrtoeol();
}

