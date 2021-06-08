use crate::map::*;
use crate::disp::*;
use crate::gcore::*;
use crate::saving::*;
use crate::config_load::*;
//use crate::renderer::endwin;
use crate::personalization::*;
use crate::containers::*;
//use crate::tree::*;
use crate::units::*;
use crate::buildings::*;
use crate::resources::*;
use crate::doctrine::*;
use crate::ai::*;
use crate::renderer::endwin;

const WORLD_MAP_SCENARIO_CONFIG_FILE: &str = "config/world_scenario/countries.txt";

fn return_start_loc(zoom_in_depth: usize, keys: &Vec<KeyPair>) -> Option<Coord> {
	let (start_y_txt, start_x_txt) = match zoom_in_depth {
		3 => ("start_3_y", "start_3_x"),
		2 => ("start_2_y", "start_2_x"),
		_ => panicq!("unsupported zoom_in_depth")
	};
	
	// return Coord:
	if let Some(start_y) = find_opt_key_parse(start_y_txt, keys) {
		Some(Coord {y: start_y, x: find_req_key_parse(start_x_txt, keys)})
	}else{None}
}

impl GameOptions {
	pub fn init_random_opts(&mut self) {
		self.zoom_in_depth = 2;
		self.player_personalizations = vec![None; 12];
		self.human_player_ind = HUMAN_PLAYER_IND;
		self.load_world_map = false;
		self.ai_bonuses = Bonuses::default();
	}
	
	pub fn set_world_scenario_start_locs(&mut self) {
		let key_sets = config_parse(read_file(WORLD_MAP_SCENARIO_CONFIG_FILE));
		for (keys, player_personalization) in key_sets.iter().zip(self.player_personalizations.iter_mut()) {
			//printlnq!("coord {} zoom_in_depth {}", return_start_loc(self.zoom_in_depth, keys).unwrap(), self.zoom_in_depth);
			player_personalization.as_mut().unwrap().start_loc = return_start_loc(self.zoom_in_depth, keys);
		}
	}
	
	pub fn init_world_map_scenario(&mut self, nms: &Nms, rng: &mut XorState) {
		self.zoom_in_depth = 2;
		self.player_personalizations.clear();
		self.human_player_ind = HUMAN_PLAYER_IND;
		self.load_world_map = true;
		self.ai_bonuses = Bonuses::default();
		
		let key_sets = config_parse(read_file(WORLD_MAP_SCENARIO_CONFIG_FILE));
		
		// loop over countries
		for (player_ind, keys) in key_sets.iter().enumerate() {
			let personality = AIPersonality {
				friendliness: find_req_key_parse("friendliness", keys),
				spirituality: find_req_key_parse("spirituality", keys)
			};
			
			let ruler_nm = PersonName {
				first: find_key_parse("leader_first", String::new(), keys),
				last: find_req_key("leader_last", keys)
			};
			
			// start location for the civilization
			let start_loc = return_start_loc(self.zoom_in_depth, keys);
			let country_nm = find_req_key("nm", keys);
			
			self.player_personalizations.push(Some(
				ScenarioPersonalization {
					start_loc,
					personalization: Personalization {
						color: PLAYER_COLORS[player_ind % PLAYER_COLORS.len()], //PLAYER_COLORS[player_ind],
						city_nm_theme: nms.return_city_theme_ind(&country_nm),
						nm: country_nm,
						nm_adj: find_req_key("nm_adj", keys),
						gender_female: find_req_key_parse("gender_female", keys),
						ruler_nm,
						personality,
						doctrine_advisor_nm: PersonName::new(nms, rng).1,
						crime_advisor_nm: PersonName::new(nms, rng).1,
						pacifism_advisor_nm: PersonName::new(nms, rng).1,
						health_advisor_nm: PersonName::new(nms, rng).1,
						unemployment_advisor_nm: PersonName::new(nms, rng).1,
						motto: find_req_key("motto", keys),
						founded: find_req_key_parse("founded", keys)
					}
				}
			));
		}
	}
	
	pub fn gen_or_load_map_root<'bt,'ut,'rt,'dt>(&self, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			rng: &mut XorState, dstate: &mut DispState) -> Vec<Map<'rt>> {
		if self.load_world_map {
			// load ScenarioMap from disk
			let scenario_map = {
				let buf = read_file("config/world_scenario/world.map");
				let mut offset = 0;
				let mut scenario_map: Vec<ScenarioMap> = Vec::new();
				scenario_map.ld(&buf, &mut offset, temps.bldgs, temps.units, temps.resources, temps.doctrines);
				scenario_map
			};
 			
			{ // convert ScenarioMap to Map
				let mut map_root: Vec<Map> = Vec::with_capacity(scenario_map.len());
				
				for scenario_map in scenario_map.iter() {
					map_root.push(scenario_map.to_map());
				}
				
				map_root
			}
		}else{
			map_gen(MapSz {h: H_ROOT, w: W_ROOT, sz: H_ROOT*W_ROOT}, rng, dstate)
		}
	}
}

// format saved to disk
#[derive(Clone, Default)]
struct ScenarioMap {
	arability: u8,
	elevation: u8,
	show_snow: bool,
	map_type: u8
}

impl_saving! { ScenarioMap {arability, elevation, show_snow, map_type} }

impl ScenarioMap {
	fn to_map<'rt>(&self) -> Map<'rt> {
		Map {
			arability: self.arability as f32,
			show_snow: self.show_snow,
			elevation: self.elevation as f32,
			map_type: MapType::from(self.map_type),
			resource: None,
			resource_cont: None
		}
	}
}

