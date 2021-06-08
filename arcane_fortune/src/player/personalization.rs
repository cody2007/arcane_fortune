use crate::saving::*;
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::containers::*;
use crate::gcore::LogType;
use crate::ai::*;
use crate::nn;
use crate::config_load::*;
use crate::renderer::endwin;
use std::path::Path;
use std::fs;

// formerly `Owner`
#[derive(Clone, PartialEq)]
pub struct Personalization {
	pub color: i32, // color index (ncurses)
	pub nm: String, // of country, ex. China
	pub nm_adj: String, // adjective of country, ex. Chinese
	
	pub gender_female: bool,
	pub ruler_nm: PersonName,
	pub personality: AIPersonality,
	
	pub doctrine_advisor_nm: PersonName,
	pub crime_advisor_nm: PersonName,
	pub pacifism_advisor_nm: PersonName,
	pub health_advisor_nm: PersonName,
	pub unemployment_advisor_nm: PersonName,
	
	pub city_nm_theme: usize, // index into nms.cities[city_nm_theme]
	pub motto: String,
	pub founded: usize // turn
}

impl_saving!{Personalization {color, nm, nm_adj, gender_female,
	ruler_nm, personality, doctrine_advisor_nm, crime_advisor_nm,
	pacifism_advisor_nm, health_advisor_nm, unemployment_advisor_nm,
	city_nm_theme, motto, founded}}

#[derive(Clone, PartialEq)]
pub struct PersonName {
	pub first: String,
	pub last: String
}

impl PersonName {
	pub fn txt(&self) -> String {
		format!("{} {}", self.first, self.last)
	}
}

impl_saving!{PersonName {first, last}}

#[derive(Clone, PartialEq, Default)]
pub struct CityTheme {
	pub theme_nm: String,
	pub build_nms_in_order: bool, // ex. if true build city_nms[0] before city_nms[1]
	pub city_nms: Vec<String>,
}
impl_saving!{CityTheme {theme_nm, build_nms_in_order, city_nms}}

#[derive(Clone, PartialEq)]
pub struct Nms { // name list used to randomly name each city and unit (loaded from config files)
	pub cities: Vec<CityTheme>, // cities[i].city_nms[:] are all city names with theme i
	pub units: Vec<String>, // aka battalions
	pub brigades: Vec<String>,
	pub sectors: Vec<String>,
	pub noble_houses: Vec<String>,
	pub females: Vec<String>,
	pub males: Vec<String>,
}

impl_saving!{Nms {cities, units, brigades, sectors, noble_houses, females, males}}

fn load_city_themes(dir_nm: &str, build_nms_in_order: bool, city_themes: &mut Vec<CityTheme>) {
	// loop over files in the directory, each file contains a seprate them of city names
	if let Result::Ok(dir_entries) = fs::read_dir(Path::new(dir_nm)) {
		for entry in dir_entries {
			if let Result::Ok(e) = entry {
				let theme_path = e.path();
				
				city_themes.push(CityTheme {
					theme_nm: String::from(theme_path.as_os_str().to_str().unwrap()),
					build_nms_in_order,
					city_nms: return_names_list(read_file(
						theme_path.as_os_str().to_str().unwrap()))
				});
			}
		}
	} else {panicq!("failed to open {}", dir_nm);}
}

const WORLD_SCENARIO_CITY_NMS: &str = "config/world_scenario/city_names/";

impl Nms {
	pub fn return_city_theme_ind(&self, theme_nm: &str) -> usize {
		let theme_nm = format!("{}{}.txt", WORLD_SCENARIO_CITY_NMS, theme_nm);
		
		if let Some(ind) = self.cities.iter().position(|city_theme| city_theme.theme_nm == theme_nm) {
			ind
		// theme not found:
		}else{
			printlnq!("could not find {}", theme_nm);
			for city in self.cities.iter() {
				printlnq!("found {}", city.theme_nm);
			}
			panicq!("");
		}
	}
	
	pub fn load_default_config(&mut self) {
		self.cities = {
			let mut city_themes: Vec<CityTheme> = Vec::new();
			
			load_city_themes("config/names/cities/", false, &mut city_themes); // randomly generated names
			load_city_themes(WORLD_SCENARIO_CITY_NMS, true, &mut city_themes);
			
			city_themes
		};
		
		self.units = return_names_list(read_file("config/names/battalion_names.txt"));
		self.brigades = return_names_list(read_file("config/names/brigade_names.txt"));
		self.sectors = return_names_list(read_file("config/names/sector_names.txt"));
		self.noble_houses = return_names_list(read_file("config/names/noble_houses/english_names.txt"));
		self.females = return_names_list(read_file("config/names/females.txt")); // names of rulers/nobility
		self.males = return_names_list(read_file("config/names/males.txt")); // names of rulers/nobility
	}
	
	// choose name of city, making sure to not choose one that's already been used
	pub fn new_city_name(&self, personalization: &Personalization, gstate: &mut GameState) -> String {
		let city_theme = &self.cities[personalization.city_nm_theme];
		
		// randomize orderings or not
		let inds = if city_theme.build_nms_in_order {
			(0..city_theme.city_nms.len()).collect()
		}else{
			gstate.rng.inds(city_theme.city_nms.len())
		};
		
		let mut prefix = String::new();
		let mut suffix = String::new();
		
		loop {
			'nm_selection: for ind in inds.iter() {
				let proposed_nm = &city_theme.city_nms[*ind];
				for log in gstate.logs.iter() {
					if let LogType::CityFounded {city_nm, ..} = &log.val {
						if *city_nm == *proposed_nm {
							continue 'nm_selection;
						}
					}
				}
				
				// if this point has been reached, no matches have been found
				return if suffix == "shire" && proposed_nm.chars().last().unwrap() == 's' {
					format!("{}{}-{}", prefix, proposed_nm, suffix)
				}else{
					format!("{}{}{}", prefix, proposed_nm, suffix)
				};
			}
			
			// if this point has been reached, every name tried has already been used,
			// so we add a suffix or prefix and search again
			
			// create suffix
			if suffix.len() == 0 && gstate.rng.gen_f32b() < 0.5 {
				const SUFFIXES: &[&str] = &["shire", "borough", "ville", " Town", " City"];
				let ind = gstate.rng.usize_range(0, SUFFIXES.len());
				
				suffix = String::from(SUFFIXES[ind]);
			// append prefix
			}else{
				if prefix.len() == 0 && gstate.rng.gen_f32b() < 0.5 {
					prefix = String::from("Fort ");
				}else{
					prefix.push_str("New ");
				}
			}
		}
	}
}

impl Personalization {
	pub fn random(personality: AIPersonality, nm: String, ruler_nm: PersonName, gender_female: bool, color: i32, 
			txt_gen: &mut nn::TxtGenerator, gstate: &mut GameState, temps: &Templates) -> Self {
		Self {
			color,
			nm_adj: nm.clone(), // of country
			nm, // of country
			gender_female,
			ruler_nm,
			personality,
			doctrine_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			crime_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			pacifism_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			health_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			unemployment_advisor_nm: PersonName::new(&temps.nms, &mut gstate.rng).1,
			city_nm_theme: gstate.rng.usize_range(0, temps.nms.cities.len()),
			motto: txt_gen.gen_str(nn::TxtCategory::from(&personality)),
			founded: gstate.turn
		}
	}
}

