use std::collections::{VecDeque};
use std::num::ParseIntError;
use std::fmt;

use crate::gcore::hashing::{HashedMap, HashStruct64, HashedFogVars};
//use crate::config_load::get_usize_map_config;
use crate::zones::{FogVars, ZoneDemandSumMap, ZoneAgnosticLocallyLogged};
use crate::units::*;
use crate::buildings::*;
use crate::saving::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::tech::TechTemplate;
use crate::disp::{CCYAN, CSAND4, CGREEN, CSAND1};
use crate::nobility::Houses;
use crate::disp_lib::*;
use crate::gcore::{Bonuses, Brigade, Sector, XorState};
use crate::localization::Localization;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

pub const METERS_PER_TILE: f32 = 5.;

// puts "N" as last element
macro_rules! enum_From{($nm:ident { $($entry:ident),* }) => (
	#[derive(Copy, Clone, PartialEq, Debug)]
	pub enum $nm { $($entry),*, N }
	
	impl From<usize> for $nm {
		#[inline]
		fn from(val: usize) -> Self {
			let entry_list: &[$nm] = &[ $($nm::$entry),* ]; // does not include N

			debug_assertq!(val < entry_list.len());
			debug_assertq!(($nm::N as usize) == entry_list.len()); // last entry should be N
			entry_list[val]
		}
	}
	impl From<isize> for $nm {
		#[inline]
		fn from(val: isize) -> Self {
			let entry_list: &[$nm] = &[ $($nm::$entry),* ]; // does not include N

			debug_assertq!((val as usize) < entry_list.len());
			debug_assertq!(($nm::N as usize) == entry_list.len()); // last entry should be N
			entry_list[val as usize]
		}
	}
	
	impl From<f32> for $nm {
		#[inline]
		fn from(val: f32) -> Self {Self::from(val.round() as usize)}
	}
	
	impl From<u32> for $nm {
		#[inline]
		fn from(val: u32) -> Self {
			let entry_list: &[$nm] = &[ $($nm::$entry),* ]; // does not include N

			debug_assertq!((val as usize) < entry_list.len(), "val: {}, entry_list.len() {}", val, entry_list.len());
			debug_assertq!(($nm::N as usize) == entry_list.len()); // last entry should be N
			entry_list[val as usize]
		}
	}
	
	impl Default for $nm {
		fn default() -> Self {$nm::N}
	}
);}

enum_From!{ MapType {ShallowWater, DeepWater, Land, Mountain} }
enum_From!{ StructureType {Road, Wall, Gate} }

impl fmt::Display for StructureType {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			StructureType::Road => {"Road"}
			StructureType::Wall => {"Wall"}
			StructureType::Gate => {"Gate"}
			StructureType::N => {panic!("invalid structure type");}
		})
	}
}

enum_From!{ ZoneDemandType {Development, ProdAvail, ConsAvail,
	UnusedResidentCapac, GovBldg, Resource} }

impl ZoneDemandType {
	pub fn to_str(&self) -> &str {
		match self {
			ZoneDemandType::Development => "Development",
			ZoneDemandType::ProdAvail => "ProdAvail",
			ZoneDemandType::ConsAvail => "ConsAvail",
			ZoneDemandType::UnusedResidentCapac => "UnusedResidentCapac",
			ZoneDemandType::GovBldg => "GovBldg",
			ZoneDemandType::Resource => "Resource",
			ZoneDemandType::N => {panicq!("invalid demand type")}
		}
	}
}

enum_From!{ ZoneType {Agricultural, Residential, Business, Industrial} }

// required to parse movement_type from text file configurations [ find_opt_key_parse() ]
impl std::str::FromStr for ZoneType {
	type Err = ParseIntError;
	
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Result::Ok (match s {
			"Agricultural" => ZoneType::Agricultural,
			"Residential" => ZoneType::Residential,
			"Business" => ZoneType::Business,
			"Industrial" => ZoneType::Industrial,
			_ => {panicq!("Failed to parse \"{}\" into movement_type. Valid options are: Agricultural, Residential, Business, or Industrial", s)}
		})
	}
}

impl ZoneType {
	pub fn to_str(&self) -> &str {
		match self {
			ZoneType::Agricultural => "Agricultural",
			ZoneType::Residential => "Residential",
			ZoneType::Business => "Business",
			ZoneType::Industrial => "Industrial",
			ZoneType::N => {panicq!("invalid zone")}
		}
	}
	
	pub fn to_color(&self) -> CInt {
		match self {
			ZoneType::Agricultural => CSAND4,
			ZoneType::Residential => CGREEN,
			ZoneType::Business => CCYAN,
			ZoneType::Industrial => CSAND1,
			ZoneType::N => {panicq!("invalid zone")}
		}
	}
}

///////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq)]
pub struct AIPersonality {
	pub friendliness: f32, // negative friendliness is agression; range [-1:1]
	// ^ (1 - friendliness) -> proportionate to war declaration probability
	pub spirituality: f32, // negative spirituality is scientific; range [-1:1]
	// ^ (1 - spirituality)/2 -> probability of scientific buildings
}

impl_saving!{AIPersonality {friendliness, spirituality} }

impl AIPersonality {
	pub fn new(rng: &mut XorState) -> Self {
		Self {
			friendliness: 2.*rng.gen_f32b() - 1., // between -1:1
			spirituality: 2.*rng.gen_f32b() - 1.
		}
	}
}

#[derive(Clone, PartialEq)]
pub enum PlayerType {
	Human,
	AI(AIPersonality),
	Barbarian
}

impl PlayerType {
	pub fn is_ai(&self) -> bool {
		if let PlayerType::AI(_) = &self {true} else {false}
	}
	
	pub fn is_human(&self) -> bool {
		if let PlayerType::Human = &self {true} else {false}
	}
}

impl fmt::Display for PlayerType {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			PlayerType::Human => {"Human"}
			PlayerType::AI(_personality) => {"AI"}
			PlayerType::Barbarian => {"Barbarian"}
		})
	}
}

#[derive(Clone, PartialEq)]
pub struct PersonName {
	pub first: String,
	pub last: String
}

impl_saving!{PersonName {first, last}}

// it's not very easy to create immutable references to Owner because the save code adds owners
// as it's placing units. would need to pre-determine number of barbarians to add near each civ
// before calling new_game() in main.rs
#[derive(Clone, PartialEq)]
pub struct Owner {
	pub id: SmSvType,
	pub color: i32,
	pub nm: String, // of country
	
	pub gender_female: bool,
	pub ruler_nm: PersonName,
	
	pub doctrine_advisor_nm: PersonName,
	pub crime_advisor_nm: PersonName,
	pub pacifism_advisor_nm: PersonName,
	pub health_advisor_nm: PersonName,
	pub unemployment_advisor_nm: PersonName,
	
	pub city_nm_theme: usize, // index into nms.cities[city_nm_theme]
	pub motto: String,
	pub player_type: PlayerType, // fog of war only updated for human players
}

impl_saving!{Owner {id, color, nm, gender_female,
	ruler_nm, doctrine_advisor_nm, crime_advisor_nm,
	pacifism_advisor_nm, health_advisor_nm, unemployment_advisor_nm,
	city_nm_theme, motto, player_type}}

#[derive(Clone, PartialEq)]
pub struct Nms { // name list used to randomly name each city and unit (loaded from config files)
	pub cities: Vec<Vec<String>>, // cities[i][:] are all city names with theme i
	pub units: Vec<String>, // aka battalions
	pub brigades: Vec<String>,
	pub sectors: Vec<String>,
	pub noble_houses: Vec<String>,
	pub females: Vec<String>,
	pub males: Vec<String>,
}

impl_saving!{Nms {cities, units, brigades, sectors, noble_houses, females, males}}

#[derive(Clone, PartialEq)]
pub enum TechProg {Prog(SmSvType), Finished}

pub const LOG_TURNS: usize = 30;

// for each player, and zoom level:
#[derive(Clone, PartialEq)]
pub struct LandDiscov {
	pub map_sz_discov: MapSz, // scaled down (or same) from map_sz_frm
	pub map_sz: MapSz, // for the current zoom level -- MapEx coordinates
	
	pub frac_i: f32, // map_sz_discov.h / map_sz.h
	pub frac_j: f32, // map_sz_discov.w / map_sz.w
	// ^ both should be <= 1.
	
	// note: `coords` may be on a sparse grid and not reprsent direct coordinates (like zone info)
	//pub coords: HashedCoords // entry indicates coordinate has been discovered
	pub discovered: Vec<u8>
	// use .discover() and .discovered() to access
}

impl_saving! {LandDiscov {map_sz_discov, map_sz, frac_i, frac_j, discovered}}

impl LandDiscov {
	// convert map coord (ex. MapEx coord) to discovered map coord
	pub fn map_to_discov_coord(&self, mut coord: Coord) -> usize {
		debug_assertq!(coord.y >= 0 && coord.y < self.map_sz.h as isize &&
				coord.x >= 0 && coord.x < self.map_sz.w as isize,
				"{} {} {}", coord, self.map_sz_discov, self.map_sz);
		
		// convert to discov coord
		coord.y = (self.frac_i * coord.y as f32).round() as isize;
		coord.x = (self.frac_j * coord.x as f32).round() as isize;
		
		debug_assertq!(coord.y >= 0 && coord.y < self.map_sz_discov.h as isize &&
				coord.x >= 0 && coord.x < self.map_sz_discov.w as isize,
				"{} {} {}", coord, self.map_sz_discov, self.map_sz);
		
		(coord.y * self.map_sz_discov.w as isize + coord.x) as usize
	}
	
	fn discov_to_map_coord(&self, discov_coord: usize) -> u64 {
		let mut c = Coord::frm_ind(discov_coord as u64, self.map_sz_discov);
		
		c.y = (c.y as f32 / self.frac_i).round() as isize;
		c.x = (c.x as f32 / self.frac_j).round() as isize;
		
		if c.y >= self.map_sz.h as isize {c.y = self.map_sz.h as isize - 1;}
		if c.x >= self.map_sz.w as isize {c.x = self.map_sz.w as isize - 1;}
		
		c.to_ind(self.map_sz) as u64
	}
	
	// inputs: in coordinates of discovered map
	fn discov_coord_discovered(&self, coord: usize) -> bool {
		debug_assertq!(coord < (self.map_sz_discov.h*self.map_sz_discov.w));
		(self.discovered[coord / 8] & (1 << (coord % 8))) != 0
	}
	
	// inputs: in coordinates of map coord (ex. MapEx)
	pub fn map_coord_discovered(&self, coord: Coord) -> bool {
		#[cfg(feature="profile")]
		let _g = Guard::new("land discovered");
		
		self.discov_coord_discovered(self.map_to_discov_coord(coord))
	}
	
	// inputs: in coordinates of map coord (ex. MapEx)
	pub fn map_coord_ind_discovered(&self, coord: u64) -> bool {
		#[cfg(feature="profile")]
		let _g = Guard::new("land discovered");
		
		let coord = self.map_to_discov_coord(Coord::frm_ind(coord, self.map_sz));
		self.discov_coord_discovered(coord)
	}
	
	// inputs: in coordinates of map coord (ex. MapEx)
	pub fn map_coord_discover(&mut self, coord: Coord) {
		#[cfg(feature="profile")]
		let _g = Guard::new("discover land");
		
		let coord = self.map_to_discov_coord(coord);
		debug_assertq!(coord < (self.map_sz_discov.h*self.map_sz_discov.w));
		self.discovered[coord / 8] |= 1 << (coord % 8);
	}
}

pub struct LandDiscovIter<'ld> {
	land_discov: &'ld LandDiscov,
	pub discov_coord: usize
}

impl <'ld>LandDiscovIter<'ld> {
	pub fn from(land_discov: &'ld LandDiscov) -> Self {
		LandDiscovIter {
			land_discov,
			discov_coord: 0
		}
	}
}

// iterator of discovered map coords
impl Iterator for LandDiscovIter<'_> {
	type Item = u64;
	
	fn next(&mut self) -> Option<u64> {
		let discov_map_sz = self.land_discov.map_sz_discov.h * self.land_discov.map_sz_discov.w;
		debug_assertq!(discov_map_sz <= (self.land_discov.discovered.len()*8));
		
		loop {
			if self.discov_coord < discov_map_sz {
				if self.land_discov.discov_coord_discovered(self.discov_coord) {
					let map_coord = self.land_discov.discov_to_map_coord(self.discov_coord);
					self.discov_coord += 1;
					return Some(map_coord);
				}
				self.discov_coord += 1;
				
			// finished
			}else {return None;}
		}
	}
}

// stats for each player
#[derive(Clone, PartialEq)]
pub struct Stats<'bt,'ut,'rt,'dt> {
	pub alive: bool,
	pub population: usize, // add_resident()/rm_resident()
	pub gold: f32,
	pub employed: usize, // # employed
	
	pub doctrine_template: &'dt DoctrineTemplate, // current doctrine to give bonuses
	
	// zone agnostic stats dependent on doctrinality
		pub locally_logged: ZoneAgnosticLocallyLogged, // happiness, doctrinality, pacifism
		// ^ and fractional contributions of doctrinality, pacifism, health, unemployment, crime
		
		// updated when a bldg is added or removed
			pub crime: f32,
			pub health: f32,
		
	pub resources_avail: Vec<usize>, // # of resources available, indexed by resource id
	
	/*
	// ^ set when zone is created and city hall is close enough (zone_ex
	//	todo: recheck if new resources are available after city hall construction
	//
	*/	
	pub resources_discov_coords: Vec<Vec<u64>>, // indexing: [resource_id][exemplar]
	
	pub bonuses: Bonuses,
	
	/////////// logs:
	pub alive_log: Vec<bool>,
	pub population_log: Vec<usize>,
	pub gold_log: Vec<f32>,
	pub net_income_log: Vec<f32>,
	pub unemployed_log: Vec<f32>,
	pub defense_power_log: Vec<usize>,
	pub offense_power_log: Vec<usize>,
	pub zone_demand_log: Vec<Vec<f32>>, // indexed as [time][zone_type]
	
	pub happiness_log: Vec<f32>,
	pub crime_log: Vec<f32>,
	pub pacifism_log: Vec<f32>,
	pub health_log: Vec<f32>,
	pub doctrinality_log: Vec<Vec<f32>>, // indexed as [time][doctrine_id]
	
	pub research_per_turn_log: Vec<u32>,
	pub research_completed_log: Vec<usize>,
	
	pub mpd_log: Vec<f32>, // milliseconds per day code execution time
	/////////////
	
	pub zone_demand_sum_map: Box<[ZoneDemandSumMap]>,
	// ^ indexed by zone_type, summed across land
	
	pub tax_income: f32,
	pub unit_expenses: f32,
	pub bldg_expenses: f32,
	
	pub techs_progress: Vec<TechProg>, // entry i is for tech_templates[i] tech progress
	pub techs_scheduled: Vec<SmSvType>, // index into tech_templates
	// ^ research is scheduled by starting at the end of the vector
	// and working back to the first entry
	
	pub research_per_turn: SmSvType,
	
	pub brigades: Vec<Brigade<'bt,'ut,'rt,'dt>>,
	pub sectors: Vec<Sector>,
	
	pub houses: Houses,
	
	// note: `land_discov` internals may be on a sparse grid and not reprsent direct coordinates (like zone info)
	pub land_discov: Vec<LandDiscov>, // use .discovered() and .discover() to access
	pub fog: Vec<HashedFogVars<'bt,'ut,'rt,'dt>> // (indexed by [zoom level][direct (non-grid) coordinate])
		// ^ if a coordinate (or its direct, non-grid analog) is in `land_discov` but not `fog`, this means a unit/bldg is present
		//	if `fog` is present, it means a unit & bldg is absent.
		//	it should never occur that a `fog` entry is present but `land_discov` is absent
}

impl_saving!{Stats<'bt,'ut,'rt,'dt> {alive, population, gold, employed,
		    doctrine_template,
		    locally_logged,
		    crime, health,
		    resources_avail, resources_discov_coords,
		    
		    bonuses,
		    
		    alive_log, population_log, gold_log, net_income_log, unemployed_log, defense_power_log,
		    offense_power_log, zone_demand_log,
		    
		    happiness_log, 
		    crime_log, pacifism_log, health_log, doctrinality_log,
		    
		    research_per_turn_log, research_completed_log,
		    
		    mpd_log,
		    
		    zone_demand_sum_map,
		    
		    tax_income, unit_expenses, bldg_expenses,
		    
		    techs_progress, techs_scheduled, research_per_turn,
		    
		    brigades, sectors,
		    
		    houses,
		    
		    land_discov,
		    fog
}}

impl <'bt,'ut,'rt,'dt>Stats<'bt,'ut,'rt,'dt> {
	pub fn default_init(bonuses: &Bonuses, tech_templates: &Vec<TechTemplate>, resource_templates: &Vec<ResourceTemplate>,
			doctrine_templates: &'dt Vec<DoctrineTemplate>, map_data: &MapData) -> Self {
		
		let (fog, land_discov) = {
			let max_zoom_ind = map_data.max_zoom_ind();
			let mut fog = Vec::with_capacity(max_zoom_ind+1);
			let mut land_discov = Vec::with_capacity(max_zoom_ind+1);
			assertq!(map_data.map_szs.len() == (max_zoom_ind+1));
			
			let max_land_discov_sz = MapSz {h: 7760/2, w: 31719/2, sz: 0};
			
			for map_sz in map_data.map_szs.iter() {
				// fog (same coordinates as the MapEx storage `exs`)
				let s: BuildHasherDefault<HashStruct64> = Default::default();
				fog.push(HashMap::with_hasher(s));
				
				// land_discov
				let map_sz_discov = if map_sz.w <= max_land_discov_sz.w {
					*map_sz
				}else{
					max_land_discov_sz
				};
				
				land_discov.push(LandDiscov {
						map_sz_discov,
						map_sz: *map_sz,
						frac_i: (map_sz_discov.h as f32 -1.) / map_sz.h as f32,
						frac_j: (map_sz_discov.w as f32 -1.) / map_sz.w as f32,
						discovered: vec![0; 
							(((map_sz_discov.h*map_sz_discov.w) as f32) / 8.).ceil() as usize]
				});
			}
			(fog, land_discov)
		};
		
		Stats {alive: true, population: 0,
			gold: 300000.,
			employed: 0,
			
			doctrine_template: &doctrine_templates[0],
			locally_logged: ZoneAgnosticLocallyLogged::default_init(doctrine_templates),
			crime: 0.,
			health: 0.,
			
			resources_avail: vec!{0; resource_templates.len()},
			resources_discov_coords: vec!{Vec::new(); resource_templates.len()},
			
			bonuses: bonuses.clone(),
			
			// log across time
			alive_log: Vec::new(),
			population_log: Vec::new(),
			unemployed_log: Vec::new(),
			gold_log: Vec::new(),
			net_income_log: Vec::new(),
			defense_power_log: Vec::new(),
			offense_power_log: Vec::new(),
			zone_demand_log: Vec::new(),
			
			happiness_log: Vec::new(),
			crime_log: Vec::new(),
			doctrinality_log: Vec::new(),
			pacifism_log: Vec::new(),
			health_log: Vec::new(),
			
			research_per_turn_log: Vec::new(),
			research_completed_log: Vec::new(),
			
			mpd_log: Vec::new(),
			
			zone_demand_sum_map: vec!{Default::default(); ZoneType::N as usize}.into_boxed_slice(),
			
			tax_income: 0., unit_expenses: 0., bldg_expenses: 0.,
			
			techs_progress: vec!{TechProg::Prog(0); tech_templates.len()}, // entry for each tech template *not a stack*
			techs_scheduled: Vec::with_capacity(tech_templates.len()), // stack. last entry researched first
			research_per_turn: 0,
			
			brigades: Vec::new(),
			sectors: Vec::new(),
			
			houses: Default::default(),
			
			land_discov, fog
		}
	}
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MapSz {
	pub h: usize,
	pub w: usize,
	pub sz: usize,
}

impl fmt::Display for MapSz {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "({}, {}, sz {})", self.h, self.w, self.sz)
	}
}

impl_saving!{MapSz {h,w,sz}}

/////////////////////////////////////////////////////
// zoned bldg distance to city hall:
#[derive(Clone, PartialEq, Copy)]
pub enum Dist {
	NotInit,
	NotPossible {turn_computed: usize},
	Is {dist: usize, bldg_ind: usize},
	ForceRecompute {dist: usize, bldg_ind: usize},
		// ^ ForceRecompute is set in uninit_city_hall_dists()
		//   and later read when the distance is recomputed by:
		//   		set_city_hall_dist() [which needs to update
		//					    the BldgArg::CityHall.population value]
		//	also: 
		//		set_owner, replace_map_bldg_ind(), add/rm_resident()
		//	also need to updatethe cityhall BldgArg enum
}

#[derive(Clone, PartialEq)]
pub struct MapEx<'bt,'ut,'rt,'dt> {
	pub actual: FogVars<'bt,'ut,'rt,'dt>,
	
	pub unit_inds: Option<Vec<usize>>,
	pub bldg_ind: Option<usize>
}

impl_saving!{ MapEx<'bt,'ut,'rt,'dt>{ actual, unit_inds, bldg_ind}  }

use super::ARABILITY_STEP;
use crate::disp::*;

enum_From!{ ArabilityType {
	Ocean, ShallowWater, Mountain,
	Tundra, Desert,
	PineForest, Savanna,
	Steppe, Heath,
	Prarie, BroadleafForest, Meadow, Wetland, TropicalBroadleaf, MixedForest}}

impl ArabilityType {
	pub fn to_str(&self, l: &Localization) -> String {
		match self {
			ArabilityType::Ocean => {l.Ocean.clone()}
			ArabilityType::ShallowWater => {l.Shallow_water.clone()}
			ArabilityType::Mountain => {l.Mountain.clone()}
			
			ArabilityType::Tundra => {l.Tundra.clone()}
			ArabilityType::Desert => {l.Desert.clone()}
			
			ArabilityType::PineForest => {l.Pine_forest.clone()}
			ArabilityType::Savanna => {l.Savanna.clone()}
			
			ArabilityType::Steppe => {l.Steppe.clone()}
			ArabilityType::Prarie => {l.Prarie.clone()}
			
			ArabilityType::Heath => {l.Heath.clone()}
			ArabilityType::MixedForest => {l.Mixed_forest.clone()}
			ArabilityType::BroadleafForest => {l.Broadleaf_forest.clone()}
			ArabilityType::TropicalBroadleaf => {l.Tropical_broadleaf.clone()}
			ArabilityType::Wetland => {l.Wetland.clone()}
			ArabilityType::Meadow => {l.Meadow.clone()}
			
			ArabilityType::N => {panicq!("could not convert unknown arability type to string")}
		}
	}
	
	pub fn to_color(&self, sel: bool) -> CInd {
		if !sel {
			match self {
				ArabilityType::Tundra => CSNOW4,
				ArabilityType::Desert => CSAND4,
				
				ArabilityType::PineForest => CSNOW3,
				ArabilityType::Savanna => CSAND3,
				
				ArabilityType::Steppe => CSNOW2,
				ArabilityType::Prarie => CSAND2,
				
				ArabilityType::Heath => CSAND1,
				ArabilityType::MixedForest => CGREEN1,
				ArabilityType::BroadleafForest => CGREEN2,
				ArabilityType::TropicalBroadleaf => CGREEN3,
				ArabilityType::Wetland => CGREEN4,
				ArabilityType::Meadow => CGREEN5,
				
				ArabilityType::Ocean | ArabilityType::ShallowWater | ArabilityType::Mountain | 
				ArabilityType::N => {panicq!("could not convert unknown arability type to color")}
			}
		}else{
			match self {
				ArabilityType::Tundra => CREDSNOW4,
				ArabilityType::Desert => CREDSAND4,
				
				ArabilityType::PineForest => CREDSNOW3,
				ArabilityType::Savanna => CREDSAND3,
				
				ArabilityType::Steppe => CREDSNOW2,
				ArabilityType::Prarie => CREDSAND2,
				
				ArabilityType::Heath => CREDSAND1,
				ArabilityType::MixedForest => CREDGREEN1,
				ArabilityType::BroadleafForest => CREDGREEN2,
				ArabilityType::TropicalBroadleaf => CREDGREEN3,
				ArabilityType::Wetland => CREDGREEN4,
				ArabilityType::Meadow => CREDGREEN5,
				
				ArabilityType::Ocean | ArabilityType::ShallowWater | ArabilityType::Mountain | 
				ArabilityType::N => {panicq!("could not convert unknown arability type to color")}
			}
		}
	}
	
	pub fn frm_arability(arability: f32, map_type: MapType, show_snow: bool) -> ArabilityType {
		match map_type {
			MapType::DeepWater => ArabilityType::Ocean,
			MapType::ShallowWater => ArabilityType::ShallowWater,
			MapType::Mountain => ArabilityType::Mountain,
			MapType::Land => {
				let a = arability as isize;
				if a <= ARABILITY_STEP {
					if show_snow {ArabilityType::Tundra} else {ArabilityType::Desert}}
				else if a <= (2*ARABILITY_STEP) {
					if show_snow {ArabilityType::PineForest} else {ArabilityType::Savanna}}
				else if a <= (3*ARABILITY_STEP) {
					if show_snow {ArabilityType::Steppe} else {ArabilityType::Prarie}}
				
				else if a <= (4*ARABILITY_STEP) {ArabilityType::Heath}
				else if a <= (5*ARABILITY_STEP) {ArabilityType::MixedForest}
				else if a <= (6*ARABILITY_STEP) {ArabilityType::BroadleafForest}
				else if a <= (7*ARABILITY_STEP) {ArabilityType::TropicalBroadleaf}
				else if a <= (8*ARABILITY_STEP) {ArabilityType::Wetland}
				else                            {ArabilityType::Meadow}
			}
			MapType::N => {panicq!("unknown map type input")}
		}
	}
	
	pub fn to_config_str(&self) -> String {
		match self {
			ArabilityType::Ocean => "ocean_prob",
			ArabilityType::ShallowWater => "shallow_water_prob",
			ArabilityType::Mountain => "mountain_prob",
			
			ArabilityType::Tundra => "tundra_prob",
			ArabilityType::Desert => "desert_prob",
			
			ArabilityType::PineForest => "pine_forest_prob",
			ArabilityType::Savanna => "savanna_prob",
			
			ArabilityType::Steppe => "steppe_prob",
			ArabilityType::Prarie => "prarie_prob",
			
			ArabilityType::Heath => "heath_prob",
			ArabilityType::MixedForest => "mixed_forest_prob",
			ArabilityType::BroadleafForest => "broadleaf_forest_prob",
			ArabilityType::TropicalBroadleaf => "tropical_broadleaf_prob",
			ArabilityType::Wetland => "wetland_prob",
			ArabilityType::Meadow => "meadow_prob",
			
			ArabilityType::N => {panicq!("Unknown arability type")}
		}.to_string()
	}
}

#[derive(Copy, Clone, PartialEq)]
pub struct ResourceCont {
	pub offset_i: u8,
	pub offset_j: u8
}

impl_saving!{ ResourceCont{ offset_i, offset_j } }

////////////////////////////////////////////////////////////////////
#[derive(Copy, Clone, PartialEq)]
pub struct Map<'rt> {
	pub arability: f32,
	pub show_snow: bool,
	pub elevation: f32,
	pub map_type: MapType,
	
	// use .get_resource() and .resource_char() to access:
	pub resource: Option<&'rt ResourceTemplate>, // `resource` is used for the far upper left extent of the resource
	pub resource_cont: Option<ResourceCont>, // `resource_cont` is set for everything within the resource's height and width, largely for display purposes
}

impl_saving!{ Map<'rt>{ arability, show_snow, elevation, map_type, resource, resource_cont}  }

// for management of map buffers (VecDeque)
#[derive(Copy, Clone, PartialEq)]
pub struct DequeZoomIn {
	pub zoom_in_ind: usize, // index into map_data.zoom_in
	pub coord: u64
}

pub const N_STD_EXPLICITLY_STORED_ZOOM_LVLS: usize = 4; // standard downsampled zoom-outs (dimensions prop. to generated map). does NOT include submap. must be sz of ZOOM_OUT_LVLS
pub const STD_EXPLICITLY_STORED_ZOOM_LVL_SPACE: &'static [usize] = &[32,17,4,1]; //&[45,15,6,3,1];
// ^ spacing between each tile of ZOOM_IND_ROOT map (the map generated originally at game start)

pub const ZOOM_IND_SUBMAP: usize = 0; // shown at bottom of screen, not necessarily proportionate to generated map dimensions
pub const ZOOM_IND_EXPANDED_SUBMAP: usize = 2; // proportionate to map size
pub const ZOOM_IND_ROOT: usize = N_STD_EXPLICITLY_STORED_ZOOM_LVLS; // full map -- generated by slower map_gen()
pub const N_EXPLICITLY_STORED_ZOOM_LVLS: usize = N_STD_EXPLICITLY_STORED_ZOOM_LVLS + 1; // also includes submap and map root (ZOOM_IND_ROOT)

// returns how many land plots at ZOOM_IND are averaged together for each land plot at zoom lvl `zoom_ind`:
// (fn limited to explicitly stored maps (the map root and maps zoomed out further from it)
pub struct MapData<'rt> {	
	// only zoom_out[ZOOM_IND_ROOT][:] needs to be saved to disk (this is the map root everything is generated from)
	pub zoom_out: Vec<Vec<Map<'rt>>>, // entries for zoom_out[zoom_lvl][coord]
	// (^ everything stored explicitly in memory. we can store all values in zoomed_out maps --
	//    we can't store the map at full zoom due to memory limitations)
	
	// not saved
	pub deque_zoom_in: VecDeque<DequeZoomIn>,
	pub zoom_in: Vec<HashedMap<'rt>>, // hashmap at each zoom level	
	
	pub map_szs: Vec<MapSz>,
	pub max_zoom_in_buffer_sz: usize,
	
	pub rt: &'rt Vec<ResourceTemplate>
}

//////////// !!!! note: custom MapData save procedure implemented in saving/mod.rs
////////////            only zoom_out, map_szs, max_zoom_in_buffer_sz are saved

use std::hash::{BuildHasherDefault};
use std::collections::HashMap;
use super::zoom_in::upsample_dim;

impl <'rt>MapData<'rt> {
	// zoom_in_depth is the number of upsamples from map_root
	pub fn default(map_root: Vec<Map<'rt>>, mut h: usize, mut w: usize, zoom_in_depth: usize, max_zoom_in_buffer_sz: usize,
			resource_templates: &'rt Vec<ResourceTemplate>) -> MapData<'rt> {
		
		///////// set map_szs
		let map_szs = {
			let mut map_szs = Vec::with_capacity(N_EXPLICITLY_STORED_ZOOM_LVLS + zoom_in_depth);
			
			// zoomed out maps (stored explicitly). to be set later in zoom_out::compute_zoom_outs()
			for _zoom_ind in 0..N_EXPLICITLY_STORED_ZOOM_LVLS {
				map_szs.push(MapSz {h: 0, w: 0, sz: 0});
			}
			
			// root of the map (everything computed from this)
			map_szs[ZOOM_IND_ROOT] = MapSz {h, w, sz: h*w};
			
			// zoomed in maps (upsampled) stored at map_szs[ZOOM_IND_ROOT+1:]
			for _depth in 1..=zoom_in_depth {
				let (h_dim, h_round_off) = upsample_dim(h);
				let (w_dim, w_round_off) = upsample_dim(w);
				
				// remove round off at bottom and right of map
				// this round off is not used for map generation at more zoomed-in levels
				{
					let map_sz = map_szs.last_mut().unwrap();
					map_sz.h -= h_round_off;
					map_sz.w -= w_round_off;
					//endwin();
					//println!("depth {} mapszh {} mapszw {} hroundoff {} wroundoff {}", depth, map_sz.h, map_sz.w, h_round_off, w_round_off);
				}
				
				map_szs.push(MapSz{h: h_dim, w: w_dim, sz: h_dim*w_dim});
				
				h = h_dim;
				w = w_dim;
			}
			map_szs
		};
		//////////
		
		/////// set zoom_out
		let mut zoom_out = vec!{Vec::new(); N_EXPLICITLY_STORED_ZOOM_LVLS};
		zoom_out[ZOOM_IND_ROOT] = map_root;
		
		/////// set zoom_in
		let mut zoom_in = Vec::with_capacity(zoom_in_depth);
		
		for _zoom_ind in 0..zoom_in_depth {
			let s: BuildHasherDefault<HashStruct64> = Default::default();
			zoom_in.push(HashMap::with_hasher(s));
		}
		///////

		MapData {
			zoom_out,
			deque_zoom_in: VecDeque::new(),
			zoom_in,
			
			map_szs,
			max_zoom_in_buffer_sz,
			rt: resource_templates
		}
	}
	
	#[inline]
	pub fn zoom_spacing_explicitly_stored(zoom_ind: usize) -> usize {
		debug_assertq!(zoom_ind != ZOOM_IND_SUBMAP);
		debug_assertq!(STD_EXPLICITLY_STORED_ZOOM_LVL_SPACE.len() == N_STD_EXPLICITLY_STORED_ZOOM_LVLS);

		STD_EXPLICITLY_STORED_ZOOM_LVL_SPACE[zoom_ind - 1]
	}
	
	// returns how many land plots at ZOOM_IND are averaged together for each land plot at zoom lvl `zoom_ind`:
	#[inline]
	pub fn zoom_spacing(&self, zoom_ind: usize) -> f32 {
		if zoom_ind < N_EXPLICITLY_STORED_ZOOM_LVLS {
			MapData::zoom_spacing_explicitly_stored(zoom_ind) as f32
		}else{
			(self.map_szs[ZOOM_IND_ROOT].w as f32) / (self.map_szs[zoom_ind].w as f32)
		}
	}
	
	#[inline]
	pub fn max_zoom_ind(&self) -> usize {self.map_szs.len() - 1}
	// ^ max allowed value for self.zoom_ind
}

// set value from sum, convert sum to value
macro_rules! convert_to_T {
	($val_out: expr, $val_sum: expr, bool) => {$val_out = $val_sum != 0.;};
	($val_out: expr, $val_sum: expr, f32) => {$val_out = $val_sum;};
	($val_out: expr, $val_sum: expr, u8) => {$val_out = $val_sum as u8;};
	($val_out: expr, $val_sum: expr, i8) => {$val_out = $val_sum as i8;};
	($val_out: expr, $val_sum: expr, MapType) => {$val_out = MapType::from($val_sum as usize);};
	($val_out: expr, $val_sum: expr, BldgType) => {$val_out = BldgType::from($val_sum as usize);};
}

