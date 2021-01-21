use std::collections::HashMap;
use std::hash::{BuildHasherDefault};

use crate::saving::*;
use crate::renderer::endwin;
use crate::map::{TechProg, LandDiscov, MapSz, MapData, ZoneType};
use crate::zones::{ZoneAgnosticLocallyLogged, ZoneDemandSumMap};
use crate::gcore::*;
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;

pub const LOG_TURNS: usize = 30;

#[derive(Clone, PartialEq)]
pub struct Stats<'bt,'ut,'rt,'dt> {
	pub id: SmSvType,
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
	
	pub tax_income: f32, // use pstats.tax_income() if the value after fiefdom taxes is desired
	pub unit_expenses: f32,
	pub bldg_expenses: f32,
	
	pub techs_progress: Vec<TechProg>, // entry i is for tech_templates[i] tech progress
	pub techs_scheduled: Vec<SmSvType>, // index into tech_templates
	// ^ research is scheduled by starting at the end of the vector
	// and working back to the first entry
	
	pub research_per_turn: SmSvType,
	
	pub brigades: Vec<Brigade<'bt,'ut,'rt,'dt>>,
	pub sectors: Vec<Sector>,
	
	// note: `land_discov` internals may be on a sparse grid and not reprsent direct coordinates (like zone info)
	pub land_discov: Vec<LandDiscov>, // use .discovered() and .discover() to access
	pub fog: Vec<HashedFogVars<'bt,'ut,'rt,'dt>> // (indexed by [zoom level][direct (non-grid) coordinate])
		// ^ if a coordinate (or its direct, non-grid analog) is in `land_discov` but not `fog`, this means a unit/bldg is present
		//	if `fog` is present, it means a unit & bldg is absent.
		//	it should never occur that a `fog` entry is present but `land_discov` is absent
}

impl_saving!{Stats<'bt,'ut,'rt,'dt> {id, alive, population, gold, employed,
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
	    
	    land_discov,
	    fog
}}

use crate::containers::Templates;
impl <'bt,'ut,'rt,'dt>Stats<'bt,'ut,'rt,'dt> {
	pub fn default_init(id: SmSvType, bonuses: &Bonuses,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &MapData,
			n_log_entries: usize) -> Self {
		
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
							(((map_sz_discov.h*map_sz_discov.w) as f32) / 8.).ceil() as usize],
						n_discovered: 0
				});
			}
			(fog, land_discov)
		};
		
		Stats {id, alive: true, population: 0,
			gold: 300000.,
			employed: 0,
			
			doctrine_template: &temps.doctrines[0],
			locally_logged: ZoneAgnosticLocallyLogged::default_init(temps.doctrines),
			crime: 0.,
			health: 0.,
			
			resources_avail: vec!{0; temps.resources.len()},
			resources_discov_coords: vec!{Vec::new(); temps.resources.len()},
			
			bonuses: bonuses.clone(),
			
			// log across time
			alive_log: vec![false; n_log_entries],
			population_log: vec![0; n_log_entries],
			unemployed_log: vec![0.; n_log_entries],
			gold_log: vec![0.; n_log_entries],
			net_income_log: vec![0.; n_log_entries],
			defense_power_log: vec![0; n_log_entries],
			offense_power_log: vec![0; n_log_entries],
			zone_demand_log: vec![vec![0.; ZoneType::N as usize]; n_log_entries],
			
			happiness_log: vec![0.; n_log_entries],
			crime_log: vec![0.; n_log_entries],
			doctrinality_log: vec![vec![0.; temps.doctrines.len()]; n_log_entries],
			pacifism_log: vec![0.; n_log_entries],
			health_log: vec![0.; n_log_entries],
			
			research_per_turn_log: vec![0; n_log_entries],
			research_completed_log: vec![0; n_log_entries],
			
			mpd_log: Vec::new(),
			
			zone_demand_sum_map: vec!{Default::default(); ZoneType::N as usize}.into_boxed_slice(),
			
			tax_income: 0., unit_expenses: 0., bldg_expenses: 0.,
			
			techs_progress: vec!{TechProg::Prog(0); temps.techs.len()}, // entry for each tech template *not a stack*
			techs_scheduled: Vec::with_capacity(temps.techs.len()), // stack. last entry researched first
			research_per_turn: 0,
			
			brigades: Vec::new(),
			sectors: Vec::new(),
			
			land_discov, fog
		}
	}
	
	pub fn military_power(&self) -> Option<usize> {
		if let Some(&offense_power) = self.offense_power_log.last() {
		if let Some(&defense_power) = self.defense_power_log.last() {
			return Some(offense_power + defense_power);
		}}
		None
	}
}

