use crate::doctrine::DoctrineTemplate;
use crate::disp::{DispChars, IfaceSettings, Coord};
use crate::map::*;
use crate::units::*;
use crate::tech::TechTemplate;
use crate::resources::ResourceTemplate;
use crate::config_load::*;
use crate::saving::*;
use crate::gcore::hashing::*;
use crate::gcore::*;
use crate::renderer::endwin;
use crate::localization::Localization;
use crate::movement::manhattan_dist;
use crate::player::*;
use crate::containers::*;
use crate::zones::*;

mod vars; pub use vars::*;
mod events; pub use events::*;

const BLDG_CONFIG_FILE: &str = "config/buildings.txt";

pub fn init_bldg_templates<'ut,'rt,'dt>(tech_templates: &Vec<TechTemplate>, unit_templates: &'ut Vec<UnitTemplate<'rt>>,
		doctrine_templates: &'dt Vec<DoctrineTemplate>,
		disp_chars: &DispChars, l: &Localization) -> Vec<BldgTemplate<'ut,'rt,'dt>> {
	
	let key_sets = config_parse(read_file(BLDG_CONFIG_FILE));
	
	let mut bldg_templates: Vec<BldgTemplate> = Vec::new();
	
	// first set of keys is the BldgConfig variables
	for (id, keys) in key_sets.iter().skip(1).enumerate() {
		let eng_nm = find_req_key("nm", keys);
		let nm = if let Some(nm) = l.bldg_nms.iter().find(|nms| nms[0] == eng_nm) {
			nm
		}else{panicq!("could not find translations of building `{}`. the localization file may need to be updated", eng_nm);};
		
		let units_producable = find_opt_key_units_producable("units_producable", keys, unit_templates);
		let units_producable_txt = find_opt_key_vec_string("units_producable_txt", keys);
		let unit_production_rate = find_key_parse("unit_production_rate", 0, keys);
		
		let bldg_type = if let Some(ztype) = find_opt_key_parse("taxable_zone", keys) {
			BldgType::Taxable(
				Zone {
					ztype,
					density: ZoneDensity::frm_str(&find_req_key("zone_density", keys))
				}
			)
		}else{
			BldgType::Gov(load_zone_bonuses(keys))
		};
		
		let barbarian_only = if let Some(_) = find_key("barbarian_only", keys) {true} else {false};
		let not_human_buildable = if let Some(_) = find_key("not_human_buildable", keys) {true} else {false};
		
		// check that all unit production variables are set or not set
		if !((units_producable.is_none() && units_producable_txt.is_none() && unit_production_rate == 0) ||
			(!units_producable.is_none() && !units_producable_txt.is_none() && unit_production_rate != 0)) {
			 	panicq!("Building entry \"{}\": `units_producable`, `units_producable_txt`, and `unit_production_rate` must all be set or not set at all.", nm[0]);
			 	//println!("{} {} {}", units_producable.is_none(), units_producable_txt.is_none(), unit_production_rate == 0);
		}
		
		bldg_templates.push( BldgTemplate {
				id: id as SmSvType,
				
				tech_req: find_tech_req(&nm[0], keys, tech_templates),
				doctrine_req: find_doctrine_req(&nm[0], keys, doctrine_templates),
				nm: nm.clone(),
				
				menu_txt: find_key("menu_txt", keys),
				
				research_prod: find_key_parse("research_prod", 0, keys),
				
				sz: find_req_key_print_sz("print_str", keys),
				print_str: find_req_key_print_str("print_str", keys, disp_chars),
				plot_zoomed: find_req_key_parse("plot_zoomed", keys),
				bldg_type,
				
				units_producable,
				units_producable_txt,
				unit_production_rate,
				water_source: find_key_parse("water_source", false, keys),
				
				construction_req: find_key_parse("construction_req", 0., keys),
				upkeep: find_req_key_parse("upkeep", keys),
				
				resident_max: find_key_parse("resident_max", 0, keys),
				cons_max: find_key_parse("consumption_max", 0, keys),
				prod_max: find_key_parse("production_max", 0, keys),
				
				crime_bonus: find_key_parse("crime_bonus", 0., keys), // + means more crime
				happiness_bonus: find_key_parse("happiness_bonus", 0., keys),
				doctrinality_bonus: find_key_parse("doctrinality_bonus", 0., keys),
				pacifism_bonus: find_key_parse("pacifism_bonus", 0., keys),
				health_bonus: find_key_parse("health_bonus", 0., keys),
				job_search_bonus: find_key_parse("job_search_bonus", 0., keys),
				
				barbarian_only, not_human_buildable
			} );
	}
	
	// check ordering is correct
	#[cfg(any(feature="opt_debug", debug_assertions))]
	for (i, bt) in bldg_templates.iter().enumerate() {
		debug_assertq!(bt.id == i as SmSvType);
	}
	
	bldg_templates
}

impl BldgConfig {
	pub fn from_config_file() -> Self {
		if let Some(keys) = config_parse(read_file(BLDG_CONFIG_FILE)).first() { // first key set are the BldgConfig variables
			Self {
				fire_damage_rate: find_req_key_parse("fire_damage_rate", &keys),
				fire_repair_rate: find_req_key_parse("fire_repair_rate", &keys),
				max_bldg_damage: find_req_key_parse("max_bldg_damage", &keys),
				
				job_search_bonus_dist: find_req_key_parse("job_search_bonus_dist", &keys),
				
				birth_celebration_bonus: find_req_key_parse("birth_celebration_bonus", &keys),
				marriage_celebration_bonus: find_req_key_parse("marriage_celebration_bonus", &keys),
				funeral_bonus: find_req_key_parse("funeral_bonus", &keys),
				
				cost_to_zone_low_density: find_req_key_parse("cost_to_zone_low_density", &keys),
				cost_to_zone_medium_density: find_req_key_parse("cost_to_zone_medium_density", &keys),
				cost_to_zone_high_density: find_req_key_parse("cost_to_zone_high_density", &keys),
			}
		}else{panicq!("could not find building configuration entries in {}", BLDG_CONFIG_FILE);}
	}
}

// should be run after init_unit_templates(): 
// req. no units, structures, and bldgs be present
pub fn land_clear_ign_zone_ign_owner(coord: u64, mfc: &Map, exf: &HashedMapEx) -> bool {
	if mfc.map_type != MapType::Land {return false;}
	
	if let Some(ex) = exf.get(&coord) {
		if !ex.unit_inds.is_none() || !ex.actual.structure.is_none() || !ex.bldg_ind.is_none() {return false;}
		debug_assertq!(!ex.actual.owner_id.is_none());
	}
	true
}

pub struct MatchZoneOwner {
	pub zone_type: ZoneType,
	pub owner_id: SmSvType
}

// req owner, and zone type to match (if set to none, req. that there be no zone)
// req no units, structures, and bldgs be present
pub fn land_clear(coord: u64, match_zone_owner: Option<MatchZoneOwner>, mfc: &Map, exf: &HashedMapEx) -> bool {
	if mfc.map_type != MapType::Land {return false;}
	
	// must match zone owner
	if let Some(zo) = match_zone_owner {
		if let Some(ex) = exf.get(&coord) {
			if !ex.unit_inds.is_none() || !ex.actual.structure.is_none() || !ex.bldg_ind.is_none() {return false;}
			if ex.actual.owner_id != Some(zo.owner_id) {return false;}
			
			// zone type must match:
			if let Some(zone_type) = ex.actual.ret_zone_type() {
				return zo.zone_type == zone_type;
			}
		} // ex present
		
		return false;
	
	// must not be any zone or owner
	}else{
		if let Some(ex) = exf.get(&coord) {
			if !ex.unit_inds.is_none() || !ex.actual.structure.is_none() || !ex.bldg_ind.is_none() {return false;}
			if ex.actual.owner_id != None {return false;}
		
			// no zone_match set, so plot must not have a zone
			return ex.actual.ret_zone_type().is_none();
		}
		return true;
	}
}

// req. no zone, structures aside from roads, no bldgs
pub fn land_clear_ign_units_roads(coord: u64, owner_id: SmSvType, mfc: &Map, exf: &HashedMapEx) -> bool{
	if mfc.map_type != MapType::Land {return false;}
	
	if let Some(ex) = exf.get(&coord) {
		if ex.actual.owner_id != None && ex.actual.owner_id != Some(owner_id) {return false;}
		if !ex.bldg_ind.is_none() {return false;}
		
		if let Some(structure) = ex.actual.structure {
			if structure.structure_type != StructureType::Road {return false;}
		}
	}
	true
}

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn bldg_ind_frm_cursor(&self, bldgs: &Vec<Bldg>, map_data: &MapData, exf: &HashedMapEx) -> Option<usize> {
		if self.zoom_ind == map_data.max_zoom_ind() || bldgs.len() != 0 {
			let map_coord = self.cursor_to_map_ind(map_data);
			if let Some(ex) = exf.get(&map_coord) {
				if let Some(bldg_ind) = &ex.bldg_ind {
					if bldgs[*bldg_ind].owner_id == self.cur_player {
						return Some(*bldg_ind);
					}
				}
			}
		}
		None
	}
}

// uses bldgs[cur_bldg_ind].coord to set map to None if replace_bldg_ind is None
// if replace_bldg_ind is not None, sets map to point to cur_bldg_ind anywhere where replace_bldg_ind should be
// bldg should be at bldgs[cur_bldg_ind]
//
// updates zoom maps. if bldg is a city hall, updates city hall indices to cur_bldg_ind on map 
// where they equal replace_bldg_ind

pub fn replace_map_bldg_ind<'bt,'ut,'rt,'dt>(cur_bldg_ind: usize, replace_bldg_ind: Option<usize>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	let b = &bldgs[cur_bldg_ind];
	let bt = &b.template;
	let c = Coord::frm_ind(b.coord, map_sz);
	
	let h = bt.sz.h as isize;
	let w = bt.sz.w as isize;
	
	// update full map w/ new index
	let exf = exs.last_mut().unwrap();
	for i_off in 0..h {
	for j_off in 0..w {
		let map_coord = map_sz.coord_wrap(c.y + i_off, c.x + j_off).unwrap();
		let ex = &mut exf.get_mut(&map_coord).unwrap();
		 
		ex.bldg_ind = if replace_bldg_ind == None {
			None
		}else{
			Some(cur_bldg_ind)
		};
	}}
	
	// update zoomed maps
	for i_off in 0..h {
	for j_off in 0..w {
		let map_coord = map_sz.coord_wrap(c.y + i_off, c.x + j_off).unwrap();
		compute_zooms_coord(map_coord, bldgs, bldg_templates, map_data, exs, players);
	}}
	
	let b = &bldgs[cur_bldg_ind];
	
	if let BldgArgs::PopulationCenter {..} = &b.args {
		// if bldg whose bldg_ind has been chgd is a city hall, update zones_info.city_hall_bldg_ind
		if let Some(replace_bldg_ind_un) = replace_bldg_ind {
			for (_, zi) in players[b.owner_id as usize].zone_exs.iter_mut() {
				zi.replace_city_hall_bldg_ind_after_swap_rm(replace_bldg_ind_un, cur_bldg_ind);
			}
		// deleting a city hall
		}else{
			for (_, zi) in players[b.owner_id as usize].zone_exs.iter_mut() {
				zi.city_hall_removed();
			}
		}
	} // city hall
}

pub enum UnitDelAction<'d,'u,'bt,'ut,'rt,'dt,'g> {
	// store to-be-deleted units in disband_unit_inds
	Record(&'d mut Vec<usize>),
		
	// deletes units immediately
	Delete {
		units: &'u mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		gstate: &'g mut GameState
	}
}

pub fn rm_bldg<'bt,'ut,'rt,'dt>(bldg_ind: usize, is_cur_player: bool, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
		bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, del_action: UnitDelAction<'_,'_,'bt,'ut,'rt,'dt,'_>, map_sz: MapSz) {
	
	let b = &mut bldgs[bldg_ind];
	let orig_coord = b.coord;
	let orig_bt = b.template;
	let b_owner_id = b.owner_id as usize;
	
	let player = &mut players[b_owner_id];
	
	// accounting
	match del_action {
		UnitDelAction::Record(disband_unit_inds) => {
			player.rm_bldg(bldg_ind, b, disband_unit_inds);
		}
		UnitDelAction::Delete {units, gstate} => {
			let mut disband_unit_inds = Vec::new();
			player.rm_bldg(bldg_ind, b, &mut disband_unit_inds);
			
			for unit_ind in disband_unit_inds.iter() {
				if !disband_unit_inds.contains(unit_ind) {
					disband_unit(*unit_ind, is_cur_player, units, map_data, exs, players, gstate, map_sz);
				}
			}
		}
	}
	
	let player = &mut players[b_owner_id];
	
	// only subtract research per turn if bldg had finished construction
	if b.construction_done.is_none() {
		player.stats.research_per_turn -= b.template.research_prod;
	}
	
	if let Some(resource) = bldg_resource(orig_coord, orig_bt, map_data, map_sz) {
		player.stats.resources_avail[resource.id as usize] -= 1;
	}
	
	// update stats for removed building
	if b.template.upkeep >= 0. { // bldg costs money to maintain
		player.stats.bldg_expenses -= b.return_taxable_upkeep();
	}else{ // pays taxes
		b.set_taxable_upkeep(0., &mut player.stats); // updates stats
	}
	
	rm_all_commutes(bldg_ind, bldgs, player, map_sz);
	
	replace_map_bldg_ind(bldg_ind, None, bldgs, bldg_templates, map_data, exs, players); 
	// ^ call before removing entry in bldgs[]
	
	bldgs.swap_remove(bldg_ind);
	
	// previous bldgs[bldgs.len()-1] was set to bldgs[bldg_ind]
	// maps and commutes need to be updated
	if bldg_ind != bldgs.len() {
		// swap remove should be already called before:
		replace_map_bldg_ind(bldg_ind, Some(bldgs.len()), bldgs, bldg_templates, map_data, exs, players);
				
		// swap remove should be already called before:
		update_commute_bldg_inds(bldg_ind, bldgs.len(), bldgs);
		
		let b = &bldgs[bldg_ind];
		bldg_map_update(Coord::frm_ind(b.coord, map_sz), b.template, bldgs, bldg_templates, map_data, exs, players, map_sz);
		
		// accounting
		let b_owner_id = b.owner_id as usize;
		players[b_owner_id].chg_bldg_ind(bldgs.len(), bldg_ind, b);
	}
	
	bldg_map_update(Coord::frm_ind(orig_coord, map_sz), orig_bt, bldgs, bldg_templates, map_data, exs, players, map_sz);
}

use crate::gcore::return_effective_tax_rate;
use crate::ai::{CITY_GRID_HEIGHT, CITY_GRID_WIDTH, city_hall_offset};

pub fn add_bldg<'bt,'ut,'rt,'dt>(coord: u64, owner_id: SmSvType, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
		bt: &'bt BldgTemplate<'ut,'rt,'dt>, doctrine_dedication: Option<&'dt DoctrineTemplate>,
		wealth_level: Option<i32>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState) -> bool {
	
	let doctrine_dedication = (|| {
		// if there is a doctrine dedication provided and this building
		// contributes + doctrine points, use it, else, use the undefined doctrine
		if let Some(dedication) = doctrine_dedication {
			if bt.doctrinality_bonus > 0. {
				return dedication;
			}
		}
		&temps.doctrines[0]
	})();
	
	let map_sz = *map_data.map_szs.last().unwrap();
	
	let c = Coord::frm_ind(coord, map_sz);
	
	//// check if bldg can be constructed
	let exf = exs.last().unwrap();
	let mfc_pos = map_data.get(ZoomInd::Full, coord);
	
	for i_off in 0..bt.sz.h as isize {
	for j_off in 0..bt.sz.w as isize {
		if let Some(coord) = map_sz.coord_wrap(c.y + i_off, c.x + j_off) {
			let mzo = if let Some(zone_type) = bt.bldg_type.zone_type() {
				Some(MatchZoneOwner {zone_type, owner_id})
			}else{
				None
			};
			let mfc = if i_off != 0 || j_off != 0 {map_data.get(ZoomInd::Full, coord)} else {mfc_pos};
			
			if !(land_clear(coord, mzo, &mfc, exf) ||
				bt.nm[0] == DOCK_NM && mfc.map_type == MapType::ShallowWater) {
					return false;
			}
		}else{
			return false;
		}
	}}
	////////
	let owner_id = owner_id as usize;
	let player = &mut players[owner_id];
	
	let args = if bt.nm[0] == CITY_HALL_NM || bt.nm[0] == MANOR_NM {
			// create new city plan if this is a human player (it would've already been created for the AI)
			if let PlayerType::Human(ai_state) = &mut player.ptype {
				let offsets = city_hall_offset(CITY_GRID_HEIGHT as isize, CITY_GRID_WIDTH as isize);
				let city_coord = Coord {
					y: c.y - offsets.0,
					x: c.x - offsets.1 - 1
				};
				ai_state.add_city_plan(city_coord, &mut gstate.rng, map_data, map_sz, temps.bldgs);
				ai_state.city_states.last_mut().unwrap().population_center_ind = Some(bldgs.len());
			}
			
			// choose name of city, making sure to not choose one that's already been used
			let nm = temps.nms.new_city_name(&player.personalization, gstate);
					
			// log
			//if bt.nm[0] == MANOR_NM {
				gstate.log_event(LogType::CityFounded {owner_id, city_nm: nm.clone()});
			//}
			
			BldgArgs::PopulationCenter {
				tax_rates: vec!{10; ZoneType::N as usize}.into_boxed_slice(),
				production: Vec::new(),
				population: vec![0; WealthLevel::N as usize],
				nm
			}
		}else if let Some(_) = bt.units_producable {
			BldgArgs::GenericProducable {production: Vec::new()}
		}else if let Some(wealth_level) = wealth_level {
			BldgArgs::Taxable {wealth_level}
		}else {BldgArgs::None};
	
	////////////////////////////////////////////////////////////////////////////////////////////
	// stats
	//	Note: update set_owner() [zones/set_owner.rs] if changed
	let (taxable_upkeep, resource_opt) = {
		// taxes
		let taxable_upkeep = if bt.upkeep > 0. {
			player.stats.bldg_expenses += bt.upkeep;
			bt.upkeep
		}else{
			let taxable_upkeep = if let BldgType::Gov(_) = bt.bldg_type {
				player.stats.tax_income -= bt.upkeep; // bt.upkeep is negative so we are adding a positive value
				-bt.upkeep // make positive then add to income
			}else{
				-bt.upkeep * return_effective_tax_rate(coord, map_data, exs, player, bldgs, temps.doctrines, map_sz, gstate.turn)
				//           ^ may call set_city_hall_dist which will not update stats because the bldg hasn't been added yet
			};
			taxable_upkeep
		};
		
		// see also when altering: worker create building, zones/set_owner
		// for taxable buildings (no construction time, these are updated immediately)
		// for gov buildings (w/ a construction time) these are updated when the worker completes them
		// counters for research, crime, happiness, doctrinality, health, pacifism
		match bt.bldg_type {
			BldgType::Gov(_) => {}
			BldgType::Taxable(_) => {
				player.stats.bldg_stats(StatsAction::Add, bt);
			}
		}
		
		// resource
		let resource_opt = bldg_resource(coord, bt, map_data, map_sz);
		
		// log resource
		if let Some(resource) = resource_opt {
			player.stats.resources_avail[resource.id as usize] += 1;
		}
		
		(taxable_upkeep, resource_opt)
	};
	/////////////////////////////////////////////////////////////////////////
	
	bldgs.push(Bldg::default(coord, owner_id as SmSvType, taxable_upkeep, resource_opt, doctrine_dedication, bt, args));
	
	//// set map
	let exf = exs.last_mut().unwrap();

	for i_off in 0..bt.sz.h as isize {
	for j_off in 0..bt.sz.w as isize {
		let coord = map_sz.coord_wrap(c.y + i_off, c.x + j_off).unwrap();
		exf.create_if_empty(coord);
		let ex = exf.get_mut(&coord).unwrap();
		
		// req taxable bldgs be entirely within the correct zone and owned by current owner
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			if let Some(zone_type) = bt.bldg_type.zone_type() {
				debug_assertq!(Some(zone_type) == ex.actual.ret_zone_type() && ex.actual.owner_id == Some(owner_id as SmSvType));
			}
			
			debug_assertq!(ex.actual.owner_id.is_none() || ex.actual.owner_id == Some(owner_id as SmSvType));
		}
		
		ex.actual.owner_id = Some(owner_id as SmSvType);
		ex.bldg_ind = Some(bldgs.len()-1);
	}}
	
	bldg_map_update(c, bt, bldgs, temps.bldgs, map_data, exs, players, map_sz);
	
	// log ai state
	players[owner_id].add_bldg(bldgs.len()-1, coord, bldgs.last().unwrap(), map_sz);
	
	true
}

//// update zoom maps if bldg is added or rmd
fn bldg_map_update<'bt,'ut,'rt,'dt>(c: Coord, bt: &BldgTemplate, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, 
		map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz) {
	for i_off in 0..(bt.sz.h as isize) {
	for j_off in 0..(bt.sz.w as isize) {
		let coord = map_sz.coord_wrap(c.y + i_off, c.x + j_off).unwrap();
		compute_zooms_coord(coord, bldgs, bldg_templates, map_data, exs, players);
	}}
}

pub fn bldg_resource<'rt>(coord: u64, bt: &BldgTemplate, map_data: &mut MapData<'rt>, map_sz: MapSz) -> Option<&'rt ResourceTemplate> {
	const DIST_SEARCH: isize = 25;
	
	if let Some(zone_type) = bt.bldg_type.zone_type() {
		let c = Coord::frm_ind(coord, map_sz);
		
		for i_off in -DIST_SEARCH..(bt.sz.h as isize + DIST_SEARCH) {
		for j_off in -DIST_SEARCH..(bt.sz.w as isize + DIST_SEARCH) {
			if let Some(coord_chk) = map_sz.coord_wrap(c.y + i_off, c.x + j_off) {
				let mfc = map_data.get(ZoomInd::Full, coord_chk);
				if let Some(resource) = mfc.get_resource(coord_chk, map_data, map_sz) {
					if zone_type == resource.zone {
						return Some(resource);
					}
				}
			}
		}}
	}
	None
}

#[derive(PartialEq)]
pub enum StatsAction {Add, Rm}

impl <'dt>Stats<'_,'_,'_,'dt> {
	pub fn bldg_stats(&mut self, action: StatsAction, bt: &BldgTemplate) {
		let f = match action {
			StatsAction::Add => {
				self.research_per_turn += bt.research_prod; // because it's a u32
				1.
			}
			StatsAction::Rm => {
				self.research_per_turn -= bt.research_prod;
				-1.
			}
		};
		
		self.crime += f*bt.crime_bonus;
		// updated in zones/happiness.rs:
		//	self.happiness += f*bt.happiness_bonus;
		//	self.doctrinality[doctrine_dedication.id] += f*bt.doctrinality_bonus;
		//	self.pacifism += f*bt.pacifism_bonus;
		self.health += f*bt.health_bonus;
		
		/*// update prevailing doctrine
		{
			let mut max_val = self.doctrinality[0];
			self.doctrine_template = &doctrine_templates[0];
			
			for (doc_pts, d) in self.doctrinality.iter()
								.zip(doctrine_templates.iter())
								.skip(1) {
				if !d.bldg_reqs_met(self) || max_val >= *doc_pts {continue;}
				
				max_val = *doc_pts;
				self.doctrine_template = d;
			}
		}*/
	}
}

pub fn bldg_ind_frm_coord(coord: u64, bldgs: &Vec<Bldg>) -> Option<usize> {
	bldgs.iter().position(|b| b.coord == coord)
}

pub fn bldg_frm_coord<'b,'bt,'ut,'rt,'dt>(coord: u64, bldgs: &'b mut Vec<Bldg<'bt,'ut,'rt,'dt>>) 
		-> Option<&'b mut Bldg<'bt,'ut,'rt,'dt>> {
	bldgs.iter_mut().find(|b| b.coord == coord)
}

// used for rioters so that they do not initially target city halls
pub fn closest_owned_non_city_hall_bldg<'b,'bt,'ut,'rt,'dt>(coord: Coord, 
		population_thresh: u32, owner_id: SmSvType, 
		bldgs: &'b Vec<Bldg<'bt,'ut,'rt,'dt>>, map_sz: MapSz) -> Option<&'b Bldg<'bt,'ut,'rt,'dt>> {
	struct BldgDist {
		bldg_ind: usize,
		dist: usize
	}
	let mut bldg_dist_opt: Option<BldgDist> = None;
	for (bldg_ind, b) in bldgs.iter().enumerate().filter(|(_, b)| {
		if let BldgArgs::PopulationCenter {population, ..} = &b.args {
			if population.iter().sum::<u32>() > population_thresh {
				return false;
			}
		}
		b.owner_id == owner_id
	}) {
		let dist = manhattan_dist(coord, Coord::frm_ind(b.coord, map_sz), map_sz);
		// check if less than previously found entry
		if let Some(bldg_dist) = &bldg_dist_opt {
			if dist < bldg_dist.dist {
				bldg_dist_opt = Some(BldgDist {bldg_ind, dist});
			}
		// no entries found yet, save this one
		}else{
			bldg_dist_opt = Some(BldgDist {bldg_ind, dist});
		}
	}
	
	if let Some(bldg_dist) = bldg_dist_opt {
		Some(&bldgs[bldg_dist.bldg_ind])
	}else{
		None
	}
}

impl Bldg<'_,'_,'_,'_> {
	// coords start at upper left and go clockwise
	pub fn perimeter_coords(&self, map_sz: MapSz) -> Vec<u64> {
		let b_coord = Coord::frm_ind(self.coord, map_sz);
		let h = self.template.sz.h as isize;
		let w = self.template.sz.w as isize;
		let mut coords = Vec::with_capacity((h*2 + w*2 + 4) as usize);
		// top row (left to right)
		for x_off in -1..=w {
			if let Some(coord) = map_sz.coord_wrap(b_coord.y - 1, b_coord.x + x_off) {
				coords.push(coord);
			}
		}
		
		// right side (top to bottom)
		for y_off in 0..h {
			if let Some(coord) = map_sz.coord_wrap(b_coord.y + y_off, b_coord.x + w) {
				coords.push(coord);
			}
		}
		
		// bottom row (right to left)
		for x_off in (-1..=w).rev() {
			if let Some(coord) = map_sz.coord_wrap(b_coord.y + h, b_coord.x + x_off) {
				coords.push(coord);
			}
		}
		
		// left side (bottom to top)
		for y_off in (0..h).rev() {
			if let Some(coord) = map_sz.coord_wrap(b_coord.y + y_off, b_coord.x - 1) {
				coords.push(coord);
			}
		}
		
		coords
	}
}

