use crate::movement::*;
use crate::disp::*;
use crate::map::*;
use crate::buildings::*;
use crate::saving::*;
use crate::units::*;
use crate::gcore::hashing::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::disp_lib::endwin;
use crate::player::Player;

pub mod disp; pub use disp::*;
pub mod utils; pub use utils::*;
pub mod set_owner; pub use set_owner::*;
pub mod sample_from_turn_computed; pub use sample_from_turn_computed::*;
pub mod happiness; pub use happiness::*;

/////////////////// Fog
pub const FOG_UNIT_DIST: usize = 6;//12;

#[derive(Clone, PartialEq, Copy)]
pub struct StructureData {
	pub structure_type: StructureType,
	pub health: u8,
	pub orientation: char // chars: /|\-
}

impl_saving! {StructureData {structure_type, health, orientation}}

// Note: `.is_empty()` should be updated if new entries added to FogVars
#[derive(Clone, PartialEq)]
pub struct FogVars<'bt,'ut,'rt,'dt> {
	pub owner_id: Option<SmSvType>,
	pub structure: Option<StructureData>,
	
	zone_type: Option<ZoneType>, // stats[owner] and zone_exs_owners map counters potentially need to be updated when chgd
	// ^ use:   add_zone(), rm_zone(), ret_zone_type()    to access
	
	// for zoomed out views:
	pub max_bldg_template: Option<&'bt BldgTemplate<'ut,'rt,'dt>>,
	pub max_city_nm: Option<String>,
}

impl_saving! { FogVars<'bt,'ut,'rt,'dt>{ owner_id, structure, zone_type, max_bldg_template, max_city_nm }}

impl Default for FogVars<'_,'_,'_,'_> {
	fn default() -> Self {
		FogVars {
			owner_id: None,
			structure: None,
			zone_type: None,
			max_bldg_template: None,
			max_city_nm: None
		}
	}
}

// zone maintenence fns, and setting structure from unit's path_coords
impl <'bt,'ut,'rt,'dt>FogVars<'bt,'ut,'rt,'dt> {
	#[inline]
	pub fn ret_zone_type(&self) -> Option<ZoneType> {self.zone_type}
	
	// decrement ZoneEx map counter
	// update pstats.zone_demand_sum_map[zt] if zone is no longer anywhere on map
	pub fn rm_zone(&mut self, coord: u64, players: &mut Vec<Player>, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz) {
		// we only have something todo if there is actually a zone placed
		if let Some(zt) = self.zone_type {
			let zt = zt as usize;
			
			let owner_id = self.owner_id.unwrap() as usize;
			let player = &mut players[owner_id];
			
			if let Some(zone_ex) = player.zone_exs.get_mut(&return_zone_coord(coord, map_sz)) {
				// ^ should already be created from add_zone
				
				// update map counter:
				let map_counter = &mut zone_ex.demand_weighted_sum_map_counter[zt];
				debug_assertq!(*map_counter > 0, "map_counter already 0. owner: {}", owner_id);
				*map_counter -= 1;
				
				// no longer on map, update stats
				if *map_counter == 0 {
					if let Some(demand_weighted_sum) = zone_ex.demand_weighted_sum[zt] {
						player.stats.zone_demand_sum_map[zt].demand_weighted_sum -= demand_weighted_sum;
						player.stats.zone_demand_sum_map[zt].n_summed -= 1;
					}
					
					// happiness, doctrinality, pacifism
					{
						let stats_old = &mut zone_ex.zone_agnostic_stats;
						
						// decrement from pstats
						player.stats.locally_logged = player.stats.locally_logged.clone() - stats_old.locally_logged.clone();
						
						// zero out in case the zone is ever added back
						//	(return_happiness() would incorrectly update
						//	 pstats if these are not cleared)
						*stats_old = ZoneAgnosticStats::default_init(0, doctrine_templates);
						
						// we have to clear the city hall dist because if this map location is ever used again
						// and the city hall distance is set, it will attempt to update the previous city hall's
						// population counter (as if the game were transfering the resident count from one city to another,
						// and would corrupt the counters)
						zone_ex.city_hall_dist = Dist::NotInit;
					}
				}
				
				if self.structure == None {
					self.owner_id = None;
				}
				self.zone_type = None;
			}else{printlnq!("could not remove zone at {}", Coord::frm_ind(coord, map_sz));}
		}
	}
	
	// increment ZoneEx map_counter
	// update pstats.zone_demand_sum_map[zt] if zone is now on map
	pub fn add_zone(&mut self, coord: u64, zone_type: ZoneType, player: &mut Player, doctrine_templates: &Vec<DoctrineTemplate>, map_sz: MapSz) {
		//self.rm_zone(coord, zone_exs_owners, stats, doctrine_templates, map_sz); // needed to update zone counters ?
		
		self.owner_id = Some(player.id);
		
		let zt = zone_type as usize;
		
		// load ZoneEx
		let zn_coord = return_zone_coord(coord, map_sz);
		player.zone_exs.create_if_empty(zn_coord, doctrine_templates);
		let zone_ex = player.zone_exs.get_mut(&return_zone_coord(coord, map_sz)).unwrap();
		
		// update ZoneEx map counter
		let map_counter = &mut zone_ex.demand_weighted_sum_map_counter[zt];
		*map_counter += 1;
		
		// zone demand just added to map, update stats if zone demand has already been computed
		if *map_counter == 1 { 
			if let Some(demand_weighted_sum) = zone_ex.demand_weighted_sum[zt] {
				player.stats.zone_demand_sum_map[zt].demand_weighted_sum += demand_weighted_sum;
				player.stats.zone_demand_sum_map[zt].n_summed += 1;
			}
		}
		
		self.zone_type = Some(zone_type);
	}
	
	// directly set zone_type-- this is needed in map/zoom_out.rs because stats[owner] should *NOT*
	// be updated for zoomed out representations of the map
	pub fn set_zone_direct(&mut self, zone_type: Option<ZoneType>) {self.zone_type = zone_type}
	
	pub fn ret_structure(&self) -> Option<StructureType> {
		if let Some(structure) = self.structure {
			Some(structure.structure_type)
		}else{
			None
		}
	}
	
	// get orientation from unit's path_coords
	pub fn set_structure(&mut self, u: &Unit, structure_type: StructureType, map_sz: MapSz){
		// get orientation
		let get_orientation = || {
			if let Some(action) = u.action.last() {
				if let Some(&p_next) = action.path_coords.last() {
					let c2 = Coord::frm_ind(p_next, map_sz);
					let c = Coord::frm_ind(u.return_coord(), map_sz);
					
					let dist_comps = manhattan_dist_components(c, c2, map_sz);
					
					return if c.x == c2.x && dist_comps.h == 1 {
						'-'
					}else if c.y == c2.y && dist_comps.w == 1 {
						'|'
					}else{
						if dist_comps.h == 0 && dist_comps.w == 0 { // starting to build structure at current position and path is more than zero
							'-' // todo
						}else{
							#[cfg(any(feature="opt_debug", debug_assertions))]
							{
								// for building a wall, the gate can leave a gap, the unit simply jumps over it and does not move there
								if dist_comps.h > 2 || dist_comps.w > 2 {
									//endwin();
									debug_assertq!(dist_comps.h <= 2 && dist_comps.w <= 2,
											"unit {} owner_id {} dist_comps {} {}", u.nm, u.owner_id, dist_comps.h, dist_comps.w);
								}
							}
							if (c.y - c2.y) == (c.x - c2.x) {
								'\\'
							}else{
								'/'
							}
						}
					};
				}
			}
			'-' // todo
		};
		let orientation = get_orientation();
			
		self.owner_id = Some(u.owner_id);
		self.structure = Some(StructureData {
			structure_type,
			health: std::u8::MAX,
			orientation
		});
	}
	
	pub fn rm_structure(&mut self, coord: u64, players: &mut Vec<Player>, map_sz: MapSz) {
		if let Some(_) = self.structure {
			// record wall as being damaged
			players[self.owner_id.unwrap() as usize].log_damaged_wall(Coord::frm_ind(coord, map_sz));
			
			self.structure = None;
			// if no zone is present, remove the owner
			if self.zone_type == None {
				self.owner_id = None;
			}
		}else{panicq!("tried to remove non-existant structure");}
	}
	
	pub fn is_empty(&self) -> bool {
		self.owner_id.is_none() &&
		self.structure.is_none() &&
		self.zone_type.is_none() &&
		self.max_bldg_template.is_none()
	}
}

//////////////////////////////////// stored in hashmap

#[derive(Clone, PartialEq)]
pub struct ZoneDemandRaw {
	pub turn_computed: usize,
	pub demand: Box<[isize]>, // indxd by ZoneDemandType
}

impl_saving!{ ZoneDemandRaw{turn_computed, demand} }

// relative contributions of each component
#[derive(Clone, PartialEq, Debug)]
pub struct ZoneAgnosticContribFracs {
	// doctrine + pacifism = 1.
	pub doctrine: f32,
	pub pacifism: f32,
	
	// health + unemployment + crime
	pub health: f32,
	pub unemployment: f32,
	pub crime: f32,
	
	pub pos_sum: f32,
	pub neg_sum: f32
}

impl_saving!{ZoneAgnosticContribFracs{doctrine, pacifism, health, unemployment, crime, pos_sum, neg_sum}}

// summed locally at each ZoneEx, then summed together across ZoneExs in pstats
#[derive(Clone, PartialEq, Debug)]
pub struct ZoneAgnosticLocallyLogged {
	pub happiness_sum: f32,
	pub doctrinality_sum: Vec<f32>,
	pub pacifism_sum: f32,
	
	pub contrib: ZoneAgnosticContribFracs, // contribution of each factor to happiness_sum
}

impl_saving!{ZoneAgnosticLocallyLogged{happiness_sum, doctrinality_sum, pacifism_sum, contrib}}

#[derive(Clone, PartialEq, Debug)]
pub struct ZoneAgnosticStats {
	pub turn_computed: usize,
	pub gov_bldg_happiness_sum: f32, // ex from things like parks; has no pstats analog
	
	pub locally_logged: ZoneAgnosticLocallyLogged, // happiness, doctrinality, pacifism
	// ^ and contributions of doctrinality, pacifism, crime, health, unemployment to local sum
	
	pub crime_sum: f32, // pstats updated w/ pstats.bldg_stats()
	pub health_sum: f32, // pstats updated w/ pstats.bldg_stats()
	pub unemployment_sum: f32 // pstats updated w/ add/rm residents(?)
	
	// **** ZoneAgnosticStats.happiness() pstats updated w/ pstats.return_happiness()
	// note for ZoneAgnosticLocallyLogged entries:
	//	they are added once per zone_ex entry (unlike zone demands which sum total usage for
	// 	every map coord that utilizes it. [multiple map coords can rely on a single zone_ex])
	//	Therefore to transfer ownership of a zone (remove from old owner, add to new),
	//	here's what needs to happen for the zone_ex's ZoneAgnosticStats entries:
	//		removing the zone: pstats should only be updated once the map counter is
	//				zero for all zone types (i.e., when the zone_ex is no longer used by
	//				any map coordinates.
	//		adding the zone: pstats does not need to be updated. it'll be updated
	//				once return_happiness() is computed on one of the zone_ex's containing coords
}

impl_saving!{ ZoneAgnosticStats{
	turn_computed,
	gov_bldg_happiness_sum,
	
	locally_logged,
	
	crime_sum,
	health_sum,
	unemployment_sum} }

//////////// accessed w/:
//	zone_exs_owners[owner_id].get(return_zone_coord(coord, ..))
#[derive(Clone, PartialEq)]
pub struct ZoneEx {
	city_hall_dist: Dist, // distance to cityhall
				    // use zone_ex.set_city_hall_dist() & .ret_city_hall_dist()
	
	////////////////////////////////
	// demand based on connections to specific zone types
	pub demand_weighted_sum_map_counter: Box<[usize]>, // indexd by zone_type -- # of times demand_weighted_sum used on map
	pub demand_weighted_sum: Box<[Option<f32>]>, // indexed by ZoneType, computed from return_potential_demand(), which is computed from demand_raw[:]
	// ^ each value can be computed across multiple demand_raw[zone_type][zone_demand_type] for any zone_type and zone_demand_type
	//   for example the residential demand_weighted_sum can depend on demand_raw[industrial zone][UnusedResidentCapac] 
	
	pub demand_raw: Box<[Option<ZoneDemandRaw>]>, // used to compute `demand_weighted_sum` entry in ZoneEx
		// by return_potential_demand() which calls return_zone_raw_demands()
		// ^ indexed by ZoneType
	// *** note: demand_weighted_sum and demand_raw should not be set by external functions outside of zones/mode.rs
	
	/////////////////////////////////
	pub zone_agnostic_stats: ZoneAgnosticStats // ex. gov happiness
}

impl_saving!{ ZoneEx {city_hall_dist, demand_weighted_sum_map_counter, demand_weighted_sum, demand_raw,
				zone_agnostic_stats} }

//////////////////
// functions to guard/update the bldgs[ch_bldg_ind].population counter
//	every time a building is added to the city, that counter should be updated
impl ZoneEx {
	pub fn ret_city_hall_dist(&self) -> Dist {
		self.city_hall_dist
	}
	
	// if the city hall for the zone has changed,
	// update the population statistics for both
	// the sending and receiving city hall
	// if the owners differ, set distance to Dist::NotInit
	// in case the player ever uses this zone again
	pub fn set_city_hall_dist(&mut self, zone_coord: u64, new_dist: Dist, bldgs: &mut Vec<Bldg>, map_sz: MapSz) {
		// check if sender and receiver are the same, if so, return
		match new_dist {
			// (the match inside of a match is a little contrived, but
			//   will alert us that we may need to re-think the logic
			//   if any new entries are added to the Dist enum)
			
			Dist::Is {bldg_ind: receiver_ind, ..} |
			Dist::ForceRecompute {bldg_ind: receiver_ind, ..} => {
				match self.city_hall_dist {
					Dist::Is {bldg_ind: sender_ind, ..} |
					Dist::ForceRecompute {bldg_ind: sender_ind, ..} => {
						if receiver_ind == sender_ind {
							self.city_hall_dist = new_dist;
							return;
						}
					}
					Dist::NotInit |
					Dist::NotPossible {..} => {}
				}
			}
			Dist::NotInit |
			Dist::NotPossible {..} => {}
		}
		
		let mut population_moved = None;
		macro_rules! count_population_moved{() => {
			// only count if we haven't already done so
			if population_moved == None {
				let mut sum = 0;
				for b in bldgs.iter().filter(|b| {
					BldgType::Taxable(ZoneType::Residential) == b.template.bldg_type &&
					return_zone_coord(b.coord, map_sz) == zone_coord
				}) {
					sum += b.n_residents();
				}
				population_moved = Some(sum);
			}
		};};
		
		// add population counts to receiver
		match new_dist {
			Dist::Is {bldg_ind: ch_bldg_ind, ..} |
			Dist::ForceRecompute {bldg_ind: ch_bldg_ind, ..} => {
				count_population_moved!();
				
				if let BldgArgs::PopulationCenter {ref mut population, ..} = bldgs[ch_bldg_ind].args {
					*population += population_moved.unwrap() as u32;
				}else{panicq!("input requires the bldg ind be a city hall");}
			}
			Dist::NotInit |
			Dist::NotPossible {..} => {}
		}
		
		// remove population counts from sender
		match self.city_hall_dist {
			Dist::Is {bldg_ind: ch_bldg_ind, ..} |
			Dist::ForceRecompute {bldg_ind: ch_bldg_ind, ..} => {
				count_population_moved!();
				
				if let BldgArgs::PopulationCenter {ref mut population, ..} = bldgs[ch_bldg_ind].args {
					debug_assertq!(*population >= population_moved.unwrap() as u32);
					*population -= population_moved.unwrap() as u32;
				}else{panicq!("input requires the bldg ind be a city hall");}
			}
			Dist::NotInit |
			Dist::NotPossible {..} => {}
		}
		
		self.city_hall_dist = new_dist;
	}
	
	// called when city hall has been removed and we need to uninitialize
	// the city hall distances.
	// no need to update bldg[ch_bldg_ind].args if it has/is being removed (the counter
	// is stored in bldgs[] so we don't need to alter it if the counter itself is deleted)
	pub fn city_hall_removed(&mut self) {
		match self.city_hall_dist {
			Dist::Is {..} |
			Dist::ForceRecompute {..} => {
				self.city_hall_dist = Dist::NotInit;
			}
			Dist::NotInit |
			Dist::NotPossible {..} => {}
		}
	}
	
	// fn sets bldg_ind to `new_bldg_ind` only if its current bldg_ind value == `old_bldg_ind`
	//	there's no need to change city hall population statistics
	//	(in its BldgArgs) because the remapping should've
	//	resulted in the index pointing to the same building
	//		if not, the population statistics will be corrupted
	// this is called in replace_map_bldg_ind()
	pub fn replace_city_hall_bldg_ind_after_swap_rm(&mut self, old_bldg_ind: usize,
			new_bldg_ind: usize) {
		match self.city_hall_dist {
			Dist::NotInit |
			Dist::NotPossible {..} => {} // nothing to update
			
			Dist::Is {ref mut bldg_ind, ..} |
			Dist::ForceRecompute {ref mut bldg_ind, ..} => {
				if *bldg_ind == old_bldg_ind {
					*bldg_ind = new_bldg_ind;
				}
			}
		}
	}
}

//////////////////////// stored in stats[player_id].zone_demand_sum[zone_type]

#[derive(Clone, PartialEq)]
pub struct ZoneDemandSumMap { 
	demand_weighted_sum: f32,
	n_summed: usize
}

impl_saving!{ZoneDemandSumMap {demand_weighted_sum, n_summed}}

#[cfg(any(feature="opt_debug", debug_assertions))]
impl ZoneDemandSumMap {
	pub fn demand_weighted_sum(&self) -> f32 {self.demand_weighted_sum}
	pub fn n_summed(&self) -> usize {self.n_summed}
}

impl Default for ZoneDemandSumMap {
	fn default() -> Self {
		ZoneDemandSumMap {demand_weighted_sum: 0., n_summed: 0}
	}
}

const N_BLDGS_BOOST_CUTOFF: usize = 15;
const BEGINNING_BOOST: f32 = 0.1;

impl ZoneDemandSumMap {
	// if less than N_BUILDINGS_BOOST_CUTOFF for zone_type, give the player a bonus
	pub fn map_avg_zone_demand(&self, zone_type: ZoneType, bldgs: &Vec<Bldg>, owner_id: SmSvType) -> f32 {
		if let Some(beginning_boost) = return_beginning_boost(zone_type, bldgs, owner_id) {
			beginning_boost
		}else if self.n_summed != 0 {
			self.demand_weighted_sum / (self.n_summed as f32)
		}else{
			BEGINNING_BOOST
		}
		/*if self.n_summed >= N_BLDGS_BOOST_CUTOFF {
			return self.demand_weighted_sum / (self.n_summed as f32);
		}
		BEGINNING_BOOST*/
	}
}
/////////////////////////////////////////

// if less than N_BUILDINGS_BOOST_CUTOFF for zone_type, give the player a bonus
fn return_beginning_boost(zone_type: ZoneType, bldgs: &Vec<Bldg>, owner_id: SmSvType) -> Option<f32> {
	let mut count = 0;
	for b in bldgs {
		if let BldgType::Taxable(bz) = b.template.bldg_type{
			if b.owner_id == owner_id && bz == zone_type {
				count += 1;
				
				// finished: no need to loop anymore
				if count == N_BLDGS_BOOST_CUTOFF {
					return None;
				}
			}
		}
	}
	
	if count < N_BLDGS_BOOST_CUTOFF {Some(BEGINNING_BOOST)} else {None}
}

// starts at map location coord, and then finds paths to the closest buildings
// that can fullfill particular demand types
//
// returns zone_exs_owners[owner_ind].get_mut(coord).demand_raw[zone_type][: (all zone_demand_type)] 
// demand_raw[zone_type] will be set to `demand_raw_zone` returned from this function
const N_ZONE_SAMPLES: usize = 9;
impl ZoneDemandRaw {
	fn new(coord: u64, zone_type_frm: ZoneType, zone_type_to: ZoneType, 
			owner_id: SmSvType, map_data: &mut MapData, 
			exs: &mut Vec<HashedMapEx>, bldgs: &Vec<Bldg>, map_sz: MapSz, turn: usize) -> Self {
		let start_c = Coord::frm_ind(coord, map_sz);
		let exf = exs.last().unwrap();
		let mut demand_raw_zone = ZoneDemandRaw::default_init(turn);
		
		if let Some(mut action_iface) = start_civil_mv_mode(coord, map_data, exf, map_sz) {
			let mut n_conn_found = vec!{0; ZoneDemandType::N as usize};
			
			///////////////////////////
			// loop over bldgs until we find enough or we run out
			for b in bldgs {
				if b.owner_id != owner_id {continue;}
				
				match &b.template.bldg_type {
					BldgType::Taxable(b_zone) => {
						if *b_zone != zone_type_frm {continue;}
						if action_iface.too_far(start_c, b, bldgs, exf, exs, map_data, map_sz) {continue;}
						
						///////////////
						// check if this bldg meets the requested demand type
						for demand_ind in 0..(ZoneDemandType::N as usize) {
							let demand_type = ZoneDemandType::from(demand_ind);
							
							if n_conn_found[demand_ind] == N_ZONE_SAMPLES || demand_type == ZoneDemandType::GovBldg {continue;}
							debug_assertq!(n_conn_found[demand_ind] < N_ZONE_SAMPLES);
							
							n_conn_found[demand_ind] += 1;
							demand_raw_zone.demand[demand_ind] += match demand_type {
								ZoneDemandType::Development => {
									b.n_residents() // number of residents/employees
									
								} ZoneDemandType::ProdAvail => {
									if *b_zone != ZoneType::Residential {
										let prod_capac = b.prod_capac(); // capacity
										let n_sold = b.n_sold(); // amount of production sold
										debug_assertq!(prod_capac >= n_sold, "{} {}", prod_capac, n_sold); // can't be selling more than producing
										
										prod_capac - n_sold
									}else{
										debug_assertq!(b.n_residents() >= b.n_sold()); // shouldn't employ more than lives here
										b.n_residents() - b.n_sold()
										//0 // residential bldgs don't produce anything
									}
									
								} ZoneDemandType::ConsAvail => {
									// consumption ability available
									let cons_capac = b.cons_capac();
									let cons = b.cons();
									
									//endwin();
									//println!("cons {} cons_capac {}", cons, cons_capac);
									
									debug_assertq!(cons_capac >= cons);
									
									cons_capac - cons
									
								} ZoneDemandType::UnusedResidentCapac => {
									// can we add more residents?
									b.template.resident_max - b.n_residents()
									// =     capacity       -      current # of residents
								
								} ZoneDemandType::Resource => {
									let ret_bonus = || {
										if let Some(resource) = b.resource {
											if let Some(bonus) = resource.zone_bonuses[zone_type_to as usize] {
												return bonus as usize;
											}
										}
										0
									};
									ret_bonus()
								
								} ZoneDemandType::GovBldg | ZoneDemandType::N => panicq!("invalid zone demand")
							} as isize;
						} // demand type loop
					}
					BldgType::Gov(bonuses) => {
						if let Some(bonus) = bonuses[zone_type_frm as usize] {
							if action_iface.too_far(start_c, b, bldgs, exf, exs, map_data, map_sz) {continue;}
							//debug_assertq!(zone_type_frm == zone_type_to);
							
							let demand_ind = ZoneDemandType::GovBldg as usize;
							
							n_conn_found[demand_ind] += 1;
							demand_raw_zone.demand[demand_ind] += bonus;
						}
					}
				}
				
				//////////
				// check if finished
				let mut finished = true;
				for n in n_conn_found.iter() {
					if *n != N_ZONE_SAMPLES {
						finished = false;
						break;
					}
				}
				if finished {break;}
			} // bldg loop
		}
		demand_raw_zone
	}
}

// zones demands are not computed for each land plot. they are spaced on a grid
// return_zone_coord corresponds to the point on the grid from a given coord
pub const ZONE_SPACING: isize = 21;
pub fn return_zone_coord(coord: u64, map_sz: MapSz) -> u64 {
	let mut coord_c = Coord::frm_ind(coord, map_sz);
	
	coord_c.x = ZONE_SPACING * (coord_c.x / ZONE_SPACING);
	coord_c.y = ZONE_SPACING * (coord_c.y / ZONE_SPACING);
	
	map_sz.coord_wrap(coord_c.y, coord_c.x).unwrap()
}

pub const N_TURNS_RECOMP_ZONE_DEMAND: usize = 30*12*5; //30*12 * 1;//75;

// return potential demand for zone type, recomputes raw demands (return_raw_demands()), if needed
pub fn return_potential_demand(mut coord: u64, map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx>, player: &mut Player, bldgs: &Vec<Bldg>, map_sz: MapSz, turn: usize) -> f32 {
	
	let exf = exs.last().unwrap();
	let ex = exf.get(&coord).unwrap();
	let zone_type = ex.actual.zone_type.unwrap();
	
	//////////// compute zone demands on a spaced grid, unless zone doesn't match the grid
	coord = return_zone_coord(coord, map_sz);
	let zone_ex = player.zone_exs.get_mut(&coord).unwrap(); // should be created by add_zone() method in FogVars
	
	let owner_id = ex.actual.owner_id.unwrap();
	debug_assertq!(owner_id == player.id);
	
	// during the early game, give the payer a bonus
	if let Some(beginning_boost) = return_beginning_boost(zone_type, bldgs, owner_id) {return beginning_boost};
	
	//////////////////////////////////////// Set param weights
	struct Params {zone_type: ZoneType, demand_type: ZoneDemandType, scale: f32}
	
	let mut params: Vec<Params> = Vec::new();
	
	macro_rules! zn_param{
		($zone_type: expr, $demand_type: expr, $scale: expr) => (
			params.push(Params {zone_type: $zone_type, demand_type: $demand_type, scale: $scale}););
	}
	
	zn_param!(zone_type, ZoneDemandType::GovBldg, 1.); // gov bldg bonus for zone type being queried
	
	// bonuses from resources
	//	(regardless of what zone the resource is from, it could give a bonus to another zone type)
	for zt in 0..(ZoneType::N as usize) {
		zn_param!(ZoneType::from(zt), ZoneDemandType::Resource, 1.);
	}
	
	let mut demand = 0.;
	let mut max_sum = 0.;
	
	// example elevation and arability values
	/* steppe: 2.4929237 58.13279
	prarie: 1.1988862 84.46308
	meadow: 0.6811216 108.76186
	broadleaf forest: 2.8568678 200.41454
	northern pine forest: 0.49236917 43.93574
	mountainous tundra: 16.93524 2.2890272
	broadleaf forest: 0.5418797 190.39809 */
	
	const LAND_MULT: f32 = 1.;//20.;
	
	match zone_type {
		ZoneType::Residential => {
			zn_param!(ZoneType::Agricultural, ZoneDemandType::Development, -0.1/2.);
			zn_param!(ZoneType::Industrial, ZoneDemandType::Development, -1./2.);
			
			zn_param!(ZoneType::Residential, ZoneDemandType::ProdAvail, -1.); // unemployment
			zn_param!(ZoneType::Business, ZoneDemandType::ProdAvail, 1.); // products to buy
			zn_param!(ZoneType::Agricultural, ZoneDemandType::ProdAvail, 1.); // food to buy
			
			// jobs:
			zn_param!(ZoneType::Business, ZoneDemandType::UnusedResidentCapac, 1.);
			zn_param!(ZoneType::Industrial, ZoneDemandType::UnusedResidentCapac, 1.);
			zn_param!(ZoneType::Agricultural, ZoneDemandType::UnusedResidentCapac, 1.);
			
			zn_param!(ZoneType::Residential, ZoneDemandType::UnusedResidentCapac, -2.);//0.5/2.); // unused houses
			
			// arability 
			const F: f32 = LAND_MULT*60./2.;
			demand += F*((map_data.get(ZoomInd::Full, coord).arability / (ARABILITY_STEP as f32)) - 4.);
			max_sum += F;
			
		} ZoneType::Agricultural => {
			// dont want to be near these:
			zn_param!(ZoneType::Industrial, ZoneDemandType::Development, -0.5/2.);
			
			zn_param!(ZoneType::Agricultural, ZoneDemandType::ProdAvail, -0.5/2.);
			
			// market for crops:
			zn_param!(ZoneType::Residential, ZoneDemandType::ConsAvail, 2.);
			zn_param!(ZoneType::Business, ZoneDemandType::ConsAvail, 2.);
			
			zn_param!(ZoneType::Agricultural, ZoneDemandType::UnusedResidentCapac, -0.5/2.); // under-used bldgs
			
			// arability 
			const F: f32 = LAND_MULT*80./2.;
			demand += F*((map_data.get(ZoomInd::Full, coord).arability / (ARABILITY_STEP as f32)) - 4.);
			max_sum += F;
			
		} ZoneType::Business => {
			// market for products:
			zn_param!(ZoneType::Residential, ZoneDemandType::ConsAvail, 1.);
			
			zn_param!(ZoneType::Residential, ZoneDemandType::ProdAvail, 1.); // employmees
			zn_param!(ZoneType::Industrial, ZoneDemandType::ProdAvail, 1.); // source material
			zn_param!(ZoneType::Agricultural, ZoneDemandType::ProdAvail, 0.25);
			
			zn_param!(ZoneType::Business, ZoneDemandType::UnusedResidentCapac, -1.1/2.); // under-used bldgs
			zn_param!(ZoneType::Business, ZoneDemandType::ProdAvail, -1./2.);
			
			// elevation (lower elevation means higher bonus)
			const F: f32 = LAND_MULT*2.;
			demand += F*(10. - map_data.get(ZoomInd::Full, coord).elevation);
			max_sum += F;
		
		} ZoneType::Industrial => {
			// markets
			zn_param!(ZoneType::Business, ZoneDemandType::ConsAvail, 1.);
			zn_param!(ZoneType::Agricultural, ZoneDemandType::ConsAvail, 1.);
			zn_param!(ZoneType::Industrial, ZoneDemandType::ProdAvail, -0.75/4.);
			
			// employees
			zn_param!(ZoneType::Residential, ZoneDemandType::ProdAvail, 1.);
			
			// elevation 
			const F: f32 = LAND_MULT*8.*8.;
			demand += F*(map_data.get(ZoomInd::Full, coord).elevation - 10.);
			max_sum += F;
		
		} ZoneType::N => panicq!("invalid zone type")
	}
	
	////////////////////////////////// Sum together weighted params
	for p in params {
		max_sum += p.scale * (std::i8::MAX as f32);
		
		let p_zone_type = p.zone_type as usize;
		let demand_raw_zone = &mut zone_ex.demand_raw[p_zone_type];
		
		////// check if we re-compute or use old vals
		if demand_raw_zone.is_none() || (demand_raw_zone.as_ref().unwrap().turn_computed + 
				 N_TURNS_RECOMP_ZONE_DEMAND) < turn {
			
			*demand_raw_zone = Some(ZoneDemandRaw::new(coord, p.zone_type, zone_type, owner_id, map_data, exs, bldgs, map_sz, turn));
			// ^ recomputes zone_ex[owner].get_mut(coord).demand_raw[p.zone_type][:] 
			// across all demand types
		}
		
		// add to sum
		demand += p.scale * (demand_raw_zone.as_ref().unwrap().demand[p.demand_type as usize] as f32);
	}
	debug_assertq!(max_sum > 0.); // otherwise the division will flip the sign
	
	let val = demand / max_sum;
	
	//////////////////////////////////// update:
	//      stats[player_id].zone_demand_sum[zone_type], subtract old val if prev set
	let zt = zone_type as usize;
	
	let demand_weighted_sum = &mut zone_ex.demand_weighted_sum[zt]; 
	// ^ specific demand value for map location
	
	let stats_zone_sum_map = &mut player.stats.zone_demand_sum_map[zt];
	// ^ demand across full map for this zone type:
	
	// we have already set a value for this land plot
	if let Some(val_prev) = demand_weighted_sum {
		stats_zone_sum_map.demand_weighted_sum += val - *val_prev;
		
	// we have not yet set a value for this land plot
	}else{
		stats_zone_sum_map.demand_weighted_sum += val;
		stats_zone_sum_map.n_summed += 1;
	}
	
	*demand_weighted_sum = Some(val);
	
	val
}

impl Default for ZoneEx {
	fn default() -> Self {
		let n_zone_types = ZoneType::N as usize;
		ZoneEx {
			city_hall_dist: Dist::NotInit,
			demand_weighted_sum_map_counter: vec!{0; n_zone_types}.into_boxed_slice(),
			demand_weighted_sum: vec!{None; n_zone_types}.into_boxed_slice(),
			demand_raw: vec!{None; n_zone_types}.into_boxed_slice(),
			zone_agnostic_stats: Default::default()
		}
	}
}

impl ZoneEx {
	pub fn default_init(doctrine_templates: &Vec<DoctrineTemplate>) -> Self {
		let n_zone_types = ZoneType::N as usize;
		ZoneEx {
			city_hall_dist: Dist::NotInit,
			demand_weighted_sum_map_counter: vec!{0; n_zone_types}.into_boxed_slice(),
			demand_weighted_sum: vec!{None; n_zone_types}.into_boxed_slice(),
			demand_raw: vec!{None; n_zone_types}.into_boxed_slice(),
			zone_agnostic_stats: ZoneAgnosticStats::default_init(0, doctrine_templates)
		}
	}
}

