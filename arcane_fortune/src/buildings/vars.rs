use crate::disp::*;
use crate::map::*;
use crate::units::*;
use crate::movement::*;
use crate::saving::*;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::zones::return_zone_coord;
use crate::gcore::rand::XorState;
use crate::disp_lib::endwin;
use crate::gcore::{Log, Relations};
use crate::ai::{BarbarianState, AIState};
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::player::{Stats, Player, Nms};
use crate::containers::Templates;

pub const CITY_HALL_NM: &str = "City Hall";
pub const BOOT_CAMP_NM: &str = "Boot Camp";
pub const DOCK_NM: &str = "Dock";
pub const ACADEMY_NM: &str = "Academy";
pub const BARBARIAN_CAMP_NM: &str = "Camp";
pub const MANOR_NM: &str = "Manor";

// global parameters
#[derive(PartialEq, Clone)]
pub struct BldgConfig {
	pub fire_damage_rate: u32,
	pub fire_repair_rate: u32,
	pub max_bldg_damage: u32,
	pub job_search_bonus_dist: u32,
	// ^ the distance a bldg gives a bonus, if it
	//   has a job_search_bonus
	//   the bonus falls off linearly as a fn of the manhattan dist
}
impl_saving!{BldgConfig {fire_damage_rate, fire_repair_rate, max_bldg_damage, job_search_bonus_dist}}

#[derive(PartialEq, Clone)]
pub enum BldgType {
	Taxable(ZoneType),
	Gov(Vec<Option<isize>>) // zone bonuses indexed by ZoneType
}

#[derive(PartialEq, Clone)]
pub struct BldgTemplate <'ut,'rt,'dt> {
	pub id: SmSvType,
	pub nm: Vec<String>,
	pub menu_txt: Option<String>,
	
	pub tech_req: Option<Vec<usize>>,
	pub doctrine_req: Option<&'dt DoctrineTemplate>,
	pub research_prod: SmSvType,
	
	pub sz: ScreenSz,
	pub print_str: String,
	pub plot_zoomed: char, // what to show when zoomed out
	pub bldg_type: BldgType, // either the zone type or the bonus (optional) the gov bldg gives to each zone
	
	pub units_producable: Option<Vec<&'ut UnitTemplate<'rt>>>,
	pub units_producable_txt: Option<Vec<String>>, // for production window
	pub unit_production_rate: SmSvType,
	
	pub construction_req: f32, // how many action turns needed to construct
	pub upkeep: f32, // negative means it's paying taxes
	
	pub resident_max: usize, // max residents (or employees)
	pub cons_max: usize, // max consumption 
	pub prod_max: usize, // max production
	
	pub crime_bonus: f32, // + means more crime
	pub happiness_bonus: f32,
	pub doctrinality_bonus: f32,
	pub pacifism_bonus: f32,
	pub health_bonus: f32,
	pub job_search_bonus: f32,
	
	pub barbarian_only: bool // only barbarians can have this bldg
}

impl_saving_template!{BldgTemplate <'ut,'rt,'dt>{id, nm, menu_txt, tech_req,
			doctrine_req, research_prod,
			sz, print_str, plot_zoomed, bldg_type,
			units_producable, units_producable_txt, 
			unit_production_rate, construction_req, upkeep, 
			resident_max, cons_max, prod_max,
			crime_bonus, happiness_bonus,
			doctrinality_bonus, pacifism_bonus,
			health_bonus, job_search_bonus,
			barbarian_only}}

#[derive(Clone, PartialEq)]
struct Commute {
	bldg_ind: Option<usize>, // residential can have "none" here
			  	// (for the residents)
	zone_type: ZoneType
}

impl Default for Commute {
	fn default() -> Self {
		Commute {bldg_ind: None, zone_type: ZoneType::N}}}

impl_saving!{Commute {bldg_ind, zone_type}}

pub enum CommuteType {Frm, To, None}

impl <'ut,'rt,'dt> BldgTemplate<'ut,'rt,'dt> {
	pub fn frm_str<'bt>(txt: &str, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>) -> &'bt BldgTemplate<'ut,'rt,'dt> {
		
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			let mut found = 0;
			for bt in bldg_templates.iter() {
				if bt.nm[0] == txt {
					found += 1;
				}
			}
			assertq!(found == 1, "Duplicate entries found for: \"{}\"", txt);
		}
		
		for bt in bldg_templates.iter() {
			if bt.nm[0] == txt {
				#[cfg(any(feature="opt_debug", debug_assertions))]
				{
					if let BldgType::Gov(_) = bt.bldg_type {} else {
						panicq!("frm_str used for non gov bldg. {}", bt.nm[0]);
					}
				}
				return bt;
			}
		}
		panicq!("Could not find building with name: \"{}\"", txt);
	}
}

#[derive(Clone, PartialEq)]
pub struct Bldg<'bt,'ut,'rt,'dt> {
	pub coord: u64,
	pub owner_id: SmSvType,
	pub template: &'bt BldgTemplate<'ut,'rt,'dt>,
	pub construction_done: Option<f32>, // action turns expended on construction. "None" means done
	
	pub doctrine_dedication: &'dt DoctrineTemplate,
	
	pub resource: Option<&'rt ResourceTemplate>,
	
	pub fire: Option<Fire>,
	pub damage: Option<u32>,
	
	taxable_upkeep_pre_operating_frac: f32, // based on dist from city hall. stats[owner.id].tax_income should be updated when this changes
	// ^ first set w/ add_bldg (which calls the default fn here), 
	//   then updated w/ set_city_hall_dist (which calls update_taxable_upkeep), then cleared by rm_bldg
	//   (also calls update_taxable_upkeep)
	taxable_upkeep: f32, // updated by self.set_taxable_upkeep
	
	/////////////////////
	// connections to and from other buildings for prod. & consumption
	//////////////
	//
	// not some stats[owner.id] need to be updated when some of the stats of these structures chgs (ex. population)
	bldgs_recv_frm: Vec<Commute>, // residential recvs residents from None bldg_ind
	bldgs_send_to: Vec<Commute>,
	
	pub args: BldgArgs<'ut,'rt>,
}

impl_saving!{Bldg<'bt,'ut,'rt,'dt> {coord, owner_id, template, construction_done, doctrine_dedication,
					resource, fire, damage, taxable_upkeep_pre_operating_frac,
					taxable_upkeep, bldgs_recv_frm, bldgs_send_to, args}}

impl <'bt,'ut,'rt,'dt> Bldg <'bt,'ut,'rt,'dt> {
	pub fn default(coord: u64, owner_id: SmSvType, taxable_upkeep_pre_operating_frac: f32,
			resource: Option<&'rt ResourceTemplate>, doctrine_dedication: &'dt DoctrineTemplate,
			bt: &'bt BldgTemplate<'ut,'rt,'dt>, args: BldgArgs<'ut,'rt>) -> Self {
		
		let (taxable_upkeep, construction_done) = if let BldgType::Gov(_) = bt.bldg_type {
			(taxable_upkeep_pre_operating_frac, Some(0.))
		}else{
			(0., None)
		};
		
		Bldg {coord, owner_id, template: bt, resource,
			bldgs_recv_frm: Vec::new(), bldgs_send_to: Vec::new(),
			construction_done, fire: None, damage: None,
			taxable_upkeep_pre_operating_frac,
			taxable_upkeep,
			doctrine_dedication,
			args}
	}
	
	pub fn set_taxable_upkeep(&mut self, mut new_val: f32, pstats: &mut Stats) {
		if let BldgType::Taxable(_) = self.template.bldg_type {
			self.taxable_upkeep_pre_operating_frac = new_val;
			
			new_val *= self.operating_frac();
			
			pstats.tax_income += new_val - self.taxable_upkeep;
			self.taxable_upkeep = new_val;
		
		}else{panicq!("cannot set taxable upkeep on bldg that pays no taxes");}
	}
	
	// update when n_residents() changes
	#[inline]
	fn population_update_taxable_upkeep(&mut self, pstats: &mut Stats) {
		self.set_taxable_upkeep(self.taxable_upkeep_pre_operating_frac, pstats);
	}
	
	#[inline]
	pub fn ret_taxable_upkeep_pre_operating_frac(&self) -> f32 {
		self.taxable_upkeep_pre_operating_frac
	}
	
	#[inline]
	pub fn return_taxable_upkeep(&self) -> f32 {
		self.taxable_upkeep
	}
	
	// check that all connections have valid bldg inds
	// and only residencies can have recv. connections when the bldg_ind = None
	#[cfg(any(feature="opt_debug", debug_assertions))]
	pub fn chk_connection_inds(&self, n_bldgs: usize) {
		for bldg_recv_frm in &self.bldgs_recv_frm {
			if let Some(bldg_ind) = bldg_recv_frm.bldg_ind {
				assertq!(bldg_ind < n_bldgs);
			}else{
				assertq!(bldg_recv_frm.zone_type == ZoneType::Residential);
				assertq!(self.template.bldg_type == BldgType::Taxable(ZoneType::Residential));
				// setting bldg_recv_frm.bldg_ind = None should only be done for residencies
				// (it is the mechanism of having a population)
			}
		}
		
		for bldg_send_to in &self.bldgs_send_to {
			if let Some(bldg_ind) = bldg_send_to.bldg_ind {
				assertq!(bldg_ind < n_bldgs);
			}else{
				panicq!("bldg is sending to unknown bldg_ind");
			}
		}
	}
	
	// number of consumption items from zone_type
	pub fn n_recv_frm(&self, zone_type: ZoneType) -> usize {
		let mut n = 0;
		for c in &self.bldgs_recv_frm {
			// should only have null bldg_inds for residents 
			// moving in to a residential bldg
			debug_assertq!(!c.bldg_ind.is_none() || self.template.bldg_type == BldgType::Taxable(ZoneType::Residential));
			
			if zone_type == c.zone_type {n += 1;}
		}
		
		debug_assertq!(zone_type != ZoneType::Residential || self.template.resident_max >= n, 
				"res max: {}, n: {}", self.template.resident_max, n);
		n
	}
	
	// residents when self is residential, employees otherwise
	pub fn n_residents(&self) -> usize {
		self.n_recv_frm(ZoneType::Residential)
	}
	
	// (n_sold is the number of people employed for residencies)
	pub fn n_sold(&self) -> usize {
		let n_sold = self.bldgs_send_to.len();
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			if let BldgType::Taxable(zone) = self.template.bldg_type {
				debug_assertq!((zone == ZoneType::Residential && n_sold <= self.n_residents()) || 
						(zone != ZoneType::Residential && n_sold <= self.prod()));
			}else {panicq!("building should be taxable"); }
		}
		n_sold
	}
	
	// current operating fraction (ex. production efficiency)
	pub fn operating_frac(&self) -> f32 {
		let n_residents = self.n_residents();
		let max_residents = self.template.resident_max;
		debug_assertq!(n_residents <= max_residents);
		debug_assertq!(max_residents > 0);
		
		(n_residents as f32) / (max_residents as f32)
	}
	
	// building production capacity
	pub fn prod_capac(&self) -> usize {
		(self.operating_frac() * (self.template.prod_max as f32)).floor() as usize
	}
	
	pub fn cons_capac(&self) -> usize {
		(self.operating_frac() * (self.template.cons_max as f32)).floor() as usize
	}
	
	// building consumption
	pub fn cons(&self) -> usize {
		self.bldgs_recv_frm.len() - self.n_residents()
	}
	
	// building production
	pub fn prod(&self) -> usize {
		if self.template.bldg_type != BldgType::Taxable(ZoneType::Residential) {
			self.bldgs_send_to.len()
		}else{
			0 // residencies don't produce anything
		}
	}
	
	// is bldg_ind connected to Bldg?
	pub fn connected(&self, bldg_ind_find: usize) -> CommuteType {
		if let Some(_) = find_commute_ind(bldg_ind_find, &self.bldgs_send_to) {
			CommuteType::To
		}else if let Some(_) = find_commute_ind(bldg_ind_find, &self.bldgs_recv_frm) {
			CommuteType::Frm
		}else{
			CommuteType::None
		}
	}
}

// building produced unit
// cur_player is from iface_settings, not the player building the unit
pub fn build_unit<'o,'bt,'ut,'rt,'dt>(bldg_ind: usize, cur_player: SmSvType, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, 
		map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		player: &mut Player<'bt,'ut,'rt,'dt>, relations: &mut Relations, logs: &mut Vec<Log>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, turn: usize, rng: &mut XorState) {
	
	enum ProdAction<'ut,'rt> {IncProg, DecProg, FinProd {coord_add: u64, unit_template: &'ut UnitTemplate<'rt>}};
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	let prod_action;
	let b = &bldgs[bldg_ind];
	let unit_production_rate = b.template.unit_production_rate;
	
	// req. building not be under construction
	if let Some(_) = b.construction_done {return;}
	
	match &b.args {
	  BldgArgs::CityHall {production, ..} |
	  BldgArgs::GenericProducable {production} => {
	  	if let Some(entry) = production.last() {
			let movement_type = entry.production.movement_type;
			
			// finished?
			prod_action = if (entry.progress as f32) >= entry.production.production_req {
				//// find bordering location of building to put unit:
				(|| {
					let c = Coord::frm_ind(b.coord, map_sz);
					let sz = b.template.sz;
					let mv_vars = MvVarsAtZoom::NonCivil {units, start_owner: b.owner_id, blind_undiscov: None};
					
					macro_rules! chk_column {($j: expr) => (
						for i_off in -1..=(sz.h as isize) {
							if let Some(coord_add) = map_sz.coord_wrap(c.y + i_off, $j) {
								// set src coord equal to coord_add+1 so that movable_to does not always return true
								if movable_to(coord_add+1, coord_add, &map_data.get(ZoomInd::Full, coord_add), exs.last().unwrap(), mv_vars, bldgs, &Dest::NoAttack, movement_type){
									return ProdAction::FinProd {coord_add, unit_template: entry.production};
								}
							}
						}
					);};
					
					macro_rules! chk_row {($i: expr) => (
						for j_off in -1..=(sz.w as isize) {
							if let Some(coord_add) = map_sz.coord_wrap($i, c.x + j_off) {
								// set src coord equal to coord_add+1 so that movable_to does not always return true
								if movable_to(coord_add+1, coord_add, &map_data.get(ZoomInd::Full, coord_add), exs.last().unwrap(), mv_vars, bldgs, &Dest::NoAttack, movement_type){
									return ProdAction::FinProd {coord_add, unit_template: entry.production};
								}
							}
						}
					);}
					
					chk_column!(c.x-1); // left
					chk_column!(c.x + sz.w as isize); // right
					chk_row!(c.y-1); // top
					chk_row!(c.y + sz.h as isize); // bottom
					
					ProdAction::DecProg
				})()
			}else{
				ProdAction::IncProg
			};
		// not producing anything
		}else{return;}
	 } BldgArgs::None => {return;}}
	
	/////////////////////
	// perform production action
	match prod_action {
		ProdAction::DecProg => (),
		ProdAction::IncProg => {
			let b = &mut bldgs[bldg_ind];
			match b.args {
			  BldgArgs::CityHall {ref mut production, ..} | BldgArgs::GenericProducable {ref mut production} => {
				production.last_mut().unwrap().progress += unit_production_rate * player.stats.bonuses.production_factor;
			  } _ => {
				  panicq!("bldg args in undefined state");
			  }
			}
		} ProdAction::FinProd {coord_add, unit_template} => {
			match bldgs[bldg_ind].args {
			  BldgArgs::CityHall {ref mut production, ..} | BldgArgs::GenericProducable {ref mut production} => {
				production.pop();
			  } _ => {
				 panicq!("bldg args in undefined state");
			  }}
			let owner_id = bldgs[bldg_ind].owner_id;
			add_unit(coord_add, owner_id == cur_player, unit_template, units, map_data, exs, bldgs, player, relations, logs, temps.units, &temps.nms, turn, rng);
		}
	}
}

fn find_commute_ind(bldg_ind_find: usize, commutes: &Vec<Commute>) -> Option<usize> {
	for (i, c) in commutes.iter().enumerate() {
		if let Some(bldg_ind) = c.bldg_ind {
			if bldg_ind == bldg_ind_find {
				return Some(i)
			}
		}
	}
	None
}

pub fn trim_cons(bldg_ind_recv: usize, bldgs: &mut Vec<Bldg>, pstats: &mut Stats){	
	while bldgs[bldg_ind_recv].cons() > bldgs[bldg_ind_recv].cons_capac(){
		let b_recv = &bldgs[bldg_ind_recv];
		
		// get a sending bldg ind (don't remove employment [residential connection])
		let mut bldg_ind_send = None;
		for b_recv_frm in b_recv.bldgs_recv_frm.iter().rev() {
			if b_recv_frm.zone_type != ZoneType::Residential {
				bldg_ind_send = b_recv_frm.bldg_ind;
				break;
			}
		}
		
		// rm commute frm sender
		let bldg_ind_send = bldg_ind_send.unwrap();
		let commute_rm = find_commute_ind(bldg_ind_recv, &bldgs[bldg_ind_send].bldgs_send_to);
		rm_commute_to(bldg_ind_send, commute_rm.unwrap(), bldgs, pstats);
	}
}

pub fn add_commute_to(bldg_ind_send: usize, bldg_ind_recv: usize, bldgs: &mut Vec<Bldg>, pstats: &mut Stats) {
	if let BldgType::Taxable(zone_send) = bldgs[bldg_ind_send].template.bldg_type {
	if let BldgType::Taxable(zone_recv) = bldgs[bldg_ind_recv].template.bldg_type {
		debug_assertq!(zone_send != ZoneType::Residential && (bldgs[bldg_ind_send].prod_capac() > bldgs[bldg_ind_send].n_sold()) ||
				zone_send == ZoneType::Residential && (bldgs[bldg_ind_send].n_residents() > bldgs[bldg_ind_send].n_sold()));
		debug_assertq!(bldgs[bldg_ind_send].owner_id == bldgs[bldg_ind_recv].owner_id);
		debug_assertq!(zone_send != ZoneType::Residential || zone_recv != ZoneType::Residential);
		// ^ otherwise this would be interpreted as a resident living there
		
		// add connection between bldgs
		bldgs[bldg_ind_recv].bldgs_recv_frm.push(Commute {bldg_ind: Some(bldg_ind_send), 
				zone_type: zone_send });
		
		bldgs[bldg_ind_send].bldgs_send_to.push(Commute {bldg_ind: Some(bldg_ind_recv), 
				zone_type: zone_recv });
		
		// record keeping
		if zone_send == ZoneType::Residential {
			let id = bldgs[bldg_ind_send].owner_id as usize;
			pstats.employed += 1;
			bldgs[bldg_ind_recv].population_update_taxable_upkeep(pstats);
		}
	}else {panicq!("no zone for recv");}
	}else {panicq!("no zone for send");}
}

pub fn rm_commute_to(bldg_ind_send: usize, commute_ind: usize, bldgs: &mut Vec<Bldg>, pstats: &mut Stats) {
	// rm frm sender
	let bldg_ind_recv = bldgs[bldg_ind_send].bldgs_send_to.swap_remove(commute_ind).bldg_ind.unwrap();
	debug_assertq!(bldgs[bldg_ind_send].owner_id == bldgs[bldg_ind_recv].owner_id);
	let b_recv = &mut bldgs[bldg_ind_recv];
	
	// find commute index to remove in recv bldg
	let commute_rm = find_commute_ind(bldg_ind_send, &b_recv.bldgs_recv_frm);
	
	// rm frm receiver
	b_recv.bldgs_recv_frm.swap_remove(commute_rm.unwrap());
	
	// cut production?
	while bldgs[bldg_ind_recv].prod_capac() < bldgs[bldg_ind_recv].prod() {
		let b_recv = &bldgs[bldg_ind_recv];
		debug_assertq!(b_recv.template.bldg_type != BldgType::Taxable(ZoneType::Residential));
		rm_commute_to(bldg_ind_recv, b_recv.bldgs_send_to.len()-1, bldgs, pstats);
	}
	
	trim_cons(bldg_ind_recv, bldgs, pstats);
	
	// record keeping
	if let BldgType::Taxable(zone_send) = bldgs[bldg_ind_send].template.bldg_type {
		if zone_send == ZoneType::Residential {
			pstats.employed -= 1;
			bldgs[bldg_ind_recv].population_update_taxable_upkeep(pstats);
		}
	}else{panicq!("bldg not taxable");}
}

// find all commutes to and from bldg_ind which use bldg_ind_old
// and set to bldg_ind.
pub fn update_commute_bldg_inds(bldg_ind: usize, bldg_ind_old: usize, bldgs: &mut Vec<Bldg>){	
	// all bldgs sending to this bldg
	for i in 0..bldgs[bldg_ind].bldgs_recv_frm.len() {
		if let Some(bldg_ind_u) = bldgs[bldg_ind].bldgs_recv_frm[i].bldg_ind {
			let bldgs_send_to = &mut bldgs[bldg_ind_u].bldgs_send_to;
			
			let c_ind = find_commute_ind(bldg_ind_old, bldgs_send_to).unwrap();
			bldgs_send_to[c_ind].bldg_ind = Some(bldg_ind);
		}
	}
	
	// all bldgs receiving from this bldg
	for i in 0..bldgs[bldg_ind].bldgs_send_to.len() {
		let bldg_ind_u = bldgs[bldg_ind].bldgs_send_to[i].bldg_ind.unwrap();
		// bldg_ind should never be none when sending. would imply sending to no bldg
		
		let bldgs_recv_frm = &mut bldgs[bldg_ind_u].bldgs_recv_frm;
		
		let c_ind = find_commute_ind(bldg_ind_old, bldgs_recv_frm).unwrap();
		bldgs_recv_frm[c_ind].bldg_ind = Some(bldg_ind);
	}
}

pub fn rm_all_commutes(bldg_ind: usize, bldgs: &mut Vec<Bldg>, player: &mut Player, map_sz: MapSz){
	// remove commutes where this bldg is receiving products
	
	// (go in reverse order to avoid unnecessary 
	//  copying of B.bldgs_recv_frm)
	while bldgs[bldg_ind].bldgs_recv_frm.len() != 0 {
		let b = &bldgs[bldg_ind];
		let commutes = &b.bldgs_recv_frm;
		let commute_last = commutes.last().unwrap();
		let bldg_ind_sender = commute_last.bldg_ind;
		
		// commute contains product or employment
		if let Some(bldg_ind_sender) = bldg_ind_sender {
			
			// residential zones only send employment, and not to other residential zones
			debug_assertq!(b.template.bldg_type != BldgType::Taxable(ZoneType::Residential) ||
				      commute_last.zone_type != ZoneType::Residential);
			
			let commutes_sender = &bldgs[bldg_ind_sender].bldgs_send_to;
			let commute_ind = find_commute_ind(bldg_ind, &commutes_sender);
			rm_commute_to(bldg_ind_sender, commute_ind.unwrap(), bldgs, &mut player.stats);
		
		// remove resident
		}else{
			debug_assertq!(b.template.bldg_type == BldgType::Taxable(ZoneType::Residential) &&
				      commute_last.zone_type == ZoneType::Residential);
			
			rm_resident(bldg_ind, bldgs, player, map_sz);
		}
	}
	
	// all employment and residents have been removed so no production should be remaining
	debug_assertq!(bldgs[bldg_ind].bldgs_send_to.len() == 0);
}

pub fn add_resident(bldg_ind: usize, bldgs: &mut Vec<Bldg>,
		zone_exs: &HashedMapZoneEx, pstats: &mut Stats, map_sz: MapSz) {
	let b = &mut bldgs[bldg_ind];
	debug_assertq!(b.template.bldg_type == BldgType::Taxable(ZoneType::Residential));
	debug_assertq!(b.n_residents() < b.template.resident_max);
	
	b.bldgs_recv_frm.push(Commute {bldg_ind: None, zone_type: ZoneType::Residential});
	
	let id = b.owner_id as usize;
	pstats.population += 1;
	b.population_update_taxable_upkeep(pstats);
	
	// update city hall population counters
	if let Some(zone_ex) = zone_exs.get(&return_zone_coord(b.coord, map_sz)) {
		match zone_ex.ret_city_hall_dist() {
			Dist::Is {bldg_ind: ch_bldg_ind, ..} |
			Dist::ForceRecompute {bldg_ind: ch_bldg_ind, ..} => {
				if let BldgArgs::CityHall {ref mut population, ..} = bldgs[ch_bldg_ind].args {
					*population += 1;
				}else{panicq!("expected city hall");}
			}
			Dist::NotInit | Dist::NotPossible {..} => {}
		}
	}
}

// needs zone_exs
pub fn rm_resident(bldg_ind: usize, bldgs: &mut Vec<Bldg>, player: &mut Player, map_sz: MapSz){
	let b = &mut bldgs[bldg_ind];
	debug_assertq!(b.template.bldg_type == BldgType::Taxable(ZoneType::Residential));
	let n_residents = b.n_residents();
	debug_assertq!(n_residents > 0);
	
	// find commute index to remove
	let mut commute_rm = None;
	for (i, b_recv_frm) in b.bldgs_recv_frm.iter().enumerate().rev() {
		if b_recv_frm.zone_type == ZoneType::Residential {
			commute_rm = Some(i);
			break;
		}
	}
	
	// if commute_rm not found, this purposely throws an error:
	b.bldgs_recv_frm.swap_remove(commute_rm.unwrap());
	
	// everyone was employed so one employee commute
	// must be removed...
	let n_employed = b.bldgs_send_to.len();
	if n_residents == n_employed {
		debug_assertq!(b.bldgs_send_to[n_employed-1].zone_type != ZoneType::Residential);
		// ^ residential bldgs shouldn't recv anything from other residential bldgs
		
		rm_commute_to(bldg_ind, n_employed-1, bldgs, &mut player.stats);
	}
	
	trim_cons(bldg_ind, bldgs, &mut player.stats);
	
	/////// record keeping
	let b = &mut bldgs[bldg_ind];
	player.stats.population -= 1;
	b.population_update_taxable_upkeep(&mut player.stats);
	
	// rm from city hall
	if let Some(zone_ex) = player.zone_exs.get(&return_zone_coord(b.coord, map_sz)) {
		match zone_ex.ret_city_hall_dist() {
			Dist::Is {bldg_ind: ch_bldg_ind, ..} |
			Dist::ForceRecompute {bldg_ind: ch_bldg_ind, ..} => {
				if let BldgArgs::CityHall {ref mut population, ..} = bldgs[ch_bldg_ind].args {
					*population -= 1;
				}else{panicq!("expected city hall");}
			}
			Dist::NotInit | Dist::NotPossible {..} => {}
		}
	}
}

/////////////
// meta args

#[derive(Clone, PartialEq)]
pub struct ProductionEntry<'ut,'rt> {
	pub production: &'ut UnitTemplate<'rt>,
	pub progress: u32
}

impl_saving!{ProductionEntry<'ut,'rt> {production, progress}}

#[derive(Clone, PartialEq)]
pub enum BldgArgs<'ut,'rt> {
	CityHall {
		tax_rates: Box<[u8]>, // for each zone type indexed by ZoneType
		production: Vec<ProductionEntry<'ut,'rt>>, // unit production
		population: u32, // updated w/: zone_ex.set_city_hall_dist() and add_resident(), rm_resident()
		nm: String
	},
	
	GenericProducable {
		production: Vec<ProductionEntry<'ut,'rt>> // unit production
	},
	None
}

impl BldgTemplate<'_,'_,'_> {
	pub fn available(&self, pstats: &Stats) -> bool {
		if !pstats.tech_met(&self.tech_req) || self.barbarian_only {return false;}
		if let Some(doctrine_req) = self.doctrine_req {
			return doctrine_req.bldg_reqs_met(pstats);
		}
		true
	}
}

