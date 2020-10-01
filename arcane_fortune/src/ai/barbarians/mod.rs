use std::cmp::max;
use crate::disp::{CGRAY, DispChars};
use crate::disp_lib::*;
use crate::units::{ActionMeta, ActionType, Unit, UnitTemplate, add_unit, WARRIOR_NM, square_clear, Quad, ARCHER_NM};
use crate::buildings::{BldgTemplate, Bldg, BARBARIAN_CAMP_NM, add_bldg, BldgArgs, ProductionEntry};
use crate::disp::ScreenSz;
use crate::map::{MapSz, PlayerType, MapData, Owner, compute_zooms_coord, Stats,
		     RecompType, StructureType, Nms, PersonName};
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::saving::{N_UNIT_PLACEMENT_ATTEMPTS, SmSvType, add_owner_vars, print_attempt_update};
use crate::zones::StructureData;
use crate::doctrine::DoctrineTemplate;
use crate::gcore::{Log, Relations, Bonuses, XorState};
use super::{AIState, ATTACK_SEARCH_TIMEOUT};
use crate::map::utils::ExFns;
use crate::tech::TechTemplate;
use crate::resources::ResourceTemplate;
#[cfg(feature = "profile")]
use crate::gcore::profiling::*;
use crate::nn::{TxtGenerator, TxtCategory};
use crate::ai::set_target_attackable;
use crate::localization::Localization;

pub mod vars; pub use vars::*;
pub mod attacking; pub use attacking::*;

pub const MAX_BARBARIAN_SEARCH: usize = 300;

impl BarbarianState {
	pub fn plan_actions<'bt,'ut,'rt,'dt>(&mut self, owner: &Owner, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
			map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
			stats: &mut Vec<Stats>, unit_templates: &'ut Vec<UnitTemplate<'rt>>,
			owners: &Vec<Owner>, map_sz: MapSz, rng: &mut XorState, turn: usize) {
		if owner.player_type != PlayerType::Barbarian {return;}
		if !stats[owner.id as usize].alive {return;}
		
		// if barbarian camp is not producing anything, set it to produce a new archer
		if !self.max_units() {
			let archer_t = UnitTemplate::frm_str(ARCHER_NM, unit_templates);
			
			let b = &mut bldgs[self.camp_ind];
			debug_assertq!(b.owner_id == owner.id);
			debug_assertq!(b.template.nm[0] == BARBARIAN_CAMP_NM);
			
			// if not producing anything
			if let BldgArgs::GenericProducable {ref mut production} = b.args {
				if production.len() == 0 {
					production.push(ProductionEntry {
						production: archer_t,
						progress: 0
					});
				}
			}
		}
		
		// launch attacks
		let mut prev_target = None;
		
		const CHANCE_SKIP: f32 = 0.025;
		
		'attacker_loop: for attacker_ind in self.attacker_inds.iter() {
			if rng.gen_f32b() < CHANCE_SKIP {continue;}
			let exf = exs.last().unwrap();
			let u = &mut units[*attacker_ind];
			
			// if action already set, check to make sure a unit still exists
			// if not, find new action.
			// if action is fortification, check if we should search again yet based on when it was fortified
			if let Some(action) = &u.action.last() {
				if let ActionType::Attack {attack_coord: Some(coord), ..} = &action.action_type {
					if let Some(ex) = exf.get(&coord) {
						let still_attackable = || {
							if ex.actual.structure != None || ex.bldg_ind != None {
								return true;
							}else if let Some(unit_inds) = &ex.unit_inds {
								if unit_inds.len() > 0 {
									return true; // don't change action
								}
							}
							false
						};
						if still_attackable() {
							continue 'attacker_loop;
						// remove previous target
						}else{
							u.action.pop();
						}
					}
				}else if let ActionType::Fortify {turn} = action.action_type {
					if (turn + ATTACK_SEARCH_TIMEOUT) >= turn {continue 'attacker_loop;}
				}else{panicq!("unknown barbarian attacker action");}
			}
			
			// check if we can re-use previous target
			if let Some(target) = &prev_target {
				if set_target_attackable(target, *attacker_ind, true, MAX_BARBARIAN_SEARCH, units, bldgs, exs, map_data, map_sz) {
					continue 'attacker_loop;
				}
			}
			// find new target
			if let Some(new_target) = find_and_set_attack(*attacker_ind, units, bldgs, exs, map_data, owners, map_sz) {
				prev_target = Some(new_target);
				
			// no new target, update as failed attempt
			}else{
				units[*attacker_ind].action = vec!{ActionMeta::new(ActionType::Fortify {turn}); 1};
			}
		}
	}
}

pub fn place_barbarians<'bt,'ut,'rt,'dt>(attempt: &mut usize, player: SmSvType, city_h: isize, city_w: isize,
		map_coord_y: u64, map_coord_x: u64, bonuses: &Bonuses,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, owners: &mut Vec<Owner>,
		map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, zone_exs_owners: &mut Vec<HashedMapZoneEx>,
		doctrine_templates: &'dt Vec<DoctrineTemplate>,
		unit_templates: &'ut Vec<UnitTemplate<'rt>>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
		tech_templates: &Vec<TechTemplate>, resource_templates: &'rt Vec<ResourceTemplate>,
		stats: &mut Vec<Stats<'bt,'ut,'rt,'dt>>, relations: &mut Relations, ai_states: &mut Vec<Option<AIState<'bt,'ut,'rt,'dt>>>,
		barbarian_states: &mut Vec<Option<BarbarianState>>, logs: &mut Vec<Log>,
		nms: &Nms, rng: &mut XorState, map_sz: MapSz, sz_prog: &MapSz, screen_sz: &mut ScreenSz,
		turn: usize, disp_chars: &DispChars, l: &Localization, txt_gen: &mut TxtGenerator, d: &mut DispState) -> bool {
	let warrior_t = UnitTemplate::frm_str(WARRIOR_NM, unit_templates);
	
	///// find bldg template for camp, set sz
	const WALL_DIST_I: isize = 2; // from camp
	const WALL_DIST_J: isize = 4;
	let mut camp_bt = &bldg_templates[0];
	let camp_sz;
	let camp_w;
	let camp_h;
	{
		let mut camp_bt_found = false;
		for bt in bldg_templates.iter() {
			if bt.nm[0] == BARBARIAN_CAMP_NM {
				camp_bt = bt;
				camp_bt_found = true;
				break;
			}
		}
		
		assertq!(camp_bt_found, "Could not find required building \"{}\" in configuration file.", BARBARIAN_CAMP_NM);
		
		camp_sz = max(camp_bt.sz.h + (WALL_DIST_I*2) as usize + 2, camp_bt.sz.w + (WALL_DIST_J*2) as usize + 2);
		camp_w = camp_bt.sz.w as isize;
		camp_h = camp_bt.sz.h as isize;
	}

	for _ in 0..rng.usize_range(1,5) {
		'barbarian_attempt: loop {
			const MAX_BARBARIAN_DIST: isize = 300;
			const MIN_BARBARIAN_DIST: usize = 30;
			
			macro_rules! try_again{() => {
				if *attempt > N_UNIT_PLACEMENT_ATTEMPTS { // try a new map
					return false;
				}else{ // try a new barbarian placement
					*attempt += 1;
					print_attempt_update(*attempt, player, &sz_prog, screen_sz, disp_chars, l, d);
					continue 'barbarian_attempt;
				}
			};};
				
			// generate location
			let by;
			let bx;
			{
				const ALPHA: f32 = 1./10.;
				let r = rng.gen_f32b();
				
				let y = map_coord_y as isize;
				let x = map_coord_x as isize;
				
				// upper left to lower left section
				if r < (3.*ALPHA) {
					by = rng.isize_range(y - MAX_BARBARIAN_DIST,    y + city_h + MAX_BARBARIAN_DIST);
					bx = rng.isize_range(x - MAX_BARBARIAN_DIST,    x);
				
				// upper right to lower right section
				}else if r < (6.*ALPHA) {
					by = rng.isize_range(y - MAX_BARBARIAN_DIST,    y + city_h + MAX_BARBARIAN_DIST);
					bx = rng.isize_range(x + city_w,    x + city_w + MAX_BARBARIAN_DIST);
					
				// top center
				}else if r < (7.*ALPHA) {
					by = rng.isize_range(y - MAX_BARBARIAN_DIST,   y);
					bx = rng.isize_range(x,    x + city_w);
					
				// bottom center
				}else{
					by = rng.isize_range(y + city_h,   y + city_h + MAX_BARBARIAN_DIST);
					bx = rng.isize_range(x,   x + city_w);
				}
				
				if ((by - y).abs() + (bx - x).abs()) < (MIN_BARBARIAN_DIST as isize) {try_again!();}
			}
			
			if let Some(map_coord_block) = map_sz.coord_wrap(by, bx) {
				// check if there's sufficient space to fit entire camp
				if square_clear(map_coord_block, ScreenSz{h: camp_sz, w: camp_sz, sz: 0}, Quad::Lr, map_data, exs.last().unwrap()) == None {
					try_again!();
				}
				
				let ruler_nm = PersonName {first: String::from("Conan"), last: String::from("Cimmeria")};
				
				let barbarian_id = owners.len();
				owners.push(Owner {id: barbarian_id as SmSvType,
						color: CGRAY,
						nm: "Barbarian".to_string(),
						gender_female: false,
						doctrine_advisor_nm: ruler_nm.clone(),
						crime_advisor_nm: ruler_nm.clone(),
						pacifism_advisor_nm: ruler_nm.clone(),
						health_advisor_nm: ruler_nm.clone(),
						unemployment_advisor_nm: ruler_nm.clone(),
						ruler_nm,
						
						city_nm_theme: 0,
						motto: txt_gen.gen_str(TxtCategory::Negative),
						player_type: PlayerType::Barbarian
				});
				add_owner_vars(bonuses, zone_exs_owners, stats, relations, barbarian_states, ai_states, owners, tech_templates, resource_templates, doctrine_templates, map_data);
				
				// location to put bldg
				let map_coord = map_sz.coord_wrap(by + WALL_DIST_I, bx + WALL_DIST_J).unwrap();
				if !add_bldg(map_coord, barbarian_id as SmSvType, bldgs, camp_bt,
						 None, bldg_templates, doctrine_templates, map_data, exs,
						 zone_exs_owners, stats, &mut ai_states[barbarian_id], owners, nms, turn, logs, rng) {
					panicq!("failed to add barbarian bldg");
				}
				
				bldgs.last_mut().unwrap().construction_done = None; // finish construction
				let barbarian_state_opt = barbarian_states.last_mut().unwrap();
				*barbarian_state_opt = Some(BarbarianState{
						camp_ind: bldgs.len() - 1,
						attacker_inds: Vec::new(),
						defender_inds: Vec::new()
				});
				
				////////// add walls
				macro_rules! add_wall{($i: expr, $j: expr, $d: expr) => {
					let coord = map_sz.coord_wrap(by + $i, bx + $j).unwrap();
					let exf = exs.last_mut().unwrap();
					exf.create_if_empty(coord);
					let ex = exf.get_mut(&coord).unwrap();
					ex.actual.structure = Some(StructureData {
							structure_type: StructureType::Wall,
							orientation: $d,
							health: std::u8::MAX
					});
					ex.actual.owner_id = Some(barbarian_id as SmSvType);
					compute_zooms_coord(coord, RecompType::Bldgs(bldgs, bldg_templates, zone_exs_owners), map_data, exs, owners);
				};};
				
				// wall gate
				let c = rng.isize_range(0, 4);
				let gate_loc = if c == 0 || c == 1 { // top and bottom
					rng.isize_range(1, WALL_DIST_J*2 + camp_w)
				}else if c == 2 || c == 3 { // left and right
					rng.isize_range(1, WALL_DIST_I*2 + camp_h)
				}else{panicq!("invalid random value");};
				
				// top and bottom walls
				for col in 0..=(WALL_DIST_J*2 + camp_w) {
					if c != 0 || gate_loc != col { // top
						add_wall!(0, col, '|');
					}
					if c != 1 || gate_loc != col { // bottom
						add_wall!(WALL_DIST_I*2 + camp_h, col, '|');
					}
				}
				
				// side walls
				for row in 1..(WALL_DIST_I*2 + camp_h) {
					if c != 2 || gate_loc != row { // left
						add_wall!(row, 0, '-');
					}
					if c != 3 || gate_loc != row { // right
						add_wall!(row, WALL_DIST_J*2 + camp_w, '-');
					}
				}
				
				// place units
				{
					macro_rules! add_u{($coord:expr, $player:expr, $type:expr) => (
					add_unit($coord, $player, false, $type, units, map_data, exs, bldgs, stats, relations, logs, barbarian_state_opt, &mut ai_states[$player as usize], unit_templates, owners, nms, turn, rng););};
					
					if c == 0 { // top
						let coord = map_sz.coord_wrap(by, bx + gate_loc).unwrap();
						add_u!(coord, barbarian_id as SmSvType, warrior_t);
					}else if c == 1 { // bottom
						let coord = map_sz.coord_wrap(by + WALL_DIST_I*2 + camp_h, bx + gate_loc).unwrap();
						add_u!(coord, barbarian_id as SmSvType, warrior_t);
					}else if c == 2 { // left
						let coord = map_sz.coord_wrap(by + gate_loc, bx).unwrap();
						add_u!(coord, barbarian_id as SmSvType, warrior_t);
					}else{ // right
						let coord = map_sz.coord_wrap(by + gate_loc, bx + WALL_DIST_J*2 + camp_w).unwrap();
						add_u!(coord, barbarian_id as SmSvType, warrior_t);
					}
				}
				
				break 'barbarian_attempt;
			} // valid coordinate
			
			try_again!();
		} // barbarian placement loop
	} // place `n_place` barbarians
	true
}
