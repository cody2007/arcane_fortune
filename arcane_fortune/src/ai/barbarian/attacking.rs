//use crate::disp_lib::endwin;
use crate::units::{ActionType, Unit};
use crate::buildings::{Bldg, BldgType};
use crate::disp::{Coord};
use crate::map::{MapSz, MapData, StructureType};
use crate::player::*;
use crate::movement::{manhattan_dist};
use crate::gcore::hashing::HashedMapEx;
#[cfg(feature = "profile")]
use crate::gcore::profiling::*;
use crate::ai::set_target_attackable;
use super::MAX_BARBARIAN_SEARCH;

// finds targets for barbarians (everyone except barbarians)
// first starts searching for units, then bldgs, then walls
pub fn find_and_set_attack<'bt,'ut,'rt,'dt>(attacker_ind: usize, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, 
			bldgs: &Vec<Bldg>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData,
			players: &Vec<Player>, map_sz: MapSz) -> Option<ActionType<'bt,'ut,'rt,'dt>> {
	#[cfg(feature="profile")]
	let _g = Guard::new("find_and_set_attack");
	
	let attacker_u = &units[attacker_ind];
	let attacker_c = Coord::frm_ind(attacker_u.return_coord(), map_sz);
	
	struct Dist<'bt,'ut,'rt,'dt> {
		action_type: ActionType<'bt,'ut,'rt,'dt>,
		manhattan_dist: usize
	}
	
	// add to dists if player is not a barbarian and it is close enough
	macro_rules! add_to_dists{($dists: expr, $attackee_coord: expr, $attackee_owner: expr) => {
		if !players[$attackee_owner as usize].ptype.is_barbarian() {
			let dist = manhattan_dist(Coord::frm_ind($attackee_coord, map_sz), attacker_c, map_sz);
			if dist < MAX_BARBARIAN_SEARCH {
				$dists.push(Dist {
					action_type: ActionType::Attack {
						attack_coord: Some($attackee_coord),
						attackee: Some($attackee_owner),
						ignore_own_walls: true
					},
					manhattan_dist: dist
				});
			}
		}
	};}
	
	// attack closest target in $dists
	macro_rules! attack_closest{($dists: expr) => {
		// sort from least to greatest (first entry being least)
		$dists.sort_unstable_by(|a, b| a.manhattan_dist.partial_cmp(&b.manhattan_dist).unwrap());
		
		// find closest attackable unit
		for dist in $dists.iter() {
			if set_target_attackable(&dist.action_type, attacker_ind, true, MAX_BARBARIAN_SEARCH, units, bldgs, exs, map_data, map_sz) {
				return Some(dist.action_type.clone());
			}
		}
	};}

	{ // attack closest unit
		#[cfg(feature="profile")]
		let _g = Guard::new("attack closest unit");
		
		let mut dists = Vec::with_capacity(units.len());
		for attackee_u in units.iter() {
			add_to_dists!(dists, attackee_u.return_coord(), attackee_u.owner_id);
		}
		attack_closest!(dists);
	}
	
	{ // attack closest bldg
		#[cfg(feature="profile")]
		let _g = Guard::new("attack closest bldg");

		let mut dists = Vec::with_capacity(bldgs.len());
		for attackee_b in bldgs.iter() {
			if let BldgType::Gov(_) = attackee_b.template.bldg_type {
				add_to_dists!(dists, attackee_b.coord, attackee_b.owner_id);
			}
		}
		attack_closest!(dists);
	}
	
	{ // attack closest wall
		#[cfg(feature="profile")]
		let _g = Guard::new("attack closest wall");
		
		let mut dists = Vec::with_capacity(exs.last().unwrap().len());
		for (coord, ex) in exs.last().unwrap().iter() {
			if ex.actual.ret_structure() == Some(StructureType::Wall) {
				add_to_dists!(dists, *coord, ex.actual.owner_id.unwrap());
			}
		}
		attack_closest!(dists);
	}
	
	return None;
}

