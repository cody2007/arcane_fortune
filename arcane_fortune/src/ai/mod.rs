use std::cmp::{min, max, Ordering};
use crate::saving::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::gcore::rand::XorState;
use crate::units::{ActionMeta, ActionType};
use crate::map::{StructureType, MapSz, ZoneType, ZoomInd, MapType};
use crate::player::Stats;
use crate::disp::*;
use crate::units::*;
use crate::buildings::*;
use crate::disp_lib::*;
use crate::map::MapData;
use crate::gcore::hashing::HashedMapEx;
use crate::gcore::*;
use crate::player::Player;
use crate::movement::{manhattan_dist, manhattan_dist_components, MvVars, movable_to};
use crate::containers::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

pub const AI_MAX_SEARCH_DEPTH: usize = 300*4*2; // 300*4 was not enough to move to the gate

pub mod config; pub use config::*;
pub mod common; pub use common::*; // `AIState` common between nobility and empire AIs
pub mod empire; pub use empire::*; // normal AI country
pub mod nobility; pub use nobility::*;
pub mod barbarian; pub use barbarian::*;

#[derive(Clone, PartialEq, Copy)]
pub struct AIPersonality {
	pub friendliness: f32, // negative friendliness is agression; range [-1:1]
	// ^ (1 - friendliness) -> proportionate to war declaration probability
	pub spirituality: f32, // negative spirituality is scientific; range [-1:1]
	// ^ (1 - spirituality)/2 -> probability of scientific buildings
}

impl_saving!{AIPersonality {friendliness, spirituality} }

// returns true on success
pub fn set_target_attackable<'bt,'ut,'rt,'dt>(target: &ActionType<'bt,'ut,'rt,'dt>, attacker_ind: usize,
		clear_action_que: bool, max_search_depth: usize,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
			map_data: &mut MapData, map_sz: MapSz) -> bool {
	#[cfg(feature="profile")]
	let _g = Guard::new("set_target_attackable");
	
	if let ActionType::Attack {attack_coord: Some(coord), attackee: Some(attackee_unw), ..} = &target {
		let u = &units[attacker_ind];
		debug_assertq!(u.owner_id != *attackee_unw);
		debug_assertq!(u.action.len() == 0, "action len {} nm {} action type {}",
				u.action.len(), u.template.nm[0], u.action.last().unwrap().action_type);
		
		let coord = Coord::frm_ind(*coord, map_sz);
		
		let u = &units[attacker_ind];
		let mut action_iface = ActionInterfaceMeta {
			action: ActionMeta::new(target.clone()),
			unit_ind: Some(attacker_ind),
			max_search_depth,
			start_coord: Coord::frm_ind(u.return_coord(), map_sz),
			movement_type: u.template.movement_type,
			movable_to: &movable_to
		};
		
		action_iface.update_move_search(coord, map_data, exs, MvVars::NonCivil{units, start_owner: units[attacker_ind].owner_id, blind_undiscov: None}, bldgs);
		
		// move possible, send unit on their way
		return if action_iface.action.path_coords.len() > 0 {
			let u = &mut units[attacker_ind];
			if clear_action_que {u.action.clear();}
			u.action.push(action_iface.action);
			////////// dbg
			/*{
				let c = Coord::frm_ind(u.return_coord(), map_sz);
				let cf = Coord::frm_ind(u.action.last().unwrap().path_coords[0], map_sz);
				printlnq!("start coord {} {} path_coords.len {}  path_coords last {} {}", c.y, c.x, u.action.last().unwrap().path_coords.len(), cf.y, cf.x);
				printlnq!("actions_req {}", u.action.last().unwrap().actions_req);
			}*/
			////////
			u.set_attack_range(map_data, exs.last().unwrap(), map_sz);
			//////////////
			/*if let ActionType::Attack {attack_coord, attackee, ..} = u.action.last().unwrap().action_type {
				printlnq!("attack_coord {}, attackee {}", attack_coord.unwrap(), attackee.unwrap());
			}
			printlnq!("ret true");*/
			true
		}else {false}; // <- move not possible
	}else{
		panicq!("invalid input to is_target_attackable()");
	}
}

