use super::*;
use crate::ai::CityState;
use crate::player::{Nms, Stats};
use std::collections::VecDeque;

#[derive(Clone, PartialEq)]
pub struct Brigade<'bt,'ut,'rt,'dt> {
	pub nm: String,
	pub unit_inds: Vec<usize>,
	pub build_list: VecDeque<ActionMeta<'bt,'ut,'rt,'dt>>,
	pub repair_sector_walls: Option<String> // string is the sector name
}

impl_saving!{Brigade<'bt,'ut,'rt,'dt> {nm, unit_inds, build_list, repair_sector_walls}}

impl Brigade<'_,'_,'_,'_> {
	pub fn has_buildable_actions(&self, units: &Vec<Unit>) -> bool {
		self.unit_inds.iter().any(|&ind| {
			let u = &units[ind];
			u.template.nm[0] == WORKER_NM ||
			u.template.repair_wall_per_turn != None
		})
	}
}

impl <'bt,'ut,'rt,'dt>Stats<'bt,'ut,'rt,'dt> {
	pub fn new_brigade_nm(&self, nms: &Nms, rng: &mut XorState) -> String {
		let mut nm_suffix = String::new();
		let brigade_nm_inds = rng.inds(nms.brigades.len());
		for i in 0..1000 {
			for brigade_nm_ind in brigade_nm_inds.iter() {
				let nm_txt = format!("{}{}", nms.brigades[*brigade_nm_ind], nm_suffix);
				if !self.brigades.iter().any(|brigade| brigade.nm == nm_txt) {
					return nm_txt;
				}
			}
			nm_suffix = format!(" {}", i);
		}
		panicq!("could not create brigade name; n_brigades: {}", self.brigades.len());
	}
	
	pub fn rm_empty_brigades(&mut self) {
		self.brigades.retain(|brigade| brigade.unit_inds.len() != 0);
	}
	
	pub fn rm_unit_frm_brigade(&mut self, unit_ind: usize) {
		for (brigade_ind, brigade) in self.brigades.iter_mut().enumerate() {
			for (list_ind, brigade_unit_ind) in brigade.unit_inds.iter().enumerate() {
				if *brigade_unit_ind == unit_ind {
					// remove entire brigade if no units remaining
					if brigade.unit_inds.len() == 1 {
						self.brigades.swap_remove(brigade_ind);
					// only remove unit
					}else{
						brigade.unit_inds.swap_remove(list_ind);
					}
					return;
				}
			}
		}
	}
	
	pub fn chg_brigade_unit_ind(&mut self, frm_ind: usize, to_ind: usize) {
		for brigade in self.brigades.iter_mut() {
			for brigade_unit_ind in brigade.unit_inds.iter_mut() {
				if *brigade_unit_ind == frm_ind {
					*brigade_unit_ind = to_ind;
					break;
				}
			}
		}
		// the unit shouldn't remain in any brigades
		debug_assert!(!self.brigades.iter().any(|brigade| brigade.unit_inds.contains(&frm_ind)));
	}
	
	pub fn unit_brigade_nm(&self, unit_ind: usize) -> Option<&str> {
		for brigade in self.brigades.iter() {
			if brigade.unit_inds.contains(&unit_ind) {
				return Some(&brigade.nm);
			}
		}
		None
	}
	
	pub fn brigade_frm_nm(&self, brigade_nm: &String) -> &Brigade {
		if let Some(brigade) = self.brigades.iter().find(|b| b.nm == *brigade_nm) {
			brigade
		}else{panicq!("could not find brigade `{}`", brigade_nm);}
	}
	
	pub fn brigade_frm_nm_mut(&mut self, brigade_nm: &String) -> &mut Brigade<'bt,'ut,'rt,'dt> {
		if let Some(brigade) = self.brigades.iter_mut().find(|b| b.nm == *brigade_nm) {
			brigade
		}else{panicq!("could not find brigade `{}`", brigade_nm);}
	}
}

impl <'bt,'ut,'rt,'dt>ActionMeta<'bt,'ut,'rt,'dt> {
	// used when setting action from build list
	pub fn set_action_meta(self, unit_ind: usize, is_cur_player: bool, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, 
			pstats: &mut Stats<'bt,'ut,'rt,'dt>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, map_data: &mut MapData,
			min_city_opt: Option<&CityState<'bt,'ut,'rt,'dt>>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, relations: &mut Relations, logs: &mut Vec<Log>, map_sz: MapSz, turn: usize) {
		let u = &mut units[unit_ind];
		
		// if repairing wall, start coord is stored in the action type, else use `path_coords`
		let start_coord = match &self.action_type {
			ActionType::WorkerRepairWall {wall_coord, ..} => {
				wall_coord.unwrap()
			} ActionType::WorkerZone {start_coord, ..} |
			  ActionType::WorkerRmZonesAndBldgs {start_coord, ..} => {
				start_coord.unwrap()
			} _ => {
				*self.path_coords.last().unwrap()
			}
		};
		
		// not at start, move to start
		if u.return_coord() != start_coord {
			let start_coord = Coord::frm_ind(start_coord, map_sz);
			
			let mut max_search_depth = 300*4;// *5;
			
			// if distance is far, set unit to be just outside the wall, then search a bit further with astar
			{
				if let Some(min_city) = min_city_opt {
					if !min_city.exit_city(&mut max_search_depth, unit_ind, is_cur_player, true, start_coord, 
								Coord::frm_ind(units[unit_ind].return_coord(), map_sz), pstats, map_data, exs, units, bldgs, relations, logs, map_sz, turn) {
						return;
					}
				}
			}
			
			// if repairing wall, action type must be WorkerRepairWall or else the move search will
			// never return the path (moving onto a wall is normally not allowed)
			let action_meta = if let ActionType::WorkerRepairWall {..} = &self.action_type {
				self.clone()
			}else{
				ActionMeta::new(ActionType::Mv)
			};
			
			let mut action_iface = ActionInterfaceMeta {
				action: action_meta,
				unit_ind: Some(unit_ind),
				max_search_depth,
				start_coord: Coord::frm_ind(units[unit_ind].return_coord(), map_sz), // starting location of unit. we want to then move to `start_coord` which is the starting coord of the action
				movement_type: units[unit_ind].template.movement_type,
				movable_to: &movable_to
			};
			
			action_iface.update_move_search(start_coord, map_data, exs, MvVars::NonCivil{units, start_owner: pstats.id, blind_undiscov: None}, bldgs);
			
			// move possible, send unit on their way
			if action_iface.action.path_coords.len() > 0 {
				let u = &mut units[unit_ind];
				
				// if repairing wall, do not actually move onto wall (remove final coord from path)
				if let ActionType::WorkerRepairWall {..} = &self.action_type {
					let path_start_coord = action_iface.action.path_coords.remove(0);
					debug_assertq!(Coord::frm_ind(path_start_coord, map_sz) == start_coord);
				}
				
				u.action.push(self); // scheduled action
				u.action.push(action_iface.action); // move to start of scheduled action
			// not possible
			}else{
				return;
			}
		// create road or zone
		}else{
			u.action.push(self);
		}
	}
}

