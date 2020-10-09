use std::hash::{BuildHasherDefault};
use std::collections::HashMap;

use crate::gcore::hashing::{HashedMapEx, HashStruct64};
use crate::map::*;
use crate::units::*;
use crate::buildings::*;
use crate::gcore::{Relations, Log};
use crate::ai::{AIState, BarbarianState};
use crate::disp::*;
use crate::disp::menus::FindType;
use crate::disp_lib::*;
use crate::player::Player;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

mod utils;
mod movable_to;
mod find_closest_square;

pub use utils::*;
pub use movable_to::*;
pub use find_closest_square::*;

// start_mv_mode(): 'action' is the action, if any, to be performed after movement
pub fn unit_action_iface<'f,'bt,'ut,'rt,'st>(start_coord: Coord, action_type: ActionType<'bt,'ut,'rt,'st>, unit_ind: usize, 
		units: &Vec<Unit>) -> ActionInterfaceMeta<'f,'bt,'ut,'rt,'st> {
	ActionInterfaceMeta {
		action: ActionMeta::new(action_type), 
		unit_ind: Some(unit_ind),
		max_search_depth: 5000,
		start_coord,
		movement_type: units[unit_ind].template.movement_type,
		movable_to: &movable_to
	}
}

impl <'f,'bt,'ut,'rt,'st>IfaceSettings<'f,'bt,'ut,'rt,'st> {
	// individual actions:
	//	should be assigned only to one unit at a time (no applying these to every unit in a brigade)
	pub fn start_individual_mv_mode(&mut self, action_type: ActionType<'bt,'ut,'rt,'st>,
			unit_inds: &Vec<usize>, units: &Vec<Unit>, map_data: &MapData) -> bool {
		if !self.add_action_to.is_none() {return false;} // the player is already moving something
		
		if let Some(unit_ind) = unit_inds.first() {
			debug_assertq!(unit_inds.len() == 1);
			self.add_action_to = AddActionTo::IndividualUnit {
				action_iface: unit_action_iface(Coord::frm_ind(units[*unit_ind].return_coord(), *map_data.map_szs.last().unwrap()), action_type.clone(), *unit_ind, units)
			};
			return true;
		}
		false
	}
	
	// broadcastable actions:
	// 	can be assigned to all units in brigade at once (also can be assigned to individual units)
	pub fn start_broadcastable_mv_mode(&mut self, action_type: ActionType<'bt,'ut,'rt,'st>,
			unit_inds: &Vec<usize>, units: &Vec<Unit>, map_data: &MapData) {
		// broadcast to all in brigade
		if let AddActionTo::AllInBrigade {ref mut action_ifaces, ..} = self.add_action_to {
			if !action_ifaces.is_none() {return;} // already started a moving units...
			
			let mut action_ifaces_add = Vec::with_capacity(units.len());
			for unit_ind in unit_inds.iter() {
				action_ifaces_add.push(unit_action_iface(Coord::frm_ind(units[*unit_ind].return_coord(), *map_data.map_szs.last().unwrap()), action_type.clone(), *unit_ind, units));
			}
			
			*action_ifaces = Some(action_ifaces_add);
		// start moving one unit only
		}else{
			self.start_individual_mv_mode(action_type, unit_inds, units, map_data);
		}
	}
	
	// build list actions:
	//	can be added to a brigade's build list but not be assigned simultanously to multiple units at once (also can be assigned to individual units)
	pub fn start_build_mv_mode(&mut self, action_type: ActionType<'bt,'ut,'rt,'st>,
			unit_inds: &Vec<usize>, units: &Vec<Unit>, map_data: &MapData) {
		// add to brigade build list
		if let AddActionTo::BrigadeBuildList {ref mut action, ..} = self.add_action_to {
			*action = Some(ActionMeta::new(action_type));
		// start moving one unit only
		}else{
			self.start_individual_mv_mode(action_type, unit_inds, units, map_data);
		}
	}
}

// travel to dest_bldg_ind if set, using only zones and roads
// returns false when no roads around start_coord
pub fn start_civil_mv_mode<'f,'bt,'ut,'rt,'st>(start_coord: u64, map_data: &mut MapData,
		exf: &HashedMapEx, map_sz: MapSz) -> Option<ActionInterfaceMeta<'f,'bt,'ut,'rt,'st>> {
	
	if let Some(start_coord_use) = find_closest_road(start_coord, map_data, exf, map_sz){
		return Some( ActionInterfaceMeta {
			action: ActionMeta::new(ActionType::CivilianMv),
			unit_ind: None,
			max_search_depth: 400,
			start_coord: Coord::frm_ind(start_coord_use, map_sz),
			movement_type: MovementType::Land,
			movable_to: &civil_movable_to});
	}
	None
}

#[derive(Copy,Clone)]
struct Node { // will be indexed by the coordinate	
	cur_dist: f32, // dist from start to coord
	sum_dist: f32, // cur + set
	prev_coord: u64,
}

// move from iface_settings.action_iface.start_coord to cursor
// Note: when action == WorkerBuildRoad, we do not use roads
// when this function returns false, do not let user move the cursor with the mouse
//	(condition where the user is moving the unit with the mouse and the requested position is
//	 not movable)
impl <'f,'bt,'ut,'rt,'dt>IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	pub fn update_move_search_ui(&mut self, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			relations: &mut Relations, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			logs: &mut Vec<Log>, map_sz: MapSz, turn: usize, d: &mut DispState) {
		#[cfg(feature="profile")]
		let _g = Guard::new("update_move_search_ui");
		
		//endwin();
		
		if self.zoom_ind != map_data.max_zoom_ind() && self.zoom_ind != ZOOM_IND_ROOT {return;} // paths only computed at these zoom levels for now
		let mut end_coord = self.cursor_to_map_coord(map_data); // has to occur outside of loop due to immutable borrow
		
		if self.zoom_ind != map_data.max_zoom_ind() {
			end_coord = end_coord.to_zoom(self.zoom_ind, map_data.max_zoom_ind(), &map_data.map_szs);
		}
		
		macro_rules! update_action_iface{($action_iface: expr) => {
			let unit_ind = $action_iface.unit_ind.unwrap();
			let mv_vars = MvVars::NonCivil {
				units,
				start_owner: units[unit_ind].owner_id,
				blind_undiscov: Some(&players[self.cur_player as usize].stats.land_discov)
			};
			
			$action_iface.action.action_meta_cont = None; // clear out path on zoomed-out map
			$action_iface.update_move_search(end_coord, map_data, exs, mv_vars, bldgs);
		};};
		
		match &mut self.add_action_to {
			// no unit in particular is being moved
			AddActionTo::None | AddActionTo::NoUnit {..} |
			AddActionTo::BrigadeBuildList {..} => {}
			
			AddActionTo::IndividualUnit {ref mut action_iface} => {
				update_action_iface!(action_iface);
				
				// move unit with cursor
				if let ActionType::MvWithCursor = &action_iface.action.action_type {
					let unit_ind = action_iface.unit_ind.unwrap();
					let u = &mut units[unit_ind];
					if self.zoom_ind == map_data.max_zoom_ind() {
						if action_iface.action.path_coords.len() > 0 {
							u.action = vec![action_iface.action.clone()];
							
							let mut disband_unit_inds = Vec::new();
							mv_unit(unit_ind, true, units, map_data, exs, bldgs, players, relations, map_sz, DelAction::Record(&mut disband_unit_inds), logs, turn);
							
							// ex. if unit boards a boat
							if disband_unit_inds.contains(&unit_ind) {
								self.add_action_to = AddActionTo::None;
							}else{
								let u = &mut units[unit_ind];
								
								// no actions remain
								if u.actions_used.is_none() {
									u.action.pop();
									self.add_action_to = AddActionTo::None;
									self.set_auto_turn(AutoTurn::Off, d);
									// \/ update move search should be false otherwise there will be infinite recursion
									self.center_on_next_unmoved_menu_item(false, FindType::Coord(u.return_coord()), map_data, exs, units, bldgs, relations, players, logs, turn, d);
									self.ui_mode = UIMode::MvWithCursorNoActionsRemainAlert {unit_ind};
									return;
								}
								
								// if the cursor is moved again, start the search from wherever the unit is
								action_iface.start_coord = Coord::frm_ind(u.return_coord(), map_sz);
							}
							
							disband_units(disband_unit_inds, self.cur_player, units, map_data, exs, players, relations, map_sz, logs, turn);
						}else{
							// \/ update move search should be false otherwise there will be infinite recursion
							self.center_on_next_unmoved_menu_item(false, FindType::Coord(u.return_coord()), map_data, exs, units, bldgs, relations, players, logs, turn, d);
						}
					}
				}
			}
			
			AddActionTo::AllInBrigade {ref mut action_ifaces, ..} => {
				if let Some(ref mut action_ifaces) = action_ifaces {
					for action_iface in action_ifaces.iter_mut() {
						update_action_iface!(action_iface);
					}
				}
			}
		}
	}
}

// add_neighbors_to_list(): add neighbors of prev_coord
// cur_dist_prev is distance from start_y,start_x to i,j
// end_y,end_x is our goal, start_y,start_x is the origin of the entire path
// dest: if we are attacking or not, map_type: if we are moving on land or water
//
// Returns: true if destination found, false otherwise
//
// ** special behavior for CIVILIAN_ACTION_MV and UNIT_ACTION_WORKER_BUILD_ROAD
#[inline]
fn add_neighbors_to_list(prev_coord_ind: u64, end_coord: u64, zoom_ind: ZoomInd, cur_dist_prev: f32, 
		nodes: &mut HashMap<u64, Node, BuildHasherDefault<HashStruct64>>, 
		nodes_fin: &mut HashMap<u64, Node, BuildHasherDefault<HashStruct64>>, 
		map_data: &mut MapData, exz: &HashedMapEx, mv_vars: MvVarsAtZoom, bldgs: &Vec<Bldg>, 
		action_iface: &ActionInterfaceMeta, map_sz: MapSz, use_roads: bool,
		dest: &Dest, movement_type: MovementType) -> bool {
	
	let prev_coord = Coord::frm_ind(prev_coord_ind, map_sz);
	
	let start_coord = action_iface.start_coord;
	let start_coord_ind = (start_coord.y*(map_sz.w as isize) + start_coord.x) as u64;
	
	let mut key = Node{cur_dist: 0., sum_dist: 0., prev_coord: prev_coord_ind};
	
	for i_off in -1..=(1 as isize) {
	for j_off in -1..=(1 as isize) {
		if i_off == 0 && j_off == 0 {continue;} // same as prev_coord
		
		if let Some(coord) = map_sz.coord_wrap(prev_coord.y + i_off, prev_coord.x + j_off){ // next potential step
			if coord == start_coord_ind {continue;} // this brought us back to the beginning
			
			// check if movable to
			if !(*action_iface.movable_to)(prev_coord_ind, coord, &map_data.get(zoom_ind, coord), exz, mv_vars, bldgs, dest, movement_type)
				{continue;}
			
			// dists
			key.cur_dist = cur_dist_prev + mv_action_cost(prev_coord_ind, coord, use_roads, map_data, exz, map_sz);
			key.sum_dist = est_mv_action_cost(coord, end_coord, map_sz) + key.cur_dist;
			
			///////////////////////
			// chk if already added, and update if path is now shorter
			
			macro_rules! update_if_shorter{($nodes_chk: expr) => (
					let node_found = (*$nodes_chk).get_mut(&coord);
					if let Some(nf) = node_found { // found
						if key.sum_dist < nf.sum_dist {
							*nf = key;
						}
						continue;
					}
				);}
			
			update_if_shorter!(nodes);
			update_if_shorter!(nodes_fin);
			
			nodes.insert(coord, key);
			
			if coord == end_coord {return true;} // finished			
		} // coord is valid
	}}
	false
}

// move from action_iface.start_coord to end_coord
//	if the path is far, we compute checkpoints from the zoomed out map and then compute an exact path (at
//    full zoom) to the first far-enough away checkpoint.
//
//		Note: when action == WorkerBuildRoad, we do not use roads
//		when action == CivilianMv, we always use roads or zoned territoriy
//		Note: units only needs to be supplied when in UI mode (civil move mode does not req. it)
impl <'f,'bt,'ut,'rt,'st> ActionInterfaceMeta<'f,'bt,'ut,'rt,'st> {
	pub fn update_move_search(&mut self, mut end_coord: Coord, map_data: &mut MapData, exs: &Vec<HashedMapEx>,
					mv_vars: MvVars, bldgs: &Vec<Bldg>){
		#[cfg(feature="profile")]
		let _g = Guard::new("update_move_search");
		
		// path shouldn't be updated once we start drawing the zoning rectangle
		if let ActionType::WorkerZone {start_coord: Some(_), ..} |
			 ActionType::WorkerRmZonesAndBldgs {start_coord: Some(_), ..} = &self.action.action_type {
				return;
		}
		
		self.action.actions_req = 0.;
		self.action.path_coords.clear();
		
		let map_sz = *map_data.map_szs.last().unwrap();
		
		debug_assertq!(self.start_coord.y >= 0 && self.start_coord.x >= 0);
		debug_assertq!(end_coord.y >= 0 && end_coord.x >= 0);
		debug_assertq!(self.start_coord.y < (map_sz.h as isize) && self.start_coord.x < (map_sz.w as isize));
		debug_assertq!(end_coord.y < (map_sz.h as isize) && end_coord.x < (map_sz.w as isize));
		
		// nothing to find
		if self.start_coord == end_coord {return;} //printlnq!("at destination"); return;}
		
		//self.blind_undiscov = false; //////??????????????????????????????
		
		let checkpoint_dist = self.max_search_depth / 5;
		
		let exf = exs.last().unwrap();
		let mv_vars_full = mv_vars.to_zoom(map_data.max_zoom_ind());
		
		///////////////////////////////////////
		// use or create checkpoints
		//
		// is the destination too far to consider? if so, shorten the exact path we will compute at full zoom to a checkpoint destination
		//	(the checkpoints are computed from an exact path computed on a zoomed out map)
		if manhattan_dist(self.start_coord, end_coord, map_sz) >= checkpoint_dist &&
				self.action.action_type != ActionType::CivilianMv {
			macro_rules! find_next_end_coord{($path_coords: expr) => {
				// step back along checkpointed path to find first coordinate which we will compute a full zoomed-in path, save in end_coord
				let mut found = false;
				
				for path_coord in $path_coords.iter() {
					let path_coord_c = Coord::frm_ind(*path_coord, map_sz);
					
					//printlnq!("path_coord b {}, start {} max depth {} dist {}", path_coord_c, self.start_coord, checkpoint_dist,
					//		manhattan_dist(self.start_coord, path_coord_c, map_sz));
					
					if manhattan_dist(self.start_coord, path_coord_c, map_sz) >= checkpoint_dist {continue;}
					
					assertq!(path_coord_c.y < map_sz.h as isize && path_coord_c.x < map_sz.w as isize, "{} map_sz {} {}", path_coord_c, map_sz.h, map_sz.w);
					
					end_coord = path_coord_c;
					found = true;
					break;
				}
				
				//assertq!(found, "path_coords len {}", $path_coords.len());
				if !found {
					end_coord = Coord::frm_ind(*$path_coords.last().unwrap(), map_sz);
				}
			}};
			
			//////////////////
			// zoomed out path already set from previous call
			// (i.e., unit has reached first checkpoint)
			if let Some(action_meta_cont) = &self.action.action_meta_cont {
				// set end_coord based on action_meta_cont.zoomed_out_path_coords
				find_next_end_coord!(action_meta_cont.checkpoint_path_coords);
				//printlnq!("using pre-computed checkpoint {} final {}", end_coord, self.action.action_meta_cont.as_ref().unwrap().final_end_coord);
			////////////////
			// compute exact zoomed out path
			}else{
				// impossible destination at full zoom
				let end_coord_ind = end_coord.to_ind(map_sz) as u64;
				if !map_type_consistent_w_mvment(end_coord_ind, &map_data.get(ZoomInd::Full, end_coord_ind), exf, self.movement_type, mv_vars_full) {
					return;
				}
				
				let map_szs = &map_data.map_szs;
				let zoom_ind_full = map_szs.len()-1;
				
				let end_coord_zoomed_out   =        end_coord.to_zoom(zoom_ind_full, ZOOM_IND_ROOT, map_szs);
				let start_coord_zoomed_out = self.start_coord.to_zoom(zoom_ind_full, ZOOM_IND_ROOT, map_szs);
				
				let max_search_depth_backup = self.max_search_depth;
				self.max_search_depth *= 15;
				
				if let Some((mut path_coords, _)) = self.update_move_search_at_zoom(end_coord_zoomed_out, start_coord_zoomed_out, ZoomInd::Val(ZOOM_IND_ROOT), 
																map_data, &exs[ZOOM_IND_ROOT], mv_vars.to_zoom(ZOOM_IND_ROOT), bldgs, map_data.map_szs[ZOOM_IND_ROOT]) {
					let map_szs = &map_data.map_szs;
					
					// convert path coordinates to full zoom
					for path_coord in path_coords.iter_mut() {
						*path_coord = Coord::frm_ind(*path_coord, map_szs[ZOOM_IND_ROOT])
									   .to_zoom(ZOOM_IND_ROOT, zoom_ind_full, map_szs)
									   .to_ind(map_sz) as u64;
						//printlnq!("path_coord {}", Coord::frm_ind(*path_coord, map_sz));
					}
					
					let final_end_coord = end_coord;
					find_next_end_coord!(path_coords);
					
					self.action.action_meta_cont = Some(ActionMetaCont {
							final_end_coord,
							checkpoint_path_coords: path_coords
					});
					self.max_search_depth = max_search_depth_backup;
					
				// path not possible
				}else{
					self.max_search_depth = max_search_depth_backup;
					//printlnq!("could not find path on zoomed out map max_search_depth {}, zoom_ind_root {}, start_coord {} end_coord {}",
					//		self.max_search_depth, ZOOM_IND_ROOT, start_coord_zoomed_out, end_coord_zoomed_out);
					return;
				}
				
				//printlnq!("start {} using checkpoint {} final {}", self.start_coord, end_coord,
				//		self.action.action_meta_cont.as_ref().unwrap().final_end_coord);
			}
		
		// we are close to the destination now to compute the remaining exact path at full zoom,
		// so we clear out all checkpoints, and prevent ignoring if a path is invalid (which 
		// 	avoids invalidly moving into a city, but is needed between intermediate checkpoints
		//	in case the checkpoint is in or around water on the zoomed-in map)
		}else{self.action.action_meta_cont = None;}
		
		// compute path at full zoom using checkpoint or actual end_coord
		if let Some((path_coords, actions_req)) = self.update_move_search_at_zoom(end_coord, self.start_coord, ZoomInd::Full, map_data,
												exf, mv_vars_full, bldgs, map_sz) {
			self.action.actions_req = actions_req;
			self.action.path_coords = path_coords;
			
		// path not possible -- ignore land movability (in conditions listed below) because we are just trying to get to a checkpoint:
		//	Conditions:
		//		-we're not about to finish the path
		//		-OR we're not presently on a movable position (ex. a land unit on water)
		}else if let Some(action_meta_cont) = &self.action.action_meta_cont {
			let start_coord_ind = self.start_coord.to_ind(map_sz) as u64;
			// conditions listed above for ignoring an impossible path:
			if end_coord != action_meta_cont.final_end_coord
			   || map_type_consistent_w_mvment(start_coord_ind, &map_data.get(ZoomInd::Full, start_coord_ind), exf, self.movement_type, mv_vars_full){
				// travel over 'air' so that we can ignore the fact that the checkpoint might even be in water
				let movement_type_back = self.movement_type;
				self.movement_type = MovementType::AllMapTypes;
				// ignoring land movability is possible
				if let Some((path_coords, actions_req)) = self.update_move_search_at_zoom(end_coord, self.start_coord, ZoomInd::Full, map_data, exf, mv_vars_full, bldgs, map_sz) {
					self.action.actions_req = actions_req;
					self.action.path_coords = path_coords;
					//println!("found path after ignoring assumption that intermediate path be on land");
				// even ignoring land movability the path is still impossible
				}else{
					self.action.action_meta_cont = None;
					//println!("could not find path even when ignoring assumption that intermediate path be on land");
				}
				self.movement_type = movement_type_back;
			// abort -- conditions for ignoring land movability not met
			}else{
				//endwin();println!("could not find path on full map, action_meta_cont exists");
				self.action.action_meta_cont = None;
			}
			
		// path not possible and not going to a checkpoint
		}else{
			//printlnq!("could not find path on full map");
			self.action.action_meta_cont = None;
		}
	}
	
	// returns: (path_coords, actions_req)
	fn update_move_search_at_zoom(&self, end_coord: Coord, start_coord: Coord, zoom_ind: ZoomInd, map_data: &mut MapData,
						exz: &HashedMapEx, mv_vars: MvVarsAtZoom, bldgs: &Vec<Bldg>, map_sz: MapSz) -> Option<(Vec<u64>, f32)> {
		// impossible source (todo: in the future, boats can move in water...)
		let start_coord_ind = start_coord.to_ind(map_sz) as u64;
		
		//debug_assertq!(map_type_consistent_w_mvment(start_coord_ind, &map_data.get(zoom_ind, start_coord_ind), exz, self.movement_type, mv_vars),
		//		"map not consistent w/ movement, start: {} end: {}, zoom: {}", start_coord, end_coord, zoom_ind);
		
		// impossible destination at zoom_ind
		let end_coord_ind = end_coord.to_ind(map_sz) as u64;
		if !map_type_consistent_w_mvment(end_coord_ind, &map_data.get(zoom_ind, end_coord_ind), exz, self.movement_type, mv_vars) {
			return None;
		}
		
		////
		let use_roads = match self.action.action_type {
			ActionType::WorkerBuildStructure {..} => false,
			_ => true
		};
		
		// trivial path (directly away)
		{
			let mdist = manhattan_dist(start_coord, end_coord, map_sz);
			if mdist <= 2 && (end_coord.y - start_coord.y).abs() <= 1 &&
					((end_coord.x - start_coord.x).abs() <= 1 ||
					 (end_coord.x == 0 && start_coord.x == (map_sz.w-1) as isize) || //  condition 1 for map wrap around
					 (end_coord.x == (map_sz.w-1) as isize && start_coord.x == 0))   //  condition 2 for map wrap around
			{
				return Some((vec![end_coord_ind], mv_action_cost(start_coord_ind, end_coord_ind, use_roads, map_data, exz, map_sz)));
			}
		}
		//printlnq!("move search");
		
		// additional variables used with calls to movable_to()
		let dest = Dest::from(&self.action.action_type, end_coord_ind,
			if let MvVarsAtZoom::NonCivil {start_owner, ..} = mv_vars {
				Some(start_owner)
			} else {None});
		
		let s: BuildHasherDefault<HashStruct64> = Default::default();
		let s2: BuildHasherDefault<HashStruct64> = Default::default();
		
		let mut nodes: HashMap<u64, Node, BuildHasherDefault<HashStruct64>> = HashMap::with_capacity_and_hasher(self.max_search_depth+1, s);
		let mut nodes_fin: HashMap<u64, Node, BuildHasherDefault<HashStruct64>> = HashMap::with_capacity_and_hasher(self.max_search_depth+1, s2);
		
		// we should've already returned if it's a trivial path
		assertq!(add_neighbors_to_list(start_coord_ind, end_coord_ind, zoom_ind, 0., &mut nodes, &mut nodes_fin,
				map_data, exz, mv_vars, bldgs, self, map_sz, use_roads, &dest, self.movement_type) == false);
		
		/////////////////////////////// astar
		loop{
			if nodes.len() == 0 {return None;}
			
			///// find next min node
			let mut min_node = Node {cur_dist: 0., sum_dist: 0., prev_coord: 0};
			let mut node_coord = 0_u64;
			for (i, (coord_i, node_i)) in nodes.iter().enumerate() {
				if i == 0 || min_node.sum_dist > node_i.sum_dist {
					min_node = *node_i;
					node_coord = *coord_i;
				}
			} // min
			
			let ret_val = add_neighbors_to_list(node_coord, end_coord_ind, zoom_ind, min_node.cur_dist, &mut nodes, &mut nodes_fin,
					map_data, exz, mv_vars, bldgs, self, map_sz, use_roads, &dest, self.movement_type);
			
			if ret_val {
				break; // done
			}else if (nodes_fin.len()+1) >= self.max_search_depth { // couldn't find path
				//printlnq!("could not find path, no more buffer {} z {}", self.max_search_depth, zoom_ind == ZoomInd::Full);
				return None;
			}
			
			// cp node to finished list
			nodes_fin.insert(node_coord, min_node);
			
			// delete from working list
			nodes.remove(&node_coord);
		}
		
		///////////////////// copy node path to path_coords
		let mut coord_add = end_coord_ind as u64;
		let mut node_add = nodes.get(&coord_add).unwrap();
		let actions_req = node_add.cur_dist;
		let mut path_coords = Vec::with_capacity(100);
		
		loop {
			// cp
			path_coords.push(coord_add);
			if node_add.prev_coord == start_coord_ind {break;} // start pos
			
			// find next
			coord_add = node_add.prev_coord;
			
			let mut node_add_res = nodes_fin.get(&coord_add);
			if node_add_res.is_none() {
				node_add_res = nodes.get(&coord_add);
				debug_assertq!(!node_add_res.is_none());
			}
			node_add = node_add_res.unwrap();
		}
		Some((path_coords, actions_req))
	}
}

