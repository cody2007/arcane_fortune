#[macro_use]
pub mod vars;
pub mod attack;
pub use vars::*;
pub use attack::*;
use crate::map::{MapType, MapData, MapSz, ZoomInd, compute_active_window, PresenceAction};
use crate::player::{Stats, Player};
use crate::movement::*;
use crate::disp::{Coord, ScreenSz, IfaceSettings};
use crate::gcore::hashing::*;
use crate::buildings::*;
use crate::config_load::*;
use crate::saving::*;
use crate::tech::TechTemplate;
use crate::resources::ResourceTemplate;
use crate::gcore::{Log, Relations};
use crate::localization::Localization;
use crate::disp_lib::endwin;

impl IfaceSettings<'_,'_,'_,'_,'_> {
	// check to see if player is clear to end the turn
	pub fn update_all_player_pieces_mvd_flag(&mut self, units: &Vec<Unit>){
		self.all_player_pieces_mvd = true;
		for u in units {
			if !u.actions_used.is_none() && u.owner_id == self.cur_player && u.action.len() == 0 &&
					u.template.nm[0] != RIOTER_NM {
				self.all_player_pieces_mvd = false;
				return;
			}
		} // unit search
	}
}

pub fn init_unit_templates<'rt>(tech_templates: &Vec<TechTemplate>,
		resource_templates: &'rt Vec<ResourceTemplate>,
		l: &Localization) -> Vec<UnitTemplate<'rt>> {
	let key_sets = config_parse(read_file("config/units.txt"));
	chk_key_unique("nm", &key_sets);
	
	let mut unit_templates: Vec<UnitTemplate<'rt>> = Vec::new();
	
	for (id, keys) in key_sets.iter().enumerate() {
		let eng_nm = find_req_key("nm", keys);
		let nm = if let Some(nm) = l.unit_nms.iter().find(|nms| nms[0] == eng_nm) {
			nm.clone()
		}else{panicq!("could not find translations of resource `{}`. the localization file may need to be updated", eng_nm);};

		let attack_per_turn = find_opt_key_parse("attack_per_turn", &keys);
		let siege_bonus_per_turn = find_opt_key_parse("siege_bonus_per_turn", &keys);
		
		// attack_per_turn must be set if siege_bonus_per_turn is set (attacking end_turn code assumes this)
		if let Some(_) = siege_bonus_per_turn {
			if attack_per_turn == None {
				panicq!("unit configuration file, `attack_per_turn` must be given (even if set to zero) if `siege_bonus_per_turn` has been supplied for a unit's entry.");
			}
		}
		
		unit_templates.push( UnitTemplate {
			id: id as SmSvType,
			
			tech_req: find_tech_req(&eng_nm, &keys, &tech_templates),
			resources_req: find_resources_req(&eng_nm, &keys, resource_templates),
			
			nm,
			
			movement_type: find_req_key_parse("movement_type", &keys),
			
			carry_capac: find_key_parse("carry_capac", 0, &keys),
			
			actions_per_turn: find_req_key_parse("actions_per_turn", &keys),
			attack_per_turn,
			siege_bonus_per_turn,
			repair_wall_per_turn: find_opt_key_parse("repair_wall_per_turn", &keys),
			attack_range: find_opt_key_parse("attack_range", &keys),
			max_health: find_req_key_parse("max_health", &keys),
			
			production_req: find_req_key_parse("production_req", &keys),
			char_disp: find_req_key_parse("char_disp", &keys),
			upkeep: find_req_key_parse("upkeep", &keys) 
		} );
	}
	
	// check ordering is correct
	#[cfg(any(feature="opt_debug", debug_assertions))]
	for (i, ut) in unit_templates.iter().enumerate() {
		debug_assertq!(ut.id == i as SmSvType);
	}
	
	unit_templates
}

#[derive(PartialEq)]
pub enum Quad {Any, Lr, Ul, Ur, Ll}

// returns end coordinate of the square
pub fn square_clear(coord: u64, blank_spot: ScreenSz, quad: Quad, map_data: &mut MapData, exf: &HashedMapEx) -> Option<u64> {
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	let c = Coord::frm_ind(coord, map_sz);
	let hi = blank_spot.h as isize;
	let wi = blank_spot.w as isize;
	
	let mut search = |start_i: isize, end_i: isize, start_j: isize, end_j: isize| {
		for i_off in start_i..end_i {
		for j_off in start_j..end_j {
			if let Some(coord) = map_sz.coord_wrap(c.y + i_off, c.x + j_off) {
				//endwin();
				//println!("{} {}, {} {} map_sz {} {}", c.y, i_off, c.x, j_off, map_sz.h, map_sz.w);
				if map_data.get(ZoomInd::Full, coord).map_type != MapType::Land {return false;}
				
				if let Some(ex) = exf.get(&coord) {
					if let Some(_) = ex.unit_inds {return false;}
					if let Some(_) = ex.bldg_ind {return false;}
					if let Some(_) = ex.actual.structure {return false;}
					if let Some(_) = ex.actual.ret_zone_type() {return false;}
				}
			
			} else {return false;}
		}}
		return true;
	};
	
	macro_rules! chk_quad{($start_i:expr, $end_i:expr, $start_j:expr, $end_j:expr,
				$fin_i:expr, $fin_j:expr) => (
		
		if search($start_i, $end_i, $start_j, $end_j) {
			return map_sz.coord_wrap(c.y + $fin_i, c.x + $fin_j);
		}
	)};
	
	// quadrant 4 (lower right)
	if quad == Quad::Any || quad == Quad::Lr {
		chk_quad!(0,hi,  0,wi,   hi-1, wi-1);
	}
	
	let nhi = -hi + 1;
	let nwi = -wi + 1;
	
	// quadrant 2 (upper left)
	if quad == Quad::Any || quad == Quad::Ul {
		chk_quad!(nhi,1,  nwi,1,   nhi, nwi);
	}
	
	// quadrant 3 (lower left)
	if quad == Quad::Any || quad == Quad::Ll {
		chk_quad!(0,hi,  nwi,1,   hi-1, nwi);
	}
	
	// quadrant 1 (upper right)
	if quad == Quad::Any || quad == Quad::Ur {
		chk_quad!(nhi,1,  0,wi,   nhi, wi-1);
	}
	
	None
}

impl IfaceSettings<'_,'_,'_,'_,'_> {
	// unit from cursor position (reqs cur_player own it or returns None)
	// also see bldg_ind_from_cur()
	pub fn unit_ind_frm_cursor(&self, units: &Vec<Unit>, map_data: &mut MapData, exf: &HashedMapEx) -> Option<usize> {
		if self.zoom_ind == map_data.max_zoom_ind() || units.len() != 0 {
			let map_coord = self.cursor_to_map_ind(map_data);
			if let Some(ex) = exf.get(&map_coord) {
				if let Some(unit_inds) = &ex.unit_inds {
					if unit_inds.len() > self.unit_subsel {
						//debug_assertq!(unit_inds.len() > self.unit_subsel);
						
						let unit_ind = unit_inds[self.unit_subsel];
						
						if units[unit_ind].owner_id == self.cur_player {
							return Some(unit_ind);
						}
					}
				}
			}
		}
		None
	}
	
	pub fn unit_inds_frm_sel(&self, pstats: &Stats, units: &Vec<Unit>, map_data: &mut MapData, exf: &HashedMapEx) -> Option<Vec<usize>> {
		if self.zoom_ind == map_data.max_zoom_ind() || units.len() != 0 {
			if let Some(brigade_nm) = &self.add_action_to.brigade_sel_nm() {
				Some(pstats.brigade_frm_nm(brigade_nm).unit_inds.clone())
			}else{
				if let Some(unit_ind) = self.unit_ind_frm_cursor(units, map_data, exf) {
					Some(vec![unit_ind])
				}else{
					None
				}
			}
		}else{None}
	}
}

// returns None if no bldg present or under construction at cursor,
// returns Some(bldg_ind) if there is
pub fn worker_can_continue_bldg(chk_path: bool, bldgs: &Vec<Bldg>, map_data: &mut MapData, exf: &HashedMapEx,
		iface_settings: &IfaceSettings) -> Option<usize> {
	if !chk_path || iface_settings.add_action_to.actions().iter().any(|action| action.path_coords.len() != 0) {
		if let Some(bldg_ind) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
			let b = &bldgs[bldg_ind];
			if b.owner_id == iface_settings.cur_player {
				if let Some(_) = bldgs[bldg_ind].construction_done {
					return Some(bldg_ind);
				}
			}
		}
	}
	None
}

#[derive(PartialEq)]
pub enum DelAction<'d> {
	Record(&'d mut Vec<usize>),
	Delete
}
// ^ Record: mv_unit() appends to vector of unit inds that should be deleted
// (because they boarded a boat, and mv_unit() can be called over loops of all units)
// Delete: delete the unit that boarded the boat (becomes stored in the boats Unit structure)

// move unit based on the next steps along its path_coords
pub fn mv_unit<'bt,'ut,'rt,'dt,'d>(unit_ind: usize, is_cur_player: bool, 
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		bldgs: &Vec<Bldg>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		relations: &mut Relations, map_sz: MapSz, del_action: DelAction, logs: &mut Vec<Log>, turn: usize) {
	
	//printlnq!("mv_unit");
	
	let u = &units[unit_ind];
	let u_coord = u.return_coord();
	
	debug_assertq!(!u.actions_used.is_none());
	debug_assertq!(u.template.actions_per_turn > 0.);
	if let Some(actions_used) = u.actions_used {
		if actions_used >= u.template.actions_per_turn {return;}
	}else{
		//panicq!("unit {} ({}), owner {} has no actions_used", u.template.nm[0], unit_ind, u.owner_id);
		return;
	}
	debug_assertq!(u.action.len() != 0);
	
	let u_owner = u.owner_id as usize;
	
	let action = u.action.last().unwrap();
	debug_assertq!(action.path_coords.len() > 0);
	debug_assertq!(action.actions_req > 0.);
	
	let actions_per_turn_use = match action.action_type {
		ActionType::WorkerBuildStructure {..} => 1., 
		_ => u.template.actions_per_turn };
	
	// used to check if path is still movable
	let dest = {
		let end_coord_ind = match &action.action_type {
			ActionType::Attack {attack_coord, ..} => {
				if let Some(coord) = attack_coord {
					*coord
				}else{
					action.path_coords[0]
				}
			}
			ActionType::Mv | ActionType::MvWithCursor |
			ActionType::MvIgnoreWalls | ActionType::MvIgnoreOwnWalls |
			ActionType::CivilianMv | ActionType::AutoExplore {..} | ActionType::WorkerBuildStructure {..} |
			ActionType::WorkerRepairWall {..} | ActionType::WorkerBuildBldg {..} | ActionType::Fortify {..} |
			ActionType::WorkerZone {..} | ActionType::WorkerZoneCoords {..} | ActionType::GroupMv {..} |
			ActionType::WorkerRmZonesAndBldgs {..} |
			ActionType::BrigadeCreation {..} | ActionType::SectorCreation {..} | ActionType::SectorAutomation {..} |
			ActionType::UIWorkerAutomateCity | ActionType::WorkerContinueBuildBldg {..} |
			ActionType::BurnBuilding {..} => {
				action.path_coords[0]
			}
		};
		
		Dest::from(&action.action_type, end_coord_ind, Some(u.owner_id))
	};
	
	////////////////////////////////////////////
	// unit can move there now
	if (actions_per_turn_use - u.actions_used.unwrap()) >= action.actions_req {
		//printlnq!("\tunit ind {} can move to destination now", unit_ind);
		
		let mut dest_coord = action.path_coords[0];
		let mut abort_mv = false; // if path becomes untraversable, set to true
		
		// check if path is still traversable (starting from cur loc) and discover land along path
		if u_coord != dest_coord { // not already at destination (don't want to inadvertedly set u.action = None in this case)			
			// if we are only traveling to an intermediate checkpoint, ignore the fact that some locations may not be movable to
			let ignore_movable_to = if let Some(_) = &action.action_meta_cont {
									true //dest_coord != action_cont.final_end_coord.to_ind(map_sz) as u64
			}else {false};
			
			for (step_i, step) in action.path_coords.iter().enumerate().rev() {
				// cannot move here
				if !ignore_movable_to && !movable_to(u_coord, *step, &map_data.get(ZoomInd::Full, *step), exs.last().unwrap(), MvVarsAtZoom::NonCivil{units, start_owner: u.owner_id, blind_undiscov: None}, 
						bldgs, &dest, u.template.movement_type) {
					
					// no moves possible
					if (step_i+1) >= action.path_coords.len() {
						//printlnq!("unit_ind {} no moves possible, step_i {}, path_coords.len {}", unit_ind, step_i, action.path_coords.len());
						units[unit_ind].action.pop();
						return;
					}
					
					// move up to previous coord in path_coords
					dest_coord = action.path_coords[step_i+1];
					
					// set the new destination as the end point
					units[unit_ind].action.last_mut().unwrap().action_meta_cont = Some(ActionMetaCont {
							final_end_coord: Coord::frm_ind(dest_coord, map_sz),
							checkpoint_path_coords: Vec::new()
					});
					
					abort_mv = true;
					break;
				
				// discover land
				}else{
					compute_active_window(*step, is_cur_player, PresenceAction::DiscoverOnly, map_data, exs, &mut players[u_owner].stats, map_sz, relations, units, logs, turn);
				}
			}
		}
		
		set_coord(dest_coord, unit_ind, is_cur_player, units, map_data, exs, &mut players[u_owner].stats, map_sz, relations, logs, turn); // mv to destination
		let u = &mut units[unit_ind];
		let action = u.action.last().unwrap();
		let actions_req = action.actions_req;
		
		// has unit moved to final destination?
		if action.action_meta_cont.is_none() || 
				dest_coord == action.action_meta_cont.as_ref().unwrap().final_end_coord.to_ind(map_sz) as u64 {
			// set unit action / clear path
			match action.action_type {
				ActionType::Mv | ActionType::MvIgnoreWalls | ActionType::MvIgnoreOwnWalls => {
					u.action.pop();
					//println!("unit_ind {} finished move", unit_ind);
				}_ => {
					if abort_mv {
						u.action.pop();
						//println!("unit_ind {} aborting move, not possible", unit_ind);
					}else{
						u.action.last_mut().unwrap().path_coords.clear();
						// useful for attacking, for instance, to signal that the attacked unit's health can
						// be drained, or killed since we've reached the destination
					}
				}
			}
		
		// continue moving until unit reaches destination (compute next set of path_coords)
		}else{
			let mut action_iface = unit_action_iface(Coord::frm_ind(dest_coord, map_sz), action.action_type.clone(), unit_ind, units);
			
			let u = &mut units[unit_ind];
			
			if let Some(action) = u.action.pop() {
				action_iface.action = action;
				let start_owner = u.owner_id;
				action_iface.update_move_search(action_iface.action.action_meta_cont.as_ref().unwrap().final_end_coord, map_data, exs, 
						MvVars::NonCivil{units, start_owner, blind_undiscov: None}, bldgs);
				
				// set attack range (only if there is a valid path to the attack point)
				if action_iface.action.path_coords.len() > 0 {
					let u = &mut units[unit_ind];
					u.action.push(action_iface.action);
					
					if let ActionType::Attack {..} = &u.action.last().unwrap().action_type {
						//endwin();
						u.set_attack_range(map_data, exs.last().unwrap(), map_sz);
					}
				}
				
			}else{panicq!("action not found");}
			
			// debug
			/*if action.path_coords.len() == 0 && (action.action_meta_cont.is_none() || 
									  action.action_meta_cont.as_ref().unwrap().checkpoint_path_coords.len() == 0) {
				endwin();
				println!("update_move_search did not return any coordinates");
			}*/
		}
		
		// update actions_used
		let u = &mut units[unit_ind];
		if (u.actions_used.unwrap() + actions_req) == actions_per_turn_use {
			u.actions_used = None;
		}else{
			u.actions_used = Some(u.actions_used.unwrap() + actions_req);
			debug_assertq!(u.actions_used.unwrap() < actions_per_turn_use);
		}
	
	//////////////////////////////////////
	// will take multiple turns
	}else{
		let ignore_movable_to = !action.action_meta_cont.is_none();
		// ^ allow checkpoints and intermediate paths to potentially traverse over water
		
		// can't even take one step, cancel action
		let next_coord = action.path_coords[action.path_coords.len()-1];
		if !ignore_movable_to && !movable_to(u_coord, next_coord, &map_data.get(ZoomInd::Full, next_coord), exs.last().unwrap(), 
								MvVarsAtZoom::NonCivil{units, start_owner: u.owner_id, blind_undiscov: None}, bldgs, &dest, u.template.movement_type) {
			
			units[unit_ind].action.pop();
			////////
			/*let cn = Coord::frm_ind(next_coord, map_sz);
			let cu = Coord::frm_ind(units[unit_ind].return_coord(), map_sz);
			printlnq!("unit_ind {} can't move even one step. {} {} -> {} {}", unit_ind, cu.y, cu.x, cn.y, cn.x);*/
			//////////
			return;
		}
		
		let use_roads = match action.action_type {
			ActionType::WorkerBuildStructure {..} => false,
			_ => true
		};
		
		let mut prev_cost = 0.; // action cost
		let mut prev_coord = u.return_coord();
		let mut moved = false;
		
		let actions_remaining = actions_per_turn_use - u.actions_used.unwrap();
		
		// step back to the beginning, see how far the unit can move
		for (step_ind, next_coord) in action.path_coords.iter().enumerate().rev() {
			debug_assertq!(step_ind == (action.path_coords.len()-1) || *next_coord != prev_coord, "step_ind {}", step_ind); // at first step or current and prev step coords are different
			#[cfg(any(feature="opt_debug", debug_assertions))]
			{
				let next_c = Coord::frm_ind(*next_coord, map_sz);
				let prev_c = Coord::frm_ind(prev_coord, map_sz);
				debug_assertq!((next_c.y - prev_c.y).abs() <= 1, "unit_ind {} {} owner {} next: {} {}  prev: {} {}", unit_ind, 
						units[unit_ind].nm, players[units[unit_ind].owner_id as usize].personalization.nm,
						next_c.y, next_c.x, prev_c.y, prev_c.x);
				
				//debug_assertq!(((next_coord % map_sz.w) as isize - (prev_coord % map_sz.w) as isize).abs() <= 1); // doesn't need to be true (wrap)
			}
			
			let next_cost = mv_action_cost(prev_coord, *next_coord, use_roads, map_data, exs.last().unwrap(), map_sz);
			let next_step_traversable = ignore_movable_to || movable_to(u_coord, *next_coord, &map_data.get(ZoomInd::Full, *next_coord), exs.last().unwrap(), 
													MvVarsAtZoom::NonCivil{units, start_owner: units[unit_ind].owner_id, blind_undiscov: None}, 
													bldgs, &dest, u.template.movement_type);
			let cost_sum = next_cost + prev_cost;
			
			// possible (or we must because path becomes not traversible) move now...
			// (stop at this mv)
			if !next_step_traversable || cost_sum >= actions_remaining {
				let dest_coord = *next_coord;
				let u = &mut units[unit_ind];
				
				// can go up to additional step
				if next_step_traversable && cost_sum == actions_remaining {
					u.actions_used = None; // no more actions available
					
					set_coord(dest_coord, unit_ind, is_cur_player, units, map_data, exs, &mut players[u_owner].stats, map_sz, relations, logs, turn);
					
					let act = units[unit_ind].action.last_mut().unwrap();
					act.path_coords.truncate(step_ind);
					act.actions_req -= cost_sum;
				
				// cannot go an additional step, just go to prev_coord for this turn...
				}else{
					u.actions_used = Some(u.actions_used.unwrap() + prev_cost.ceil());
					
					set_coord(prev_coord, unit_ind, is_cur_player, units, map_data, exs, &mut players[u_owner].stats, map_sz, relations, logs, turn);
					
					let act = units[unit_ind].action.last_mut().unwrap();
					act.path_coords.truncate(step_ind + 1);
					act.actions_req -= prev_cost;
				}
				
				debug_assertq!(units[unit_ind].action.last().unwrap().actions_req > 0.);
				moved = true;
				break;
			
			// movable, discover land
			}else if next_step_traversable {
				compute_active_window(*next_coord, is_cur_player, PresenceAction::DiscoverOnly, map_data, exs, &mut players[u_owner].stats, map_sz, relations, units, logs, turn);
			}
			
			prev_coord = *next_coord;
			prev_cost += next_cost;
		} // step through path to dest
		
		debug_assertq!(moved);
	} // multi-step move
	
	// boarding boat
	let u = &units[unit_ind];
	let mfc = map_data.get(ZoomInd::Full, u.return_coord());
	if mfc.map_type == MapType::ShallowWater && 
			u.template.movement_type == MovementType::Land {
		
		let unit_inds = exs.last().unwrap().get(&u.return_coord()).unwrap().unit_inds.as_ref().unwrap();
		
		// boat is owned by current owner and it has room?
		for boat_unit_ind in unit_inds {
			let bu = &units[*boat_unit_ind];
			if bu.template.carry_capac == 0 || bu.owner_id != u.owner_id {continue;}
			if let Some(units_carried) = &bu.units_carried {
				if (units_carried.len() + 1) > bu.template.carry_capac {
					continue;
				}
			}
			
			// board boat
			debug_assertq!(u.units_carried == None);
			let u = u.clone();
			players[u.owner_id as usize].stats.unit_expenses += u.template.upkeep;
			let bu = &mut units[*boat_unit_ind];
			if bu.units_carried == None {
				bu.units_carried = Some(Vec::with_capacity(bu.template.carry_capac));
			}
			if let Some(ref mut units_carried) = bu.units_carried {
				units_carried.push(u);
			}
			
			match del_action {
				DelAction::Delete => {
					disband_unit(unit_ind, is_cur_player, units, map_data, exs, players, relations, map_sz, logs, turn);
				} DelAction::Record(disband_unit_inds) => {
					disband_unit_inds.push(unit_ind);
				}
			}
			return;
		}
	}
}

impl <'bt,'ut,'rt,'dt> Unit<'bt,'ut,'rt,'dt> {
	// figure out what range we should move the unit to before it can attack (it may not need to move all the way to the end of the path to attack)
	// or if the unit even needs to move at all from its current position
	// inputs: first entry (last traversed) of path_coords should be the attack destination
	pub fn set_attack_range(&mut self, map_data: &mut MapData, exf: &HashedMapEx, map_sz: MapSz) {
		let u_coord = self.return_coord();
		let action = self.action.last_mut().unwrap();
		
		//printlnq!("{} {}", Coord::frm_ind(u_coord, map_sz), self.template.nm[0]);
		let default_attack_range = self.template.attack_range.unwrap() as f32;
		
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			if let ActionType::Attack {..} = &action.action_type {} else {
				panicq!("unit must be attacking to set attack range");
			}
			debug_assertq!(!self.template.attack_range.is_none());
		}
		
		let initial_step_cost = mv_action_cost(u_coord, *action.path_coords.last().unwrap(), true, map_data, exf, map_sz);
		
		////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
		//debug_assertq!((mv_path_cost(&action.path_coords, true, map_data, exf, map_sz) + initial_step_cost) == action.actions_req, 
		//			"({} + {}) != {}", mv_path_cost(&action.path_coords, true, map_data, exf, map_sz), initial_step_cost, action.actions_req);
		
		let ret_attack_range = || {
			// if destination is a wall, require unit to be directly next to it
			if let Some(attack_coord) = action.path_coords.first() {
				if let Some(ex) = exf.get(attack_coord) {
					if ex.actual.structure != None {
						return 1.;
					}
				}
			}
			
			default_attack_range
		};
		
		let attack_range = ret_attack_range();
		//printlnq!("unit nm {} owner {}, actions_req {}, attack_range {}", self.template.nm, self.owner_id, action.actions_req, attack_range);
		
		// some path will remain after removing the last portion
		if action.actions_req > attack_range {
			let mut found = false;
			let mut cost = 0.;
			
			//printlnq!("start loop path_coords {}, actions_req {} attack_range {}", action.path_coords.len(),
			//		action.actions_req, attack_range);
			
			// start at destination, work back until we reached the max attack range
			for i in 0..(action.path_coords.len()-1) {
				cost += mv_action_cost(action.path_coords[i+1], action.path_coords[i], true, map_data, exf, map_sz);
				////////////
				/*let c1 = Coord::frm_ind(action.path_coords[i+1], map_sz);
				let c2 = Coord::frm_ind(action.path_coords[i], map_sz);
				println!("i {} cost {} coords {} {} -> {} {}", i, cost, c1.y, c1.x, c2.y, c2.x);*/
				//////////////
				
				// the unit can't attack from a greater distance, so it stops here
				if cost >= attack_range {
					// i+1 is the last coordinate the unit will move to before attacking 
					// (and is included in the returned split)
					action.path_coords = action.path_coords.split_off(i+1);
					action.actions_req -= cost;
					
					//////////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					//debug_assertq!((mv_path_cost(&action.path_coords, true, map_data, exf, map_sz) +
					//			initial_step_cost) == action.actions_req);
					
					found = true;
					break;
				}
			} // path loop
			
			// unit can traverse entire path except its initial step from its start position
			if !found {
				debug_assertq!((initial_step_cost + cost) >= attack_range);
				debug_assertq!((cost + initial_step_cost) == action.actions_req); // check that the cost sum was correct
				action.path_coords = vec!{*action.path_coords.last().unwrap(); 1}; // keep only first step
				action.actions_req = initial_step_cost; // ?
				//println!("only one coord left actions_req {}", action.actions_req);
			}
			//println!("end loop");
		
		// no path remains
		}else{
			action.path_coords.clear();
			action.actions_req = 0.;
		}
	}
}

pub fn disband_units<'bt,'ut,'rt,'dt>(mut disband_unit_inds: Vec<usize>, cur_ui_player: SmSvType, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, 
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
		map_sz: MapSz, logs: &mut Vec<Log>, turn: usize) {
	#[cfg(any(feature="opt_debug", debug_assertions))]
	{
		// require each entry to only occur once
		for unit_ind in disband_unit_inds.iter() {
			debug_assertq!(disband_unit_inds.iter()
					.filter(|&&ind_test| ind_test == *unit_ind).count() == 1);
		}
	}
	
	// loop should be done starting from the greatest index and proceed
	// to the smallest, otherwise we will delete the wrong units
	
	disband_unit_inds.sort_unstable(); // results in smallest to largest
	for unit_ind in disband_unit_inds.iter().rev() {
		debug_assertq!(units.len() > *unit_ind, "units len {}, unit_ind {} turn {}", units.len(), *unit_ind, turn);
		if let Some(u) = units.get(*unit_ind) {
			disband_unit(*unit_ind, u.owner_id == cur_ui_player, units, map_data, exs, players, relations, map_sz, logs, turn);
		}
	}
}

impl Unit<'_,'_,'_,'_> {
	pub fn health(&self) -> f32 {
		100.*(self.health as f32)/(self.template.max_health as f32)
	}
}

