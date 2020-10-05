use crate::movement::*;
use crate::units::*;
use crate::doctrine::*;
use crate::tech::*;
use crate::map::*;
use crate::gcore::*;
use crate::keyboard::KeyboardMap;
use crate::disp::menus::{OptionsUI};
use super::*;

impl <'f,'bt,'ut,'rt,'dt>IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	// Step 2:
	// called when adding action to: `individual unit` or `all in brigade`.
	// *not called when adding an action to the brigade list* because no unit (and thus action_iface)
	// in particular is being referred to
	pub fn assign_action_iface_to_unit(&mut self, action_iface: ActionInterfaceMeta<'f,'bt,'ut,'rt,'dt>,
			cur_mi: u64, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
			bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			stats: &mut Vec<Stats<'bt,'ut,'rt,'dt>>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			ai_states: &mut Vec<Option<AIState<'bt,'ut,'rt,'dt>>>,
			barbarian_states: &mut Vec<Option<BarbarianState>>,
			relations: &mut Relations, owners: &Vec<Owner>, bldg_config: &BldgConfig,
			unit_templates: &'ut Vec<UnitTemplate<'rt>>,
			bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
			tech_templates: &Vec<TechTemplate>,
			doctrine_templates: &'dt Vec<DoctrineTemplate>,
			zone_exs_owners: &mut Vec<HashedMapZoneEx>,
			map_data: &mut MapData<'rt>, map_sz: MapSz,
			disp_settings: &DispSettings, disp_chars: &DispChars, menu_options: &mut OptionsUI,
			logs: &mut Vec<Log>, turn: usize, rng: &mut XorState, frame_stats: &mut FrameStats,
			kbd: &KeyboardMap, l: &Localization, buttons: &mut Buttons,
			txt_list: &mut TxtList, d: &mut DispState) -> bool {
		if !self.pre_process_action_chk_valid(cur_mi, units, bldgs, exs.last().unwrap(), map_data) {return false;}
		let unit_ind = action_iface.unit_ind.unwrap();
		let u = &mut units[unit_ind];
		let u_coord = u.return_coord();
		
		// if moving, or attacking, req there be a path
		if let ActionType::Attack {..} | ActionType::Mv {..} = action_iface.action.action_type {
			if action_iface.action.path_coords.len() == 0 {return false;}
		}
		
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			debug_assertq!(u.owner_id == self.cur_player);
			debug_assertq!(!u.actions_used.is_none());
			debug_assertq!(u.actions_used.unwrap() < u.template.actions_per_turn);
			
			let start_coord = action_iface.start_coord;
			debug_assertq!(start_coord.y >= 0 && start_coord.x >= 0);
			debug_assertq!(start_coord.y < (map_sz.h as isize) && start_coord.x < (map_sz.w as isize));
			let map_coord_prev = (start_coord.y as usize)*map_sz.w + (start_coord.x as usize);
			debug_assertq!(map_coord_prev as u64 == u.return_coord());
		}
		
		// move self.action_iface.action to u
		u.action.pop();
		u.action.push(action_iface.action);
		let action = u.action.last_mut().unwrap();
		
		///////////////////////////////////////
		// action specific preparations:
		/////////////////////////////////////
		
		// attacking additional steps:
		//
		//	1. last move is the attack, remove it from path_coords, store in "U"
		//		(unless the path is checkpointed, in which case the final coordinates in action.path_coords
		//		 will not have been computed and instead this step is performed in mv_unit() upon reaching the
		//		 final checkpoint and computing the final round of action.path_coords)
		//	2. store owner to be attacked
		
		if let ActionType::Attack {..} = action.action_type {
			debug_assertq!(!u.template.attack_range.is_none());
			let exf = exs.last().unwrap();
			
			//////////////////// 1) remove last steps from path (based on attack range)
			let dest_coord = {
				// path_coords does not contain destination (it is checkpointed)
				if let Some(action_meta_cont) = &action.action_meta_cont {
					action_meta_cont.final_end_coord.to_ind(map_sz) as u64
				// traversing entire path with checkpoints (path_coords re-computed when the checkpoint is reached)
				}else{
					let dest_coord = action.path_coords[0];
					// ^ note: zeroth entry of path_coords is the destination coord
					u.set_attack_range(map_data, exf, map_sz); // shorten path_coords (therefore, dest_coord should be saved before doing this step)
					dest_coord
				}
			};
			
			//////////////////// 2) save owner to be attacked in u
			if let Some(ex) = exf.get(&dest_coord) {
				let contains_rioters = || {
					if let Some(dest_unit_inds) = &ex.unit_inds {
						dest_unit_inds.iter().any(|&ind|
							units[ind].template.nm[0] == RIOTER_NM
						)
					}else{false}
				};
				
				// set action if destination is not owned by attacking player
				macro_rules! set_action{($dest_owner:expr) => {
					if $dest_owner != self.cur_player || contains_rioters() { // not attacking self
						units[unit_ind].action.last_mut().unwrap().action_type = 
							ActionType::Attack {
									attack_coord: Some(dest_coord), 
									attackee: Some($dest_owner),
									ignore_own_walls: false
							};
					}
				};};
				
				// check if there is a structure at the destination
				if let Some(_) = ex.actual.structure {
					set_action!(ex.actual.owner_id.unwrap());
					
				// check if there is a bldg at the destination
				}else if let Some(bldg_ind) = ex.bldg_ind {
					set_action!(bldgs[bldg_ind].owner_id);
					
				// check that there are units at destination
				}else if let Some(dest_unit_inds) = &ex.unit_inds {
					debug_assertq!(dest_unit_inds.len() > 0);
					
					set_action!(units[dest_unit_inds[0]].owner_id);
				}
			}
		
		//////////////////////////////
		// group move specific: select all units in rectangle and move them
		}else if let ActionType::GroupMv {start_coord: Some(rect_start_c), end_coord: Some(rect_end_c)} = action.action_type {				
			// movement possible
			if let Some(mv_dest) = action.path_coords.first() {
				let mv_dest = Coord::frm_ind(*mv_dest, map_sz);
				let moving_owner = u.owner_id;
				let src_u_orig_coord = Coord::frm_ind(u_coord, map_sz);
				// ^ original coordinate for source group unit -- used to determine rel positions for other units
				
				// convert original unit's action to a standard Mv
				// 	(GroupMv should not be used aside from UI selection of units)
				action.action_type = ActionType::Mv;
				mv_unit(unit_ind, true, units, map_data, exs, bldgs, stats, relations, owners, map_sz, DelAction::Delete {barbarian_states, ai_states}, logs, turn);
				
				// mv group
				{
					// rectangle selecting group
					let (rect_start_c, rect_sz) = {
						let rect_start_c = Coord::frm_ind(rect_start_c, map_sz);
						let rect_end_c = Coord::frm_ind(rect_end_c, map_sz);
						
						start_coord_use(rect_start_c, rect_end_c, map_sz)
					};
					
					// compute paths & move all units in group
					for i_off in 0..rect_sz.h as isize {
					for j_off in 0..rect_sz.w as isize {
					if let Some(coord) = map_sz.coord_wrap(rect_start_c.y + i_off, rect_start_c.x + j_off) {
						let exf = exs.last().unwrap();
						
						if let Some(ex) = exf.get(&coord) {
						if let Some(unit_inds) = &ex.unit_inds {
							for group_unit_ind in unit_inds.clone().iter() {
								let u = &units[*group_unit_ind];
								
								// not owned by player or original moving unit (latter case handled below)
								if u.owner_id != moving_owner || *group_unit_ind == unit_ind {return false;}
								
								// destination relative to the source (arbitrarily chosen unit)
								let mv_dest = {
									let u_coord = Coord::frm_ind(u.return_coord(), map_sz);
									Coord {
										y: mv_dest.y + u_coord.y - src_u_orig_coord.y,
										x: mv_dest.x + u_coord.x - src_u_orig_coord.x
									}
								};
								
								let mut action_iface = unit_action_iface(Coord::frm_ind(u.return_coord(), map_sz),
													ActionType::Mv, *group_unit_ind, units);
								
								action_iface.update_move_search(mv_dest, map_data, exs,
										MvVars::NonCivil{units, start_owner: u.owner_id, blind_undiscov: Some(&stats[u.owner_id as usize].land_discov)}, bldgs);
								
								if action_iface.action.path_coords.len() > 0 {
									let u = &mut units[*group_unit_ind];
									u.action.pop();
									u.action.push(action_iface.action);
									
									mv_unit(*group_unit_ind, true, units, map_data, exs, bldgs, stats, relations, owners, map_sz, DelAction::Delete {barbarian_states, ai_states}, logs, turn);
								}
							}
						}}
					}}}
				}
				
				self.reset_unit_subsel();
				self.update_all_player_pieces_mvd_flag(units);
				return true;
				
			// no path computed -- remove action
			}else{
				u.action.pop();
			}
			
			return false;
		
		// req there be a building that's unfinished  
		}else if let ActionType::WorkerContinueBuildBldg = action.action_type {
			if let Some(bldg_ind) = worker_can_continue_bldg(true, bldgs, map_data, exs.last().unwrap(), self) {
				action.action_type = ActionType::WorkerBuildBldg {
					valid_placement: true,
					template: bldgs[bldg_ind].template,
					doctrine_dedication: None,
					bldg_coord: None
				};
				
				//if self.add_action_to.specific_unit_or_units() {
				//	let action = self.add_action_to.first_action_mut().unwrap();
					action.path_coords.remove(0); // do not attempt to move onto the building
				//}
			}else{return false;}
		}
		
		let mut disband_unit_inds = Vec::new();
		
		// start move
		let action = units[unit_ind].action.last().unwrap();
		if action.path_coords.len() > 0 {
			match action.action_type {
				ActionType::WorkerBuildStructure {..} => {},
				_ => {mv_unit(unit_ind, true, units, map_data, exs, bldgs, stats, relations, owners, map_sz, DelAction::Record(&mut disband_unit_inds), logs, turn);}
			}
		}
		
		// attacks (if relevant and not disbanded (ex. boarding boat))
		do_attack_action(unit_ind, &mut disband_unit_inds, units, bldg_config, bldgs, tech_templates, unit_templates, bldg_templates, 
			doctrine_templates, stats, relations, ai_states, barbarian_states,
			map_data, exs, zone_exs_owners, owners, logs, self, disp_chars, disp_settings, menu_options, self.cur_player_paused(ai_states),
			map_sz, frame_stats, turn, rng, kbd, l, buttons, txt_list, d);
		
		// do not allow repeat attacking
		if !disband_unit_inds.contains(&unit_ind) {
			let u = &mut units[unit_ind];
			if let Some(action) = u.action.last() {
				if let ActionType::Attack {..} = action.action_type {
					u.actions_used = None;
				}
			}
		}
		
		// could occur even when not attacking due to boarding boat
		disband_units(disband_unit_inds, self.cur_player, units, map_data, exs, stats, relations, barbarian_states, ai_states, owners, map_sz, logs, turn);
		
		return true;
	}
	
	// Step 1:
	// returns true if the action can be executed now, false if additional information needed
	// from the player (mostly used w/ building actions, to check and set the start and end locations)
	// may alter `action` with action-specific pre-processing
	// `cur`_mi is the cursor map location
	// ***Note: not run when AddActionTo::AllInBrigade
	pub fn pre_process_action_chk_valid(&mut self, cur_mi: u64, units: &Vec<Unit<'bt,'ut,'rt,'dt>>,
			bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>,
			exf: &HashedMapEx, map_data: &mut MapData) -> bool {
		debug_assert!(self.add_action_to.is_build_list() || self.add_action_to.is_individual_unit());
		
		// the start coordinate was not set, prevent the action from being set in the unit or build list
		if !self.set_action_start_coord_if_not_set(cur_mi, units, exf, map_data) {
			return false;
		}
		
		////////////////////////
		// check that action is valid	
		return match &mut self.add_action_to.first_action_mut().unwrap().action_type {
			// if building or zoning, req valid placement
			ActionType::WorkerBuildBldg {valid_placement: false, ..} |
			ActionType::WorkerZone {valid_placement: false, ..} => {false
			
			// req zone be drawn already
			} ActionType::WorkerZone {ref mut end_coord, start_coord, valid_placement: true, ..} => {
				// don't allow rectangles of size one
				if cur_mi != start_coord.unwrap() {
					*end_coord = Some(cur_mi);
					true
				}else{
					false
				}
			
			// set end coord, if the start coord was already set
			} ActionType::WorkerRmZonesAndBldgs {ref mut end_coord, start_coord: Some(start_coord)} => {
				// don't allow rectangles of size one
				if cur_mi != *start_coord {
					*end_coord = Some(cur_mi);
					true
				}else{
					false
				}

			// req there be a building that's unfinished  
			} ActionType::WorkerContinueBuildBldg => {
				if !worker_can_continue_bldg(true, bldgs, map_data, exf, self).is_none() {
					true
				}else{false}
				
			//////////////////////////////
			// repair wall: last move is the wall, remove it from path coords, store in `action_type`
			} ActionType::WorkerRepairWall {..} => {
				if self.add_action_to.specific_unit_or_units() {
					let action = self.add_action_to.first_action_mut().unwrap();
					if action.path_coords.len() == 0 {return false;} // we need a wall to repair
					action.path_coords.remove(0);
					
					if let ActionType::WorkerRepairWall {ref mut wall_coord, ..} = action.action_type {
						*wall_coord = Some(cur_mi);
					}else{panicq!("invalid action");}
				}
				true
			} _ => {true}
		};
	}
	
	// Step 0:
	// used w/ actions requiring rectangular selection (ex. zones)
	// returns true if the start coordinate was already set (or not needed), false if 
	// it was either set in this function or the coordinate is invalid
	pub fn set_action_start_coord_if_not_set(&mut self, cur_mi: u64, units: &Vec<Unit<'bt,'ut,'rt,'dt>>,
			exf: &HashedMapEx, map_data: &mut MapData) -> bool {
		return if let Some(action) = self.add_action_to.first_action_mut() {
			match &mut action.action_type {
				// check if the proposed start coordinate is clear
				ActionType::WorkerZone {start_coord: None, ..} |
				ActionType::WorkerRmZonesAndBldgs {start_coord: None, ..} => {
					// if an individual unit is moving, see if it can travel to the start location
					if let AddActionTo::IndividualUnit {action_iface} = &self.add_action_to {
						let unit_ind = action_iface.unit_ind.unwrap();
						let u = &units[unit_ind];
					
						// if the cursor is not on the unit and there's no path, this is not a valid location
						if cur_mi != u.return_coord() && action_iface.action.path_coords.len() == 0 {return false;}
					}
					
					// set start_coord
					if let ActionType::WorkerRmZonesAndBldgs {ref mut start_coord, ..} = self.add_action_to.first_action_mut().unwrap().action_type {
						*start_coord = Some(cur_mi);
					
					// check if the zone placement is valid
					}else if land_clear_ign_units_roads(cur_mi, self.cur_player, &map_data.get(ZoomInd::Full, cur_mi), exf) {
						if let ActionType::WorkerZone {ref mut start_coord, ..} = self.add_action_to.first_action_mut().unwrap().action_type {
							*start_coord = Some(cur_mi);
						}else{panicq!("invalid action");}
					}
					
					false
				
				} ActionType::WorkerZone {start_coord: Some(_), ..} |
				  ActionType::WorkerRmZonesAndBldgs {start_coord: Some(_), ..} => {true
				
				// accept any position as the start coordinate
				} ActionType::GroupMv {ref mut start_coord, ..} |
				  ActionType::BrigadeCreation {ref mut start_coord, ..} |
				  ActionType::SectorCreation {ref mut start_coord, ..} => {
					if start_coord.is_none() {
						*start_coord = Some(cur_mi);
						false
					}else{true}
					
				// actions w/o rectangle drawing
				} ActionType::Mv | ActionType::MvWithCursor | ActionType::MvIgnoreWalls |
				  ActionType::MvIgnoreOwnWalls | ActionType::CivilianMv | ActionType::AutoExplore {..} |
				  ActionType::WorkerBuildStructure {..} | ActionType::WorkerRepairWall {..} |
				  ActionType::SectorAutomation {..} |
				  ActionType::WorkerBuildBldg {..} | ActionType::Attack {..} | ActionType::Fortify {..} |
				  ActionType::WorkerZoneCoords {..} | ActionType::UIWorkerAutomateCity | ActionType::BurnBuilding {..} |
				  ActionType::WorkerContinueBuildBldg {..} => {true} 
			}
		}else{false};
	}
}

