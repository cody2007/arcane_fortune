use super::*;
use crate::map::*;
use crate::movement::*;

const TURNS_RECHECK_FORTIFY: usize = 15;

// common actions among empires and nobility:
//	attack, defense, & worker actions
impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	// if target_city_coord_opt is some value, then that location is targeted exclusively
	pub fn common_actions(ai_ind: usize, is_cur_player: bool, target_city_coord_opt: Option<u64>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			personality: AIPersonality, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			gstate: &mut GameState, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			disband_unit_inds: &mut Vec<usize>, map_sz: MapSz, temps: &Templates<'bt,'ut,'rt,'dt,'_>) {
		#[cfg(feature="profile")]
		let _g = Guard::new("AIState::common_actions");
		
		{ // city unit actions (produce new buildings and improvements)
			#[cfg(feature="profile")]
			let _g = Guard::new("city unit actions");
			
			let boot_camp_template = &BldgTemplate::frm_str(BOOT_CAMP_NM, temps.bldgs);
			let academy_template = &BldgTemplate::frm_str(ACADEMY_NM, temps.bldgs);
			let pstats = &players[ai_ind].stats;
			let net_income = pstats.net_income(players, &gstate.relations);
			
			let player = &mut players[ai_ind];
			let pstats = &mut player.stats;
			if let Some(ai_state) = player.ptype.any_ai_state_mut() {
				if ai_state.next_bonus_bldg.is_none() {ai_state.set_next_bonus_bldg(pstats, &personality, temps.bldgs, &mut gstate.rng);}
				
				let mut next_bonus_bldg = ai_state.next_bonus_bldg.as_ref().unwrap_or_else(|| panicq!("next bonus bldg unset"));
				let is_offense_strong_enough = ai_state.is_offense_strong_enough(pstats, &personality, units, bldgs);
				let any_academy_constructed = ai_state.city_states.iter().any(|c| !c.academy_ind.is_none());
				
				for city_ind in 0..ai_state.city_states.len() {
					let city = &mut ai_state.city_states[city_ind];
					
					city.execute_defense_actions(ai_ind, is_cur_player, units, bldgs, pstats, gstate, map_data, exs, disband_unit_inds, map_sz);
					if gstate.rng.gen_f32b() < 0.25 {
						city.execute_worker_actions(is_cur_player, units, pstats, bldgs, map_data, exs, gstate, map_sz);
					}
					
					/////////// worker actions
					{
						let exf = exs.last_mut().unwrap();
						
						macro_rules! add_bldg_action{($bldg_template: expr) => {
							let mut already_planned = false;
							
							// check if action not already planned
							for action in city.worker_actions.iter().rev() {
								if let ActionType::WorkerBuildBldg {template, ..} = &action.action_type {
									if template == $bldg_template {already_planned = true; break;}
								}
							}
							
							// check if no workers are presently working on it
							if !already_planned {
								for unit_ind in city.worker_inds.iter().rev() {
									for action in units[*unit_ind].action.iter() {
										if let ActionType::WorkerBuildBldg {template, ..} = &action.action_type {
											if template == $bldg_template {already_planned = true; break;}
										}
									}
								}
								
								if !already_planned {
									const CH_DIST: isize = 20;
									let mut search_start = Coord::frm_ind(city.coord, map_sz);
									search_start.y += gstate.rng.isize_range(-CH_DIST, CH_DIST);
									search_start.x += gstate.rng.isize_range(-CH_DIST, CH_DIST);
									
									if let Some(boot_camp_coord) = find_square_buildable(search_start, $bldg_template, map_data, exf, map_sz) {
										city.worker_actions.push( ActionMeta {
											action_type: ActionType::WorkerBuildBldg{
												valid_placement: true,
												template: $bldg_template,
												doctrine_dedication: ai_state.goal_doctrine,
												bldg_coord: None
											},
											actions_req: 1.,
											path_coords: vec!{boot_camp_coord; 1},
											action_meta_cont: None
										});
										//dbg_log(&format!("worker_actions len {}", city.worker_actions.len()), owner.id, logs, turn);
									} // buildable location
								}
							}
						};}
						
						// 1. wait for city hall to be constructed
						// 2. repair walls
						// 3. build boot camp
						// 4. build academy
						// 5. build bonus bldg
						if city.population_center_ind != None {
							// repair walls
							if city.damaged_wall_coords.len() != 0 {
								'wall_loop: for damaged_wall_coord in city.damaged_wall_coords.iter() {
									let damaged_wall_coord = damaged_wall_coord.to_ind(map_sz) as u64;
									
									macro_rules! skip_if_repair{($action: expr) => {
										if let ActionType::WorkerRepairWall {wall_coord: Some(coord), ..} = &$action.action_type {
											if *coord == damaged_wall_coord {
												continue 'wall_loop;
											}
										}
									};}
									
									// check that it's not being actively repaired
									for worker_ind in city.worker_inds.iter() {
										for action in units[*worker_ind].action.iter().rev() {skip_if_repair!(action);}
									}
									
									// check that it's not already scheduled to be repaired
									for action in city.worker_actions.iter() {skip_if_repair!(action);}
									
									city.worker_actions.push(ActionMeta::new(
										ActionType::WorkerRepairWall {
											wall_coord: Some(damaged_wall_coord),
											turns_expended: 0
										}));
								}
							}else if city.boot_camp_ind == None {
								add_bldg_action!(boot_camp_template);
							}else if !any_academy_constructed && net_income >= academy_template.upkeep {
								add_bldg_action!(academy_template);
							// bonus bldg
							}else if is_offense_strong_enough && net_income >= next_bonus_bldg.upkeep {
								add_bldg_action!(next_bonus_bldg);
								ai_state.set_next_bonus_bldg(pstats, &personality, temps.bldgs, &mut gstate.rng);
								next_bonus_bldg = ai_state.next_bonus_bldg.as_ref().unwrap_or_else(|| panicq!("next bonus bldg unset"));
								//printlnq!("next bonus bldg {}", next_bonus_bldg.nm[0]);
							}
						}
					}
				} // city action loop
			}
		}
		
		{ // building productions: (produce new workers, defense units, attack units)
			#[cfg(feature="profile")]
			let _g = Guard::new("building productions");
			
			const WORKERS_PER_CITY: usize = 2;
			const MIN_DEFENDERS_BEFORE_CONSTRUCTING_ACADEMY: usize = 2;
			
			let pstats = &players[ai_ind].stats;
			let max_defensive_unit = pstats.max_defensive_unit(temps.units);
			let net_income = pstats.net_income(players, &gstate.relations);
			
			let player = &mut players[ai_ind];
			let pstats = &player.stats;
			if let Some(ai_state) = player.ptype.any_ai_state_mut() {
				for city in ai_state.city_states.iter() {
					// city hall
					if let Some(population_center_ind) = city.population_center_ind {
						if let BldgArgs::PopulationCenter {ref mut production, ..} = &mut bldgs[population_center_ind].args {
							// check if not producing anything already and we do not already have enough workers for this city
							if production.len() == 0 && city.worker_inds.len() < WORKERS_PER_CITY {
								// produce new worker
								production.push(ProductionEntry {
									production: UnitTemplate::frm_str(WORKER_NM, temps.units),
									progress: 0
								});
							}
						}else{panicq!("ai city hall has no population center arguments");}
					}
					
					// boot camp
					// if an academy is not already contructed, only produce MIN_DEFENDERS_BEFORE_CONSTRUCTING_ACADEMY
					if city.defenders.len() < MIN_DEFENDERS_BEFORE_CONSTRUCTING_ACADEMY || city.academy_ind != None {
						if let Some(boot_camp_ind) = city.boot_camp_ind {
							if let BldgArgs::GenericProducable {ref mut production} = &mut bldgs[boot_camp_ind].args {
								// check if not producing anything already and we do not already have enough defenders for this city
								// and we can afford it
								if production.len() == 0 {
									// produce new defensive unit
									if city.defenders.len() < city.max_defenders() && net_income >= max_defensive_unit.upkeep {
										production.push(ProductionEntry {
											production: max_defensive_unit,
											progress: 0
										});
										//dbg_log("setting bootcamp", owner.id, logs, turn);
									
									// produce new attack unit
									}else if let Some(next_attack_unit) = ai_state.attack_fronts.next_req_unit(ai_state.city_states.len(), pstats, temps.units) {
										if net_income >= next_attack_unit.upkeep {
											production.push(ProductionEntry {
												production: next_attack_unit,
												progress: 0
											});
										}
									}
								}
							}else{panicq!("boot camp has no arguments");}
						}
					}
				} // building productions
			}
		}
		
		if let Some(ai_state) = players[ai_ind].ptype.any_ai_state_mut() {
			ai_state.attack_fronts.execute_actions(ai_ind, target_city_coord_opt, units, bldgs, map_data, exs, gstate, map_sz);//, iface_settings);
			
			//////////////////////////////
			// icbm actions
			if ai_state.icbm_inds.len() > 0 && gstate.rng.gen_f32b() < (1./10.){
				#[cfg(feature="profile")]
				let _g = Guard::new("icbm actions");
				
				let war_enemies = gstate.relations.at_war_with(ai_ind);
				
				if war_enemies.len() != 0 {
					'icbm_loop: for icbm_ind in ai_state.icbm_inds.iter() {
						let u = &units[*icbm_ind];
						if u.action.len() != 0 {continue;} // unit pressumably already moving to a target
						
						// find target
						for b in bldgs.iter() {
							if !war_enemies.contains(&(b.owner_id as usize)) {continue;}
							
							// attack
							if let BldgArgs::PopulationCenter {..} = &b.args {
								let action_type = ActionType::Attack {
									attack_coord: Some(b.coord),
									attackee:  Some(b.owner_id),
									ignore_own_walls: false
								};
								
								if set_target_attackable(&action_type, *icbm_ind, true, AI_MAX_SEARCH_DEPTH, units, bldgs, exs, map_data, map_sz) {
									break 'icbm_loop; // success
								}
							}
						}
						
						// no targets found
						break 'icbm_loop;
					}
				}
			}
		}
	}
}

// defensive & worker actions
impl <'bt,'ut,'rt,'dt>CityState<'bt,'ut,'rt,'dt> {
	pub fn execute_defense_actions(&mut self, ai_ind: usize, is_cur_player: bool, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>,
			pstats: &mut Stats<'bt,'ut,'rt,'dt>, gstate: &mut GameState, map_data: &mut MapData,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, disband_unit_inds: &mut Vec<usize>, map_sz: MapSz) {
		#[cfg(feature="profile")]
		let _g = Guard::new("execute_defense_actions");
		
		////////////////
		// first check if there are any nearby enemies that need to be attacked
		if self.defenders.len() > 0 {
			#[cfg(feature="profile")]
			let _g = Guard::new("execute_defense_actions (nearby enemies)");
			
			let war_enemies = gstate.relations.at_war_with(ai_ind);
			
			let mut ordered_defenders = Vec::with_capacity(self.defenders.len()); // index into self.defenders NOT units[]
			
			'unowned_unit_loop: for unowned_unit_ind in 0..units.len() {
				let unowned_unit = &units[unowned_unit_ind];
				
				if unowned_unit.owner_id == ai_ind as SmSvType {continue;}
				
				let unowned_owner = unowned_unit.owner_id as usize;
				
				// at war w/ and close enough to attack
				if (war_enemies.contains(&unowned_owner)) && self.within_city_defense_area(Coord::frm_ind(unowned_unit.return_coord(), map_sz)) {
					// order defenders
					// 	add idlers, then fortified
					if ordered_defenders.len() == 0 {
						// idlers
						for (list_ind, defender) in self.defenders.iter().enumerate() {
							if units[defender.unit_ind].action.len() == 0 {
								ordered_defenders.push(list_ind);
							}
						}
						
						// fortified
						for (list_ind, defender) in self.defenders.iter().enumerate() {
							if let Some(action) = &units[defender.unit_ind].action.last() {
								if let ActionType::Fortify {turn: turn_fortified} = &action.action_type {
									if (*turn_fortified + TURNS_RECHECK_FORTIFY) < gstate.turn {
										ordered_defenders.push(list_ind);
									}
								}
							}
						}
						
						// no one available
						if ordered_defenders.len() == 0 {break 'unowned_unit_loop;}
					}
					
					let attack_coord = unowned_unit.return_coord();
					
					let action_type = ActionType::Attack {
						attack_coord: Some(attack_coord),
						attackee:  Some(unowned_owner as SmSvType),
						ignore_own_walls: true
					};
					
					// find defender in `ordered_defenders` to attack `unowned_unit`
					'defender_loop: for list_ind in ordered_defenders.iter().rev() {
						let defender_unit_ind = self.defenders[*list_ind].unit_ind;
						let u = &mut units[defender_unit_ind];
						
						u.action.clear();
						
						// if moving over land & not ignoring walls, exit city first (to prevent being blocked by the wall)
						match u.template.movement_type {
							MovementType::Land => {
								let mut max_search_depth = AI_MAX_SEARCH_DEPTH;
								if !self.exit_city(&mut max_search_depth, defender_unit_ind, is_cur_player, false,
									Coord::frm_ind(attack_coord, map_sz),
									Coord::frm_ind(units[defender_unit_ind].return_coord(), map_sz),
									pstats, map_data, exs, units, bldgs, gstate, map_sz) {
								continue 'defender_loop;
							}}
							
							MovementType::AllWater | MovementType::ShallowWater |
							MovementType::LandAndOwnedWalls |
							MovementType::Air => {}
							MovementType::AllMapTypes | MovementType::N => {panicq!("invalid movement type");}
						}

						if set_target_attackable(&action_type, defender_unit_ind, true, AI_MAX_SEARCH_DEPTH, units, bldgs, exs, map_data, map_sz) {
							// target set successfully
							ordered_defenders.pop();
							continue 'unowned_unit_loop;
						}//else{
						//	printlnq!("defender failed to attack. from {} to {}, owner {}",
						//			Coord::frm_ind(units[defender_unit_ind].return_coord(), map_sz),
						//			Coord::frm_ind(units[unowned_unit_ind].return_coord(), map_sz), owner.nm);
						//}
					}
				}
			}
		}
		
		////////////////////
		// if no need to attack, move to defense positions
		{
			#[cfg(feature="profile")]
			let _g = Guard::new("execute_defense_actions (mv to defense positions)");
			
			let next_defense_pos_ind_opt = self.next_unfilled_defense_pos_ind(units);
			
			// units that have no current actions or if there's a new defense position, include fortified units that are at lower priority
			//for defender in self.defenders.iter_mut() {
			for defender_list_ind in 0..self.defenders.len() {
				let defender = &self.defenders[defender_list_ind];
				let u = &mut units[defender.unit_ind];
				debug_assertq!(u.owner_id == ai_ind as SmSvType && u.template.nm[0] != WORKER_NM);
				
				let chk_alter = || {
					// not doing any actions -- either fortify and move to next position
					// (unit will be fortified after, so it must not already be at a defense position)
					if u.actions_used == Some(0.) && u.action.len() == 0 {
						return true;
					}
					
					// if fortified, only check so often
					if let Some(ActionMeta {action_type: ActionType::Fortify {turn: turn_fortified}, ..}) = u.action.last() {
						if (*turn_fortified + TURNS_RECHECK_FORTIFY) > gstate.turn {return false;}
					}
					
					// if a new defense position is available, include units that are:
					// 	-fortified at positions of lower priority than the next available position
					// 	-fortified not in a defensive position
					if let Some(next_defense_pos_ind) = next_defense_pos_ind_opt {
						if let Some(ActionMeta {action_type: ActionType::Fortify {turn: turn_fortified}, ..}) = u.action.last() {
							// only check every so often
							if (*turn_fortified + TURNS_RECHECK_FORTIFY) < gstate.turn {
								// unit already at a defensive position
								if let Some(cur_defense_pos_ind) = self.current_defense_pos_ind(u.return_coord()) {
									// higher priority, move to next defense pos
									if cur_defense_pos_ind > next_defense_pos_ind {
										return true;
									}
								// unit not at a defensive position
								}else {return true;}
							}
						}
					}
					
					// either the unit has actions or it is fortified at a higher priority defense position already, so skip:
					false
				};
				
				if !chk_alter() {continue;}
				
				let u_coord = u.return_coord();
				
				// fortify (pushed first, executed last if movement is pushed after)
				u.action.clear();
				u.action.push(ActionMeta::new(ActionType::Fortify {turn: gstate.turn}));
				
				// move unit to next defense position
				if let Some(next_defense_pos_ind) = next_defense_pos_ind_opt {
					let mut action_iface = ActionInterfaceMeta {
						action: ActionMeta::new(ActionType::Mv),
						unit_ind: Some(defender.unit_ind),
						max_search_depth: AI_MAX_SEARCH_DEPTH,
						start_coord: Coord::frm_ind(u_coord, map_sz), // starting location of unit
						movement_type: u.template.movement_type,
						movable_to: &movable_to
					};
					
					action_iface.update_move_search(Coord::frm_ind(self.defense_positions[next_defense_pos_ind], map_sz), map_data, exs, MvVars::NonCivil{units, start_owner: ai_ind as SmSvType, blind_undiscov: None}, bldgs);
					
					let defender = &mut self.defenders[defender_list_ind];
					
					// move possible, send unit on their way
					if action_iface.action.path_coords.len() > 0 {
						units[defender.unit_ind].action.push(action_iface.action);
						defender.failed_mv_attempts = 0;
						//dbg_log("mv possible", owner.id, logs, turn);
					}else{
						//dbg_log("mv not possible", owner.id, logs, turn);
						
						// delete unit if it has consistently failed to move to the defense position
						defender.failed_mv_attempts += 1;
						if defender.failed_mv_attempts > MAX_FAILED_MV_ATTEMPTS {
							//disband_unit(defender.unit_ind, units, map_data, exs, stats, relations, barbarian_states, ai_states, owners, map_sz, logs, turn);
							// ^ cannot do this here because it requires a mutable copy of ai_states
							disband_unit_inds.push(defender.unit_ind);
						}else{
							units[defender.unit_ind].action.push(ActionMeta::new(ActionType::Fortify {turn: gstate.turn}));
						}
					}
				}
			}
		}
	}
	
	pub fn execute_worker_actions(&mut self, is_cur_player: bool, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, pstats: &mut Stats<'bt,'ut,'rt,'dt>,
			bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, map_data: &mut MapData,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz) {
		if self.worker_actions.len() == 0 {return;}
		#[cfg(feature="profile")]
		let _g = Guard::new("execute_worker_actions");
		
		// find first unit that has no current actions
		if let Some(&unit_ind) = self.worker_inds.iter().find(|&&unit_ind| {
			let u = &units[unit_ind];
			//debug_assertq!(u.owner_id == owner.id);
			u.template.nm[0] == WORKER_NM && 
			u.actions_used == Some(0.) &&
			u.action.len() == 0
		}) {
			self.set_worker_action(unit_ind, is_cur_player, units, pstats, bldgs, map_data, exs, gstate, map_sz);
		} // idle worker exists
	}
	
	pub fn set_worker_action(&mut self, unit_ind: usize, is_cur_player: bool, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, pstats: &mut Stats<'bt,'ut,'rt,'dt>,
			bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz) {
		if let Some(action) = self.worker_actions.pop() {
			action.set_action_meta(unit_ind, is_cur_player, units, pstats, bldgs, map_data, Some(self), exs, gstate, map_sz);
		}
	}
	
	// set unit to be just outside the wall, then search a bit further with astar
	// 	`min_dist_chk` if true, requires destination be far from target for the unit to be placed outside the city
	// returns:
	//	true if we've either exited the city or probably already out or not exiting
	//	false if we cannot exit
	pub fn exit_city(&self, max_search_depth: &mut usize, unit_ind: usize, is_cur_player: bool, min_dist_chk: bool, 
			start_coord: Coord, u_coord: Coord, pstats: &mut Stats<'bt,'ut,'rt,'dt>, map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz) -> bool {
		#[cfg(feature="profile")]
		let _g = Guard::new("exit_city");
		
		// not traveling far
		if min_dist_chk && manhattan_dist(start_coord, u_coord, map_sz) <= (CITY_WIDTH + CITY_HEIGHT) {return true;}
		
		// not close enough to city
		if manhattan_dist(u_coord, Coord::frm_ind(self.coord, map_sz), map_sz) >= 2*(CITY_WIDTH + CITY_HEIGHT) {return true;}
		
		*max_search_depth *= 2*5; // was 2
		//dbg_log(&format!("attempting to go further {} {} -> {} {}", u_coord.y, u_coord.x, start_coord.y, start_coord.x), owner.id, logs, turn);
		
		// find the closest wall to the final destination (start_coord -- the start of the next action)
		let mut min_coord = *self.wall_coords.iter().min_by_key(|c| manhattan_dist(start_coord, **c, map_sz)).unwrap();
		
		// move just outside the wall
		const WALL_DIST: isize = 3;
		min_coord.y += if min_coord.y < start_coord.y {WALL_DIST} else {-WALL_DIST};
		min_coord.x += if min_coord.x < start_coord.x {WALL_DIST} else {-WALL_DIST};
		
		// check that location is valid and movable to, then set unit to that location
		if let Some(min_coord_ind) = map_sz.coord_wrap(min_coord.y, min_coord.x) {
			let u_coord_ind = units[unit_ind].return_coord();
			if movable_to(u_coord_ind, min_coord_ind, &map_data.get(ZoomInd::Full, min_coord_ind), exs.last().unwrap(), MvVarsAtZoom::NonCivil {units, start_owner: pstats.id, blind_undiscov: None},
					bldgs, &Dest::NoAttack, units[unit_ind].template.movement_type) {
				set_coord(min_coord_ind, unit_ind, is_cur_player, units, map_data, exs, pstats, map_sz, gstate);
			}else {return false;} // can't travel outside city
		}else {return false;} // can't travel outside city
		
		true
	}
}

impl <'bt,'ut,'rt,'dt>CityState<'bt,'ut,'rt,'dt> {
	#[inline]
	pub fn current_defense_pos_ind(&self, coord: u64) -> Option<usize> {
		self.defense_positions.iter().position(|&pos| pos == coord)
	}
	
	#[inline]
	pub fn next_unfilled_defense_pos_ind(&self, units: &Vec<Unit>) -> Option<usize> {
		// find first position that is unoccupied
		self.defense_positions.iter().position(|&defense_pos| {
			// determine if max defenders are here
			let mut count = 0;
			for defender in self.defenders.iter() {
				if units[defender.unit_ind].return_coord() == defense_pos {
					count += 1;
					if count == 2 {return false;}
				}
			}
			true
		})
	}
}

