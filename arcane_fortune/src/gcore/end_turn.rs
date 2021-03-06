use std::time::*;
use std::cmp::{Ordering};
use crate::disp::*;
use crate::disp::menus::*;
use crate::map::*;
use crate::units::*; // do_attack_actions() <- units/attack.rs
use crate::buildings::*;
use crate::zones::*;
use crate::movement::*;
use crate::tech::{research_techs};
use crate::ai::*;
use crate::gcore::hashing::*;
use crate::player::{LOG_TURNS, Player, PlayerType};
use crate::containers::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;
use super::*;

pub fn end_turn<'f,'bt,'ut,'rt,'dt>(gstate: &mut GameState, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp<'f,'_,'bt,'ut,'rt,'dt>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, frame_stats: &mut FrameStats) {
	#[cfg(feature="profile")]
	let _g = Guard::new("end_turn");
	
	let frame_start = Instant::now();
	
	let bldg_config = &temps.bldg_config;
	
	// for menu indicators
	let cur_ui_ai_player_is_paused = disp.state.iface_settings.cur_player_paused(players);
	
	if disp.state.iface_settings.auto_turn == AutoTurn::Off {
		disp.ui_mode = UIMode::None;
		disp.state.iface_settings.add_action_to = AddActionTo::None;
	}
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	/////////////////////////
	// events
	rm_old_events(disp.state.iface_settings.cur_player, bldgs, units, gstate, &temps.bldgs, map_data, exs, players, map_sz);
	
	//////////////////////////////////////////////
	// nobility
	new_unaffiliated_nobility(players, units, bldgs, map_data, exs, gstate, temps, players[0].stats.alive_log.len(), map_sz);
	
	//////////////////////////////////////////////
	// ai & barbarians
	let mut disband_unit_inds = Vec::new();
	
	{
		#[cfg(feature="profile")]
		let _g = Guard::new("ai planning");
		
		for player_ind in 0..players.len() {
			let player = &players[player_ind];
			if !player.stats.alive {continue;}
			if let Some(AIState {paused: false, ..}) = player.ptype.any_ai_state() {
				NobilityState::plan_actions(player_ind, player_ind == disp.state.iface_settings.cur_player as usize, players, units, bldgs, map_data, exs, &mut disband_unit_inds, gstate, map_sz, temps, disp);
				EmpireState::plan_actions(player_ind, players, units, bldgs, gstate, map_data, exs, temps, &mut disband_unit_inds, map_sz, disp);
			}
			BarbarianState::plan_actions(player_ind, units, bldgs, map_data, exs, players, temps.units, map_sz, gstate);
		}
	}
	
	///////////////////////////////////////////
	{ // rioting
		#[cfg(feature="profile")]
		let _g = Guard::new("rioting");
		
		{ // citizen demands
			const MAX_PROB: f32 = 0.01;
			const DEMANDS_THRESH: f32 = -100.;
			const SATURATING_DEMANDS_THRESH: f32 = -500.; // at this happiness level,
			// the demands probability will be MAX_PROB
			
			// desired function:
			// 	prob(happiness) = PROB_FACTOR * happiness + PROB_OFFSET
			
			// constraints:
			//	(1) MAX_PROB = PROB_FACTOR * SATURATING_DEMANDS_THRESH + PROB_OFFSET
			//	(2) 0 = PROB_FACTOR * DEMANDS_THRESH + PROB_OFFSET
			//
			// solving constraints for PROB_FACTOR, PROB_OFFSET
			// 	(3) => PROB_FACTOR = -PROB_OFFSET / DEMANDS_THRESH (from eq 2)
			//	=> MAX_PROB = -(PROB_OFFSET / DEMANDS_THRESH)*SATURATING_DEMANDS_THRESH + PROB_OFFSET (eq 3 into eq 1)
			//	=> MAX_PROB = PROB_OFFSET * (1 - SATURATING_DEMANDS_THRESH/DEMANDS_THRESH)
			//	(4) => PROB_OFFSET = MAX_PROB / (1-SATURATING_DEMANDS_THRESH/DEMANDS_THRESH) 
			//
			//	=> PROB_FACTOR = -MAX_PROB / (DEMANDS_THRESH - SATURATING_DEMANDS_THRESH) ** (eq 4 into eq 3)
			
			const PROB_FACTOR: f32 = -MAX_PROB / (DEMANDS_THRESH - SATURATING_DEMANDS_THRESH);
			const PROB_OFFSET: f32 = MAX_PROB / (1. - SATURATING_DEMANDS_THRESH/DEMANDS_THRESH);
			
			const LOG_SPACING_DAYS: usize = 30*12*5;
			
			// create demands
			'demand_loop: for player in players.iter() {
				let pstats = &player.stats;
				if !pstats.alive || pstats.locally_logged.happiness_sum > DEMANDS_THRESH {continue;}
				
				let mut prob = PROB_FACTOR * pstats.locally_logged.happiness_sum + PROB_OFFSET;
				if prob > MAX_PROB {prob = MAX_PROB};
				
				if gstate.rng.gen_f32b() < prob {
					let contrib = &pstats.locally_logged.contrib;
					let pos_sum = contrib.doctrine + contrib.pacifism;
					let neg_sum = contrib.health + contrib.unemployment + contrib.crime;
					
					// see zones/happiness.rs
					let vals = [contrib.doctrine/pos_sum,
							contrib.pacifism/pos_sum,
							contrib.health/neg_sum,
							contrib.unemployment/neg_sum,
							contrib.crime/neg_sum];
					
					let max_ind = {
						let mut max_ind = 0;
						let mut max_val = vals[0];
						for (ind, val) in vals.iter().enumerate() {
							if max_val >= *val {continue;}
							max_ind = ind;
							max_val = *val;
						}
						max_ind
					};
					
					let reason = match max_ind {
						0 => HappinessCategory::Doctrine,
						1 => {
							let pacifism_militarism = if pstats.locally_logged.pacifism_sum < 0. {
								PacifismMilitarism::Militarism
							}else{
								PacifismMilitarism::Pacifism
							};
							HappinessCategory::PacifismOrMilitarism(pacifism_militarism)
						}
						2 => HappinessCategory::Health,
						3 => HappinessCategory::Unemployment,
						4 => HappinessCategory::Crime,
						_ => {panicq!("invalid max ind for happiness reason");}
					};
					
					// prevent logging too frequently
					for log in gstate.logs.iter().rev() {
						// no logs found, do not go further back in time
						if (log.turn + LOG_SPACING_DAYS) <= gstate.turn {break;}
						
						if let LogType::CitizenDemand {owner_id, reason: reason_logged} = &log.val {
							if *owner_id == player.id as usize && *reason_logged == reason {
								continue 'demand_loop;
							}
						}
					}
					
					gstate.log_event(LogType::CitizenDemand {owner_id: player.id as usize, reason});
					
					if player.id == disp.state.iface_settings.cur_player {
						disp.create_interrupt_window(UIMode::CitizenDemandAlert(CitizenDemandAlertState {reason}));
					}
				}
			}
		}
		
		const RIOTING_THRESH: f32 = -275.;
		const MAX_PROB: f32 = 0.1;
		
		let ut = UnitTemplate::frm_str(RIOTER_NM, temps.units);
		
		{ // create rioters
			const SATURATING_RIOTING_THRESH: f32 = -500.; // at this happiness level,
			// the rioting probability will be MAX_PROB
			
			// desired function:
			// 	prob(happiness) = PROB_FACTOR * happiness + PROB_OFFSET
			
			// constraints:
			//	(1) MAX_PROB = PROB_FACTOR * SATURATING_RIOTING_THRESH + PROB_OFFSET
			//	(2) 0 = PROB_FACTOR * RIOTING_THRESH + PROB_OFFSET
			//
			// solving constraints for PROB_FACTOR, PROB_OFFSET
			// 	(3) => PROB_FACTOR = -PROB_OFFSET / RIOTING_THRESH (from eq 2)
			//	=> MAX_PROB = -(PROB_OFFSET / RIOTING_THRESH)*SATURATING_RIOTING_THRESH + PROB_OFFSET (eq 3 into eq 1)
			//	=> MAX_PROB = PROB_OFFSET * (1 - SATURATING_RIOTING_THRESH/RIOTING_THRESH)
			//	(4) => PROB_OFFSET = MAX_PROB / (1-SATURATING_RIOTING_THRESH/RIOTING_THRESH) 
			//
			//	=> PROB_FACTOR = -MAX_PROB / (RIOTING_THRESH - SATURATING_RIOTING_THRESH) ** (eq 4 into eq 3)
			
			const PROB_FACTOR: f32 = -MAX_PROB / (RIOTING_THRESH - SATURATING_RIOTING_THRESH);
			const PROB_OFFSET: f32 = MAX_PROB / (1. - SATURATING_RIOTING_THRESH/RIOTING_THRESH);
			
			const LOG_SPACING_DAYS: usize = 30*5;
			
			// create rioters
			'owner_loop: for owner_ind in 0..players.len() {
				let player = &players[owner_ind];
				let pstats = &player.stats;
				if !pstats.alive || pstats.locally_logged.happiness_sum > RIOTING_THRESH {continue;}
				
				let mut prob = PROB_FACTOR * pstats.locally_logged.happiness_sum + PROB_OFFSET;
				if prob > MAX_PROB {prob = MAX_PROB};
				
				if gstate.rng.gen_f32b() < prob {
					if let Some(coord) = sample_low_happiness_coords(&player.zone_exs, exs.last().unwrap(), map_sz, &mut gstate.rng) {
						let owner_id = owner_ind as SmSvType;
						
						//printlnq!("adding rioter to owner {} is_empire {} is_barbarian {} is_nobility {} is_human {} {}", owner_id,
						//		player.ptype.is_empire(), player.ptype.is_barbarian(), player.ptype.is_nobility(), player.ptype.is_human(), player.personalization.nm);
						add_unit(coord, owner_id == disp.state.iface_settings.cur_player, ut, units, map_data, exs, bldgs, &mut players[owner_ind], gstate, temps);
						
						{ // log and create alert window if relevant
							let c = Coord::frm_ind(coord, map_sz);
							
							// find closest city
							if let Some(b) = bldgs.iter_mut().filter(|b| {
										if let BldgArgs::PopulationCenter {..} = b.args {
											b.owner_id == owner_id
										}else{false}
									}).min_by_key(|b| 
										manhattan_dist(Coord::frm_ind(b.coord, map_sz), c, map_sz)
									) {
								if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
									// prevent logging to frequently
									for log in gstate.logs.iter().rev() {
										// no logs found, do not go further back in time
										if (log.turn + LOG_SPACING_DAYS) <= gstate.turn {break;}
										
										if let LogType::Rioting {city_nm, owner_id} = &log.val {
											if *city_nm == *nm && *owner_id == owner_ind {
												continue 'owner_loop;
											}
										}
									}
									
									// log
									gstate.log_event(LogType::Rioting {city_nm: nm.clone(), owner_id: owner_ind});
									
									// create alert window
									if owner_id == disp.state.iface_settings.cur_player {
										disp.create_interrupt_window(UIMode::RiotingAlert(RiotingAlertState {city_nm: nm.clone()}));
										disp.center_on_next_unmoved_menu_item(true, FindType::Coord(coord), map_data, exs, units, bldgs, gstate, players);
									}
								}else{panicq!("rioting did not filter non-city halls");}
							}
						}
					}
				}
			}
		}
		
		{ // rm rioters
			const SATURATING_RIOTING_THRESH: f32 = 0.;
			
			// desired function:
			// 	prob(happiness) = PROB_FACTOR * happiness + PROB_OFFSET
			
			// constraints:
			//	(1) MAX_PROB = PROB_FACTOR * SATURATING_RIOTING_THRESH + PROB_OFFSET
			//	(2) 0 = PROB_FACTOR * RIOTING_THRESH + PROB_OFFSET
			//
			// solving constraints for PROB_FACTOR, PROB_OFFSET
			// 	(3) => PROB_FACTOR = -PROB_OFFSET / RIOTING_THRESH (from eq 2)
			//	=> MAX_PROB = -(PROB_OFFSET / RIOTING_THRESH)*SATURATING_RIOTING_THRESH + PROB_OFFSET (eq 3 into eq 1)
			//	=> MAX_PROB = PROB_OFFSET * (1 - SATURATING_RIOTING_THRESH/RIOTING_THRESH)
			//	(4) => PROB_OFFSET = MAX_PROB / (1-SATURATING_RIOTING_THRESH/RIOTING_THRESH) 
			//
			//	=> PROB_FACTOR = -MAX_PROB / (RIOTING_THRESH - SATURATING_RIOTING_THRESH) ** (eq 4 into eq 3)
			
			const PROB_FACTOR: f32 = -MAX_PROB / (RIOTING_THRESH - SATURATING_RIOTING_THRESH);
			const PROB_OFFSET: f32 = MAX_PROB / (1. - SATURATING_RIOTING_THRESH/RIOTING_THRESH);
			
			for owner_ind in 0..players.len() {
				let pstats = &players[owner_ind].stats;
				if !pstats.alive || pstats.locally_logged.happiness_sum < RIOTING_THRESH {continue;}
				
				let mut prob = PROB_FACTOR * pstats.locally_logged.happiness_sum + PROB_OFFSET;
				if prob > MAX_PROB {prob = MAX_PROB};
				
				if gstate.rng.gen_f32b() < prob {
					if let Some((unit_ind, u)) = units.iter().enumerate().find(|(_,u)|
						u.owner_id == owner_ind as SmSvType &&
						u.template == ut
					) {
						disband_unit(unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, players, gstate, map_sz);
					}
				}
			}
			
			//////////// remove old rioters
			const MAX_RIOTER_AGE: usize = 4*TURNS_PER_YEAR;
			if let Some((unit_ind, u)) = units.iter().enumerate().find(|(_,u)|
				u.template == ut &&
				u.action.len() == 0 &&
				(gstate.turn - u.creation_turn as usize) > MAX_RIOTER_AGE
			) {
				disband_unit(unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, players, gstate, map_sz);
			}
		}
		
		// assign rioter actions
		const CITY_POPULATION_THRESH: u32 = 20;
		for u in units.iter_mut().filter(|u| u.template.nm[0] == RIOTER_NM && u.action.len() == 0) {
			let u_coord = Coord::frm_ind(u.return_coord(), map_sz);
			if let Some(b) = closest_owned_non_city_hall_bldg(u_coord, CITY_POPULATION_THRESH, u.owner_id, bldgs, map_sz) {
				u.action.push(ActionMeta::new(ActionType::BurnBuilding {coord: b.coord}));
			}
		}
	}
	
	/////////////////////////////////////////
	{ // brigades: build list & repair walls
		#[cfg(feature="profile")]
		let _g = Guard::new("brigades");
		
		// repair walls
		'player_loop: for player in players.iter_mut() {
			if let Some(ai_state) = player.ptype.any_ai_state() {
				if ai_state.damaged_wall_coords.len() == 0 {continue;}
				
				let pstats = &mut player.stats;
				for brigade_ind in 0..pstats.brigades.len() {
					let brigade = &mut pstats.brigades[brigade_ind];
					if let Some(repair_sector_walls) = &brigade.repair_sector_walls {
						let repair_sector_walls = repair_sector_walls.clone();
						// sector existant
						// 1) check for a unit not already repairing a wall
						// 2) check over each entry in ai_states.damaged_wall_coords for anything in this sector
						if let Some(sector) = pstats.sector_frm_nm_checked(&repair_sector_walls) {
							for unit_ind in pstats.brigades[brigade_ind].unit_inds.iter() {
								let u = &mut units[*unit_ind];
								// 1) already repairing a wall
								if let Some(ActionMeta {
									action_type: ActionType::WorkerRepairWall {..},
									..
								}) = u.action.last() {continue;}
								
								// 2) find coordinate in sector
								for wall_coord in ai_state.damaged_wall_coords.iter()
										.filter(|wall_coord| sector.contains(wall_coord, map_sz)) {
									
									let action_meta = ActionMeta::new(ActionType::WorkerRepairWall {
										wall_coord: Some(wall_coord.to_ind(map_sz) as u64),
										turns_expended: 0
									});
									
									u.action.pop(); // remove whatever its working on now
									// ^ (if this is not done it'll start resuming the action wherever it started
									//    repairing the wall, which could result in zones or something else in an undesired location)
									
									let min_city_opt = None;
									action_meta.set_action_meta(*unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, &mut player.stats, bldgs, map_data, min_city_opt, exs, gstate, map_sz);
									continue 'player_loop;
								}
							}
						// sector no longer existant
						}else{
							let brigade = &mut pstats.brigades[brigade_ind];
							brigade.repair_sector_walls = None;
						}
					}
				}
			}
		}
		
		// build list
		for player in players.iter_mut() {
			'brigade_loop: for brigade_ind in (0..player.stats.brigades.len()).rev() {
				let brigade = &mut player.stats.brigades[brigade_ind];
				if let Some(next_action) = brigade.build_list.get(0) {
					// find unit that can do the action and set it
					if let Some(unit_ind) = brigade.unit_inds.iter().find(|&unit_ind| {
						let u = &units[*unit_ind];
						
						// only consider idle units
						if u.actions_used != Some(0.) || u.action.len() != 0 {
							return false;
						}
						
						match next_action.action_type {
							ActionType::WorkerRepairWall {..} => {
								u.template.repair_wall_per_turn != None
							} _ => {
								u.template.nm[0] == WORKER_NM
							}
						}
					}) {
						// set action
						let u = &units[*unit_ind];
						let is_cur_player = u.owner_id == disp.state.iface_settings.cur_player;
						
						let action = brigade.build_list.pop_front().unwrap();
						
						let min_city_opt = {
							if let Some(ai_state) = player.ptype.any_ai_state() {
								let unit_coord = Coord::frm_ind(u.return_coord(), map_sz);
								ai_state.city_states.iter().min_by_key(|c| 
									manhattan_dist(Coord::frm_ind(c.coord, map_sz), unit_coord, map_sz))
							}else{None}
						};
						
						action.set_action_meta(*unit_ind, is_cur_player, units, &mut player.stats, bldgs, map_data, min_city_opt, exs, gstate, map_sz);
						continue 'brigade_loop;
					}
				}
			}
		}
	}
	
	////////////////////////////////////////////
	{ // unit management
		#[cfg(feature="profile")]
		let _g = Guard::new("end_turn unit management");
		
		// attacks
		for unit_ind in 0..units.len() {
			do_attack_action(unit_ind, &mut disband_unit_inds, units, bldgs, temps, players, gstate, map_data, exs, disp, cur_ui_ai_player_is_paused, map_sz, frame_stats);
		}
		
		const MAX_UNIT_INDS_COMP: usize = 5;
		let unit_inds = gstate.rng.inds(units.len());
		//let unit_action_start = Instant::now();
		
		// all other actions aside from attacking
		'outer: for (unit_count, unit_ind) in unit_inds.iter().enumerate() {
			let u = &mut units[*unit_ind];
			
			//if (unit_count > MAX_UNIT_INDS_COMP && !players[u.owner_id as usize].ptype.is_human()) || // limit # of non-human movements per turn
			if u.actions_used.is_none() || // no actions left to take
			   disband_unit_inds.contains(&unit_ind) // to be deleted
			   {continue;}
			
			////////////// fority: restore health
			// (if no actions have been taken and we can restore health and unit is idle or fortified)
			const HEALTH_GAIN_PER_TURN: usize = 1;
			if u.actions_used == Some(0.) && u.template.max_health > u.health {
				macro_rules! inc_health{() => (u.health += HEALTH_GAIN_PER_TURN; continue 'outer;);}
				
				//if u.action.is_none() || let ActionType::Fortify {..} = u.action.as_ref().unwrap().action_type {inc_health!();}
				if let Some(ActionMeta {action_type: ActionType::Fortify {..}, ..}) | None = u.action.last() {inc_health!();}
			}
			
			// idle
			if u.action.len() == 0 {continue;}
			
			let u_coord = u.return_coord();
			let action = u.action.last_mut().unwrap();
			let path_coords_len = action.path_coords.len();
			let action_type = &mut action.action_type;
			
			///////////////////////////////////////// pipe
			if let ActionType::WorkerBuildPipe = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerBuildStructurePipe");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				let mfc = map_data.get(ZoomInd::Full, u.return_coord());
				debug_assertq!(mfc.map_type == MapType::Land);
				
				let exf = &mut exs.last_mut().unwrap();
				exf.create_if_empty(u.return_coord());
				let ex = exf.get_mut(&u.return_coord()).unwrap();
				
				// if there is already an owner and it is not the current player, do not continue
				if let Some(current_owner_id) = ex.actual.owner_id {
					if current_owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
				}else{
					ex.actual.owner_id = Some(u.owner_id);
				}
				
				// add pipe
				ex.actual.pipe_health = Some(u8::MAX);
				
				//printlnq!("adding road at {}, unit owner {}", Coord::frm_ind(u.return_coord(), map_sz), owners[u.owner_id as usize].nm);
				compute_zooms_coord(u.return_coord(), bldgs, temps.bldgs, map_data, exs, players);
				
				// end action or mv unit
				if u.action.last().unwrap().path_coords.len() == 0 {
					u.action.pop();
					//uninit_city_hall_dists(u.owner_id, exs, bldgs);
				}else{
					debug_assertq!(u.action.last().unwrap().actions_req > 0.);
					debug_assertq!(u.action.last().unwrap().path_coords.len() > 0);
					mv_unit(*unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Record(&mut disband_unit_inds), &mut Some(disp));
				}
			
			///////////////////////////////////////// road
			}else if let ActionType::WorkerBuildStructure {structure_type: StructureType::Road, ..} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerBuildStructure {road}");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				let mfc = map_data.get(ZoomInd::Full, u.return_coord());
				debug_assertq!(mfc.map_type == MapType::Land);
				let exf = &mut exs.last_mut().unwrap();
				
				// land already has some other structure on it
				if let Some(ex) = exf.get(&u.return_coord()) {
					if (ex.actual.structure != None && ex.actual.ret_structure() != Some(StructureType::Road)) || !ex.bldg_ind.is_none() {
						u.action.pop();
						//uninit_city_hall_dists(u.owner_id, exs, bldgs);
						continue;
					}
				}
				
				exf.create_if_empty(u.return_coord());
				let ex = exf.get_mut(&u.return_coord()).unwrap();
				
				// if there is already an owner and it is not the current player, do not continue
				if let Some(current_owner_id) = ex.actual.owner_id {
					if current_owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
				}else{
					ex.actual.owner_id = Some(u.owner_id);
				}
				
				// add road
				ex.actual.set_structure(u, StructureType::Road, map_sz);
				ex.actual.rm_zone(u_coord, players, temps.doctrines, map_sz);
				
				//printlnq!("adding road at {}, unit owner {}", Coord::frm_ind(u.return_coord(), map_sz), owners[u.owner_id as usize].nm);
				compute_zooms_coord(u.return_coord(), bldgs, temps.bldgs, map_data, exs, players);
				
				// end action or mv unit
				if u.action.last().unwrap().path_coords.len() == 0 {
					u.action.pop();
					//uninit_city_hall_dists(u.owner_id, exs, bldgs);
				}else{
					debug_assertq!(u.action.last().unwrap().actions_req > 0.);
					debug_assertq!(u.action.last().unwrap().path_coords.len() > 0);
					mv_unit(*unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Record(&mut disband_unit_inds), &mut Some(disp));
				}
			
			/////////////////////////////// wall (see also WorkerRepairWall)
			}else if let ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, ref mut turns_expended} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerBuildStructure {wall}");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				let mfc = map_data.get(ZoomInd::Full, u_coord);
				debug_assertq!(mfc.map_type == MapType::Land);
				
				let ex = exs.last().unwrap().get(&u_coord).unwrap();
				let n_units = ex.unit_inds.as_ref().unwrap().len();
				debug_assertq!(n_units > 0);
				
				// land already has some other structure on it
				// or has a unit aside from the current worker on it
				// or has a building on it
				if (ex.actual.structure != None && ex.actual.ret_structure() != Some(StructureType::Wall)) ||
						n_units != 1 || ex.bldg_ind != None || ex.actual.ret_zone_type() != None {
					//if u.owner_id == 0 {
					//	endwin(); println!("aborting wall creation n_units {}", n_units);
					//}
					
					if u.owner_id == disp.state.iface_settings.cur_player {
						u.action.pop();
						continue;
					}
					*turns_expended = 0;
				}else{
					// add wall or skip turn; update `turns_expended`
					if *turns_expended == WORKER_WALL_CONSTRUCTION_TURNS {
						*turns_expended = 0;
						
						let exf = exs.last_mut().unwrap();
						exf.create_if_empty(u_coord);
						let ex = exf.get_mut(&u_coord).unwrap();
						ex.actual.set_structure(u, StructureType::Wall, map_sz);
						
						compute_zooms_coord(u_coord, bldgs, temps.bldgs, map_data, exs, players);
					}else{
						*turns_expended += 1;
						continue 'outer;
					}
				}
				
				// mv unit or abort action
				let action = u.action.last_mut().unwrap();
				if let Some(&next_coord) = action.path_coords.last() {
					debug_assertq!(action.actions_req > 0.);
					
					// already at next location
					if next_coord == u_coord {
						action.path_coords.pop();
						continue 'outer;
					}
					
					let u_owner_id = u.owner_id;
					let movement_type = u.template.movement_type;
					
					// for ai only
					macro_rules! skip_to_next{() => {
						loop {
							let action = units[*unit_ind].action.last_mut().unwrap();
							if let Some(next_coord) = action.path_coords.pop() {
								if movable_to(u_coord, next_coord, &map_data.get(ZoomInd::Full, next_coord), exs.last().unwrap(), MvVarsAtZoom::NonCivil {units, start_owner: u_owner_id, blind_undiscov: None}, bldgs, &Dest::NoAttack, movement_type) {
									set_coord(next_coord, *unit_ind, u_owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, &mut players[u_owner_id as usize].stats, map_sz, gstate);
									continue 'outer;
								}
							}else{
								units[*unit_ind].action.pop();
								continue 'outer;
							}
						}
					};}
					
					// gap indicates wall, just skip over it
					match manhattan_dist_inds(next_coord, u_coord, map_sz) {
						d if d < 2 => {
							if disp.state.iface_settings.cur_player == u.owner_id || movable_to(u_coord, next_coord, &map_data.get(ZoomInd::Full, next_coord), exs.last().unwrap(), MvVarsAtZoom::NonCivil {units, start_owner: u_owner_id, blind_undiscov: None}, bldgs, &Dest::NoAttack, movement_type) {
								mv_unit(*unit_ind, u_owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Record(&mut disband_unit_inds), &mut Some(disp));
							}else{
								skip_to_next!();
							}
						} _ => {skip_to_next!();}
					}
					
				// no actions left
				}else {u.action.pop();}
			
			//////////////////////////////////////////// auto-explore
			}else if let ActionType::AutoExplore {start_coord, explore_type} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("AutoExplore");
				
				if !players[u.owner_id as usize].ptype.is_empire() || gstate.rng.gen_f32b() < (1./50.) {
					let start_coord = *start_coord;
					let explore_type = *explore_type;
					
					// move to unexplored territory
					if path_coords_len != 0 {
						mv_unit(*unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Record(&mut disband_unit_inds), &mut Some(disp));
					}
					
					let u = &mut units[*unit_ind];
					
					// reached unexplored destination OR out of moves => add more to go to
					// find next closest unexplored territory
					if u.action.len() == 0 || (u.action.last().unwrap().path_coords.len() == 0){
						u.action.pop();
						let land_discov = &players[u.owner_id as usize].stats.land_discov.last().unwrap();
						let is_cur_player = disp.state.iface_settings.cur_player == u.owner_id;
						
						// found new unexplored territory near current position
						if let Some(new_action) = explore_type.find_square_unexplored(*unit_ind, start_coord, map_data, exs, units, bldgs, land_discov, map_sz, is_cur_player, &mut gstate.rng) {
							units[*unit_ind].action.push(new_action);
						
						// go back to start
						}else if units[*unit_ind].return_coord() != start_coord {
							match explore_type {
								ExploreType::SpiralOut => {
								} ExploreType::Random => {
									let mfc = &map_data.get(ZoomInd::Full, start_coord);
									let u = &units[*unit_ind];
									if movable_to(u.return_coord(), start_coord, mfc, exs.last().unwrap(), MvVarsAtZoom::NonCivil {units, start_owner: u.owner_id, blind_undiscov: None}, bldgs, &Dest::NoAttack, u.template.movement_type) {
										let pstats = &mut players[u.owner_id as usize].stats;
										set_coord(start_coord, *unit_ind, disp.state.iface_settings.cur_player == units[*unit_ind].owner_id, units, map_data, exs, pstats, map_sz, gstate);
										units[*unit_ind].action.push(ActionMeta::new(ActionType::AutoExplore {start_coord, explore_type})); //prev_action.unwrap());
									}
								} ExploreType::N => {panicq!("invalid exploration type");}
							}
						}
					}
				}
			
			//////////////////////////////////////////// zoning (from path_coords for AI) 
			// [this should be above the move units check, otherwise it won't run because path_coords_len != 0
			}else if let ActionType::WorkerZoneCoords {zone} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerZoneCoords");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				
				// owned by someone else
				let ex = exs.last_mut().unwrap().get_mut(&u_coord).unwrap();
				if let Some(owner_id) = ex.actual.owner_id {
					if owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
				}
				
				/////////////////////////////////////////////////////////
				// create zone
				if ex.actual.structure == None {
					let player = &mut players[u.owner_id as usize];
					{ // update gold if the player can afford to zone
						let cost = zone.cost_per_tile(temps);
						if player.stats.gold < cost {
							u.action.pop();
							continue;
						}else{
							player.stats.gold -= cost;
						}
					}
					
					ex.actual.add_zone(u_coord, *zone, player, temps.doctrines, map_sz);
				}
				
				/////////////////////////////////////////////
				// move to next plot
				
				// not finished, move unit
				if let Some(coord_new) = u.action.last_mut().unwrap().path_coords.pop() {
					// verify land still movable to
					if !movable_to(u_coord, coord_new, &map_data.get(ZoomInd::Full, coord_new), exs.last().unwrap(), MvVarsAtZoom::NonCivil{units, start_owner: units[*unit_ind].owner_id, blind_undiscov: None},
							bldgs, &Dest::NoAttack, units[*unit_ind].template.movement_type) {
						units[*unit_ind].action.pop();
						continue;
					}
					
					// finally, move unit
					compute_zooms_coord(u_coord, bldgs, temps.bldgs, map_data, exs, players);
					let owner_id = units[*unit_ind].owner_id;
					set_coord(coord_new, *unit_ind, owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, &mut players[owner_id as usize].stats, map_sz, gstate);
					
				// finished
				}else{
					//////
					//if units[unit_ind].owner_id == 7 {//if owners[units[unit_ind].owner_id as usize].nm == "Swaodor" {
					//	println!("unit_ind {} ending zone", unit_ind);
					//}
					////////
					units[*unit_ind].action.pop();
					continue;
				}
			
			///////////////////////////////////////// move units
			}else if path_coords_len != 0 {
				if let ActionType::Mv | ActionType::MvIgnoreWallsAndOntoPopulationCenters | ActionType::MvIgnoreOwnWalls |
						ActionType::WorkerBuildBldg{..} | ActionType::WorkerBuildStructure {..} | ActionType::ScaleWalls |
						ActionType::WorkerZone{..} | ActionType::WorkerRmZonesAndBldgs {..} |
						ActionType::WorkerRepairWall{..} = action_type {
					
					#[cfg(feature="profile")]
					let _g = Guard::new("mv units");
					
					// limit # of non-human movements per turn
					if (|| {
						let u_owner = u.owner_id as usize;
						
						if unit_count > MAX_UNIT_INDS_COMP && // exceeded turn quota
						   !players[u_owner].ptype.is_human() && // and player isn't human
						   u.template.nm[0] != WORKER_NM  // and unit isn't a worker
						{
							// fiefdom of human -> do not limit movement
							if let Some(fiefdom_owner) = gstate.relations.fiefdom_of(u_owner) {
								if players[fiefdom_owner].ptype.is_human() {
									return false; // do not skip
								}
							}
							
							// the moving AI is at war with a human, do not skip
							if gstate.relations.at_war_with(u_owner).iter()
								.any(|&war_owner| players[war_owner].ptype.is_human()) {
								return false; // do not skip
							}
							
							return true; // skip movement
						}
						false
					})() {continue;}
					
					debug_assertq!(u.action.last().unwrap().actions_req > 0.);
					debug_assertq!(u.action.last().unwrap().path_coords.len() > 0);
					
					mv_unit(*unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Record(&mut disband_unit_inds), &mut Some(disp));
				}
			
			//////////////////////////////////////// assassinate
			}else if let ActionType::Assassinate {attack_coord} = action_type {
				do_assassinate_action(*unit_ind, *attack_coord, &mut disband_unit_inds, units, bldgs, players, exs, map_data, gstate, disp, temps);
				
			///////////////////////////////////////// gate
			}else if let ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, ..} = action_type {
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				let mfc = map_data.get(ZoomInd::Full, u.return_coord());
				debug_assertq!(mfc.map_type == MapType::Land);
				
				// land already has some other structure on it
				if let Some(ex) = exs.last().unwrap().get(&u.return_coord()) {
					if !ex.bldg_ind.is_none() {
						u.action.pop();
						//uninit_city_hall_dists(u.owner_id, exs, bldgs);
						continue;
					}
				}
				
				let exf = &mut exs.last_mut().unwrap();
				exf.create_if_empty(u.return_coord());
				let ex = exf.get_mut(&u.return_coord()).unwrap();
				
				// if there is already an owner and it is not the current player, do not continue
				if let Some(current_owner_id) = ex.actual.owner_id {
					if current_owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
				}
				
				// add gate
				ex.actual.set_structure(u, StructureType::Gate, map_sz);
				ex.actual.owner_id = Some(u.owner_id);
				ex.actual.rm_zone(u_coord, players, temps.doctrines, map_sz);
				
				compute_zooms_coord(u.return_coord(), bldgs, temps.bldgs, map_data, exs, players);
				
				u.action.pop();
			
			//////////////////////////////////////////// remove zones & bldgs (from start/end coords from human UI)
			}else if let ActionType::WorkerRmZonesAndBldgs {start_coord, end_coord, ..} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerRmZonesAndBldgs");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				
				// owned by someone else
				let ex = exs.last_mut().unwrap().get_mut(&u_coord).unwrap();
				if let Some(owner_id) = ex.actual.owner_id {
					if owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
				}
				
				// break apart coords
				let current_c = Coord::frm_ind(u_coord, map_sz); // current coords
				let end_coord = end_coord.unwrap();
				let finished_c = Coord::frm_ind(end_coord, map_sz); // finished coords
				let start_c = Coord::frm_ind(start_coord.unwrap(), map_sz); // start coords
				
				// map dimensions
				let rect_sz = start_coord_use(start_c, finished_c, map_sz).1;
				let w = rect_sz.w as isize;
				
				let removable_bldg = |bldg_ind: usize| -> bool {
					let b = &bldgs[bldg_ind];
					let bt = &b.template;
					if let BldgType::Taxable(_) = &bt.bldg_type {
						true
					}else if let BldgArgs::PopulationCenter {..} = &b.args {
						false
					}else{ bt.doctrinality_bonus == 0. }
				};
				
				/////////////////////////////////////////////////////////
				// rm bldg
				if let Some(bldg_ind) = ex.bldg_ind {
					if removable_bldg(bldg_ind) {
						rm_bldg(bldg_ind, u.owner_id == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
					}else{
						u.action.pop();
					}
					continue;
				}
				
				ex.actual.rm_zone(u_coord, players, temps.doctrines, map_sz);
				ex.actual.structure = None;
				
				/////////////////////////////////////////////
				// move to next plot
				
				let direction_i = if start_c.y < finished_c.y {1} else {-1};
				let mut direction_j = if start_c.x < finished_c.x {1} else {-1};
				if (start_c.x - finished_c.x).abs() != (w-1) {direction_j *= -1;} // wrap
				
				let odd_line = ((current_c.y - start_c.y).abs() % 2) != 0;
				if odd_line {direction_j *= -1};
				
				// finished?
				if current_c.y == finished_c.y {
					let corner_coord = map_sz.coord_wrap(finished_c.y, start_c.x).unwrap();
					
					let chk = |coord| {
						if let Some(ex) = exs.last().unwrap().get(&coord) {
							ex.actual.ret_zone_type().is_none() && ex.actual.ret_structure().is_none() &&
							ex.bldg_ind.is_none()
						} else {true}
					};
					
					if chk(corner_coord) && chk(end_coord) {
						u.action.pop();
						continue;
						// both corners don't have anything on them, so quit
					}
				}
				
				// not finished, move unit
				let mut new_c = current_c.clone();
				
				if (current_c.x == finished_c.x && !odd_line) || (current_c.x == start_c.x && odd_line ) { // next line
					new_c.y += direction_i;
				}else{
					new_c.x += direction_j;
				}
				
				let coord_new = map_sz.coord_wrap(new_c.y, new_c.x).unwrap();
				
				// check if next coordinate has a bldg (and it matches the unit's owner), if so, remove it
				if let Some(ex) = exs.last().unwrap().get(&coord_new) {
					if let Some(bldg_ind) = ex.bldg_ind {
						// owners do not match
						if bldgs[bldg_ind].owner_id != u.owner_id || !removable_bldg(bldg_ind) {
							u.action.pop();
							continue;
						}
							
						rm_bldg(bldg_ind, u.owner_id == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
					}
				}
				
				// verify land still movable to
				let u_owner = units[*unit_ind].owner_id;
				if !movable_to(units[*unit_ind].return_coord(), coord_new, &map_data.get(ZoomInd::Full, coord_new), exs.last().unwrap(), MvVarsAtZoom::NonCivil{units, start_owner: u_owner, blind_undiscov: None}, bldgs, 
						&Dest::NoAttack, units[*unit_ind].template.movement_type) {
					units[*unit_ind].action.pop();
					continue;
				}
				
				// finally, move unit
				compute_zooms_coord(u_coord, bldgs, temps.bldgs, map_data, exs, players);
				set_coord(coord_new, *unit_ind, u_owner == disp.state.iface_settings.cur_player, units, map_data, exs, &mut players[u_owner as usize].stats, map_sz, gstate);
			
			//////////////////////////////////////////// zoning (from start/end coords from human UI)
			}else if let ActionType::WorkerZone {zone, start_coord, end_coord, ..} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerZone");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				
				// owned by someone else
				let ex = exs.last_mut().unwrap().get_mut(&u_coord).unwrap();
				if let Some(owner_id) = ex.actual.owner_id {
					if owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
				}
				
				// break apart coords
				let current_c = Coord::frm_ind(u_coord, map_sz); // current coords
				let finished_c = Coord::frm_ind(end_coord.unwrap(), map_sz); // finished coords
				let start_c = Coord::frm_ind(start_coord.unwrap(), map_sz); // start coords
				
				// road map dimensions
				let (start_c_wrap, rect_sz) = start_coord_use(start_c, finished_c, map_sz);
				let w = rect_sz.w as isize;
				
				/////////////////////////////////////////////////////////
				// create zone or road
				if ex.actual.structure == None {
					let player = &mut players[u.owner_id as usize];
					
					{ // update gold if the player can afford to zone
						let cost = zone.cost_per_tile(temps);
						if player.stats.gold < cost {
							u.action.pop();
							continue;
						}else{
							player.stats.gold -= cost;
						}
					}
					
					let h = rect_sz.h as isize;
					
					// wrapped
					let cx = if (current_c.x - start_c_wrap.x).abs() > ((current_c.x + map_sz.w as isize) - start_c_wrap.x).abs(){
						current_c.x + map_sz.w as isize
					}else{ current_c.x};
					
					// road map
					let exf = &mut exs.last_mut().unwrap();
					let roads = new_zone_w_roads(start_coord.unwrap(), end_coord.unwrap(), map_sz, exf);
				
					exf.create_if_empty(u_coord);
					let ex = exf.get_mut(&u_coord).unwrap();
					
					// current position should be within the zone
					debug_assertq!(cx >= start_c_wrap.x && (cx - start_c_wrap.x) < w, "cx {} start_c_wrap.x {} w {} current_c {} {} start_c {} {} finished_c {} {} owner {}",
							cx, start_c_wrap.x, w, current_c.y, current_c.x, start_c.y, start_c.x, finished_c.y, finished_c.x, u.owner_id);
					debug_assertq!(current_c.y >= start_c_wrap.y && (current_c.y - start_c_wrap.y) < h);
					
					/////////////////// place road
					let ind = (current_c.y - start_c_wrap.y)*w + (cx - start_c_wrap.x);
					if roads[ind as usize] {
						if ex.actual.ret_structure() != Some(StructureType::Road) {
							ex.actual.set_structure(u, StructureType::Road, map_sz);
							continue;
						}
						
					//////////////////// place zone
					}else {ex.actual.add_zone(u_coord, *zone, player, temps.doctrines, map_sz);}
				} // create zone or road
				
				/////////////////////////////////////////////
				// move to next plot
				
				// finished?
				let direction_i = if start_c.y < finished_c.y {1} else {-1};
				let mut direction_j = if start_c.x < finished_c.x {1} else {-1};
				if (start_c.x - finished_c.x).abs() != (w-1) {direction_j *= -1;} // wrap
				
				let odd_line = ((current_c.y - start_c.y).abs() % 2) != 0;
				if odd_line {direction_j *= -1};
				
				let corner_coord = map_sz.coord_wrap(finished_c.y, start_c.x).unwrap();
				
				// finished (check that we got the other corner)
				
				let chk_zone_or_road = |coord| {
					if let Some(ex) = exs.last().unwrap().get(&coord) {
						ex.actual.ret_zone() == Some(*zone) || ex.actual.ret_structure() == Some(StructureType::Road)
					} else {false}
				};
				
				if chk_zone_or_road(corner_coord) && chk_zone_or_road(end_coord.unwrap()) {
					u.action.pop();
					continue;
					// both corners have been zoned, so quit
					// even if they aren't the right zone type, the user should decide
					// since someone else must be working close by
				}
				
				// not finished, move unit
				let mut new_c = current_c.clone();
				
				if (current_c.x == finished_c.x && !odd_line) || (current_c.x == start_c.x && odd_line ) { // next line
					new_c.y += direction_i;
				}else{
					new_c.x += direction_j;
				}
				
				let coord_new = map_sz.coord_wrap(new_c.y, new_c.x).unwrap();
				
				// verify land still movable to
				let u_owner = units[*unit_ind].owner_id;
				if !movable_to(units[*unit_ind].return_coord(), coord_new, &map_data.get(ZoomInd::Full, coord_new), exs.last().unwrap(), MvVarsAtZoom::NonCivil{units, start_owner: u_owner, blind_undiscov: None}, bldgs, 
						&Dest::NoAttack, units[*unit_ind].template.movement_type) {
					
					units[*unit_ind].action.pop();
					continue;
				}
				
				// finally, move unit
				compute_zooms_coord(u_coord, bldgs, temps.bldgs, map_data, exs, players);
				set_coord(coord_new, *unit_ind, u_owner == disp.state.iface_settings.cur_player, units, map_data, exs, &mut players[u_owner as usize].stats, map_sz, gstate);
				
			///////////////////////////////////////// buildings
			}else if let ActionType::WorkerBuildBldg {template, bldg_coord, doctrine_dedication, ..} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerBuildBldg");
				
				debug_assertq!(u.template.nm[0] == WORKER_NM);
				
				// bldg already existant
				let bldg_ind = if let Some(bldg_coord) = bldg_coord {
					if let Some(bldg_ind) = bldg_ind_frm_coord(*bldg_coord, bldgs) {
						bldg_ind
					}else{
						u.action.pop();
						continue;
					}
				// bldg needs to be created
				}else{
					// add building next to unit's location:
					let map_coord = map_sz.coord_wrap((u_coord as usize/map_sz.w) as isize, (u_coord as usize % map_sz.w) as isize + 1).unwrap();
					
					// if the building hasn't been created
					let exf = exs.last().unwrap();
					if exf.get(&map_coord).is_none() || exf.get(&map_coord).unwrap().bldg_ind == None {
						let u_owner_id = u.owner_id as usize;
						let doctrine_dedication = if let Some(dedication) = doctrine_dedication {
							//printlnq!("owner {} target: {} closest avail: {}", u_owner_id, dedication.nm[0],
							//		dedication.closest_available(&stats[u_owner_id], temps.doctrines).nm[0]);
							Some(dedication.closest_available(&players[u_owner_id].stats, temps.doctrines))
						}else{None};
						
						if !add_bldg(map_coord, u.owner_id, bldgs, template, doctrine_dedication, None, temps, map_data, exs, players, gstate) {
							u.action.pop(); // couldn't create bldg
							continue;
						}
						
						////////////////////////// dbg
						//endwin();
						//
						//for (c, ex) in exs.last().unwrap().iter() {
						//	if let Some(bldg_ind) = ex.bldg_ind {
						//		println!("{} {}", c, bldg_ind);
						//	}
						//}
						///////////////////////
					}
				
					exs.last().unwrap().get(&map_coord).unwrap().bldg_ind.unwrap() // bldg_ind
				};
				
				let b = &mut bldgs[bldg_ind];
				
				// bldg owned by someone else
				if b.owner_id != u.owner_id {
					u.action.pop();
					continue;
				// building is damaged
				}else if let Some(damage) = b.damage {
					b.damage = if damage > bldg_config.fire_repair_rate {
						Some( damage - bldg_config.fire_repair_rate )
					// completely repaired, extinguish any fires
					}else{
						b.fire = None;
						None
					};
				// increment construction
				}else if let Some(construction_done) = b.construction_done {
					let new_const_val = construction_done + u.template.actions_per_turn;
					b.construction_done = if new_const_val < b.template.construction_req {
						Some(new_const_val)
					}else {None};
				}
				
				// finished constructing & repairing
				if b.construction_done == None && b.damage == None {
					u.action.pop();
					let pstats = &mut players[b.owner_id as usize].stats;
					
					// launch tech tree
					if pstats.research_per_turn == 0 && b.template.research_prod != 0 &&
						disp.state.iface_settings.cur_player == b.owner_id && pstats.techs_scheduled.len() == 0 &&
						disp.state.iface_settings.interrupt_auto_turn {
							
							disp.create_tech_window(true);
					}
					
					// see also when altering: zones/set_owner, buildings/add_bldg
					pstats.bldg_stats(StatsAction::Add, b.template);
					
					if b.template.nm[0] == CITY_HALL_NM {
						uninit_city_hall_dists(u.owner_id, &mut players[b.owner_id as usize].zone_exs, bldgs, map_sz);
					}
				}
			
			///////////////////////////////////////// repair wall
			}else if let ActionType::WorkerRepairWall {wall_coord, ref mut turns_expended} = action_type {
				#[cfg(feature="profile")]
				let _g = Guard::new("WorkerRepairWall");
				
				let wall_coord = wall_coord.unwrap();
				let exf = exs.last_mut().unwrap();
				
				macro_rules! create_wall{
					// use provided MapEx
					($map_ex: expr) => {
						if *turns_expended == WORKER_WALL_CONSTRUCTION_TURNS {
							$map_ex.actual.set_structure(u, StructureType::Wall, map_sz);
							compute_zooms_coord(u.return_coord(), bldgs, temps.bldgs, map_data, exs, players);
						}else{
							*turns_expended += 1;
							continue 'outer;
						}
					};
					
					// create MapEx if needed
					() => {
						if *turns_expended == WORKER_WALL_CONSTRUCTION_TURNS {
							exf.create_if_empty(u.return_coord());
							let ex = exf.get_mut(&u.return_coord()).unwrap();
							
							ex.actual.set_structure(u, StructureType::Wall, map_sz);
							compute_zooms_coord(u.return_coord(), bldgs, temps.bldgs, map_data, exs, players);
						}else{
							*turns_expended += 1;
							continue 'outer;
						}
					};
				}
				
				// repair or create wall
				if let Some(ref mut ex) = exf.get_mut(&wall_coord) {
					if let Some(owner_id) = ex.actual.owner_id {
						if u.owner_id == owner_id {
							// repair wall
							if let Some(ref mut s) = ex.actual.structure {
								if s.health != std::u8::MAX && s.structure_type == StructureType::Wall {
									let repair_per_turn = u.template.repair_wall_per_turn.unwrap();
									
									if ((s.health as usize) + repair_per_turn) < (std::u8::MAX as usize) {
										s.health += repair_per_turn as u8;
										continue;
										
									// now finished:
									}else {s.health = std::u8::MAX;}
								}
							
							// create new wall
							}else {create_wall!(ex);}
						}
					// create new wall
					}else {create_wall!(ex);}
				
				// create new wall and MapEx
				}else {create_wall!();}
				
				// record wall as repaired
				players[u.owner_id as usize].log_repaired_wall(Coord::frm_ind(wall_coord, map_sz));
				
				u.action.pop();
			
			////////////////////////////////////// automate worker
			}else if let ActionType::UIWorkerAutomateCity = action_type {
				let u_owner = u.owner_id as usize;
				let player = &mut players[u_owner];
				if let Some(ai_state) = player.ptype.any_ai_state_mut() {
					let unit_coord = Coord::frm_ind(u.return_coord(), map_sz);
					let is_cur_player = u.owner_id == disp.state.iface_settings.cur_player;
					
					// find closest city
					if let Some(min_city) = ai_state.city_states.iter_mut().min_by_key(|c| 
							manhattan_dist(Coord::frm_ind(c.coord, map_sz), unit_coord, map_sz)) {
						if min_city.worker_actions.len() != 0 {
							min_city.set_worker_action(*unit_ind, is_cur_player, units, &mut player.stats, bldgs, map_data, exs, gstate, map_sz);
						// no actions left
						}else{u.action.pop();}
					// no minimum city found
					}else{u.action.pop();}
				}else{panicq!("UI worker automation on invalid player");}
			
			//////////////////////////////////////// burn building
			}else if let ActionType::BurnBuilding {coord} = action_type {
				if let Some(b) = bldg_frm_coord(*coord, bldgs) {
					if b.owner_id != u.owner_id {
						u.action.pop();
						continue;
					}
					
					b.create_fire(&mut gstate.rng);
					
					//// move unit around perimeter (just a visual effect)
					if gstate.rng.gen_f32b() < 0.1 {
						let u_coord = u.return_coord();
						let u_owner = u.owner_id;
						let movement_type = u.template.movement_type;
						let dest = Dest::NoAttack;
						
						let perimeter_coords = b.perimeter_coords(map_sz);
						
						// find next coordinate in vector
						let start_ind = if let Some(perim_ind) = perimeter_coords.iter().position(|&c| c == u_coord) {
							if perim_ind != (perimeter_coords.len() - 1) {
								perim_ind + 1
							// wrap
							}else{0}
						}else{0};
						
						let clockwise = gstate.rng.gen_f32b() < 0.5;
						
						macro_rules! chk_and_set {($coord: expr, $mv_vars: expr, $exf: expr) => {
							if movable_to(u_coord, $coord, &map_data.get(ZoomInd::Full, $coord), $exf, $mv_vars, bldgs, &dest, movement_type) {
								set_coord($coord, *unit_ind, false, units, map_data, exs, &mut players[u_owner as usize].stats, map_sz, gstate);
								continue 'outer;
							}
						};}
						
						// find the next movable coordinate to place unit
						macro_rules! find_and_set_next_coord {($skip: expr) => {
							let exf = exs.last().unwrap();
							let mv_vars = MvVarsAtZoom::NonCivil {units, start_owner: u_owner, blind_undiscov: None};
							
							if clockwise {
								for perim_coord in perimeter_coords.iter().skip($skip) {
									chk_and_set!(*perim_coord, mv_vars, exf);
								}
							}else{
								for perim_coord in perimeter_coords.iter().skip($skip).rev() {
									chk_and_set!(*perim_coord, mv_vars, exf);
								}
							}
						};}
						
						find_and_set_next_coord!(start_ind);
						
						// try again from the start
						if start_ind != 0 {find_and_set_next_coord!(0);}
					}
				}else{u.action.pop();}
			
			//////////////////////////////////// automate soldier
			}else if let ActionType::SectorAutomation {unit_enter_action, sector_nm, ..} = action_type {
				let automated_unit_owner_id = u.owner_id;
				let pstats = &mut players[automated_unit_owner_id as usize].stats;
				let sector = pstats.sector_frm_nm(sector_nm);
				let unit_enter_action = *unit_enter_action;
				let attacker_ind = unit_ind;
				let u_coord = u.return_coord();
				let automated_unit_nm = u.nm.clone();
				let movement_type = u.template.movement_type;
				
				// check if any units have entered the sector
				if let Some(u) = units.iter().find(|u| u.owner_id != automated_unit_owner_id &&
						sector.contains(&Coord::frm_ind(u.return_coord(), map_sz), map_sz)) {
					const MAX_SEARCH_DEPTH: usize = 200;
					
					macro_rules! attack_unit{() => {
						let target = ActionType::Attack {
							attack_coord: Some(u.return_coord()),
							attackee: Some(u.owner_id),
							ignore_own_walls: false
						};
						if set_target_attackable(&target, *attacker_ind, false, MAX_SEARCH_DEPTH, units, bldgs, exs, map_data, map_sz) {
							continue 'outer;
						}
					};}
					
					match unit_enter_action {
						SectorUnitEnterAction::Assault => {attack_unit!();
						} SectorUnitEnterAction::Defense => {
							if gstate.relations.at_war(automated_unit_owner_id as usize, u.owner_id as usize) {
								attack_unit!();
							}
						} SectorUnitEnterAction::Report => {
							units[*unit_ind].action.pop();
							if automated_unit_owner_id == disp.state.iface_settings.cur_player {
								disp.create_interrupt_window(UIMode::ForeignUnitInSectorAlert(ForeignUnitInSectorAlertState {
									sector_nm: sector.nm.clone(),
									battalion_nm: automated_unit_nm
								}));
							}
							continue;
						} SectorUnitEnterAction::N => {panicq!("invalid unit enter action");}
					}
				}
				
				let u = &mut units[*unit_ind];
				
				// repair health
				if u.template.max_health > u.health {
					u.health += HEALTH_GAIN_PER_TURN;
					continue 'outer;
				}
				
				// do the idle action
				if let Some(action) = u.action.last_mut() {
					if let ActionType::SectorAutomation {ref mut idle_action, ..} = action.action_type {
						match idle_action {
							SectorIdleAction::Patrol {dist_monitor:_, perim_coord_ind, perim_coord_turn_computed} => {
								let perim_coords = &sector.perim_coords;
								
								let mut action_iface = ActionInterfaceMeta {
									action: ActionMeta::new(ActionType::Mv),
									unit_ind: Some(automated_unit_owner_id as usize),
									max_search_depth: 500,
									start_coord: Coord::frm_ind(u_coord, map_sz),
									movement_type,
									movable_to: &movable_to
								};
								
								// if perim_coord_ind is not stale
								if *perim_coord_turn_computed == perim_coords.turn_computed {
									// inc & wrap
									if *perim_coord_ind < (perim_coords.coords.len()-1) {
										*perim_coord_ind += 1;
									}else{
										*perim_coord_ind = 0;
									}
									
									let next_c = Coord::frm_ind(perim_coords.coords[*perim_coord_ind], map_sz);
									action_iface.update_move_search(next_c, map_data, exs, MvVars::NonCivil{units, start_owner: automated_unit_owner_id, blind_undiscov: None}, bldgs);
									
									let u = &mut units[*unit_ind];
									
									// if next coord is traversable (w/ move search) -> go there
									if action_iface.action.path_coords.len() > 0 {
										u.action.push(action_iface.action);
									// else, move in direction of next location
									}else{
										let cur_loc = u.return_coord();
										let cur_loc_c = Coord::frm_ind(cur_loc, map_sz);
										let exf = exs.last().unwrap();
										let mv_vars = MvVarsAtZoom::NonCivil {units, start_owner: automated_unit_owner_id, blind_undiscov: None};
										let dest = Dest::NoAttack;
										
										macro_rules! chk_and_mv{($y_off: expr, $x_off: expr) => {
											if let Some(cand_c) = map_sz.coord_wrap(cur_loc_c.y + $y_off, cur_loc_c.x + $x_off) {
												let mfc = map_data.get(ZoomInd::Full, cand_c);
												if movable_to(cur_loc, cand_c, &mfc, exf, mv_vars, bldgs, &dest, movement_type) {
													let pstats = &mut players[automated_unit_owner_id as usize].stats;
													set_coord(cand_c, *unit_ind, disp.state.iface_settings.cur_player == automated_unit_owner_id, units, map_data, exs, pstats, map_sz, gstate);
													continue;
												}
											}
										};}
										
										if cur_loc_c.y < next_c.y {chk_and_mv!(1, 0);}
										if cur_loc_c.y > next_c.y {chk_and_mv!(-1, 0);}
										if cur_loc_c.x < next_c.x {chk_and_mv!(0, 1);}
										if cur_loc_c.x > next_c.x {chk_and_mv!(0, -1);}
									}
								// perim_coord_ind is stale
								}else{
									*perim_coord_ind = 0;
									*perim_coord_turn_computed = perim_coords.turn_computed;
									
									action_iface.update_move_search(Coord::frm_ind(perim_coords.coords[*perim_coord_ind], map_sz), map_data, exs, MvVars::NonCivil{units, start_owner: automated_unit_owner_id, blind_undiscov: None}, bldgs);
									
									// if next coord is traversable (w/ move search) -> go there
									if action_iface.action.path_coords.len() > 0 {
										units[*unit_ind].action.push(action_iface.action);
									}
								}
							} SectorIdleAction::Sentry => {}
						}
					}
				}
			
			////////////////////////////////// mv w cursor
			}else if let ActionType::MvWithCursor = action_type {
				u.action.pop();
			}
		} // unit loop
		
		//printlnq!("{}", unit_action_start.elapsed().as_millis());
		
		///////////
		// delete units
		//
		// have to do this step after the unit loop above, because the unit indices will otherwise
		// change during the loop.
		disband_units(disband_unit_inds, disp.state.iface_settings.cur_player, units, map_data, exs, players, gstate, map_sz);
		
		/////////////////
		// reset unit actions
		for u in units.iter_mut() {u.actions_used = Some(0.);}
	}
	
	///////////////////////////////////////////
	{ // bldg management
		#[cfg(feature="profile")]
		let _g = Guard::new("end_turn bldg management");
		
		const MAX_BLDG_INDS_COMP: usize = 300;//150;//0;
		let mut bldg_inds = gstate.rng.inds_max(bldgs.len(), MAX_BLDG_INDS_COMP);
		
		/////////////////
		{ // recompute city hall and water dists if not init
			#[cfg(feature="profile")]
			let _g = Guard::new("recompute city hall dists");
			
			for bldg_ind in bldg_inds.iter() {
				let b = &bldgs[*bldg_ind];
				let b_coord = b.coord;
				let player = &mut players[b.owner_id as usize];
				if let Some(_zone_ex) = player.zone_exs.get(&return_zone_coord(b_coord, map_sz)) {
					//let water_source_not_init = zone_ex.water_source_dist == Dist::NotInit; // req for the borrow checker. ref to player is used which contains zone_ex
					
					//if let Dist::NotInit = zone_ex.ret_city_hall_dist() {
					//	player.set_city_hall_dist(b_coord, map_data, exs, bldgs, temps.doctrines, map_sz, gstate.turn);
					//}
					
					//if water_source_not_init {
					//	set_water_dist(b_coord, &mut player.zone_exs, map_data, exs, bldgs, map_sz, gstate.turn);
					//}
					
					ret_water_dist_recomp(b_coord, &mut player.zone_exs, map_data, exs, bldgs, temps.doctrines, map_sz, gstate.turn);
					player.ret_city_hall_dist_recomp(b_coord, map_data, exs, bldgs, temps.doctrines, map_sz, gstate.turn);
				}
			}
		}
		
		/////////////
		{ // building productions
			#[cfg(feature="profile")]
			let _g = Guard::new("building productions");
			
			for bldg_ind in 0..bldgs.len() {
				let player = &mut players[bldgs[bldg_ind].owner_id as usize];
				build_unit(bldg_ind, disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, player, gstate, temps);
			}
		}
		
		///////////
		for _ in 0..3 {
			create_taxable_constructions(map_data, temps, exs, players, bldgs, map_sz, gstate);
		}
		
		////////////////////////
		// destroy taxable constructions
		//
		// go in reverse order because building indices may shift around behind the deleted index
		{
			#[cfg(feature="profile")]
			let _g = Guard::new("destroy taxable constructions");
			
			bldg_inds.sort_unstable(); // results in smallest to largest
			
			for bldg_ind in bldg_inds.iter().rev() {
				const FULL_PROB_RES_RM: f32 = 1.5/3.; //800.);
				let bt = &bldgs[*bldg_ind].template;
				if let BldgType::Taxable(_) = bt.bldg_type {
					let effective_tax = bldgs[*bldg_ind].ret_taxable_upkeep_pre_operating_frac();
					let rand_cutoff = FULL_PROB_RES_RM * effective_tax * effective_tax;
					//assertq!(rand_cutoff >= 0., "{} {} {}", effective_tax, bldgs[bldg_ind].return_taxable_upkeep(), bt.upkeep);
					
					// destroy
					if gstate.rng.gen_f32b() < rand_cutoff {
						rm_bldg(*bldg_ind, bldgs[*bldg_ind].owner_id == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
					}
				}
			}
		}
		
		//////////////////////////////
		{ // sell products, add/rm residents, find jobs
			#[cfg(feature="profile")]
			let _g = Guard::new("sell products, add/rm residents, find jobs");
			
			const MAX_BLDG_INDS_PROD_COMP: usize = 25;//50;//2*50;//25;//50;//2*50;
			let bldg_inds = gstate.rng.inds_max(bldgs.len(), MAX_BLDG_INDS_PROD_COMP);
			
			// get bldgs w/ any job_search_bonus
			#[derive(Clone)]
			struct BonusWCoord {bonus: f32, coord: Coord}
			let job_bonus_bldgs = {
				let mut job_bonus_bldgs = vec![Vec::new(); players.len()];
				for b in bldgs.iter().filter(|b| b.template.job_search_bonus != 0.) {
					job_bonus_bldgs[b.owner_id as usize].push(BonusWCoord {
						bonus: b.template.job_search_bonus,
						coord: Coord::frm_ind(b.coord, map_sz)
					});
				}
				job_bonus_bldgs
			};
			
			for bldg_ind in bldg_inds.iter() {
				let b = &bldgs[*bldg_ind];
				let coord = b.coord;
				let exf = exs.last_mut().unwrap();
				let ex = exf.get(&coord).unwrap();
				if let Some(zone_type) = ex.actual.ret_zone_type() {
					debug_assertq!(b.template.resident_max >= b.n_residents());
					debug_assertq!(b.cons_capac() >= b.cons());
					
					////////////////////
					// taxable bldgs
					if zone_type != ZoneType::Residential {
						// everything is already sold
						if b.n_sold() == b.prod_capac() {continue;}
						debug_assertq!(b.n_sold() < b.prod_capac());
						
					///////////
					// residents
					}else{
						let mut demand = None;
						let player = &mut players[b.owner_id as usize];
						
						// add residents
						if b.template.resident_max > b.n_residents() {
							demand = Some(return_potential_demand(coord, map_data, exs, player, bldgs, map_sz, gstate.turn));
							return_happiness(coord, map_data, exs, bldgs, player, temps, gstate, map_sz);
							let effective_tax = b.ret_taxable_upkeep_pre_operating_frac();
							
							//println!("demand {} effective_tax {}", demand.unwrap(), -effective_tax);
							
							if gstate.rng.gen_f32b() < 2.*(4.*demand.unwrap() - effective_tax) {
								add_resident(*bldg_ind, bldgs, &player.zone_exs, &mut player.stats, map_sz);
							}
						}
						let b = &bldgs[*bldg_ind];
						
						// rm residents
						if b.n_residents() > 0 { // && rng.gen_f32b() > (1.-(1./RES_PROB_SCALE)) {
							let demand = demand.unwrap_or_else(||
								return_potential_demand(coord, map_data, exs, player, bldgs, map_sz, gstate.turn)
							);
							
							let effective_tax = b.ret_taxable_upkeep_pre_operating_frac();
							
							if gstate.rng.gen_f32b() > (1. + 4.*demand - effective_tax) {
								rm_resident(*bldg_ind, bldgs, player, map_sz);
							}
						}
						
						///////////////
						// find employer?
						let b = &bldgs[*bldg_ind];
						let n_residents = b.n_residents();
						
						// no residents
						if n_residents == 0 {continue;}
						
						// everyone is already employed
						if b.n_sold() == n_residents {continue;}
					} // residents
					
					//////////////////////
					// sell product or find employer
					
					// sum of bldg bonuses for job search (used for determining chance of not searching)
					let job_search_bonus_sum = {
						let b = &bldgs[*bldg_ind];
						
						let mut bonus_sum = 0.;
						let b_coord = Coord::frm_ind(b.coord, map_sz);
						for bonus_bldg in job_bonus_bldgs[b.owner_id as usize].iter() {
							let dist = manhattan_dist(b_coord, bonus_bldg.coord, map_sz);
							if dist > bldg_config.job_search_bonus_dist as usize {continue;}
							// y(0) = bldg_job_search_bonus
							// y(bldg_config.job_search_bonus_dist) = 0
							let slope = -(bonus_bldg.bonus / bldg_config.job_search_bonus_dist as f32);
							bonus_sum += slope*(dist as f32) + bonus_bldg.bonus;
						}
						bonus_sum
					};
					
					// chance of not searching
					if gstate.rng.gen_f32b() < (0.95 - job_search_bonus_sum) {continue;}
					
					sell_prod(*bldg_ind, bldgs, map_data, exs, map_sz, &mut players[bldgs[*bldg_ind].owner_id as usize].stats, &mut gstate.rng);
					debug_assertq!(bldgs[*bldg_ind].n_residents() <= bldgs[*bldg_ind].template.resident_max);
				} // in zone
			} // bldg loop
		}
		/////////////
		
		randomly_update_happiness(map_data, exs, players, bldgs, temps, gstate, map_sz);
		
		////////////////////////
		{ // fire damage
			#[cfg(feature="profile")]
			let _g = Guard::new("sell products, add/rm residents, find jobs");
			
			for bldg_ind in (0..bldgs.len()).rev() {
				let b = &mut bldgs[bldg_ind];
				if let Some(_) = &b.fire {
					b.damage = Some(
						if let Some(damage) = b.damage {
							let new_damage = damage + bldg_config.fire_damage_rate;
							// remove bldg
							if new_damage >= bldg_config.max_bldg_damage {
								rm_bldg(bldg_ind, b.owner_id == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
								continue;
						
							// save incremented damage
							} else {new_damage}
						} else {bldg_config.fire_damage_rate}
					);
				}
			}
		}
	}
	
	{ // update prevailing doctrine
		#[cfg(feature="profile")]
		let _g = Guard::new("update prevailing doctrine");
		
		for player in players.iter_mut() {
			let pstats = &mut player.stats;
			let mut max_val = pstats.locally_logged.doctrinality_sum[0];
			let prev_prevailing = pstats.doctrine_template;
			pstats.doctrine_template = &temps.doctrines[0];
			
			for (doc_pts, d) in pstats.locally_logged.doctrinality_sum.iter()
								.zip(temps.doctrines.iter())
								.skip(1) {
				if !d.bldg_reqs_met(pstats) || max_val >= *doc_pts {continue;}
				
				max_val = *doc_pts;
				pstats.doctrine_template = d;
			}
			
			// log change of doctrine and potentially open alert window
			if pstats.doctrine_template != prev_prevailing {
				gstate.log_event(LogType::PrevailingDoctrineChanged {
					owner_id: player.id as usize,
					doctrine_frm_id: prev_prevailing.id,
					doctrine_to_id: pstats.doctrine_template.id,
				});
				
				if player.id == disp.state.iface_settings.cur_player {
					disp.create_interrupt_window(UIMode::PrevailingDoctrineChangedWindow(PrevailingDoctrineChangedWindowState {}));
				}
			}
		}
	}
	
	research_techs(players, &mut gstate.relations, temps, disp);
	
	gstate.turn += 1;
	disp.state.iface_settings.update_all_player_pieces_mvd_flag(units);
	
	// prevent unit_subsel out-of-bounds error -- in the case a unit was deleted and was prev selected
	// we should not reset unit_subsel if we are moving an individual unit
	match &disp.state.iface_settings.add_action_to {
		AddActionTo::None | AddActionTo::NoUnit {..} |
		AddActionTo::BrigadeBuildList {..} | AddActionTo::AllInBrigade {..} => {
			if let Some(ex) = exs.last().unwrap().get(&disp.state.iface_settings.cursor_to_map_ind(map_data)) {
				if let Some(unit_inds) = &ex.unit_inds {
					if disp.state.iface_settings.unit_subsel >= unit_inds.len() {
						disp.state.iface_settings.unit_subsel = 0;
					}
				}else{
					disp.state.iface_settings.unit_subsel = 0;
				}
			}else{
				disp.state.iface_settings.unit_subsel = 0;
			}
		}
		AddActionTo::IndividualUnit {..} => {}
	}
	
	{ ///////////////////////// update gold
		#[cfg(feature="profile")]
		let _g = Guard::new("update gold");
		
		for owner_id in 0..players.len() {
			let player = &players[owner_id];
			match player.ptype {
				PlayerType::Barbarian(_) => {continue;}
				PlayerType::Empire(_) | PlayerType::Human(_) | PlayerType::Nobility(_) => {}
			}
			if !player.stats.alive {continue;}
			
			struct Expense { // for units and blgs (separately)
				upkeep: f32,
				ind: usize
			}
			
			//////////// remove units
			if in_debt(&player.stats, players, &gstate.relations) {
				/////////// find max unit to remove
				let mut unit_expenses = Vec::with_capacity(units.len());
				
				for (unit_ind, u) in units.iter().enumerate().filter(|(_, u)| u.owner_id == owner_id as u32) {
					unit_expenses.push(Expense {upkeep: u.template.upkeep, ind: unit_ind});
				}
				
				// sort from least to greatest
				unit_expenses.sort_by(|a, b| a.upkeep.partial_cmp(&b.upkeep).unwrap_or(Ordering::Less));
				
				// remove units
				while in_debt(&players[owner_id].stats, players, &gstate.relations) {
					if let Some(unit_expense) = unit_expenses.pop() {
						// log
						gstate.log_event(LogType::UnitDisbanded {
							owner_id,
							unit_nm: units[unit_expense.ind].nm.clone(),
							unit_type_nm: units[unit_expense.ind].template.nm[disp.state.local.lang_ind].clone()
						});
						
						disband_unit(unit_expense.ind, owner_id as SmSvType == disp.state.iface_settings.cur_player, units, map_data, exs, players, gstate, map_sz);
						
						// last unit_ind should be updated to unit_expense.ind because it swapped position
						for unit_expense_update in unit_expenses.iter_mut().filter(|unit_expense_update| unit_expense_update.ind == units.len()) {
							*unit_expense_update = unit_expense;
							break;
						}
					} else {break;} // no more units to remove
				}
			}
			
			let mut n_cities = bldgs.iter().filter(|b| { 
				if let BldgArgs::PopulationCenter {..} = &b.args {
					return b.owner_id == owner_id as SmSvType;
				}
				false
			}).count();
			
			///////// remove buildings and possibly collapse civ
			if in_debt(&players[owner_id].stats, players, &gstate.relations) {
				/////////// find max bldg to remove
				let mut bldg_expenses = Vec::with_capacity(bldgs.len());
				
				for (bldg_ind, b) in bldgs.iter().enumerate() {
					if b.owner_id != owner_id as u32 {continue;}
					bldg_expenses.push(Expense {upkeep: b.template.upkeep, ind: bldg_ind});
				}
				
				// sort from least to greatest
				bldg_expenses.sort_by(|a, b| a.upkeep.partial_cmp(&b.upkeep).unwrap_or(Ordering::Less));
				
				// remove bldgs
				while in_debt(&players[owner_id].stats, players, &gstate.relations) {
					if let Some(bldg_expense) = bldg_expenses.pop() {
						{ // log bldg or city being destroyed 
							let b_rm = &bldgs[bldg_expense.ind];
							if let BldgArgs::PopulationCenter {nm, ..} = &b_rm.args {
								gstate.log_event(LogType::CityDisbanded {owner_id, city_nm: nm.clone()});
								
								n_cities -= 1;
							}else if let BldgType::Gov(_) = b_rm.template.bldg_type {
								gstate.log_event(LogType::BldgDisbanded {
									owner_id,
									bldg_nm: b_rm.template.nm[disp.state.local.lang_ind].clone()
								});
							}
						}
						
						rm_bldg(bldg_expense.ind, owner_id as SmSvType == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
						
						// last bldg_ind should be updated to bldg_expense.ind because it swapped position
						for bldg_expense_update in bldg_expenses.iter_mut() {
							if bldg_expense_update.ind != bldgs.len() {continue;}
							*bldg_expense_update = bldg_expense;
							break;
						}
					} else { // no more bldgs to remove, but still in debt?
						#[cfg(any(feature="opt_debug", debug_assertions))]
						{
							endwin();
							let pstats = &players[owner_id].stats;
							
							println!("owner_id {}", owner_id);
							println!("gold {}", pstats.gold);
							println!("employed {}", pstats.employed);
							println!("tax_income {}", pstats.tax_income);
							println!("unit_expenses {}", pstats.unit_expenses);
							println!("bldg_expenses {}", pstats.bldg_expenses);
							panicq!("civilization in debt but has no buildings or units");
						}
						#[cfg(not(any(feature="opt_debug", debug_assertions)))]
						break;
					}
				}	
			}
			
			/////////// civ collapsed -- removed all units, bldgs, zones
			if n_cities == 0 && players[owner_id].req_population_center(gstate.turn) {
				gstate.log_event(LogType::CivCollapsed {owner_id});
				civ_collapsed(owner_id, &mut None, players, units, bldgs, map_data, exs, gstate, map_sz, temps, disp);
			}
			
			///////// update gold
			let net_income = players[owner_id].stats.net_income(players, &gstate.relations);
			let pstats = &mut players[owner_id].stats;
			pstats.gold += net_income;
			debug_assertq!(pstats.gold >= 0. || approx_eq_tol(pstats.gold, 0., TOL), "negative gold {}, owner: {}, tax_income {} unit_expenses {} bldg_expenses {} bldgs.len() {}",
					pstats.gold, owner_id, pstats.tax_income, pstats.unit_expenses, pstats.bldg_expenses, bldgs.len());
		}
	}
	
	////////////////////// logging
	if (gstate.turn % LOG_TURNS) == 0 {
		#[cfg(feature="profile")]
		let _g = Guard::new("logging");
	
		// log gold, population, zone demands
		//for player in players.iter_mut() {
		for player_ind in 0..players.len() {
			let net_income = players[player_ind].stats.net_income(players, &gstate.relations);
			players[player_ind].stats.net_income_log.push(net_income);
			
			let player = &mut players[player_ind];
			let pstats = &mut player.stats;
			pstats.alive_log.push(pstats.alive);
			pstats.population_log.push(pstats.population);
			pstats.population_wealth_level_log.push(pstats.population_wealth_level.clone());
			pstats.unemployed_log.push(if pstats.population == 0 {0.} else {
					100.*(pstats.population - pstats.employed) as f32 / pstats.population as f32});
			pstats.gold_log.push(pstats.gold);
			
			// happiness, crime, doctrinality, pacifism, health
			// 	(all affected by doctrinality bonuses)
			{
				let d = pstats.doctrine_template;
				
				pstats.happiness_log.push(pstats.locally_logged.happiness_sum); // don't include doctrine bonus because that was already included when computing pstats.happiness
				pstats.crime_log.push(pstats.crime + d.crime_bonus);
				pstats.doctrinality_log.push(pstats.locally_logged.doctrinality_sum.clone());
				pstats.pacifism_log.push(pstats.locally_logged.pacifism_sum + d.pacifism_bonus);
				pstats.health_log.push(pstats.health + d.health_bonus);
			}
			
			pstats.defense_power_log.push(0);
			pstats.offense_power_log.push(0);
			
			///// zone demand logging
			let mut zone_demand_sums = Vec::with_capacity(4); // entry for each zone type
			
			for (zone_ind, zdsm) in pstats.zone_demand_sum_map.iter().enumerate() { // across zone types
				zone_demand_sums.push(zdsm.map_avg_zone_demand(ZoneType::from(zone_ind), bldgs, player.id));
			}
			//printlnq!("{:#?}", zone_demand_sums);
			pstats.zone_demand_log.push(zone_demand_sums);
			
			pstats.research_per_turn_log.push(pstats.research_per_turn);
			
			{ // research completed
				let mut research_completed = 0;
				for (tp, tech_template) in pstats.techs_progress.iter()
									.zip(temps.techs.iter()) {
					research_completed += match tp {
						TechProg::Prog(prog) => {*prog}
						TechProg::Finished => tech_template.research_req
					} as usize;
				}
				pstats.research_completed_log.push(research_completed);
			}
			
			if player.id == HUMAN_PLAYER_ID {
				pstats.mpd_log.push(frame_stats.dur_mean);
			}
			
			//endwin();
			//panicq!("zdsm len {} pstats {}", pstats.zone_demand_sum_map.len(),
			//		pstats.zone_demand_log.last().unwrap().len());
		}
		
		///////////////////
		// log offense and defense power
		let point_update = players[0].stats.defense_power_log.len() - 1;
		
		for u in units.iter() {
			let pstats = &mut players[u.owner_id as usize].stats;
			pstats.defense_power_log[point_update] += u.template.max_health;
			
			if let Some(attack_per_turn) = u.template.attack_per_turn {
				pstats.offense_power_log[point_update] += attack_per_turn;
			}
		}
	}
	
	{ //////////////////////// auto-save
		let day = (gstate.turn) % (30*12);
		if day == 0 {
			let year = gstate.turn/(30*12);
			if disp.state.iface_settings.checkpoint_freq != 0 && (year % disp.state.iface_settings.checkpoint_freq as usize) == 0 {
				save_game(SaveType::Auto, gstate, map_data, exs, temps, bldgs, units, players, &mut disp.state, frame_stats);
			}
		}
	}
	
	#[cfg(any(feature="opt_debug", debug_assertions))]
	chk_data(units, bldgs, exs, players, &gstate.relations, temps.doctrines, map_data, map_sz);
	
	frame_stats.update(frame_start);
}

