use super::*;
use crate::gcore::*;
use crate::map::*;
use crate::disp::*;
use crate::disp::menus::*;
use crate::renderer::*;
use crate::ai::*;
use crate::zones::*;
use crate::buildings::*;
//use std::time::Duration;
//use std::thread;
use crate::player::{PlayerType};
use crate::containers::*;

pub fn do_attack_action<'f,'bt,'ut,'rt,'dt>(unit_ind: usize, disband_unit_inds: &mut Vec<usize>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
		map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		disp: &mut Disp<'f,'bt,'ut,'rt,'dt>, cur_ui_ai_player_is_paused: Option<bool>,
		map_sz: MapSz, frame_stats: &mut FrameStats) {
	#[cfg(feature="profile")]
	let _g = Guard::new("do_attack_actions");
	
	let cur_player = disp.state.iface_settings.cur_player as usize;
	
	// perform attacks first (before any other action) to prevent units from constantly moving away
	let u = &mut units[unit_ind];
	
	// no actions left to take
	if u.actions_used.is_none() {return;}
	
	// to be deleted
	if disband_unit_inds.contains(&unit_ind) {return;}
	
	// idle
	if u.action.len() == 0 {return;}
	
	let action = u.action.last_mut().unwrap();
	let path_coords_len = action.path_coords.len();
	let action_type = &mut action.action_type;
	
	macro_rules! rm_units_and_destroy_civ{($owner_prev: expr) => {
		// remove all units
		for unit_ind in (0..units.len()).rev() {
			if units[unit_ind].owner_id == ($owner_prev as SmSvType) {
				if !disband_unit_inds.contains(&unit_ind) {
					disband_unit_inds.push(unit_ind);
				}
			}
		}
		
		// log
		gstate.logs.push(Log {turn: gstate.turn,
				   val: LogType::CivDestroyed {
				   owner_attackee_id: $owner_prev,
				   owner_attacker_id: units[unit_ind].owner_id as usize
				}});
		
		civ_destroyed(&mut players[$owner_prev], gstate, disp);
	};};

	///////////////////////////////////////// move units
	if path_coords_len != 0 {
		if let ActionType::Attack{..} = action_type {
			debug_assertq!(u.action.last().unwrap().actions_req > 0.);
			debug_assertq!(u.action.last().unwrap().path_coords.len() > 0);
			
			////////// debug
			/*{
				printlnq!("moving {} for attack, actions: {}", unit_ind, units[unit_ind].action.len());
				if let Some(action) = &units[unit_ind].action.last() {
					println!("\t\t{} path coords len {} action_meta_cont {}",
							action.action_type, action.path_coords.len(),
							action.action_meta_cont.is_none());
				}
			}*/
			////////////////////
			
			mv_unit(unit_ind, u.owner_id == disp.state.iface_settings.cur_player, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Record(disband_unit_inds));
			
			//////////////////// debug
			/*{
				println!("\tpost moving {} for attack, actions: {}", unit_ind, units[unit_ind].action.len());
				if let Some(action) = &units[unit_ind].action.last() {
					println!("\t\t{} path coords len {} action_meta_cont {}",
								action.action_type, action.path_coords.len(),
								action.action_meta_cont.is_none());
				}
			}*/
			////////////////////
		}
	//////////////////////////////////////// attack
	}else if let ActionType::Attack {attack_coord, attackee, ..} = action_type {
		//printlnq!("performing attack for unit {} (owner {})", unit_ind, u.owner_id);
		
		macro_rules! reset_cont{() => {
			units[unit_ind].action.pop();
			return;
		};};
		
		if attackee.is_none() {reset_cont!();}
		
		let attack_owner = attackee.unwrap();
		let attack_coord = attack_coord.unwrap(); // final dest, where the attacked units should be
		
		// shouldn't be attacking self
		debug_assertq!(attack_owner != u.owner_id);
		
		// cannot attack due to active peace treaty
		if let Some(_) = gstate.relations.peace_treaty_turns_remaining(u.owner_id as usize, attack_owner as usize, gstate.turn) {
			reset_cont!();
		}
		
		macro_rules! disable_auto_turn{() => {
			if disp.state.iface_settings.interrupt_auto_turn && attack_owner == disp.state.iface_settings.cur_player {
				disp.state.set_auto_turn(AutoTurn::Off);
				disp.state.update_menu_indicators(cur_ui_ai_player_is_paused);
				disp.center_on_next_unmoved_menu_item(false, FindType::Coord(attack_coord), map_data, exs, units, bldgs, gstate, players);
			}
		};};
		
		// special animation & actions for ICBMs
		if u.template.nm[0] == ICBM_NM {
			//printlnq!("icbm dropped on {} from {}", attack_owner, u.owner_id);
			gstate.logs.push(Log {turn: gstate.turn, val: LogType::ICBMDetonation {owner_id: u.owner_id as usize}});
			gstate.relations.declare_war(u.owner_id as usize, attack_owner as usize, &mut gstate.logs, players, gstate.turn, cur_ui_ai_player_is_paused, &mut gstate.rng, disp);
			
			// lower mood
			{
				const N_THREATEN_EQUIVS: usize = 10;
				for _ in 0..N_THREATEN_EQUIVS {
					gstate.relations.threaten(u.owner_id as usize, attack_owner as usize, gstate.turn);
				}
			}
			
			disable_auto_turn!();
			
			disp.end_window();
			
			// center on ICBM
			let u = &mut units[unit_ind];
			disp.center_on_next_unmoved_menu_item(false, FindType::Coord(u.return_coord()), map_data, exs, units, bldgs, gstate, players);
			
			disp.print_map(map_data, units, bldgs, players, temps, exs, gstate, frame_stats, 0);
			disp.update_cursor(&players[cur_player].stats, map_data);
			disp.state.renderer.refresh();
			
			let u = &units[unit_ind];
			let attacker_owner_id = u.owner_id as usize;
			let attacker_is_cur_player = cur_player == attacker_owner_id;
			
			const BLAST_RADIUS: i32 = 40;
			
			let mut owners_w_city_halls_rmd = Vec::with_capacity(players.len());
			let mut coords_zone_or_ex_rmd = Vec::with_capacity((BLAST_RADIUS*BLAST_RADIUS) as usize);
			
			for radius in 1_i32..BLAST_RADIUS {
				// r^2 = y^2 + x^2
				for y in (-radius)..radius {
					let x_max = (((radius*radius - y*y) as f32).sqrt() * 2.).round() as i32;
					
					let y_pos = disp.state.iface_settings.cur.y as i32 + y;
					if y_pos >= 0 && y_pos < disp.state.iface_settings.screen_sz.h as i32 {
						disp.mv(y_pos, disp.state.iface_settings.cur.x as i32 - x_max);
					}
					for x in (-x_max)..x_max {
						let x_pos = disp.state.iface_settings.cur.x as i32 + x;
						
						// display explosion or bomb
						if y_pos >= 0 && y_pos < disp.state.iface_settings.screen_sz.h as i32 && x_pos >= 0 && x_pos < disp.state.iface_settings.screen_sz.w as i32 {
							// show explosion
							if y != 0 || x != 0 {
								let cur_radius = ((y*y) as f32 + ((x*x) as f32 / 4.)).sqrt();
								
								let color = COLOR_PAIR(
									// show random color
									if gstate.rng.gen_f32b() < ((1. + cur_radius) / BLAST_RADIUS as f32) {
										const N_COLORS: f32 = 3.;
										
										let rand = gstate.rng.gen_f32b();
										if rand < 1./N_COLORS {CREDSAND3}
										else if rand < 2./N_COLORS {CREDSAND4}
										else {CRED}
										
									// show color based on distance from center
									}else{
										const N_COLORS: f32 = 4.;
										let step = radius as f32 / N_COLORS;
										
										if cur_radius > (3.*step) {CREDSAND3}
										else if cur_radius > (2.*step) {CREDSAND4}
										else if cur_radius > step {CRED}
										else {CWHITE}
									});
								
								disp.attron(color);
								disp.addch(disp.state.chars.land_char);
								disp.attroff(color);
							// show bomb
							}else{
								set_player_color(&players[u.owner_id as usize], true, &mut disp.state.renderer);
								disp.addch(u.template.char_disp);
								set_player_color(&players[u.owner_id as usize], false, &mut disp.state.renderer);
							}
						}
						
						// rm zones, units, bldgs
						{
							let cur_map_coord = disp.state.iface_settings.screen_coord_to_map_ind(Coord {
									y: y_pos as isize, x: x_pos as isize}, map_data);
							
							// already removed
							if coords_zone_or_ex_rmd.contains(&cur_map_coord) {continue;}
							
							if let Some(ex) = exs.last_mut().unwrap().get_mut(&cur_map_coord) {
								coords_zone_or_ex_rmd.push(cur_map_coord);
								
								// rm zone
								ex.actual.rm_zone(cur_map_coord, players, temps.doctrines, map_sz);
								
								// disband units
								if let Some(unit_inds) = &ex.unit_inds {
									for unit_ind in unit_inds.iter() {
										if !disband_unit_inds.contains(unit_ind) {
											disband_unit_inds.push(*unit_ind);
										}
									}
								}
								
								// rm wall
								if let Some(structure) = &ex.actual.structure {
									match structure.structure_type {
										StructureType::Wall | StructureType::Gate => {
											ex.actual.rm_structure(cur_map_coord, players, map_sz);
										} StructureType::Road => {
										} StructureType::N => {panicq!("invalid structure");}
									}
								}
								
								// rm bldg
								if let Some(bldg_ind) = ex.bldg_ind {
									let b = &bldgs[bldg_ind];
									
									// log
									if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
										let b_owner = b.owner_id as usize;
										if !owners_w_city_halls_rmd.contains(&b_owner) {
											owners_w_city_halls_rmd.push(b_owner);
										}
										
										gstate.logs.push(Log {turn: gstate.turn,
											   val: LogType::CityDestroyed {
													city_attackee_nm: nm.clone(),
													
													owner_attackee_id: b_owner,
													owner_attacker_id: u.owner_id as usize,
										}});
									}
									rm_bldg(bldg_ind, b.owner_id == disp.state.iface_settings.cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Record (disband_unit_inds), map_sz);	
								}
							}
						}
					}
				}
				
				disp.update_cursor(&players[cur_player].stats, map_data);
				disp.state.renderer.refresh();
				//thread::sleep(Duration::from_millis(2));
			}
			
			// update map
			for coord in coords_zone_or_ex_rmd {
				compute_zooms_coord(coord, bldgs, temps.bldgs, map_data, exs, players);
				compute_active_window(coord, attacker_is_cur_player, PresenceAction::SetAbsent, map_data, exs, &mut players[attacker_owner_id].stats, map_sz, gstate, units);
			}

			// remove ICBM
			if !disband_unit_inds.contains(&unit_ind) {
				disband_unit_inds.push(unit_ind);
			}
			
			///////////////////////////
			// check if any countries have been destroyed
			for owner_prev in owners_w_city_halls_rmd {
				// no remaining city halls?
				if bldgs.iter().any(|b| b.owner_id == (owner_prev as SmSvType) && b.template.nm[0] == CITY_HALL_NM) {
					continue;
				}
				
				let is_cur_player = owner_prev as SmSvType == disp.state.iface_settings.cur_player;
				
				// rm bldgs in reverse order to avoid index issues
				for bldg_ind in (0..bldgs.len()).rev() {
					let b = &bldgs[bldg_ind];
					let b_coord = b.coord;
					
					if (owner_prev as SmSvType) == b.owner_id {
						rm_bldg(bldg_ind, is_cur_player, bldgs, temps.bldgs, map_data, exs, players, UnitDelAction::Record(disband_unit_inds), map_sz);
						compute_active_window(b_coord, attacker_is_cur_player, PresenceAction::SetAbsent, map_data, exs, &mut players[attacker_owner_id].stats, map_sz, gstate, units);
					}
				}
				
				rm_player_zones(owner_prev, bldgs, temps, players, exs, map_data, map_sz);
				rm_units_and_destroy_civ!(owner_prev);
			}
			
			return;
		}
		
		// no ex data
		let ex_wrapped = exs.last_mut().unwrap().get_mut(&attack_coord);
		if ex_wrapped.is_none() {reset_cont!();}
		let ex = &mut ex_wrapped.unwrap();
		
		/////////////////
		// attack unit
		if let Some(unit_inds) = &ex.unit_inds {
			///////////////// check owner of attacked unit & that it hasn't already been disbanded				
			for unit_ind in unit_inds {
				if units[*unit_ind].owner_id != attack_owner || disband_unit_inds.contains(unit_ind) {
					reset_cont!();
				}
			}
			
			//////////////////// attack
			let ut = units[unit_ind].template;
			
			// find healthiest unit to attack
			let mut max_health = 0;
			let mut max_health_unit_ind = 0;
			
			for (i, unit_ind) in unit_inds.iter().enumerate() {
				if i == 0 || max_health < units[*unit_ind].health {
					max_health = units[*unit_ind].health;
					max_health_unit_ind = *unit_ind;
				}
			}
			
			// fortify bonus
			let (fortify_scale_down, fortify_scale_up) = if let Some(ActionMeta {action_type: ActionType::Fortify{..}, ..}) = 
											&units[max_health_unit_ind].action.last() {
				const FORTIFY_BONUS: f32 = 0.25;
				(1. - FORTIFY_BONUS, 1. + FORTIFY_BONUS)
			}else{
				(1., 1.)
			};
			
			let attackee_id = units[max_health_unit_ind].owner_id as usize;
			let attacker_id = units[unit_ind].owner_id as usize;
			
			if attackee_id != attacker_id {
				gstate.relations.declare_war(attackee_id, attacker_id, &mut gstate.logs, players, gstate.turn, cur_ui_ai_player_is_paused, &mut gstate.rng, disp);
			}
			
			let attackee_bonus = players[attackee_id].stats.bonuses.combat_factor;
			let attacker_bonus = players[attacker_id].stats.bonuses.combat_factor;
			
			// attackee health and survival
			let attackee_survives = {
				// attackee does not survive
				let effective_attack_damage = ((ut.attack_per_turn.unwrap() as f32)*fortify_scale_down*attacker_bonus).round() as usize;
				if effective_attack_damage >= max_health {
					if !disband_unit_inds.contains(&max_health_unit_ind) {
						disband_unit_inds.push(max_health_unit_ind);
					}
					
					// log
					{
						let u_attackee = &units[max_health_unit_ind];
						let u_attacker = &units[unit_ind];
						let l = &disp.state.local;
						gstate.logs.push(Log {turn: gstate.turn,
								   val: if u_attackee.template.nm[0] != RIOTER_NM {
								   	LogType::UnitDestroyed {
										unit_attackee_nm: u_attackee.nm.clone(),
										unit_attacker_nm: u_attacker.nm.clone(),
										
										unit_attackee_type_nm: u_attackee.template.nm[l.lang_ind].clone(),
										unit_attacker_type_nm: u_attacker.template.nm[l.lang_ind].clone(),
										
										owner_attackee_id: u_attackee.owner_id as usize,
										owner_attacker_id: u_attacker.owner_id as usize}
									}else{
										LogType::RiotersAttacked {owner_id: u_attacker.owner_id as usize}
							}});
					}
					
					false
					
				// unit survives attack -- decrease its health and log
				}else{
					units[max_health_unit_ind].health -= effective_attack_damage;
					debug_assertq!(units[max_health_unit_ind].health > 0);
					
					// log unit attacked
					if attack_owner as usize == cur_player {
						let u_attackee = &units[max_health_unit_ind];
						let u_attacker = &units[unit_ind];
						let l = &disp.state.local;
						if u_attackee.template.nm[0] != RIOTER_NM {
							gstate.logs.push(Log {turn: gstate.turn,
									   val: LogType::UnitAttacked {
												unit_attackee_nm: u_attackee.nm.clone(),
												unit_attacker_nm: u_attacker.nm.clone(),
												
												unit_attackee_type_nm: u_attackee.template.nm[l.lang_ind].clone(),
												unit_attacker_type_nm: u_attacker.template.nm[l.lang_ind].clone(),
												
												owner_attackee_id: u_attackee.owner_id as usize,
												owner_attacker_id: u_attacker.owner_id as usize,
							}});
						}
					} // log
					
					true
				} // unit survives
			};
			
			// attacker health and survival
			{
				if let Some(attack_per_turn) = units[max_health_unit_ind].template.attack_per_turn {
					// attacker does not survive
					let effective_attack_damage = ((attack_per_turn as f32)*fortify_scale_up*attackee_bonus).round() as usize;
					if effective_attack_damage >= units[unit_ind].health {
						debug_assertq!(!disband_unit_inds.contains(&unit_ind));
						disband_unit_inds.push(unit_ind);
						
						// log
						{
							let u_attackee = &units[unit_ind];
							let u_attacker = &units[max_health_unit_ind];
							let l = &disp.state.local;
							gstate.logs.push(Log {turn: gstate.turn, val: LogType::UnitDestroyed {
											unit_attackee_nm: u_attackee.nm.clone(),
											unit_attacker_nm: u_attacker.nm.clone(),
											
											unit_attackee_type_nm: u_attackee.template.nm[l.lang_ind].clone(),
											unit_attacker_type_nm: u_attacker.template.nm[l.lang_ind].clone(),
											
											owner_attackee_id: u_attackee.owner_id as usize,
											owner_attacker_id: u_attacker.owner_id as usize}
							});
						}
						
					// attacker survives, decrease health
					}else{
						units[unit_ind].health -= effective_attack_damage;
					}
				}
			}
			
			// prevent attack action from being removed
			if attackee_survives {
				disable_auto_turn!();
				return;
			}
		
		/////////////
		// attack structure
		}else if let Some(structure) = &mut ex.actual.structure {
			// if it's owned by the attacking player or a road, cancel the action
			if ex.actual.owner_id == Some(u.owner_id) || match structure.structure_type {
					StructureType::Wall | StructureType::Gate => {false}
					StructureType::Road => {true}
					StructureType::N => {panicq!("invalid structure");}
			} {reset_cont!();}
			
			let mut effective_attack_damage = u.template.attack_per_turn.unwrap() / 2;
			if let Some(siege_bonus) = u.template.siege_bonus_per_turn {
				effective_attack_damage += siege_bonus;
			}
			
			// gates destroyed more quickly
			match structure.structure_type {
				StructureType::Wall | StructureType::Road => {}
				StructureType::Gate => {effective_attack_damage *= 2;}
				StructureType::N => {panicq!("invalid structure");}
			}
			
			let attackee_owner_id = ex.actual.owner_id.unwrap() as usize;
			let structure_type = structure.structure_type;
			
			// structure is destroyed
			if (structure.health as usize) <= effective_attack_damage {
				ex.actual.rm_structure(attack_coord, players, map_sz);
				compute_zooms_coord(attack_coord, bldgs, temps.bldgs, map_data, exs, players);
				u.action.pop();
			// structure is damaged, continue attacking
			}else{
				// record wall as being damaged
				players[attackee_owner_id].log_damaged_wall(Coord::frm_ind(attack_coord, map_sz));
				
				structure.health -= effective_attack_damage as u8;
			}
			
			// log
			{
				let u_attacker = &units[unit_ind];
				gstate.relations.declare_war(u_attacker.owner_id as usize, attackee_owner_id, &mut gstate.logs, players, gstate.turn, cur_ui_ai_player_is_paused, &mut gstate.rng, disp);
				
				// only log humans and active UI AI as having walls/structures destroyed
				// also only log if this attacker has not attacked this wall in WALL_ATTACK_LOG_DELAY days
				const WALL_ATTACK_LOG_DELAY: usize = 30;
				if players[attackee_owner_id].ptype.is_human() || attackee_owner_id == cur_player {
					let l = &disp.state.local;
					let attacked_log = Log {turn: gstate.turn,
						   val: LogType::StructureAttacked {
							structure_coord: attack_coord,
							unit_attacker_nm: u_attacker.nm.clone(),
							unit_attacker_type_nm: u_attacker.template.nm[l.lang_ind].clone(),
							
							structure_type,
							
							owner_attackee_id: attackee_owner_id,
							owner_attacker_id: u_attacker.owner_id as usize,
					}};
					
					// check if already logged and return from this fn so auto turn increment isn't disabled
					let turn_threshold = if gstate.turn > WALL_ATTACK_LOG_DELAY
						{gstate.turn - WALL_ATTACK_LOG_DELAY} else {0};
					
					for log in gstate.logs.iter().rev() {
						// only check so far back in time
						if log.turn < turn_threshold {
							break;
						}
						
						if log.val == attacked_log.val {
							reset_cont!();
						}
					}
					
					gstate.logs.push(attacked_log);
				}
			}
			
			disable_auto_turn!();
			return; // return to avoid removing action
			
		//////////////////
		// attack building
		}else if let Some(bldg_ind) = ex.bldg_ind {
			let b = &mut bldgs[bldg_ind];
			
			if b.owner_id == u.owner_id {reset_cont!();}
			
			// city hall
			if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
				let city_attackee_nm = nm.clone();
				
				// if at city hall, check at entrance
				let bldg_coord = Coord::frm_ind(b.coord, map_sz);
				
				let sz = b.template.sz;
				
				let w = sz.w as isize;
				let h = sz.h as isize;
				
				//// not at entrance
				//if map_sz.coord_wrap(bldg_coord.y + h-1, bldg_coord.x + w/2).unwrap() != attack_coord {reset_cont!();}
				
				let owner_prev = b.owner_id as usize;
				players[owner_prev].stats.bldg_expenses -= b.template.upkeep;
				let u_owner_id = u.owner_id as usize;
				// set owner, update map
				b.owner_id = u.owner_id;
				
				players[u_owner_id].stats.bldg_expenses += b.template.upkeep;
				
				for i_off in 0..h {
				for j_off in 0..w {
					let coord = map_sz.coord_wrap(bldg_coord.y + i_off, bldg_coord.x + j_off).unwrap();
					let ex = &mut exs.last_mut().unwrap().get_mut(&coord).unwrap();
					ex.actual.owner_id = Some(u_owner_id as SmSvType);
					compute_zooms_coord(coord, bldgs, temps.bldgs, map_data, exs, players);
				}}
				
				//printlnq!("{} attacking {} at {}", u.owner_id, owner_prev, bldg_coord);
				
				gstate.relations.declare_war(u_owner_id, owner_prev, &mut gstate.logs, players, gstate.turn, cur_ui_ai_player_is_paused, &mut gstate.rng, disp);
				
				// loading screen
				let prev_visibility = {
					disp.state.print_clear_centered_logo_txt(if u_owner_id == cur_player || owner_prev == cur_player {
						disp.state.local.A_new_order_is_being_established.clone()
					}else{
						disp.state.local.Please_wait_violent_take_over.clone()
					});
					disp.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE)
				};
				
				//printlnq!("bldg coord {} from {} to {}", Coord::frm_ind(bldgs[bldg_ind].coord, map_sz), owner_prev, u_owner_id);
				let mut to_city_state_opt = transfer_city_ai_state(bldg_ind, owner_prev, u_owner_id, cur_player, units, bldgs, temps, exs, map_data, players, gstate, map_sz);
				
				////////////////////////////////////////////
				// set surrounding land ownership, update stats
				{
					let exf = exs.last().unwrap();
					let zone_exs = &players[owner_prev].zone_exs;
					
					// save all exs which are zoned which use the city hall that is changing ownership
					let mut adj_stack = Vec::new();
					for (coord, ex) in exf.iter() {
						// make sure owner matches and ex is actually a zone
						if ex.actual.owner_id != Some(owner_prev as SmSvType) || ex.actual.ret_zone_type() == None {continue;}
						
						// ex has demand set
						if let Some(zone_ex) = zone_exs.get(&return_zone_coord(*coord, map_sz)) {
							// demand is calculated as using the city hall that changed ownership
							if let Dist::Is {bldg_ind: bldg_ind_ch, ..} = zone_ex.ret_city_hall_dist() {
								if bldg_ind_ch == bldg_ind {
									adj_stack.push(*coord);
								}
							}
						}
					}
					
					///////// update owner info/bldg info in adj_stack[]
					if let Some(to_city_state) = &mut to_city_state_opt {
						set_all_adj_owner(adj_stack, u_owner_id, owner_prev, cur_player, &mut Some(to_city_state), units, bldgs, temps, exs, players, map_data, gstate, map_sz);
					}else{
						set_all_adj_owner(adj_stack, u_owner_id, owner_prev, cur_player, &mut None, units, bldgs, temps, exs, players, map_data, gstate, map_sz);
					}
				}
				
				// log
				gstate.logs.push(Log {turn: gstate.turn,
					   val: LogType::CityCaptured {
							city_attackee_nm,
							owner_attackee_id: owner_prev,
							owner_attacker_id: units[unit_ind].owner_id as usize
						}});
				
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// country destroyed because of no remaining city halls?
				if !bldgs.iter().any(|b| b.owner_id == (owner_prev as SmSvType) && b.template.nm[0] == CITY_HALL_NM) {
					////////////////////////////////// set all land over to attacker
					
					// gather coords
					let exf = exs.last().unwrap();
					let mut coords = Vec::with_capacity(exf.len());
					for (coord, ex) in exf.iter() {
						if ex.actual.owner_id == Some(owner_prev as SmSvType) {
							coords.push(*coord);
						}
					}
					
					// set land over to attacker
					for coord in coords {
						if let Some(to_city_state) = &mut to_city_state_opt {
							set_owner(coord, u_owner_id, owner_prev, cur_player, &mut Some(to_city_state), units, bldgs, temps, exs, players, map_data, gstate, map_sz);
						}else{
							set_owner(coord, u_owner_id, owner_prev, cur_player, &mut None, units, bldgs, temps, exs, players, map_data, gstate, map_sz);
						}
					}
					
					////////////////////////////////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					
					rm_units_and_destroy_civ!(owner_prev);
				}
				
				// push city state into attacker
				if let Some(to_city_state) = to_city_state_opt {
					if let Some(ai_state) = players[u_owner_id].ptype.any_ai_state_mut() {
						ai_state.city_states.push(to_city_state);
					}
				}
				
				disp.state.renderer.curs_set(prev_visibility);
			
			// barbarian camp
			}else if b.template.nm[0] == BARBARIAN_CAMP_NM {
				let attackee_id = b.owner_id as usize;
				let attackee = &mut players[attackee_id];
				debug_assertq!(attackee.stats.alive);
				if let PlayerType::Barbarian {..} = attackee.ptype {
					attackee.stats.alive = false;
					
					// remove units
					if let PlayerType::Barbarian(barbarian_state) = &mut attackee.ptype {
						for defender in barbarian_state.defender_inds.iter() {
							debug_assertq!(!disband_unit_inds.contains(defender));
							disband_unit_inds.push(*defender);
						}
						
						for attacker in barbarian_state.attacker_inds.iter() {
							debug_assertq!(!disband_unit_inds.contains(attacker));
							disband_unit_inds.push(*attacker);
						}
					}
					
					// log
					gstate.logs.push(Log {turn: gstate.turn,
						   val: LogType::CivDestroyed {
							owner_attackee_id: attackee_id,
							owner_attacker_id: units[unit_ind].owner_id as usize
						}});					
				} // pressumably barbarian_states = None occurs when a barbarian takes over a city
				// then if the AI later takes over the city, it then owns the camp the 
				// barbarian owned, so the AI can have the camp but no barbarian state
				
				rm_bldg(bldg_ind, b.owner_id == cur_player as SmSvType, bldgs, temps.bldgs, map_data, exs, players,
						  UnitDelAction::Record (disband_unit_inds), map_sz);
				
				//players[attackee_id].ptype = None;
				
			}//else {reset_cont!();}
		}else{reset_cont!();}
		
		disable_auto_turn!();
		reset_cont!(); // removes attack action
	} // attack
}

