use super::*;
use crate::map::*;
use crate::player::*;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;
use crate::saving::SmSvType;
//use crate::gcore::dbg_log;
//use crate::disp_lib::endwin;

impl <'bt,'ut,'rt,'dt>EmpireState<'bt,'ut,'rt,'dt> {
	pub fn plan_actions(ai_ind: usize, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			gstate: &mut GameState, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, disband_unit_inds: &mut Vec<usize>, map_sz: MapSz, disp: &mut Disp, cur_ui_ai_player_is_paused: Option<bool>) {
		#[cfg(feature="profile")]
		let _g = Guard::new("plan_unit_actions");
		
		let player = &players[ai_ind];
		let pstats = &player.stats;
		if let PlayerType::Empire(EmpireState {ai_state, personality}) = &player.ptype {
			if ai_state.paused {return;}
			
			let is_cur_player = ai_ind as SmSvType == disp.state.iface_settings.cur_player;
			
			//////////////////////////// immutable accessing of players
			{ // war planning (declaration)
				const WAR_CHECK_PROB: f32 = 1./(15.*12.);
				const MAX_WAR_DIST: usize = 5000;
				
				let icbm_producable = pstats.unit_producable(UnitTemplate::frm_str(ICBM_NM, temps.units));
				
				if gstate.rng.gen_f32b() <= WAR_CHECK_PROB*(1. - personality.friendliness) {
					let war_enemies = gstate.relations.at_war_with(player.id as usize);
					
					// potential targets
					'attackee_loop: for player_chk in players.iter() {
						if player.id == player_chk.id {continue;} // don't launch war against self
						if war_enemies.contains(&(player_chk.id as usize)) {continue;} // already at war
						
						match player_chk.ptype {
							PlayerType::Barbarian(_) | PlayerType::Nobility(_) => {continue;}
							PlayerType::Empire(_) => {}
							PlayerType::Human(_) => {
								// civ and the human player have not yet met:
								if !gstate.relations.discovered(player_chk.id as usize, player.id as usize) {continue;}
							}
						}
						
						if let Some(adv) = current_war_advantage(player.id as usize, players, &gstate.relations) {
							if adv <= *player_chk.stats.defense_power_log.last().unwrap() as isize {continue;} // cannot overpower target with current wars
							
							// is target close enough to any of the attacker's cities? (being able to produce ICBMs removes this limit)
							for city in ai_state.city_states.iter() {
								let city_loc = Coord::frm_ind(city.coord, map_sz);
								for b in bldgs.iter() {
									if b.owner_id != player_chk.id {continue;}
									if !icbm_producable && MAX_WAR_DIST <= manhattan_dist(Coord::frm_ind(b.coord, map_sz), city_loc, map_sz) {continue;}
									
									gstate.relations.declare_war(player.id as usize, player_chk.id as usize, &mut gstate.logs, players, gstate.turn, cur_ui_ai_player_is_paused, &mut gstate.rng, disp);
									continue 'attackee_loop;
								} // loop over bldgs
							} // loop over attacker cities
						}else {break;} // no defense/offense has been logged yet
					} // loop over players
				} // if we should check if ai declares new wars
			}
			
			/////////////////////////////// mutable accessing of player
			let player = &mut players[ai_ind];
			let pstats = &mut player.stats;
			if let PlayerType::Empire(EmpireState {ai_state, personality}) = &mut player.ptype {
				{ // create city? also realloc unit to new city
					const POPULATION_PER_CITY_THRESH: usize = 150;
					let n_cities = ai_state.city_states.len();
					if pstats.population > (n_cities * POPULATION_PER_CITY_THRESH) {
						let rand_city_ind = gstate.rng.usize_range(0, n_cities);
						let city_parent = &ai_state.city_states[rand_city_ind];
						
						if let Some(&worker_ind) = city_parent.worker_inds.last() {
							match city_parent.neighbors_possible {
								Neighbors::Possible | Neighbors::NotKnown => {
									// check if new location possible
									if let Some(coord) = city_parent.find_new_city_loc(ai_state, pstats, map_data, exs.last().unwrap(), &temps.ai_config, &mut gstate.rng, map_sz) {
										let city_parent = &mut ai_state.city_states[rand_city_ind];
										
										city_parent.neighbors_possible = Neighbors::Possible;
										city_parent.worker_inds.pop(); // remove worker_ind from old city
										
										ai_state.add_city_plan(coord, &mut gstate.rng, map_data, map_sz, temps.bldgs);
										ai_state.city_states.last_mut().unwrap().worker_inds.push(worker_ind); // add worker_ind to new city
										
										//dbg_log("creating new city", owner.id, logs, turn);
									}else{
										ai_state.city_states[rand_city_ind].neighbors_possible = Neighbors::NotPossible;
									}
								} Neighbors::NotPossible => {
									//dbg_log("can't find city location", owner.id, logs, turn);
								} Neighbors:: N => {panicq!("unknown city neighbor state");}
							} // match if neighbors possible
						} // worker in city to create it
					} // create new city?
				}
				
				ai_state.common_actions(ai_ind, is_cur_player, pstats, personality, units, bldgs, gstate, map_data, exs, disband_unit_inds, map_sz, temps);
			}
		}
	} // plan unit actions
	
	pub fn accept_peace_treaty(&self, opposition_id: usize, cur_player_id: usize, gold_offering: f32, gstate: &mut GameState, players: &Vec<Player>) -> bool {
		const ADVANTAGE_TO_GOLD_FACTOR: f32 = 10.;
		const MIN_WAR_DURATION_DAYS: usize = 30;
		
		if (gstate.turn - gstate.relations.turn_war_started(cur_player_id, opposition_id)) < MIN_WAR_DURATION_DAYS {return false;}
		
		let friendliness = gstate.relations.friendliness_toward(opposition_id, cur_player_id, players);
		
		if let Some(war_advantage) = current_war_advantage(opposition_id, players, &gstate.relations) {
			let war_advantage = war_advantage as f32;
			
			// at an advantage
			if war_advantage > 0. {
				// player is requesting gold	
				if gold_offering < 0. {return false;}
				
				// player is sending gold
				debug_printlnq!("ai would accept: {}, friendliness {}", war_advantage*ADVANTAGE_TO_GOLD_FACTOR*(2.-friendliness), friendliness);
				
				return gold_offering > war_advantage*ADVANTAGE_TO_GOLD_FACTOR*(2.-friendliness);
				// ^ negative moods raise the price the AI will accept
			// at a disadvantage
			}else{
				// player is sending gold
				if gold_offering > 0. {return true;}
				
				// player is requesting gold
				debug_printlnq!("ai would send: {}, friendliness {}", war_advantage.abs()*ADVANTAGE_TO_GOLD_FACTOR*(2.+friendliness), friendliness);
				
				return gold_offering.abs() < war_advantage.abs()*ADVANTAGE_TO_GOLD_FACTOR*(2.+friendliness);
				// ^ negative moods lower the price the AI is willing to send
			}
		}else{
			debug_printlnq!("could not estimate war advantage");
			
			return false;
		}
	}
}

