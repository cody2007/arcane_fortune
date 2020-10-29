use super::*;
use crate::map::{MapSz};
use crate::renderer::*;

const DELAY_TURNS_BEFORE_JOIN_REQ: usize = 8 * TURNS_PER_YEAR;

const MARRIAGE_PROB_PER_TURN: f32 = 1./(TURNS_PER_YEAR as f32*10.);
const CHILD_PROB_PER_TURN: f32 = 1./(TURNS_PER_YEAR as f32*10.);

// req to join empire, nobility management
impl House {
	pub fn plan_actions<'bt,'ut,'rt,'dt>(player_ind: usize, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz, 
			nms: &Nms, disp: &mut Disp<'_, 'bt,'ut,'rt,'dt>) {
		// join empire?
		let player = &players[player_ind];
		if gstate.turn > (player.personalization.founded + DELAY_TURNS_BEFORE_JOIN_REQ) {
			if let Some(
				NobilityState {
					house: House {
						has_req_to_join: false, ..
					},
					ai_state
				}
			) = player.ptype.nobility_state() {
				if let Some(city_state) = ai_state.city_states.first() {
					if let Some(empire_ind) = city_state.nearby_empire_ind(MAX_NOBILITY_CITY_DIST, players, map_sz) {
						let empire = &players[empire_ind];
						
						// join automatically
						if empire.ptype.is_empire() {
							gstate.relations.join_as_fiefdom(player_ind, empire_ind, players, &mut gstate.logs, gstate.turn);
							
						// ask to join human player
						}else if empire.ptype.is_human() {
							disp.create_interrupt_window(UIMode::AcceptNobilityIntoEmpire(AcceptNobilityIntoEmpireState {
								mode: 0,
								house_ind: player_ind
							}));
						}
						
						// set flag
						if let Some(house) = players[player_ind].ptype.house_mut() {
							house.has_req_to_join = true;
						}else{panicq!("invalid player state");}
					}
				}
			}
		}
		
		// add children, marriages
		let player = &mut players[player_ind];
		if let Some(house) = player.ptype.house_mut() {
			// add children and marriages
			let mut child_ind = house.noble_pairs.len()-1;
			for noble_pair in house.noble_pairs.iter_mut() {
				// add new child to marriage?
				if let Some(marriage) = &mut noble_pair.marriage {
					if gstate.rng.gen_f32b() > CHILD_PROB_PER_TURN {continue;}
					
					child_ind += 1;
					marriage.children.push(child_ind);
					
				// add marriage?
				}else{
					if gstate.rng.gen_f32b() > MARRIAGE_PROB_PER_TURN {continue;}
					noble_pair.marriage = Marriage::new(&noble_pair.noble, nms, gstate);
				}
			}
			
			// add children to noble_pairs
			for _ in (house.noble_pairs.len()-1)..child_ind {
				house.noble_pairs.push(NoblePair {
					noble: Noble::new(nms, 0, None, gstate),
					marriage: None
				});
			}
		}
	}
}

impl <'bt,'ut,'rt,'dt>NobilityState<'bt,'ut,'rt,'dt> {
	pub fn plan_actions(player_ind: usize, is_cur_player: bool, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, disband_unit_inds: &mut Vec<usize>, gstate: &mut GameState, map_sz: MapSz, 
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp<'_, 'bt,'ut,'rt,'dt>) {
		House::plan_actions(player_ind, players, gstate, map_sz, &temps.nms, disp);
		
		/////////////////////////// AIState common actions
		let player = &mut players[player_ind];
		match &mut player.ptype {
			PlayerType::Nobility(NobilityState {house, ai_state}) => {
				ai_state.common_actions(player_ind, is_cur_player, &mut player.stats, &house.head_personality(), units, bldgs, gstate, map_data, exs, disband_unit_inds, map_sz, temps);
			}
			PlayerType::Empire(_) | PlayerType::Human(_) | PlayerType::Barbarian(_) => {}
		}
	}
}

