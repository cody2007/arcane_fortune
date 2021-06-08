use super::*;
use crate::map::{MapSz};
use crate::renderer::*;

const DELAY_TURNS_BEFORE_JOIN_REQ: usize = 8 * TURNS_PER_YEAR;

const MARRIAGE_PROB_PER_TURN: f32 = 1./(TURNS_PER_YEAR as f32*10.);
const CHILD_PROB_PER_TURN: f32 = 1./(TURNS_PER_YEAR as f32*10.);
const DEATH_PROB_PER_YEAR_PER_TURN: f32 = 0.9/(110. * TURNS_PER_YEAR as f32);

const REQ_RESOURCE_FROM_PARENT_PROB_PER_TURN: f32 = 1./(2.*19. * TURNS_PER_YEAR as f32);
const REQ_WAR_PEACE_FROM_PARENT_PROB_PER_TURN: f32 = 1./(2.*19. * TURNS_PER_YEAR as f32);
const REQ_BLDG_FROM_PARENT_PROB_PER_TURN: f32 = 1./(2.*19. * TURNS_PER_YEAR as f32);

const INDEPENDENCE_DECLARATION_PROB_PER_TURN: f32 = 1./(2.*8. * TURNS_PER_YEAR as f32);
const STEAL_GOLD_PROB_PER_TURN: f32 = 1./(3. * TURNS_PER_YEAR as f32);
const STEAL_GOLD_FRAC: f32 = 0.8;

// req to join empire, nobility management (births, deaths, marriages)
impl House {
	pub fn plan_actions<'bt,'ut,'rt,'dt>(player_ind: usize, disband_unit_inds: &mut Vec<usize>,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
			bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp<'_,'_, 'bt,'ut,'rt,'dt>) {
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
							gstate.relations.join_as_fiefdom(player_ind, empire_ind, players, &mut gstate.logs, gstate.turn, &mut gstate.rng, &mut disp.state);
							
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
		
		// add children, marriages, deaths
		let player = &mut players[player_ind];
		
		if let Some(CityState {coord, ..}) = player.ptype.any_ai_state().unwrap().city_states.first() {
			let house_coord = *coord;
			if let Some(house) = player.ptype.house_mut() {
				// add children and marriages
				let mut child_ind = house.noble_pairs.len()-1;
				let mut marriage_event_types = Vec::new();
				for noble_pair in house.noble_pairs.iter_mut().filter(|noble_pair| noble_pair.noble.alive) {
					// add new child to marriage?
					if let Some(marriage) = &mut noble_pair.marriage {
						if gstate.rng.gen_f32b() > CHILD_PROB_PER_TURN {continue;}
						
						child_ind += 1;
						marriage.children.push(child_ind);
						
					// add marriage?
					}else if noble_pair.marriage.is_none() {
						if gstate.rng.gen_f32b() > MARRIAGE_PROB_PER_TURN {continue;}
						noble_pair.marriage = Marriage::new(&noble_pair.noble, &temps.nms, gstate);
						
						// add event?
						if let Some(marriage) = &noble_pair.marriage {
							// to be added later because adding events reqs a mutable borrow of players
							marriage_event_types.push(disp.state.local.wedding_ceremony
								.replace("[name1]", &noble_pair.noble.name.txt())
								.replace("[name2]", &marriage.partner.name.txt())
							);
						}
					}
				}
				
				let children = { // create children and add events
					let head_last_name = house.head_noble().name.last.clone();
					let n_children_add = 1 + child_ind - house.noble_pairs.len();
					let mut children = Vec::with_capacity(n_children_add);
					for _ in 0..n_children_add {
						let child = NoblePair {
							noble: Noble::new(&temps.nms, 0, Some(&head_last_name), None, gstate),
							marriage: None
						};
						
						{ // add event
							let event_type = disp.state.local.ceremony_celebrating_the_birth_of_X
								.replace("[name]", &child.noble.name.first)
								.replace("[house_nm]", &players[player_ind].personalization.nm);
							PublicEventType::Birth.create_near(house_coord, event_type, player_ind as SmSvType, bldgs, temps, map_data, exs, players, gstate, disp);
						}
						
						children.push(child);
					}
					children
				};
				
				// add children to noble_pairs
				// must be done separately because adding events requires mutable borrowing of players
				if let Some(house) = players[player_ind].ptype.house_mut() {
					for child in children {
						house.noble_pairs.push(child);
					}
				}
				
				{ // add marriage events
					for event_type in marriage_event_types {
						PublicEventType::Marriage.create_near(house_coord, event_type, player_ind as SmSvType, bldgs, temps, map_data, exs, players, gstate, disp);
					}
				}
			}else{return;}
			
			// deaths
			if let Some(house) = players[player_ind].ptype.house_mut() {
				let mut death_event_types = Vec::new();
				for noble_pair in house.noble_pairs.iter_mut().filter(|noble_pair| noble_pair.noble.alive) {
					if gstate.rng.gen_f32b() > DEATH_PROB_PER_YEAR_PER_TURN {continue;}
					
					// partner dies
					if let Some(marriage) = &mut noble_pair.marriage {
						if gstate.rng.gen_f32b() > 0.5 {
							death_event_types.push(marriage.partner.die(&disp.state.local));
							continue;
						}
					}
					
					// main noble dies (descended from the noble family)
					death_event_types.push(noble_pair.noble.die(&disp.state.local));
				}
				
				// if the head noble has died, set a successor,
				// if there is no successor the house collapses
				if !house.head_noble().alive {
					let prev_head_noble_nm = house.head_noble().name.clone();
					
					if !house.set_next_successor() {
						collapse_house_due_to_no_successor(player_ind, disband_unit_inds, players, units, bldgs, exs, map_data, gstate, temps, disp);
						return;
					}
					
					let new_head_noble_nm = house.head_noble().name.clone();
					
					// notify player that the head noble has changed
					if let Some(parent_empire_ind) = gstate.relations.fiefdom_of(player_ind) {
						if parent_empire_ind as SmSvType == disp.state.iface_settings.cur_player {
							disp.create_interrupt_window(UIMode::GenericAlert(GenericAlertState {
								txt: disp.state.local.Noble_head_changed
									.replace("[old_name]", &prev_head_noble_nm.txt())
									.replace("[house_name]", &prev_head_noble_nm.last)
									.replace("[new_name]", &new_head_noble_nm.txt())
							}));
						}
					}
					
					// update personalization.ruler_nm
					players[player_ind].personalization.ruler_nm = new_head_noble_nm;
				}
				
				{ // add death events
					for event_type in death_event_types {
						PublicEventType::Funeral.create_near(house_coord, event_type, player_ind as SmSvType, bldgs, temps, map_data, exs, players, gstate, disp);
					}
				}
			}
		}
	}
}

impl <'bt,'ut,'rt,'dt>NobilityState<'bt,'ut,'rt,'dt> {
	pub fn plan_actions(player_ind: usize, is_cur_player: bool, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, disband_unit_inds: &mut Vec<usize>, gstate: &mut GameState, map_sz: MapSz, 
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp<'_,'_, 'bt,'ut,'rt,'dt>) {
		#[cfg(feature="profile")]
		let _g = Guard::new("NobilityState::plan_actions");
		
		// AI common actions & request something from the parent empire?
		// function returns if the player is not a noble
		let player = &mut players[player_ind];
		match &mut player.ptype {
			PlayerType::Nobility(NobilityState {house, ..}) => {
				let personality = house.head_personality();
				let cur_player = disp.state.iface_settings.cur_player as usize;
				let target_city_coord = house.target_city_coord;
				if gstate.relations.fiefdom(player_ind, cur_player) {
					// request resource
					if gstate.rng.gen_f32b() < REQ_RESOURCE_FROM_PARENT_PROB_PER_TURN {
						// from the fiefdom's parent empire's resources, which would the nobility want the most?
						let resource_req = {
							let mut resource_req: Option<&ResourceTemplate> = None;
							for (_resource_avail, resource) in players[cur_player].stats.resources_avail.iter()
									.zip(temps.resources)
									.filter(|(&resource_avail, _)| resource_avail != 0) {
								if resource_req.is_none() || resource_req.unwrap().ai_valuation < resource.ai_valuation {
									resource_req = Some(resource);
								}
							}
							
							resource_req
						};
						
						if let Some(resource_req) = resource_req {
							disp.create_interrupt_window(UIMode::NobilityRequestWindow(
								NobilityRequestWindowState::new(player_ind, NobilityRequestType::Resource(resource_req))
							));
						}
						
					// request war/peace
					} else if gstate.rng.gen_f32b() < REQ_WAR_PEACE_FROM_PARENT_PROB_PER_TURN {
						if let Some(request_type) = {
							// request war
							if house.head_personality().friendliness < 0. {
								NobilityRequestType::preferred_war_target(player_ind, cur_player, players, gstate)
							// request peace
							}else{
								NobilityRequestType::preferred_peace_target(player_ind, cur_player, players, gstate)
							}
						} {
							disp.create_interrupt_window(UIMode::NobilityRequestWindow(
								NobilityRequestWindowState::new(player_ind, request_type)
							));
						}
					// request building
					} else if gstate.rng.gen_f32b() < REQ_BLDG_FROM_PARENT_PROB_PER_TURN {
						if let Some(request_window_state) = {
							let exf = exs.last().unwrap();
							
							// request scientific bldg
							if house.head_personality().spirituality < 0. {
								NobilityRequestWindowState::new_science_bldg_request(player_ind, players, bldgs, temps, map_data, exf, &mut gstate.rng, map_sz, &disp.state)
							// request doctrine bldg
							}else{
								NobilityRequestWindowState::new_doctrine_bldg_request(player_ind, players, bldgs, temps, map_data, exf, &mut gstate.rng, map_sz, &disp.state)
							}
						} {
							disp.create_interrupt_window(UIMode::NobilityRequestWindow(request_window_state));
						}
					}
				}
				
				AIState::common_actions(player_ind, is_cur_player, target_city_coord, players, personality, units, bldgs, gstate, map_data, exs, disband_unit_inds, map_sz, temps);
			}
			PlayerType::Empire(_) | PlayerType::Human(_) | PlayerType::Barbarian(_) => {return;}
		}
		
		House::plan_actions(player_ind, disband_unit_inds, players, gstate, map_sz, units, bldgs, map_data, exs, temps, disp);
		gstate.house_steal_gold_if_unhappy(player_ind, players, disp);
		gstate.house_declare_independence_if_unhappy(player_ind, players, disp);
	}
}

use crate::nn::TxtCategory;
// rogue nobility action
impl GameState {
	fn house_declare_independence_if_unhappy(&mut self, house_id: usize, players: &mut Vec<Player>, disp: &mut Disp) {
		if let Some(empire_id) = self.relations.fiefdom_of(house_id) {
			match TxtCategory::from_relations(&self.relations, house_id, empire_id, players) {
				TxtCategory::Negative => {
					if self.rng.gen_f32b() > INDEPENDENCE_DECLARATION_PROB_PER_TURN {return;}
					
					self.log_event(LogType::HouseDeclaresIndependence {house_id, empire_id});
					self.relations.declare_peace_wo_logging(house_id, empire_id, self.turn);
					players[house_id].personalization.color = NOBILITY_COLOR;
					
					if empire_id as SmSvType == disp.state.iface_settings.cur_player {
						disp.create_interrupt_window(UIMode::NobilityDeclaresIndependenceWindow(
							NobilityDeclaresIndependenceWindowState {mode: 0, owner_id: house_id}
						));
					}
				}
				TxtCategory::Neutral | TxtCategory::Positive => {}
			}
		}
	}
	
	fn house_steal_gold_if_unhappy(&mut self, house_id: usize, players: &mut Vec<Player>, disp: &mut Disp) {
		if let Some(empire_id) = self.relations.fiefdom_of(house_id) {
			// only do this if the empire is the human player
			if house_id != disp.state.iface_settings.cur_player as usize {return;}
			
			if self.rng.gen_f32b() > STEAL_GOLD_PROB_PER_TURN {return;}
			
			// not yet unhappy enough:
			if self.relations.friendliness_toward(house_id, empire_id, players) > friendliness_thresh(1.5) {return;}
			
			let empire_stats = &mut players[empire_id].stats;
			let steal_gold = empire_stats.gold * STEAL_GOLD_FRAC;
			
			empire_stats.gold -= steal_gold;
			let house = &mut players[house_id];
			house.stats.gold += steal_gold;
			
			let txt = disp.state.local.Nobility_steal_gold
				.replace("[noble_nm]", &house.personalization.ruler_nm.txt())
				.replace("[house_nm]", &house.personalization.nm);
			
			disp.create_interrupt_window(UIMode::GenericAlert(GenericAlertState {txt}));
		}
	}
}

impl NobilityRequestType<'_,'_,'_,'_> {
	// returns the noble's preferred war target
	fn preferred_war_target(noble_owner_id: usize, human_owner_id: usize, players: &Vec<Player>, gstate: &GameState) -> Option<Self> {
		if let Some(war_target_id) = players.iter().enumerate()
			.filter(|(owner_id, player)| 
				*owner_id != human_owner_id && 
				player.stats.alive &&
				gstate.relations.discovered(human_owner_id, *owner_id) && 
				!gstate.relations.fiefdom(human_owner_id, *owner_id) &&
				!gstate.relations.at_war(human_owner_id, *owner_id))
			// select the owner which the noble likes least
			.min_by(|(owner_i, _), (owner_j, _)| {
				let friendliness_i = gstate.relations.friendliness_toward(noble_owner_id, *owner_i, players);
				let friendliness_j = gstate.relations.friendliness_toward(noble_owner_id, *owner_j, players);
				friendliness_i.partial_cmp(&friendliness_j).unwrap_or(Ordering::Equal)
			})
			.map(|(owner_id, _)| owner_id)
		{
			Some(Self::DeclareWarAgainst(war_target_id))
		}else{
			None
		}
	}
	
	fn preferred_peace_target(noble_owner_id: usize, human_owner_id: usize, players: &Vec<Player>, gstate: &GameState) -> Option<Self> {
		if let Some(peace_target_id) = gstate.relations.at_war_with(human_owner_id).iter()
			// select the owner which the noble likes most
			.max_by(|&owner_i, &owner_j| {
				let friendliness_i = gstate.relations.friendliness_toward(noble_owner_id, *owner_i, players);
				let friendliness_j = gstate.relations.friendliness_toward(noble_owner_id, *owner_j, players);
				friendliness_i.partial_cmp(&friendliness_j).unwrap_or(Ordering::Equal)
			})
		{
			Some(Self::DeclarePeaceAgainst(*peace_target_id))
		}else{
			None
		}
	}
}

pub fn collapse_house_due_to_no_successor<'bt,'ut,'rt,'dt>(player_ind: usize, disband_unit_inds: &mut Vec<usize>, 
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, 
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		map_data: &mut MapData<'rt>, gstate: &mut GameState,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp) {
	gstate.log_event(LogType::NoNobleSuccessor {owner_id: player_ind});
	let map_sz = *map_data.map_szs.last().unwrap();
	civ_collapsed(player_ind, &mut Some(disband_unit_inds), players, units, bldgs, map_data, exs, gstate, map_sz, temps, disp);	
}

