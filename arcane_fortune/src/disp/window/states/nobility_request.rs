// nobility requests gold, resources, [war declaration, peace, doctrine changes] of parent empire
use crate::disp::*;
use super::*;

pub struct NobilityRequestWindowState<'bt,'ut,'rt,'dt> {
	pub mode: usize, // option selected (accept/decline request)
	pub owner_id: usize, // nobility owner making the request
	pub nobility_request_type: NobilityRequestType<'bt,'ut,'rt,'dt>
}

pub enum NobilityRequestType<'bt,'ut,'rt,'dt> {
	GoldForEvent(String), // string is the text describing the event
	
	Resource(&'rt ResourceTemplate),
	
	DeclareWarAgainst(usize), // owner_id to declare war against
	DeclarePeaceAgainst(usize), // "
	
	BuildDoctrineBldg(DoctrineBldg<'bt,'ut,'rt,'dt>),
	BuildScienceBldg(ScienceBldg<'bt,'ut,'rt,'dt>)
}

pub struct DoctrineBldg<'bt,'ut,'rt,'dt> {
	bldg_template: &'bt BldgTemplate<'ut,'rt,'dt>,
	doctrine: &'dt DoctrineTemplate,
	coord: u64
}

pub struct ScienceBldg<'bt,'ut,'rt,'dt> {
	bldg_template: &'bt BldgTemplate<'ut,'rt,'dt>,
	coord: u64
}

impl Stats<'_,'_,'_,'_> {
	// amount fiefdoms request from main empire. self is the main empire
	fn request_event_gold(&self) -> f32 {
		let mut gold_req = self.gold*0.25;
		const MIN_GOLD: f32 = 10_000.;
		if gold_req < MIN_GOLD {
			gold_req = MIN_GOLD;
		}
		
		if self.gold < gold_req {
			gold_req = self.gold;
		}
		
		gold_req.floor()
	}
}

// find closest city (of the current player) to the noble house
fn find_nearest_city_loc(noble_ai_state: &AIState, bldgs: &Vec<Bldg>, map_sz: MapSz, dstate: &DispState) -> Coord {
	let noble_house_coord = Coord::frm_ind(noble_ai_state.city_states[0].coord, map_sz);
	let coord_ind = bldgs.iter()
		// cur_player owns building & it's a population center
		.filter(|b| {
			if b.owner_id == dstate.iface_settings.cur_player {
				if let BldgArgs::PopulationCenter {..} = &b.args {
					return true;
				}
			}
			false
		})
		// find closest population center to the nobility's manor
		.min_by_key(|b| manhattan_dist(Coord::frm_ind(b.coord, map_sz), noble_house_coord, map_sz))
		
		.unwrap().coord;
	
	Coord::frm_ind(coord_ind, map_sz)
}

use crate::movement::find_square_buildable;
use crate::ai::BonusBldg;
impl <'bt,'ut,'rt,'dt>NobilityRequestWindowState<'bt,'ut,'rt,'dt> {
	// use new_doctrine_bldg_request & new_science_bldg_request() for those request types
	pub fn new(nobility_owner_id: usize, nobility_request_type: NobilityRequestType<'bt,'ut,'rt,'dt>) -> Self {
		Self {
			mode: 0,
			owner_id: nobility_owner_id,
			nobility_request_type
		}
	}
	
	// assumes `nobility_owner_id` is a fiefdom of the current iface_settings player
	pub fn new_doctrine_bldg_request(nobility_owner_id: usize, players: &Vec<Player<'bt,'ut,'rt,'dt>>, 
			bldgs: &Vec<Bldg>, temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData,
			exf: &HashedMapEx, rng: &mut XorState, map_sz: MapSz, dstate: &DispState) -> Option<Self> {
		
		let nobility_player = &players[nobility_owner_id];
		let noble_ai_state = nobility_player.ptype.any_ai_state().unwrap();
		
		// the nobility's goal doctrine
		if let Some(goal_doctrine) = noble_ai_state.goal_doctrine {
			let player_cur_stats = &players[dstate.iface_settings.cur_player as usize].stats;
			
			// closest doctrine the current player can create buildings for
			let intermediate_doctrine = goal_doctrine.closest_available(player_cur_stats, &temps.doctrines);
			
			// choose doctrine bldg to build
			let bldg_template = {
				let bonus_bldgs = BonusBldg::new(temps.bldgs, player_cur_stats).doctrine_bldgs;
				bonus_bldgs[rng.usize_range(0, bonus_bldgs.len())]
			};
			
			// closest city to the noble house (near where the building will be built)
			let city_coord = find_nearest_city_loc(noble_ai_state, bldgs, map_sz, dstate);
			
			// find place near the city coord to build the building
			if let Some(coord) = find_square_buildable(city_coord, bldg_template, map_data, exf, map_sz) {
				return Some(Self {
					mode: 0,
					owner_id: nobility_owner_id,
					nobility_request_type: NobilityRequestType::BuildDoctrineBldg(
						DoctrineBldg {
							bldg_template,
							doctrine: intermediate_doctrine,
							coord
						}
					)
				});
			}
		}
		None
	}
	
	pub fn new_science_bldg_request(nobility_owner_id: usize, players: &Vec<Player<'bt,'ut,'rt,'dt>>, 
			bldgs: &Vec<Bldg>, temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData,
			exf: &HashedMapEx, rng: &mut XorState, map_sz: MapSz, dstate: &DispState) -> Option<Self> {
		let noble_ai_state = players[nobility_owner_id].ptype.any_ai_state().unwrap();
		let player_cur_stats = &players[dstate.iface_settings.cur_player as usize].stats;
		
		// choose science bldg to build
		let bldg_template = {
			let bonus_bldgs = BonusBldg::new(temps.bldgs, player_cur_stats).scientific_bldgs;
			if bonus_bldgs.len() == 0 {
				return None; // ?
			}
			bonus_bldgs[rng.usize_range(0, bonus_bldgs.len())]
		};
		
		// closest city to the noble house (near where the building will be built)
		let city_coord = find_nearest_city_loc(noble_ai_state, bldgs, map_sz, dstate);
		
		// find place near the city coord to build the building
		if let Some(coord) = find_square_buildable(city_coord, bldg_template, map_data, exf, map_sz) {
			return Some(Self {
				mode: 0,
				owner_id: nobility_owner_id,
				nobility_request_type: NobilityRequestType::BuildScienceBldg(
					ScienceBldg {
						bldg_template,
						coord
					}
				)
			});
		}
		None
	}
	
	pub fn print(&self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let nobility_player = &players[self.owner_id];
		
		let intro_txt = { // "[noble_nm] from the House of [house_nm] greets you with the following request:"
			dstate.local.Nobility_request
				.replace("[noble_nm]", &nobility_player.personalization.ruler_nm.txt())
				.replace("[house_nm]", &nobility_player.personalization.nm)
		};
		
		let request_txt = {
			match &self.nobility_request_type {
				NobilityRequestType::GoldForEvent(event_type) => {
					let gold_request = players[dstate.iface_settings.cur_player as usize].stats.request_event_gold();
				
					// "Our house is having a [event_txt] and we could use financial help in hosting it. Could you spare us [gold] gold?"
					
					dstate.local.Nobility_request_money
						.replace("[event_txt]", event_type)
						.replace("[gold]", &format!("{}", gold_request as usize))
				} NobilityRequestType::Resource(resource) => {
					// "Our House's advisors have counseled us that our fiefdom would likely prosper with access to [resource]. Given the prosperity of your empire, could you spare us access to your [resource]? We doubt you'd even notice our usage."
					
					dstate.local.Nobility_request_resource
						.replace("[resource]", &resource.nm[dstate.local.lang_ind])
				} NobilityRequestType::DeclareWarAgainst(owner_id) => {
					// "We are starting to become embarrassed to call ourselves part of your empire. You seem weak on the world stage and this is something our House finds disgraceful. With all the deceit and treachery surrounding us, what better time to start a war? How about we attack the [civ_nm]? The world would be better off without them." 
					
					dstate.local.Nobility_request_war
						.replace("[civ_nm]", &players[*owner_id].personalization.nm)
				} NobilityRequestType::DeclarePeaceAgainst(owner_id) => {
					// "If we wanted to be associated with a tyrannical steam-roller of free states, we would've joined ourselves with local barbarians. It's time to put violence aside and declare peace against the [civ_nm]. Luckily for you, we have already taken it upon ourselves to contact the [civ_nm]. They agreed with our sentiments and would be happy to call off the war without imposing any terms and conditions. What do you say?"
					
					dstate.local.Nobility_request_peace
						.replace("[civ_nm]", &players[*owner_id].personalization.nm)
				} NobilityRequestType::BuildDoctrineBldg(doctrine_bldg) => {
					// "It seems that your leadership lacks scope and perspective. Both of which would be provided if you strived to incorporate more [doctrine] throughout the empire. We'd like to help you along this journey. Some members of our House have even been so kind to offer assigning their servants to the task of building a [bldg] dedicated to [doctrine_intermediate] in one of your cities. We trust that you will have no problem paying the maintence of this building. What do you say, would you like to follow us on this great path toward [doctrine]?"
					
					// the nobility's goal doctrine
					if let Some(goal_doctrine) = nobility_player.ptype.any_ai_state().unwrap().goal_doctrine {
						dstate.local.Nobility_build_doctrine_bldg
							.replace("[doctrine]", &goal_doctrine.nm[dstate.local.lang_ind])
							.replace("[doctrine_intermediate]", &doctrine_bldg.doctrine.nm[dstate.local.lang_ind])
							.replace("[bldg]", &doctrine_bldg.bldg_template.nm[dstate.local.lang_ind])
					}else{
						return UIModeControl::Closed;
					}
				} NobilityRequestType::BuildScienceBldg(science_bldg) => {
					// "There are great wonders beyond this world that are waiting to be discovered. Wonders that will unlock new levels of prosperity and control over our surroundings. Scholars from our House would like to support the advancement of this empire and, with your approval, will see to it that a [bldg] is built in one of your cities. We trust that your empire will be able to afford the salaries and building upkeep of the [bldg]."
					
					dstate.local.Nobility_build_science_bldg
						.replace("[bldg]", &science_bldg.bldg_template.nm[dstate.local.lang_ind])
				}
			}
		};
		
		let w = 70;
		
		let intro_txt_wrapped = wrap_txt(&intro_txt, w as usize - 4);
		let request_txt_wrapped = wrap_txt(&request_txt, w as usize - 4);
		
		let h = 7 + intro_txt_wrapped.len() + request_txt_wrapped.len();
		let w_pos = dstate.print_window(ScreenSz {w, h, sz:0});
		
		let mut row = w_pos.y + 1;
		
		// print: 
		// "[noble_nm] from the House of phouse_nm] greets you with the following request:"
		for intro_txt in intro_txt_wrapped { // for each line
			dstate.mv(row, w_pos.x + 2); row += 1;
			dstate.addstr(&intro_txt);
		}
		row += 1;
		
		// print request (each line)
		for request_txt in request_txt_wrapped {
			dstate.mv(row, w_pos.x + 2); row += 1;
			dstate.addstr(request_txt);
		}
		
		let mut screen_reader_cur_loc = (0,0);
		{ // options
			row += 1;
			let screen_w = dstate.iface_settings.screen_sz.w as i32;
			
			dstate.buttons.It_would_be_my_pleasure.print_centered_selection(row as i32, self.mode == 0, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer); row += 1;
			dstate.buttons.Not_a_chance.print_centered_selection(row as i32, self.mode == 1, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
		}
		
		dstate.renderer.mv(screen_reader_cur_loc.0, screen_reader_cur_loc.1);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			gstate: &mut GameState, temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData<'rt>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(mode) = button_mode_update_and_action(&mut self.mode, vec![
					&mut dstate.buttons.It_would_be_my_pleasure,
					&mut dstate.buttons.Not_a_chance
				], dstate.key_pressed, &dstate.mouse_event, &dstate.kbd) {
			let cur_player = dstate.iface_settings.cur_player as usize;
			
			return UIModeControl::New(UIMode::GenericAlert(GenericAlertState {
				txt: match mode {
				// accept
				0 => {
					match &self.nobility_request_type {
						NobilityRequestType::GoldForEvent(_) => {
							let pstats = &mut players[cur_player].stats;
							let gold_req = pstats.request_event_gold();
							pstats.gold -= gold_req;
							players[self.owner_id].stats.gold += gold_req;
						} NobilityRequestType::Resource(resource) => {
							let mut trade_deal = TradeDeal::new(gstate.turn);
							trade_deal.add_item(TradeItem::Resource(resource.id as usize), cur_player, self.owner_id);
							gstate.relations.add_trade(&trade_deal, cur_player, self.owner_id, players, temps, map_data, dstate, gstate.turn);
						} NobilityRequestType::DeclareWarAgainst(owner_id) => {
							gstate.relations.declare_war_ret_ui_mode(cur_player, *owner_id, &mut gstate.logs, players, gstate.turn, None, &mut gstate.rng, dstate);
						} NobilityRequestType::DeclarePeaceAgainst(owner_id) => {
							gstate.relations.declare_peace(cur_player, *owner_id, &mut gstate.logs, gstate.turn);
						} NobilityRequestType::BuildDoctrineBldg(doctrine_bldg) => {
							add_bldg(doctrine_bldg.coord, cur_player as SmSvType, bldgs, doctrine_bldg.bldg_template,
									Some(doctrine_bldg.doctrine), None, temps, map_data, exs, players, gstate);
							bldgs.last_mut().unwrap().construction_done = None;
						} NobilityRequestType::BuildScienceBldg(science_bldg) => {
							add_bldg(science_bldg.coord, cur_player as SmSvType, bldgs, science_bldg.bldg_template,
									None, None, temps, map_data, exs, players, gstate);
							bldgs.last_mut().unwrap().construction_done = None;
						}
					}
					
					gstate.relations.add_mood_factor(cur_player, self.owner_id, MoodType::YouHelpedUs, gstate.turn);
					
					// show message to player
					dstate.local.Nobility_request_accepted.replace("[house_nm]", &players[self.owner_id].personalization.nm)
					
				// decline
				} 1 => {
					gstate.relations.add_mood_factor(cur_player, self.owner_id, MoodType::YouRefusedToHelpUs, gstate.turn);
					
					// show message to player
					dstate.local.Nobility_request_declined.replace("[house_nm]", &players[self.owner_id].personalization.nm)
				} _ => {panic!("invalid return");}
			}}));
		}
		
		UIModeControl::UnChgd
	}
}
