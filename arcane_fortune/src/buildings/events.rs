use super::*;
use crate::movement::find_square_buildable;
use crate::disp::*;

const EVENT_DURATION_TURNS: usize = 1*TURNS_PER_YEAR;

enum_From!{ PublicEventType {Marriage, Birth, Funeral} }

impl PublicEventType {
	// `event_type` is text to be shown in the game log
	pub fn create_near<'bt,'ut,'rt,'dt>(&self, coord: u64, event_type: String,
			owner_id: SmSvType, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, 
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, disp: &mut Disp) {
		let bt = BldgTemplate::frm_str(PUBLIC_EVENT_NM, temps.bldgs);
		let map_sz = *map_data.map_szs.last().unwrap();
		if let Some(coord_add) = find_square_buildable(Coord::frm_ind(coord, map_sz), bt, map_data, exs.last().unwrap(), map_sz) {
			if add_bldg(coord_add, owner_id, bldgs, bt, None, temps, map_data, exs, players, gstate) {
				// get city name at `coord`
				if let Some(ex) = exs.last().unwrap().get(&coord) {
					if let Some(bldg_ind) = ex.bldg_ind {
						let b = &mut bldgs[bldg_ind];
						if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
							// log
							gstate.log_event(LogType::GenericEvent {
								owner_id: owner_id as usize,
								location: nm.clone(),
								event_type: event_type.clone()
							});
							
							// set bldg args of event bldg
							let event_bldg = bldgs.last_mut().unwrap();
							event_bldg.construction_done = None;
							event_bldg.args = BldgArgs::PublicEvent {
								public_event_type: *self,
								nm: event_type.clone(),
								turn_created: gstate.turn
							};
							
							// ask owning empire for money?
							if gstate.rng.gen_f32b() < 0.5 {
								if let Some(parent_id) = gstate.relations.fiefdom_of(owner_id as usize) {
									if parent_id == disp.state.iface_settings.cur_player as usize && 
											disp.state.iface_settings.interrupt_auto_turn {
										disp.create_interrupt_window(UIMode::NobilityRequestWindow(
											NobilityRequestWindowState {
												mode: 0,
												owner_id: owner_id as usize,
												nobility_request_type: NobilityRequestType::GoldForEvent(event_type)
											}
										));
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	pub fn happiness_bonus(&self, bldg_config: &BldgConfig) -> f32 {
		match self {
			Self::Marriage => {bldg_config.marriage_celebration_bonus}
			Self::Birth => {bldg_config.birth_celebration_bonus}
			Self::Funeral => {bldg_config.funeral_bonus}
			Self::N => {panic!("invalid PublicEventType");}
		}
	}
}

pub fn rm_old_events<'bt,'ut,'rt,'dt>(cur_player: SmSvType, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		gstate: &mut GameState, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
		map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz) {
	#[cfg(feature="profile")]
	let _g = Guard::new("rm_old_events");
	
	for bldg_ind in (0..bldgs.len()).rev() {
		let b = &bldgs[bldg_ind];
		if let BldgArgs::PublicEvent {turn_created, ..} = b.args {
			// event not old enough to be removed
			if (gstate.turn - turn_created) < EVENT_DURATION_TURNS {continue;}
			
			let is_cur_player = b.owner_id == cur_player;
			rm_bldg(bldg_ind, is_cur_player, bldgs, bldg_templates, map_data, exs, players, UnitDelAction::Delete {units, gstate}, map_sz);
		}
	}	
}

