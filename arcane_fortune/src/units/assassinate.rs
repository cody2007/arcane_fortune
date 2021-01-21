use super::*;
use crate::gcore::*;
use crate::ai::collapse_house_due_to_no_successor;

const CHANCE_ASSASSINATION_FAILS: f32 = 0.5;
pub fn do_assassinate_action<'bt,'ut,'rt,'dt>(unit_ind: usize, attack_coord: Option<u64>,
		disband_unit_inds: &mut Vec<usize>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>,
		bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>, gstate: &mut GameState,
		disp: &mut Disp, temps: &Templates<'bt,'ut,'rt,'dt,'_>) {
	let u = &mut units[unit_ind];
	u.action.pop();
	if let Some(attack_coord) = attack_coord {
		if let Some(MapEx {bldg_ind: Some(bldg_ind), ..}) = exs.last().unwrap().get(&attack_coord) {
			let b = &bldgs[*bldg_ind as usize];
			let attackee = b.owner_id as usize;
			let attacker = units[unit_ind].owner_id as usize;
			
			if attackee == attacker {return;}
			
			// assassination fails
			if gstate.rng.gen_f32b() < CHANCE_ASSASSINATION_FAILS || gstate.relations.at_war(attackee, attacker) {
				assassination_failed(attacker, attackee, unit_ind, units, exs, map_data, gstate, &mut DelAction::Record(disband_unit_inds), players, disp);
				return;
			}
			
			// city hall
			if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
				// log
				gstate.log_event(LogType::LeaderAssassinated {
					owner_id: attackee,
					city_nm: nm.clone()
				});
				
				// remove head noble
				if let Some(house) = players[attackee].ptype.house_mut() {
					house.head_noble_mut().die(&disp.state.local);
					// cannot set a new successor
					if !house.set_next_successor() {
						collapse_house_due_to_no_successor(attackee, disband_unit_inds, players, units, bldgs, exs, map_data, gstate, temps, disp);
						return;
					}
					
					players[attackee].personalization.ruler_nm = house.head_noble().name.clone();
				// transfer city over to barbarians
				}else{
					let barbarian_ind = get_alive_barbarian_ind(players);
					transfer_population_center_ownership(barbarian_ind, attackee, attacker, *bldg_ind, disband_unit_inds, bldgs, units, players, map_data, exs, temps, gstate, disp);
				}
				
			// barbarian camp
			}else if b.template.nm[0] == BARBARIAN_CAMP_NM {
				destroy_barbarian_camp(unit_ind, *bldg_ind, disband_unit_inds, disp.state.iface_settings.cur_player as usize, players, bldgs, units, exs, map_data, gstate, temps);
			}
		}
	}
}

// ---- called from mv_unit() at each land tile traversed over
// check if wall scaling has been detected, declare war if so
// otherwise, discover land
// `disp` should be supplied if the movement is WallScaling (needed to set disp.ui_mode if war is declared)
// returns true if wall scaling is detected, false if not
const CHANCE_WALL_SCALING_DISCOVERED: f32 = 0.5;
pub fn is_wall_scale_detected_else_discover_land<'bt,'ut,'rt,'dt>(coord: u64, unit_ind: usize,
		action_type: &ActionType, del_action: &mut DelAction,
		is_cur_player: bool, moving_player_owner_id: usize, map_data: &mut MapData,
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, disp_opt: &mut Option<&mut Disp>) -> bool {
	// wall scaling detected?
	let wall_scaling_detected_owner = (|| {
		if let ActionType::ScaleWalls = &action_type {
			if let Some(ex) = exs.last().unwrap().get(&coord) {
				if let Some(StructureData {structure_type: StructureType::Wall, ..}) = ex.actual.structure {
					let wall_owner_id = ex.actual.owner_id.unwrap() as usize;
					
					// the player is allowed to scale its own walls
					if wall_owner_id == moving_player_owner_id {return None;}
					
					// if at war, scaling is always detected
					if gstate.relations.at_war(moving_player_owner_id, wall_owner_id) || // if at war, scaling is always detected
					   gstate.rng.gen_f32b() < CHANCE_WALL_SCALING_DISCOVERED { // if not at war, randomly detect
						return Some(wall_owner_id);
					}
				}
			}
		}
		None
	})();
	
	// wall scaling detected
	if let Some(wall_scaling_detected_owner) = wall_scaling_detected_owner {
		let disp = &mut disp_opt.as_mut().unwrap();
		assassination_failed(moving_player_owner_id, wall_scaling_detected_owner, unit_ind, units, exs, map_data, gstate, del_action, players, disp);
		true
	// discover land
	}else{
		compute_active_window(coord, is_cur_player, PresenceAction::DiscoverOnly, map_data, exs, &mut players[moving_player_owner_id].stats, *map_data.map_szs.last().unwrap(), gstate, units);
		false
	}
}

fn assassination_failed<'bt,'ut,'rt,'dt>(assassinater_owner_id: usize, assassinatee_owner_id: usize, unit_ind: usize,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData,
		gstate: &mut GameState, del_action: &mut DelAction,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, disp: &mut Disp) {
	let is_cur_player = assassinater_owner_id == disp.state.iface_settings.cur_player as usize;
	let map_sz = *map_data.map_szs.last().unwrap();
	
	gstate.relations.add_mood_factor(assassinater_owner_id, assassinatee_owner_id, MoodType::FailedAssassination, gstate.turn);
	gstate.relations.declare_war(assassinatee_owner_id, assassinater_owner_id, &mut gstate.logs, players, gstate.turn, &mut gstate.rng, disp);
	del_action.execute(unit_ind, is_cur_player, units, map_data, exs, players, gstate, map_sz); // disband unit
	disp.create_alert_window(disp.state.local.Assassination_caught_message.clone());
}

fn get_alive_barbarian_ind(players: &Vec<Player>) -> usize {
	players.iter().find(|p| p.stats.alive && p.ptype.is_barbarian()).unwrap().id as usize
}

