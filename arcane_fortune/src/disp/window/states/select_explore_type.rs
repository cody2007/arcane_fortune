use super::*;
use crate::units::*;
pub struct SelectExploreTypeState {pub mode: usize}

impl SelectExploreTypeState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_exploration_type.clone(), explore_types_list(&dstate.local), None, None, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>,
			gstate: &mut GameState,
			map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		if let Some(unit_inds) = dstate.iface_settings.unit_inds_frm_sel(&players[cur_player].stats, units, map_data, exs.last().unwrap()) {
			if explore_types_list(&dstate.local).list_mode_update_and_action(&mut self.mode, dstate) {
				let map_sz = *map_data.map_szs.last().unwrap();
				let explore_type = ExploreType::from(self.mode);
				
				for unit_ind in unit_inds {
					let u = &mut units[unit_ind];
					u.action.pop();
					let land_discov = &players[cur_player].stats.land_discov.last().unwrap();
					
					if let Some(new_action) = explore_type.find_square_unexplored(unit_ind, u.return_coord(), map_data, exs, units, bldgs, land_discov, map_sz, true, &mut gstate.rng) {
						units[unit_ind].action.push(new_action);
						mv_unit(unit_ind, true, units, map_data, exs, bldgs, players, gstate, map_sz, DelAction::Delete, &mut None);
						//dstate.iface_settings.reset_unit_subsel();
						dstate.iface_settings.update_all_player_pieces_mvd_flag(units);
					}
				}
				dstate.iface_settings.add_action_to = AddActionTo::None;
				return UIModeControl::Closed;
			}
			
			return UIModeControl::UnChgd;
		}
		UIModeControl::Closed
	}
}
