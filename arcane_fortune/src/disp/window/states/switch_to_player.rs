use super::*;
pub struct SwitchToPlayerWindowState {pub mode: usize}

impl SwitchToPlayerWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let all_civs = all_civilizations_list(players);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_civilization.clone(), all_civs, None, None, 0, Some(players));
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData, dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = all_civilizations_list(players);
		macro_rules! enter_action{($mode: expr) => {
			if let ArgOptionUI::OwnerInd(owner_ind) = list.options[$mode].arg {
				dstate.iface_settings.cur_player = owner_ind as SmSvType;
				
			}else{panicq!("invalid UI setting");}
			
			dstate.iface_settings.unit_subsel = 0;
			dstate.iface_settings.add_action_to = AddActionTo::None;
			
			let pstats = &mut players[dstate.iface_settings.cur_player as usize].stats;
			dstate.production_options = init_bldg_prod_windows(temps.bldgs, pstats, &dstate.local);
			dstate.update_menu_indicators(dstate.iface_settings.cur_player_paused(players));
			compute_zoomed_out_discoveries(map_data, &mut players[dstate.iface_settings.cur_player as usize].stats);
			
			return UIModeControl::Closed;
		};};
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	enter_action!(ind);}
		
		let kbd = &dstate.kbd;
		match dstate.key_pressed {
			// down
			k if kbd.down(k) => {
				if (self.mode + 1) <= (list.options.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if kbd.up(k) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == kbd.enter => {
				enter_action!(self.mode);
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}
