use super::*;

pub struct NobleUnitsWindowState {
	pub mode: usize,
	pub house_nm: Option<String>
}

impl NobleUnitsWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, units: &Vec<Unit>, map_data: &mut MapData,
			temps: &Templates, gstate: &GameState, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		
		// find and set house_nm
		if self.house_nm.is_none() {
			let houses = gstate.relations.noble_houses(cur_player);
			
			// no houses to show
			if houses.len() == 0 {
				return UIModeControl::New(UIMode::GenericAlert(GenericAlertState {
					txt: dstate.local.No_nobility_in_empire.clone()
				}));
				
			// only one house to show
			}else if houses.len() == 1 {
				self.house_nm = Some(players[houses[0]].personalization.nm.clone());
				
			// ask which house to show
			}else{
				let options = noble_houses_list(cur_player, &gstate.relations, players, &dstate.local);
				let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_a_noble_house.clone(), options, None, None, 0, None);
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
				return UIModeControl::UnChgd;
			}
		}
		
		// print the battalions owned by house_nm
		if let Some(house_nm) = &self.house_nm {
			if let Some(player) = players.iter().find(|h| h.personalization.nm == *house_nm) {
				dstate.print_owned_unit_list(self.mode, player.id, players, units, map_data, temps);
				return UIModeControl::UnChgd;
			}
			
			return UIModeControl::Closed;
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, gstate: &GameState, units: &Vec<Unit>,
			map_data: &mut MapData, dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// show units
		if let Some(house_nm) = &self.house_nm {
			if let Some(house) = players.iter().find(|player| player.personalization.nm == *house_nm) {
				let mut w = 0;
				let mut label_txt_opt = None;
				let map_sz = *map_data.map_szs.last().unwrap();
				let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
				
				let list = owned_unit_list(units, house.id, cursor_coord, players, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
				
				if list.list_mode_update_and_action(&mut self.mode, dstate) {
					// move cursor to entry
					let coord = match list.options[self.mode].arg {
						ArgOptionUI::UnitInd(unit_ind) => {units[unit_ind].return_coord()}
						_ => {panicq!("unit inventory list argument option not properly set");}
					};
					
					return UIModeControl::CloseAndGoTo(coord);
				}
				
			}else{return UIModeControl::Closed;}
		
		// get noble house to show units of
		}else{
			let cur_player = dstate.iface_settings.cur_player as usize;
			let list = gstate.relations.noble_houses(cur_player);
			
			if noble_houses_list(cur_player, &gstate.relations, players, &dstate.local).list_mode_update_and_action(&mut self.mode, dstate) {
				if let Some(house_ind) = list.get(self.mode) {
					self.house_nm = Some(players[*house_ind].personalization.nm.clone());
					self.mode = 0;
				}else{
					return UIModeControl::Closed;
				}
				return UIModeControl::UnChgd;
			}
		}
		
		UIModeControl::UnChgd
	}
}

