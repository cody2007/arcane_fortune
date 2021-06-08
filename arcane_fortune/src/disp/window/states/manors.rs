use super::*;
pub struct ManorsWindowState {pub mode: usize}

////////////// list of manors -- clicking goes to the manor
impl ManorsWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, bldgs: &Vec<Bldg>,
			map_data: &mut MapData, gstate: &GameState, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		if gstate.relations.noble_houses(cur_player as usize).len() == 0 {
			return UIModeControl::New(UIMode::GenericAlert(GenericAlertState {
				txt: dstate.local.No_nobility_in_empire.clone()
			}));
		}
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let owned_manors = manors_list(bldgs, cur_player as SmSvType, players, gstate, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_manor.clone(), owned_manors, Some(w), label_txt_opt, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, bldgs: &Vec<Bldg>, gstate: &GameState,
			map_data: &mut MapData, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let entries = {
			let mut w = 0;
			let mut label_txt_opt = None;
			let map_sz = *map_data.map_szs.last().unwrap();
			let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
			
			manors_list(bldgs, dstate.iface_settings.cur_player, players, gstate, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local)
		};
		
		if entries.list_mode_update_and_action(&mut self.mode, dstate) {
			// move cursor to entry
			let coord = match entries.options[self.mode].arg {
				ArgOptionUI::BldgInd(bldg_ind) => {bldgs[bldg_ind].coord}
				ArgOptionUI::CityInd(city_ind) => {bldgs[city_ind].coord}
				_ => {panicq!("unit inventory list argument option not properly set");}
			};
			
			return UIModeControl::CloseAndGoTo(coord);
		}
		return UIModeControl::UnChgd	
	}
}

