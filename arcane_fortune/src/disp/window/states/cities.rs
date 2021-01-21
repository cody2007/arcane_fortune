use super::*;
pub struct CitiesWindowState {pub mode: usize}

////////////// show owned cities
impl CitiesWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, map_data: &mut MapData, bldgs: &Vec<Bldg>, gstate: &GameState, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let owned_cities = owned_city_list(bldgs, dstate.iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &gstate.logs, &dstate.local);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_city.clone(), owned_cities, Some(w), label_txt_opt, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, bldgs: &Vec<Bldg>, map_data: &mut MapData, gstate: &GameState, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let entries = owned_city_list(bldgs, dstate.iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &gstate.logs, &dstate.local);
		let entries_present = entries.options.len() > 0;
		
		//////////////////////// handle keys
		macro_rules! enter_action {($mode: expr) => {
			// move cursor to entry
			let coord = match entries.options[$mode].arg {
				ArgOptionUI::CityInd(bldg_ind) => {bldgs[bldg_ind].coord}
				_ => {panicq!("inventory list argument option not properly set");}
			};
			
			return UIModeControl::CloseAndGoTo(coord);
		};};
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
		
		match dstate.key_pressed {
			k if entries_present && (dstate.kbd.down(k)) => {
				if (self.mode + 1) <= (entries.options.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if entries_present && (dstate.kbd.up(k)) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = entries.options.len() - 1;
				}
				
			// enter
			} k if entries_present && (k == dstate.kbd.enter) => {
				enter_action!(self.mode);
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}
