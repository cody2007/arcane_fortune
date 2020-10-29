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
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, units: &Vec<Unit>, bldgs: &Vec<Bldg>, gstate: &GameState,
			map_data: &mut MapData, exs: &Vec<HashedMapEx>, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let entries = manors_list(bldgs, dstate.iface_settings.cur_player, players, gstate, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
		
		let entries_present = entries.options.len() > 0;
		
		//////////////////////// handle keys
		macro_rules! enter_action {($mode: expr) => {
			// move cursor to entry
			let coord = match entries.options[$mode].arg {
				ArgOptionUI::BldgInd(bldg_ind) => {bldgs[bldg_ind].coord}
				ArgOptionUI::CityInd(city_ind) => {bldgs[city_ind].coord}
				_ => {panicq!("unit inventory list argument option not properly set");}
			};
			
			return UIModeControl::CloseAndGoTo(coord);
		};};
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	enter_action!(ind);}
		
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
		return UIModeControl::UnChgd	
	}
}
