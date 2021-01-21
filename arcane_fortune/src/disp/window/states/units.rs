use super::*;
pub struct UnitsWindowState {
	pub mode: usize
}

impl UnitsWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, units: &Vec<Unit>,
			map_data: &mut MapData, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		dstate.print_owned_unit_list(self.mode, dstate.iface_settings.cur_player, players, units, map_data, temps);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, units: &Vec<Unit>, map_data: &mut MapData,
			dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let cur_player = dstate.iface_settings.cur_player;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let entries = owned_unit_list(units, cur_player, cursor_coord, players, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
		let entries_present = entries.options.len() > 0;
		
		macro_rules! enter_action {($mode: expr) => {
			// move cursor to entry
			let coord = match entries.options[$mode].arg {
				ArgOptionUI::UnitInd(unit_ind) => {units[unit_ind].return_coord()}
				_ => {panicq!("unit inventory list argument option not properly set");}
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

