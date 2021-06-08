use super::*;
//use crate::units::*;
pub struct PlaceUnitWindowState {pub mode: usize}

impl PlaceUnitWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let unit_opts = discovered_units_list(pstats, temps.units, &dstate.local);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_a_unit.clone(), unit_opts, None, None, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, player: &mut Player<'bt,'ut,'rt,'dt>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, gstate: &mut GameState, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = discovered_units_list(&player.stats, temps.units, &dstate.local);
		macro_rules! enter_action{($mode:expr) => {
			if let ArgOptionUI::UnitTemplate(Some(ut)) = list.options[$mode].arg {
				if dstate.iface_settings.zoom_ind == map_data.max_zoom_ind() {
					let c = dstate.iface_settings.cursor_to_map_ind(map_data);
					add_unit(c, true, ut, units, map_data, exs, bldgs, player, gstate, temps);
				}
			}else{panicq!("invalid UI setting");}
			
			return UIModeControl::Closed;
		};}
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
		
		match dstate.key_pressed {
			// down
			k if dstate.kbd.down(k) => {
				if (self.mode + 1) <= (list.options.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if dstate.kbd.up(k) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = list.options.len() - 1;
				}
				
			// enter
			} k if k == dstate.kbd.enter => {
				enter_action!(self.mode);
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}
