use super::*;

//////// production by buildings
pub struct CurrentBldgProdState {pub mode: usize}

impl CurrentBldgProdState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, bldgs: &Vec<Bldg>, players: &Vec<Player>, map_data: &mut MapData, exf: &HashedMapEx,
			temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		debug_assertq!(dstate.iface_settings.zoom_ind == map_data.max_zoom_ind());
		
		let w = 29;
		let pstats = &players[dstate.iface_settings.cur_player as usize].stats;
		
		if let Some(bldg_ind) = dstate.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
			let options = bldg_prod_list(&bldgs[bldg_ind], &dstate.local); 
			let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_an_item_to_remove.clone(), options.clone(), None, None, 0, None);
			
			// print details for selected bldg
			if let Some(option) = options.options.get(self.mode) {
				if let ArgOptionUI::UnitTemplate(Some(ut)) = option.arg {
					let template_ind = temps.units.iter().position(|r| r == ut).unwrap();
					dstate.show_exemplar_info(template_ind, EncyclopediaCategory::Unit, OffsetOrPos::Offset(w+3), None, OffsetOrPos::Offset(self.mode+4), InfoLevel::Abbrev, temps, pstats);
				// zeroth entry only should be none
				}else if self.mode != 0 {
					panicq!("could not find unit template {}", self.mode);
				}
			}
			dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		}else{
			return UIModeControl::Closed;
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, bldgs: &mut Vec<Bldg>, map_data: &mut MapData,
			exf: &HashedMapEx, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(bldg_ind) = dstate.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
			let b = &mut bldgs[bldg_ind];
			debug_assertq!(dstate.iface_settings.zoom_ind == map_data.max_zoom_ind());
			
			let l = &dstate.local;
			let buttons = &mut dstate.buttons;
			let list = bldg_prod_list(&b, l);
			macro_rules! enter_action{($mode: expr) => {
				if let BldgArgs::PopulationCenter {ref mut production, ..} |
						BldgArgs::GenericProducable {ref mut production, ..} = b.args {
					production.swap_remove($mode);
				}
				return UIModeControl::UnChgd;
			};};
			if let Some(ind) = buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
			
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
		}
		UIModeControl::UnChgd
	}
}

