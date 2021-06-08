use super::*;
pub struct BldgsWindowState {
	pub mode: usize,
	pub bldgs_show: BldgsShow
}

pub enum BldgsShow {Military, Improvements}

impl BldgsWindowState {
	pub fn print<'bt,'ut,'rt,'dt> (&self, pstats: &Stats, bldgs: &Vec<Bldg>, temps: &Templates,
			map_data: &mut MapData, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt>  {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		let cur_player = dstate.iface_settings.cur_player;
		
		let owned_bldgs = match self.bldgs_show {
			BldgsShow::Improvements {..} => {owned_improvement_bldgs_list(bldgs, temps.doctrines, cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local)}
			BldgsShow::Military {..} => {owned_military_bldgs_list(bldgs, cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local)}
		};
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_a_building.clone(), owned_bldgs.clone(), Some(w), label_txt_opt, 0, None);
		
		// show info box
		if owned_bldgs.options.len() > 0 {
			if let ArgOptionUI::BldgInd(bldg_ind) = owned_bldgs.options[self.mode].arg {
				let top_right = list_pos.top_right;
				dstate.show_exemplar_info(bldgs[bldg_ind].template.id as usize, EncyclopediaCategory::Bldg, OffsetOrPos::Pos(top_right.x as usize - 1), None, OffsetOrPos::Pos(top_right.y as usize + self.mode + 4), InfoLevel::AbbrevNoCostNoProdTime, temps, pstats);
			}else{panicq!("invalid UI setting");}
		}
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, temps: &Templates, bldgs: &Vec<Bldg>, map_data: &mut MapData, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let entries = match self.bldgs_show {
			BldgsShow::Improvements => {
				owned_improvement_bldgs_list(bldgs, &temps.doctrines, dstate.iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local)
			} BldgsShow::Military => {
				owned_military_bldgs_list(bldgs, dstate.iface_settings.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local)
			}
		};
		let entries_present = entries.options.len() > 0;
		
		macro_rules! enter_action {($mode: expr) => {
			// move cursor to entry
			let coord = match entries.options[$mode].arg {
				ArgOptionUI::BldgInd(bldg_ind) => {bldgs[bldg_ind].coord}
				_ => {panicq!("unit inventory list argument option not properly set");}
			};
			
			return UIModeControl::CloseAndGoTo(coord);
		};}
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
