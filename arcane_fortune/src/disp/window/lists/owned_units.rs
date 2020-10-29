use super::*;

impl DispState<'_,'_,'_,'_,'_> {
	pub fn print_owned_unit_list(&mut self, mode: usize, player_ind: SmSvType, players: &Vec<Player>,
			units: &Vec<Unit>, map_data: &mut MapData, temps: &Templates) {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = self.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let owned_units = owned_unit_list(units, player_ind, cursor_coord, players, &mut w, &mut label_txt_opt, map_sz, &self.local);
		
		let list_pos = self.print_list_window(mode, self.local.Select_battalion.clone(), owned_units.clone(), Some(w), label_txt_opt, 0, None);
		
		// show info box
		if owned_units.options.len() > 0 {
			let pstats = &players[self.iface_settings.cur_player as usize].stats;
			if let ArgOptionUI::UnitInd(unit_ind) = owned_units.options[mode].arg {
				let top_right = list_pos.top_right;
				self.show_exemplar_info(units[unit_ind].template.id as usize, EncyclopediaCategory::Unit, OffsetOrPos::Pos(top_right.x as usize - 1), None, OffsetOrPos::Pos(top_right.y as usize + mode + 4), InfoLevel::AbbrevNoCostNoProdTime, temps, pstats);
			}else{panicq!("invalid UI setting");}
		}
		self.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
	}
}
