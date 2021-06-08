use super::*;
pub struct NoblePedigreeState {
	pub mode: usize,
	pub house_nm: Option<String>
}

impl NoblePedigreeState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, gstate: &GameState, players: &Vec<Player>,
			dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		let l = &dstate.local;
		
		// find and set house_nm
		if self.house_nm.is_none() {
			let houses = gstate.relations.noble_houses(cur_player);
			
			// no houses to show
			if houses.len() == 0 {
				return UIModeControl::New(UIMode::GenericAlert(GenericAlertState {
					txt: l.No_nobility_in_empire.clone()
				}));
				
			// only one pedigree to show
			}else if houses.len() == 1 {
				self.house_nm = Some(players[houses[0]].personalization.nm.clone());
				
			// ask which pedigree to show
			}else{
				let options = noble_houses_list(cur_player, &gstate.relations, players, l);
				let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_a_noble_house.clone(), options, None, None, 0, None);
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
				return UIModeControl::UnChgd;
			}
		}
		
		// print the pedigree
		if let Some(house_nm) = &self.house_nm {
			if let Some(player) = players.iter().find(|h| h.personalization.nm == *house_nm) {
				if let Some(house) = player.ptype.house() {
					house.print_pedigree(&player.personalization, gstate.turn, dstate);
					return UIModeControl::UnChgd;
				}
			}
			
			return UIModeControl::Closed;
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &GameState, 
			players: &Vec<Player>, dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = gstate.relations.noble_houses(dstate.iface_settings.cur_player as usize);
		
		macro_rules! enter_action{($mode: expr) => {
			if let Some(house_ind) = list.get($mode) {
				self.house_nm = Some(players[*house_ind].personalization.nm.clone());
			}else{
				return UIModeControl::Closed;
			}
		};}
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
		
		match dstate.key_pressed {
			// down
			k if dstate.kbd.down(k) => {
				if (self.mode + 1) <= (list.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if dstate.kbd.up(k) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = list.len() - 1;
				}
				
			// enter
			} k if k == dstate.kbd.enter => {
				enter_action!(self.mode);
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}

