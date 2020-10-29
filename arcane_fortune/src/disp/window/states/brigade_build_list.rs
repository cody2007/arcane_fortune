use super::*;
pub struct BrigadeBuildListState {
	pub mode: usize,
	pub brigade_nm: String
}

impl BrigadeBuildListState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let l = &dstate.local;
		let w = 1 + l.Or_select_an_action_and_press.len() + 5 + l.to_remove.len() + 1 + 4 + 4;
		let label_txt_opt = None;
		let entries = brigade_build_list(&self.brigade_nm, pstats, l);
		
		let n_gap_lines = 1;
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_battalion.clone(), entries, Some(w), label_txt_opt, n_gap_lines, None);
		
		let top_left = list_pos.top_left;
		let mut roff = top_left.y as i32 + 2;
		macro_rules! mvl{() => {dstate.mv(roff, top_left.x as i32 + 2); roff += 1;};
			($fin: expr) => {dstate.mv(roff, top_left.x as i32 + 2);};};
		
		// Press 'a' to add an action
		// Or select an action and press <Delete> to remove:			
		
		mvl!();
		dstate.buttons.Press_to_add_action_to_brigade_build_list.print(None, &dstate.local, &mut dstate.renderer);
		
		mvl!(1);
		dstate.renderer.addstr(&dstate.local.Or_select_an_action_and_press); dstate.addch(' ');
		print_key_always_active(KEY_DC, &dstate.local, &mut dstate.renderer);
		dstate.addch(' '); dstate.renderer.addstr(&dstate.local.to_remove);
		
		dstate.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, pstats: &mut Stats, units: &Vec<Unit>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = brigade_build_list(&self.brigade_nm, pstats, &dstate.local);
		
		// add aciton to brigade build list
		if dstate.buttons.Press_to_add_action_to_brigade_build_list.activated(dstate.key_pressed, &dstate.mouse_event) &&
				pstats.brigade_frm_nm(&self.brigade_nm).has_buildable_actions(units) {
			dstate.iface_settings.add_action_to = AddActionTo::BrigadeBuildList {
				brigade_nm: self.brigade_nm.clone(),
				action: None
			};
			
			return UIModeControl::Closed;
		}
		
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
			} k if k == KEY_DC as i32 => {
				let brigade = pstats.brigade_frm_nm_mut(&self.brigade_nm);
				if brigade.build_list.len() != 0 {
					brigade.build_list.remove(self.mode);
					if self.mode >= brigade.build_list.len() {
						self.mode = brigade.build_list.len() - 1;
					}
				}				
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}
