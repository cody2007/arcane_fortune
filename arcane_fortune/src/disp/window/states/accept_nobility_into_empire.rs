use super::*;

// nobility request to join empire
pub struct AcceptNobilityIntoEmpireState {
	pub mode: usize,
	pub house_ind: usize
}

impl AcceptNobilityIntoEmpireState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const MAX_W: usize = 70;
		
		if let Some(house) = players[self.house_ind].ptype.house() {
			let txt = {
				let noble = &house.noble_pairs[house.head_noble_pair_ind].noble;
				
				dstate.local.noble_req_to_join
					.replace("[title]", 
							&if noble.gender_female
								{dstate.local.noble_female_title.clone()} else {dstate.local.noble_male_title.clone()}
						)
					.replace("[first_name]", &noble.name.first)
					.replace("[last_name]", &noble.name.last)
			};
			
			let wrapped_txt = wrap_txt(&txt, MAX_W);
			let w = min(MAX_W, txt.len()) + 4;
			let h = wrapped_txt.len() + 8;
			let w_pos = dstate.print_window(ScreenSz{w, h, sz:0});
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {dstate.renderer.mv(row + y, x); row += 1;};
						     ($final: expr) => {dstate.renderer.mv(row + y, x);}};
			
			mvl!(); dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer); mvl!();
			
			for line in wrapped_txt.iter() {
				mvl!();
				dstate.renderer.addstr(line);
			}
			
			mvl!();
			mvl!();
			let l = &dstate.local;
			let txt = if self.mode == 0 {format!("* {} *", l.Accept_nobility)} else {l.Accept_nobility.clone()};
			center_txt(&txt, w as i32, None, &mut dstate.renderer);
			
			dstate.renderer.mv(row + y, x);
			let txt = if self.mode == 1 {format!("* {} *", l.Reject_nobility)} else {l.Reject_nobility.clone()};
			center_txt(&txt, w as i32, None, &mut dstate.renderer);
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &mut GameState, players: &mut Vec<Player>,
			dstate: &DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const N_OPTS: usize = 2;
		let kbd = &dstate.kbd;
		match dstate.key_pressed {
			k if kbd.up(k) => {if self.mode > 0 {self.mode -= 1;} else {self.mode = N_OPTS-1;}}
			k if kbd.down(k) => {if (self.mode + 1) <= (N_OPTS-1) {self.mode += 1;} else {self.mode = 0;}}
			k if kbd.enter == k => {
				match self.mode {
					// accept nobility into empire
					0 => {
						//printlnq!("house_ind: {} {}", house_ind, players[house_ind].personalization.nm);
						gstate.relations.join_as_fiefdom(self.house_ind, dstate.iface_settings.cur_player as usize, players, &mut gstate.logs, gstate.turn);
					// reject nobility
					} 1 => {
					} _ => {panicq!("unknown UI state");}
				}
				return UIModeControl::Closed;
			} _ => {}
		}
		
		UIModeControl::UnChgd
	}
}

