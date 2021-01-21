use super::*;

// nobility request to join empire
pub struct AcceptNobilityIntoEmpireState {
	pub mode: usize,
	pub house_ind: usize
}

impl AcceptNobilityIntoEmpireState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const MAX_W: usize = 70;
		
		if let Some(nobility_state) = players[self.house_ind].ptype.nobility_state() {
			let noble = nobility_state.house.head_noble();
			let txt = dstate.local.noble_req_to_join
				.replace("[title]", 
						&if noble.gender_female
							{dstate.local.noble_female_title.clone()} else {dstate.local.noble_male_title.clone()}
					)
				.replace("[first_name]", &noble.name.first)
				.replace("[last_name]", &noble.name.last);
			
			let wrapped_txt = wrap_txt(&txt, MAX_W);
			let w = min(MAX_W, txt.len()) + 4;
			let h = wrapped_txt.len() + 8 + 4;
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
			
			{ // print personality & doctrine
				let txt = dstate.local.Noble_personality_join.replace("[name]", &noble.name.txt());
				dstate.renderer.addstr(&txt);
				dstate.addch(' ');
				
				let format_str = format!("{} & {}", FRIENDLINESS_TAG, SPIRITUALITY_TAG);
				let tags = noble.personality.color_tags(&dstate.local);
				color_tags_print(&format_str, &tags, None, &mut dstate.renderer);
				
				mvl!();
				dstate.renderer.addstr(&dstate.local.Noble_doctrine_join);
				dstate.addch(' ');
				dstate.renderer.addstr(&nobility_state.ai_state.doctrine_txt(&dstate.local));
			}
			mvl!();
			mvl!();
			
			let mut screen_reader_cur_loc = (0,0);
			{ // options
				let screen_w = dstate.iface_settings.screen_sz.w as i32;
				
				dstate.buttons.Accept_nobility.print_centered_selection(row + y, self.mode == 0, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer); row += 1;
				dstate.buttons.Reject_nobility.print_centered_selection(row + y, self.mode == 1, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
			}
			
			dstate.renderer.mv(screen_reader_cur_loc.0, screen_reader_cur_loc.1);
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &mut GameState, players: &mut Vec<Player>,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(mode) = button_mode_update_and_action(&mut self.mode, vec![
							&mut dstate.buttons.Accept_nobility,
							&mut dstate.buttons.Reject_nobility
						], dstate.key_pressed, &dstate.mouse_event, &dstate.kbd) {
			match mode {
				// accept nobility into empire
				0 => {
					gstate.relations.join_as_fiefdom(self.house_ind, dstate.iface_settings.cur_player as usize, players, &mut gstate.logs, gstate.turn, &mut gstate.rng, dstate);
				// reject nobility
				} 1 => {
				} _ => {panicq!("unknown UI state");}
			}
			return UIModeControl::Closed;
		}
		UIModeControl::UnChgd
	}
}

