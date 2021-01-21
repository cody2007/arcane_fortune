use super::*;

pub struct NobilityDeclaresIndependenceWindowState {
	pub mode: usize,
	pub owner_id: usize // nobility that declared independence
}

impl NobilityDeclaresIndependenceWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>,
			dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let nobility_player = &players[self.owner_id];
		
		let intro_txt = dstate.local.Nobility_declare_independence.replace("[house_nm]", &nobility_player.personalization.nm);
		
		let w = 100;
		
		let intro_txt_wrapped = wrap_txt(&intro_txt, w as usize - 4);
		
		let h = 6 + intro_txt_wrapped.len();
		let w_pos = dstate.print_window(ScreenSz {w, h, sz:0});
		
		let mut row = w_pos.y + 1;
		
		// print lines
		for intro_txt in intro_txt_wrapped { // for each line
			dstate.mv(row, w_pos.x + 2); row += 1;
			dstate.addstr(&intro_txt);
		}
		row += 1;
		
		let mut screen_reader_cur_loc = (0,0);
		{ // options
			row += 1;
			let screen_w = dstate.iface_settings.screen_sz.w as i32;
			
			dstate.buttons.Independence_do_nothing.print_centered_selection(row as i32, self.mode == 0, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer); row += 1;
			dstate.buttons.Independence_declare_war.print_centered_selection(row as i32, self.mode == 1, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
		}
		
		dstate.renderer.mv(screen_reader_cur_loc.0, screen_reader_cur_loc.1);
	
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>,
			gstate: &mut GameState,
			dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(mode) = button_mode_update_and_action(&mut self.mode, vec![
					&mut dstate.buttons.Independence_do_nothing,
					&mut dstate.buttons.Independence_declare_war
				], dstate.key_pressed, &dstate.mouse_event, &dstate.kbd) {
			let cur_player = dstate.iface_settings.cur_player as usize;
			
			let mood = match mode {
				// player chose to do nothing
				0 => MoodType::IndependenceNotHandled,
				// declare war
				1 => {
					gstate.relations.declare_war_ret_ui_mode(cur_player, self.owner_id, &mut gstate.logs, players, gstate.turn, None, &mut gstate.rng, dstate);
					MoodType::IndependenceNotHandled
				} _ => {panicq!("invalid return");}
			};
			
			// alter moods of all players
			for owner2 in (0..players.len()).filter(|&owner2| owner2 != cur_player && owner2 != self.owner_id) {
				gstate.relations.add_mood_factor(cur_player, owner2, mood, gstate.turn);
			}
			
			return UIModeControl::Closed;
		}
		
		UIModeControl::UnChgd
	}
}

