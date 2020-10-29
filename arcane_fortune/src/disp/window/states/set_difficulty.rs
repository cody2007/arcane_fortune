use super::*;
pub struct SetDifficultyWindowState {pub mode: usize}

impl SetDifficultyWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, game_difficulties: &GameDifficulties, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = game_difficulty_list(game_difficulties);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_difficulty.clone(), list, None, None, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player>, game_difficulties: &GameDifficulties,
			dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = game_difficulty_list(game_difficulties);
		macro_rules! enter_action{($mode:expr) => {
			let new_difficulty = &game_difficulties.difficulties[$mode];
			for player in players.iter_mut() {
				if player.ptype.is_human() {continue;}
				
				player.stats.bonuses = new_difficulty.ai_bonuses.clone();
			}
			
			return UIModeControl::Closed;
		};};
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
