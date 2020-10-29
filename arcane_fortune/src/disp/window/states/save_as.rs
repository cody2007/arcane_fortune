use super::*;
use crate::saving::*;
use std::convert::TryFrom;

pub struct SaveAsWindowState {
	pub prev_auto_turn: AutoTurn,
	pub save_nm: String,
	pub curs_col: isize
}

impl SaveAsWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let w = min(MAX_SAVE_AS_W, dstate.iface_settings.screen_sz.w);
		let h = 7;
		
		let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 1;
		
		let w = (w - 2) as i32;
		
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		d.mv(y,x);
		center_txt(&l.Save_as, w, Some(COLOR_PAIR(TITLE_COLOR)), d);
		
		// print file name
		d.mv(y+2,x+1);
		d.addstr(&self.save_nm);
					
		// instructions
		{
			let instructions_w = format!("{}   {}", dstate.buttons.Esc_to_close.print_txt(l), dstate.buttons.Save.print_txt(l)).len() as i32;
			let gap = ((w - instructions_w)/2) as i32;
			d.mv(y + 4, x - 1 + gap);
			dstate.buttons.Esc_to_close.print(None, l, d); d.addstr("   ");
			dstate.buttons.Save.print(None, l, d);
		}
		
		// mv to cursor location
		d.mv(y + 2, x + 1 + self.curs_col as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &GameState, map_data: &MapData<'rt>, exs: &Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, 
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, units: &Vec<Unit<'bt,'ut,'rt,'dt>>, players: &Vec<Player>, 
			frame_stats: &FrameStats, dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if dstate.buttons.Save.activated(dstate.key_pressed, &dstate.mouse_event) && self.save_nm.len() > 0 {
			dstate.iface_settings.save_nm = self.save_nm.clone();
			//dstate.reset_auto_turn(); // save_game will clear disp.ui_mode which contains the prior value of the auto turn setting
			save_game(SaveType::Manual, gstate, map_data, exs, temps, bldgs, units, players, dstate, frame_stats);
			return UIModeControl::Closed;
		}else{
			do_txt_entry_keys!(dstate.key_pressed, self.curs_col, self.save_nm, Printable::FileNm, dstate);
		}
		UIModeControl::UnChgd
	}
}
