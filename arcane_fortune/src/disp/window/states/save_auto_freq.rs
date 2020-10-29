use super::*;
use std::convert::TryFrom;
pub struct SaveAutoFreqWindowState {
	pub freq: String,
	pub curs_col: isize
}

impl SaveAutoFreqWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let buttons = &mut dstate.buttons;
		let l = &dstate.local;
		
		let instructions_w = format!("{}   {}", buttons.Esc_to_close.print_txt(l), buttons.Confirm.print_txt(l)).len() as i32;
		let w = max(l.years_0_disables.len(), instructions_w as usize) + 4;
		let h = 9;
		
		let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
		
		let buttons = &mut dstate.buttons;
		let l = &dstate.local;
		let d = &mut dstate.renderer;
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 1;
		
		let w = (w - 2) as i32;
		
		d.mv(y,x);
		center_txt(&l.Save_auto_freq, w, Some(COLOR_PAIR(TITLE_COLOR)), d);
		
		d.mv(y+2,x+1);
		d.addstr(&l.Auto_save_the_game_every);
		
		// print file name
		d.mv(y+3,x+1);
		d.addstr(&self.freq);
		
		d.mv(y+4,x+1);
		d.addstr(&l.years_0_disables);
		
		// instructions
		{
			let gap = ((w - instructions_w)/2) as i32;
			d.mv(y + 6, x + gap);
			buttons.Esc_to_close.print(None, l, d); d.addstr("  ");
			buttons.Confirm.print(None, l, d);
		}
		
		// mv to cursor location
		d.mv(y + 3, x + 1 + self.curs_col as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if dstate.buttons.Confirm.activated(dstate.key_pressed, &dstate.mouse_event) && self.freq.len() > 0 {
			if let Result::Ok(val) = self.freq.parse() {
				dstate.iface_settings.checkpoint_freq = val;
			}
			return UIModeControl::Closed;
		}
		
		match dstate.key_pressed {
			KEY_LEFT => {if self.curs_col != 0 {self.curs_col -= 1;}}
			KEY_RIGHT => {
				if self.curs_col < (self.freq.len() as isize) {
					self.curs_col += 1;
				}
			}
			
			KEY_HOME | KEY_UP => {self.curs_col = 0;}
			
			// end key
			KEY_DOWN | 0x166 | 0602 => {self.curs_col = self.freq.len() as isize;}
			
			// backspace
			KEY_BACKSPACE | 127 | 0x8  => {
				if self.curs_col != 0 {
					self.curs_col -= 1;
					self.freq.remove(self.curs_col as usize);
				}
			}
			
			// delete
			KEY_DC => {
				if self.curs_col != self.freq.len() as isize {
					self.freq.remove(self.curs_col as usize);
				}
			}
			_ => { // insert character
				if self.freq.len() < (min(MAX_SAVE_AS_W, dstate.iface_settings.screen_sz.w)-5) {
					if let Result::Ok(c) = u8::try_from(dstate.key_pressed) {
						if let Result::Ok(ch) = char::try_from(c) {
							if "012345679".contains(ch) {
								self.freq.insert(self.curs_col as usize, ch);
								self.curs_col += 1;
							}
						}
					}
				}
			}
		}
		UIModeControl::UnChgd
	}
}
