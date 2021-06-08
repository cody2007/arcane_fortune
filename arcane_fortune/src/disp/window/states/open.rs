use super::*;
use crate::config_load::SaveFile;
use crate::saving::*;

// open game
pub struct OpenWindowState {
	pub prev_auto_turn: AutoTurn,
	pub save_files: Vec<SaveFile>,
	pub mode: usize
}

impl OpenWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// max file name length
		let mut max_txt_len = self.save_files.iter().map(|f| f.nm.len()).max().unwrap_or(0);
		max_txt_len += "     XXX XX-XX-XXXX  XX:XX XX (GMT)".len() + 2;
		
		let mut w = min(max_txt_len+1, dstate.iface_settings.screen_sz.w);
		let h = min(dstate.iface_settings.screen_sz.h, self.save_files.len() + 2 + 3 + 2);
		
		let n_rows_plot = min(self.save_files.len(), dstate.iface_settings.screen_sz.h - 4 - 2) as i32;
		if n_rows_plot < self.save_files.len() as i32 { // add additional width if scrolling
			w += 1;
		}
		
		let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
		
		let d = &mut dstate.renderer;
		let buttons = &mut dstate.buttons;
		let l = &dstate.local;
		let title_c = Some(COLOR_PAIR(TITLE_COLOR));
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 1;
		
		d.mv(y,x);
		center_txt("Open game", w as i32, title_c, d);
		
		// first file to print
		let first_ln = if self.mode >= (n_rows_plot as usize) {
			self.mode - n_rows_plot as usize + 1
		}else{0};
		
		// print files
		if self.save_files.len() != 0 {
			for (i, s) in self.save_files.iter().enumerate() {
				if i < first_ln {continue;}
				if i >= (first_ln + n_rows_plot as usize) {break;}
				
				let row_start = y + i as i32 + 2 - first_ln as i32;
				//if row_start >= (self.screen_sz.h - 4) as i32 {break;}
				
				// modified time start column
				let mut mod_col_start = (x-1)+  w as i32  - 2 - 3 - (s.modified.len() as i32);
				if n_rows_plot < self.save_files.len() as i32 { // move over to left for scrollbar
					mod_col_start -= 1;
				}
				
				// file name
				d.mv(row_start, x + 1);
				let button_start = cursor_pos(d);
				
				if i == self.mode {d.attron(A_REVERSE());}
				for i in 0..min(mod_col_start-1-x, s.nm.chars().count() as i32) {
					d.addch(s.nm.chars().nth(i as usize).unwrap());
				}
				
				// print ... if file name too long
				if mod_col_start < (x+1+s.nm.chars().count() as i32) {
					d.mv(row_start, mod_col_start - 3);
					d.addstr("...");
				}else{
					for _ in 0..((mod_col_start-1-x) - (s.nm.chars().count() as i32)) {
						d.addch(' ');
					}
				}
				
				// modified
				d.mv(row_start, mod_col_start);
				d.addstr("   ");
				d.addstr(&s.modified);
				buttons.add(button_start, i, d);
				if i == self.mode {d.attroff(A_REVERSE());}
			}
		// no files
		}else{
			let row_start = y + 2 - first_ln as i32;
			d.mv(row_start, x + 1);
			d.addstr("No files");
		}
		
		// instructions
		{
			let instructions_w = buttons.Esc_to_close.print_txt(l).len() + 
				"   ".len() + buttons.Open.print_txt(l).len();
			let gap = ((w - instructions_w)/2) as i32;
			
			let row_start = if self.save_files.len() as i32 > n_rows_plot {
				1 + (dstate.iface_settings.screen_sz.h - 4) as i32
			}else{
				y + self.save_files.len() as i32 + 3
			};
			d.mv(row_start, x - 1 + gap);
			buttons.Esc_to_close.print(None, l, d);
			d.addstr("   ");
			buttons.Open.print(None, l, d);
		}
		
		//////// print scroll bars
		if self.save_files.len() > n_rows_plot as usize {
			let h = h as i32;
			let w = x as i32 + w as i32 - 2;
			let scroll_track_h = n_rows_plot;
			let frac_covered = n_rows_plot as f32 / self.save_files.len() as f32;
			let scroll_bar_h = ((scroll_track_h as f32) * frac_covered).round() as i32;
			debug_assertq!(frac_covered <= 1.);
			
			let frac_at_numer = if self.mode < n_rows_plot as usize {
				0
			} else {
				first_ln + 2//+ n_rows_plot as usize
			};
			
			let frac_at = frac_at_numer as f32 / self.save_files.len() as f32;
			let scroll_bar_start = ((scroll_track_h as f32) * frac_at).round() as i32;
			
			d.mv(LOG_START_ROW, w-1);
			d.attron(COLOR_PAIR(CLOGO));
			d.addch(dstate.chars.hline_char);
			for row in 0..scroll_bar_h-1 {
				d.mv(row + 1 + scroll_bar_start + LOG_START_ROW, w-1);
				d.addch(dstate.chars.vline_char);
				//d.addch('#' as chtype);
			}
			d.mv(h-LOG_STOP_ROW, w-1);
			d.addch(dstate.chars.hline_char);
			d.attroff(COLOR_PAIR(CLOGO));
		}
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, game_control: &mut GameControl, dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		macro_rules! enter_action {($mode: expr) => {
			*game_control = GameControl::Load(format!("{}/{}", SAVE_DIR, self.save_files[$mode].nm));
			return UIModeControl::ChgGameControl;
		};}
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	enter_action!(ind);}
		else if dstate.buttons.Open.activated(dstate.key_pressed, &dstate.mouse_event) && self.save_files.len() != 0 {
			enter_action!(self.mode);
		}
		
		match dstate.key_pressed {
			// down
			k if dstate.kbd.down(k) => {
				if (self.mode + 1) <= (self.save_files.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if dstate.kbd.up(k) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = self.save_files.len() - 1;
				}
			} _ => {}
		}
		
		return UIModeControl::UnChgd;
	}
}
