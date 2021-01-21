use std::process::exit;
use std::time::Instant;

use crate::renderer::*;
use crate::saving::*;
use crate::saving::save_game::SAVE_DIR;
use crate::config_load::return_save_files;
use super::vars::{ScreenSz};
use super::fns::getmaxyxu;
use super::logo_vars::*;
use super::color::*;
use super::version_status::{VERSION};
use crate::containers::*;
use super::*;

const IGN_KEY_TIME: u64 = 50;

enum TitleOptionSel { LoadGm=0, NewGm=1, Exit=2, N=3 }

impl <'f,'bt,'ut,'rt,'dt>DispState<'f,'_,'bt,'ut,'rt,'dt> {
	fn print_raster(&mut self, raster: &[u8], row_start: i32, col_start: i32, n_cols: i32,
			n_rows: i32, col_skip_b: i32, col_skip_e: i32){
		
		let mut ind = 0;
		for row in 0..n_rows {
			self.renderer.mv((row_start + row) as i32, col_start as i32);
			ind += col_skip_b;
			for _col in 0..(n_cols - col_skip_b - col_skip_e) {
				if raster[ind as usize] != 0 {
					self.renderer.addch(self.chars.land_char as chtype | COLOR_PAIR(CLOGO) | A_BOLD());
				}else{
					self.renderer.addch(' ' as chtype);
				}
				ind += 1;
			}
			ind += col_skip_e;
		}
	}

	pub fn print_centered_logo(&mut self, screen_sz: ScreenSz, additional_txt: usize) -> usize {
		let sub = LOGO_HEIGHT + 6 + additional_txt;
		if screen_sz.h < sub {return 0;}
		let txt_under_logo_row_off = (screen_sz.h - sub)/2;
		
		let mut col_off = (screen_sz.w as i32 - LOGO_WIDTH as i32)/2;
		let mut sub_arcane = 0;
		if screen_sz.w <= (N_COLS_ARCANE_LOGO + 2) {
			sub_arcane = (N_COLS_ARCANE_LOGO + 2) as i32 - screen_sz.w as i32;
			col_off = 0;
		}
		
		let mut sub_fortune = 0;
		let mut col_skip_b_fortune = 0;
		if screen_sz.w <= (N_COLS_FORTUNE_LOGO - 3) {
			col_skip_b_fortune = 2;
			sub_fortune = (N_COLS_FORTUNE_LOGO - 3 - col_skip_b_fortune) as i32 - screen_sz.w as i32;
			col_off = 0;
		}
		
		// print centered
		if !screen_reader_mode() {
			self.print_raster(LOGO_ARCANE, txt_under_logo_row_off as i32, col_off+2, N_COLS_ARCANE_LOGO as i32, N_ROWS_ARCANE_LOGO as i32, 0, sub_arcane);
			self.print_raster(LOGO_FORTUNE, txt_under_logo_row_off as i32 + N_ROWS_ARCANE_LOGO as i32, col_off, N_COLS_FORTUNE_LOGO as i32, N_ROWS_FORTUNE_LOGO as i32, 2 + col_skip_b_fortune as i32, 1 + sub_fortune as i32);
		}else{
			let txt = "Arcane Fortune";
			self.renderer.mv(txt_under_logo_row_off as i32, ((screen_sz.w - txt.len())/2) as i32);
			self.renderer.addstr(txt);
		}
		txt_under_logo_row_off
	}
	
	// clear then print centered logo with text:
	pub fn print_clear_centered_logo_txt(&mut self, txt: String){
		self.renderer.clear();
		let screen_sz = getmaxyxu(&mut self.renderer);
		let mut txt_under_logo_row_off = self.print_centered_logo(screen_sz, 0) as i32;
		txt_under_logo_row_off += (LOGO_HEIGHT + 2) as i32;
		
		let col = ((screen_sz.w - txt.len())/2) as i32;
		self.renderer.mv(txt_under_logo_row_off, col as i32);
		self.renderer.addstr(&txt);
		self.renderer.refresh();
	}
}

const VISIT: &str = "";//Visit ";
pub const URL: &str = "https://arcanefortune.com";

impl <'f,'bt,'ut,'rt,'dt>DispState<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn show_title_screen(&mut self) -> GameControl {
		self.renderer.curs_set(if screen_reader_mode() {CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE
		}else{CURSOR_VISIBILITY::CURSOR_INVISIBLE});
		
		let mut screen_reader_cur_loc = (0,0); // cursor loc for screen readers
		
		let screen_sz = getmaxyxu(&self.renderer);
		let mut screen_sz_prev = screen_sz.clone();
		
		let mut key_pressed = ERR;
		
		let mut visit_url = String::new();
		visit_url.push_str(VISIT);
		visit_url.push_str(URL);
		
		let save_file_exists = return_save_files().len() != 0;
		let min_option = if save_file_exists {0} else {1}; // don't show load game option if no files exist
		let mut title_option_sel = min_option;
		//let langs_supported = Localization::supported();
		
		let start_time = Instant::now();

		loop {
			let screen_sz = getmaxyxu(&self.renderer);
			let mouse_event = self.renderer.getmouse(key_pressed);
			
			// clear screen if terminal sz changes
			if screen_sz != screen_sz_prev {
				self.renderer.clear();
				screen_sz_prev = screen_sz.clone();
			}
			
			macro_rules! load_game{() => {
				// load most recently modified file
				let save_files = return_save_files();
				return GameControl::Load(format!("{}/{}", SAVE_DIR, save_files[0].nm));
			};};
			macro_rules! new_game{() => {return GameControl::NewOptions;};};
			macro_rules! exit{() => {endwin(); exit(0);};};
			
			if self.buttons.Load_game.activated(0, &mouse_event) {load_game!();}
			if self.buttons.New_game.activated(0, &mouse_event) {new_game!();}
			if self.buttons.Exit.activated(0, &mouse_event) {exit!();}
			
			if self.buttons.Load_game.hovered(&mouse_event) {title_option_sel = 0;} else
			if self.buttons.New_game.hovered(&mouse_event) {title_option_sel = 1;} else
			if self.buttons.Exit.hovered(&mouse_event) {title_option_sel = 2;}
			
			// handle keys
			match key_pressed {
				k if k == KEY_UP || k == self.kbd.up => {
					title_option_sel -= 1;
					if title_option_sel < min_option {
						title_option_sel = TitleOptionSel::N as isize - 1;
					}},
				k if k == KEY_DOWN || k == self.kbd.down => {
					title_option_sel += 1;
					if title_option_sel > (TitleOptionSel::N as isize - 1) { // wrap
						title_option_sel = min_option;
					}},
				k if k == ('\n' as i32) => { // enter
					self.renderer.set_mouse_to_arrow();
					match title_option_sel {
						i if i == TitleOptionSel::LoadGm as isize => {load_game!();}
						i if i == TitleOptionSel::NewGm as isize => {new_game!();}
						i if i == TitleOptionSel::Exit as isize => {exit!();},
						_ => panicq!("unknown title menu option selection")
					}},
				// change language preferences
				/*k if k == ('\t' as i32) => {
					let mut lang_ind_new = l.lang_ind + 1;
					if lang_ind_new >= langs_supported.len() {
						lang_ind_new = 0;
					}
					l.set(lang_ind_new);
					d.clear();
				}*/
				_ => () 
			}
			
			// printing
			let mut roff = self.print_centered_logo(screen_sz, 0) as i32;
			
			roff += LOGO_HEIGHT as i32 + 1;
			
			macro_rules! center_txt{($txt: expr) => {
				self.renderer.mv(roff as i32, ((screen_sz.w - $txt.len())/2) as i32);
				self.renderer.addstr(&$txt);
			};};
			
			center_txt!(&format!("{} -- v{}", self.local.Edition, VERSION)); roff += 2;
			center_txt!(&visit_url); roff += 2;
			
			{ // colorize url
				let mut y = 0;
				let mut x = 0;
				self.renderer.getyx(stdscr(), &mut y, &mut x);
				self.renderer.mv(y, x - (URL.len()) as i32);
			}
			addstr_attr(URL, self.chars.shortcut_indicator as chtype, &mut self.renderer);
			
			let w = screen_sz.w as i32;
			if save_file_exists {
				let highlight = title_option_sel == TitleOptionSel::LoadGm as isize;
				self.buttons.Load_game.print_centered_selection(roff, highlight, &mut screen_reader_cur_loc, w, &mut self.renderer);
				roff += 2;
			}
			
			let highlight = title_option_sel == TitleOptionSel::NewGm as isize;
			self.buttons.New_game.print_centered_selection(roff, highlight, &mut screen_reader_cur_loc, w, &mut self.renderer);
			roff += 2;
			
			let highlight = title_option_sel == TitleOptionSel::Exit as isize;
			self.buttons.Exit.print_centered_selection(roff, highlight, &mut screen_reader_cur_loc, w, &mut self.renderer);
			
			/*{ // language prefs
				let mut lang_txt = String::from(": ");
				for (lang_ind, lang_supported) in langs_supported.iter().enumerate() {
					if lang_ind == l.lang_ind {lang_txt.push('*');}
					lang_txt.push_str(lang_supported);
					if lang_ind == l.lang_ind {lang_txt.push('*');}
					if lang_ind != (langs_supported.len()-1) {
						lang_txt.push_str(", ");
					}
				}
					
				d.mv((screen_sz.h-1) as i32, (screen_sz.w - 1 - l.Tab_key.len() - lang_txt.len()) as i32);
				addstr_c(&l.Tab_key, ESC_COLOR, d);
				
				d.addstr(&lang_txt);
			}*/
			
			self.renderer.mv(screen_reader_cur_loc.0, screen_reader_cur_loc.1);
			self.renderer.refresh();
			self.renderer.set_mouse_to_arrow();
			loop {
				key_pressed = self.renderer.getch();
				// ignore enter key for a short amount of time
				if key_pressed == '\n' as i32 && (start_time.elapsed().as_millis() as u64) < IGN_KEY_TIME {continue;}
				
				if key_pressed != ERR {break;}
			}
		}
	}
}

