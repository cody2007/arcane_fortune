use std::process::exit;
use std::time::Instant;

use crate::disp_lib::*;
use crate::saving::*;
use crate::saving::save_game::SAVE_DIR;
use crate::config_load::return_save_files;
use super::vars::{ScreenSz};
use super::fns::getmaxyxu;
use super::logo_vars::*;
use super::color::*;
use super::version_status::{VERSION};
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;
use super::{addstr_attr, Button, Buttons, cursor_pos, screen_reader_mode};

const IGN_KEY_TIME: u64 = 50;

enum TitleOptionSel { LoadGm=0, NewGm=1, Exit=2, N=3 }

fn print_raster(raster: &[u8], row_start: i32, col_start: i32, n_cols: i32,
		n_rows: i32, col_skip_b: i32, col_skip_e: i32, disp_chars: &DispChars, d: &mut DispState){
	
	let mut ind = 0;
	for row in 0..n_rows {
		d.mv((row_start + row) as i32, col_start as i32);
		ind += col_skip_b;
		for _col in 0..(n_cols - col_skip_b - col_skip_e) {
			if raster[ind as usize] != 0 {
				d.addch(disp_chars.land_char as chtype | COLOR_PAIR(CLOGO) | A_BOLD());
			}else{
				d.addch(' ' as chtype);
			}
			ind += 1;
		}
		ind += col_skip_e;
	}
}

pub fn print_centered_logo(screen_sz: ScreenSz, disp_chars: &DispChars, additional_txt: usize, d: &mut DispState) -> usize {
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
	print_raster(LOGO_ARCANE, txt_under_logo_row_off as i32, col_off+2, N_COLS_ARCANE_LOGO as i32, N_ROWS_ARCANE_LOGO as i32, 0, sub_arcane, disp_chars, d);
	print_raster(LOGO_FORTUNE, txt_under_logo_row_off as i32 + N_ROWS_ARCANE_LOGO as i32, col_off, N_COLS_FORTUNE_LOGO as i32, N_ROWS_FORTUNE_LOGO as i32, 2 + col_skip_b_fortune as i32, 1 + sub_fortune as i32, disp_chars, d);
	txt_under_logo_row_off
}

fn print_ln(button: &mut Button, roff: usize, highlight: bool, sel_loc: &mut (i32, i32),
		screen_sz: ScreenSz, d: &mut DispState) {
	let col = ((screen_sz.w - button.txt.len() - 4) as i32)/2;
	
	d.mv(roff as i32, col);
	if highlight {
		d.addstr("* ");
		{ // set position for screen readers
			let curs = cursor_pos(d);
			*sel_loc = (curs.y as i32, curs.x as i32);
		}
		button.print_without_parsing(d);
		d.addstr(" *");
	}else{
		d.addstr("  ");
		button.print_without_parsing(d);
		d.addstr("  ");
	}
}

// clear then print centered logo with text:
pub fn print_clear_centered_logo_txt(txt: &str, disp_chars: &DispChars, d: &mut DispState){
	d.clear();
	let screen_sz = getmaxyxu(d);
	let mut txt_under_logo_row_off = print_centered_logo(screen_sz, disp_chars, 0, d) as i32;
	txt_under_logo_row_off += (LOGO_HEIGHT + 2) as i32;
	
	let col = ((screen_sz.w - txt.len())/2) as i32;
	d.mv(txt_under_logo_row_off, col as i32);
	d.addstr(txt);
	d.refresh();
}

const VISIT: &str = "";//Visit ";
pub const URL: &str = "https://arcanefortune.com";

pub fn show_title_screen<'f,'bt,'ut>(disp_chars: &DispChars, kbd: &KeyboardMap,
		buttons: &mut Buttons, l: &mut Localization, d: &mut DispState) -> GameState {
	d.curs_set(if screen_reader_mode() {CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE
	}else{CURSOR_VISIBILITY::CURSOR_INVISIBLE});
	
	let mut sel_loc = (0,0); // cursor loc for screen readers
	
	let screen_sz = getmaxyxu(d);
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
		let screen_sz = getmaxyxu(d);
		let mouse_event = d.getmouse(key_pressed);
		
		// clear screen if terminal sz changes
		if screen_sz != screen_sz_prev {
			d.clear();
			screen_sz_prev = screen_sz.clone();
		}
		
		macro_rules! load_game{() => {
			// load most recently modified file
			let save_files = return_save_files();
			return GameState::Load(format!("{}/{}", SAVE_DIR, save_files[0].nm));
		};};
		macro_rules! new_game{() => {return GameState::NewOptions;};};
		macro_rules! exit{() => {endwin(); exit(0);};};
		
		if buttons.Load_game.activated(0, &mouse_event) {load_game!();}
		if buttons.New_game.activated(0, &mouse_event) {new_game!();}
		if buttons.Exit.activated(0, &mouse_event) {exit!();}
		
		if buttons.Load_game.hovered(&mouse_event) {title_option_sel = 0;} else
		if buttons.New_game.hovered(&mouse_event) {title_option_sel = 1;} else
		if buttons.Exit.hovered(&mouse_event) {title_option_sel = 2;}
		
		// handle keys
		match key_pressed {
			k if k == KEY_UP || k == kbd.up => {
				title_option_sel -= 1;
				if title_option_sel < min_option {
					title_option_sel = TitleOptionSel::N as isize - 1;
				}},
			k if k == KEY_DOWN || k == kbd.down => {
				title_option_sel += 1;
				if title_option_sel > (TitleOptionSel::N as isize - 1) { // wrap
					title_option_sel = min_option;
				}},
			k if k == ('\n' as i32) => { // enter
				d.set_mouse_to_arrow();
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
		let mut roff = print_centered_logo(screen_sz, disp_chars, 0, d);
		
		roff += LOGO_HEIGHT + 1;
		
		macro_rules! center_txt{($txt: expr) => {
			d.mv(roff as i32, ((screen_sz.w - $txt.len())/2) as i32);
			d.addstr(&$txt);
		};};
		
		center_txt!(&format!("{} -- v{}", l.Edition, VERSION)); roff += 2;
		center_txt!(&visit_url); roff += 2;
		
		// colorize url
		{
			let mut y = 0;
			let mut x = 0;
			d.getyx(stdscr(), &mut y, &mut x);
			d.mv(y, x - (URL.len()) as i32);
		}
		addstr_attr(URL, disp_chars.shortcut_indicator as chtype, d);
		
		if save_file_exists {
			let highlight = title_option_sel == TitleOptionSel::LoadGm as isize;
			print_ln(&mut buttons.Load_game, roff, highlight, &mut sel_loc, screen_sz, d);
			roff += 2;
		}
		
		let highlight = title_option_sel == TitleOptionSel::NewGm as isize;
		print_ln(&mut buttons.New_game, roff, highlight, &mut sel_loc, screen_sz, d);
		roff += 2;
		
		let highlight = title_option_sel == TitleOptionSel::Exit as isize;
		print_ln(&mut buttons.Exit, roff, highlight, &mut sel_loc, screen_sz, d);
		
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
		
		d.mv(sel_loc.0, sel_loc.1);
		d.refresh();
		d.set_mouse_to_arrow();
		loop {
			key_pressed = d.getch();
			// ignore enter key for a short amount of time
			if key_pressed == '\n' as i32 && (start_time.elapsed().as_millis() as u64) < IGN_KEY_TIME {continue;}
			
			if key_pressed != ERR {break;}
		}
	}
}

