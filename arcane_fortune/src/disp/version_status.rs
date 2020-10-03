extern crate ironnetpro;

use std::process::exit;
use std::env;
use crate::disp_lib::*;
use super::vars::ScreenSz;
use super::{URL, print_centered_logo, DispChars};
use super::logo_vars::{LOGO_HEIGHT, LOGO_WIDTH};
use crate::disp::{Buttons};
use crate::localization::Localization;

const COMMIT_ID: &str = include_str!("../../../.git/refs/heads/v0.3.0");//master");
const IRONNETPRO_COMMIT_ID: &str = COMMIT_ID; //include_str!("../../../../ironnetpro/.git/refs/heads/master");

const COMMIT_LEN: usize = 9;

pub const VERSION_NM: &str = "League of War";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const TARGET: &str = env!("TARGET");
const PROFILE: &str = env!("PROFILE");
const RUSTV: &str = env!("RUSTV");
const OPT_LEVEL: &str = env!("OPT_LEVEL");
const COMPILE_DATE: &str = env!("COMPILE_DATE");

/////////////////////////////////////
// detect arguments, print version (to console)
pub fn show_version_status_console() {
	let args: Vec<String> = env::args().collect();
	if args.len() > 1 {
		let mut commit = String::from(COMMIT_ID);
		let mut ironnetpro_commit = String::from(IRONNETPRO_COMMIT_ID);
		
		commit.truncate(COMMIT_LEN);
		ironnetpro_commit.truncate(COMMIT_LEN);
		
		let w = 50;
		let mut eq = String::new();
		let mut min = String::new();

		for _ in 0..w {
			eq.push('=');
			min.push('-');
		}
		
		let center_txt = |txt: &str| {
			let g = (w - txt.len())/2;
			let mut sp = String::new();
			for _ in 0..g {
				sp.push(' ');
			}
			println!("{}{}", sp, txt);
		};
		
		println!("\n{}\n", eq);
		center_txt("Arcane Fortune");
		println!("\n{}\n", eq);
		
		center_txt(VERSION_NM);
		center_txt(&format!("v{}-{}-{}", VERSION, PROFILE, commit));
		center_txt("");
		center_txt("By Darin Straus");
		center_txt(&format!("Visit {}", URL));
		
		println!("\n{}\n", min);
		
		println!("Licensing: See the included license.txt file.");
		println!("Compiled: {}", COMPILE_DATE);
		println!("Target: {} (opt-level: {})", TARGET, OPT_LEVEL);
		println!("Ironnet Pro: {}-{}", ironnetpro::VERSION, ironnetpro_commit);
		println!("Rust version: {}\n", RUSTV);
		
		exit(0);
	}
}

/////////////////////////
// print version and logo in ncurses/sdl
pub fn show_version_status(screen_sz: ScreenSz, disp_chars: &DispChars,
		buttons: &mut Buttons, l: &Localization, d: &mut DispState) {
	let mut commit = String::from(COMMIT_ID);
	let mut ironnetpro_commit = String::from(IRONNETPRO_COMMIT_ID);
	
	commit.truncate(COMMIT_LEN);
	ironnetpro_commit.truncate(COMMIT_LEN);
	
	const LOGO_ROW_OFFSET: usize = 8;
	let mut row = (print_centered_logo(screen_sz, disp_chars, LOGO_ROW_OFFSET, d) + LOGO_HEIGHT) as i32;
	
	if LOGO_WIDTH >= screen_sz.w {return;}
	let col = ((screen_sz.w - LOGO_WIDTH) / 2) as i32 - 2;
	
	// print top line
	{
		d.mv(row - LOGO_HEIGHT as i32 - 1, col);
		d.addch(disp_chars.ulcorner_char);
		for _ in 0..LOGO_WIDTH {d.addch(disp_chars.hline_char);}
		d.addch(disp_chars.urcorner_char);
	}
	
	macro_rules! clr_ln{() => {
			d.mv(row, col);
			d.addch(disp_chars.vline_char);
			for _ in 0..LOGO_WIDTH {d.addch(' ');}
			d.addch(disp_chars.vline_char);
		};
		($i: expr) => {
			d.mv(row+$i, col);
			d.addch(disp_chars.vline_char);
			for _ in 0..LOGO_WIDTH {d.addch(' ');}
			d.addch(disp_chars.vline_char);
	};};
	
	macro_rules! center_txt{($txt: expr) => {
		clr_ln!();
		d.mv(row, ((screen_sz.w - $txt.len()) / 2) as i32);
		d.addstr($txt);
		row += 1;
	};};
	
	for r in -(LOGO_HEIGHT as i32)..=0 {clr_ln!(r);}
	
	// run a second time to print over the spaces that were just printed
	// (running the first time was to get the row)
	row = (print_centered_logo(screen_sz, disp_chars, LOGO_ROW_OFFSET, d) + LOGO_HEIGHT + 1) as i32;
	
	center_txt!(VERSION_NM);
	center_txt!(&format!("v{}-{}-{}", VERSION, PROFILE, commit));
	center_txt!("");
	center_txt!("By Darin Straus");
	
	// print url
	{
		clr_ln!();
		d.mv(row, ((screen_sz.w - "Visit ".len() - URL.len()) / 2) as i32);
		d.addstr("Visit ");
		d.attron(disp_chars.shortcut_indicator);
		d.addstr(URL);
		d.attroff(disp_chars.shortcut_indicator);
		row += 1;
	}
	
	clr_ln!(); row += 1;
	clr_ln!(); row += 1;
	clr_ln!(); row += 1;

	////////// print next 3 lines in a box, spaced left and right text
	let window_w = (LOGO_WIDTH - 35) as i32;
	let col_sub = (screen_sz.w as i32 - window_w)/2;
	
	macro_rules! lr_txt{($l_txt: expr, $r_txt: expr) => {
		clr_ln!();
		d.mv(row,col_sub);
		d.addstr($l_txt);
		d.mv(row, col_sub + window_w - $r_txt.len() as i32);
		d.addstr($r_txt);
		row += 1;
	};};
	
	lr_txt!("Licensing:", "See the included license.txt file.");
	lr_txt!("Compiled:", COMPILE_DATE);
	lr_txt!("Target:", &format!("{} (opt-level: {})", TARGET, OPT_LEVEL));
	lr_txt!("Ironnet Pro:", &format!("{}-{}", ironnetpro::VERSION, ironnetpro_commit));
	lr_txt!("Rust version:", RUSTV);
	
	///////// end
	clr_ln!(); row += 1;
	
	clr_ln!();
	{ // Esc to close
		let button = &mut buttons.Esc_to_close;
		d.mv(row, ((screen_sz.w - button.print_txt(l).len()) / 2) as i32);
		button.print(None, l, d);
		row += 1;
	}

	// print bottom line
	{
		d.mv(row,col);
		d.addch(disp_chars.llcorner_char);
		for _ in 0..LOGO_WIDTH {d.addch(disp_chars.hline_char);}
		d.addch(disp_chars.lrcorner_char);
	}
}

