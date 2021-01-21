/*
    Copyright © 2013 Free Software Foundation, Inc
    See licensing in LICENSE file

    Derivation of ncurses Rust module
    Original author: Jesse 'Jeaye' Wilkerson
*/

// /home/tapa/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses/include/ncursesw/curses.h
#![allow(dead_code)]
use std::ptr;
use std::ffi::CString;
use std::convert::TryInto;
use super::*;
use super::direct_ncurses;
use crate::renderer::*;
pub use super::direct_ncurses::{CInt, CShort, chtype,
	WINDOW, mousemask};

pub const COLOR_RED: i16 = 1;
pub const COLOR_GREEN: i16 = 2;
pub const COLOR_BLUE: i16 = 4;

pub const COLOR_CYAN: i16 = 6;
pub const COLOR_MAGENTA: i16 = 5;
pub const COLOR_YELLOW: i16 = 3;

pub const KEY_MOUSE: i32 = 0o631;

macro_rules! wrap{($nm:ident) => {pub fn $nm(&self) {unsafe{direct_ncurses::$nm();}}}}
macro_rules! wrap_sep{($nm:ident) => {pub fn $nm() {unsafe{direct_ncurses::$nm();}}}}

wrap_sep!(initscr);
wrap_sep!(flushinp);
wrap_sep!(noecho);
wrap_sep!(start_color);
wrap_sep!(endwin);

#[derive(Debug, Clone, Copy)]
pub enum CURSOR_VISIBILITY{CURSOR_INVISIBLE = 0, CURSOR_VISIBLE, CURSOR_VERY_VISIBLE}

macro_rules! wrap_try_into{($nm: ident) => {
	pub fn $nm<T: TryInto<chtype>>(&mut self, a: T) {
		unsafe {
			if let Ok(val) = a.try_into() {
				direct_ncurses::$nm(val);
			}else{
				panicq!("could not convert to chtype");
			}
		}
	}
};}

impl Renderer {
	wrap!(clrtoeol);
	wrap!(refresh);
	wrap!(clear);
	
	pub fn init_pair(&mut self, a: CShort, b: CShort, c: CShort) {unsafe{direct_ncurses::init_pair(a,b,c);}}
	pub fn mv<Y: TryInto<CInt>, X: TryInto<CInt>>(&mut self, y: Y, x: X) {
		if let Ok(y) = y.try_into() {
		if let Ok(x) = x.try_into() {
			unsafe{direct_ncurses::mv(y,x);}
			return;
		}}
		panicq!("could not convert arguments into integers");
	}
	wrap_try_into!(addch);
	//wrap_try_into!(attroff);
	wrap_try_into!(attron);
	
	pub fn attroff<T: TryInto<chtype>>(&mut self, a: T) {
		unsafe {
			if let Ok(val) = a.try_into() {
				direct_ncurses::attroff(val);
			}else{
				panicq!("could not convert to chtype");
			}
		}
		#[cfg(not(target_env="musl"))]
		self.attron(COLOR_PAIR(CWHITE));
	}

	pub fn getch(&self) -> CInt {unsafe{direct_ncurses::getch()}}
	pub fn inch(&self) -> chtype {unsafe{direct_ncurses::inch()}}
	pub fn curs_set(&mut self, mode: CURSOR_VISIBILITY) -> CURSOR_VISIBILITY {
		match unsafe{direct_ncurses::curs_set(mode as CInt)} {
			0 => CURSOR_VISIBILITY::CURSOR_INVISIBLE,
			1 => CURSOR_VISIBILITY::CURSOR_VISIBLE,
			2 => CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE,
			_ => {panicq!("could not set cursor visiblity");}
		}
	}
	pub fn timeout(&mut self, time_out: CInt) {unsafe{direct_ncurses::timeout(time_out);}}
	pub fn addnstr(&mut self, txt: &str, len: CInt) {
		unsafe {direct_ncurses::addnstr(CString::new(txt).unwrap().as_c_str().as_ptr(), len);}
	}
	
	pub fn addstr(&mut self, txt: &str) {
		unsafe {direct_ncurses::addstr(CString::new(txt).unwrap().as_c_str().as_ptr());}
	}
	
	pub fn getmaxyx(&self, win: direct_ncurses::WINDOW, y: &mut i32, x: &mut i32)
		{ unsafe { *y = direct_ncurses::getmaxy(win); *x = direct_ncurses::getmaxx(win) } }
	
	pub fn getyx(&self, win: WINDOW, y: &mut i32, x: &mut i32)
		{ unsafe { *y = direct_ncurses::getcury(win); *x = direct_ncurses::getcurx(win); } }
	
	pub fn getmouse(&self, key_pressed: CInt) -> Option<MEVENT> {
		let mut event = MEVENT {
			id: 0,
			x: 0,
			y: 0,
			z: 0,
			bstate: 0
		};
		
		if key_pressed == KEY_MOUSE && unsafe {direct_ncurses::getmouse(&mut event)} == OK {
			Some(event)
		}else{
			None
		}
	}
}

pub fn COLOR_PAIR(pair: CInt) -> chtype {unsafe{direct_ncurses::COLOR_PAIR(pair) as chtype}}
pub fn stdscr() -> direct_ncurses::WINDOW {unsafe{direct_ncurses::stdscr}}
pub fn can_change_color() -> bool {unsafe{direct_ncurses::can_change_color()}}
pub fn has_colors() -> bool {unsafe{direct_ncurses::has_colors()}}
fn keypad(w: direct_ncurses::WINDOW, s: bool) {unsafe{direct_ncurses::keypad(w,s);}}

#[allow(dead_code)]
fn add_wch(ch: [chtype; 5]) {
	unsafe {
		let f = direct_ncurses::cchar_t {attr: 0, ch};
		direct_ncurses::add_wch(&f);
	}
}

const NCURSES_ATTR_SHIFT: u32 = 8;
pub const ERR: i32 = -1;
pub const OK: i32 = 0;

pub const fn NCURSES_BITS(m: u32, s: u32) -> u32 {m << (s + NCURSES_ATTR_SHIFT) as usize}
pub const fn NCURSES_MOUSE_MASK(b: mmask_t, m: mmask_t) -> mmask_t {m << ((b-1)*5)}

const NCURSES_BUTTON_RELEASED: mmask_t = 0o1;
const NCURSES_BUTTON_PRESSED: mmask_t =  0o2;
const NCURSES_BUTTON_CLICKED: mmask_t =  0o4;
const NCURSES_DOUBLE_CLICKED: mmask_t =  0o10;
const NCURSES_TRIPLE_CLICKED: mmask_t =  0o20;
const NCURSES_RESERVED_EVENT: mmask_t =  0o40;

pub const BUTTON1_RELEASED: mmask_t = NCURSES_MOUSE_MASK(1, NCURSES_BUTTON_RELEASED);
pub const BUTTON1_PRESSED: mmask_t = NCURSES_MOUSE_MASK(1, NCURSES_BUTTON_PRESSED);
pub const BUTTON1_CLICKED: mmask_t = NCURSES_MOUSE_MASK(1, NCURSES_BUTTON_CLICKED);
pub const BUTTON1_DOUBLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(1, NCURSES_DOUBLE_CLICKED);
pub const BUTTON1_TRIPLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(1, NCURSES_TRIPLE_CLICKED);

pub const BUTTON2_RELEASED: mmask_t = NCURSES_MOUSE_MASK(2, NCURSES_BUTTON_RELEASED);
pub const BUTTON2_PRESSED: mmask_t = NCURSES_MOUSE_MASK(2, NCURSES_BUTTON_PRESSED);
pub const BUTTON2_CLICKED: mmask_t = NCURSES_MOUSE_MASK(2, NCURSES_BUTTON_CLICKED);
pub const BUTTON2_DOUBLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(2, NCURSES_DOUBLE_CLICKED);
pub const BUTTON2_TRIPLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(2, NCURSES_TRIPLE_CLICKED);

pub const BUTTON3_RELEASED: mmask_t = NCURSES_MOUSE_MASK(3, NCURSES_BUTTON_RELEASED);
pub const BUTTON3_PRESSED: mmask_t = NCURSES_MOUSE_MASK(3, NCURSES_BUTTON_PRESSED);
pub const BUTTON3_CLICKED: mmask_t = NCURSES_MOUSE_MASK(3, NCURSES_BUTTON_CLICKED);
pub const BUTTON3_DOUBLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(3, NCURSES_DOUBLE_CLICKED);
pub const BUTTON3_TRIPLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(3, NCURSES_TRIPLE_CLICKED);

pub const BUTTON4_RELEASED: mmask_t = NCURSES_MOUSE_MASK(4, NCURSES_BUTTON_RELEASED);
pub const BUTTON4_PRESSED: mmask_t = NCURSES_MOUSE_MASK(4, NCURSES_BUTTON_PRESSED);
pub const BUTTON4_CLICKED: mmask_t = NCURSES_MOUSE_MASK(4, NCURSES_BUTTON_CLICKED);
pub const BUTTON4_DOUBLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(4, NCURSES_DOUBLE_CLICKED);
pub const BUTTON4_TRIPLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(4, NCURSES_TRIPLE_CLICKED);

pub const BUTTON5_RELEASED: mmask_t = NCURSES_MOUSE_MASK(5, NCURSES_BUTTON_RELEASED);
pub const BUTTON5_PRESSED: mmask_t = NCURSES_MOUSE_MASK(5, NCURSES_BUTTON_PRESSED);
pub const BUTTON5_CLICKED: mmask_t = NCURSES_MOUSE_MASK(5, NCURSES_BUTTON_CLICKED);
pub const BUTTON5_DOUBLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(5, NCURSES_DOUBLE_CLICKED);
pub const BUTTON5_TRIPLE_CLICKED: mmask_t = NCURSES_MOUSE_MASK(5, NCURSES_TRIPLE_CLICKED);

pub const REPORT_MOUSE_POSITION: mmask_t = NCURSES_MOUSE_MASK(6, 0o10);
pub const ALL_MOUSE_EVENTS: mmask_t = REPORT_MOUSE_POSITION - 1;

pub const fn A_CHARTEXT() -> chtype {(NCURSES_BITS(1, 0) - 1) as chtype}
pub const fn A_UNDERLINE() -> chtype {NCURSES_BITS(1, 9) as chtype}
pub const fn A_REVERSE() -> chtype {NCURSES_BITS(1, 10) as chtype}
pub const fn A_DIM() -> chtype {NCURSES_BITS(1, 12) as chtype}
pub const fn A_BOLD() -> chtype {NCURSES_BITS(1, 13) as chtype}

pub fn NCURSES_ACS(c: char) -> chtype {
    unsafe {*direct_ncurses::acs_map().offset((c as u8) as isize) as chtype}
}

pub fn ACS_ULCORNER() -> chtype {NCURSES_ACS('l')} 
pub fn ACS_LLCORNER() -> chtype {NCURSES_ACS('m')}
pub fn ACS_URCORNER() -> chtype {NCURSES_ACS('k')}
pub fn ACS_LRCORNER() -> chtype {NCURSES_ACS('j')}
pub fn ACS_HLINE() -> chtype {NCURSES_ACS('q')}
pub fn ACS_VLINE() -> chtype {NCURSES_ACS('x')}
pub fn ACS_CKBOARD() -> chtype {NCURSES_ACS('a')}

//use std::io::{self, Write};
pub fn setup_disp_lib() -> Renderer {
	unsafe{
		direct_ncurses::setlocale(direct_ncurses::LC_ALL, &direct_ncurses::LC_VARS);
		initscr();
		keypad(stdscr(), true); // to allow the arrow keys, for example, as inputs through getch()
		direct_ncurses::timeout(MAX_DELAY_FRAMES);
		noecho();
		start_color();
		direct_ncurses::set_escdelay(MAX_DELAY_FRAMES); // in milliseconds
		mousemask(BUTTON1_PRESSED | BUTTON1_RELEASED | BUTTON1_CLICKED |
			    BUTTON3_PRESSED | BUTTON3_RELEASED | BUTTON3_CLICKED |
			    BUTTON4_PRESSED | BUTTON5_PRESSED |
			    REPORT_MOUSE_POSITION, ptr::null_mut());
		
		// to enable mouse reporting with gnome: (seems to work fine in Konsole)
		//printf("\033[?1003h\n"); // https://gist.github.com/sylt/93d3f7b77e7f3a881603 (Accessed September 10, 2020)
		/*io::stdout().write_all(&[0o33]);
		let txt = "[?1003h\n";
		io::stdout().write_all(txt.as_bytes());*/
		
		// ^ note the problem with the above is that gnome starts 
		//   reporting button1 pressed, released, and clicked events
		//   when the mouse is moved
	}
	Renderer {}
}

pub fn COLOR_PAIRS() -> CInt { unsafe{direct_ncurses::COLOR_PAIRS}}

