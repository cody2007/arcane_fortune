use std::ptr;
use std::ffi::CString;
use std::convert::TryInto;
use super::*;
use super::direct_pdcurses;
use crate::disp_lib::MAX_DELAY_FRAMES;
pub use super::direct_pdcurses::{CInt, CShort, chtype,
	WINDOW, mousemask};

// see ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/curses.h
pub const COLOR_RED: i16 = 4;
pub const COLOR_GREEN: i16 = 2;
pub const COLOR_BLUE: i16 = 1;

pub const COLOR_CYAN: i16 = (COLOR_BLUE | COLOR_GREEN);
pub const COLOR_MAGENTA: i16 = (COLOR_RED | COLOR_BLUE);
pub const COLOR_YELLOW: i16 = (COLOR_RED | COLOR_GREEN);

pub const KEY_MOUSE: i32 = 0x21b;

macro_rules! wrap{($nm:ident) => {pub fn $nm(&self) {unsafe{direct_pdcurses::$nm();}}}}
macro_rules! wrap_sep{($nm:ident) => {pub fn $nm() {unsafe{direct_pdcurses::$nm();}}}}

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
				direct_pdcurses::$nm(val);
			}else{
				panicq!("could not convert to chtype");
			}
		}
	}
};}

impl DispState {
	wrap!(clrtoeol);
	wrap!(refresh);
	wrap!(clear);
	
	pub fn init_pair(&mut self, a: CShort, b: CShort, c: CShort) {unsafe{direct_pdcurses::init_pair(a,b,c);}}
	pub fn mv(&mut self, y: CInt, x: CInt) {unsafe{direct_pdcurses::mv(y,x);}}
	wrap_try_into!(addch);
	wrap_try_into!(attroff);
	wrap_try_into!(attron);
	pub fn getch(&self) -> CInt {unsafe{direct_pdcurses::wgetch(stdscr())}}
	pub fn inch(&self) -> chtype {unsafe{direct_pdcurses::inch()}}
	pub fn curs_set(&mut self, mode: CURSOR_VISIBILITY) -> CURSOR_VISIBILITY {
		match unsafe{direct_pdcurses::curs_set(mode as CInt)} {
			0 => CURSOR_VISIBILITY::CURSOR_INVISIBLE,
			1 => CURSOR_VISIBILITY::CURSOR_VISIBLE,
			2 => CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE,
			_ => {panicq!("could not set cursor visiblity");}
		}
	}
	pub fn timeout(&mut self, time_out: CInt) {unsafe{direct_pdcurses::timeout(time_out);}}
	pub fn addnstr(&mut self, txt: &str, len: CInt) {
		unsafe {direct_pdcurses::addnstr(CString::new(txt).unwrap().as_c_str().as_ptr(), len);}
	}
	
	pub fn addstr(&self, txt: &str) {
		unsafe {direct_pdcurses::addstr(CString::new(txt).unwrap().as_c_str().as_ptr());}
	}
	
	pub fn getmaxyx(&self, win: direct_pdcurses::WINDOW, y: &mut i32, x: &mut i32)
		{ unsafe { *y = direct_pdcurses::getmaxy(win); *x = direct_pdcurses::getmaxx(win) } }
	
	pub fn getyx(&self, win: WINDOW, y: &mut i32, x: &mut i32)
		{ unsafe { *y = direct_pdcurses::getcury(win); *x = direct_pdcurses::getcurx(win); } }
	
	pub fn getmouse(&self, key_pressed: CInt) -> Option<MEVENT> {
		let mut event = MEVENT {
			id: 0,
			x: 0,
			y: 0,
			z: 0,
			bstate: 0
		};
		
		if key_pressed == KEY_MOUSE && unsafe {direct_pdcurses::nc_getmouse(&mut event)} == OK {
			Some(event)
		}else{
			None
		}
	}
}

pub fn stdscr() -> direct_pdcurses::WINDOW {unsafe{direct_pdcurses::stdscr}}
pub fn can_change_color() -> bool {unsafe{direct_pdcurses::can_change_color()}}
pub fn has_colors() -> bool {unsafe{direct_pdcurses::has_colors()}}
fn keypad(w: direct_pdcurses::WINDOW, s: bool) {unsafe{direct_pdcurses::keypad(w,s);}}

fn add_wch(ch: [chtype; 5]) {
	unsafe {
		let f = direct_pdcurses::cchar_t {attr: 0, ch};
		direct_pdcurses::add_wch(&f);
	}
}

pub const ERR: i32 = -1;
pub const OK: i32 = 0;

pub const BUTTON1_RELEASED: mmask_t = 0x1;
pub const BUTTON1_PRESSED: mmask_t = 0x2;
pub const BUTTON1_CLICKED: mmask_t = 0x4;
pub const BUTTON1_MOVED: mmask_t = 0x100;

pub const BUTTON2_RELEASED: mmask_t = 0x20;
pub const BUTTON2_PRESSED: mmask_t = 0x40;
pub const BUTTON2_CLICKED: mmask_t = 0x80;

pub const BUTTON3_RELEASED: mmask_t = 0x400;
pub const BUTTON3_PRESSED: mmask_t = 0x800;
pub const BUTTON3_CLICKED: mmask_t = 0x1000;
pub const BUTTON3_MOVED: mmask_t = 0x4000;

pub const BUTTON4_PRESSED: mmask_t = 0x00010000;
pub const BUTTON5_PRESSED: mmask_t = 0x00200000;

pub const REPORT_MOUSE_POSITION: mmask_t = 0x20000000;

pub const fn A_CHARTEXT() -> chtype {0x0000ffff as chtype}
pub const fn A_UNDERLINE() -> chtype {0x00100000 as chtype}
pub const fn A_REVERSE() -> chtype {0x00200000 as chtype}
pub const fn A_DIM() -> chtype {0 as chtype}
pub const fn A_BOLD() -> chtype {0x00800000 as chtype}

const A_COLOR: chtype = 0xff000000;
const A_ALTCHARSET: chtype = 0x00010000;

const PDC_COLOR_SHIFT: usize = 24;
pub const fn COLOR_PAIR(pair: i32) -> chtype {((pair as chtype) << PDC_COLOR_SHIFT) & A_COLOR}


pub const fn PDC_ACS(c: char) -> chtype {
    ((c as u8) as isize) as chtype | A_ALTCHARSET
}

pub fn ACS_ULCORNER() -> chtype {PDC_ACS('l')} 
pub fn ACS_LLCORNER() -> chtype {PDC_ACS('m')}
pub fn ACS_URCORNER() -> chtype {PDC_ACS('k')}
pub fn ACS_LRCORNER() -> chtype {PDC_ACS('j')}
pub fn ACS_HLINE() -> chtype {PDC_ACS('q')}
pub fn ACS_VLINE() -> chtype {PDC_ACS('x')}
pub fn ACS_CKBOARD() -> chtype {PDC_ACS('a')}

pub fn setup_disp_lib() -> DispState {
	unsafe{
		direct_pdcurses::setlocale(direct_pdcurses::LC_ALL, &direct_pdcurses::LC_VARS);
		initscr();
		keypad(stdscr(), true); // to allow the arrow keys, for example, as inputs through getch()
		direct_pdcurses::timeout(MAX_DELAY_FRAMES);
		noecho();
		start_color();
		
		mousemask(BUTTON1_PRESSED | BUTTON1_RELEASED | BUTTON1_CLICKED | BUTTON1_MOVED |
			    BUTTON3_PRESSED | BUTTON3_RELEASED | BUTTON3_CLICKED | BUTTON3_MOVED |
			    BUTTON4_PRESSED | BUTTON5_PRESSED |
			    REPORT_MOUSE_POSITION, ptr::null_mut());
	}
	DispState {}
}

pub fn COLOR_PAIRS() -> CInt { unsafe{direct_pdcurses::COLOR_PAIRS}}

