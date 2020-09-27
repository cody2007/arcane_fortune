#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(target_os = "windows")]
pub mod pdcurses;
#[cfg(target_os = "windows")]
mod direct_pdcurses;
#[cfg(target_os = "windows")]
pub use pdcurses::*;

#[cfg(not(target_os = "windows"))]
pub mod ncurses;
#[cfg(not(target_os = "windows"))]
mod direct_ncurses;
#[cfg(not(target_os = "windows"))]
pub use ncurses::*;

// see ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/curses.h
// /usr/include/ncurses/ncurses.h

pub const KEY_DOWN: i32 = 0x102;
pub const KEY_UP: i32 = 0x103;
pub const KEY_LEFT: i32 = 0x104;
pub const KEY_RIGHT: i32 = 0x105;
//pub const KEY_SLEFT: i32 = 0x189;
//pub const KEY_SRIGHT: i32 = 0x192;
pub const KEY_ENTER: i32 = 0x157;

pub const KEY_HOME: i32 = 0x106;
//pub const KEY_END: i32 = 0x166;
//pub const KEY_END: i32 = 0550;

pub const KEY_BACKSPACE: i32 = 0x107;
//pub const KEY_BACKSPACE: i32 = 127;
pub const KEY_DC: i32 = 0x14a;

pub const KEY_ESC: i32 = 27;

pub const COLOR_BLACK: i16 = 0;
pub const COLOR_WHITE: i16 = 7;

use std::os::raw::{c_int, c_short};
pub type mmask_t = u32;//c_uint;

#[derive(Debug)]
#[repr(C)]
pub struct MEVENT {
	pub id: c_short, // used to distinguish multiple devices
	pub x: c_int,
	pub y: c_int,
	pub z: c_int,
	pub bstate: mmask_t // button state bits
}

pub fn shift_pressed() -> bool {false} // ncurses has no way of detecting shift pressed in isolation w/o other keys

pub fn rbutton_clicked(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON3_CLICKED) != 0
	}else{
		false
	}
}

pub fn rbutton_released(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON3_RELEASED) != 0
	}else{
		false
	}
}

pub fn rbutton_pressed(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON3_PRESSED) != 0
	}else{
		false
	}
}

pub fn lbutton_clicked(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON1_CLICKED) != 0
	}else{
		false
	}
}

pub fn lbutton_released(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON1_RELEASED) != 0
	}else{
		false
	}
}

pub fn lbutton_pressed(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON1_PRESSED) != 0
	}else{
		false
	}
}

pub fn dragging(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & REPORT_MOUSE_POSITION) != 0
	}else{
		false
	}
}

pub fn scroll_up(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON4_PRESSED) != 0
	}else{
		false
	}
}

pub fn scroll_down(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate, ..}) = &mouse_event {
		(bstate & BUTTON5_PRESSED) != 0
	}else{
		false
	}
}

pub struct DispState {}

// not supported...
impl DispState {
	pub fn mouse_pos(&self) -> Option<(i32, i32)> {None}
	pub fn set_mouse_to_arrow(&mut self) {}
	pub fn set_mouse_to_hand(&mut self) {}
	pub fn set_mouse_to_crosshair(&mut self) {}
	pub fn toggle_fullscreen(&mut self) {}
	pub fn inc_font_sz(&mut self) {}
	pub fn dec_font_sz(&mut self) {}
}

