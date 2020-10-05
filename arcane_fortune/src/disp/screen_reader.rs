use std::env;
use crate::disp_lib::{DispState, endwin};
use super::cursor_pos;

pub fn screen_reader_mode() -> bool {
	let args: Vec<String> = env::args().collect();
	for arg in args.iter().skip(1) {
		if arg == "screen_reader_mode" {return true;}
	}
	false
}

// to be tabbed through when ui_mode is UIMode::TextTab
#[derive(Debug)]
pub struct TxtList {
	pub bottom: Vec<(i32, i32)>,
	pub right: Vec<(i32, i32)>
}

impl TxtList {
	pub fn new() -> Self {
		Self {
			bottom: Vec::new(),
			right: Vec::new()
		}
	}
	
	pub fn clear(&mut self) {
		self.bottom.clear();
		self.right.clear();
	}
	
	pub fn add_r(&mut self, d: &DispState) {
		let curs = cursor_pos(d);
		self.right.push((curs.y as i32, curs.x as i32));
	}
	
	pub fn add_b(&mut self, d: &DispState) {
		let curs = cursor_pos(d);
		//printlnq!("{} {}", curs.y, curs.x);
		self.bottom.push((curs.y as i32, curs.x as i32));
	}
}

