use std::env;
use crate::renderer::Renderer;
use super::cursor_pos;

pub fn screen_reader_mode() -> bool {
	let args: Vec<String> = env::args().collect();
	for arg in args.iter().skip(1) {
		if arg == "screen_reader_mode" {return true;}
	}
	false
}

// to be tabbed through when ui_mode is UIMode::TextTab (row,col)
#[derive(Debug)]
pub struct TxtList {
	pub bottom: Vec<(i32, i32)>,
	pub right: Vec<(i32, i32)>,
	pub window: Vec<(i32, i32)>
}

impl TxtList {
	pub fn new() -> Self {
		Self {
			bottom: Vec::new(),
			right: Vec::new(),
			window: Vec::new()
		}
	}
	
	pub fn clear(&mut self) {
		self.bottom.clear();
		self.right.clear();
		self.window.clear();
	}
	
	pub fn add_r(&mut self, r: &Renderer) {
		let curs = cursor_pos(r);
		self.right.push((curs.y as i32, curs.x as i32));
	}
	
	pub fn add_b(&mut self, r: &Renderer) {
		let curs = cursor_pos(r);
		//printlnq!("{} {}", curs.y, curs.x);
		self.bottom.push((curs.y as i32, curs.x as i32));
	}
	
	pub fn add_w(&mut self, r: &Renderer) {
		let curs = cursor_pos(r);
		self.window.push((curs.y as i32, curs.x as i32));
	}

}

