use std::os::raw::c_int;
use std::{thread, time};
extern crate sdl2_lib; use sdl2_lib::*;
const FRAME_PAUSE: u64 = 20;
pub const CWHITE: CInd = 0;
pub const CYELLOW: CInd = 3;

/*pub fn draw_chess_board(d: &mut DispState) {
	let d_area = d.get_viewport();
	
	for row in 0..8 {
		let mut x = row % 2;
		for column in 0..(4+(row%2)) {
			d.set_draw_color(25, 25, 25, 0xFF);
			
			let w = d_area.w/8;
			let h = d_area.h/8;
			
			let rect = SDL_Rect {
				w, h,
				x: x * w,
				y: row * h
			};
			x += 2;
			d.fill_rect(rect);
		}
	}
}*/

pub fn main() {
	println!("test");
	setup_disp_lib();
	
	loop {}
}

