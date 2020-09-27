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
	let mut d = setup_disp_lib();
	
	let img = Texture::from_img(&d.txt_state.renderer, "/home/tapa/Pictures/komi_compare.png");
	let mut rect = SDL_Rect {w: 200, h: 200, x: 0, y: 0};
	
	d.init_pair(CWHITE as i16, COLOR_WHITE, COLOR_BLACK);
	d.init_pair(CYELLOW as i16, COLOR_RED, COLOR_GREEN);
	d.attron(COLOR_PAIR(CYELLOW));
	d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
	
	let mut ch = ' ';
	loop {
		if ch =='A' {
			rect.x -= 10;
		}else if ch == 'D' {
			rect.x += 10;
		}else if ch == 'S' {
			rect.y += 10;
		}else if ch == 'W' {
			rect.y -= 10;
		}
		
		//draw_chess_board(&mut d);
		d.txt_state.renderer.copy(&img, &rect);
		
		d.mv(0,0);
		d.attron(A_UNDERLINE());
		d.addstr(&format!("Arcane Fortune test display {} {}", rect.x, rect.y));
		d.attroff(A_UNDERLINE());
		d.attron(A_REVERSE() | A_UNDERLINE());
		d.addstr("additional text");
		d.attroff(A_REVERSE() | A_UNDERLINE());
		d.mv(1,0);
		d.addstr("additional text2");
		d.mv(2,0);
		for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-=_+.\\;:[]{}|`~".chars() {
			d.addch(c as chtype);
		}
		d.addch('a' as chtype | COLOR_PAIR(CYELLOW));
		
		let mut h = 0;
		let mut w = 0;
		d.getmaxyx(stdscr(), &mut h, &mut w);
		d.mv(h-1, w-1);
		d.addch('@' as chtype);
		
		d.refresh();
		ch = d.getch() as u8 as char;
	}
	sdl_quit();
}

