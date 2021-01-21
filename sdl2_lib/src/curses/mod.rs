#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::convert::TryInto;
use std::{thread, time};
use std::time::{Instant};
use std::path::Path;
//use std::fs::File;
//use std::io::Write;
use super::*;
pub mod constants; pub use constants::*;

const PAD_DISPLAY_PX: i32 = 3;

const FONT_FILE: &str = "fonts/OxygenMono-Regular.ttf";
const ICON_FILE: &str = "icon.ico";

const FONT_CHG_STEP: c_int = 1;
const FONT_SZ_MAX: c_int = 50;

const CHAR_BUF_SZ: usize = 1000;

// returns true if the flag (`bit`) is set in the `attribute` used with attr/on/attroff/addch
pub fn attr_set(attribute: chtype, bit: chtype) -> bool {
	(attribute & (1 << bit)) != 0
}

// returns the index of the color pair from the `attribute` used with attron/attroff/addch
pub fn color_pair_ind_frm_attr(attribute: chtype) -> usize {
	(attribute >> A_COLOR_PAIR_OFFSET) as usize
}

// returns the value that can be `OR`d into the `attribute` used with attron/attroff/addch
pub fn COLOR_PAIR(pair: CInt) -> chtype {
	((pair as chtype) << A_COLOR_PAIR_OFFSET) | (1 << A_COLOR_PAIR_FLAG_OFFSET)
}

struct ColorPair {
	fg: SDL_Color, // foreground
	bg: SDL_Color, // background
	
	// font caches
	font_cache_fg: Vec<Option<Texture>>, // rendered as fg color
	font_cache_bg: Vec<Option<Texture>> // rendered as bg color for attron(A_INVERT)
}

impl ColorPair {
	pub fn new(fg: SDL_Color, bg: SDL_Color) -> Self {
		const FONT_CACHE_SZ: usize = 256; // non-unicode (2^8 = 256)
		
		let mut font_cache_fg = Vec::with_capacity(FONT_CACHE_SZ);
		let mut font_cache_bg = Vec::with_capacity(FONT_CACHE_SZ);
		
		for _ in 0..FONT_CACHE_SZ {
			font_cache_fg.push(None);
			font_cache_bg.push(None);
		}
		
		ColorPair {
			fg, bg,
			font_cache_fg,
			font_cache_bg
		}
	}
	
	pub fn texture_from_cache(&mut self, ch: char, font: &Font, invert: bool, sdl_renderer: &SDLRenderer) -> &Texture {
		macro_rules! get_or_set_cache{($ch: expr, $cache: expr, $color: expr) => {
			let font_cache_entry = &mut $cache[$ch as usize];
			return if let Some(texture) = font_cache_entry {
				texture
			}else{
				*font_cache_entry = Some(
					Texture::from_font(sdl_renderer, font, &String::from($ch), $color)
				);
				font_cache_entry.as_ref().unwrap()
			};
		};};
		
		if !invert {
			get_or_set_cache!(ch, self.font_cache_fg, self.fg);
		}else{
			get_or_set_cache!(ch, self.font_cache_bg, self.bg);
		}
	}
	
	pub fn invalidate_cache(&mut self) {
		for (fg, bg) in self.font_cache_fg.iter_mut()
				.zip(self.font_cache_bg.iter_mut()) {
			*fg = None;
			*bg = None;
		}
	}
}

pub struct TxtState {
	pub sdl_renderer: SDLRenderer,
	
	x: c_int,
	y: c_int,
	
	font: Font,
	font_ch_sz: Size, // pixel size of the font rendered at the current font size setting
	
	//////////////////////////////
	// text character attributes
	color_pair_ind: usize,
	
	reverse: bool, // reverse fg & bg
	ckboard: bool,
	underline: bool,
	special_char: SpecialChar,
	/////////////////////////////
	
	char_buf: Vec<chtype>, // chars on the screen (saved as a raveled square vector)
	char_buf_w: usize, // width of the square ^
}

impl TxtState {
	fn current_attr(&self, ch: char) -> chtype {
		let mut attr = match self.special_char {
			SpecialChar::None => ch as chtype,
			SpecialChar::VLine => {1 << A_VLINE_OFFSET}
			SpecialChar::HLine => {1 << A_HLINE_OFFSET}
			SpecialChar::LLCorner => {1 << A_LLCORNER_OFFSET}
			SpecialChar::LRCorner => {1 << A_LRCORNER_OFFSET}
			SpecialChar::URCorner => {1 << A_URCORNER_OFFSET}
			SpecialChar::ULCorner => {1 << A_ULCORNER_OFFSET}
		};
		
		if self.reverse {
			attr |= 1 << A_REVERSE_OFFSET;
		}
		
		if self.ckboard {
			attr |= 1 << A_CKBOARD_OFFSET;
		}
		
		attr |= COLOR_PAIR(self.color_pair_ind as i32);
		
		attr
	}
	
	// prints a single char
	fn print_char(&mut self, ch: char, color_pair: &mut ColorPair) {
		if ch == '\n' {
			self.y += 1;
			self.x = 0;
			return;
		}
		
		let (invert, fg, mut bg) = if (!self.reverse) && (!self.ckboard) {
			(false, color_pair.fg, color_pair.bg)
		}else{
			(true, color_pair.bg, color_pair.fg)
		};
		
		let w = self.font_ch_sz.w;
		let h = self.font_ch_sz.h;
		
		let rect = SDL_Rect {
			w, h,
			x: self.x*w + PAD_DISPLAY_PX,
			y: self.y*h + PAD_DISPLAY_PX
		};
		
		{ // background
			if self.ckboard {
				bg.r /= 2;
				bg.b /= 2;
				bg.g /= 2;
			}
			
			self.sdl_renderer.set_draw_color(bg.r, bg.g, bg.b, bg.a);
			self.sdl_renderer.fill_rect(rect);
		}
		
		// foreground
		macro_rules! set_fg{() => {
			self.sdl_renderer.set_draw_color(fg.r, fg.g, fg.b, fg.a);
		};};
		
		let half = |val| { // also takes the ceil
			((val as f32)/2.).ceil() as c_int
		};
		
		match self.special_char {
			SpecialChar::None => {
				if ch != ' ' {
					let font_texture = color_pair.texture_from_cache(ch, &self.font, invert, &self.sdl_renderer);
					self.sdl_renderer.copy(&font_texture, &rect);
					
					// underline
					if self.underline {set_fg!();
						self.sdl_renderer.fill_rect(SDL_Rect {
							w,
							h: 1,
							x: self.x*w + PAD_DISPLAY_PX,
							y: self.y*h + PAD_DISPLAY_PX + h-3
						});
					}
				}
			} SpecialChar::VLine => {set_fg!();
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: 1,
					h,
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX
				});
			} SpecialChar::HLine => {set_fg!();
				self.sdl_renderer.fill_rect(SDL_Rect {
					w,
					h: 1,
					x: self.x*w + PAD_DISPLAY_PX,
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
			} SpecialChar::LLCorner => {set_fg!();
				// h -
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: half(w),
					h: 1,
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
				
				// v |
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: 1,
					h: half(h),
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX
				});
			} SpecialChar::LRCorner => {set_fg!();
				// h -
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: half(w),
					h: 1,
					x: self.x*w + PAD_DISPLAY_PX,
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
				
				// v |
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: 1,
					h: half(h),
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX
				});
			} SpecialChar::URCorner => {set_fg!();
				// h -
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: half(w),
					h: 1,
					x: self.x*w + PAD_DISPLAY_PX,
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
				
				// v |
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: 1,
					h: half(h),
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
			} SpecialChar::ULCorner => {set_fg!();
				// h -
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: half(w),
					h: 1,
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
				
				// v |
				self.sdl_renderer.fill_rect(SDL_Rect {
					w: 1,
					h: half(h),
					x: self.x*w + PAD_DISPLAY_PX + half(w),
					y: self.y*h + PAD_DISPLAY_PX + half(h)
				});
			}
		}
		
		{ // update the screen buffer (for inch())
			let ind = self.y as usize * self.char_buf_w + self.x as usize;
			self.char_buf[ind] = self.current_attr(ch);
		}
		self.x += 1;
	}
}

pub struct Cursors {
	arrow: Cursor,
	hand: Cursor,
	crosshair: Cursor
}

pub struct Renderer {
	pub txt_state: TxtState,
	fullscreen: bool,
	window: Window,
	key_timeout: CInt, // milliseconds
	color_pairs: Vec<ColorPair>,
	
	cursor_visibility: CURSOR_VISIBILITY,
	mouse_event: Option<MEVENT>,
	cursors: Cursors,
	cursor_new: SDL_SystemCursor, // the cursor is updated only in d.refresh()
	cursor_prev: SDL_SystemCursor
}

#[derive(PartialEq)]
enum SpecialChar {
	None, VLine, HLine, LLCorner, LRCorner, ULCorner, URCorner
}

impl Renderer {
	fn new(window: Window, font_sz: c_int) -> Renderer {
		let sdl_renderer = SDLRenderer::new(&window);
		let font = Font::new(Path::new(FONT_FILE).as_os_str().to_str().unwrap(), font_sz);
		let fg = SDL_Color::from(COLOR_WHITE);
		let bg = SDL_Color::from(COLOR_BLACK);
		let font_ch_sz = Texture::from_font(&sdl_renderer, &font, "W", fg).size();
		
		let color_pairs = {
			const N_COLOR_PAIRS: usize = 256;
			let mut color_pairs = Vec::with_capacity(N_COLOR_PAIRS);
			for _ in 0..N_COLOR_PAIRS {
				color_pairs.push(ColorPair::new(fg, bg));
			}
			color_pairs
		};
		
		Renderer {
			txt_state: TxtState {
				sdl_renderer,
				
				x: 0,
				y: 0,
				
				font,
				font_ch_sz,
				
				color_pair_ind: 0,
				
				reverse: false,
				ckboard: false,
				underline: false,
				special_char: SpecialChar::None,
				
				char_buf: vec![' ' as chtype; CHAR_BUF_SZ*CHAR_BUF_SZ],
				char_buf_w: CHAR_BUF_SZ
			},
			fullscreen: false,
			window,
			key_timeout: 100,
			color_pairs,
			
			cursor_visibility: CURSOR_VISIBILITY::CURSOR_VISIBLE,
			mouse_event: None,
			cursors: Cursors {
				arrow: Cursor::new(SDL_SystemCursor::SDL_SYSTEM_CURSOR_ARROW),
				hand:  Cursor::new(SDL_SystemCursor::SDL_SYSTEM_CURSOR_HAND),
				crosshair: Cursor::new(SDL_SystemCursor::SDL_SYSTEM_CURSOR_CROSSHAIR)
			},
			cursor_new: SDL_SystemCursor::SDL_SYSTEM_CURSOR_ARROW,
			cursor_prev: SDL_SystemCursor::SDL_SYSTEM_CURSOR_ARROW
		}
	}
	
	// self.refresh() actually updates the cursor to avoid flickering
	pub fn set_mouse_to_arrow(&mut self) {self.cursor_new = SDL_SystemCursor::SDL_SYSTEM_CURSOR_ARROW;}
	pub fn set_mouse_to_hand(&mut self) {self.cursor_new = SDL_SystemCursor::SDL_SYSTEM_CURSOR_HAND;}
	pub fn set_mouse_to_crosshair(&mut self) {self.cursor_new = SDL_SystemCursor::SDL_SYSTEM_CURSOR_CROSSHAIR;}
	
	pub fn init_pair(&mut self, pair: CShort, fg: CShort, bg: CShort) {
		if let Some(color_pair) = self.color_pairs.get_mut(pair as usize) {
			*color_pair = ColorPair::new(SDL_Color::from(fg), SDL_Color::from(bg));
		}else{
			panic!("invalid color pair: {}", pair);
		}
	}
	
	pub fn mv<Y: TryInto<c_int>, X: TryInto<c_int>>(&mut self, y: Y, x: X) {
		if let Ok(y) = y.try_into() {
		if let Ok(x) = x.try_into() {
			if y < 0 || x < 0 {return;}
			self.txt_state.y = y;
			self.txt_state.x = x;
			return;
		}}
		panic!("could not convert arguments into integers");
	}
	
	pub fn addch<T: TryInto<chtype>>(&mut self, ch: T) {
		#[cfg(feature="profile")]
		let _g = Guard::new("addch");
		
		if let Ok(val) = ch.try_into() {
			let attr_non_zero = (val & (!A_CHARTEXT())) != 0;
			if attr_non_zero {self.attron(val);}
			self.addstr(&String::from((val & A_CHARTEXT()) as u8 as char));
			if attr_non_zero {self.attroff(val);}
		}else{
			panic!("could not convert to chtype");
		}
	}
	
	pub fn addnstr(&mut self, txt: &str, len: CInt) {
		for c in txt.chars().take(len as usize) {
			self.addch(c as chtype);
		}
	}
	
	pub fn addstr(&mut self, txt: &str) {
		#[cfg(feature="profile")]
		let _g = Guard::new("addstr");
		
		if txt.len() == 0 {return;}
		
		let color_pair = &mut self.color_pairs[self.txt_state.color_pair_ind];
		
		for ch in txt.chars() {
			self.txt_state.print_char(ch, color_pair);
		}
	}
	
	pub fn inch(&self) -> chtype {
		#[cfg(feature="profile")]
		let _g = Guard::new("inch");
		
		let txt_state = &self.txt_state;
		
		let ind = txt_state.y as usize * txt_state.char_buf_w + txt_state.x as usize;
		txt_state.char_buf[ind] as chtype
	}
	
	// see description above for embedding of attributes and color pairs
	pub fn attron<T: TryInto<chtype>>(&mut self, attr: T) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attron");
		
		if let Ok(attr) = attr.try_into() {
			let txt_state = &mut self.txt_state;
			
			// handle attributes
			if attr_set(attr, A_REVERSE_OFFSET) {
				txt_state.reverse = true;
			}
			
			if attr_set(attr, A_CKBOARD_OFFSET) {
				txt_state.ckboard = true;
			}
			
			if attr_set(attr, A_VLINE_OFFSET) {txt_state.special_char = SpecialChar::VLine;
			}else if attr_set(attr, A_HLINE_OFFSET) {txt_state.special_char = SpecialChar::HLine;
			}else if attr_set(attr, A_LLCORNER_OFFSET) {txt_state.special_char = SpecialChar::LLCorner;
			}else if attr_set(attr, A_LRCORNER_OFFSET) {txt_state.special_char = SpecialChar::LRCorner;
			}else if attr_set(attr, A_ULCORNER_OFFSET) {txt_state.special_char = SpecialChar::ULCorner;
			}else if attr_set(attr, A_URCORNER_OFFSET) {txt_state.special_char = SpecialChar::URCorner;}
			
			if attr_set(attr, A_UNDERLINE_OFFSET) {
				txt_state.underline = true;
			}
			
			// handle color settings
			if attr_set(attr, A_COLOR_PAIR_FLAG_OFFSET) {
				txt_state.color_pair_ind = color_pair_ind_frm_attr(attr);
			}
		}else{panic!("failed setting attron");}
	}
	
	// see description above for embedding of attributes and color pairs
	pub fn attroff<T: TryInto<chtype>>(&mut self, attr: T) {
		#[cfg(feature="profile")]
		let _g = Guard::new("attroff");

		if let Ok(attr) = attr.try_into() {
			let txt_state = &mut self.txt_state;
			// handle attributes
			if attr_set(attr, A_REVERSE_OFFSET) {
				txt_state.reverse = false;
			}
			
			if attr_set(attr, A_VLINE_OFFSET) || attr_set(attr, A_HLINE_OFFSET) ||
					attr_set(attr, A_LLCORNER_OFFSET) || attr_set(attr, A_LRCORNER_OFFSET) ||
					attr_set(attr, A_ULCORNER_OFFSET) || attr_set(attr, A_URCORNER_OFFSET) {
				txt_state.special_char = SpecialChar::None;
			}
			
			if attr_set(attr, A_CKBOARD_OFFSET) {
				txt_state.ckboard = false;
			}
			
			if attr_set(attr, A_UNDERLINE_OFFSET) {
				txt_state.underline = false;
			}
			
			if attr_set(attr, A_COLOR_PAIR_FLAG_OFFSET) {
				txt_state.color_pair_ind = 0;
			}
		}else{panic!("failed setting attroff");}
	}
	
	pub fn getmaxyx(&self, _: WINDOW, y: &mut i32, x: &mut i32) {
		#[cfg(feature="profile")]
		let _g = Guard::new("getmaxyx");
		
		let screen_sz = self.txt_state.sdl_renderer.get_viewport();
		*y = (screen_sz.h - PAD_DISPLAY_PX*2) / self.txt_state.font_ch_sz.h;
		*x = (screen_sz.w - PAD_DISPLAY_PX*2) / self.txt_state.font_ch_sz.w;
	}
	
	pub fn getyx(&self, _: WINDOW, y: &mut i32, x: &mut i32) {
		#[cfg(feature="profile")]
		let _g = Guard::new("getyx");

		*y = self.txt_state.y;
		*x = self.txt_state.x;
	}
	
	pub fn curs_set(&mut self, mode: CURSOR_VISIBILITY) -> CURSOR_VISIBILITY {
		let prev = self.cursor_visibility;
		self.cursor_visibility = mode;
		prev
	}
	
	pub fn refresh(&mut self) {
		#[cfg(feature="profile")]
		let _g = Guard::new("refresh");
		
		// show txt cursor
		match self.cursor_visibility {
			CURSOR_VISIBILITY::CURSOR_VISIBLE | CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE => {
				let txt_state = &self.txt_state;
				
				let ind = txt_state.y as usize * txt_state.char_buf_w + txt_state.x as usize;
				let buf = txt_state.char_buf[ind];
				self.addch(buf | A_REVERSE());
				
				/*let rect = {
					let w = txt_state.font_ch_sz.w;
					let h = txt_state.font_ch_sz.h;
					SDL_Rect {
						w, h,
						x: txt_state.x*w,
						y: txt_state.y*h
					}
				};
				
				let c = def_color_pair.fg;
				sdl_renderer.set_draw_color(c.r, c.b, c.g, c.a);
				sdl_renderer.fill_rect(rect);*/
			} CURSOR_VISIBILITY::CURSOR_INVISIBLE => {}
		}
		
		// update cursor
		if self.cursor_prev != self.cursor_new {
			match self.cursor_new {
				SDL_SystemCursor::SDL_SYSTEM_CURSOR_ARROW => {self.cursors.arrow.set();}
				SDL_SystemCursor::SDL_SYSTEM_CURSOR_HAND => {self.cursors.hand.set();}
				SDL_SystemCursor::SDL_SYSTEM_CURSOR_CROSSHAIR => {self.cursors.crosshair.set();}
				_ => {panic!("Cursor type not supported.");}
			}
			self.cursor_prev = self.cursor_new;
		}
		
		{ // update display
			let sdl_renderer = &self.txt_state.sdl_renderer;
			let def_color_pair = &self.color_pairs[0];
			
			sdl_renderer.present();
			let c = def_color_pair.bg;
			sdl_renderer.set_draw_color(c.r, c.b, c.g, c.a);
			sdl_renderer.clear();
		}
	}
	
	pub fn toggle_fullscreen(&mut self) {
		let flags = if self.fullscreen {0} else {SDL_WindowFlags::SDL_WINDOW_FULLSCREEN as u32};
		assert!(unsafe {SDL_SetWindowFullscreen(self.window.val, flags)} == 0, "Failed toggling fullscreen mode.");
		self.fullscreen ^= true;
	}
	
	pub fn getch(&mut self) -> CInt {
		#[cfg(feature="profile")]
		let _g = Guard::new("getch");
		
		macro_rules! terminate{() => {
			sdl_quit();
			
			#[cfg(feature="profile")]
			write_prof();
			
			std::process::exit(0);
		};};
		
		let start_time = Instant::now();
		'event_loop: loop {
			if let Some(event) = Event::poll() {
				match event {
					Event::Quit => {terminate!();
					} Event::TextInput(txt) => {
						//println!("text: {}", txt.text[0] as u8 as char);
						//let c_str: &CStr = unsafe { CStr::from_ptr(txt.text as *const i8) };
						//let ch_string = c_str.to_str().unwrap();
						//let ch_string = std::str::from_utf8(&txt.text).unwrap();
						let ch_string = txt.text[0] as u8 as char;
						if txt.text[1] != 0 {continue;} // len of string not 1
						
						// change display size? or terminate program (ctrl + c)?
						if ctrl_pressed() {
							if ch_string == '-' {
								self.dec_font_sz();
							}else if ch_string == '+' || ch_string == '=' {
								self.inc_font_sz();
							}else if ch_string == 'c' {
								terminate!();
							}
						}
						
						return ch_string as u8 as CInt;
					} Event::KeyDown(key) => {
						//println!("key down: {}", key.ch());
						return match key.keysym.sym {
							k if k == SDLK_AT as i32 => '@' as u8 as CInt,
							//k if k == SDLK_SPACE as i32 => ' ' as u8 as CInt,
							k if k == SDLK_DOWN as i32 => KEY_DOWN,
							k if k == SDLK_UP as i32 => KEY_UP,
							k if k == SDLK_LEFT as i32 => KEY_LEFT,
							k if k == SDLK_RIGHT as i32 => KEY_RIGHT,
							k if k == SDLK_KP_ENTER as i32 || k == SDLK_RETURN as i32 => {
								'\n' as u8 as CInt}
							k if k == SDLK_TAB as i32 => {
								'\t' as u8 as CInt}
							k if k == SDLK_AC_HOME as i32 => KEY_HOME,
							k if k == SDLK_BACKSPACE as i32 => KEY_BACKSPACE,
							k if k == SDLK_DELETE as i32 => KEY_DC,
							k if k == SDLK_ESCAPE as i32 => KEY_ESC,
							k if k == SDLK_F11 as i32 => {self.toggle_fullscreen(); continue;}
							_ => {
								let ch_string = key.ch();
								
								let shift_pressed = (key.keysym.mod_ & ((SDL_Keymod::KMOD_LSHIFT as u16) | (SDL_Keymod::KMOD_RSHIFT as u16))) != 0;
								let ctrl_pressed = (key.keysym.mod_ & ((SDL_Keymod::KMOD_LCTRL as u16) | (SDL_Keymod::KMOD_RCTRL as u16))) != 0;
								
								// change display size? or terminate program (ctrl + c)?
								if ctrl_pressed && ch_string == String::from("C") && !shift_pressed {
									terminate!();
								}
								continue;
							}
						};
					} Event::Window(_) => {
						self.clear();
						return KEY_WINDOW;
					} Event::MouseMotion(motion_event) => { // SDL_MouseMotionEvent
						let bstate = 
							if (motion_event.state & SDL_BUTTON_LMASK) != 0 {MouseState::LeftDragging
							}else if (motion_event.state & SDL_BUTTON_RMASK) != 0 {MouseState::RightDragging
							}else if (motion_event.state & SDL_BUTTON_MMASK) != 0 {MouseState::MiddleDragging
							}else{MouseState::Motion};
						
						self.set_mouse_event(bstate, motion_event.x, motion_event.y);
						return KEY_MOUSE;
					} Event::MouseButtonDown(button_event) => { // SDL_MouseButtonEvent
						let bstate = match button_event.button {
							ButtonInd::SDL_BUTTON_LEFT => MouseState::LeftPressed,
							ButtonInd::SDL_BUTTON_MIDDLE => MouseState::MiddlePressed,
							ButtonInd::SDL_BUTTON_RIGHT => MouseState::RightPressed,
							_ => {continue 'event_loop;}
						};
						
						self.set_mouse_event(bstate, button_event.x, button_event.y);
						return KEY_MOUSE;
					} Event::MouseButtonUp(button_event) => { // SDL_MouseButtonEvent
						let bstate = match button_event.button {
							ButtonInd::SDL_BUTTON_LEFT => MouseState::LeftReleased,
							ButtonInd::SDL_BUTTON_MIDDLE => MouseState::MiddleReleased,
							ButtonInd::SDL_BUTTON_RIGHT => MouseState::RightReleased,
							_ => {continue 'event_loop;}
						};
						
						self.set_mouse_event(bstate, button_event.x, button_event.y);
						return KEY_MOUSE;
					} Event::MouseWheel(wheel_event) => { // SDL_MouseWheelEvent
						let mut bstate = if wheel_event.y > 0 {
							MouseState::ScrollUp
						}else{
							MouseState::ScrollDown
						};
						
						// change font size
						if ctrl_pressed() {
							if bstate == MouseState::ScrollDown {
								self.dec_font_sz();
							}else if bstate == MouseState::ScrollUp {
								self.inc_font_sz();
							}
							bstate = MouseState::CtrlScroll;
							// ^ mostly a hack to prevent scroll_up()/scroll_down() from returning true and the game also zooming in/out the map
						}
						
						let (mut y, mut x) = (0,0);
						unsafe {SDL_GetMouseState(&mut x, &mut y)};
						
						self.set_mouse_event(bstate, x, y);
						return KEY_MOUSE;
					}
				}
			}else{
				if start_time.elapsed().as_millis() as u64 > self.key_timeout as u64 {
					return ERR;
				}
				thread::sleep(time::Duration::from_millis(15));
			}
		}
	}
	
	// converts x,y to grid coordinates
	fn set_mouse_event(&mut self, bstate: MouseState, x: c_int, y: c_int) {
		let grid_sz = &self.txt_state.font_ch_sz;
		
		self.mouse_event = Some(MEVENT {
			x: x / grid_sz.w,
			y: y / grid_sz.h,
			z: 0,
			bstate
		});
	}
	
	pub fn getmouse(&self, key_pressed: CInt) -> Option<MEVENT> {
		if key_pressed == KEY_MOUSE {
			self.mouse_event.clone()
		}else{
			None
		}
	}
	
	pub fn inc_font_sz(&mut self) {
		let font_sz = self.txt_state.font.font_sz;
		self.set_font_sz(font_sz + FONT_CHG_STEP);
	}
	pub fn dec_font_sz(&mut self) {
		let font_sz = self.txt_state.font.font_sz;
		self.set_font_sz(font_sz - FONT_CHG_STEP);
	}

	fn set_font_sz(&mut self, font_sz: c_int) {
		if font_sz <= 0 || font_sz > FONT_SZ_MAX {return;}
		
		let txt_state = &mut self.txt_state;
		txt_state.font = Font::new(FONT_FILE, font_sz);
		txt_state.font_ch_sz = Texture::from_font(&txt_state.sdl_renderer, &txt_state.font, "W", SDL_Color::from(COLOR_WHITE)).size();
		
		// invalidate caches
		for color_pair in self.color_pairs.iter_mut() {
			color_pair.invalidate_cache();
		}
	}
	
	pub fn clrtoeol(&mut self) {
		let txt_state = &self.txt_state;
		let cur_x = txt_state.x;
		let sz = txt_state.sdl_renderer.get_viewport();
		let chars_per_line = sz.w / txt_state.font_ch_sz.w;
		
		// we are already at the edge of the line
		if chars_per_line < cur_x {return;}
		
		let mut blanks = String::with_capacity((chars_per_line - cur_x) as usize);
		for _ in 0..(chars_per_line - cur_x) {
			blanks.push(' ');
		}
		
		self.addstr(&blanks);
		self.txt_state.x = cur_x;
	}
	
	pub fn clear(&mut self) {
		let sdl_renderer = &self.txt_state.sdl_renderer;
		let c = self.color_pairs[0].bg;
		sdl_renderer.set_draw_color(c.r, c.b, c.b, c.a);
		sdl_renderer.clear();
		self.txt_state.char_buf = vec![' ' as chtype; CHAR_BUF_SZ*CHAR_BUF_SZ];
	}
	pub fn timeout(&mut self, t: CInt) {self.key_timeout = t;}
	
	// returns (y, x)
	pub fn mouse_pos(&self) -> Option<(i32, i32)> {
		let mut y = 0;
		let mut x = 0;
		unsafe {SDL_GetMouseState(&mut x, &mut y)};
		let font_sz = &self.txt_state.font_ch_sz;
		Some((y / font_sz.h, x / font_sz.w))
	}
}

fn ctrl_pressed() -> bool {
	is_key_down(SDL_Scancode::SDL_SCANCODE_LCTRL) || 
	is_key_down(SDL_Scancode::SDL_SCANCODE_RCTRL)
}

pub fn shift_pressed() -> bool {
	is_key_down(SDL_Scancode::SDL_SCANCODE_LSHIFT) || 
	is_key_down(SDL_Scancode::SDL_SCANCODE_RSHIFT)
}

pub fn flushinp() {
	// inputs: SDL_EventType
	//unsafe{SDL_FlushEvents(SDL_EventType::SDL_KEYDOWN as u32, SDL_EventType::SDL_KEYUP as u32);}
	unsafe{SDL_FlushEvents(SDL_EventType::SDL_KEYDOWN as u32, SDL_EventType::SDL_USEREVENT as u32);}
}

impl SDL_Color {
	pub fn from(ind: CShort) -> Self {
		// https://jonasjacek.github.io/colors/ Accessed September 6, 2020
		let (r, g, b) = match ind {
			0 => (0,0,0), // BLACK
			1 => (175,0,0), // RED
			2 => (0,175,0), // GREEN
			3 => (178,104,24),//(128,128,0), // YELLOW
			4 => (0,0,128), // BLUE
			5 => (128,0,128), // MAGENTA
			6 => (0,175,175), // CYAN
			7 => (192,192,192), // WHITE
			8 => (128,128,128),
			9 => (255,0,0),
			10 => (0,255,0),
			11 => (255,255,0),
			12 => (0,0,255),
			13 => (255,0,255),
			14 => (0,255,255),
			15 => (255,255,255),
			16 => (0,0,0),
			17 => (0,0,95),
			18 => (0,0,135),
			19 => (0,0,175),
			20 => (0,0,215),
			21 => (0,0,255),
			22 => (0,95,0),
			23 => (0,95,95),
			24 => (0,95,135),
			25 => (0,95,175),
			26 => (0,95,215),
			27 => (0,95,255),
			28 => (0,135,0),
			29 => (0,135,95),
			30 => (0,135,135),
			31 => (0,135,135),
			32 => (0,135,215),
			33 => (0,135,255),
			34 => (0,175,0),
			35 => (0,175,95),
			36 => (0,175,135),
			37 => (0,175,175),
			38 => (0,175,215),
			39 => (0,175,255),
			40 => (0,215,0),
			41 => (0,215,95),
			42 => (0,215,135),
			43 => (0,215,175),
			44 => (0,215,215),
			45 => (0,215,255),
			46 => (0,255,0),
			47 => (0,255,95),
			48 => (0,255,135),
			49 => (0,255,175),
			50 => (0,255,215),
			51 => (0,255,255),
			52 => (95,0,0),
			53 => (95,0,95),
			54 => (95,0,135),
			55 => (95,0,175),
			56 => (95,0,215),
			57 => (95,0,255),
			58 => (95,95,0),
			59 => (95,95,95),
			60 => (95,95,135),
			61 => (95,95,175),
			62 => (95,95,215),
			63 => (95,95,255),
			64 => (95,135,0),
			65 => (95,135,95),
			66 => (95,135,135),
			67 => (95,135,175),
			68 => (95,135,215),
			69 => (95,135,255),
			70 => (95,175,0),
			71 => (95,175,95),
			72 => (95,175,135),
			73 => (95,175,175),
			74 => (95,175,215),
			75 => (95,175,255),
			76 => (95,215,0),
			77 => (95,215,95),
			78 => (95,215,135),
			79 => (95,215,175),
			80 => (95,215,215),
			81 => (95,215,255),
			82 => (95,255,0),
			83 => (95,255,95),
			84 => (95,255,135),
			85 => (95,255,175),
			86 => (95,255,215),
			87 => (95,255,255),
			88 => (135,0,0),
			89 => (135,0,95),
			90 => (135,0,135),
			91 => (135,0,175),
			92 => (135,0,215),
			93 => (135,0,255),
			94 => (135,95,0),
			95 => (135,95,95),
			96 => (135,95,135),
			97 => (135,95,175),
			98 => (135,95,215),
			99 => (135,95,255),
			100 => (135,135,0),
			101 => (135,135,95),
			102 => (135,135,135),
			103 => (135,135,175),
			104 => (135,135,215),
			105 => (135,135,255),
			106 => (135,175,0),
			107 => (135,175,95),
			108 => (135,175,135),
			109 => (135,175,175),
			110 => (135,175,215),
			111 => (135,175,255),
			112 => (135,215,0),
			113 => (135,215,95),
			114 => (135,215,135),
			115 => (135,215,175),
			116 => (135,215,215),
			117 => (135,215,255),
			118 => (135,255,0),
			119 => (135,255,95),
			120 => (135,255,135),
			121 => (135,255,175),
			122 => (135,255,215),
			123 => (135,255,255),
			124 => (175,0,0),
			125 => (175,0,95),
			126 => (175,0,135),
			127 => (175,0,175),
			128 => (175,0,215),
			129 => (175,0,255),
			130 => (175,95,0),
			131 => (175,95,95),
			132 => (175,95,135),
			133 => (175,95,175),
			134 => (175,95,215),
			135 => (175,95,255),
			136 => (175,135,0),
			137 => (175,135,95),
			138 => (175,135,135),
			139 => (175,135,175),
			140 => (175,135,215),
			141 => (175,135,255),
			142 => (175,175,0),
			143 => (175,175,95),
			144 => (175,175,135),
			145 => (175,175,175),
			146 => (175,175,215),
			147 => (175,175,255),
			148 => (175,215,0),
			149 => (175,215,95),
			150 => (175,215,135),
			151 => (175,215,175),
			152 => (175,215,215),
			153 => (175,215,255),
			154 => (175,255,0),
			155 => (175,255,95),
			156 => (175,255,135),
			157 => (175,255,175),
			158 => (175,255,215),
			159 => (175,255,255),
			160 => (215,0,0),
			161 => (215,0,95),
			162 => (215,0,135),
			163 => (215,0,175),
			164 => (215,0,215),
			165 => (215,0,255),
			166 => (215,95,0),
			167 => (215,95,95),
			168 => (215,95,135),
			169 => (215,95,175),
			170 => (215,95,215),
			171 => (215,95,255),
			172 => (215,135,0),
			173 => (215,135,95),
			174 => (215,135,135),
			175 => (215,135,175),
			176 => (215,135,215),
			177 => (215,135,255),
			178 => (215,175,0),
			179 => (215,175,95),
			180 => (215,175,135),
			181 => (215,175,175),
			182 => (215,175,215),
			183 => (215,175,255),
			184 => (215,215,0),
			185 => (215,215,95),
			186 => (215,215,135),
			187 => (215,215,175),
			188 => (215,215,215),
			189 => (215,215,255),
			190 => (215,255,0),
			191 => (215,255,95),
			192 => (215,255,135),
			193 => (215,255,175),
			194 => (215,255,215),
			195 => (215,255,255),
			196 => (255,0,0),
			197 => (255,0,95),
			198 => (255,0,135),
			199 => (255,0,175),
			200 => (255,0,215),
			201 => (255,0,255),
			202 => (255,95,0),
			203 => (255,95,95),
			204 => (255,95,135),
			205 => (255,95,175),
			206 => (255,95,215),
			207 => (255,95,255),
			208 => (255,135,0),
			209 => (255,135,95),
			210 => (255,135,135),
			211 => (255,135,175),
			212 => (255,135,215),
			213 => (255,135,255),
			214 => (255,175,0),
			215 => (255,175,95),
			216 => (255,175,135),
			217 => (255,175,175),
			218 => (255,175,215),
			219 => (255,175,255),
			220 => (255,215,0),
			221 => (255,215,95),
			222 => (255,215,135),
			223 => (255,215,175),
			224 => (255,215,215),
			225 => (255,215,255),
			226 => (255,255,0),
			227 => (255,255,95),
			228 => (255,255,135),
			229 => (255,255,175),
			230 => (255,255,215),
			231 => (255,255,255),
			232 => (8,8,8),
			233 => (18,18,18),
			234 => (28,28,28),
			235 => (38,38,38),
			236 => (48,48,48),
			237 => (58,58,58),
			238 => (68,68,68),
			239 => (78,78,78),
			240 => (88,88,88),
			241 => (98,98,98),
			242 => (108,108,108),
			243 => (118,118,118),
			244 => (128,128,128),
			245 => (138,138,138),
			246 => (148,148,148),
			247 => (158,158,158),
			248 => (168,168,168),
			249 => (178,178,178),
			250 => (188,188,188),
			251 => (198,198,198),
			252 => (208,208,208),
			253 => (218,218,218),
			254 => (228,228,228),
			255 => (238,238,238),	
			_ => {panic!("unsupport color: {}", ind);}
		};
		Self {r, g, b, a: 255}
	}
}

pub fn setup_disp_lib() -> Renderer {
	sdl_init();
	ttf_init();
	
	let window = Window::new("Arcane Fortune", Some(ICON_FILE));
	unsafe {SDL_StartTextInput();}
	Renderer::new(window, 14)
}

pub fn setup_disp_lib_custom_font_sz(font_sz: c_int) -> Renderer {
	sdl_init();
	ttf_init();
	
	let window = Window::new("Arcane Fortune", Some(ICON_FILE));
	unsafe {SDL_StartTextInput();}
	Renderer::new(window, font_sz)
}


#[derive(Clone, PartialEq)]
pub enum MouseState {
	Motion, // left button not pressed
	LeftDragging, // motion w/ left button pressed
	LeftReleased,
	LeftPressed,
	RightDragging,
	RightReleased,
	RightPressed,
	MiddleDragging,
	MiddleReleased,
	MiddlePressed,
	ScrollUp,
	ScrollDown,
	CtrlScroll
}

#[derive(Clone)]
pub struct MEVENT {
	pub x: c_int,
	pub y: c_int,
	pub z: c_int,
	pub bstate: MouseState
}

pub fn rbutton_clicked(_mouse_event: &Option<MEVENT>) -> bool {false}
pub fn mbutton_clicked(_mouse_event: &Option<MEVENT>) -> bool {false}
pub fn lbutton_clicked(_mouse_event: &Option<MEVENT>) -> bool {false}

pub fn rbutton_released(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::RightReleased, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn rbutton_pressed(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::RightPressed, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn mbutton_released(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::MiddleReleased, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn mbutton_pressed(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::MiddlePressed, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn lbutton_released(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::LeftReleased, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn lbutton_pressed(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::LeftPressed, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn ldragging(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::LeftDragging, ..}) = mouse_event {true} else {false}
}

pub fn rdragging(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::RightDragging, ..}) = mouse_event {true} else {false}
}

pub fn mdragging(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::MiddleDragging, ..}) = mouse_event {true} else {false}
}

pub fn motion(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::Motion, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn scroll_up(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::ScrollUp, ..}) = mouse_event {
		true
	}else{
		false
	}
}

pub fn scroll_down(mouse_event: &Option<MEVENT>) -> bool {
	if let Some(MEVENT {bstate: MouseState::ScrollDown, ..}) = mouse_event {
		true
	}else{
		false
	}
}

