#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// SDL2: 2.0.12 (Accessed September 6, 2020)
// SDL2_image: 2.0.5 (Accessed September 6, 2020)
// SDL2_ttf: 2.0.15 (Accessed September 6, 2020)

// All paths are on Centos 8

// SDL2_image installed to:
// /usr/local/lib
// /usr/local/include/SDL2
// /usr/local/include/SDL2/SDL_image.h

// SDl2_tff installed to:
// /usr/local/lib
// /usr/local/include/SDL2
// /usr/local/include/SDL2/SDL_tff.h

// to prevent linker errors: (may not be necessary, errors were caused by specifing the wrong #[link(name=...)]
// cp /usr/local/lib/libSDL2_ttf.so ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib
// cp /usr/local/lib/libSDL2_ttf.so /home/tapa/.rustup/toolchains/1.46.0-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib

const fn SDL_BUTTON(x: u32) -> u32 {1 << (x-1)}
pub const SDL_BUTTON_LMASK: u32 = SDL_BUTTON(ButtonInd::SDL_BUTTON_LEFT as u32);
pub const SDL_BUTTON_RMASK: u32 = SDL_BUTTON(ButtonInd::SDL_BUTTON_RIGHT as u32);

use std::slice;
use std::os::raw::{c_int, c_char};
use std::ffi::CString;
use std::ptr;

#[cfg(feature="profile")]
mod profiling;
#[cfg(feature="profile")]
pub use profiling::*;

pub mod sdl2_defs; pub use sdl2_defs::*;
pub mod curses; pub use curses::*;

// /usr/include/SDL2/SDL.h
pub const SDL_INIT_VIDEO: u32 = 0x00000020;
pub const SDL_WINDOWPOS_UNDEFINED: i32 = 0x1FFF0000;
pub const SDL_WINDOW_RESIZABLE: u32 = 0x00000020;

pub enum SDL_Window_Struct {}
pub enum SDL_Suface_Struct {}
pub enum SDL_Renderer_Struct {}
pub enum SDL_Texture_Struct {}
pub enum SDL_Cursor_Struct {}
pub enum TTF_Font_Struct {}

pub type SDL_Window = *mut SDL_Window_Struct;
pub type SDL_Surface = *mut SDL_Suface_Struct;
pub type SDL_Renderer = *mut SDL_Renderer_Struct;
pub type SDL_Texture = *mut SDL_Texture_Struct;
pub type SDL_Cursor = *mut SDL_Cursor_Struct;
pub type TTF_Font = *mut TTF_Font_Struct;

pub struct Window {pub val: SDL_Window}
pub struct Surface {pub val: SDL_Surface}
pub struct Renderer {pub val: SDL_Renderer}
pub struct Texture {pub val: SDL_Texture}
pub struct Cursor {pub val: SDL_Cursor}
pub struct Font {
	pub val: TTF_Font,
	pub font_sz: c_int
}

#[derive(Clone, Debug)]
pub struct Size {
	pub w: c_int,
	pub h: c_int
}

impl Drop for Renderer {
	fn drop(&mut self) {
		unsafe {SDL_DestroyRenderer(self.val);}
	}
}

impl Drop for Texture {
	fn drop(&mut self) {
		unsafe {SDL_DestroyTexture(self.val);}
	}
}

impl Drop for Surface {
	fn drop(&mut self) {
		unsafe {SDL_FreeSurface(self.val);}
	}
}

impl Drop for Font {
	fn drop(&mut self) {
		unsafe {TTF_CloseFont(self.val);}
	}
}

impl Drop for Cursor {
	fn drop(&mut self) {
		unsafe {SDL_FreeCursor(self.val);}
	}
}

impl Cursor {
	pub fn new(id: SDL_SystemCursor) -> Self {
		let cursor = unsafe {SDL_CreateSystemCursor(id)};
		assert!(!cursor.is_null(), "Failed getting system cursor.");
		Self {val: cursor}
	}
	
	pub fn set(&self) {unsafe {SDL_SetCursor(self.val);}}
}

impl Surface {
	pub fn from_img(file: &str) -> Self {
		let c_str = CString::new(file).expect("CString::new failed in creating file name");
		let surface = unsafe {IMG_Load(c_str.as_ptr())};
		assert!(!surface.is_null(), "Failed loading image: `{}` to surface", file);
		Self {val: surface}
	}
}

impl Texture {
	pub fn from_img(renderer: &Renderer, file: &str) -> Self {
		let c_str = CString::new(file).expect("CString::new failed in creating file name");
		let tex = unsafe {IMG_LoadTexture(renderer.val, c_str.as_ptr())};
		assert!(!tex.is_null(), "Failed loading image: `{}` to texture", file);
		Self {val: tex}
	}
	
	pub fn from_surface(renderer: &Renderer, surface: Surface) -> Self {
		#[cfg(feature="profile")]
		let _g = Guard::new("texture.from_surface() [SDL_CreateTextureFromSurface]");

		let tex = unsafe {SDL_CreateTextureFromSurface(renderer.val, surface.val)};
		assert!(!tex.is_null(), "Failed creating texture from surface");
		Self {val: tex}
	}
	
	pub fn from_font(renderer: &Renderer, font: &Font, txt: &str, fg: SDL_Color) -> Self {
		Texture::from_surface(renderer, font.render_blended(txt, fg))
	}
	
	pub fn size(&self) -> Size {
		#[cfg(feature="profile")]
		let _g = Guard::new("texture.size() [SDL_QueryTexture]");

		let mut sz = Size {h: 0, w: 0};
		assert!(unsafe {SDL_QueryTexture(self.val, ptr::null_mut(), ptr::null_mut(), &mut sz.w, &mut sz.h)} == 0, "Failed to query texture");
		sz
	}
}

impl Window {
	pub fn new(title: &str, icon_file: Option<&str>) -> Self {
		let c_str = CString::new(title).expect("CString::new failed in creating window title");
		let window = unsafe {SDL_CreateWindow(c_str.as_ptr(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1024, 768, SDL_WINDOW_RESIZABLE)};
		assert!(!window.is_null(), "Failed creating SDL window");
		
		if let Some(icon_file) = icon_file {
			let icon = Surface::from_img(icon_file);
			unsafe {SDL_SetWindowIcon(window, icon.val);}
		}
		
		Self {val: window}
	}
	
	// not used
	pub fn get_surface(&self) -> Surface {
		let surface = unsafe {SDL_GetWindowSurface(self.val)};
		assert!(!surface.is_null(), "Failed getting window surface");
		Surface {val: surface}
	}
	
	// not used
	pub fn update_surface(&self) {
		assert!(unsafe {SDL_UpdateWindowSurface(self.val)} == 0, "Failed updating window surface");
	}
}

impl Renderer {
	pub fn new(window: &Window) -> Self {
		let mut flags = SDL_RendererFlags::SDL_RENDERER_ACCELERATED as u32;// | SDL_RendererFlags::SDL_RENDERER_PRESENTVSYNC as u32;
		let mut renderer = unsafe {SDL_CreateRenderer(window.val, -1, flags)};
		if renderer.is_null() {
			println!("Failed creating hardware-accelerated renderer. Will use software renderer instead.");
			flags = SDL_RendererFlags::SDL_RENDERER_SOFTWARE as u32;
			renderer = unsafe {SDL_CreateRenderer(window.val, -1, flags)};
		}
		assert!(!renderer.is_null(), "Failed creating renderer");
		Self {val: renderer}
	}
	
	// not used
	pub fn software(surface: Surface) -> Self {
		let renderer = unsafe {SDL_CreateSoftwareRenderer(surface.val)};
		assert!(!renderer.is_null(), "Failed creating software renderer");
		Self {val: renderer}
	}
	
	pub fn set_draw_color(&self, r: u8, g: u8, b: u8, a: u8) {
		#[cfg(feature="profile")]
		let _g = Guard::new("renderer.set_draw_color() [SDL_SetRenderDrawColor]");

		assert!(unsafe{SDL_SetRenderDrawColor(self.val, r, g, b, a)} == 0, "Failed setting render draw color");
	}
	
	pub fn clear(&self) {
		#[cfg(feature="profile")]
		let _g = Guard::new("renderer.clear() [SDL_RenderClear]");

		assert!(unsafe{SDL_RenderClear(self.val)} == 0, "Failed clearing renderer");
	}
	
	pub fn get_viewport(&self) -> SDL_Rect {
		#[cfg(feature="profile")]
		let _g = Guard::new("renderer.get_viewport [SDL_RenderGetViewport");

		let mut rect = SDL_Rect {x: 0, y: 0, w: 0, h: 0};
		unsafe {SDL_RenderGetViewport(self.val, &mut rect);}
		rect
	}
	
	pub fn fill_rect(&self, rect: SDL_Rect) {
		#[cfg(feature="profile")]
		let _g = Guard::new("renderer.fill_rect() [SDL_RenderFillRect]");

		assert!(unsafe {SDL_RenderFillRect(self.val, &rect)} == 0, "Failed to fill renderer rectangle");
	}
	
	pub fn copy_ex(&self, texture: &Texture, src: &SDL_Rect, dest: &SDL_Rect) {
		assert!(unsafe {SDL_RenderCopy(self.val, texture.val, src, dest)} == 0, "Failed copying texture to renderer");
	}
	
	pub fn copy(&self, texture: &Texture, dest: &SDL_Rect) {
		#[cfg(feature="profile")]
		let _g = Guard::new("renderer.copy() [SDL_RenderCopy]");
		
		assert!(unsafe {SDL_RenderCopy(self.val, texture.val, ptr::null(), dest)} == 0, "Failed copying texture to renderer");
	}
	
	pub fn present(&self) {
		#[cfg(feature="profile")]
		let _g = Guard::new("renderer.present() [SDL_RenderPresent]");
		
		unsafe {SDL_RenderPresent(self.val)};
	}
}

impl Font {
	pub fn new(file: &str, font_sz: c_int) -> Self {
		let c_str = CString::new(file).expect("CString::new failed in creating font file name");
		let font = unsafe {TTF_OpenFont(c_str.as_ptr(), font_sz)};
		assert!(!font.is_null(), "Failed loading font: `{}`", file);
		Self {val: font, font_sz}
	}
	
	pub fn set_style(&self, style: c_int) {
		unsafe {TTF_SetFontStyle(self.val, style);}
	}
	
	// faster but lower quality than render_solid()
	pub fn render_solid(&self, txt: &str, fg: SDL_Color) -> Surface {
		let c_str = CString::new(txt).expect("CString::new failed in rendering font");
		//let surface = unsafe {TTF_RenderText_Blended(self.val, c_str.as_ptr(), fg)};
		let surface = unsafe {TTF_RenderText_Solid(self.val, c_str.as_ptr(), fg)};
		assert!(!surface.is_null(), "Failed rendering text `{}`", txt);
		Surface {val: surface}
	}
	
	pub fn render_blended(&self, txt: &str, fg: SDL_Color) -> Surface {
		let c_str = CString::new(txt).expect("CString::new failed in rendering font");
		//let surface = unsafe {TTF_RenderText_Blended(self.val, c_str.as_ptr(), fg)};
		let surface = unsafe {TTF_RenderText_Blended(self.val, c_str.as_ptr(), fg)};
		assert!(!surface.is_null(), "Failed rendering text `{}`", txt);
		Surface {val: surface}
	}
}

pub enum Event { // created from SDL_Event
	Quit,
	KeyDown(SDL_KeyboardEvent),
	Window(SDL_WindowEvent),
	MouseMotion(SDL_MouseMotionEvent),
	MouseButtonDown(SDL_MouseButtonEvent),
	MouseButtonUp(SDL_MouseButtonEvent),
	MouseWheel(SDL_MouseWheelEvent),
	TextInput(SDL_TextInputEvent)
}

use std::ffi::CStr;
impl SDL_KeyboardEvent {
	pub fn ch(&self) -> String {
		let c_buf = unsafe {SDL_GetKeyName(self.keysym.sym)};
		let c_str: &CStr = unsafe { CStr::from_ptr(c_buf) };
		String::from(c_str.to_str().unwrap())
	}
}

pub fn is_key_down(key: SDL_Scancode) -> bool {
	let mut array_len = 0;
	let state = unsafe{SDL_GetKeyboardState(&mut array_len)};
	
	assert!(array_len as i32 > key as i32, "SDL_GetKeyboardState did not return a long enough array for querying. Found: {}, Needed: {}", array_len, key as isize);
	assert!(!state.is_null(), "SDL_GetKeyboardState returned null");
	
	let state = unsafe {slice::from_raw_parts(state, array_len as usize)};
	state[key as usize] == 1
}

impl Event {
	pub fn poll() -> Option<Event> {
		let mut e = SDL_Event {type_: SDL_EventType::SDL_QUIT};
		if unsafe {SDL_PollEvent(&mut e)} == 1 {
			unsafe {match e.type_ {
				SDL_EventType::SDL_QUIT => {Some(Event::Quit)}
				SDL_EventType::SDL_WINDOWEVENT => {Some(Event::Window(e.window))}
				SDL_EventType::SDL_KEYDOWN => {Some(Event::KeyDown(e.key))}
				SDL_EventType::SDL_MOUSEMOTION => {Some(Event::MouseMotion(e.motion))}
				SDL_EventType::SDL_MOUSEBUTTONDOWN => {Some(Event::MouseButtonDown(e.button))}
				SDL_EventType::SDL_MOUSEBUTTONUP => {Some(Event::MouseButtonUp(e.button))}
				SDL_EventType::SDL_MOUSEWHEEL => {Some(Event::MouseWheel(e.wheel))}
				SDL_EventType::SDL_TEXTINPUT => {Some(Event::TextInput(e.text))}
				_ => {None}
			}}
		}else{
			None
		}
	}
}

pub fn sdl_init() {
	assert!(unsafe {SDL_Init(SDL_INIT_VIDEO)} == 0, "SDL init failed.");
}

pub fn sdl_quit() {
	unsafe {SDL_Quit();}
}

pub fn ttf_init() {
	assert!(unsafe {TTF_Init()} == 0, "Failed initializing SDL_ttf");
}

#[link(name="SDL2")]
extern "C" {
	pub fn SDL_StartTextInput();
	
	// /usr/include/SDL2/SDL.h
	pub fn SDL_Init(flags: u32) -> c_int;
	pub fn SDL_Quit();
	
	// /usr/include/SDL2/SDL_log.h
	//pub fn SDL_LogError(category: c_int, 
	
	///////////////////////
	// /usr/include/SDL2/SDL_video.h
	pub fn SDL_CreateWindow(title: *const c_char,
			x: c_int, y: c_int, w: c_int, h: c_int, flags: u32) -> SDL_Window;
	
	pub fn SDL_GetWindowSurface(window: SDL_Window) -> SDL_Surface;
	pub fn SDL_UpdateWindowSurface(window: SDL_Window) -> c_int;
	pub fn SDL_SetWindowIcon(window: SDL_Window, icon: SDL_Surface);
	pub fn SDL_SetWindowFullscreen(window: SDL_Window, flags: u32) -> c_int;
	
	// /usr/include/SDL2/SDL_render.h
	pub fn SDL_CreateSoftwareRenderer(surface: SDL_Surface) -> SDL_Renderer;
	pub fn SDL_SetRenderDrawColor(renderer: SDL_Renderer, r: u8, g: u8, b: u8, a: u8) -> c_int;
	pub fn SDL_RenderClear(renderer: SDL_Renderer) -> c_int;
	pub fn SDL_RenderGetViewport(renderer: SDL_Renderer, rect: *mut SDL_Rect);
	pub fn SDL_RenderFillRect(renderer: SDL_Renderer, rect: *const SDL_Rect) -> c_int;
	pub fn SDL_RenderCopy(renderer: SDL_Renderer, texture: SDL_Texture, src: *const SDL_Rect, dest: *const SDL_Rect) -> c_int;
	pub fn SDL_RenderPresent(renderer: SDL_Renderer);
	pub fn SDL_CreateRenderer(window: SDL_Window, index: c_int, flags: u32) -> SDL_Renderer;
	pub fn SDL_CreateTextureFromSurface(renderer: SDL_Renderer, surface: SDL_Surface) -> SDL_Texture; 
	pub fn SDL_QueryTexture(texture: SDL_Texture, format: *mut u32, access: *mut c_int, w: &mut c_int, h: &mut c_int) -> c_int;
	
	pub fn SDL_DestroyRenderer(renderer: SDL_Renderer);
	pub fn SDL_DestroyTexture(texture: SDL_Texture);
	pub fn SDL_FreeSurface(surface: SDL_Surface);
	
	// /usr/include/SDL2/SDL_events.h
	pub fn SDL_PollEvent(event: *mut SDL_Event) -> c_int;
	pub fn SDL_FlushEvents(min_type: u32, max_type: u32); // inputs: SDL_EventType
	
	// /usr/include/SDL2/SDL_keyboard.h
	pub fn SDL_GetKeyboardState(array_len: &mut c_int) -> *const u8;
	
	// /usr/include/SDL2/SDL_mouse.h
	pub fn SDL_GetMouseState(x: &mut c_int, y: &mut c_int) -> u32;
	pub fn SDL_CreateSystemCursor(id: SDL_SystemCursor) -> SDL_Cursor;
	pub fn SDL_SetCursor(cursor: SDL_Cursor);
	
	pub fn SDL_FreeCursor(cursor: SDL_Cursor);
}

#[link(name="SDL2_image")]
extern "C" {
	// /usr/local/include/SDL2/SDL_image.h
	pub fn IMG_Load(file: *const c_char) -> SDL_Surface;
	pub fn IMG_LoadTexture(renderer: SDL_Renderer, file: *const c_char) -> SDL_Texture;
}

#[link(name="SDL2_ttf")]
extern "C" {
	// /usr/local/include/SDL2/SDL_tff.h
	pub fn TTF_Init() -> c_int;
	pub fn TTF_OpenFont(file: *const c_char, font_size: c_int) -> TTF_Font;
	pub fn TTF_RenderText_Blended(font: TTF_Font, text: *const c_char, fg: SDL_Color) -> SDL_Surface;
	pub fn TTF_RenderText_Solid(font: TTF_Font, text: *const c_char, fg: SDL_Color) -> SDL_Surface; // faster
	pub fn TTF_SetFontStyle(font: TTF_Font, style: c_int);
	pub fn TTF_CloseFont(font: TTF_Font);
}

pub const TTF_STYLE_NORMAL: c_int = 0x00;
pub const TTF_STYLE_BOLD: c_int = 0x01;
pub const TTF_STYLE_ITALIC: c_int = 0x02;
pub const TTF_STYLE_UNDERLINE: c_int = 0x04;
pub const TTF_STYLE_STRIKETHROUGH: c_int = 0x08;

