/////////////////////////////////////////////////
// Windows 256 color support w/ pdcurses 3.8 (wincon):
//
// (pdcurses will be statically linked into the af binary)
//
// to cross-compile on Linux for windows:
// 1. install mingw64 and rust support: 
//		sudo yum install mingw64-gcc
// 		rustup install stable-x86_64-pc-windows-gnu
// 		rustup target add x86_64-pc-windows-gnu
//
// 2. Compile PDCurses:
//   2a. cd ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/wincon
//   2b. alter makefile to include:
//		CC          = x86_64-w64-mingw32-gcc
//		AR          = x86_64-w64-mingw32-ar
//		STRIP       = x86_64-w64-mingw32-strip
//		LINK        = x86_64-w64-mingw32-gcc
//
// 3. make WIDE=Y
//	cp ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/wincon/pdcurses.a ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-pc-windows-gnu/lib/libpdcurses.a
//
// 4. Building af for Windows on Linux:
//		cd ~/af/arcane_fortune/src
//		cargo build --target=x86_64-pc-windows-gnu [--release]


///////////////////////////////////////////////////////
// Windows 256 color support PDCurses SDL2 *fails*
//
// 1.  sudo yum install SDL2
//     sudo yum install SDL2-devel
//
// [? 1b. Download SDL2 source code https://www.libsdl.org/download-2.0.php (https://www.libsdl.org/release/SDL2-2.0.10.tar.gz)
//     cd ~/Downloads/SDL2-2.0.10
//     ./configure --host=x86_64-w64-mingw32
//     make
//     sudo make install
//     cp /usr/local/cross-tools/x86_64-w64-mingw32/lib/libSDL2.a ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-pc-windows-gnu/li
//     cp /usr/local/cross-tools/x86_64-w64-mingw32/lib/libSDL2main.a ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-pc-windows-gnu/lib/
//     cp /usr/local/cross-tools/x86_64-w64-mingw32/lib/libSDL2_test.a ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-pc-windows-gnu/lib/

// ]
//
// 2.  alter make file to include the following definitions (such that there are no over-writing statements after re-setting these variables):
//		CC          = x86_64-w64-mingw32-gcc
//		AR          = x86_64-w64-mingw32-ar
//		STRIP       = x86_64-w64-mingw32-strip
//		LINK        = x86_64-w64-mingw32-gcc
//
// 3. make WIDE=Y
//    cp ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/sdl2/pdcurses.a ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-pc-windows-gnu/lib/libpdcurses.a
//
// FAILS with SDL link errors
//
// also cannot cross-compile demo program (diff. link errors than from rust)
// :~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/demos$ x86_64-w64-mingw32-gcc xmas.c ../sdl2/pdcurses.a /usr/local/cross-tools/x86_64-w64-mingw32/lib/libSDL2.a /usr/local/cross-tools/x86_64-w64-mingw32/lib/libSDL2main.a /usr/local/cross-tools/x86_64-w64-mingw32/lib/libSDL2_test.a -I.. -o xmas_wincon.exe -lole32 -loleaut32 -DZMQ_STATIC=1 -DENABLE-DRAFTS=no -lversion -limm32

//////////////////////////////////////////////
// Windows 256 color support PDCurses SDL1 *fails*
// 
// 1. sudo yum install SDL-devel
//    sudo yum install libiconv-devel [fails]
//
// 2. cd ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/sdl1
//    make [fails... cannot find iconv.h]

//////////////////////////////
// to build on Windows for Windows:

// (not actually done for pdcurses, was tried for ncurses
// which seemed to not work -- mingw ncurses can't seem to support 256 colors--was not able to build
// af for windows supporting cygwin using ncurses). old instructions:
//
// on Windows:
// 1. Install mingw64 first(?) [needed at all]
// 2. Install rust
// 3. Install cygwin64


pub type chtype = u32;
pub type CInt = i32;
pub type CShort = i16;
use crate::renderer::curses::{MEVENT, mmask_t};
use std::os::raw::{c_int};

#[repr(C, packed)]
pub struct cchar_t {
	pub attr: chtype,
	pub ch: [chtype; 5]
}

pub type WINDOW = *mut i8;

#[link(name = "SDL2")]
#[link(name = "SDL2main")]
#[link(name = "SDL2_test")]
#[link(name = "ole32")]
#[link(name = "oleaut32")]
#[link(name = "imm32")]
#[link(name = "pdcurses")]
extern "C" {
	pub fn initscr() -> *mut WINDOW;
	pub fn inch() -> chtype;
	pub fn refresh() -> CInt;
	pub fn addch(ch: chtype) -> CInt;
	pub fn addnstr(txt: *const i8, len: CInt) -> CInt;
	pub fn getcurx(w: WINDOW) -> CInt;
	pub fn getcury(w: WINDOW) -> CInt;
	pub fn clear() -> CInt;
	pub fn clrtoeol() -> CInt;
	pub fn flushinp() -> CInt;
	pub fn curs_set(a: CInt) -> CInt;
	pub fn wgetch(w: WINDOW) -> CInt;
	pub fn noecho() -> CInt;
	pub fn add_wch(ch: *const cchar_t) -> CInt;
	pub fn keypad(w: WINDOW, s: bool) -> CInt;
	pub fn timeout(t: CInt);
	pub fn start_color() -> CInt;
	pub fn has_colors() -> bool;
	pub fn can_change_color() -> bool;
	pub fn endwin() -> CInt;
	pub fn addstr(txt: *const i8) -> CInt;
	pub fn attron(attr: chtype) -> CInt;
	pub fn attroff(attr: chtype) -> CInt;
	
	pub fn mousemask(new_mask: mmask_t, old_mask: *mut mmask_t) -> mmask_t;
	pub fn nc_getmouse(event: *mut MEVENT) -> c_int;
	
	pub fn init_pair(_: CShort, _: CShort, _: CShort) -> CInt;
	
	#[link_name = "move"]
	pub fn mv(y: CInt, x: CInt) -> CInt;
	
	pub fn getmaxx(_: WINDOW) -> CInt;
	pub fn getmaxy(_: WINDOW) -> CInt;
	
	pub static COLOR_PAIRS: CInt;
	pub static stdscr: WINDOW;

	pub fn setlocale(t: i32, locale: *const i8) -> *mut i8;
}

//////////////
pub const LC_ALL: i32 = 0;
pub const LC_VARS: i8 = 0;

