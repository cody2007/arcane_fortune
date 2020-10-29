/*
    Copyright © 2013 Free Software Foundation, Inc
    See licensing in LICENSE file

    Derivation of ncurses Rust module
    Original author: Jesse 'Jeaye' Wilkerson
*/

///////////////////////////////////
// Linux: for statically linked ncurses 6.1 w/ musl (produces FULLY static binaries):
//
// 1. Add rust support:
//		rustup target add x86_64-unknown-linux-musl
//
// 2. Build musl (located in ~/af/arcane_fortune/non_rust_dependencies/src/musl-1.1.24):
//		./configure; make; sudo make install # (installs to non-overwriting locations; musl-gcc)
//
// Todo: add entries in /usr/share/terminfo to with-fallbacks option in step 3
// 		wget https://github.com/kovidgoyal/kitty/blob/master/terminfo/x/xterm-kitty
//		cp xterm-kitty /usr/share/terminfo/x/
//
// 3. Compile ncurses 6.1 (w/ musl):
//		cd ~/af/arcane_fortune/non_rust_dependencies/tmp_build_dirs
//		export CC=/usr/local/musl/bin/musl-gcc
//		make clean
//		/home/tapa/af/arcane_fortune/non_rust_dependencies/src/ncurses-6.2/configure --prefix=/home/tapa/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses --enable-termcap --enable-widec --disable-database --without-ada --without-cxx --without-cxx-binding --disable-db-install --without-manpages --without-progs --with-fallbacks=xterm-256color,xterm,linux,xterm-kitty,xfce,xterm-x11mouse,xterm-xfree86,gnome,gnome-256color,konsole,konsole-256color,putty-256color,putty,xterm-utf8,Apple_Terminal --enable-ext-colors --with-gpm --with-sysmouse --enable-ext-mouse
//		#/home/tapa/af/arcane_fortune/non_rust_dependencies/src/ncurses-6.1/configure --prefix=/home/tapa/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses --enable-termcap --enable-widec --disable-database --without-ada --without-cxx --without-cxx-binding --disable-db-install --without-manpages --without-progs --with-fallbacks=xterm-256color,xterm,linux,xterm-kitty,xfce,xterm-x11mouse,xterm-xfree86,gnome,gnome-256color,konsole,konsole-256color,putty-256color,putty,xterm-utf8,Apple_Terminal --enable-ext-colors --with-gpm --with-sysmouse --enable-ext-mouse
//		#/home/tapa/af/arcane_fortune/non_rust_dependencies/src/ncurses-6.1/configure --prefix=/home/tapa/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses --enable-termcap --enable-widec --disable-database --without-ada --without-cxx --without-cxx-binding --disable-db-install --without-manpages --without-progs --with-fallbacks=xterm-256color,xterm,linux --enable-ext-colors
//		make; make install
//		cp ~/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses/lib/libncursesw.a ~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-musl/lib/
//		cp ~/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses/lib/libncursesw.a ~/.rustup/toolchains/1.46.0-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-musl/lib/
//
// compile af:
//	cd ~/af/arcane_fortune/src
//	cargo build --target=x86_64-unknown-linux-gnu [--release]


//////////////////////////////////////////////
// Linux: for statically linking ncurses 6.1 w/ GLIBC
//
// ****does not produce fully static libraries because GLIBC must be dynamically loaded -- results in needing to compile
//  on older distributions because otherwise the resulting binary will require whatever version of GLIBC it was compiled with****
//
// Build ncurses 6.1:
//	make clean
//	./configure --prefix=/home/tapa/Downloads/ncurses --enable-termcap --enable-widec --disable-database --without-ada --without-cxx --without-cxx-binding --disable-db-install --without-manpages --without-progs --with-fallbacks=xterm-256color,xterm,linux --enable-ext-colors
//	make; make install
//
//	Copy /home/tapa/Downloads/ncurses/lib/ncursesw.a to:
//
// 	~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib/
///////////////

// /home/tapa/af/arcane_fortune/non_rust_dependencies/prefix_dirs/ncurses/include/ncursesw/curses.h

pub type chtype = u32;
pub type CInt = i32;
pub type CShort = i16;
use crate::renderer::curses::{MEVENT, mmask_t};
use std::os::raw::{c_int};

// if characters are not printing correctly verify the
// size of wchar_t in C. compile/run in C:
// printf("%i\n", LC_ALL, sizeof(wchar_t));

#[repr(C, packed)]
pub struct cchar_t {
	pub attr: chtype,
	pub ch: [chtype; 5]
}

pub type WINDOW = *mut i8;

#[cfg(target_os = "linux")]
#[link(name = "ncursesw")]
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
	pub fn getch() -> CInt;
	pub fn noecho() -> CInt;
	pub fn add_wch(ch: *const cchar_t) -> CInt;
	pub fn keypad(w: WINDOW, s: bool) -> CInt;
	pub fn timeout(t: CInt);
	pub fn start_color() -> CInt;
	pub fn has_colors() -> bool;
	pub fn set_escdelay(t: CInt) -> CInt;
	pub fn can_change_color() -> bool;
	pub fn endwin() -> CInt;
	pub fn addstr(txt: *const i8) -> CInt;
	pub fn attron(attr: chtype) -> CInt;
	pub fn attroff(attr: chtype) -> CInt;
	
	pub fn mousemask(new_mask: mmask_t, old_mask: *mut mmask_t) -> mmask_t;
	pub fn getmouse(event: *mut MEVENT) -> c_int;
	
	pub fn init_pair(_: CShort, _: CShort, _: CShort) -> CInt;
	pub fn COLOR_PAIR(pair: CInt) -> CInt;
	
	#[link_name = "move"]
	pub fn mv(y: CInt, x: CInt) -> CInt;
	
	pub fn getmaxx(_: WINDOW) -> CInt;
	pub fn getmaxy(_: WINDOW) -> CInt;
	
	#[link_name = "acs_map"]
	pub static mut acs_map_intern: [chtype; 0];
	pub static COLOR_PAIRS: CInt;
	pub static stdscr: WINDOW;

	pub fn setlocale(t: i32, locale: *const i8) -> *mut i8;
}

#[cfg(target_os = "macos")]
#[link(name = "ncurses")]
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
	pub fn getch() -> CInt;
	pub fn noecho() -> CInt;
	pub fn add_wch(ch: *const cchar_t) -> CInt;
	pub fn keypad(w: WINDOW, s: bool) -> CInt;
	pub fn timeout(t: CInt);
	pub fn start_color() -> CInt;
	pub fn has_colors() -> bool;
	pub fn set_escdelay(t: CInt) -> CInt;
	pub fn can_change_color() -> bool;
	pub fn endwin() -> CInt;
	pub fn addstr(txt: *const i8) -> CInt;
	pub fn attron(attr: chtype) -> CInt;
	pub fn attroff(attr: chtype) -> CInt;
	
	pub fn mousemask(new_mask: mmask_t, old_mask: *mut mmask_t) -> mmask_t;
	pub fn getmouse(event: *mut MEVENT) -> c_int;
	
	pub fn init_pair(_: CShort, _: CShort, _: CShort) -> CInt;
	pub fn COLOR_PAIR(pair: CInt) -> CInt;
	
	#[link_name = "move"]
	pub fn mv(y: CInt, x: CInt) -> CInt;
	
	pub fn getmaxx(_: WINDOW) -> CInt;
	pub fn getmaxy(_: WINDOW) -> CInt;
	
	#[link_name = "acs_map"]
	pub static mut acs_map_intern: [chtype; 0];
	pub static COLOR_PAIRS: CInt;
	pub static stdscr: WINDOW;

	pub fn setlocale(t: i32, locale: *const i8) -> *mut i8;
}

pub fn acs_map() -> *const chtype {
	unsafe {&acs_map_intern as *const chtype}
}

//////////////
pub const LC_ALL: i32 = 0;
pub const LC_VARS: i8 = 0;

