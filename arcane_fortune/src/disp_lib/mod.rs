pub const MAX_DELAY_FRAMES: CInt = 100; // max delay between frame printings when no keys pressed

pub const ALT_DELAY: u32 = 490; // min time req. before alternating to the next unit on screen (when multp. per plot)

use std::os::raw::c_int;
pub const CWHITE: c_int = 53;
// ^ some versions of ncurses don't allow re-writing of the first (8?) color pairs
//   hence setting this to a higher value allows it to be re-defined (note ncurses statically compiled (v6.2 allows
//	redefining but not the centos 8 ncurses)

#[cfg(feature="sdl")]
extern crate sdl2_lib;
#[cfg(feature="sdl")]
pub use sdl2_lib::*;

#[cfg(not(feature="sdl"))]
pub mod curses;
#[cfg(not(feature="sdl"))]
pub use curses::*;

