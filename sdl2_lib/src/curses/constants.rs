pub const TEXT_MODE: bool = false;

pub type chtype = u32;
pub type CInd = i32;
pub type CShort = i16;
pub type CInt = i32;

// addch() takes as input a chtype.
// the first 8 bits are the character to be printed (ex. 'a')
// the next bits are binary flags
// the A_COLOR_PAIR_OFFSET-31 (zero indexed) bits are the index to the color pair
pub const A_REVERSE_OFFSET: chtype = 8;
pub const A_UNDERLINE_OFFSET: chtype = 9;
pub const A_CKBOARD_OFFSET: chtype = 10;
pub const A_VLINE_OFFSET: chtype = 11;
pub const A_HLINE_OFFSET: chtype = 12;
pub const A_LLCORNER_OFFSET: chtype = 13;
pub const A_LRCORNER_OFFSET: chtype = 14;
pub const A_ULCORNER_OFFSET: chtype = 15;
pub const A_URCORNER_OFFSET: chtype = 16;
pub const A_COLOR_PAIR_FLAG_OFFSET: chtype = 17; // setting to one means attron() should change the color pair
pub const A_COLOR_PAIR_OFFSET: chtype = 18; // (input >> A_COLOR_PAIR_OFFSET) is the index of the color pair

pub const COLOR_BLACK: i16 = 0;
pub const COLOR_WHITE: i16 = 7;
pub const COLOR_RED: i16 = 1;
pub const COLOR_GREEN: i16 = 2;
pub const COLOR_BLUE: i16 = 4;

pub const COLOR_CYAN: i16 = 6;
pub const COLOR_MAGENTA: i16 = 5;
pub const COLOR_YELLOW: i16 = 3;

pub const ERR: CInt = -1;

#[derive(Debug, Clone, Copy)]
pub enum CURSOR_VISIBILITY{CURSOR_INVISIBLE = 0, CURSOR_VISIBLE, CURSOR_VERY_VISIBLE}

pub struct WINDOW {}

// see ~/af/arcane_fortune/non_rust_dependencies/src/PDCurses-3.8/curses.h
// /usr/include/ncurses/ncurses.h

pub const KEY_DOWN: i32 = 0x102;
pub const KEY_UP: i32 = 0x103;
pub const KEY_LEFT: i32 = 0x104;
pub const KEY_RIGHT: i32 = 0x105;
//pub const KEY_SLEFT: i32 = 0x189;
//pub const KEY_SRIGHT: i32 = 0x192;
pub const KEY_ENTER: i32 = 0x157;
pub const KEY_MOUSE: i32 = 0x199;

pub const KEY_HOME: i32 = 0x106;
//pub const KEY_END: i32 = 0x166;
//pub const KEY_END: i32 = 0550;

pub const KEY_BACKSPACE: i32 = 0x107;
pub const KEY_WINDOW: i32 = 0xFFFF;
//pub const KEY_BACKSPACE: i32 = 127;
pub const KEY_DC: i32 = 0x14a;
pub const KEY_ESC: i32 = 27;

pub const fn A_CHARTEXT() -> chtype {0xFF}
pub const fn A_UNDERLINE() -> chtype {1 << A_UNDERLINE_OFFSET}
pub const fn A_REVERSE() -> chtype {1 << A_REVERSE_OFFSET}
pub const fn A_DIM() -> chtype {0}
pub const fn A_BOLD() -> chtype {0}

pub const fn endwin() {}
pub const fn stdscr() -> WINDOW {WINDOW {}}
pub const fn can_change_color() -> bool {true}
pub const fn has_colors() -> bool {true}
pub const fn start_color() {}
pub const fn COLOR_PAIRS() -> CInt {256}

pub const fn ACS_CKBOARD() -> chtype {(1 << A_CKBOARD_OFFSET) | (' ' as u8 as chtype)}
pub const fn ACS_VLINE() -> chtype  {(1 << A_VLINE_OFFSET) | ('x' as u8 as chtype)}
pub const fn ACS_HLINE() -> chtype {(1 << A_HLINE_OFFSET) | ('q' as u8 as chtype)}
pub const fn ACS_LLCORNER() -> chtype {(1 << A_LLCORNER_OFFSET) | ('m' as u8 as chtype)}
pub const fn ACS_LRCORNER() -> chtype {(1 << A_LRCORNER_OFFSET) | ('j' as u8 as chtype)}
pub const fn ACS_ULCORNER() -> chtype {(1 << A_ULCORNER_OFFSET) | ('l' as u8 as chtype)}
pub const fn ACS_URCORNER() -> chtype {(1 << A_URCORNER_OFFSET) | ('k' as u8 as chtype)}


