// https://localhost:20082/notebooks/modeling_notebooks/logo_generation.ipynb
// maketext.io; the Atomic Age font -- last accessed august 19, 2020

pub const LOGO_ARCANE: &'static [u8] = &[0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,255,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,0,255,255,0,0,0,255,255,255,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,0,255,255,0,255,255,0,0,255,255,0,0,0,0,255,255,255,0,0,0,0,0,255,255,0,0,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,0,0,0,0,0,0,255,255,0,0,0,0,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,255,255,255,0,0,0,255,255,0,0,0,0,255,255,255,255,255,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,0,0,0,255,255,0,0,0,0,0,255,255,255,255,0,0,0,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; 

pub const N_ROWS_ARCANE_LOGO: usize = 9;
pub const N_COLS_ARCANE_LOGO: usize = 82;

pub const LOGO_FORTUNE: &'static [u8] = &[0,0,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,255,255,255,255,255,255,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,255,255,255,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,0,0,0,0,0,0,255,255,0,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,255,255,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,0,0,0,0,0,0,255,255,0,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,0,0,0,0,0,0,255,255,0,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,255,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,255,255,255,255,255,255,255,255,255,0,0,0,0,0,255,255,0,0,0,0,255,255,255,255,255,0,0,0,255,255,255,255,255,255,0,255,255,255,255,255,255,255,255,255,255,255,255,0,0,255,255,255,0,0,0,0,255,255,255,255,0,0,0,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

pub const N_ROWS_FORTUNE_LOGO: usize = 9;
pub const N_COLS_FORTUNE_LOGO: usize = 89;

pub const LOGO_HEIGHT: usize = N_ROWS_ARCANE_LOGO + N_ROWS_FORTUNE_LOGO;
pub const LOGO_WIDTH: usize = N_COLS_FORTUNE_LOGO;

