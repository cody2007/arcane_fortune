use crate::disp_lib::*;
use crate::player::Player;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::config_load::*;
use crate::saving::*;

pub type CInd = i32;

pub const CRED: CInd = 1;
pub const CGREEN: CInd = 2;
pub const CYELLOW: CInd = 3;
pub const CBLUE: CInd = 4;
pub const CMAGENTA: CInd = 5;
pub const CCYAN: CInd = 6;
pub const CBLACK: CInd = 7;
pub const CGRAY: CInd = 8;

//pub const N_ARABILITY_LVLS: usize = 9;
pub const CGREEN5: CInd = 11;
pub const CGREEN4: CInd = 12;
pub const CGREEN3: CInd = 13;
pub const CGREEN2: CInd = 14;
pub const CGREEN1: CInd = 15;

pub const CSAND1: CInd = 20;
pub const CSAND2: CInd = 21;
pub const CSAND3: CInd = 22;
pub const CSAND4: CInd = 23;

pub const CSNOW2: CInd = 24;
pub const CSNOW3: CInd = 25;
pub const CSNOW4: CInd = 26;

pub const CLOGO: CInd = 38;

pub const CREDGREEN: CInd = 16;
pub const CREDBLUE: CInd = 17;
pub const CREDGREEN5: CInd = 18;
pub const CREDGRAY: CInd = 19;

pub const CREDGREEN4: CInd = 27;
pub const CREDGREEN3: CInd = 28;
pub const CREDGREEN2: CInd = 29;
pub const CREDGREEN1: CInd = 30;

pub const CREDSAND1: CInd = 31;
pub const CREDSAND2: CInd = 32;
pub const CREDSAND3: CInd = 33;
pub const CREDSAND4: CInd = 34;

pub const CREDSNOW2: CInd = 35;
pub const CREDSNOW3: CInd = 36;
pub const CREDSNOW4: CInd = 37;

pub const CSHALLOW_WATER: CInd = 39; // 27: rgb(0,95,255)
pub const CDEEP_WATER: CInd = 40; // 17: rgb(0,0,95)

pub const CREDSHALLOW_WATER: CInd = 41; // 99: rgb(135,95,255)
pub const CREDDEEP_WATER: CInd = 42; // 53: rgb(95,0,95)

pub const CGREENWHITE: CInd = 43;
pub const CBLUEWHITE: CInd = 44;

pub const CDARKRED: CInd = 45;
pub const CDARKGRAY: CInd = 46;

pub const CBLUERED0: CInd = 47;
pub const CBLUERED1: CInd = 48;
pub const CBLUERED2: CInd = 49;
pub const CBLUERED3: CInd = 50;
pub const CBLUERED4: CInd = 51;
pub const CBLUERED5: CInd = 52;

const FG_BLACK_OFFSET: i16 = 55; // note: see disp_lib/mod.rs (CWHITE; it should be less than the number defined here)

pub const ESC_COLOR: CInd = CSAND4;

pub const PLAYER_COLORS: &[i32] = &[1, 2, 3, CREDGREEN4, 5, 6, CBLUERED2, CREDSAND1, CGREEN4, 38];
// ^ zeroth entry is the human player

pub const NOBILITY_COLOR: CInd = CREDBLUE;

#[derive(PartialEq, Clone)]
pub struct DispSettings {
	pub limit_schars: bool,
	pub limit_colors: bool
}

impl_saving!{DispSettings {limit_schars, limit_colors}}

#[derive(Clone)]
pub struct DispChars {
	pub land_char: u64,
	
	pub hline_char: u64,
	pub vline_char: u64,
	pub llcorner_char: u64,
	pub lrcorner_char: u64,
	pub ulcorner_char: u64,
	pub urcorner_char: u64,
	
	pub shortcut_indicator: u64,
}

pub fn white_fg(color: CInt) -> CInt {
	color + FG_BLACK_OFFSET as CInt
}

pub fn init_color_pairs(disp_settings: &DispSettings, d: &mut DispState) -> DispChars {
	const COLOR_CONFIG_FILE: &str = "config/colors.txt";
	let key_sets = config_parse(read_file(COLOR_CONFIG_FILE));
	
	let ret_color = |nm| {
		for keys in key_sets.iter() {
			if let Some(key) = keys.iter().find(|k| k.key == nm) {
				// an integer was supplied
				return if let Result::Ok(val) = key.val.parse() {
					val
				// does this match a known default color?
				}else{
					match key.val.as_str() {
						"WHITE" => COLOR_WHITE,
						"RED" => COLOR_RED,
						"GREEN" => COLOR_GREEN,
						"YELLOW" => COLOR_YELLOW,
						"BLUE" => COLOR_BLUE,
						"MAGENTA" => COLOR_MAGENTA,
						"CYAN" => COLOR_CYAN,
						"BLACK" => COLOR_BLACK,
						_ => {
							const DEFAULT_CONFIG_COLORS: &[&str] = &["WHITE", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "BLACK"];
							panicq!("Could not interpret input `{}` for color configuration value `{}` in {}. allowed values: {:#?}",
								key.val, key.key, COLOR_CONFIG_FILE, DEFAULT_CONFIG_COLORS);
						}
					}
				};
			}
		}
		panicq!("could not find required color configuration `{}` in {}", nm, COLOR_CONFIG_FILE);
	};
	
	let (black, white) = if disp_settings.limit_colors == false {
			(ret_color("black"), ret_color("white"))
		}else{
			(ret_color("8_color_black"), ret_color("8_color_white"))
		};
	
	for (color_offset, fg_color) in [(0, black), (FG_BLACK_OFFSET, white)].iter().cloned() {
		macro_rules! set{($color: expr, $bg: expr) => {
			assertq!($color < FG_BLACK_OFFSET as CInd, "all color indices must be less than FG_BLACK_OFFSET");
			if fg_color == black {
				d.init_pair($color as i16, ret_color($bg), fg_color)
			}else{
				d.init_pair($color as i16 + color_offset, fg_color, ret_color($bg))
			}
		};};
		
		set!(CWHITE, "white");
		set!(CRED, "red");
		set!(CGREEN, "green");
		set!(CYELLOW, "yellow");
		set!(CBLUE, "blue");
		set!(CMAGENTA, "magenta");
		set!(CCYAN, "cyan");
		set!(CBLACK, "black");
		
		set!(CGREENWHITE, "green");
		set!(CBLUEWHITE, "blue");
		
		if disp_settings.limit_colors == false { // use 256 range
			set!(CGRAY, "gray");
			
			d.init_pair(9, black, ret_color("green")); //?
			d.init_pair(10, black, ret_color("blue"));
			
			set!(CGREEN5, "green5"); // meadow
			set!(CGREEN4, "green4");//28); // wetland
			set!(CGREEN3, "green3");//34); // tropical broadleaf
			set!(CGREEN2, "green2");//40); // broadleaf forest
			set!(CGREEN1, "green1");//40);//46); // mixed forest
			
			set!(CSAND1, "sand1");//112); // heath
			set!(CSAND2, "sand2");//106);//100);//106); // prarie
			set!(CSAND3, "sand3");//94);//100); // savanna
			set!(CSAND4, "sand4");//166);//94); // desert
			
			set!(CSNOW2, "snow2"); // steppe
			set!(CSNOW3, "snow3");//245); // pine forest
 			set!(CSNOW4, "snow4"); // tundra
			
			// submap
			set!(CREDGREEN, "red_green");
			set!(CREDBLUE, "red_blue");
			set!(CREDGREEN5, "red_green5");
			set!(CREDGRAY, "red_gray");
			
			set!(CREDGREEN4, "red_green4");
			set!(CREDGREEN3, "red_green3");
			set!(CREDGREEN2, "red_green2");
			set!(CREDGREEN1, "red_green1");
			
			set!(CREDSAND1, "red_sand1");
			set!(CREDSAND2, "red_sand2");
			set!(CREDSAND3, "red_sand3");
			set!(CREDSAND4, "red_sand4");
			
			set!(CREDSNOW2, "red_snow2");
			set!(CREDSNOW3, "red_snow3"); // 203
			set!(CREDSNOW4, "red_snow4"); // 217
			
			set!(CLOGO, "logo");
			
			set!(CSHALLOW_WATER, "shallow_water");//21);
			set!(CDEEP_WATER, "deep_water");//19);
			
			set!(CREDSHALLOW_WATER, "red_shallow_water");//99);
			set!(CREDDEEP_WATER, "red_deep_water");
			
			set!(CDARKRED, "dark_red");
			set!(CDARKGRAY, "dark_gray"); // note: 236 is too dark
			
			set!(CBLUERED0, "bluered0");
			set!(CBLUERED1, "bluered1");
			set!(CBLUERED2, "bluered2");
			set!(CBLUERED3, "bluered3");
			set!(CBLUERED4, "bluered4");
			set!(CBLUERED5, "bluered5");
			
		}else{ // 8 colors
			set!(CGRAY, "8_color_gray");
			
			d.init_pair(9, black, ret_color("8_color_green"));
			d.init_pair(10, black, ret_color("8_color_blue"));
			
			set!(CGREEN5, "8_color_green5");
			set!(CGREEN4, "8_color_green4");
			set!(CGREEN3, "8_color_green3");
			set!(CGREEN2, "8_color_green2");
			set!(CGREEN1, "8_color_green1");
			
			set!(CSAND1, "8_color_sand1");
			set!(CSAND2, "8_color_sand2");
			set!(CSAND3, "8_color_sand3");
			set!(CSAND4, "8_color_sand4");
			
			set!(CSNOW2, "8_color_snow2");
			set!(CSNOW3, "8_color_snow3");
			set!(CSNOW4, "8_color_snow4");
			
			// submap
			set!(CREDGREEN, "8_color_red_green");
			set!(CREDBLUE, "8_color_red_blue");
			set!(CREDGREEN5, "8_color_green5");
			set!(CREDGRAY, "8_color_red_gray");
			
			set!(CREDGREEN4, "8_color_red_green4");
			set!(CREDGREEN3, "8_color_red_green3");
			set!(CREDGREEN2, "8_color_red_green2");
			set!(CREDGREEN1, "8_color_red_green1");
			
			set!(CREDSAND1, "8_color_red_sand1");
			set!(CREDSAND2, "8_color_red_sand2");
			set!(CREDSAND3, "8_color_red_sand3");
			set!(CREDSAND4, "8_color_red_sand4");
			
			set!(CREDSNOW2, "8_color_red_snow2");
			set!(CREDSNOW3, "8_color_red_snow3"); // 203
			set!(CREDSNOW4, "8_color_red_snow4"); // 217
			
			set!(CLOGO, "8_color_logo");
			
			set!(CSHALLOW_WATER, "8_color_shallow_water");
			set!(CDEEP_WATER, "8_color_deep_water");
			
			set!(CREDSHALLOW_WATER, "8_color_red_shallow_water");
			set!(CREDDEEP_WATER, "8_color_red_deep_water");
			
			set!(CDARKRED, "8_color_dark_red");
			set!(CDARKGRAY, "8_color_dark_gray");
			
			set!(CBLUERED0, "8_color_bluered0");
			set!(CBLUERED1, "8_color_bluered1");
			set!(CBLUERED2, "8_color_bluered2");
			set!(CBLUERED3, "8_color_bluered3");
			set!(CBLUERED4, "8_color_bluered4");
			set!(CBLUERED5, "8_color_bluered5");
		}
	}
	
	let shortcut_indicator = || A_UNDERLINE() | COLOR_PAIR(CCYAN);
	
	let disp_chars = if disp_settings.limit_schars == false { // use all chars
				DispChars {
					land_char: ACS_CKBOARD() as u64,
					
					hline_char: ACS_HLINE() as u64,
					vline_char: ACS_VLINE() as u64,
					llcorner_char: ACS_LLCORNER() as u64,
					lrcorner_char: ACS_LRCORNER() as u64,
					ulcorner_char: ACS_ULCORNER() as u64,
					urcorner_char: ACS_URCORNER() as u64,
					
					shortcut_indicator: shortcut_indicator() as u64
				}
			}else{
				DispChars { // limit to ASCII
					land_char: '*' as u64,
					
					hline_char: '-' as u64,
					vline_char: '|' as u64,
					llcorner_char: '-' as u64,
					lrcorner_char: '-' as u64,
					ulcorner_char: '-' as u64,
					urcorner_char: '-' as u64,
				
					shortcut_indicator: shortcut_indicator() as u64
				}
			};

	d.clear();
	d.attroff(0); // hack (for the ncurses binding code to load in the settings for the default white text -- see that binding for more info)
	disp_chars
}

pub fn set_player_color(player: &Player, on: bool, d: &mut DispState){
	if on { d.attron(COLOR_PAIR(player.personalization.color)); }else{
		d.attroff(COLOR_PAIR(player.personalization.color)); }
}

