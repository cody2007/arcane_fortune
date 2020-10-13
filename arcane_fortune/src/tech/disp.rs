use crate::disp_lib::*;
use crate::player::Stats;
use crate::saving::SmSvType;
use crate::disp::*;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;
use crate::containers::Templates;

use super::*;

pub const TECH_SZ_PRINT: ScreenSz = ScreenSz {h: 5, w: 22, sz: 5*22};

const C_UN_DISCOV: CInd = CCYAN;
const C_DISCOV: CInd = CGRAY;
const C_SCHEDULED: CInd = CYELLOW;

pub fn print_tech_tree(temps: &Templates, disp_chars: &DispChars, pstats: &Stats, 
		tech_sel: &mut Option<SmSvType>, tech_sel_mv: &mut TreeSelMv,
		tree_offsets: &mut Option<TreeOffsets>, screen_sz: ScreenSz, prompt_tech: bool,
		kbd: &KeyboardMap, l: &Localization, buttons: &mut Buttons, d: &mut DispState) {
	
	////// init tech sel
	if tech_sel.is_none() {
		// tech is scheduled
		if let Some(tech_scheduled) = pstats.techs_scheduled.last() {
			*tech_sel = Some(*tech_scheduled);
		}else if temps.techs.len() > 0 {
			*tech_sel = Some(0);//templates.len() as SmSvType - 1);
		}else {panicq!("No techs present. Check configuration file.");}
	}
	
	let disp_properties = print_tree(temps.techs, disp_chars, pstats, tech_sel, tech_sel_mv, tree_offsets, screen_sz, TECH_SZ_PRINT, l, buttons, d);
	
	let w = screen_sz.w as i32;
	
	/////////////////////////////// plotting before/after the tree
	{
		let title_c = COLOR_PAIR(CGREEN);
		
		macro_rules! center_txt{($txt: expr, $w: expr) => {
			let g = ($w as usize - $txt.len())/2;
			let mut sp = String::new();
			for _ in 0..g {sp.push(' ');}
			d.attron(title_c);
			d.addstr(&format!("{}{}", sp, $txt));
			d.attroff(title_c);
		};};
		
		center_txt!(if prompt_tech {&l.Select_a_new_tech} else {&l.Technology_tree}, w);
		
		let gap_width = "     ".len() as i32;
		let key_width = "<Arrow keys>: Change selection".len() as i32;
		let color_width = "undiscovered tech".len() as i32;
		let txt_width = key_width + gap_width + color_width;
		
		let mut row = disp_properties.instructions_start_row;
		let mut col = (w - txt_width)/2;
		
		macro_rules! nl{() => {d.mv(row, col); row += 1;};
				($last:expr) => {d.mv(row,col);};};
		
		/////// instructions
		nl!();
		center_txt!(&l.Keys, key_width);
		nl!(); nl!();
		
		if disp_properties.down_scroll || disp_properties.right_scroll {
			for key in &[kbd.left, kbd.down, kbd.right] {
				print_key_always_active(*key, l, d);
				d.addstr(", ");
			}
			
			print_key_always_active(kbd.up, l, d);
			d.addstr(": ");
			d.addstr(&l.Scroll_view);
			nl!();
		}
		
		addstr_c(&format!("<{}>", l.Arrow_keys), ESC_COLOR, d);
		d.addstr(": ");
		d.addstr(&l.Change_selection);
		
		nl!(1);
		addstr_c(&l.Enter_key, ESC_COLOR, d);
		d.addstr(": ");
		d.addstr(&l.Start_researching);
		
		////////////// colors
		row = disp_properties.instructions_start_row;
		col += (key_width + gap_width) as i32;
		
		nl!();
		center_txt!(&l.Color_indicators, color_width);
		nl!();
		
		macro_rules! show_key{($c:expr, $txt:expr) => {
			d.attron(COLOR_PAIR($c));
			d.addch(disp_chars.land_char);
			d.attroff(COLOR_PAIR($c));
			d.addstr(" ");
			d.addstr($txt);
		};};
		
		nl!();show_key!(CRED, "selected tech");
		nl!();show_key!(C_UN_DISCOV, "undiscovered tech");
		nl!();show_key!(C_DISCOV, "discovered tech");
		nl!(1);show_key!(C_SCHEDULED, "researching");
	}
	
	// show infobox for selected tech (units & bldgs it can unlock)
	if let Some((mut row, mut col, t)) = disp_properties.sel_loc {
		// off the top edge of the screen
		if row < 0 || col < 0 {return;}
		
		// if the tech isn't used to discover any units or bldgs, return
		if !temps.units.iter().any(|ut| if let Some(tech_req) = &ut.tech_req {tech_req.contains(&(t as usize))} else {false}) &&
		   !temps.bldgs.iter().any(|bt| if let Some(tech_req) = &bt.tech_req {tech_req.contains(&(t as usize))} else {false}) {
			return;
		}
		
		let window_w = 20;
		if window_w > w {return;} // screen too small
		
		// off the right side of the screen
		if (col + window_w) > w {
			row += 5;
			col = w - window_w - 2;
		}
		
		// print window top line
		{
			d.mv(row, col); row += 1;
			d.addch(disp_chars.ulcorner_char);
			for _ in 0..window_w {d.addch(disp_chars.hline_char);}
			d.addch(disp_chars.urcorner_char);
		}
		
		macro_rules! lr_txt{($l_txt: expr, $r_txt: expr) => {
			// clear line
			d.mv(row,col);
			d.addch(disp_chars.vline_char);
			for _ in 0..window_w {d.addch(' ');}
			d.addch(disp_chars.vline_char);
			
			// print txt
			d.mv(row,col+2);
			d.addstr($l_txt);
			d.mv(row, col + window_w - $r_txt.len() as i32);
			d.addstr($r_txt);
			row += 1;
		};};
		
		
		// units, bldgs, and resources discovered by the selected technology
		{
			macro_rules! print_unit_bldg_discov{($templates: expr) => {
				let mut any_discov = false;
				for template in $templates.iter() {
					if let Some(tech_req) = &template.tech_req {
						if tech_req.contains(&(t as usize)) {
							// first entry
							if !any_discov {
								// clear line
								d.mv(row,col);
								d.addch(disp_chars.vline_char);
								for _ in 0..window_w {d.addch(' ');}
								d.addch(disp_chars.vline_char);
								
								// print txt
								d.mv(row,col+2);
								d.attron(COLOR_PAIR(CGREEN));
								d.addstr(&l.Req_for);
								d.attroff(COLOR_PAIR(CGREEN));
								row += 1;
								
								any_discov = true;
							}
							lr_txt!("", &template.nm[0]);
						}
					}
				}
			};};
			
			print_unit_bldg_discov!(temps.units); // temp
			print_unit_bldg_discov!(temps.bldgs);
			
			// resources discovered (stored slightly differently in `temps.resources`)
			{
				let mut any_discov = false;
				for template in temps.resources.iter() {
					if template.tech_req.contains(&(t as usize)) {
						// first entry
						if !any_discov {
							// clear line
							d.mv(row,col);
							d.addch(disp_chars.vline_char);
							for _ in 0..window_w {d.addch(' ');}
							d.addch(disp_chars.vline_char);
							
							// print txt
							d.mv(row,col+2);
							d.attron(COLOR_PAIR(CGREEN));
							d.addstr("Discovers:");
							d.attroff(COLOR_PAIR(CGREEN));
							row += 1;
							
							any_discov = true;
						}
						lr_txt!("", &template.nm[l.lang_ind]);
					}
				}
			}
		}
		
		// print window bottom line
		{
			d.mv(row,col);
			d.addch(disp_chars.llcorner_char);
			for _ in 0..window_w {d.addch(disp_chars.hline_char);}
			d.addch(disp_chars.lrcorner_char);
		}
	}
}

impl TreeTemplate for TechTemplate {
	fn reqs(&self) -> Option<Vec<SmSvType>> {
		self.tech_req.clone()
	}
	
	fn nm(&self, l: &Localization) -> String {
		self.nm[l.lang_ind].clone()
	}
	
	fn line_color(&self, row: usize, pstats: &Stats) -> CInt {
		if row != 0 && row != 2 {
			CWHITE // not the title and not the additional research
		}else if row == 2 {
			CREDGRAY // additional research
		}else if pstats.techs_progress[self.id as usize] == TechProg::Finished { // title
			C_DISCOV
		}else if pstats.techs_scheduled.contains(&self.id) { // title, scheduled to be researched
			C_SCHEDULED
		}else{
			C_UN_DISCOV // title, undiscovered
		}
	}
	
	fn requirements_txt(&self, _: &Stats, l: &Localization) -> String {
		l.Reqs_X_rsrch.replace("[]", &format!("{}", self.research_req))
	}
}

