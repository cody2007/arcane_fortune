use crate::renderer::*;
use crate::player::Stats;
use crate::saving::SmSvType;
use crate::disp::*;
use crate::disp::window::keys::tree_window_movement;
use crate::localization::Localization;
use crate::containers::*;

use super::*;

pub const TECH_SZ_PRINT: ScreenSz = ScreenSz {h: 5, w: 22, sz: 5*22};

const C_UN_DISCOV: CInd = CCYAN;
const C_DISCOV: CInd = CGRAY;
const C_SCHEDULED: CInd = CYELLOW;

pub struct TechWindowState {
	pub sel: Option<SmSvType>,
	pub sel_mv: TreeSelMv,
	pub tree_offsets: Option<TreeOffsets>,
	pub prompt_tech: bool, // tell player we finished researching tech
	pub prev_auto_turn: AutoTurn
}

impl TechWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, temps: &Templates, pstats: &Stats,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		////// init tech sel
		if self.sel.is_none() {
			// tech is scheduled
			if let Some(tech_scheduled) = pstats.techs_scheduled.last() {
				self.sel = Some(*tech_scheduled);
			}else if temps.techs.len() > 0 {
				self.sel = Some(0);//templates.len() as SmSvType - 1);
			}else {panicq!("No techs present. Check configuration file.");}
		}
		
		let disp_properties = dstate.print_tree(temps.techs, pstats, &mut self.sel, &mut self.sel_mv, &mut self.tree_offsets, TECH_SZ_PRINT);
		
		let w = dstate.iface_settings.screen_sz.w as i32;
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		
		{ /////////////////////////////// plotting before/after the tree
			let title_c = COLOR_PAIR(CGREEN);
			
			macro_rules! center_txt{($txt: expr, $w: expr) => {
				let g = ($w as usize - $txt.len())/2;
				let mut sp = String::new();
				for _ in 0..g {sp.push(' ');}
				d.attron(title_c);
				d.addstr(&format!("{}{}", sp, $txt));
				d.attroff(title_c);
			};}
			
			center_txt!(if self.prompt_tech {&l.Select_a_new_tech} else {&l.Technology_tree}, w);
			
			let gap_width = "     ".len() as i32;
			let key_width = "<Arrow keys>: Change selection".len() as i32;
			let color_width = "undiscovered tech".len() as i32;
			let txt_width = key_width + gap_width + color_width;
			
			let mut row = disp_properties.instructions_start_row;
			let mut col = (w - txt_width)/2;
			
			macro_rules! nl{() => {d.mv(row, col); row += 1;};
					($last:expr) => {d.mv(row,col);};}
			
			/////// instructions
			nl!();
			center_txt!(&l.Keys, key_width);
			nl!(); nl!();
			
			let kbd = &dstate.kbd;
			
			// scroll view
			if !screen_reader_mode() && (disp_properties.down_scroll || disp_properties.right_scroll) {
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
			if !screen_reader_mode() {
				row = disp_properties.instructions_start_row;
				col += (key_width + gap_width) as i32;
				
				nl!();
				center_txt!(&l.Color_indicators, color_width);
				nl!();
				
				macro_rules! show_key{($c:expr, $txt:expr) => {
					d.attron(COLOR_PAIR($c));
					d.addch(dstate.chars.land_char);
					d.attroff(COLOR_PAIR($c));
					d.addstr(" ");
					d.addstr($txt);
				};}
				
				nl!();show_key!(CRED, &l.selected_tech);
				nl!();show_key!(C_UN_DISCOV, &l.undiscovered_tech);
				nl!();show_key!(C_DISCOV, &l.discovered_tech);
				nl!(1);show_key!(C_SCHEDULED, &l.researching);
			}
		}
		
		// sets txt cursor to the title
		macro_rules! exit_and_set_txt_cursor{() => {
			if let Some((row, col, _)) = disp_properties.sel_loc {
				d.mv(row, col);
			}
			return UIModeControl::UnChgd;
		};}
		
		// show infobox for selected tech (units & bldgs it can unlock)
		if let Some((mut row, mut col, t)) = disp_properties.sel_loc {
			// show under the tech instead of beside (on the right)
			if screen_reader_mode() {
				row = 12;
				col = 0;
			}else{
				// off the top edge of the screen
				if row < 0 || col < 0 {return UIModeControl::UnChgd;}
			}
			
			// if the tech isn't used to discover any units or bldgs, return and set text cursor for screen readers
			if !temps.units.iter().any(|ut| if let Some(tech_req) = &ut.tech_req {tech_req.contains(&(t as usize))} else {false}) &&
			   !temps.bldgs.iter().any(|bt| if let Some(tech_req) = &bt.tech_req {tech_req.contains(&(t as usize))} else {false}) {
				exit_and_set_txt_cursor!();
			}
			
			const WINDOW_W: i32 = 20;
			if WINDOW_W > w {return UIModeControl::UnChgd;} // screen too small
			
			// off the right side of the screen
			if (col + WINDOW_W) > w {
				row += 5;
				col = w - WINDOW_W - 2;
			}
			
			// print window top line
			if !screen_reader_mode() {
				d.mv(row, col); row += 1;
				d.addch(dstate.chars.ulcorner_char);
				for _ in 0..WINDOW_W {d.addch(dstate.chars.hline_char);}
				d.addch(dstate.chars.urcorner_char);
			}
			
			macro_rules! clr_ln{() => {
				if !screen_reader_mode() {
					d.mv(row,col);
					d.addch(dstate.chars.vline_char);
					for _ in 0..WINDOW_W {d.addch(' ');}
					d.addch(dstate.chars.vline_char);
				}
			}}
			
			macro_rules! lr_txt{($l_txt: expr, $r_txt: expr) => {
				clr_ln!();
				
				// print txt
				d.mv(row,col+2);
				d.addstr($l_txt);
				d.mv(row, col + WINDOW_W - $r_txt.len() as i32);
				d.addstr($r_txt);
				row += 1;
			};}
			
			
			{ // units, bldgs, and resources discovered by the selected technology
				macro_rules! print_unit_bldg_discov{($templates: expr) => {
					let mut any_discov = false;
					for template in $templates.iter() {
						if let Some(tech_req) = &template.tech_req {
							if tech_req.contains(&(t as usize)) {
								// first entry
								if !any_discov {
									clr_ln!();
																	
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
				};}
				
				print_unit_bldg_discov!(temps.units); // temp
				print_unit_bldg_discov!(temps.bldgs);
				
				{ // resources discovered (stored slightly differently in `temps.resources`)
					let mut any_discov = false;
					for template in temps.resources.iter() {
						if template.tech_req.contains(&(t as usize)) {
							// first entry
							if !any_discov {
								clr_ln!();
								
								// print txt
								d.mv(row,col+2);
								d.attron(COLOR_PAIR(CGREEN));
								d.addstr(&l.Discovers);
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
			if !screen_reader_mode() {
				d.mv(row,col);
				d.addch(dstate.chars.llcorner_char);
				for _ in 0..WINDOW_W {d.addch(dstate.chars.hline_char);}
				d.addch(dstate.chars.lrcorner_char);
			}
		}
		
		// set text cursor at the name of the tech for screen readers
		exit_and_set_txt_cursor!();
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, pstats: &mut Stats, temps: &Templates,
			dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		self.sel_mv = TreeSelMv::None;
		
		tree_window_movement(dstate.key_pressed, &mut self.sel_mv, &mut self.tree_offsets, &TECH_SZ_PRINT, &dstate.kbd);
		
		// start researching tech
		if dstate.key_pressed == dstate.kbd.enter {
			if let Some(sel) = self.sel {
				// not already scheduled
				if !pstats.techs_scheduled.contains(&sel) {
					pstats.start_researching(sel, temps.techs);
					
				// unschedule
				}else{
					pstats.stop_researching(sel, temps.techs);
				}
			}
		}
		UIModeControl::UnChgd
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
	
	fn requirements_txt(&self, pstats: &Stats, l: &Localization) -> String {
		let mut txt = l.Reqs_X_rsrch.replace("[]", &format!("{}", self.research_req));
		
		if screen_reader_mode() {
			txt.push_str(". ");
			txt.push_str(match pstats.techs_progress[self.id as usize] {
				TechProg::Finished => {&l.Your_empire_has_discov_tech}
				TechProg::Prog(_) => {
					if let Some(que_pos) = pstats.techs_scheduled.iter().position(|&sched_id| sched_id == self.id) {
						if que_pos == (pstats.techs_scheduled.len() - 1) {
							&l.Your_empire_hasnt_discov_tech_but_researching
						}else{
							&l.Your_empire_hasnt_discov_tech_but_sched
						}
					}else{
						&l.Your_empire_hasnt_discov_tech
					}
				}
			});
		}
		
		txt
	}
}

