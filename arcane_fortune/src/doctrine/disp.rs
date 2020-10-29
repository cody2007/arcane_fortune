use crate::renderer::*;
use crate::disp::*;
use crate::disp::window::keys::tree_window_movement;
use super::*;
use crate::localization::Localization;
use crate::containers::*;

pub const DOCTRINE_SZ_PRINT: ScreenSz = ScreenSz {h: 5, w: 29, sz: 5*29};

const C_CURRENT: CInd = CYELLOW; // i.e., prevailing doctrinality
const C_PRESENT: CInd = CCYAN; // those which have any buildings in the empire

pub struct DoctrineWindowState {
	pub sel: Option<SmSvType>,
	pub sel_mv: TreeSelMv,
	pub tree_offsets: Option<TreeOffsets>,
	pub prev_auto_turn: AutoTurn
}

impl DoctrineWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, temps: &Templates, pstats: &Stats, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {	
		/////// init selection
		if self.sel.is_none() {
			self.sel = Some(pstats.doctrine_template.id as SmSvType);
		}
		
		let disp_properties = dstate.print_tree(&temps.doctrines, pstats, &mut self.sel, &mut self.sel_mv, &mut self.tree_offsets, DOCTRINE_SZ_PRINT);
		
		let w = dstate.iface_settings.screen_sz.w as i32;
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		let kbd = &dstate.kbd;
		
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
			
			center_txt!(&dstate.local.Doctrine_tree, w);
			
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
				nl!(1);
			}
			
			d.attron(COLOR_PAIR(ESC_COLOR));
			d.addstr(&format!("<{}>", l.Arrow_keys));
			d.attroff(COLOR_PAIR(ESC_COLOR));
			d.addstr(": ");
			d.addstr(&l.Change_selection);
			
			////////////// colors
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
			};};
			
			nl!();show_key!(CRED, &l.selected);
			nl!();show_key!(C_CURRENT, &l.prevailing_doctrinality);
			nl!(1);show_key!(C_PRESENT, &l.doctrine_w_presence);
		}
		
		// show infobox for selected doctrine (ex bonuses and bldgs it can unlock)
		if let Some((mut row, mut col, doc_ind)) = disp_properties.sel_loc {
			row += 1;
			// off the top edge of the screen
			if row < 0 || col < 0 {return UIModeControl::UnChgd;}
			
			let doc = &temps.doctrines[doc_ind as usize];
			
			let bldgs_unlocks = doc.bldgs_unlocks(temps.bldgs);
			
			// if there are no bonuses, don't show anything
			if doc.health_bonus == 0. && doc.crime_bonus == 0. && doc.pacifism_bonus == 0. && 
				doc.happiness_bonus == 0. && doc.tax_aversion == 0. && bldgs_unlocks.len() == 0 {
				return UIModeControl::UnChgd;
			}
			
			let window_w = 29;
			if window_w > w {return UIModeControl::UnChgd;} // screen too small
			
			// off the right side of the screen
			if (col + window_w) > w {
				row += 5;
				col = w - window_w - 2;
			}
			
			// print window top line
			{
				d.mv(row, col); row += 1;
				d.addch(dstate.chars.ulcorner_char);
				for _ in 0..window_w {d.addch(dstate.chars.hline_char);}
				d.addch(dstate.chars.urcorner_char);
			}
			
			macro_rules! lr_txt{($l_txt: expr, $r_txt: expr) => {
				// clear line
				d.mv(row,col);
				d.addch(dstate.chars.vline_char);
				for _ in 0..window_w {d.addch(' ');}
				d.addch(dstate.chars.vline_char);
				
				// print txt
				d.mv(row,col+2);
				d.attron(COLOR_PAIR(CGREEN));
				d.addstr($l_txt);
				d.attroff(COLOR_PAIR(CGREEN));
				d.mv(row, col + window_w - $r_txt.len() as i32);
				d.addstr($r_txt);
				row += 1;
			};};
			
			// bonuses and bldgs discovered by this doctrine
			{
				if doc.health_bonus != 0. {
					lr_txt!(&l.Health_bonus, &format!("{}", doc.health_bonus));
				}
				
				if doc.crime_bonus != 0. {
					//lr_txt!(&l.Safety_bonus, &format!("{}", -d.crime_bonus));
					lr_txt!(&l.Crime_bonus, &format!("{}", doc.crime_bonus));
				}
				
				if doc.pacifism_bonus > 0. {
					lr_txt!(&l.Pacifism_bonus, &format!("{}", doc.pacifism_bonus));
				}else if doc.pacifism_bonus < 0. {
					lr_txt!(&l.Militarism_bonus, &format!("{}", -doc.pacifism_bonus));
				}
				
				if doc.happiness_bonus != 0. {
					lr_txt!(&l.Happiness_bonus, &format!("{}", doc.happiness_bonus));
				}
				
				if doc.tax_aversion != 0. {
					lr_txt!(&l.Tax_aversion, &format!("{}", doc.tax_aversion));
				}
				
				if let Some(bldg_unlocked) = bldgs_unlocks.first() {
					lr_txt!(&l.Discovers, &bldg_unlocked.nm[l.lang_ind]);
					for bldg_unlocked in bldgs_unlocks.iter().skip(1) {
						lr_txt!("", &bldg_unlocked.nm[l.lang_ind]);
					}
				}
			}
			
			// print window bottom line
			{
				d.mv(row,col);
				d.addch(dstate.chars.llcorner_char);
				for _ in 0..window_w {d.addch(dstate.chars.hline_char);}
				d.addch(dstate.chars.lrcorner_char);
			}
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, dstate: &DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		self.sel_mv = TreeSelMv::None;
		
		tree_window_movement(dstate.key_pressed, &mut self.sel_mv, &mut self.tree_offsets, &DOCTRINE_SZ_PRINT, &dstate.kbd);
		UIModeControl::UnChgd
	}
}

impl TreeTemplate for DoctrineTemplate {
	fn reqs(&self) -> Option<Vec<SmSvType>> {
		if let Some(req) = self.pre_req_ind {
			Some(vec![req as SmSvType])
		}else {None}
	}
	
	fn nm(&self, l: &Localization) -> String {
		self.nm[l.lang_ind].clone()
	}
	
	fn line_color(&self, row: usize, pstats: &Stats) -> CInt {
		// title text
		if row == 0 {
			// current doctrinality
			if pstats.doctrine_template == self {
				C_CURRENT
			}else if pstats.locally_logged.doctrinality_sum[self.id] != 0. {
				C_PRESENT
			}else{
				CGRAY
			}
		} else {CWHITE}
	}
	
	fn requirements_txt(&self, pstats: &Stats, l: &Localization) -> String { // ex doctrinality points
		if self.bldg_req > 0. {
			format!("{} {:.1}/{}", l.Bldg_pts, pstats.locally_logged.doctrinality_sum[self.id], self.bldg_req)
		}else {String::new()}
	}
}

