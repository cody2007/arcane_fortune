use std::cmp::max;
use crate::disp_lib::{DispState, COLOR_PAIR};
use crate::disp::{Buttons, ScreenSz, CGREEN, DispChars, crop_txt};
use crate::localization::Localization;
use crate::player::Personalization;
use super::*;

const BOX_WIDTH: usize = 20;
const BOX_HEIGHT: usize = 4;
const SPACE_BETWEEN_PAIRS_X: usize = 3;
const SPACE_BETWEEN_GENERATIONS_Y: usize = 2;

impl Noble {
	fn print(&self, show_parent_connection: bool, coord: ScreenCoord, disp_chars: &DispChars, l: &Localization,
			d: &mut DispState, turn: usize) {
		let y = coord.y as i32;
		let x = coord.x as i32;
		
		// top line
		{
			d.mv(y, x);
			d.addch(disp_chars.ulcorner_char);
			for col in 0..(BOX_WIDTH-2) {
				d.addch(if !show_parent_connection || col != (BOX_WIDTH-2)/2 {
					disp_chars.hline_char
				}else{
					disp_chars.llcorner_char
				});
			}
			d.addch(disp_chars.urcorner_char);
		}
		
		macro_rules! center_txt{($row: expr, $txt: expr) => {
			d.mv(y+$row, x);
			d.addch(disp_chars.vline_char);
			
			let txt = crop_txt($txt, BOX_WIDTH);
			d.mv(y+$row, x + (BOX_WIDTH as i32- txt.len() as i32)/2);
			d.addstr(&txt);
			
			d.mv(y+$row, x + BOX_WIDTH as i32-1);
			d.addch(disp_chars.vline_char);
		};};
		
		center_txt!(1, &format!("{} {}", self.name.first, self.name.last));
		center_txt!(2, &format!("{}: {} {}", l.Age, ((turn - self.born_turn) as f32/TURNS_PER_YEAR as f32).round(), l.yrs));
		
		// bottom line
		{
			d.mv(y+3, x);
			d.addch(disp_chars.llcorner_char);
			for _ in 0..(BOX_WIDTH-2) {d.addch(disp_chars.hline_char);}
			d.addch(disp_chars.lrcorner_char);
		}
	}
}

impl NoblePair {
	fn print(&self, show_parent_connection: bool, coord: ScreenCoord,
			disp_chars: &DispChars, l: &Localization, d: &mut DispState, turn: usize) {
		self.noble.print(show_parent_connection, coord, disp_chars, l, d, turn);
		if let Some(marriage) = &self.marriage {
			let y = coord.y as i32;
			let x = coord.x as i32;
			
			// line between the two:        | Partner 1 |-------| Partner 2 |
			let box_middle_row = y + BOX_HEIGHT as i32/2;
			d.mv(box_middle_row, x + BOX_WIDTH as i32);
			for col in 0..SPACE_BETWEEN_PAIRS_X {
				d.addch(if col != (SPACE_BETWEEN_PAIRS_X/2) {
					disp_chars.hline_char
				}else{
					disp_chars.urcorner_char
				});
			}
			
			let coord = ScreenCoord {
				y: y as isize,
				x: x as isize + (BOX_WIDTH + SPACE_BETWEEN_PAIRS_X) as isize
			};
			marriage.partner.print(show_parent_connection, coord, disp_chars, l, d, turn);
			
			// vertical line between the two
			if marriage.children.len() != 0 {
				for row in (box_middle_row+1)..(coord.y as i32 + SPACE_BETWEEN_GENERATIONS_Y as i32 + BOX_HEIGHT as i32) {
					d.mv(row, x + BOX_WIDTH as i32 + 1);
					d.addch(disp_chars.vline_char);
				}
			}
		}
	}
	
	// width of an individual noble or both partners in a marriage
	fn disp_width(&self) -> usize {
		if !self.marriage.is_none() {
			2*BOX_WIDTH + SPACE_BETWEEN_PAIRS_X
		}else{BOX_WIDTH}
	}	
}

impl House<'_,'_,'_,'_> {
	pub fn print_pedigree(&self, personalization: &Personalization, buttons: &mut Buttons, screen_sz: ScreenSz,
			disp_chars: &DispChars,	l: &Localization, d: &mut DispState, turn: usize) {
		d.clear();
		
		let title_c = COLOR_PAIR(CGREEN);
		
		d.mv(0,0); buttons.Esc_to_close.print(None, l, d);
		
		macro_rules! center_txt{($txt: expr, $w: expr) => {
			let g = ($w as usize - $txt.len())/2;
			let mut sp = String::new();
			for _ in 0..g {sp.push(' ');}
			d.attron(title_c);
			d.addstr(&format!("{}{}", sp, $txt));
			d.attroff(title_c);
		};};
		
		let w = screen_sz.w as i32;
		center_txt!(l.House_of.replace("[]", &personalization.nm), w);
		
		if let Some(head_pair) = self.noble_pairs.first() {
			let head_branch_w = self.branch_disp_width(head_pair) as i32;
			debug_assertq!(w > head_branch_w);
			self.print_branch(1, (w - head_branch_w)/2, head_pair, w, disp_chars, l, d, turn);
		}
	}
	
	fn print_branch(&self, start_row: i32, start_col: i32, head_pair: &NoblePair, screen_sz_w: i32,
			disp_chars: &DispChars, l: &Localization, d: &mut DispState, turn: usize) {
		let head_branch_w = self.branch_disp_width(head_pair) as isize;
		let head_pair_w = head_pair.disp_width() as isize;
		
		{ // print head pair
			// head pair is wider than the children
			let head_start_col = if head_pair_w == head_branch_w {
				start_col
			// center head pair on top of the children
			}else{
				debug_assertq!(head_pair_w < head_branch_w);
				start_col + (head_branch_w - head_pair_w) as i32 /2
			};
			
			head_pair.print(false, ScreenCoord {y: start_row as isize, x: head_start_col as isize}, disp_chars, l, d, turn);
		}
		
		// show children
		if let Some(marriage) = &head_pair.marriage {
			let child_branch_ws = marriage.children.iter().map(|child| 
				self.branch_disp_width(&self.noble_pairs[*child])
			).collect::<Vec<usize>>();
			
			let mut start_row = start_row + 1 + BOX_HEIGHT as i32;
			let mut start_col = 
				// center children under the head pair
				if head_pair_w == head_branch_w {
					start_col + (head_pair_w - head_branch_w) as i32/2
				// children are wider than the head pair
				} else {start_col};
			
			(|| { // horizontal line connection between parents and children
				if child_branch_ws.len() == 0 {return;} // no children
				
				d.mv(start_row, start_col + BOX_WIDTH as i32/2);
				
				// only one child and no partner, only draw: |
				if child_branch_ws.len() == 1 && self.noble_pairs[marriage.children[0]].marriage.is_none() {
					d.addch(disp_chars.vline_char);
					return;
				}
				
				// far left child
				if let Some(child_w) = child_branch_ws.first() {
					d.addch(disp_chars.ulcorner_char);
					for _ in 0..(child_w - (BOX_WIDTH/2)) {d.addch(disp_chars.hline_char);}
				}
				
				// middle children
				for (child_w_ind, child_w) in child_branch_ws.iter().enumerate().skip(1) {
					// skip last child
					if child_w_ind == (child_branch_ws.len()-1) {break;}
					
					for _ in 0..(*child_w + SPACE_BETWEEN_PAIRS_X) {d.addch(disp_chars.hline_char);}
				}
				
				// last child
				if child_branch_ws.len() > 1 {
					let child_w = child_branch_ws.last().unwrap();
					
					for _ in 0..(2+child_w - (BOX_WIDTH/2)) {d.addch(disp_chars.hline_char);}
					d.addch(disp_chars.urcorner_char);
				}
			})();
			
			start_row += 1;
			
			for (child_ind, child_branch_w) in marriage.children.iter().zip(child_branch_ws.iter()) {
				let child = &self.noble_pairs[*child_ind];
				self.print_branch(start_row, start_col, child, screen_sz_w, disp_chars, l, d, turn);
				start_col += (child_branch_w + SPACE_BETWEEN_PAIRS_X) as i32;
			}
		}
	}
	
	// max width of any generation that is a successor of the head_pair
	fn branch_disp_width(&self, head_pair: &NoblePair) -> usize {
		let head_w = head_pair.disp_width();
		
		if let Some(marriage) = &head_pair.marriage {
			let mut children_w = SPACE_BETWEEN_PAIRS_X * (marriage.children.len() - 1);
			for child_ind in marriage.children.iter() {
				children_w += self.branch_disp_width(&self.noble_pairs[*child_ind]);
			}
			max(children_w, head_w)
		}else{head_w}
	}
}

