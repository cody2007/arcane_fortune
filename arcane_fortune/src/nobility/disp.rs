use std::cmp::max;
use crate::disp_lib::{DispState, COLOR_PAIR};
use crate::disp::{Buttons, ScreenSz, CGREEN, DispChars, crop_txt};
use crate::localization::Localization;
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
			for _ in 0..SPACE_BETWEEN_PAIRS_X {
				d.addch(disp_chars.hline_char);
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

impl House {
	pub fn print_pedigree(&self, buttons: &mut Buttons, screen_sz: ScreenSz,
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
		center_txt!(l.House_of.replace("[]", &self.name), w);
		
		if let Some(head_pair) = self.noble_pairs.first() {
			let coord = ScreenCoord {
				y: 1,
				x: (screen_sz.w - head_pair.disp_width()) as isize / 2
			};
			
			head_pair.print(false, coord, disp_chars, l, d, turn);
			let head_branch_w = self.branch_disp_width(0) as isize;
			let child_start_row = (screen_sz.w as i32 - head_branch_w as i32)/2;
			
			{ // horizontal line between parents & direct children
				d.mv(BOX_HEIGHT as i32 + 2, child_start_row);
				d.addch(disp_chars.ulcorner_char);
				for col in 0..(head_branch_w-2) {
					d.addch(if col != (head_branch_w/2)-1 {
						disp_chars.hline_char
					}else{
						disp_chars.llcorner_char
					});
				}
				d.addch(disp_chars.urcorner_char);
			}
			
			// show children
			if let Some(marriage) = &head_pair.marriage {
				let mut coord = ScreenCoord {
					y: (1 + SPACE_BETWEEN_GENERATIONS_Y + BOX_HEIGHT) as isize,
					x: (screen_sz.w as isize - head_branch_w as isize) / 2
				};
				
				for child in marriage.children.iter() {
					self.noble_pairs[*child].print(true, coord, disp_chars, l, d, turn);
					coord.x += (self.branch_disp_width(*child) + SPACE_BETWEEN_PAIRS_X) as isize;
				}
			}
		}
	}
	
	// max width of any generation that is a successor of the pair_ind noble_pair
	fn branch_disp_width(&self, pair_ind: usize) -> usize {
		let noble_pair = &self.noble_pairs[pair_ind];
		if let Some(marriage) = &noble_pair.marriage {
			let mut children_w = SPACE_BETWEEN_PAIRS_X * (marriage.children.len() - 1);
			for child_ind in marriage.children.iter() {
				children_w += self.branch_disp_width(*child_ind);
			}
			max(children_w, noble_pair.disp_width())
		}else{noble_pair.disp_width()}
	}
}

