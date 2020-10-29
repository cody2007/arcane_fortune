use std::cmp::max;
use crate::renderer::COLOR_PAIR;
use crate::disp::*;
use crate::player::Personalization;
use super::*;

const BOX_WIDTH: usize = 20;
const BOX_HEIGHT: usize = 4;
const SPACE_BETWEEN_PAIRS_X: usize = 3;
const SPACE_BETWEEN_GENERATIONS_Y: usize = 2;

impl Noble {
	fn print(&self, show_parent_connection: bool, coord: ScreenCoord, dstate: &mut DispState, turn: usize) {
		let y = coord.y as i32;
		let x = coord.x as i32;
		
		// top line
		{
			dstate.mv(y, x);
			dstate.addch(dstate.chars.ulcorner_char);
			for col in 0..(BOX_WIDTH-2) {
				dstate.addch(if !show_parent_connection || col != (BOX_WIDTH-2)/2 {
					dstate.chars.hline_char
				}else{
					dstate.chars.llcorner_char
				});
			}
			dstate.addch(dstate.chars.urcorner_char);
		}
		
		macro_rules! center_txt{($row: expr, $txt: expr) => {
			dstate.mv(y+$row, x);
			dstate.addch(dstate.chars.vline_char);
			
			let txt = crop_txt($txt, BOX_WIDTH);
			dstate.mv(y+$row, x + (BOX_WIDTH as i32- txt.len() as i32)/2);
			dstate.addstr(&txt);
			
			dstate.mv(y+$row, x + BOX_WIDTH as i32-1);
			dstate.addch(dstate.chars.vline_char);
		};};
		
		center_txt!(1, &format!("{} {}", self.name.first, self.name.last));
		center_txt!(2, &format!("{}: {} {}", dstate.local.Age, ((turn - self.born_turn) as f32/TURNS_PER_YEAR as f32).round(), dstate.local.yrs));
		
		// bottom line
		{
			dstate.mv(y+3, x);
			dstate.addch(dstate.chars.llcorner_char);
			for _ in 0..(BOX_WIDTH-2) {dstate.addch(dstate.chars.hline_char);}
			dstate.addch(dstate.chars.lrcorner_char);
		}
	}
}

impl NoblePair {
	fn print(&self, show_parent_connection: bool, coord: ScreenCoord, dstate: &mut DispState, turn: usize) {
		self.noble.print(show_parent_connection, coord, dstate, turn);
		if let Some(marriage) = &self.marriage {
			let y = coord.y as i32;
			let x = coord.x as i32;
			let d = &mut dstate.renderer;
			
			// line between the two:        | Partner 1 |-------| Partner 2 |
			let box_middle_row = y + BOX_HEIGHT as i32/2;
			dstate.mv(box_middle_row, x + BOX_WIDTH as i32);
			for col in 0..SPACE_BETWEEN_PAIRS_X {
				dstate.addch(if col != (SPACE_BETWEEN_PAIRS_X/2) {
					dstate.chars.hline_char
				}else{
					dstate.chars.urcorner_char
				});
			}
			
			let coord = ScreenCoord {
				y: y as isize,
				x: x as isize + (BOX_WIDTH + SPACE_BETWEEN_PAIRS_X) as isize
			};
			marriage.partner.print(show_parent_connection, coord, dstate, turn);
			
			// vertical line between the two
			if marriage.children.len() != 0 {
				for row in (box_middle_row+1)..(coord.y as i32 + SPACE_BETWEEN_GENERATIONS_Y as i32 + BOX_HEIGHT as i32) {
					dstate.mv(row, x + BOX_WIDTH as i32 + 1);
					dstate.addch(dstate.chars.vline_char);
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
	pub fn print_pedigree(&self, personalization: &Personalization, turn: usize, dstate: &mut DispState) {
		dstate.clear();
		
		let title_c = COLOR_PAIR(CGREEN);
		
		dstate.mv(0,0); dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
		
		macro_rules! center_txt{($txt: expr, $w: expr) => {
			let g = ($w as usize - $txt.len())/2;
			let mut sp = String::new();
			for _ in 0..g {sp.push(' ');}
			dstate.attron(title_c);
			dstate.addstr(&format!("{}{}", sp, $txt));
			dstate.attroff(title_c);
		};};
		
		let w = dstate.iface_settings.screen_sz.w as i32;
		center_txt!(dstate.local.house_nm.replace("[]", &personalization.nm), w);
		
		if let Some(head_pair) = self.noble_pairs.first() {
			let head_branch_w = self.branch_disp_width(head_pair) as i32;
			debug_assertq!(w > head_branch_w);
			self.print_branch(1, (w - head_branch_w)/2, head_pair, turn, dstate);
		}
	}
	
	fn print_branch(&self, start_row: i32, start_col: i32, head_pair: &NoblePair, turn: usize, dstate: &mut DispState) {
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
			
			head_pair.print(false, ScreenCoord {y: start_row as isize, x: head_start_col as isize}, dstate, turn);
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
				
				dstate.mv(start_row, start_col + BOX_WIDTH as i32/2);
				
				// only one child and no partner, only draw: |
				if child_branch_ws.len() == 1 && self.noble_pairs[marriage.children[0]].marriage.is_none() {
					dstate.addch(dstate.chars.vline_char);
					return;
				}
				
				// far left child
				if let Some(child_w) = child_branch_ws.first() {
					dstate.addch(dstate.chars.ulcorner_char);
					for _ in 0..(child_w - (BOX_WIDTH/2)) {dstate.addch(dstate.chars.hline_char);}
				}
				
				// middle children
				for (child_w_ind, child_w) in child_branch_ws.iter().enumerate().skip(1) {
					// skip last child
					if child_w_ind == (child_branch_ws.len()-1) {break;}
					
					for _ in 0..(*child_w + SPACE_BETWEEN_PAIRS_X) {dstate.addch(dstate.chars.hline_char);}
				}
				
				// last child
				if child_branch_ws.len() > 1 {
					let child_w = child_branch_ws.last().unwrap();
					
					for _ in 0..(2+child_w - (BOX_WIDTH/2)) {dstate.addch(dstate.chars.hline_char);}
					dstate.addch(dstate.chars.urcorner_char);
				}
			})();
			
			start_row += 1;
			
			for (child_ind, child_branch_w) in marriage.children.iter().zip(child_branch_ws.iter()) {
				let child = &self.noble_pairs[*child_ind];
				self.print_branch(start_row, start_col, child, turn, dstate);
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

