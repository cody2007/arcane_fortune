// Generic tree printing functions

use crate::disp_lib::*;
use crate::saving::defaults::SmSvType;
use super::{ScreenSz, DispChars, CRED, CLOGO, Buttons};
use crate::player::Stats;
use crate::localization::Localization;

pub trait TreeTemplate {
	fn reqs(&self) -> Option<Vec<SmSvType>>;
	fn nm(&self, l: &Localization) -> String;
	fn line_color(&self, row: usize, pstats: &Stats) -> CInt;
	fn requirements_txt(&self, pstats: &Stats, l: &Localization) -> String;
}

#[derive(PartialEq, Clone, Copy)]
pub enum TreeSelMv {Up, Down, Left, Right, None}
// ^ these denote the arrow keys pressed to change current template selection

#[derive(PartialEq, Debug)]
pub struct TreeOffsets {pub row: i32, pub col: i32}

// printable_t aka Vec<PrintableTree>
// contains the children and parents *shown* for each template entry
// may not include all children for templates that have multiple reqs
// because each template is only shown once on the screen
pub struct PrintableTree {
	pub id: usize, // index into templates (Tech or Spirituality templates)
	pub children: Vec<usize>, // index into printable_t
	pub parent: Option<usize>, // index into printable_t
	pub tree_offs: TreeOffsets // location to show template on screen
}

pub const BOX_GAP: i32 = 3;

///////////////////////////////
// the tree will be printed starting at row_start and will end before row_stop and col_stop
// row, and col are the current position to print the child nodes (what isn't in the row_stop:row_stop
// bounds is not printed (similarly for 0:col_stop)
fn print_tree_recur<T: TreeTemplate>(mut row: i32, col: i32, root_children: &Vec<SmSvType>,
		templates: &Vec<T>, cur_depth: u32, shown: &mut Vec<bool>, 
		parent_tech: Option<SmSvType>, disp_chars: &DispChars, pstats: &Stats,
		entry_sel: Option<SmSvType>, row_start: i32, row_stop: i32, col_stop: i32, entry_sz_print: ScreenSz,
		sel_loc: &mut Option<(i32, i32, SmSvType)>, l: &Localization, d: &mut DispState) {
	
	let box_cols_per_entry = entry_sz_print.w as i32 - BOX_GAP*2;
	
	macro_rules! addch_chk{($row:expr, $col:expr, $c:expr) => {
		if $col < col_stop && $row < row_stop && $row >= row_start && $col >= 0 {
			d.mv($row, $col);
			d.addch($c);
		}
	};}
	
	for t in root_children.iter() {
		// don't show more than once
		if shown[*t as usize] {continue;}
		
		// only show at deepest extent
		if max_req_depth(*t, templates) != cur_depth {continue;}
		
		shown[*t as usize] = true;
		
		let tp = &templates[*t as usize];
		let mut strs_show = Vec::new();
		
		// name and research req
		strs_show.push(tp.nm(l));
		strs_show.push(tp.requirements_txt(pstats, l));
		
		// requires multiple techs
		if let Some(techs_req) = tp.reqs() {
			if techs_req.len() != 1 {
				strs_show.push(l.Also_req.clone());
				let str_added = strs_show.len() - 1;
				for (i, tech_req) in techs_req.iter().enumerate() {
					if *tech_req != parent_tech.unwrap() {
						// last entry or the next one is the last but won't be shown 
						// because it's the current parent
						if (i+1) == techs_req.len() || techs_req[i+1] == parent_tech.unwrap() {
							strs_show[str_added].push_str(&format!("{}", templates[*tech_req as usize].nm(l)));
						}else{
							strs_show[str_added].push_str(&format!("{}, ", templates[*tech_req as usize].nm(l)));
						}
					}
				}
			}
		}
		
		let n_lines = strs_show.len() as i32;
		
		if entry_sel == Some(*t) {
			d.attron(COLOR_PAIR(CRED));
			*sel_loc = Some((row + BOX_GAP - 2, col + box_cols_per_entry + 2, *t));
		}
		
		// corners
		addch_chk!(row + BOX_GAP - 1, col - 1, disp_chars.ulcorner_char);
		addch_chk!(row + BOX_GAP - 1, col + box_cols_per_entry, disp_chars.urcorner_char);
		addch_chk!(row + BOX_GAP + n_lines, col - 1, disp_chars.llcorner_char);
		addch_chk!(row + BOX_GAP + n_lines, col + box_cols_per_entry, disp_chars.lrcorner_char);
		
		// left/right
		for row_offset in 0..n_lines {
			addch_chk!(row + BOX_GAP + row_offset, col - 1, disp_chars.vline_char);
			addch_chk!(row + BOX_GAP + row_offset, col + box_cols_per_entry, disp_chars.vline_char);
		}
		
		// top/bottom
		for col_offset in 0..box_cols_per_entry {
			addch_chk!(row + BOX_GAP - 1, col + col_offset, disp_chars.hline_char);
			addch_chk!(row + BOX_GAP + n_lines, col + col_offset, disp_chars.hline_char);
		}
		
		if entry_sel == Some(*t) {d.attroff(COLOR_PAIR(CRED));}
		
		// print centered (loop over rows)
		for (i, txt) in strs_show.iter().enumerate() {
			let line_len = txt.chars().count() as i32;
			let gap_len = (box_cols_per_entry - line_len)/2; // column
			
			let row_print = row + i as i32 + BOX_GAP;
			let col_print = col + gap_len;
			
			if row_print >= row_stop || col_print >= col_stop || row_print < row_start {break;}
			
			let color = COLOR_PAIR(tp.line_color(i, pstats));
			
			// print
			if col_print >= 0 {
				d.mv(row_print, col_print);
				d.attron(color);
				d.addnstr(txt, col_stop - col_print);
				d.attroff(color);
			}else if (col_print + line_len) > 0{
				d.mv(row_print, 0);
				d.attron(color);
				d.addnstr(&txt[((-col_print) as usize)..], col_stop);
				d.attroff(color);
			}
		}
		
		// show children
		let sub_children = ret_direct_children(Some(*t), templates);
		let n_children = sub_children.len() as i32;
		let n_t_c = n_terminal_children_not_shown(Some(*t), templates, shown) as i32;
		
		print_tree_recur(row, col + entry_sz_print.w as i32, &sub_children, templates, cur_depth+1, 
				shown, Some(*t), disp_chars, pstats, entry_sel, row_start, row_stop, col_stop,
				entry_sz_print, sel_loc, l, d);
		
		// increment row counter, show connections to children
		if n_t_c == 0 {
			row += entry_sz_print.h as i32;
		}else{
			///////////// connections to children
			let row_off = row + BOX_GAP + 1;
			let col_off = col + box_cols_per_entry + 1;
			
			//// -
			addch_chk!(row_off, col_off, disp_chars.hline_char);
			addch_chk!(row_off, col_off + 1, disp_chars.hline_char);
			
			//// |
			for row_offset in 0..(entry_sz_print.h as i32 * (n_children - 1)) {
				addch_chk!(row_off + 1 + row_offset, col_off + 1, disp_chars.vline_char);
			}
			
			//// - to each child
			for child in 0..n_children {
				let mut col_off_i = col_off + 2;
							
				for _ in 0..(entry_sz_print.w as i32 - BOX_GAP - box_cols_per_entry - 2) {
					addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i, disp_chars.hline_char);
					col_off_i += 1;
				}
				addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i, '>' as chtype);
				
				// (hack) print '...' if nothing has been shown for the child (i.e., it is shown further down the tree in another location)
				if (' ' as chtype) == (d.inch() & A_CHARTEXT()) {
					addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i + 1, ' ' as chtype);
					addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i + 2, ' ' as chtype);
					addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i + 3, '.' as chtype);
					addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i + 4, '.' as chtype);
					addch_chk!(row_off + child*entry_sz_print.h as i32, col_off_i + 5, '.' as chtype);
				}
			}
			
			row += n_t_c*entry_sz_print.h as i32;
		}
	}
}

////////// creates printable_t 
// row, col are the current locations on screen that the template will be shown
fn create_printable_tree<T: TreeTemplate>(mut row: i32, col: i32, parent_template: Option<usize>,
		root_children: &Vec<SmSvType>,
		templates: &Vec<T>, cur_depth: u32, shown: &mut Vec<bool>, 
		row_max: &mut i32, col_max: &mut i32, printable_t: &mut Vec<PrintableTree>,
		entry_sz_print: ScreenSz){
	
	for t in root_children.iter() { // index into templates		
		let t_u = *t as usize;
		
		// don't show more than once
		if shown[t_u] {continue;}
		
		// only show at deepest extent
		if cur_depth != max_req_depth(*t, templates) {continue;}
		
		shown[t_u] = true;
		
		// children of child
		let sub_children = ret_direct_children(Some(*t), templates);
		let n_t_c = n_terminal_children_not_shown(Some(*t), templates, shown) as i32; 
		
		// entry of current child
		let child_template_entry = PrintableTree {
			id: *t as usize,
			children: Vec::with_capacity(sub_children.len()),
			parent: if let Some(parent_template) = parent_template { // < index into template_template
					let parent_ind = find_template(parent_template, printable_t); // index into printable_t
					
					// add child to parent
					let child_ind = printable_t.len();
					printable_t[parent_ind].children.push(child_ind);
					
					// store in child entry
					Some(parent_ind)
				} else {None},
			tree_offs: TreeOffsets {row, col}
		};
		
		printable_t.push(child_template_entry);
		
		create_printable_tree(row, col + entry_sz_print.w as i32, Some(t_u), &sub_children,
				templates, cur_depth+1, shown, row_max, col_max, printable_t, entry_sz_print);
		
		if n_t_c == 0 {
			row += entry_sz_print.h as i32;
		}else{
			row += n_t_c*entry_sz_print.h as i32;
		}
	}
	
	if row > *row_max {*row_max = row;}
	if col > *col_max {*col_max = col;}
}

// scroll to the currently selected template
fn scroll_to_sel(tree_offsets: &mut TreeOffsets, row_tree_start: i32,
		row_tree_stop: i32, col_tree_stop: i32, template_sel: u32, 
		printable_t: &Vec<PrintableTree>, entry_sz_print: ScreenSz) {
	let new_t = &printable_t[find_template(template_sel as usize, &printable_t)];
	
	/////////////// left/right scrolling
	let show_col_top = tree_offsets.col + new_t.tree_offs.col; // loc template will be shown
	let show_col_bottom = show_col_top + entry_sz_print.w as i32;
	
	if show_col_bottom > col_tree_stop {
		tree_offsets.col -= show_col_bottom - col_tree_stop;
	}
	
	if show_col_top < 1 {
		// we want: show_row_top = row_tree_start
		// tree_offsets.row + new_t.tree_offs.row = row_tree_start
		tree_offsets.col = 1 - new_t.tree_offs.col;
	}
	
	////////////////////// up/down scrolling
	let show_row_top = tree_offsets.row + new_t.tree_offs.row; // loc template will be shown
	let show_row_bottom = show_row_top + 2 + entry_sz_print.h as i32;
	
	if show_row_bottom > row_tree_stop {
		tree_offsets.row -= show_row_bottom - row_tree_stop;
	}
	
	if show_row_top < row_tree_start {
		// we want: show_row_top = row_tree_start
		// tree_offsets.row + new_t.tree_offs.row = row_tree_start
		tree_offsets.row = row_tree_start - new_t.tree_offs.row;
	}
}

// find entry in Vec<PrintableTree> that has id, the index into templates[]
fn find_template(id: usize, printable_t: &Vec<PrintableTree>) -> usize {
	for (i, t) in printable_t.iter().enumerate() {
		if t.id == id {return i;}
	}
	panicq!("could not find tree entry id {}", id);
}

// does chk_template require `req`
//	`req`: an index into either Vec<TechTemplate> or Vec<SpiritualityTemplate>
fn chk_req<T: TreeTemplate>(chk_template: &T, req: Option<SmSvType>) -> bool {
	// chk_template has templatetemplate reqs and they contain template_req
	if let Some(chk_template_reqs) = chk_template.reqs() {
		if let Some(req) = &req {
			return chk_template_reqs.contains(req);
		}
	// chk_template has no template reqs and we are searching for those that have no reqs
	}else{
		return req.is_none();
	}
	
	// either chk_template has reqs and they aren't matched OR it has no reqs and we are searching for a template_req
	false
}

fn n_children<T: TreeTemplate>(req: Option<SmSvType>, templates: &Vec<T>) -> u32 {
	let mut n_children_found = 0;
	for (i, chk_template) in templates.iter().enumerate() {	
		if chk_req(chk_template, req) {
			n_children_found += n_children(Some(i as SmSvType), templates) + 1;
		
		}
	}
	n_children_found
}

// all template inds requiring `req`
fn ret_direct_children<T: TreeTemplate>(req: Option<SmSvType>, templates: &Vec<T>) -> Vec<SmSvType> {
	let mut children = Vec::new();
	
	for (i, chk_template) in templates.iter().enumerate() {
		if chk_req(chk_template, req) {
			children.push(i as SmSvType);
		}
	}
	children
}

fn ret_children<T: TreeTemplate>(req: Option<SmSvType>, templates: &Vec<T>) -> Vec<SmSvType> {
	let mut children = Vec::new();
	
	for (i, chk_template) in templates.iter().enumerate() {
		if chk_req(chk_template, req){
			children.push(i as SmSvType);
			children.append(&mut ret_children(Some(i as SmSvType), templates));
		}
	}
	children
}

fn n_terminal_children_not_shown<T: TreeTemplate>(req: Option<SmSvType>, templates: &Vec<T>, 
		shown: &Vec<bool>) -> u32 {
	let mut n_term = 0;
	
	let children = ret_children(req, templates);
	
	for c in children {
		if shown[c as usize] {continue;}
		if n_children(Some(c), templates) == 0 {
			n_term += 1;
		}
	}
	n_term
}

fn max_req_depth<T: TreeTemplate>(chk_template: SmSvType, templates: &Vec<T>) -> u32 {
	let mut depths = Vec::with_capacity(templates.len());
	if let Some(reqs) = &templates[chk_template as usize].reqs() {
		for req in reqs {
			depths.push(1 + max_req_depth(*req, templates));
		}
	}
	
	return if let Some(max_val) = depths.iter().max() {
		*max_val
	}else{
		1
	};
}

pub struct TreeDisplayProperties {
	pub instructions_start_row: i32,
	pub down_scroll: bool,
	pub right_scroll: bool,
	pub sel_loc: Option<(i32, i32, SmSvType)> // (row, col, template_ind)
}

pub fn print_tree<T: TreeTemplate>(templates: &Vec<T>, disp_chars: &DispChars, pstats: &Stats, 
		tech_sel: &mut Option<SmSvType>, tech_sel_mv: &mut TreeSelMv,
		tree_offsets: &mut Option<TreeOffsets>, screen_sz: ScreenSz,
		entry_sz_print: ScreenSz, l: &Localization, buttons: &mut Buttons, d: &mut DispState) -> TreeDisplayProperties {
	let h = screen_sz.h as i32;
	let w = screen_sz.w as i32;
	
	let instructions_start_row = h - 6;
	let cur_depth = 1;
	
	let parent_tech = None;
	let root_children = ret_direct_children(parent_tech, templates);
	
	/////////////////////////// create printable tech tree
	
	// sz of tree computed in printable_tree_dims:
	let mut tree_rows = 0;
	let mut tree_cols = 0;
	
	let mut shown = vec!{false; templates.len()}; // show each tech only once
	let mut printable_tt = Vec::new();
	
	//endwin();
	create_printable_tree(0,0, None, &root_children, templates, cur_depth, &mut shown,
			&mut tree_rows, &mut tree_cols, &mut printable_tt, entry_sz_print);
	///// debug
	/*{
		endwin();
		for ptt in printable_tt.iter() {
			println!("id: {} {}", templates[ptt.id].nm, ptt.id);
			if let Some(parent) = ptt.parent {
				println!("parent: {}", templates[printable_tt[parent].id].nm);
			}
			println!("children:");
			for child in ptt.children.iter().cloned() {
				println!("{}", templates[printable_tt[child].id].nm);
			}
			println!("offsets: {} {}", ptt.tree_offs.row, ptt.tree_offs.col);
			println!("");
		}
		println!("{} {}", templates.len(), printable_tt.len());
	}*/
	
	debug_assertq!(shown == vec!{true; templates.len()}, "Some tech was not shown");
	tree_cols -= BOX_GAP;
	
	////////// location on screen to show tree and if it needs scroll bars
	
	let row_tree_start = 1; // title takes up the first line, so skip it
	
	// max h, w to show tree (based on the instructions/title around the tree
	let row_tree_stop = instructions_start_row - 1; // minus the scroll bar
	let col_tree_stop = w - 2;
	
	// location to show tree (initialize centered)
	let col_tree_centered = (col_tree_stop - tree_cols) / 2;
	let row_tree_centered = (row_tree_stop - tree_rows) / 2;
	
	if tree_offsets.is_none() {
		*tree_offsets = Some(TreeOffsets {row: row_tree_centered, col: col_tree_centered});
		if let Some(tech_sel) = tech_sel {
			scroll_to_sel(tree_offsets.as_mut().unwrap(), row_tree_start, row_tree_stop, col_tree_stop, *tech_sel, &printable_tt, entry_sz_print);
		}
	}
	
	// show scroll bars?
	let down_scroll = tree_rows > (row_tree_stop - row_tree_start);
	let right_scroll = tree_cols > col_tree_stop;
	
	// shorten so we don't have to unwrap every time
	let tree_offsets = if let Some(ref mut tree_offsets) = tree_offsets {
		tree_offsets
	}else{panicq!("could not unwrap tree offsets")};
	
	/////////////////////////// chk left/right scroll loc
	
	// check that tree location does not scroll out of bounds
	if right_scroll {
		if (tree_offsets.col + tree_cols) < col_tree_stop {
			tree_offsets.col = col_tree_stop - tree_cols;
		}
		
		if tree_offsets.col > 1 {
			tree_offsets.col = 1;
		}
	
	// should always be centered when no scrolling possible
	}else{tree_offsets.col = col_tree_centered;}
	
	///////////////////////// chk up/down scroll loc
	
	// check that tree location does not scroll out of bounds
	if down_scroll {
		if (tree_offsets.row + tree_rows) < (row_tree_stop - row_tree_start) {
			tree_offsets.row = row_tree_stop - row_tree_start - tree_rows;
		}
		
		if tree_offsets.row > 0 {
			tree_offsets.row = 0;
		}
		
	// should always be centered when no scrolling possible
	}else{tree_offsets.row = row_tree_centered;}
	
		
	/////////////// update tech_sel	based on arrow key pressed
	if *tech_sel_mv != TreeSelMv::None && !tech_sel.is_none() {
		let sel_ind_printable_tt = find_template(tech_sel.unwrap() as usize, &mut printable_tt);
		let printable_sel = &printable_tt[sel_ind_printable_tt];
		debug_assertq!(Some(printable_sel.id as u32) == *tech_sel);
		
		// move up to other sibling
		if *tech_sel_mv == TreeSelMv::Up || *tech_sel_mv == TreeSelMv::Down {
			// currently selecting some intermediate tech
			if let Some(parent_ind) = printable_sel.parent {
				let parent_tech = &printable_tt[parent_ind as usize];
				let child_ind = parent_tech.children.iter().position(|&r| r == sel_ind_printable_tt).unwrap();
				
				if *tech_sel_mv == TreeSelMv::Up && child_ind != 0 {
					*tech_sel = Some(printable_tt[parent_tech.children[child_ind - 1]].id as u32);
					scroll_to_sel(tree_offsets, row_tree_start, row_tree_stop, col_tree_stop, tech_sel.unwrap(), &printable_tt, entry_sz_print);
				
				}else if *tech_sel_mv == TreeSelMv::Down && child_ind < (parent_tech.children.len() - 1) {
					*tech_sel = Some(printable_tt[parent_tech.children[child_ind+1]].id as u32);
					scroll_to_sel(tree_offsets, row_tree_start, row_tree_stop, col_tree_stop, tech_sel.unwrap(), &printable_tt, entry_sz_print);

				}
			// currently selecting tech with no tech requirements (at root)
			}else{
				let child_ind = root_children.iter().position(|&r| Some(r) == *tech_sel).unwrap();
				
				if *tech_sel_mv == TreeSelMv::Up && child_ind != 0 {
					*tech_sel = Some(root_children[child_ind - 1]);
					scroll_to_sel(tree_offsets, row_tree_start, row_tree_stop, col_tree_stop, tech_sel.unwrap(), &printable_tt, entry_sz_print);
					
				}else if *tech_sel_mv == TreeSelMv::Down && 
						child_ind < (root_children.len() - 1) {
					
					*tech_sel = Some(root_children[child_ind + 1]);
					scroll_to_sel(tree_offsets, row_tree_start, row_tree_stop, col_tree_stop, tech_sel.unwrap(), &printable_tt, entry_sz_print);
					
				}
			}
		
		// move to child
		}else if *tech_sel_mv == TreeSelMv::Right && printable_sel.children.len() != 0 {
			*tech_sel = Some(printable_tt[printable_sel.children[0]].id as u32);
			scroll_to_sel(tree_offsets, row_tree_start, row_tree_stop, col_tree_stop, tech_sel.unwrap(), &printable_tt, entry_sz_print);
		
		// move to parent (research requirement)
		}else if *tech_sel_mv == TreeSelMv::Left {	
			if let Some(parent_ind) = printable_sel.parent {
				*tech_sel = Some(printable_tt[parent_ind].id as u32);
				scroll_to_sel(tree_offsets, row_tree_start, row_tree_stop, col_tree_stop, tech_sel.unwrap(), &printable_tt, entry_sz_print);
			}
		}
	}
	
	d.clear();
	
	///////////////// show tree
	let mut sel_loc = None; // screen location of current tech selection (row, col, tech_ind)
	let mut shown = vec!{false; templates.len()}; // show each tech only once
	print_tree_recur(tree_offsets.row, tree_offsets.col, &root_children, 
			templates, cur_depth, &mut shown, parent_tech,
			disp_chars, pstats, *tech_sel, row_tree_start, row_tree_stop, col_tree_stop,
			entry_sz_print, &mut sel_loc, l, d);
	
	// scroll bars (left-right)
	if right_scroll {		
		let scroll_track_w = w-3;
		let frac_covered = (col_tree_stop as f32) / (tree_cols as f32);
		let scroll_bar_w = ((scroll_track_w as f32) * frac_covered).round() as i32;
		debug_assertq!(frac_covered <= 1.);
		
		let frac_at = ((-(tree_offsets.col - 1)) as f32) / (tree_cols as f32);
		let scroll_bar_start = ((scroll_track_w as f32) * frac_at).round() as i32;
		
		// print
		d.mv(h-7, 0);
		d.attron(COLOR_PAIR(CLOGO));
		d.addch('[' as chtype);
		d.mv(h-7, 1 + scroll_bar_start);
		d.addch('<' as chtype);
		for _ in 0..scroll_bar_w-2 {
			d.addch('=' as chtype);
		}
		d.addch('>' as chtype);
		d.mv(h-7, w-2);
		d.addch(']' as chtype);
		d.attroff(COLOR_PAIR(CLOGO));
	}
	
	// scroll bars (up-down)
	if down_scroll {			
		let scroll_track_h = h-7;
		let frac_covered = ((row_tree_stop - row_tree_start) as f32) / (tree_rows as f32);
		let scroll_bar_h = ((scroll_track_h as f32) * frac_covered).round() as i32;
		debug_assertq!(frac_covered <= 1.);
		
		let frac_at = (-tree_offsets.row as f32) / (tree_rows as f32);
		let scroll_bar_start = ((scroll_track_h as f32) * frac_at).round() as i32;
		
		d.mv(0, w-1);
		d.attron(COLOR_PAIR(CLOGO));
		d.addch(disp_chars.hline_char);
		for row in 0..scroll_bar_h-1 {
			d.mv(row+1+scroll_bar_start, w-1);
			d.addch('#' as chtype);
		}
		d.mv(h-7, w-1);
		d.addch(disp_chars.hline_char);
		d.attroff(COLOR_PAIR(CLOGO));
	}
	
	// esc to close
	d.mv(0,0); buttons.Esc_to_close.print(None, l, d);
	
	TreeDisplayProperties {
		instructions_start_row,
		down_scroll,
		right_scroll,
		sel_loc
	}
}

