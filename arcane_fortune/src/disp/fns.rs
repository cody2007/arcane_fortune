use crate::renderer::*;
use super::vars::*;

pub fn getmaxyxu(r: &Renderer) -> ScreenSz {
	let mut h_i32 = 0;
	let mut w_i32 = 0;
	r.getmaxyx(stdscr(), &mut h_i32, &mut w_i32);
	ScreenSz {h: h_i32 as usize, w: w_i32 as usize,
		sz: (h_i32 * w_i32) as usize}
}

#[derive(Clone)]
pub struct KeyValColor {
	pub key: String,
	pub val: String,
	pub attr: chtype // output of COLOR_PAIR()
}

pub fn color_tags_txt(txt: &str, tags: &Vec<KeyValColor>) -> String {
	if let Some(tag) = tags.first() {
		let mut new_txt = txt.replace(&tag.key, &tag.val);
		for tag in tags.iter().skip(1) {
			new_txt = new_txt.replace(&tag.key, &tag.val);
		}
		new_txt
	}else{
		txt.to_string()
	}
}

use super::addstr_attr;
// each tag has a `key`, `val` and `color`.
// the `key` text is replaced with the `val` text and colorized with `color`
pub fn color_tags_print(txt: &str, tags: &Vec<KeyValColor>, 
		def_txt_color_pair: Option<chtype>, r: &mut Renderer) {
	if let Some(tag) = tags.first() {
		let splits = txt.split(&tag.key).collect::<Vec<&str>>();
		let remaining_tags = if tags.len() > 1 {tags[1..].to_vec()} else {Vec::new()};
		
		// parse remaining
		color_tags_print(splits[0], &remaining_tags, def_txt_color_pair, r);
		for split in splits.iter().skip(1) {
			// print `val` in place of `key`
			addstr_attr(&tag.val, tag.attr, r);
			// parse remaining
			color_tags_print(split, &remaining_tags, def_txt_color_pair, r);
		}
	////////////////////////
	// nothing to replace just print (optionally) colored text
	}else if let Some(color_pair) = def_txt_color_pair {
		addstr_attr(txt, color_pair, r);
	}else{r.addstr(txt);}
}

////////////////////////////////////////////////////////////////////////////////////
// these fns are due to a strange bug in ncurses not correctly activating the color attribute
// in the functions above

#[derive(Clone)]
pub struct KeyValColorInput {
	pub key: String,
	pub val: String,
	pub color: i32 // input of COLOR_PAIR()
}

pub fn color_input_tags_txt(txt: &str, tags: &Vec<KeyValColorInput>) -> String {
	if let Some(tag) = tags.first() {
		let mut new_txt = txt.replace(&tag.key, &tag.val);
		for tag in tags.iter().skip(1) {
			new_txt = new_txt.replace(&tag.key, &tag.val);
		}
		new_txt
	}else{
		txt.to_string()
	}
}

// each tag has a `key`, `val` and `color`.
// the `key` text is replaced with the `val` text and colorized with `color`
pub fn color_input_tags_print(txt: &str, tags: &Vec<KeyValColorInput>, 
		def_txt_color_pair: Option<chtype>, r: &mut Renderer) {
	if let Some(tag) = tags.first() {
		let splits = txt.split(&tag.key).collect::<Vec<&str>>();
		let remaining_tags = if tags.len() > 1 {tags[1..].to_vec()} else {Vec::new()};
		
		// parse remaining
		color_input_tags_print(splits[0], &remaining_tags, def_txt_color_pair, r);
		for split in splits.iter().skip(1) {
			// print `val` in place of `key`
			r.attron(COLOR_PAIR(tag.color));
			r.addstr(&tag.val);
			r.attroff(COLOR_PAIR(tag.color));
			
			// parse remaining
			color_input_tags_print(split, &remaining_tags, def_txt_color_pair, r);
		}
	////////////////////////
	// nothing to replace just print (optionally) colored text
	}else if let Some(color_pair) = def_txt_color_pair {
		addstr_attr(txt, color_pair, r);
	}else{r.addstr(txt);}
}

