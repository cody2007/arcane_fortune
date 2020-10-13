use crate::disp_lib::*;
use super::vars::*;

pub fn getmaxyxu(d: &mut DispState) -> ScreenSz {
	let mut h_i32 = 0;
	let mut w_i32 = 0;
	d.getmaxyx(stdscr(), &mut h_i32, &mut w_i32);
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
		def_txt_color_pair: Option<chtype>, d: &mut DispState) {
	if let Some(tag) = tags.first() {
		let splits = txt.split(&tag.key).collect::<Vec<&str>>();
		let remaining_tags = if tags.len() > 1 {tags[1..].to_vec()} else {Vec::new()};
		
		// parse remaining
		color_tags_print(splits[0], &remaining_tags, def_txt_color_pair, d);
		for split in splits.iter().skip(1) {
			// print `val` in place of `key`
			addstr_attr(&tag.val, tag.attr, d);
			// parse remaining
			color_tags_print(split, &remaining_tags, def_txt_color_pair, d);
		}
	////////////////////////
	// nothing to replace just print (optionally) colored text
	}else if let Some(color_pair) = def_txt_color_pair {
		addstr_attr(txt, color_pair, d);
	}else{d.addstr(txt);}
}

