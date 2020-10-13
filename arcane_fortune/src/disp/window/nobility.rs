use super::*;

// UIMode::AcceptNobilityIntoEmpire

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn print_accept_nobility_into_empire(&self, mode: usize, house_ind: usize, 
			players: &Vec<Player>, l: &Localization,
			disp_chars: &DispChars, buttons: &mut Buttons, d: &mut DispState) {
		const MAX_W: usize = 70;
		
		if let Some(house) = players[house_ind].ptype.house() {
			let txt = {
				let noble = &house.noble_pairs[house.head_noble_pair_ind].noble;
				
				l.noble_req_to_join
					.replace("[title]", 
							&if noble.gender_female
								{l.noble_female_title.clone()} else {l.noble_male_title.clone()}
						)
					.replace("[first_name]", &noble.name.first)
					.replace("[last_name]", &noble.name.last)
			};
			
			let wrapped_txt = wrap_txt(&txt, MAX_W);
			let w = min(MAX_W, txt.len()) + 4;
			let h = wrapped_txt.len() + 8;
			let w_pos = print_window(ScreenSz{w, h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d); mvl!();
			
			for line in wrapped_txt.iter() {
				mvl!();
				d.addstr(line);
			}
			
			mvl!();
			mvl!();
			let txt = if mode == 0 {format!("* {} *", l.Accept_nobility)} else {l.Accept_nobility.clone()};
			center_txt(&txt, w as i32, None, d);
			
			d.mv(row + y, x);
			let txt = if mode == 1 {format!("* {} *", l.Reject_nobility)} else {l.Reject_nobility.clone()};
			center_txt(&txt, w as i32, None, d);
		}
	}
}

pub fn accept_nobility_into_empire_keys(mode: &mut usize, house_ind: usize, cur_player: usize,
		key_pressed: i32, kbd: &KeyboardMap, relations: &mut Relations, players: &mut Vec<Player>,
		logs: &mut Vec<Log>, turn: usize) -> UIRet {
	const N_OPTS: usize = 2;
	match key_pressed {
		k if kbd.up(k) => {if *mode > 0 {*mode -= 1;} else {*mode = N_OPTS-1;}}
		k if kbd.down(k) => {if (*mode + 1) <= (N_OPTS-1) {*mode += 1;} else {*mode = 0;}}
		k if kbd.enter == k => {
			match *mode {
				// accept nobility into empire
				0 => {relations.join_as_fiefdom(house_ind, cur_player, players, logs, turn);
				// reject nobility
				} 1 => {
				} _ => {panicq!("unknown UI state");}
			}
			return UIRet::Inactive;
		} _ => {}
	}
	
	UIRet::Active
}


