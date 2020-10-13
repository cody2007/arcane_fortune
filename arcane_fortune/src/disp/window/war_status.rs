use crate::disp::{ScreenSz, DispChars, set_player_color};
use crate::gcore::Relations;
use crate::player::*;
use crate::disp_lib::{DispState};
use super::*;

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn show_war_status_window(&self, relations: &Relations, players: &Vec<Player>,
			title_c: Option<chtype>, disp_chars: &DispChars,
			l: &Localization, buttons: &mut Buttons, d: &mut DispState) {
		let mut players_discov = Vec::with_capacity(players.len());
		let mut max_len = 0_i32;
		for player in players.iter() {
			if !player.stats.alive || !relations.discovered(self.cur_player as usize, player.id as usize) {
				continue;
			}
			match player.ptype {
				PlayerType::Barbarian(_) | PlayerType::Nobility(_) => {continue;}
				PlayerType::Empire(_) | PlayerType::Human(_) => {}
			}
			if player.personalization.nm.len() as i32 > max_len {max_len = player.personalization.nm.len() as i32;}
			players_discov.push(player);
		}
		
		let (h, w) = if players_discov.len() != 1 {
			(players_discov.len()*2 + 8,   max_len as usize + 3 + 3 + players_discov.len()*3)
		}else{
			(7, l.No_other_civs_discovered.len() + 5)
		};
		
		let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
		
		let mut y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let w = (w - 2) as i32;
		
		d.mv(y,x);
		center_txt(&l.Current_wars, w, title_c, d);
		y += 2;
		
		if players_discov.len() != 1 {
			macro_rules! hline{($left: expr, $right: expr) => {
				/*d.mv(y+1, x);
				for _ in 0..(max_len + 1 + 3*(players_discov.len() as i32)) {
					d.addch(disp_chars.hline_char);
				}*/
				d.mv(y+1, x + max_len + 1);
				d.addch($left);
				for _ in 1..(3*(players_discov.len() as i32)) {
					d.addch(disp_chars.hline_char);
				}
				d.addch($right);
			}};
			
			// top line labels
			d.mv(y, x + max_len + 2);
			for owner_i in players_discov.iter() {
				set_player_color(owner_i, true, d);
				d.addstr(&format!("{}.", owner_i.personalization.nm.chars().nth(0).unwrap()));
				set_player_color(owner_i, false, d);
				d.addch(' ');
			}
			hline!(disp_chars.ulcorner_char, disp_chars.urcorner_char);
			y += 2;
			
			// print each row and column
			for (row_offset_i, owner_i) in players_discov.iter().enumerate() {
				d.mv(y, x + max_len - owner_i.personalization.nm.len() as i32);
				set_player_color(owner_i, true, d);
				d.addstr(&owner_i.personalization.nm);
				set_player_color(owner_i, false, d);
				
				d.mv(y, x + max_len + 1);
				d.addch(disp_chars.vline_char);
				for (row_offset_j, owner_j) in players_discov.iter().enumerate() {
					let color = COLOR_PAIR(
						if row_offset_i == row_offset_j {
							CBLACK
						}else	if relations.at_war(owner_i.id as usize, owner_j.id as usize) 
							{CRED} else {CBLUE});
					d.attron(color);
					d.addch(disp_chars.land_char);
					d.addch(disp_chars.land_char);
					d.attroff(color);
					d.addch(disp_chars.vline_char);
				}
				
				hline!(disp_chars.llcorner_char, disp_chars.lrcorner_char);
				y += 2;
			}
		
		// no one else discovered yet
		}else{
			d.mv(y,x);
			center_txt(&l.No_other_civs_discovered, w, None, d);
			y += 1;
		}
		
		// instructions
		{
			let button = &mut buttons.Esc_to_close;
			let gap = ((w - button.print_txt(l).len() as i32)/2) as i32;
			d.mv(y + 1, x - 1 + gap);
			button.print(None, l, d);
		}
	}
}
