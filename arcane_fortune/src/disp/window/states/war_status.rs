use crate::disp::*;
use crate::player::*;
use super::*;

pub struct WarStatusWindowState {}

impl WarStatusWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, relations: &Relations, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let title_c = Some(COLOR_PAIR(TITLE_COLOR));
		let mut players_discov = Vec::with_capacity(players.len());
		let mut max_len = 0_i32;
		for player in players.iter() {
			if !player.stats.alive || !relations.discovered(dstate.iface_settings.cur_player as usize, player.id as usize) {
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
			(7, dstate.local.No_other_civs_discovered.len() + 5)
		};
		
		let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
		
		let mut y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let w = (w - 2) as i32;
		
		dstate.mv(y,x);
		center_txt(&dstate.local.Current_wars, w, title_c, &mut dstate.renderer);
		y += 2;
		
		if players_discov.len() != 1 {
			macro_rules! hline{($left: expr, $right: expr) => {
				/*dstate.mv(y+1, x);
				for _ in 0..(max_len + 1 + 3*(players_discov.len() as i32)) {
					dstate.addch(dstate.chars.hline_char);
				}*/
				dstate.mv(y+1, x + max_len + 1);
				dstate.addch($left);
				for _ in 1..(3*(players_discov.len() as i32)) {
					dstate.addch(dstate.chars.hline_char);
				}
				dstate.addch($right);
			}};
			
			// top line labels
			dstate.mv(y, x + max_len + 2);
			for owner_i in players_discov.iter() {
				set_player_color(owner_i, true, &mut dstate.renderer);
				dstate.renderer.addstr(&format!("{}.", owner_i.personalization.nm.chars().nth(0).unwrap()));
				set_player_color(owner_i, false, &mut dstate.renderer);
				dstate.addch(' ');
			}
			hline!(dstate.chars.ulcorner_char, dstate.chars.urcorner_char);
			y += 2;
			
			// print each row and column
			for (row_offset_i, owner_i) in players_discov.iter().enumerate() {
				dstate.mv(y, x + max_len - owner_i.personalization.nm.len() as i32);
				set_player_color(owner_i, true, &mut dstate.renderer);
				dstate.renderer.addstr(&owner_i.personalization.nm);
				set_player_color(owner_i, false, &mut dstate.renderer);
				
				dstate.mv(y, x + max_len + 1);
				dstate.addch(dstate.chars.vline_char);
				for (row_offset_j, owner_j) in players_discov.iter().enumerate() {
					let color = COLOR_PAIR(
						if row_offset_i == row_offset_j {
							CBLACK
						}else	if relations.at_war(owner_i.id as usize, owner_j.id as usize) 
							{CRED} else {CBLUE});
					dstate.attron(color);
					dstate.addch(dstate.chars.land_char);
					dstate.addch(dstate.chars.land_char);
					dstate.attroff(color);
					dstate.addch(dstate.chars.vline_char);
				}
				
				hline!(dstate.chars.llcorner_char, dstate.chars.lrcorner_char);
				y += 2;
			}
		
		// no one else discovered yet
		}else{
			dstate.mv(y,x);
			center_txt(&dstate.local.No_other_civs_discovered, w, None, &mut dstate.renderer);
			y += 1;
		}
		
		// instructions
		{
			let button = &mut dstate.buttons.Esc_to_close;
			let gap = ((w - button.print_txt(&dstate.local).len() as i32)/2) as i32;
			dstate.renderer.mv(y + 1, x - 1 + gap);
			button.print(None, &dstate.local, &mut dstate.renderer);
		}
		UIModeControl::UnChgd
	}
}

