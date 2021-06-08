// heatmap of who is at war with who
use crate::disp::*;
use crate::player::*;
use super::*;

pub struct WarStatusWindowState {}

pub struct PlayerPrintColors<'p,'bt,'ut,'rt,'dt> {
	pub player: &'p Player<'bt,'ut,'rt,'dt>,
	pub colors: Vec<i32>
}

pub fn print_player_color_grid_window(players_print: &Vec<PlayerPrintColors>, title: String, 
		n_lines_skip: i32, dstate: &mut DispState) -> (Coord, ScreenSz) {
	let title_c = Some(COLOR_PAIR(TITLE_COLOR));
	
	let max_len = players_print.iter().map(|player| player.player.personalization.nm.len()).max().unwrap() as i32;
	
	let (h, w) = if players_print.len() != 1 {
		(players_print.len()*2 + 8 + n_lines_skip as usize,   max_len as usize + 3 + 3 + players_print.len()*3)
	}else{
		(7 + n_lines_skip as usize, dstate.local.No_other_civs_discovered.len() + 5)
	};
	
	let w = max(w, 38);
	
	let w_sz = ScreenSz {w, h, sz:0};
	let w_pos = dstate.print_window(w_sz);
	
	let mut y = w_pos.y as i32 + 1;
	let x = w_pos.x as i32 + 2;
	
	let w = (w - 2) as i32;
	
	dstate.mv(y,x);
	center_txt(&title, w, title_c, &mut dstate.renderer);
	y += 2 + n_lines_skip;
	
	if players_print.len() != 1 {
		macro_rules! hline{($left: expr, $right: expr) => {
			/*dstate.mv(y+1, x);
			for _ in 0..(max_len + 1 + 3*(players_discov.len() as i32)) {
				dstate.addch(dstate.chars.hline_char);
			}*/
			dstate.mv(y+1, x + max_len + 1);
			dstate.addch($left);
			for _ in 1..(3*(players_print.len() as i32)) {
				dstate.addch(dstate.chars.hline_char);
			}
			dstate.addch($right);
		}}
		
		// top line labels
		dstate.mv(y, x + max_len + 2);
		for owner_i in players_print.iter() {
			set_player_color(owner_i.player, true, &mut dstate.renderer);
			dstate.renderer.addstr(&format!("{}.", owner_i.player.personalization.nm.chars().nth(0).unwrap()));
			set_player_color(owner_i.player, false, &mut dstate.renderer);
			dstate.addch(' ');
		}
		hline!(dstate.chars.ulcorner_char, dstate.chars.urcorner_char);
		y += 2;
		
		// print each row and column
		for player_print in players_print.iter() {
			dstate.mv(y, x + max_len - player_print.player.personalization.nm.len() as i32);
			set_player_color(player_print.player, true, &mut dstate.renderer);
			dstate.renderer.addstr(&player_print.player.personalization.nm);
			set_player_color(player_print.player, false, &mut dstate.renderer);
			
			dstate.mv(y, x + max_len + 1);
			dstate.addch(dstate.chars.vline_char);
			for color in player_print.colors.iter() {
				let color = COLOR_PAIR(*color);
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
	
	{ // instructions
		let button = &mut dstate.buttons.Esc_to_close;
		let gap = ((w - button.print_txt(&dstate.local).len() as i32)/2) as i32;
		dstate.renderer.mv(y + 1, x - 1 + gap);
		button.print(None, &dstate.local, &mut dstate.renderer);
	}
	
	(w_pos, w_sz)
}

const WAR_COLOR: CInt = CRED;
const PEACE_COLOR: CInt = CBLUE;
const DEFENSIVE_PACT_COLOR: CInt = CGREEN3;
const FIEFDOM_COLOR: CInt = CGREEN2;
const KINGDOM_COLOR: CInt = CLOGO;

impl WarStatusWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, relations: &Relations, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut players_discov = Vec::with_capacity(players.len());
		
		let player_discov_filter = |player: &Player| {
			match player.ptype {
				PlayerType::Barbarian(_) | PlayerType::Nobility(_) => false,
				PlayerType::Empire(_) | PlayerType::Human(_) => {
					player.stats.alive && relations.discovered(dstate.iface_settings.cur_player as usize, player.id as usize)
				}
			}
		};
		
		for player in players.iter().filter(|player| player_discov_filter(player)) {
			let colors = { // colors for each row
				let mut colors = Vec::with_capacity(players.len());
				for player_j in players.iter().filter(|player| player_discov_filter(player)) {
					colors.push(if player.id == player_j.id {
							CBLACK
						}else{
							match relations.status(player.id as usize, player_j.id as usize) {
								RelationStatus::Undiscovered | RelationStatus::Peace(_) => {
									// defensive pact
									if relations.defensive_pact_owners(player_j.id as usize).contains(&(player.id as usize)) {
										DEFENSIVE_PACT_COLOR
									// peace only, no defensive pact
									}else{
										PEACE_COLOR
									}
								}
								RelationStatus::War {..} => WAR_COLOR,
								RelationStatus::Fiefdom {..} => {
									if player_j.ptype.is_nobility() {
										FIEFDOM_COLOR
									}else{
										PEACE_COLOR
									}
								}
								RelationStatus::Kingdom {kingdom_id, ..} => {
									if *kingdom_id == player_j.id as usize {
										KINGDOM_COLOR
									}else{
										PEACE_COLOR
									}
								}
							}
						}
					);
				}
				colors
			};
			
			players_discov.push(PlayerPrintColors {
				player,
				colors
			});
		}
		
		// legend values
		let colors = vec![WAR_COLOR, PEACE_COLOR, DEFENSIVE_PACT_COLOR,
			    		KINGDOM_COLOR];//, FIEFDOM_COLOR];
		
		let (w_pos, w_sz) = print_player_color_grid_window(&players_discov, dstate.local.Current_wars.clone(), colors.len() as i32+1, dstate);
		
		// legend values part 2. needs to be below the print call because the compiler will complain about the dstate mutable ref
		let lbls = vec![&dstate.local.War, &dstate.local.Peace, &dstate.local.Defensive_pact,
				&dstate.local.Kingdom];//, &dstate.local.Fiefdom];
		
		// How does each row see each column?
		let mut row = w_pos.y + 3;
		
		{ // print legend
			let max_len = lbls.iter().map(|lbl| lbl.len()).max().unwrap();
			
			for (color, lbl) in colors.iter().zip(lbls.iter()) {
				dstate.renderer.mv(row, w_pos.x + 2 + (w_sz.w - max_len) as isize/2); row += 1;
				dstate.renderer.attron(COLOR_PAIR(*color));
				dstate.renderer.addch(dstate.chars.land_char);
				dstate.renderer.attroff(COLOR_PAIR(*color));
				dstate.renderer.addstr(" = ");
				dstate.renderer.addstr(lbl);
			}
		}
	
		UIModeControl::UnChgd
	}
}

