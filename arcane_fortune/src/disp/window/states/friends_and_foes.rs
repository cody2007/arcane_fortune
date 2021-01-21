// heatmap of player moods
use crate::disp::*;
use crate::player::*;
use super::*;

pub struct FriendsAndFoesWindowState {}

impl FriendsAndFoesWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, relations: &Relations, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let players_discov = {
			let mut players_discov = Vec::with_capacity(players.len());
			
			let player_discov_filter = |player: &Player| {
				match player.ptype {
					PlayerType::Barbarian(_) => false,
					PlayerType::Nobility(_) | PlayerType::Empire(_) | PlayerType::Human(_) => {
						player.stats.alive && relations.discovered(dstate.iface_settings.cur_player as usize, player.id as usize)
					}
				}
			};
			
			for player in players.iter().filter(|player| player_discov_filter(player)) {
				let colors = { // colors for each row
					let mut colors = Vec::with_capacity(players.len());
					for player_j in players.iter().filter(|player| player_discov_filter(player)) {
						colors.push(if player.id == player_j.id || player.id == dstate.iface_settings.cur_player {
								CBLACK
							}else{
								let friendliness = relations.friendliness_toward(player.id as usize, player_j.id as usize, players);
								let thresh = |step| {
									const N_SCALE_COLORS: usize = 6;
									const INCREMENT: f32 = 2.*0.75/N_SCALE_COLORS as f32;
									
									step as f32 * INCREMENT - 0.75
								};
								
								if friendliness < thresh(0) {CBLUERED5} else
								if friendliness < thresh(1) {CBLUERED4} else
								if friendliness < thresh(2) {CBLUERED3} else
								if friendliness < thresh(3) {CBLUERED2} else
								if friendliness < thresh(4) {CBLUERED1} else {CBLUERED0}
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
			players_discov
		};
		
		let (w_pos, w_sz) = print_player_color_grid_window(&players_discov, dstate.local.Friends_and_foes.clone(), 5, dstate);
		
		// How does each row see each column?
		let mut row = w_pos.y + 3;
		dstate.mv(row, w_pos.x + (w_sz.w - dstate.local.How_does_each_row_see_each_column.len()) as isize / 2); row += 1;
		dstate.renderer.addstr(&dstate.local.How_does_each_row_see_each_column);
		
		{ // legend
			let colors = vec![CBLUERED5, CBLUERED2, CBLUERED0];
			let lbls = vec![&dstate.local.furiously, &dstate.local.neutrally, &dstate.local.enthusiastically];
			
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

