use crate::renderer::*;
use super::fns::getmaxyxu;
use super::*;
use crate::map::{H_ROOT, W_ROOT};
use crate::gcore::*;
use crate::player::{Nms, Personalization};

// for each country
#[derive(Clone)]
pub struct ScenarioPersonalization {
	pub start_loc: Option<Coord>,
	pub personalization: Personalization
}

pub struct GameOptions {
	pub zoom_in_depth: usize, // how many upsampled zoom levels to get to full zoom
	pub player_personalizations: Vec<Option<ScenarioPersonalization>>, // for each country
	pub human_player_ind: usize,
	pub load_world_map: bool,
	
	pub ai_bonuses: Bonuses
}

impl GameOptions {
	pub fn new() -> Self {
		Self {
			zoom_in_depth: 2,
			player_personalizations: vec![None; 12],
			human_player_ind: HUMAN_PLAYER_IND,
			load_world_map: false,
			ai_bonuses: Bonuses::default()
		}
	}
	
	pub fn n_players(&self) -> usize {
		self.player_personalizations.len()
	}
}

impl <'f,'bt,'ut,'rt,'dt>DispState<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn new_game_options(&mut self, game_opts: &mut GameOptions, game_difficulties: &GameDifficulties,
			nms: &Nms, rng: &mut XorState) -> bool {
		let screen_sz = getmaxyxu(&self.renderer);
		
		let title_c = Some(COLOR_PAIR(CGREEN));
		
		macro_rules! center_txt{($txt: expr, $w: expr, $color: expr) => {
			let g = ($w as usize - $txt.len())/2;
			let mut sp = String::new();
			for _ in 0..g {sp.push(' ');}
			
			if let Some(c) = $color {self.attron::<chtype>(c);}
			
			self.renderer.addstr(&format!("{}{}", sp, $txt));
			
			if let Some(c) = $color {self.attroff::<chtype>(c);}
		};}
		
		let mut screen_sz_prev = screen_sz.clone();
		let mut key_pressed = 0_i32;
		
		self.renderer.clear();
		
		let window_w = 75;
		
		enum NewGamePage {
			ScenarioOrNot, // step 1
			
			/////////////// world map options
			CivilizationSelection, // step 2
			ScenarioWorldSzSelection, // step 3
			
			/////////////// random game options
			WorldSzAndNPlayers, // step 2
			
			//////////////// last step for both the world map/scenario & random game options
			DifficultySelection
		}
		
		let mut page = NewGamePage::ScenarioOrNot;
		let mut difficulty_ind_sel = game_difficulties.default_ind;
		
		loop {
			let screen_sz = getmaxyxu(&mut self.renderer);
			let mouse_event = self.renderer.getmouse(key_pressed);
			
			// clear screen if terminal sz changes
			if screen_sz != screen_sz_prev {
				self.renderer.clear();
				screen_sz_prev = screen_sz.clone();
			}
			
			const N_LINES: usize = 25;
			let mut row = ((screen_sz.h - N_LINES)/2) as i32;
			self.mv(row,0);
			center_txt!(&self.local.New_game_options, screen_sz.w, title_c);
			
			row += 2;
			
			macro_rules! print_hline{() => {
				self.mv(row, ((screen_sz.w - window_w)/2) as i32);
				for _ in 0..window_w {self.addch(self.chars.hline_char);}
			};}
			
			macro_rules! print_button{($button: ident) => { // print "<Esc> to cancel"
				row += 2;
				let col = (screen_sz.w - self.buttons.$button.print_txt(&self.local).len()) / 2;
				self.mv(row, col as i32);
				
				self.buttons.$button.print(None, &self.local, &mut self.renderer);
			};}
			
			macro_rules! print_enter_esc{() => {
				row += 2; print_hline!();
				print_button!(Confirm);
				print_button!(Cancel);
			};}
			
			macro_rules! force_refresh{() => {
				// force refresh w/o key press
				self.renderer.refresh();
				self.renderer.set_mouse_to_arrow();
				key_pressed = ERR;
			};}
			
			macro_rules! use_arrow_keys_to_select{() => {
				self.mv(row, 0);
				self.attron(COLOR_PAIR(CGRAY));
				center_txt!(&self.local.Use_arrow_keys_to_select, screen_sz.w, None);
				self.attroff(COLOR_PAIR(CGRAY));
				row += 2;
			};}
			
			macro_rules! proceed_to_difficulty_selection{() => {
				self.renderer.clear();
				page = NewGamePage::DifficultySelection;
				
				force_refresh!();
				continue;
			};}
			
			print_hline!();
			
			row += 2;
			self.mv(row, 0); row += 2;
			
			match page {
				NewGamePage::ScenarioOrNot => {
					center_txt!(&self.local.What_type_of_game_do_you_want_to_play, screen_sz.w, None);
					
					print_button!(A_historical_scenario);
					
					{ // Random map and countries (centered w/ historical scenario option
						row += 2;
						let col = (screen_sz.w - self.buttons.A_historical_scenario.print_txt(&self.local).len()) / 2;
						self.mv(row, col as i32);
						
						self.buttons.Random_map_and_countries.print(None, &self.local, &mut self.renderer);
					}
					
					row += 2; print_hline!();
					print_button!(Cancel);
					
					/////////////////////////////////////////////////
					// keys
					if self.buttons.A_historical_scenario.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						page = NewGamePage::CivilizationSelection;
						game_opts.init_world_map_scenario(nms, rng);
						force_refresh!();
						continue;
					}else if self.buttons.Random_map_and_countries.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						page = NewGamePage::WorldSzAndNPlayers;
						game_opts.init_random_opts();
						force_refresh!();
						continue;
					}else if self.buttons.Cancel.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						return false;
					}
				
				////////////////////////////////
				// world map options
				} NewGamePage::CivilizationSelection => {
					center_txt!(&self.local.What_civilization_do_you_want_to_play_as, screen_sz.w, None);
					use_arrow_keys_to_select!();
					
					if screen_reader_mode() {
						center_txt!(&self.local.txt_shown_as_two_cols, screen_sz.w, None);
					}
					
					let row_start = row;
					const COL_WIDTH: usize = 10;
					let mut col = ((screen_sz.w - "[X]  ".len() - COL_WIDTH*2)/2) as i32;
					
					for (player_ind, player) in game_opts.player_personalizations.iter().enumerate() {
						if let Some(player) = player {
							self.mv(row, col); row += 1;
							self.renderer.addstr("  [");
							if player_ind != game_opts.human_player_ind {
								self.addch(' ');
							}else{
								self.attron(COLOR_PAIR(CRED));
								self.addch('X');
								self.attroff(COLOR_PAIR(CRED));
							}
							self.renderer.addstr(&format!("]  {}", player.personalization.nm));
							
							// move to second column
							if (game_opts.player_personalizations.len()/2) == player_ind {
								col += COL_WIDTH as i32 + 5;
								row = row_start;
							}
						}
					}
					
					print_enter_esc!();
					
					//////////////////////////////
					// keys
					if self.buttons.Confirm.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						page = NewGamePage::ScenarioWorldSzSelection;
						
						force_refresh!();
						continue;
					}else if self.buttons.Cancel.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						return false;
					}else{
						match key_pressed {
							KEY_DOWN => {
								if game_opts.human_player_ind < (game_opts.player_personalizations.len()-1) {
									game_opts.human_player_ind += 1;
								}else{
									game_opts.human_player_ind = 0;
								}
							}
							
							KEY_UP => {
								if game_opts.human_player_ind > 0 {
									game_opts.human_player_ind -= 1;
								}else{
									game_opts.human_player_ind = game_opts.player_personalizations.len()-1;
								}
							}
							
							_ => {}
						}
					}
				
				////////////////////////////////
				// scenario world size selection
				} NewGamePage::ScenarioWorldSzSelection => {
					const MIN_ZOOM_IN: usize = 2;
					const MAX_ZOOM_IN: usize = 3;
					
					{ // printing
						center_txt!(&self.local.World_size, screen_sz.w, None);
						
						self.mv(row, 0);
						self.attron(COLOR_PAIR(CGRAY));
						center_txt!(&self.local.Use_arrow_keys_to_select, screen_sz.w, None);
						self.attroff(COLOR_PAIR(CGRAY));
						row += 2;
						
						////// game size
						let col = ((screen_sz.w - "2  [X]  approximately 1.5x Rhode Island  (Recommended)".len())/2) as i32;
						
						let mut h = H_ROOT;
						let mut w = W_ROOT;
						
						for depth in MIN_ZOOM_IN..=MAX_ZOOM_IN {
							let (h_dim, _) = upsample_dim(h);
							let (w_dim, _) = upsample_dim(w);
							
							let (area_compare, nm) = if depth <= 2 {
									(      4_001., "Rhode Island")
								}else{
									(    104_656., "Kentucky")
							};
							
							let area = (h_dim*5/1000)*(w_dim*5/1000);
							
							self.mv(row, col); row += 1;
							
							self.attron(shortcut_indicator());
							self.renderer.addstr(&format!("{}", depth));
							self.attroff(shortcut_indicator());
							
							self.renderer.addstr("  [");
							if depth != game_opts.zoom_in_depth {
								self.addch(' ');
							}else{
								self.attron(COLOR_PAIR(CRED));
								self.addch('X');
								self.attroff(COLOR_PAIR(CRED));
							}
							self.renderer.addstr(&format!("]  {} {:.1}x {}", self.local.approximately, area as f32 / area_compare, nm));
							
							let recommend_col = col + format!("1 [ ] {} 1.5x Rhode Island  ", self.local.approximately).len() as i32;
							if depth > MIN_ZOOM_IN {
								self.mv(row-1, recommend_col);
								self.attron(COLOR_PAIR(CSAND3));
								self.addch(' ');
								self.renderer.addstr(&self.local.Experimental);
								self.attroff(COLOR_PAIR(CSAND3));
							}else{
								self.mv(row-1, recommend_col);
								self.attron(COLOR_PAIR(CGREEN));
								self.addch(' ');
								self.renderer.addstr(&self.local.Recommended);
								self.attroff(COLOR_PAIR(CGREEN));
							}
							
							h = h_dim;
							w = w_dim;
						}
						
						row += 1; print_hline!(); row += 2;
						
						print_enter_esc!();
					}
					
					////////////////////////////
					// keys
					if self.buttons.Confirm.activated(key_pressed, &mouse_event) {
						game_opts.set_world_scenario_start_locs();
						proceed_to_difficulty_selection!();
					}else if self.buttons.Cancel.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						return false;
					}else{
						match key_pressed {
							k if k == '2' as i32 => {game_opts.zoom_in_depth = 2;}
							k if k == '3' as i32 => {game_opts.zoom_in_depth = 3;}
							
							KEY_DOWN => {
								if game_opts.zoom_in_depth < MAX_ZOOM_IN {
									game_opts.zoom_in_depth += 1;
								}else{
									game_opts.zoom_in_depth = MIN_ZOOM_IN;
								}
							}
							
							KEY_UP => {
								if game_opts.zoom_in_depth > MIN_ZOOM_IN {
									game_opts.zoom_in_depth -= 1;
								}else{
									game_opts.zoom_in_depth = MAX_ZOOM_IN;
								}
							}	
							_ => {}
						}
					}
				
				////////////////////////////////
				// random game options
				} NewGamePage::WorldSzAndNPlayers => {
					const MAX_ZOOM_IN: usize = 7;
					
					{ // printing
						center_txt!(&self.local.World_size, screen_sz.w, None);
						
						self.mv(row, 0);
						self.attron(COLOR_PAIR(CGRAY));
						center_txt!(&self.local.Use_arrow_keys_to_select, screen_sz.w, None);
						self.attroff(COLOR_PAIR(CGRAY));
						row += 2;
						
						////// game size
						let col = ((screen_sz.w - "2  [X]  approximately 1.5x Rhode Island  (Recommended)".len())/2) as i32;
						
						let mut h = H_ROOT;
						let mut w = W_ROOT;
						
						for depth in 1..=MAX_ZOOM_IN {
							let (h_dim, _) = upsample_dim(h);
							let (w_dim, _) = upsample_dim(w);
							
							let (area_compare, nm) = if depth <= 1 {
									(        232.14, "Boston")
								}else if depth <= 2 {
									(      4_001., "Rhode Island")
								}else if depth <= 3 {
									(    104_656., "Kentucky")
								}else if depth <= 4 {
									(  1_723_337., "Alaska")
								}else if depth < 6 {
									( 17_098_240., "Russia")
								}else{ // 144_798_500 Mars
									(510_071_000., "Earth")
							};
							
							let area = (h_dim*5/1000)*(w_dim*5/1000);
							
							self.mv(row, col); row += 1;
							
							self.attron(shortcut_indicator());
							self.renderer.addstr(&format!("{}", depth));
							self.attroff(shortcut_indicator());
							
							self.renderer.addstr("  [");
							if depth != game_opts.zoom_in_depth {
								self.addch(' ');
							}else{
								self.attron(COLOR_PAIR(CRED));
								self.addch('X');
								self.attroff(COLOR_PAIR(CRED));
							}
							self.renderer.addstr(&format!("]  {} {:.1}x {}", self.local.approximately, area as f32 / area_compare, nm));
							
							let recommend_col = col + format!("1 [ ] {} 1.5x Rhode Island  ", self.local.approximately).len() as i32;
							if depth >= (MAX_ZOOM_IN-4) {
								self.mv(row-1, recommend_col);
								self.attron(COLOR_PAIR(CSAND3));
								self.addch(' ');
								self.renderer.addstr(&self.local.Experimental);
								self.attroff(COLOR_PAIR(CSAND3));
							}else if depth == 2 {
								self.mv(row-1, recommend_col);
								self.attron(COLOR_PAIR(CGREEN));
								self.addch(' ');
								self.renderer.addstr(&self.local.Recommended);
								self.attroff(COLOR_PAIR(CGREEN));
							}
							
							h = h_dim;
							w = w_dim;
						}
						
						row += 1; print_hline!(); row += 2;
						
						{ // print Number of players
							let col = (screen_sz.w - window_w)/2;
							self.mv(row, col as i32);
							self.renderer.addstr(&self.local.Number_of_players);
							
							self.renderer.clrtoeol();
							let player_str = format!("{} ", game_opts.n_players());
							self.mv(row, (col + window_w - (player_str.len() + 5)) as i32);
							self.renderer.addstr(&player_str);
							
							self.attron(shortcut_indicator());
							self.addch('+');
							self.attroff(shortcut_indicator());
							
							self.renderer.addstr(" | ");
							
							self.attron(shortcut_indicator());
							self.addch('-');
							self.attroff(shortcut_indicator());
						}
						
						print_enter_esc!();
					}
					
					////////////////////////////
					// keys
					if self.buttons.Confirm.activated(key_pressed, &mouse_event) {proceed_to_difficulty_selection!();
					}else if self.buttons.Cancel.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						return false;
					}else{
						match key_pressed {
							k if k == '1' as i32 => {game_opts.zoom_in_depth = 1;}
							k if k == '2' as i32 => {game_opts.zoom_in_depth = 2;}
							k if k == '3' as i32 => {game_opts.zoom_in_depth = 3;}
							k if k == '4' as i32 => {game_opts.zoom_in_depth = 4;}
							k if k == '5' as i32 => {game_opts.zoom_in_depth = 5;}
							k if k == '6' as i32 => {game_opts.zoom_in_depth = 6;}
							
							KEY_DOWN => {
								if game_opts.zoom_in_depth < MAX_ZOOM_IN {
									game_opts.zoom_in_depth += 1;
								}else{
									game_opts.zoom_in_depth = 1;
								}
							}
							
							KEY_UP => {
								if game_opts.zoom_in_depth > 1 {
									game_opts.zoom_in_depth -= 1;
								}else{
									game_opts.zoom_in_depth = MAX_ZOOM_IN;
								}
							}
							
							// increment # players
							k if k == '+' as i32 || k == '=' as i32 => {
								if game_opts.n_players() < PLAYER_COLORS.len() {
									game_opts.player_personalizations.push(None);
								}
							}
							
							// decrement # players
							k if k == '-' as i32 => {
								if game_opts.n_players() > 1 {
									game_opts.player_personalizations.pop();
								}
							}	
							_ => {}
						}
					}
				}
				
				///////////////////
				// last page before creating game (used for both world map and random games)
				// sets game_opts.ai_bonuses
				NewGamePage::DifficultySelection => {
					center_txt!(&self.local.At_which_level_are_your_leadership_skills, screen_sz.w, None);
					use_arrow_keys_to_select!();
					
					let col = ((screen_sz.w - "[X]  ".len() - game_difficulties.longest_nm)/2) as i32;
					
					for (difficulty_ind, game_difficulty) in game_difficulties.difficulties.iter().enumerate() {
						self.mv(row, col); row += 1;
						self.renderer.addstr("  [");
						if difficulty_ind != difficulty_ind_sel {
							self.addch(' ');
						}else{
							self.attron(COLOR_PAIR(CRED));
							self.addch('X');
							self.attroff(COLOR_PAIR(CRED));
						}
						self.renderer.addstr(&format!("]  {}", game_difficulty.nm));
					}
					
					print_enter_esc!();
					
					if self.buttons.Confirm.activated(key_pressed, &mouse_event) {
						game_opts.ai_bonuses = game_difficulties.difficulties[difficulty_ind_sel].ai_bonuses.clone();
						self.renderer.set_mouse_to_arrow();
						return true;
					}else if self.buttons.Cancel.activated(key_pressed, &mouse_event) {
						self.renderer.clear();
						self.renderer.set_mouse_to_arrow();
						return false;
					}else{
						match key_pressed {
							KEY_DOWN => {
								if difficulty_ind_sel < (game_difficulties.difficulties.len() - 1) {
									difficulty_ind_sel += 1;
								}else{
									difficulty_ind_sel = 0;
								}
							}
							
							KEY_UP => {
								if difficulty_ind_sel > 0 {
									difficulty_ind_sel -= 1;
								}else{
									difficulty_ind_sel = game_difficulties.difficulties.len() - 1;
								}
							}
							_ => {}
						}
					}
				}
			}
			
			self.renderer.refresh();
			self.renderer.set_mouse_to_arrow();
			
			// key handling occurs after printing. therefore, the frame that was
			// last displayed will be stale if a selection was changed.
			// we could put the key handling before the printing but that has issues
			// because we'd be handling self.buttons that had never been printed and this 
			// may not behave nicely.
			if key_pressed != ERR {
				key_pressed = self.renderer.getch();
				continue;
			}
			
			// prevent burning cpu cycles -- don't update the screen unless something changed
			loop {
				key_pressed = self.renderer.getch();
				if key_pressed != ERR {break;}
			}
		}
	}
}

