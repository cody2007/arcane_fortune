use crate::disp_lib::*;
use super::fns::getmaxyxu;
use super::*;
use crate::map::{H_ROOT, W_ROOT};
use crate::gcore::{GameDifficulties, Bonuses};
use crate::localization::Localization;

pub struct GameOptions {
	pub zoom_in_depth: usize, // how many upsampled zoom levels to get to full zoom
	pub n_players: usize,
	
	pub ai_bonuses: Bonuses
}

pub fn new_game_options(game_opts: &mut GameOptions, game_difficulties: &GameDifficulties,
		l: &Localization, disp_chars: &DispChars, buttons: &mut Buttons, d: &mut DispState) -> bool {
	let screen_sz = getmaxyxu(d);
	
	let title_c = Some(COLOR_PAIR(CGREEN));
	
	macro_rules! center_txt{($txt: expr, $w: expr, $color: expr) => {
		let g = ($w as usize - $txt.len())/2;
		let mut sp = String::new();
		for _ in 0..g {sp.push(' ');}
		
		if let Some(c) = $color {d.attron::<chtype>(c);}
		
		d.addstr(&format!("{}{}", sp, $txt));
		
		if let Some(c) = $color {d.attroff::<chtype>(c);}
	};};
	
	let mut screen_sz_prev = screen_sz.clone();
	let mut key_pressed = 0_i32;
	
	d.clear();
	
	let window_w = 75;
	
	enum NewGamePage {
		Initial,
		DifficultySelection
	}
	
	let mut page = NewGamePage::Initial;
	let mut difficulty_ind_sel = game_difficulties.default_ind;
	
	loop {
		let screen_sz = getmaxyxu(d);
		let mouse_event = d.getmouse(key_pressed);

		// clear screen if terminal sz changes
		if screen_sz != screen_sz_prev {
			d.clear();
			screen_sz_prev = screen_sz.clone();
		}
		
		const N_LINES: usize = 25;
		let mut row = ((screen_sz.h - N_LINES)/2) as i32;
		d.mv(row,0);
		center_txt!(&l.New_game_options, screen_sz.w, title_c);
		
		row += 2;
		
		macro_rules! print_hline{($row: expr) => {
			d.mv($row, ((screen_sz.w - window_w)/2) as i32);
			for _ in 0..window_w {d.addch(disp_chars.hline_char);}
		};};
		
		macro_rules! print_enter_esc{($row: expr) => {
			// print "<Enter> to continue"
			{
				$row += 2; print_hline!($row); $row += 2;
				let col = (screen_sz.w - buttons.Confirm.print_txt(l).len() - 1) / 2;
				d.mv($row, col as i32);
				
				buttons.Confirm.print(None, l, d);
			}
			
			// print "<Esc> to cancel"
			{
				$row += 2;
				let col = (screen_sz.w - buttons.Cancel.print_txt(l).len()) / 2;
				d.mv($row, col as i32);
				
				buttons.Cancel.print(None, l, d);
			}
		};};
		
		print_hline!(row);
		
		match page {
			NewGamePage::Initial => {
				row += 2;
				d.mv(row, 0); row += 2;
				center_txt!(&l.World_size, screen_sz.w, None);
				
				d.mv(row, 0);
				d.attron(COLOR_PAIR(CGRAY));
				center_txt!(&l.Use_arrow_keys_to_select, screen_sz.w, None);
				d.attroff(COLOR_PAIR(CGRAY));
				row += 2;
				
				////// game size
				const MAX_ZOOM_IN: usize = 7;
				
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
					
					d.mv(row, col); row += 1;
					
					d.attron(shortcut_indicator());
					d.addstr(&format!("{}", depth));
					d.attroff(shortcut_indicator());
					
					d.addstr("  [");
					if depth != game_opts.zoom_in_depth {
						d.addch(' ');
					}else{
						d.attron(COLOR_PAIR(CRED));
						d.addch('X');
						d.attroff(COLOR_PAIR(CRED));
					}
					d.addstr(&format!("]  {} {:.1}x {}", l.approximately, area as f32 / area_compare, nm));
					
					let recommend_col = col + format!("1 [ ] {} 1.5x Rhode Island  ", l.approximately).len() as i32;
					if depth >= (MAX_ZOOM_IN-4) {
						d.mv(row-1, recommend_col);
						d.attron(COLOR_PAIR(CSAND3));
						d.addch(' ');
						d.addstr(&l.Experimental);
						d.attroff(COLOR_PAIR(CSAND3));
					}else if depth == 2 {
						d.mv(row-1, recommend_col);
						d.attron(COLOR_PAIR(CGREEN));
						d.addch(' ');
						d.addstr(&l.Recommended);
						d.attroff(COLOR_PAIR(CGREEN));
					}
					
					h = h_dim;
					w = w_dim;
				}
				
				row += 1; print_hline!(row); row += 2;
				
				// print Number of players
				{
					let col = (screen_sz.w - window_w)/2;
					d.mv(row, col as i32);
					d.addstr(&l.Number_of_players);
					
					d.clrtoeol();
					let player_str = format!("{} ", game_opts.n_players);
					d.mv(row, (col + window_w - (player_str.len() + 5)) as i32);
					d.addstr(&player_str);
					
					d.attron(shortcut_indicator());
					d.addch('+');
					d.attroff(shortcut_indicator());
					
					d.addstr(" | ");
					
					d.attron(shortcut_indicator());
					d.addch('-');
					d.attroff(shortcut_indicator());
				}
				
				print_enter_esc!(row);
				
				if buttons.Confirm.activated(key_pressed, &mouse_event) {
					d.clear();
					page = NewGamePage::DifficultySelection;
					
					// force refresh w/o key press
					d.refresh();
					d.set_mouse_to_arrow();
					key_pressed = ERR;
					continue;
				}else if buttons.Cancel.activated(key_pressed, &mouse_event) {
					d.clear();
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
						
						k if k == '+' as i32 || k == '=' as i32 => {
							if game_opts.n_players < PLAYER_COLORS.len() {
								game_opts.n_players += 1;
							}
						}
						
						k if k == '-' as i32 => {
							if game_opts.n_players > 1 {
								game_opts.n_players -= 1;
							}
						}	
						_ => {}
					}
				}
			}
			NewGamePage::DifficultySelection => {
				row += 2;
				d.mv(row, 0); row += 2;
				center_txt!(&l.At_which_level_are_your_leadership_skills, screen_sz.w, None);
				
				d.mv(row, 0);
				d.attron(COLOR_PAIR(CGRAY));
				center_txt!(&l.Use_arrow_keys_to_select, screen_sz.w, None);
				d.attroff(COLOR_PAIR(CGRAY));
				row += 2;
				
				let col = ((screen_sz.w - "[X]  ".len() - game_difficulties.longest_nm)/2) as i32;
				
				for (difficulty_ind, game_difficulty) in game_difficulties.difficulties.iter().enumerate() {
					d.mv(row, col); row += 1;
					d.addstr("  [");
					if difficulty_ind != difficulty_ind_sel {
						d.addch(' ');
					}else{
						d.attron(COLOR_PAIR(CRED));
						d.addch('X');
						d.attroff(COLOR_PAIR(CRED));
					}
					d.addstr(&format!("]  {}", game_difficulty.nm));
				}
				
				print_enter_esc!(row);
				
				if buttons.Confirm.activated(key_pressed, &mouse_event) {
					game_opts.ai_bonuses = game_difficulties.difficulties[difficulty_ind_sel].ai_bonuses.clone();
					d.set_mouse_to_arrow();
					return true;
				}else if buttons.Cancel.activated(key_pressed, &mouse_event) {
					d.clear();
					d.set_mouse_to_arrow();
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
		
		d.refresh();
		d.set_mouse_to_arrow();
		loop {
			key_pressed = d.getch();
			if key_pressed != ERR {break;}
		}
	}
}

