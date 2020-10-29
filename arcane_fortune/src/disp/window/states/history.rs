use super::*;
pub struct HistoryWindowState {
	pub scroll_first_line: usize,
	pub htype: HistoryType
}

pub enum HistoryType {World, Battle, Economic}

impl HistoryWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, gstate: &GameState, temps: &Templates,
			dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		let h = dstate.iface_settings.screen_sz.h as i32;
		let w = dstate.iface_settings.screen_sz.w as i32;
		
		let title_c = Some(COLOR_PAIR(TITLE_COLOR));
		
		dstate.renderer.clear();
		let events = match self.htype {
			HistoryType::World => {
				center_txt(&dstate.local.World_History, w, title_c, &mut dstate.renderer);
				world_history_events(cur_player, gstate)
			}
			HistoryType::Battle => {
				center_txt(&dstate.local.Battle_History, w, title_c, &mut dstate.renderer);
				battle_history_events(cur_player, gstate)
			}
			HistoryType::Economic => {
				center_txt(&dstate.local.Economic_History, w, title_c, &mut dstate.renderer);
				economic_history_events(cur_player, gstate)
			}
		};
		
		dstate.mv(0,0); dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
		
		// no data
		if events.len() == 0 {
			dstate.mv(h/2, 0);
			center_txt(&dstate.local.No_logs, w, None, &mut dstate.renderer);
			return UIModeControl::UnChgd;
		}
		
		if h <= (LOG_START_ROW + LOG_STOP_ROW) {return UIModeControl::UnChgd;} // screen not tall enough
		
		let n_rows_plot = h - LOG_START_ROW - LOG_STOP_ROW;
		
		// check to make sure scroll line is within range (ex. if screen sz chgs)
		if events.len() <= self.scroll_first_line || (events.len() - self.scroll_first_line) < n_rows_plot as usize {
			if events.len() > n_rows_plot as usize {
				self.scroll_first_line = events.len() - n_rows_plot as usize;
			}else{
				self.scroll_first_line = 0;
			}
		}
		
		// plot log
		{
			let mut row_counter = 0;
			
			for log in events[self.scroll_first_line..].iter() {
				if row_counter >= n_rows_plot {break;}
				if !log.visible(cur_player, &gstate.relations) {continue;}
				
				dstate.mv(LOG_START_ROW + row_counter, 0);
				dstate.local.print_date_log(log.turn, &mut dstate.renderer);
				print_log(&log.val, true, players, temps.doctrines, dstate);
				
				row_counter += 1;
			}
		}
		
		/////// print scroll instructions
		{
			/*let center_txt = |txt: &str, w| {
				let g = (w as usize - txt.len())/2;
				let mut sp = String::new();
				for _ in 0..g {sp.push(' ');}
				dstate.attron(title_c.unwrap());
				d.addstr(&format!("{}{}", sp, txt));
				dstate.attroff(title_c.unwrap());
			};*/
			
			let row = h - LOG_START_ROW;
			let instructions_width = "<Arrow keys>: Scroll".len() as i32;
			let col = (w - instructions_width)/2;
			/*macro_rules! nl{() => {dstate.mv(row, col); row += 1;};
			($last:expr) => {dstate.mv(row,col);};};*/
			
			// instructions
			/*nl!();
			center_txt("Keys", instructions_width);
			nl!(); nl!();*/
			dstate.mv(row,col);
			dstate.attron(COLOR_PAIR(ESC_COLOR));
			dstate.addstr(&format!("<{}>", dstate.local.Arrow_keys));
			dstate.attroff(COLOR_PAIR(ESC_COLOR));
			dstate.addstr(&format!(": {}", dstate.local.Scroll_view));
		}
		
		//////// print scroll bars
		if self.scroll_first_line != 0 || (events.len() - self.scroll_first_line) > n_rows_plot as usize {
			let scroll_track_h = n_rows_plot;
			let frac_covered = n_rows_plot as f32 / events.len() as f32;
			let scroll_bar_h = max(((scroll_track_h as f32) * frac_covered).round() as i32, 1);
			debug_assertq!(frac_covered <= 1.);
			
			let frac_at = self.scroll_first_line as f32 / events.len() as f32;
			let scroll_bar_start = ((scroll_track_h as f32) * frac_at).round() as i32;
			
			dstate.mv(LOG_START_ROW, w-1);
			dstate.attron(COLOR_PAIR(CLOGO));
			dstate.addch(dstate.chars.hline_char);
			for row in 0..=scroll_bar_h-1 {
				dstate.mv(row + 1 + scroll_bar_start + LOG_START_ROW, w-1);
				dstate.addch(dstate.chars.vline_char);
				//dstate.addch('#' as chtype);
			}
			dstate.mv(h-LOG_STOP_ROW, w-1);
			dstate.addch(dstate.chars.hline_char);
			dstate.attroff(COLOR_PAIR(CLOGO));
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &GameState, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const SCROLL_FASTER_SPEED: usize = 10;
		let cur_player = dstate.iface_settings.cur_player as usize;
		
		let events = match self.htype {
			HistoryType::World {..} => {
				world_history_events(cur_player, gstate)
			} HistoryType::Battle {..} => {
				battle_history_events(cur_player, gstate)
			} HistoryType::Economic {..} => {
				economic_history_events(cur_player, gstate)
			} _ => {panicq!("unhandled UI mode");}
		};
		
		let h = dstate.iface_settings.screen_sz.h;
		
		match dstate.key_pressed {
			// scroll down
			k if dstate.kbd.down_normal(k) => {
				if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
					let n_rows_plot = h - (LOG_START_ROW + LOG_STOP_ROW) as usize;
					
					// check to make sure the log is long enough to keep scrolling
					if (events.len() - self.scroll_first_line) > n_rows_plot {
						self.scroll_first_line += 1;
					}
				}
			
			// scroll down (faster)
			} k if k == dstate.kbd.fast_down as i32 => {
				if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
					let n_rows_plot = h - (LOG_START_ROW + LOG_STOP_ROW) as usize;
					
					// check to make sure the log is long enough to keep scrolling
					if (events.len() - self.scroll_first_line) > n_rows_plot {
						self.scroll_first_line += SCROLL_FASTER_SPEED;
						if (events.len() - self.scroll_first_line) <= n_rows_plot {
							self.scroll_first_line = events.len() - n_rows_plot;
						}
					}
				}
			
			// scroll up
			} k if dstate.kbd.up_normal(k) => {
				if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
					// check to make sure the log is long enough to keep scrolling
					if self.scroll_first_line > 0 {
						self.scroll_first_line -= 1;
					}
				}
			
			// scroll up (faster)
			} k if k == dstate.kbd.fast_up as i32 => {
				if h > (LOG_START_ROW + LOG_STOP_ROW) as usize { // screen tall enough
					// check to make sure the log is long enough to keep scrolling
					if self.scroll_first_line > 0 {
						if self.scroll_first_line >= SCROLL_FASTER_SPEED {
							self.scroll_first_line -= SCROLL_FASTER_SPEED;
						}else{
							self.scroll_first_line = 0;
						}
					}
				}	
			} _ => {}
		}
	
		UIModeControl::UnChgd
	}
}

pub fn world_history_events(player_id: usize, gstate: &GameState) -> Vec<Log> {
	let mut events = Vec::with_capacity(gstate.logs.len());
	for log in gstate.logs.iter()
			.filter(|log| log.visible(player_id, &gstate.relations)) {
		match log.val {
			LogType::CivCollapsed {..} |
			LogType::CivDestroyed {..} |
			LogType::CityCaptured {..} |
			LogType::CityDisbanded {..} |
			LogType::CityDestroyed {..} |
			LogType::CityFounded {..} |
			LogType::CivDiscov {..} |
			LogType::WarDeclaration {..} |
			LogType::PeaceDeclaration {..} |
			LogType::PrevailingDoctrineChanged {..} |
			LogType::Rioting {..} |
			LogType::NobleHouseJoinedEmpire {..} |
			LogType::RiotersAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::ICBMDetonation {..} => {
				events.push(log.clone());
			}
			
			LogType::Debug {..} | 
			LogType::UnitDestroyed {..} |
			LogType::UnitAttacked {..} |
			LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} |
			LogType::StructureAttacked {..} => {}
		}
	}
	events
}

pub fn battle_history_events(player_id: usize, gstate: &GameState) -> Vec<Log> {
	let mut events = Vec::with_capacity(gstate.logs.len());
	for log in gstate.logs.iter().filter(|log| log.visible(player_id, &gstate.relations)) {
		match log.val {
			LogType::CivCollapsed {..} |
			LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} |
			LogType::CityDisbanded {..} |
			LogType::CityDestroyed {..} |
			LogType::CityFounded {..} |
			LogType::CivDiscov {..} |
			LogType::WarDeclaration {..} |
			LogType::PeaceDeclaration {..} |
			LogType::PrevailingDoctrineChanged {..} |
			LogType::Rioting {..} |
			LogType::RiotersAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::NobleHouseJoinedEmpire {..} |
			LogType::Debug {..} => {}
			
			LogType::CivDestroyed {..} |
			LogType::CityCaptured {..} |
			LogType::ICBMDetonation {..} |
			LogType::UnitDestroyed {..} |
			LogType::UnitAttacked {..} |
			LogType::StructureAttacked {..} => {
				events.push(log.clone());
			}
		}
	}
	events
}

pub fn economic_history_events(player_id: usize, gstate: &GameState) -> Vec<Log> {
	let mut events = Vec::with_capacity(gstate.logs.len());
	for log in gstate.logs.iter()
			.filter(|log| log.visible(player_id, &gstate.relations)) {
		match log.val {
			LogType::CivCollapsed {..} |
			LogType::CityDisbanded {..} |
			LogType::CityDestroyed {..} |
			LogType::CityFounded {..} |
			LogType::CivDiscov {..} |
			LogType::WarDeclaration {..} |
			LogType::Rioting {..} |
			LogType::RiotersAttacked {..} |
			LogType::PeaceDeclaration {..} |
			LogType::Debug {..} |
			LogType::CivDestroyed {..} |
			LogType::CityCaptured {..} |
			LogType::NobleHouseJoinedEmpire {..} |
			LogType::ICBMDetonation {..} |
			LogType::PrevailingDoctrineChanged {..} |
			LogType::UnitDestroyed {..} |
			LogType::UnitAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::StructureAttacked {..} => {}
			
			LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} => {
				events.push(log.clone());
			}
		}
	}
	events
}

