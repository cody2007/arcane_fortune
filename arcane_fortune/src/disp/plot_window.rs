use crate::map::{MapData};
use crate::player::*;
use crate::disp_lib::*;
use crate::gcore::Relations;
use super::{float_string, set_player_color, Buttons, addstr_c};
use super::color::{DispChars, CGREEN, ESC_COLOR};
use super::vars::IfaceSettings;
use super::window::{center_txt};
use crate::localization::Localization;

pub enum ColoringType<'o> {
	Players,
	Supplied {
		colors: &'o Vec<CInt>,
		lbls: &'o Vec<String>,
		ign_cur_player_alive: bool
		// if false: stop plotting once the current player is no longer alive
	}
}

// data format: data[owner][time]
// ColoringType::Owners will plot each line in the color of the owners and a legend with the owner nms
// ColoringType::Supplied will plot each line in the color supplied and create a legend with `lbls`
pub fn plot_window_data<T: Into<f32> + Copy>(coloring_type: ColoringType, title_txt: &str, 
		data: &Vec<Vec<T>>, disp_chars: &DispChars, iface_settings: &IfaceSettings, players: &Vec<Player>,
		relations: &Relations, map_data: &MapData, plot_first_player_only: bool, l: &Localization,
		buttons: &mut Buttons, d: &mut DispState) {
	d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);

	let h = iface_settings.screen_sz.h as i32;
	let w = iface_settings.screen_sz.w as i32;
	
	let title_c = Some(COLOR_PAIR(CGREEN));
	
	const ROW_TOP_GAP: i32 = 2; // row to start main plotting
	const ROW_BOTTOM_GAP: i32 = 3 + 1; // row to stop main plotting
	const COL_START: i32 = 8; // column to start main plotting
	
	let h_use = h - ROW_TOP_GAP - ROW_BOTTOM_GAP; // height of plot
	let w_use = w - COL_START - 1; // width of plot
	
	let cur_player = iface_settings.cur_player as usize;
	
	////////// plot title and close txt
	{
		d.clear(); d.mv(0,0);
		center_txt(&format!("{} {}", title_txt, l.vs_time), w, title_c, d);
		d.mv(0,0); buttons.Esc_to_close.print(None, l, d);
		
		d.mv(1,0);
		addstr_c(&format!("<{}>", l.Arrow_keys), ESC_COLOR, d);
		d.addstr(": ");
		d.addstr(&l.Next_prev_plot);
	}
	
	if let Some(first_tseries) = data.first() {
		let n_t_points = first_tseries.len(); // length for first player/time series
		
		if n_t_points < 2 {
			d.mv(h/2, 0);
			center_txt(&format!("In this prehistorical time, no logs exist yet. Check back after {} or so days.", 2*LOG_TURNS), w, None, d);
			return;
		}
		
		//////////////////////////// get min and max value to set plot limits
		let mut min_val;
		let mut max_val;
		
		match coloring_type {
			ColoringType::Players => {
				min_val = data[0][0].into();
				max_val = min_val;
				
				for (owner_id, pstats_data) in data.iter().enumerate() {
					// only plot if discov
					if !relations.discovered(cur_player, owner_id) {continue;}
					
					for val in pstats_data.iter() {
						let v: f32 = (*val).into();
						if v < min_val {min_val = v;}
						if v > max_val {max_val = v;}
					}
				}

			}
			ColoringType::Supplied {..} => { // zoning
				min_val = first_tseries[0].into();
				max_val = min_val;
				
				for tseries in data.iter() {
					for val in tseries.iter() {
						let v: f32 = (*val).into();
						if v < min_val {min_val = v;}
						if v > max_val {max_val = v;}
					}
				}
			}
		}
		
		if min_val == max_val {
			min_val -= 1.;
			max_val += 1.;
		}
		
		/////////////////// plot scales
		{
			////////////// y-axis
			d.mv(ROW_TOP_GAP, 0); // top
			d.addstr(&float_string(max_val));
			
			d.mv(ROW_TOP_GAP + h_use/2, 0); // mid-point
			d.addstr(&float_string(((max_val + min_val)/2.).round()));
			
			d.mv(ROW_TOP_GAP + h_use, 0); // bottom
			d.addstr(&float_string(min_val));
			
			///////////// x-axis
			let max_days_logged = n_t_points * LOG_TURNS;
			
			// if more than 3 years logged, show y labels in years, not months
			let y_unit_txt;
			let y_div = if max_days_logged > (30*12*3) {
				y_unit_txt = &l.years;
				30*12 // scale by year
			}else{
				y_unit_txt = &l.months;
				30 // scale by month
			};
			
			// mid-point
			d.mv(h-2, 0);
			center_txt(&format!("{}", max_days_logged/(y_div*2)), w, None, d);
			
			// left-point
			d.mv(h-2, COL_START);
			d.addstr("0");
			
			// right-point
			let txt = format!("{}", max_days_logged/y_div);
			d.mv(h-2, COL_START + 1 + w_use - txt.len() as i32);
			d.addstr(&txt);
			
			// y-label
			d.mv(h-1, 0);
			center_txt(&format!("{} ({})", l.Time, y_unit_txt), w, title_c, d);
		}
		
		//////////////////// plot data
		{
			// convert data d to row for plotting
			let y = |d: T| -> i32 {
				let d: f32 = d.into();
				let h_use = h_use as f32;
				
				let slope = h_use / (min_val - max_val);
				let c = ROW_TOP_GAP as f32  - slope*max_val;
				
				(slope * d + c).round() as i32
			};
			
			// convert data d to column for plotting (time)
			let x = |d| -> i32 {
				let d = d as f32;
				let w_use = w_use as f32;
				
				let slope = w_use / (n_t_points as f32);
				let c = COL_START as f32;
				
				(slope * d + c).round() as i32
			};
			
			// plot
			match coloring_type {
				ColoringType::Players => {
					for (player_data, player) in data.iter().zip(players.iter()) {
						// only plot if civ discovered
						if !relations.discovered(cur_player, player.id as usize) {continue;}
						match player.ptype {
							PlayerType::Barbarian(_) | PlayerType::Nobility(_) => {continue;}
							PlayerType::Human(_) | PlayerType::Empire(_) => {}
						}
						
						set_player_color(player, true, d);
						// loop over time
						for (x_d, (y_d, _alive)) in player_data.iter()
										.zip(player.stats.alive_log.iter()).enumerate()
										.filter(|(_, (_, &alive))| plot_first_player_only || alive) {
							d.mv(y(*y_d), x(x_d));
							d.addch(disp_chars.land_char as chtype);
						}
						set_player_color(player, false, d);
						
						if plot_first_player_only {break;}
					}
				}
				ColoringType::Supplied {colors, ign_cur_player_alive, ..} => { // ex. zoning, world prevailing doctrines
					let pstats = &players[iface_settings.cur_player as usize].stats;
					// loop over time series
					for (tseries, color) in data.iter().zip(colors) {
						d.attron(COLOR_PAIR(*color));
						// loop over time
						for (x_d, (y_d, alive)) in tseries.iter().zip(pstats.alive_log.iter()).enumerate() {
							if !ign_cur_player_alive && !*alive {break;}
							d.mv(y(*y_d), x(x_d));
							d.addch(disp_chars.land_char as chtype);
						}
						d.attroff(COLOR_PAIR(*color));
					}
				}
			}
		}
		
		const LEGEND_ROW_GAP: i32 = 3;
		const LEGEND_COL_GAP: i32 = 3;
		
		///////////////// plot legend
		if !plot_first_player_only {		
			let mut max_len = l.Legend.len();
			
			match coloring_type {
				ColoringType::Players => {
					let mut row_counter = 0;
					
					for (owner_id, player) in players.iter().enumerate() {
						// only plot if discov
						if !relations.discovered(cur_player, owner_id) {continue;}
						match player.ptype {
							PlayerType::Barbarian(_) | PlayerType::Nobility(_) => {continue;}
							PlayerType::Human(_) | PlayerType::Empire(_) => {}
						}
						
						d.mv(row_counter + ROW_TOP_GAP + LEGEND_ROW_GAP + 2, COL_START + LEGEND_COL_GAP);
						set_player_color(player, true, d);
						d.addch(disp_chars.land_char as chtype);
						set_player_color(player, false, d);
						d.addstr(&format!(" {}", player.personalization.nm));
						
						let owner_len = if !player.stats.alive {
							d.addstr(" (Historic)");
							" (Historic)".len()
						}else {0} + player.personalization.nm.len();
						
						if max_len < owner_len {max_len = owner_len;}
						
						row_counter += 1;
					}
				}
				ColoringType::Supplied {lbls, colors, ..} => {
					for ((i, lbl), color) in lbls.iter().enumerate().zip(colors) {
						d.mv(i as i32 + ROW_TOP_GAP + LEGEND_ROW_GAP + 2, COL_START + LEGEND_COL_GAP);
						d.attron(COLOR_PAIR(*color));
						d.addch(disp_chars.land_char as chtype);
						d.attroff(COLOR_PAIR(*color));
						d.addstr(&format!(" {}", lbl));
						
						if max_len < lbl.len() {max_len = lbl.len();}
					}
				}
			}
			
			let gap = ((2 + max_len) - l.Legend.len()) / 2; // adding 2 to max_len to include border around legend
			d.mv(ROW_TOP_GAP + LEGEND_ROW_GAP, COL_START + LEGEND_COL_GAP + gap as i32);
			d.addstr(&l.Legend);
		
		// show map buffer utilization
		}else{
			d.mv(ROW_TOP_GAP + LEGEND_ROW_GAP, COL_START + LEGEND_COL_GAP);
			d.addstr(&format!("{} {:.1}%", l.Map_buffer_used,
					(100.*map_data.deque_zoom_in.len() as f32) / (map_data.max_zoom_in_buffer_sz as f32)));
		}
		
		///////// plot axis
		{
			// x-axis
			d.mv(h - ROW_BOTTOM_GAP + 1, COL_START);
			d.addch('|');
			for col in 1..w_use {
				if col != ((w_use/2) - 2) {
					d.addch(disp_chars.hline_char as chtype);
				}else{
					d.addch('|');
				}
			}
			d.addch('|');
			
			// y-axis
			d.mv(ROW_TOP_GAP, COL_START - 1);
			d.addch(disp_chars.hline_char as chtype);
			for row in 1..h_use {
				d.mv(ROW_TOP_GAP + row, COL_START-1);
				if row != (h_use/2) {
					d.addch(disp_chars.vline_char as chtype);
				}else{
					d.addch(disp_chars.hline_char as chtype);
				}
			}
			d.mv(ROW_TOP_GAP + h_use, COL_START-1);
			d.addch(disp_chars.hline_char as chtype);
		}
	}else{
		d.mv(h/2, 0);
		center_txt(&l.No_info_to_plot, w, None, d)
	}
}

