use super::*;

pub struct PlotWindowState {pub data: PlotData}

impl PlotWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, gstate: &GameState, map_data: &mut MapData,
			temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		macro_rules! plot_data{($data: ident, $title: expr, $plot_first_player_only: expr) => {
			// collate and convert data to be plotted
			let mut data = Vec::with_capacity(players.len());
			for player in players.iter() {
				let mut d = Vec::with_capacity(player.stats.$data.len());
				for x in player.stats.$data.iter() {
					d.push(*x as f32);
				}
				data.push(d);
			}
			plot_window_data(ColoringType::Players, $title.clone(), &data, dstate, players, &gstate.relations, map_data, $plot_first_player_only);
		};}
		let pstats = &players[dstate.iface_settings.cur_player as usize].stats;
		
		match self.data {
			PlotData::DefensivePower => {plot_data!(defense_power_log, dstate.local.Defensive_power, false);}
			PlotData::OffensivePower => {plot_data!(offense_power_log, dstate.local.Offensive_power, false);}
			PlotData::Population => {plot_data!(population_log, dstate.local.Population, false);}
			PlotData::Unemployed => {plot_data!(unemployed_log, dstate.local.Unemployed, false);}
			PlotData::Gold => {plot_data!(gold_log, dstate.local.Gold, false);}
			PlotData::NetIncome => {plot_data!(net_income_log, dstate.local.Net_Income, false);}
			PlotData::ResearchPerTurn => {plot_data!(research_per_turn_log, dstate.local.Research_Output, false);}
			PlotData::ResearchCompleted => {plot_data!(research_completed_log, dstate.local.Technological_Development, false);}
			PlotData::Happiness => {plot_data!(happiness_log, dstate.local.Happiness, false);}
			PlotData::Crime => {plot_data!(crime_log, dstate.local.Crime, false);}
			PlotData::Pacifism => {plot_data!(pacifism_log, dstate.local.Pacifism_Militarism, false);}
			PlotData::Health => {plot_data!(health_log, dstate.local.Health, false);}
			PlotData::MPD => {plot_data!(mpd_log, dstate.local.Milliseconds_runtime_per_game_day, true);}
			PlotData::DoctrineScienceAxis => {
				// collate and convert data to be plotted
				let mut data = Vec::with_capacity(players.len());
				for player in players.iter() {
					let mut d = Vec::with_capacity(player.stats.doctrinality_log.len());
					// sum across doctrine types
					for x in player.stats.doctrinality_log.iter() {
						d.push(x.iter().sum::<f32>());
					}
					data.push(d);
				}
				plot_window_data(ColoringType::Players, dstate.local.Doctrinality_Methodicalism.clone(), &data, dstate, players, &gstate.relations, map_data, false);
				
			/*PlotData::DoctrineScienceAxis => {
				// collate and convert data to be plotted
				let mut data = Vec::with_capacity(owners.len());
				for pstats in stats.iter() {
					let mut d = Vec::with_capacity(pstats.doctrinality_log.len());
					for x in pstats.doctrinality_log.iter() {
						d.push(x.iter().sum::<f32>()); // sum across doctrine types
					}
					data.push(d);
				}
				plot_window_data(ColoringType::Owners(owners), &dstate.local.Doctrinality_Methodicalism, &data, disp_chars, self, stats, relations, map_data, false, l);*/
			} PlotData::YourPrevailingDoctrines => {
				let mut lbls = Vec::with_capacity(PLAYER_COLORS.len());
				let mut data: Vec<Vec<f32>> = Vec::with_capacity(PLAYER_COLORS.len());
				
				// only plot top doctrines
				if let Some(last_t) = pstats.doctrinality_log.last() { // most recent log for each doctrine type
					// most popular doctrines
					let last_ts = {
						struct LastTDoctrine<'dt> {
							template: &'dt DoctrineTemplate,
							val: f32
						}
						
						// last_ts: entry for each doctrine type
						let mut last_ts = Vec::with_capacity(temps.doctrines.len());
						for (val, template) in last_t.iter().zip(temps.doctrines.iter()) {
							last_ts.push(LastTDoctrine {template, val: *val});
						}
						
						// sort from greatest to least
						last_ts.sort_by(|a, b| b.val.partial_cmp(&a.val).unwrap_or(Ordering::Less));
						
						last_ts
					};
					
					// re-organize data into [doctrinality][time], keeping only the top ones (using no more than PLAYER_COLORS.len())
					for last_t in last_ts.iter().take(PLAYER_COLORS.len()) { // loop over doctrinality
						let doc_ind = last_t.template.id;
						if doc_ind == 0 {continue;} // skip undefined spirituality
						let mut d = Vec::with_capacity(pstats.doctrinality_log.len());
						let mut max_val = 0.;
						for ys in pstats.doctrinality_log.iter() { // loop over time
							let y = ys[doc_ind];
							d.push(y);
							if max_val < y {
								max_val = y;
							}
						}
						
						if max_val == 0. {continue;} // only plot if the data is not all zero
						
						lbls.push(last_t.template.nm[dstate.local.lang_ind].clone());
						data.push(d);
					}
				}
				
				plot_window_data(ColoringType::Supplied {colors: &PLAYER_COLORS.to_vec(), lbls: &lbls, ign_cur_player_alive: false}, dstate.local.Your_empires_prevailing_doctrines.clone(), &data, dstate, players, &gstate.relations, map_data, false);
				
			} PlotData::WorldPrevailingDoctrines => {
				let mut lbls = Vec::with_capacity(PLAYER_COLORS.len());
				let mut data: Vec<Vec<f32>> = Vec::with_capacity(PLAYER_COLORS.len());
				let n_t_points = players[0].stats.doctrinality_log.len();
				
				// only bother if we have any logged data
				if n_t_points != 0 {
					// find top doctrines to plot
					let sum_last_ts = {
						struct LastTDoctrine<'dt> {
							template: &'dt DoctrineTemplate,
							val: f32
						}
						
						let mut sum_last_ts = Vec::with_capacity(temps.doctrines.len());
						// init
						for d in temps.doctrines.iter() {
							sum_last_ts.push(LastTDoctrine {template: d, val: 0.});
						}
						
						// loop over players
						for player in players.iter() {
							// player_last_ts: Vec<f32>, indexed by temps.doctrines
							if let Some(player_last_ts) = player.stats.doctrinality_log.last() {
								// loop over doctrines
								for (player_last_t, sum_last_t) in player_last_ts.iter()
											.zip(sum_last_ts.iter_mut()) {
									sum_last_t.val += *player_last_t;
								}
							}
						}
						
						// sort from greatest to least
						sum_last_ts.sort_by(|a, b| b.val.partial_cmp(&a.val).unwrap_or(Ordering::Less));
						
						sum_last_ts
					};
					
					// re-organize data into data[doctrinality][time], keeping only the top ones (using no more than PLAYER_COLORS.len())
					// from stats[player].doctrinality_log[time][doctrinality]
					{
						// init
						for sum_last_t in sum_last_ts.iter().take(PLAYER_COLORS.len()) {
							lbls.push(sum_last_t.template.nm[dstate.local.lang_ind].clone());
							data.push(vec![0.; n_t_points]);
						}
						
						for player in players.iter() {
							// loop over time
							for (t_ind, ys) in player.stats.doctrinality_log.iter().enumerate() {
								// loop over doctrinality
								for (data_val, sum_last_t) in data.iter_mut().zip(sum_last_ts.iter())
													.take(PLAYER_COLORS.len()) {
									data_val[t_ind] += ys[sum_last_t.template.id];
								}
							}
						}
					}
					
					// remove empty time series
					for doc_ind in (0..data.len()).rev() {
						if data[doc_ind].iter().any(|&val| val != 0.) {continue;}
						lbls.swap_remove(doc_ind);
						data.swap_remove(doc_ind);
					}
					
					//printlnq!("{:#?}", data[0]);
				}
				
				plot_window_data(ColoringType::Supplied {colors: &PLAYER_COLORS.to_vec(), lbls: &lbls, ign_cur_player_alive: true}, dstate.local.World_prevailing_doctrines.clone(), &data, dstate, players, &gstate.relations, map_data, false);
				
			} PlotData::ZoneDemands => {
				let mut colors = Vec::with_capacity(4);
				let mut lbls = Vec::with_capacity(4);
				let mut data: Vec<Vec<f32>> = Vec::with_capacity(4);
				for zone_type_ind in 0_usize..4 {
					let ztype = ZoneType::from(zone_type_ind);
					lbls.push(String::from(ztype.to_str(&dstate.local)));
					colors.push(ztype.to_color());
					
					let mut tseries = Vec::with_capacity(pstats.zone_demand_log.len());
					for tpoint in pstats.zone_demand_log.iter() {
						tseries.push(tpoint[zone_type_ind]);
					}
					data.push(tseries);
				}
				
				//printlnq!("{:#?}", pstats.zone_demand_log);
				
				plot_window_data(ColoringType::Supplied {colors: &colors, lbls: &lbls, ign_cur_player_alive: false}, dstate.local.Zone_Demands.clone(), &data, dstate, players, &gstate.relations, map_data, false);
			} PlotData::PopulationByWealthLevel => {
				let mut data: Vec<Vec<f32>> = Vec::with_capacity(WealthLevel::N as usize); // [wealth_level][time]
				let lbls = vec![dstate.local.Low.clone(), dstate.local.Medium.clone(), dstate.local.High.clone()];
				let colors = vec![CRED, CGREEN4, CGREEN1];
				for wealth_level in 0..WealthLevel::N as usize {
					let mut tseries = Vec::with_capacity(pstats.population_wealth_level_log.len());
					for tpoint in pstats.population_wealth_level_log.iter() {
						tseries.push(tpoint[wealth_level] as f32);
					}
					data.push(tseries);
				}
				
				plot_window_data(ColoringType::Supplied {colors: &colors, lbls: &lbls, ign_cur_player_alive: false}, dstate.local.Population_by_wealth_level.clone(), &data, dstate, players, &gstate.relations, map_data, false);
			} PlotData::N => {panicq!("invalid plot data setting");}
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		match dstate.key_pressed {
			k if k == dstate.kbd.right as i32 || k == KEY_RIGHT => {
				self.data.next();
			} k if k == dstate.kbd.left as i32 || k == KEY_LEFT => {
				self.data.prev();
			} _ => {}
		}
		
		if dstate.buttons.Esc_to_close.activated(dstate.key_pressed, &dstate.mouse_event) {
			return UIModeControl::Closed;
		}
		
		UIModeControl::UnChgd
	}
}

enum_From!{PlotData {DefensivePower, OffensivePower, Population,
	Unemployed, Gold, NetIncome, ResearchPerTurn, 
	ResearchCompleted, Happiness, Crime, DoctrineScienceAxis, 
	YourPrevailingDoctrines, WorldPrevailingDoctrines,
	Pacifism, Health, PopulationByWealthLevel,
	ZoneDemands, MPD}}

impl PlotData {
	pub fn next(&mut self) {
		let mut ind = *self as usize;
		if ind != ((PlotData::N as usize) - 1) {
			ind += 1;
		}else{
			ind = 0;
		}
		
		*self = PlotData::from(ind);
	}
	
	pub fn prev(&mut self) {
		let mut ind = *self as usize;
		if ind != 0 {
			ind -= 1;
		}else{
			ind = (PlotData::N as usize) - 1;
		}
		
		*self = PlotData::from(ind);
	}
}

