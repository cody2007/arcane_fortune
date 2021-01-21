use crate::zones;
use crate::zones::disp::ZonePlotType;
use crate::gcore::hashing::*;
use crate::buildings::Bldg;
use crate::player::*;
use crate::containers::Templates;
use super::*;

pub fn crop_txt(txt: &str, w: usize) -> String {
	if w < 3 {
		".".to_string()
	}else if txt.len() > w {
		let mut txt_new = String::with_capacity(w);
		for i in 0..(w-3) {
			if let Some(c) = txt.chars().nth(i) {
				txt_new.push(c);
			}else {panicq!("i {} w {} len {} {}", i, w, txt.len(), txt);}
		}
		txt_new.push_str("...");
		txt_new
	}else{
		txt.to_string()
	}
}

impl Disp<'_,'_,'_,'_,'_,'_> {
	pub fn print_rside_stats(&mut self, frame_stats: &FrameStats, gstate: &GameState, bldgs: &Vec<Bldg>,
			players: &Vec<Player>, temps: &Templates, exf: &HashedMapEx, map_data: &MapData, map_sz: MapSz) {
		
		let screen_sz = self.state.iface_settings.screen_sz;
		
		let turn_col = (self.state.iface_settings.map_screen_sz.w + 1) as i32;
		for row in 1..self.state.iface_settings.map_screen_sz.h {self.mv(row as i32, turn_col-1); self.addch(' ' as chtype);}
		
		macro_rules! center_txt{($txt: expr) => {
			debug_assertq!(screen_sz.w > turn_col as usize);
			let w = screen_sz.w - turn_col as usize;
			debug_assertq!(w > $txt.len());
			let pad = (w - $txt.len()) / 2;
			for _i in 0..pad {self.addch(' ' as chtype);}
			self.state.txt_list.add_r(&mut self.state.renderer);
			self.state.renderer.addstr($txt);
		};};
		
		macro_rules! ralign_txt{($txt: expr) => {
			let (mut cy, mut cx) = (0,0);
			self.state.renderer.getyx(stdscr(), &mut cy, &mut cx);
			let w = screen_sz.w - cx as usize;
			let txt = crop_txt($txt, w);
			debug_assertq!(w >= txt.len());
			let pad = w - txt.len();
			for _i in 0..pad {self.addch(' ' as chtype);}
			self.state.txt_list.add_r(&mut self.state.renderer);
			self.state.renderer.addstr(&txt);
		};};
		
		let mut roff = TURN_ROW + 5;
		macro_rules! mvclr{() => (self.mv(roff, turn_col); roff += 1; self.state.renderer.clrtoeol());}
		
		let pstats = &players[self.state.iface_settings.cur_player as usize].stats;
		
		{ ////////// display date & next turn keys
			self.mv(TURN_ROW, turn_col);
			self.state.renderer.clrtoeol();
			ralign_txt!(&self.state.local.date_str(gstate.turn));
			
			for j in 1..=6 {
				self.mv(TURN_ROW+j, turn_col);
				self.state.renderer.clrtoeol();
			}
			
			macro_rules! button{($nm: ident) => {
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.buttons.$nm.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);};
			};
			
			match self.state.iface_settings.auto_turn {
				AutoTurn::Off => {
					self.mv(TURN_ROW+1, turn_col);
					if self.state.iface_settings.all_player_pieces_mvd {
						button!(progress_day);
						
						self.mv(TURN_ROW+2, turn_col);
						button!(progress_month);
						
						self.mv(TURN_ROW+3, turn_col);
						button!(finish_all_unit_actions);
					}else{
						self.mv(TURN_ROW+1, turn_col);
						self.attron(COLOR_PAIR(CGRAY));
						center_txt!(&self.state.local.Unmvd_units);
						self.attroff(COLOR_PAIR(CGRAY));
						
						if self.ui_mode.is_none() {
							self.mv(TURN_ROW+2, turn_col);
							button!(progress_day_ign_unmoved_units);
							
							self.mv(TURN_ROW+3, turn_col);
							button!(progress_month);
						}
					}
				} AutoTurn::FinishAllActions => {
					self.mv(TURN_ROW+1, turn_col);
					button!(stop_fin_all_unit_actions);
				} AutoTurn::On => {
				} AutoTurn::N => {panicq!("invalid auto turn setting");}
			}
		}
		
		// budget, demographics, tech, zone demands
		if pstats.alive {
			{ ///////// display budget
				mvclr!(); center_txt!(&self.state.local.Budget); mvclr!();
				
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Taxes); self.addch(':');
				ralign_txt!(&format!("{:.1}", pstats.tax_income(players, &gstate.relations)));
				mvclr!();
				
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Units); self.addch(':');
				ralign_txt!(&format!("{:.1}", -pstats.unit_expenses));
				mvclr!();
				
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Bldgs); self.addch(':');
				ralign_txt!(&format!("{:.1}", -pstats.bldg_expenses));
				mvclr!();
				
				// show line
				for _ in (turn_col as usize)..screen_sz.w {self.addch(self.state.chars.hline_char);}
				
				mvclr!();
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Net); self.addch(':');
				let net_val = pstats.net_income(players, &gstate.relations);
				let net_color = COLOR_PAIR(if net_val >= 0. {CGREEN} else {CRED});
				self.attron(net_color);
				ralign_txt!(&format!("{:.1}", net_val));
				self.attroff(net_color);
				
				mvclr!();
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Gold); self.addch(':');
				ralign_txt!(&float_string(pstats.gold));
				
				if net_val < 0.01 {
					mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&self.state.local.Bnkrpt); self.addch(':');
					mvclr!();
					ralign_txt!(&self.state.local.date_interval_str(-pstats.gold/net_val));
				}
				
				mvclr!();
				mvclr!();
			}
			
			{ ///////// demographics
				mvclr!(); center_txt!(&self.state.local.Demographics); mvclr!();
				self.state.txt_list.add_r(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Rsdnts);
				ralign_txt!(&format!("{}", pstats.population));
				
				if pstats.population != 0 {
					let unemp = ((pstats.population - pstats.employed) as f32) / (pstats.population as f32);
					
					mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&self.state.local.Unemp);
					ralign_txt!(&format!("{}%", (unemp*100.).round()));
				}
				mvclr!();
				mvclr!();
			}
		
			{ /////// tech
				mvclr!(); center_txt!(&self.state.local.Tech);
				mvclr!(); self.state.txt_list.add_r(&mut self.state.renderer); self.state.renderer.addstr(&format!("{}: ", &self.state.local.Rsrch_turn));
				self.state.txt_list.add_r(&mut self.state.renderer); self.state.renderer.addstr(&format!("{}", pstats.research_per_turn));
				mvclr!(); self.state.txt_list.add_r(&mut self.state.renderer); self.state.renderer.addstr(&self.state.local.Rsrching);
				
				if let Some(tech_scheduled) = pstats.techs_scheduled.last() {
					let tech_ind = *tech_scheduled as usize;
					let tech = &temps.techs[tech_ind];
					mvclr!(); ralign_txt!(&tech.nm[self.state.local.lang_ind]);
					//self.mv(roff, turn_col); ///////////////////////////////////// !!!!!! mvclr!();
					mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&self.state.local.Fin); self.addch(':');
					if pstats.research_per_turn == 0 {
						ralign_txt!(&self.state.local.Never);
					}else{
						let turns_req = if let TechProg::Prog(prog) = pstats.techs_progress[tech_ind] {
							((tech.research_req - prog) as f32 / (pstats.research_per_turn as f32)).round()
						}else {0.};
						
						ralign_txt!(&self.state.local.date_interval_str(turns_req));	
					}
				}else{
					ralign_txt!(&self.state.local.None);
					mvclr!();
					self.mv(roff, turn_col); self.state.renderer.clrtoeol(); //mvclr!();
				}
			}
			
			{ ///////// zone demands
				mvclr!();
				mvclr!();
				center_txt!(&self.state.local.Zone_demands);
				
				zones::disp::show_zone_demand_scales(&mut roff, turn_col, ZonePlotType::All {
					pstats: &pstats,
					bldgs,
					owner_id: self.state.iface_settings.cur_player
				}, &mut self.state);
			}
			
			{ //////// local zone demand
				let plot_local_demand = {
					let mut ret_val = false;
					if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
						let coord = self.state.iface_settings.cursor_to_map_ind(map_data);
						
						// only plot if owned by current player and zoned
						if let Some(ex) = exf.get(&coord) {
							if ex.actual.owner_id == Some(self.state.iface_settings.cur_player) {
								if let Some(zone_type) = ex.actual.ret_zone_type() {
									if let Some(zone_ex) = players[self.state.iface_settings.cur_player as usize].zone_exs.get(&return_zone_coord(coord, map_sz)) {
										if self.state.iface_settings.show_all_zone_information {
											self.mv(2,0);
											let zs = &zone_ex.zone_agnostic_stats;
											self.state.renderer.addstr(&format!("Happiness: {} (local), {} (empire)", zs.locally_logged.happiness_sum, pstats.locally_logged.happiness_sum));
											self.mv(3,0);
											self.state.renderer.addstr(&format!("Frm bldgs: {}", zs.gov_bldg_happiness_sum));
											self.mv(4,0);
											self.state.renderer.addstr(&format!("Crime: {}", zs.crime_sum));
											self.mv(5,0);
											self.state.renderer.addstr("Docrines: ");
											for (dsum, dt) in zs.locally_logged.doctrinality_sum.iter().zip(temps.doctrines.iter()) {
												if *dsum == 0. {continue;}
												self.state.renderer.addstr(&format!("{}: {}, ", dt.nm[self.state.local.lang_ind], dsum));
											}
											self.mv(6,0);
											self.state.renderer.addstr(&format!("Pacifism: {}", zs.locally_logged.pacifism_sum));
											self.mv(7,0);
											self.state.renderer.addstr(&format!("Health: {}", zs.health_sum));
											self.mv(8,0);
											self.state.renderer.addstr(&format!("Unemployment: {}", zs.unemployment_sum));
											self.mv(9,0);
											self.state.renderer.addstr(&format!("As of {}", &self.state.local.date_str(zs.turn_computed)));
										}
										
										if let Some(val) = zone_ex.demand_weighted_sum[zone_type as usize] {
											mvclr!();
											mvclr!();
											center_txt!(&self.state.local.Local_demand);
											
											zones::disp::show_zone_demand_scales(&mut roff, turn_col, ZonePlotType::Single(val, zone_type), &mut self.state);
											mvclr!();
											mvclr!();
											ret_val = true;
										}
									}
								}
							}
						}
					}
					ret_val
				};
				
				if !plot_local_demand {
					for _ in 0..4 {mvclr!();}
				}
			}
		}else{mvclr!();mvclr!();}
		
		{ ////// arrow keys mode
			self.state.txt_list.add_r(&mut self.state.renderer);
			self.state.renderer.addstr(&self.state.local.Arrow_keys_mv); mvclr!();
			
			macro_rules! show_toggle{() => {
				self.state.renderer.addstr(" (");
				self.print_key(self.state.kbd.toggle_cursor_mode);
				self.addch(')');
			};};
			
			match self.state.iface_settings.view_mv_mode {
				ViewMvMode::Screen => {
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("* {}", &self.state.local.Screen)); mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("  {}", &self.state.local.Cursor)); show_toggle!(); mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("  {}", &self.state.local.Float));
				}
				ViewMvMode::Cursor => {
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("  {}", &self.state.local.Screen)); mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("* {}", &self.state.local.Cursor)); mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("  {}", &self.state.local.Float)); show_toggle!();
				}
				ViewMvMode::Float => {
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("  {}", &self.state.local.Screen)); show_toggle!(); mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("  {}", &self.state.local.Cursor)); mvclr!();
					self.state.txt_list.add_r(&mut self.state.renderer);
					self.state.renderer.addstr(&format!("* {}", &self.state.local.Float));
				}
	
				ViewMvMode::N => {}
			}
			
			for _ in 0..2 {
				mvclr!();
			}
			self.mv(roff, turn_col); self.state.renderer.clrtoeol();
		}
		
		{ // cursor location
			mvclr!();
			self.state.txt_list.add_r(&mut self.state.renderer);
			self.state.renderer.addstr(&self.state.local.Cursor_location);
			self.mv(roff, turn_col); self.state.renderer.clrtoeol();
			self.state.txt_list.add_r(&mut self.state.renderer);
			self.state.renderer.addstr(&self.state.iface_settings.cursor_to_map_string(map_data));
		}
		
		/////////////// FPS
		if frame_stats.init == false {
			self.mv(screen_sz.h as i32 - 3, turn_col);
			ralign_txt!(&format!("{:.1} {}", frame_stats.dur_mean, &self.state.local.MPD));
			
			self.mv(screen_sz.h as i32 - 2, turn_col);
			ralign_txt!(&format!("{:.1} {}", 1e3/ frame_stats.dur_mean, &self.state.local.DPS));
			
			self.mv(screen_sz.h as i32 - 1, turn_col);
			ralign_txt!(&format!("{:.1} {}", frame_stats.days_per_frame(), &self.state.local.DPF));
		}
	}
}

