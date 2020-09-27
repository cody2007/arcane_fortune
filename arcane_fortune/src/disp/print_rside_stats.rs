use crate::zones;
use crate::zones::disp::ZonePlotType;
use crate::tech::TechTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::gcore::hashing::HashedMapZoneEx;
use crate::buildings::Bldg;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;
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

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn print_rside_stats(&self, frame_stats: &FrameStats, turn: usize, bldgs: &Vec<Bldg>,
			stats: &Vec<Stats>, tech_templates: &Vec<TechTemplate>, doctrine_templates: &Vec<DoctrineTemplate>,
			disp_chars: &DispChars, l: &Localization,
			exf: &HashedMapEx, map_data: &MapData, zone_exs: &HashedMapZoneEx,
			map_sz: MapSz, kbd: &KeyboardMap, buttons: &mut Buttons, d: &mut DispState) {
		
		let screen_sz = self.screen_sz;
		
		let turn_col = (self.map_screen_sz.w + 1) as i32;
		for row in 0..self.map_screen_sz.h {d.mv(row as i32, turn_col-1); d.addch(' ' as chtype);}
		
		macro_rules! center_txt{($txt: expr) => {
			debug_assertq!(self.screen_sz.w > turn_col as usize);
			let w = self.screen_sz.w - turn_col as usize;
			debug_assertq!(w > $txt.len());
			let pad = (w - $txt.len()) / 2;
			for _i in 0..pad {d.addch(' ' as chtype);}
			d.addstr($txt);
		};};
		
		macro_rules! ralign_txt{($txt: expr) => {
			let (mut cy, mut cx) = (0,0);
			d.getyx(stdscr(), &mut cy, &mut cx);
			let w = self.screen_sz.w - cx as usize;
			let txt = crop_txt($txt, w);
			debug_assertq!(w >= txt.len());
			let pad = w - txt.len();
			for _i in 0..pad {d.addch(' ' as chtype);}
			d.addstr(&txt);
		};};
		
		let mut roff = TURN_ROW + 5;
		macro_rules! mvclr{() => (d.mv(roff, turn_col); roff += 1; d.clrtoeol());}
		macro_rules! mvclr3{() => (mvclr!(); mvclr!(); mvclr!(););}
		
		let pstats = &stats[self.cur_player as usize];
		
		///////////////////////////////////////// display date and next turn keys
		{
			d.mv(TURN_ROW, turn_col);
			d.clrtoeol();
			ralign_txt!(&l.date_str(turn));
			
			for j in 1..=6 {
				d.mv(TURN_ROW+j, turn_col);
				d.clrtoeol();
			}
			
			macro_rules! button{($nm: ident) => {buttons.$nm.print(Some(self), l, d);};};
			
			match self.auto_turn {
				AutoTurn::Off => {
					d.mv(TURN_ROW+1, turn_col);
					if self.all_player_pieces_mvd {
						button!(progress_day);
						
						d.mv(TURN_ROW+2, turn_col);
						button!(progress_month);
						
						d.mv(TURN_ROW+3, turn_col);
						button!(finish_all_unit_actions);
					}else{
						d.mv(TURN_ROW+1, turn_col);
						d.attron(COLOR_PAIR(CGRAY));
						center_txt!(&l.Unmvd_units);
						d.attroff(COLOR_PAIR(CGRAY));
						
						if self.ui_mode.is_none() {
							d.mv(TURN_ROW+2, turn_col);
							button!(progress_day_ign_unmoved_units);
							
							d.mv(TURN_ROW+3, turn_col);
							button!(progress_month);
						}
					}
				} AutoTurn::FinishAllActions => {
					d.mv(TURN_ROW+1, turn_col);
					button!(stop_fin_all_unit_actions);
				} AutoTurn::On => {
				} AutoTurn::N => {panicq!("invalid auto turn setting");}
			}
		}
		
		// budget, demographics, tech, zone demands
		if pstats.alive {
			////// display budget
			{
				mvclr!(); center_txt!(&l.Budget); mvclr!();
				
				d.addstr(&l.Taxes); d.addch(':');
				ralign_txt!(&format!("{:.1}", pstats.tax_income));
				mvclr!();
				
				d.addstr(&l.Units); d.addch(':');
				ralign_txt!(&format!("{:.1}", -pstats.unit_expenses));
				mvclr!();
				
				d.addstr(&l.Bldgs); d.addch(':');
				ralign_txt!(&format!("{:.1}", -pstats.bldg_expenses));
				mvclr!();
				
				// show line
				for _ in (turn_col as usize)..screen_sz.w {d.addch(disp_chars.hline_char);}
				
				mvclr!();
				d.addstr(&l.Net); d.addch(':');
				let net_val = pstats.tax_income - pstats.unit_expenses - pstats.bldg_expenses;
				let net_color = COLOR_PAIR(if net_val >= 0. {CGREEN} else {CRED});
				d.attron(net_color);
				ralign_txt!(&format!("{:.1}", net_val));
				d.attroff(net_color);
				
				mvclr!();
				d.addstr(&l.Gold); d.addch(':');
				ralign_txt!(&float_string(pstats.gold));
				
				if net_val < 0.01 {
					mvclr!();
					d.addstr(&l.Bnkrpt); d.addch(':');
					mvclr!();
					ralign_txt!(&l.date_interval_str(-pstats.gold/net_val));
				}
				
				mvclr3!();
			}
			
			/////////// demographics
			{
				mvclr!(); center_txt!(&l.Demographics); mvclr!();
				d.addstr(&l.Rsdnts);
				ralign_txt!(&format!("{}", pstats.population));
				
				if pstats.population != 0 {
					let unemp = ((pstats.population - pstats.employed) as f32) / (pstats.population as f32);

					mvclr!();
					d.addstr(&l.Unemp);
					ralign_txt!(&format!("{}%", (unemp*100.).round()));
				}
				mvclr3!();
			}
		
			///////// tech
			{
				mvclr!(); center_txt!(&l.Tech);
				mvclr!(); d.addstr(&format!("{}: {}", l.Rsrch_turn, pstats.research_per_turn));
				mvclr!(); d.addstr(&l.Rsrching);
				
				if let Some(tech_scheduled) = pstats.techs_scheduled.last() {
					let tech_ind = *tech_scheduled as usize;
					let tech = &tech_templates[tech_ind];
					mvclr!(); ralign_txt!(&tech.nm[l.lang_ind]);
					//d.mv(roff, turn_col); ///////////////////////////////////// !!!!!! mvclr!();
					mvclr!();
					d.addstr(&l.Fin); d.addch(':');
					if pstats.research_per_turn == 0 {
						ralign_txt!(&l.Never);
					}else{
						let turns_req = if let TechProg::Prog(prog) = pstats.techs_progress[tech_ind] {
							((tech.research_req - prog) as f32 / (pstats.research_per_turn as f32)).round()
						}else {0.};
						
						ralign_txt!(&l.date_interval_str(turns_req));	
					}
				}else{
					ralign_txt!(&l.None);
					mvclr!();
					d.mv(roff, turn_col); d.clrtoeol(); //mvclr!();
				}
			}
			
			////// zone demands
			{
				mvclr3!();
				center_txt!(&l.Zone_demands);
				
				zones::disp::show_zone_demand_scales(&mut roff, turn_col, self, disp_chars, ZonePlotType::All {
					pstats: &pstats,
					bldgs,
					owner_id: self.cur_player
				}, d);
			}
		
			////// local zone demand
			{
				let plot_local_demand = {
					let mut ret_val = false;
					if self.zoom_ind == map_data.max_zoom_ind() {
						let coord = self.cursor_to_map_ind(map_data);
						
						// only plot if owned by current player and zoned
						if let Some(ex) = exf.get(&coord) {
							if ex.actual.owner_id == Some(self.cur_player) {
								if let Some(zone_type) = ex.actual.ret_zone_type() {
									if let Some(zone_ex) = zone_exs.get(&return_zone_coord(coord, map_sz)) {
										if self.show_all_zone_information {
											d.mv(2,0);
											let zs = &zone_ex.zone_agnostic_stats;
											d.addstr(&format!("Happiness: {} (local), {} (empire)", zs.locally_logged.happiness_sum, pstats.locally_logged.happiness_sum));
											d.mv(3,0);
											d.addstr(&format!("Frm bldgs: {}", zs.gov_bldg_happiness_sum));
											d.mv(4,0);
											d.addstr(&format!("Crime: {}", zs.crime_sum));
											d.mv(5,0);
											d.addstr("Docrines: ");
											for (dsum, dt) in zs.locally_logged.doctrinality_sum.iter().zip(doctrine_templates.iter()) {
												if *dsum == 0. {continue;}
												d.addstr(&format!("{}: {}, ", dt.nm[l.lang_ind], dsum));
											}
											d.mv(6,0);
											d.addstr(&format!("Pacifism: {}", zs.locally_logged.pacifism_sum));
											d.mv(7,0);
											d.addstr(&format!("Health: {}", zs.health_sum));
											d.mv(8,0);
											d.addstr(&format!("Unemployment: {}", zs.unemployment_sum));
											d.mv(9,0);
											d.addstr(&format!("As of {}", l.date_str(zs.turn_computed)));
										}
										
										if let Some(val) = zone_ex.demand_weighted_sum[zone_type as usize] {
											mvclr!();
											mvclr!();
											center_txt!(&l.Local_demand);
											
											zones::disp::show_zone_demand_scales(&mut roff, turn_col, self, disp_chars, 
													ZonePlotType::Single(val, zone_type), d);
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
		
		///////// arrow keys mode
		{
			d.addstr(&l.Arrow_keys_mv); mvclr!();
			
			macro_rules! show_toggle{() => {
				d.addstr(" (");
				self.print_key(kbd.toggle_cursor_mode, l, d);
				d.addch(')');
			};};
			
			match self.view_mv_mode {
				ViewMvMode::Screen => {
					d.addstr(&format!("* {}", l.Screen)); mvclr!();
					d.addstr(&format!("  {}", l.Cursor)); show_toggle!();
				}
				ViewMvMode::Cursor => {
					d.addstr(&format!("  {}", l.Screen)); show_toggle!(); mvclr!();
					d.addstr(&format!("* {}", l.Cursor));
				}
				ViewMvMode::N => {}
			}
			
			for _ in 0..2 {
				mvclr!();
			}
			d.mv(roff, turn_col); d.clrtoeol();
		}
		
		/////////////// FPS
		if frame_stats.init == false {
			d.mv(screen_sz.h as i32 - 3, turn_col);
			ralign_txt!(&format!("{:.1} {}", frame_stats.dur_mean, l.MPD));
			
			d.mv(screen_sz.h as i32 - 2, turn_col);
			ralign_txt!(&format!("{:.1} {}", 1e3/ frame_stats.dur_mean, l.DPS));
			
			d.mv(screen_sz.h as i32 - 1, turn_col);
			ralign_txt!(&format!("{:.1} {}", frame_stats.days_per_frame(), l.DPF));
		}
	}
}

