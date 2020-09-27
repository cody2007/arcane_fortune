use std::cmp::{min, max};
use super::vars::*;
use super::menus::*;
use super::color::*;

#[macro_use]
pub mod text_entry; pub use text_entry::*;
pub mod list; pub use list::*;
pub mod keys; pub use keys::*;
pub mod war_status; pub use war_status::*;
pub mod encyclopedia_info; pub use encyclopedia_info::*;
pub mod history; pub use history::*;

use std::cmp::Ordering;
use crate::disp_lib::*;
use crate::disp::*;
use crate::doctrine::{DoctrineTemplate, print_doctrine_tree};
use crate::tech::disp::{print_tech_tree};
use crate::tech::{TechTemplate};
use crate::saving::{SmSvType};
use crate::gcore::{Log, print_log, Relations};
use crate::resources::ResourceTemplate;
//use crate::nn::{TxtPrinter, TxtCategory};
use crate::gcore::{GameDifficulties, LogType};
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;

// row bounds to show log
const LOG_START_ROW: i32 = 2;
const LOG_STOP_ROW: i32 = 2;

const MAX_SAVE_AS_W: usize = 78;

const ENCYCLOPEDIA_CATEGORY_NMS: &[&str] = &["Military |u|nits", "B|uildings", "T|echnology", "D|octrines", "|R|esources"];

pub struct ProdOptions<'bt,'ut,'rt,'dt> {
	pub bldgs: Box<[Option<OptionsUI<'bt,'ut,'rt,'dt>>]>,
	worker: OptionsUI<'bt,'ut,'rt,'dt>
}

pub fn init_bldg_prod_windows<'bt,'ut,'rt,'dt>(bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, pstats: &Stats,
		l: &Localization) -> ProdOptions<'bt,'ut,'rt,'dt> {
	////////////////////////////////////// buildings workers produce
	let worker_options = {
		let mut txt_w_none = vec!{"N|one"};
		let mut bldg_template_inds = vec!{None};
		
		for bt in bldg_templates.iter() {
			if bt.barbarian_only {continue;}
			
			if let BldgType::Gov(_) = bt.bldg_type {
				if bt.available(pstats) {
					if let Some(menu_txt) = &bt.menu_txt {
						txt_w_none.push(menu_txt);
					}else{
						txt_w_none.push(&bt.nm[l.lang_ind]);
					}
					bldg_template_inds.push(Some(bt));
				}
			}
		}
		let mut worker_options = OptionsUI {options: Vec::with_capacity(txt_w_none.len()), max_strlen: 0};
		register_shortcuts(&txt_w_none, &mut worker_options);
		
		// set argument options to associate unit_template index with menu entry
		for (opt, bldg_template_ind) in worker_options.options.iter_mut().zip(bldg_template_inds.iter().cloned()) {
			opt.arg = ArgOptionUI::BldgTemplate(bldg_template_ind);
		}
		
		worker_options
	};
	
	///////////
	let mut production_options = ProdOptions {
			bldgs: vec!{None; bldg_templates.len()}.into_boxed_slice(),
			worker: worker_options
	};
	
	/////////////////////////////////// building unit productions
	for bt in bldg_templates.iter() {
		if let Some(units_producable_txt) = &bt.units_producable_txt {
			let units_producable = bt.units_producable.as_ref().unwrap();
			
			let mut txt_w_none = vec!{"N|one"};
			let mut unit_template_inds = vec!{None};
			for (txt, unit_producable) in units_producable_txt.iter().zip(units_producable.iter()) {
				if pstats.unit_producable(unit_producable) {
					txt_w_none.push(txt);
					unit_template_inds.push(Some(*unit_producable));
					//printlnq!("{} {}", bt.nm, unit_producable.nm);
				}
			}
			
			production_options.bldgs[bt.id as usize] = Some(OptionsUI {
								options: Vec::with_capacity(txt_w_none.len()),
								max_strlen: 0});
			
			register_shortcuts(&txt_w_none, production_options.bldgs[bt.id as usize].as_mut().unwrap());
			
			// set argument options to associate unit_template index with menu entry
			for (opt, unit_template_ind) in production_options.bldgs[bt.id as usize].as_mut().unwrap().options.iter_mut().
									zip(unit_template_inds.iter().cloned()) {
				opt.arg = ArgOptionUI::UnitTemplate(unit_template_ind);
			}
			
			assertq!(units_producable_txt.len() == bt.units_producable.as_ref().unwrap().len(), 
					"Building \"{}\"'s list of units_producable and units_producable_txt are not the same size ({}, and {}). The configuration file needs to be altered.",
					bt.nm[0], units_producable_txt.len(), bt.units_producable.as_ref().unwrap().len());
		}
	}
	
	production_options
}

pub fn center_txt(txt: &str, w: i32, color: Option<chtype>, d: &mut DispState) {
	let g = (w as usize - txt.len())/2;
	let mut sp = String::new();
	for _ in 0..g {sp.push(' ');}
	
	d.addstr(&sp);
	
	if let Some(c) = color {d.attron(c);}
	d.addstr(&txt);
	if let Some(c) = color {d.attroff(c);}
}

fn print_window(window_sz: ScreenSz, screen_sz: ScreenSz, disp_chars: &DispChars, d: &mut DispState) -> Coord {
	debug_assertq!(screen_sz.h >= window_sz.h);
	debug_assertq!(screen_sz.w >= window_sz.w);
	
	let row_initial = ((screen_sz.h - window_sz.h)/2) as i32;
	let col = ((screen_sz.w - window_sz.w)/2) as i32;
	
	let h = window_sz.h as i32;
	let w = window_sz.w as i32;
	
	// print top line
	{
		d.mv(row_initial, col);
		d.addch(disp_chars.ulcorner_char);
		for _ in 2..w {d.addch(disp_chars.hline_char);}
		d.addch(disp_chars.urcorner_char);
	}
	
	// print intermediate lines
	for row_off in 1..(h-1) {
		d.mv(row_initial + row_off, col);
		d.addch(disp_chars.vline_char);
		for _ in 2..w {d.addch(' ');}
		d.addch(disp_chars.vline_char);
	}
	
	// print bottom line
	{
		d.mv(row_initial + h-1, col);
		d.addch(disp_chars.llcorner_char);
		for _ in 2..w {d.addch(disp_chars.hline_char);}
		d.addch(disp_chars.lrcorner_char);
	}
	
	Coord {y: row_initial as isize, x: col as isize}
}

// prints windows
impl <'f,'bt,'ut,'rt,'dt>IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	pub fn print_windows(&mut self, map_data: &mut MapData, exf: &HashedMapEx, units: &Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, production_options: &ProdOptions, 
			disp_chars: &DispChars, unit_templates: &'ut Vec<UnitTemplate<'rt>>, bldg_templates: &'bt Vec<BldgTemplate>,
			tech_templates: &Vec<TechTemplate>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &Vec<DoctrineTemplate>,
			owners: &Vec<Owner>, stats: &Vec<Stats>, relations: &Relations, game_difficulties: &GameDifficulties, 
			logs: &Vec<Log>, turn: usize, kbd: &KeyboardMap, l: &Localization, buttons: &mut Buttons, d: &mut DispState) {
		let h = self.screen_sz.h as i32;
		let w = self.screen_sz.w as i32;
		
		const TITLE_COLOR: i32 = CGREEN;
		let title_c = Some(COLOR_PAIR(TITLE_COLOR));
		
		/////////// production by workers or buildings
		if let UIMode::ProdListWindow {mode} = self.ui_mode {
			debug_assertq!(self.zoom_ind == map_data.max_zoom_ind());
			
			let w = 29;
			let pstats = &stats[self.cur_player as usize];
			
			////////////////////// worker producing bldg
			if self.unit_inds_frm_sel(pstats, units, map_data, exf) != None {
				print_list_window(mode, &l.Select_production, production_options.worker.clone(), self, disp_chars, Some(w), None, 0, None, l, buttons, d);
				
				// print details for selected bldg
				if let ArgOptionUI::BldgTemplate(Some(bt)) = production_options.worker.options[mode].arg {
					let template_ind = bldg_templates.iter().position(|r| r == bt).unwrap();
					self.show_exemplar_info(template_ind, EncyclopediaCategory::Bldg, OffsetOrPos::Offset(w), None, OffsetOrPos::Offset(mode-4), InfoLevel::Abbrev, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
				// zeroth entry only should be none
				}else if mode != 0 {
					panicq!("could not find building template {}", mode);
				}
			////////////////////// building producing unit
			}else if let Some(bldg_ind) = self.bldg_ind_frm_cursor(bldgs, map_data, exf) {
				let b = &bldgs[bldg_ind];
				
				// look-up menu listings for the selected bldg:
				if let Some(options) = &production_options.bldgs[b.template.id as usize] {
					
					print_list_window(mode, &l.Select_production, options.clone(), self, disp_chars, None, None, 0, None, l, buttons, d);
					
					// print details for selected bldg
					if let ArgOptionUI::UnitTemplate(Some(ut)) = options.options[mode].arg {
						let template_ind = unit_templates.iter().position(|r| r == ut).unwrap();
						self.show_exemplar_info(template_ind, EncyclopediaCategory::Unit, OffsetOrPos::Offset(w), None, OffsetOrPos::Offset(mode+4), InfoLevel::Abbrev, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
					// zeroth entry only should be none
					}else if mode != 0 {
						panicq!("could not find unit template {}", mode);
					}
					
				}else {panicq!("could not find building production options");}
	
			}else{
				end_window(self, d);
				return
			}//panicq!("window active but no printing available");}
		
		/////////// production by buildings
		}else if let UIMode::CurrentBldgProd {mode} = self.ui_mode {
			debug_assertq!(self.zoom_ind == map_data.max_zoom_ind());
			
			let w = 29;
			let pstats = &stats[self.cur_player as usize];
			
			if let Some(bldg_ind) = self.bldg_ind_frm_cursor(bldgs, map_data, exf) {
				let options = bldg_prod_list(&bldgs[bldg_ind], l); 
				print_list_window(mode, &l.Select_an_item_to_remove, options.clone(), self, disp_chars, None, None, 0, None, l, buttons, d);
				
				// print details for selected bldg
				if let Some(option) = options.options.get(mode) {
					if let ArgOptionUI::UnitTemplate(Some(ut)) = option.arg {
						let template_ind = unit_templates.iter().position(|r| r == ut).unwrap();
						self.show_exemplar_info(template_ind, EncyclopediaCategory::Unit, OffsetOrPos::Offset(w+3), None, OffsetOrPos::Offset(mode+4), InfoLevel::Abbrev, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
					// zeroth entry only should be none
					}else if mode != 0 {
						panicq!("could not find unit template {}", mode);
					}
				}
			}else{
				end_window(self, d);
				return
			}
		
		//////////// tech window
		}else if let UIMode::TechWindow {sel, sel_mv, tree_offsets, prompt_tech, ..} = &mut self.ui_mode {
			let pstats = &stats[self.cur_player as usize];
			print_tech_tree(unit_templates, bldg_templates, tech_templates, resource_templates,
					disp_chars, pstats, sel, sel_mv, tree_offsets, self.screen_sz, *prompt_tech, kbd, l, buttons, d);
		
		//////////// doctrine tree window
		}else if let UIMode::DoctrineWindow {sel, sel_mv, tree_offsets, ..} = &mut self.ui_mode {
			let pstats = &stats[self.cur_player as usize];
			print_doctrine_tree(doctrine_templates, bldg_templates, disp_chars, pstats, sel, sel_mv, tree_offsets, self.screen_sz, kbd, l, buttons, d);

		////////////// tech discovered window (when new tech discovered)
		}else if let UIMode::TechDiscoveredWindow {tech_ind, ..} = self.ui_mode {
			self.show_tech_discovered(&tech_templates[tech_ind], unit_templates, bldg_templates, resource_templates, disp_chars, l, buttons, d);
			
		/////////// plotting (full screen)
		}else if let UIMode::PlotWindow {data} = &self.ui_mode {
			macro_rules! plot_data{($data: ident, $title: expr, $plot_first_player_only: expr) => {
				// collate and convert data to be plotted
				let mut data = Vec::with_capacity(owners.len());
				for pstats in stats.iter() {
					let mut d = Vec::with_capacity(pstats.$data.len());
					for x in pstats.$data.iter() {
						d.push(*x as f32);
					}
					data.push(d);
				}
				plot_window_data(ColoringType::Owners(owners), $title, &data, disp_chars, self, stats, relations, map_data, $plot_first_player_only, l, buttons, d);
			};};
			
			match data {
				PlotData::DefensivePower => {plot_data!(defense_power_log, &l.Defensive_power, false);}
				PlotData::OffensivePower => {plot_data!(offense_power_log, &l.Offensive_power, false);}
				PlotData::Population => {plot_data!(population_log, &l.Population, false);}
				PlotData::Unemployed => {plot_data!(unemployed_log, &l.Unemployed, false);}
				PlotData::Gold => {plot_data!(gold_log, &l.Gold, false);}
				PlotData::NetIncome => {plot_data!(net_income_log, &l.Net_Income, false);}
				PlotData::ResearchPerTurn => {plot_data!(research_per_turn_log, &l.Research_Output, false);}
				PlotData::ResearchCompleted => {plot_data!(research_completed_log, &l.Technological_Development, false);}
				PlotData::Happiness => {plot_data!(happiness_log, &l.Happiness, false);}
				PlotData::Crime => {plot_data!(crime_log, &l.Crime, false);}
				PlotData::Pacifism => {plot_data!(pacifism_log, &l.Pacifism_Militarism, false);}
				PlotData::Health => {plot_data!(health_log, &l.Health, false);}
				PlotData::MPD => {plot_data!(mpd_log, &l.Milliseconds_runtime_per_game_day, true);}
				PlotData::DoctrineScienceAxis => {
					// collate and convert data to be plotted
					let mut data = Vec::with_capacity(owners.len());
					for pstats in stats.iter() {
						let mut d = Vec::with_capacity(pstats.doctrinality_log.len());
						// sum across doctrine types
						for x in pstats.doctrinality_log.iter() {
							d.push(x.iter().sum::<f32>());
						}
						data.push(d);
					}
					plot_window_data(ColoringType::Owners(owners), &l.Doctrinality_Methodicalism, &data, disp_chars, self, stats, relations, map_data, false, l, buttons, d);
					
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
					plot_window_data(ColoringType::Owners(owners), &l.Doctrinality_Methodicalism, &data, disp_chars, self, stats, relations, map_data, false, l);*/
				} PlotData::YourPrevailingDoctrines => {
					let pstats = &stats[self.cur_player as usize];
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
							let mut last_ts = Vec::with_capacity(doctrine_templates.len());
							for (val, template) in last_t.iter().zip(doctrine_templates.iter()) {
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
							
							lbls.push(last_t.template.nm[l.lang_ind].clone());
							data.push(d);
						}
					}
					
					plot_window_data(ColoringType::Supplied {colors: &PLAYER_COLORS.to_vec(), lbls: &lbls, ign_cur_player_alive: false}, &l.Your_empires_prevailing_doctrines, &data, disp_chars, self, stats, relations, map_data, false, l, buttons, d);
					
				} PlotData::WorldPrevailingDoctrines => {
					let mut lbls = Vec::with_capacity(PLAYER_COLORS.len());
					let mut data: Vec<Vec<f32>> = Vec::with_capacity(PLAYER_COLORS.len());
					let n_t_points = stats[0].doctrinality_log.len();
					
					// only bother if we have any logged data
					if n_t_points != 0 {
						// find top doctrines to plot
						let sum_last_ts = {
							struct LastTDoctrine<'dt> {
								template: &'dt DoctrineTemplate,
								val: f32
							}
							
							let mut sum_last_ts = Vec::with_capacity(doctrine_templates.len());
							// init
							for d in doctrine_templates.iter() {
								sum_last_ts.push(LastTDoctrine {template: d, val: 0.});
							}
							
							// loop over players
							for pstats in stats.iter() {
								// player_last_ts: Vec<f32>, indexed by doctrine_templates
								if let Some(player_last_ts) = pstats.doctrinality_log.last() {
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
								lbls.push(sum_last_t.template.nm[l.lang_ind].clone());
								data.push(vec![0.; n_t_points]);
							}
							
							for pstats in stats.iter() {
								// loop over time
								for (t_ind, ys) in pstats.doctrinality_log.iter().enumerate() {
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
					
					plot_window_data(ColoringType::Supplied {colors: &PLAYER_COLORS.to_vec(), lbls: &lbls, ign_cur_player_alive: true}, &l.World_prevailing_doctrines, &data, disp_chars, self, stats, relations, map_data, false, l, buttons, d);
					
				} PlotData::ZoneDemands => {
					let pstats = &stats[self.cur_player as usize];
					let mut colors = Vec::with_capacity(4);
					let mut lbls = Vec::with_capacity(4);
					let mut data: Vec<Vec<f32>> = Vec::with_capacity(4);
					for zone_type_ind in 0_usize..4 {
						let zone_type = ZoneType::from(zone_type_ind);
						lbls.push(String::from(zone_type.to_str()));
						colors.push(zone_type.to_color());
						
						let mut tseries = Vec::with_capacity(pstats.zone_demand_log.len());
						for tpoint in pstats.zone_demand_log.iter() {
							tseries.push(tpoint[zone_type_ind]);
						}
						data.push(tseries);
					}
					
					//printlnq!("{:#?}", pstats.zone_demand_log);
					
					plot_window_data(ColoringType::Supplied {colors: &colors, lbls: &lbls, ign_cur_player_alive: false}, &l.Zone_Demands, &data, disp_chars, self, stats, relations, map_data, false, l, buttons, d);
				} PlotData::N => {panicq!("invalid plot data setting");}
			}
		
		/////////////// save as
		}else if let UIMode::SaveAsWindow {save_nm, curs_col, ..} = &self.ui_mode {
			let w = min(MAX_SAVE_AS_W, self.screen_sz.w);
			let h = 7;
			
			let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			d.mv(y,x);
			center_txt(&l.Save_as, w, title_c, d);
			
			// print file name
			d.mv(y+2,x+1);
			d.addstr(&save_nm);
						
			// instructions
			{
				let instructions_w = format!("{}   {}", buttons.Esc_to_close.print_txt(l), buttons.Save.print_txt(l)).len() as i32;
				let gap = ((w - instructions_w)/2) as i32;
				d.mv(y + 4, x - 1 + gap);
				buttons.Esc_to_close.print(None, l, d); d.addstr("   ");
				buttons.Save.print(None, l, d);
			}
			
			// mv to cursor location
			d.mv(y + 2, x + 1 + *curs_col as i32);
		
		/////////////// go to coordinate
		}else if let UIMode::GoToCoordinateWindow {coordinate, curs_col} = &self.ui_mode {
			let w = min(30, self.screen_sz.w);
			let h = 7;
			
			let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			d.mv(y,x);
			center_txt(&l.Go_to_coordinate, w, title_c, d);
			
			// print current location
			d.mv(y+2,x+1);
			d.addstr(&coordinate);
						
			// instructions
			{
				let instructions_w = "<Esc>: Cancel  <Enter>: Go".len() as i32;
				let gap = ((w - instructions_w)/2) as i32;
				d.mv(y + 4, x - 1 + gap);
				buttons.Esc_to_close.print(None, l, d);
				d.addstr("  ");
				d.attron(COLOR_PAIR(ESC_COLOR));
				d.addstr(&l.Enter_key);
				d.attroff(COLOR_PAIR(ESC_COLOR));
				d.addstr(&format!(": {}", l.Go));
			}
			
			// mv to cursor location
			d.mv(y + 2, x + 1 + *curs_col as i32);
		
		/////////////// get text
		}else if let UIMode::GetTextWindow {txt, curs_col, txt_type} = &self.ui_mode {
			let w = min(40, self.screen_sz.w);
			let h = 7;
			
			let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			d.mv(y,x);
			match txt_type {
				TxtType::BrigadeNm => {center_txt(&l.Choose_a_name_for_the_brigade, w, title_c, d);}
				TxtType::SectorNm => {center_txt(&l.Choose_a_name_for_the_sector, w, title_c, d);}
			}
			
			// print entered txt
			d.mv(y+2,x+1);
			d.addstr(&txt);
						
			// instructions
			{
				let instructions_w = format!("{}  {}", buttons.Esc_to_close.print_txt(l), buttons.Confirm.print_txt(l)).len() as i32;
				let gap = ((w - instructions_w)/2) as i32;
				d.mv(y + 4, x - 1 + gap);
				buttons.Esc_to_close.print(None, l, d); d.addstr("  ");
				buttons.Confirm.print(None, l, d);
			}
			
			// mv to cursor location
			d.mv(y + 2, x + 1 + *curs_col as i32);
			
		///////////// save auto freq
		}else if let UIMode::SaveAutoFreqWindow {freq, curs_col, ..} = &self.ui_mode {
			let instructions_w = format!("{}   {}", buttons.Esc_to_close.print_txt(l), buttons.Confirm.print_txt(l)).len() as i32;
			let w = max(l.years_0_disables.len(), instructions_w as usize) + 4;
			let h = 9;
			
			let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			d.mv(y,x);
			center_txt(&l.Save_auto_freq, w, title_c, d);
			
			d.mv(y+2,x+1);
			d.addstr(&l.Auto_save_the_game_every);
			
			// print file name
			d.mv(y+3,x+1);
			d.addstr(&freq);
			
			d.mv(y+4,x+1);
			d.addstr(&l.years_0_disables);
			
			// instructions
			{
				let gap = ((w - instructions_w)/2) as i32;
				d.mv(y + 6, x + gap);
				buttons.Esc_to_close.print(None, l, d); d.addstr("  ");
				buttons.Confirm.print(None, l, d);
			}
			
			// mv to cursor location
			d.mv(y + 3, x + 1 + *curs_col as i32);
			
		////////// open game
		}else if let UIMode::OpenWindow {save_files, mode, ..} = &self.ui_mode {
			// max file name length
			let mut max_txt_len = if let Some(max_f) = save_files.iter().max_by_key(|f| f.nm.len()) {max_f.nm.len()} else {0};
			max_txt_len += "     XXX XX-XX-XXXX  XX:XX XX (GMT)".len() + 2;
			
			let mut w = min(max_txt_len+1, self.screen_sz.w);
			let h = min(self.screen_sz.h, save_files.len() + 2 + 3 + 2);
			
			let n_rows_plot = min(save_files.len(), self.screen_sz.h - 4 - 2) as i32;
			if n_rows_plot < save_files.len() as i32 { // add additional width if scrolling
				w += 1;
			}
			
			let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			d.mv(y,x);
			center_txt("Open game", w as i32, title_c, d);
			
			// first file to print
			let first_ln = if *mode >= (n_rows_plot as usize) {
				*mode - n_rows_plot as usize + 1
			}else{0};
			
			// print files
			if save_files.len() != 0 {
				for (i, s) in save_files.iter().enumerate() {
					if i < first_ln {continue;}
					if i >= (first_ln + n_rows_plot as usize) {break;}
					
					let row_start = y + i as i32 + 2 - first_ln as i32;
					//if row_start >= (self.screen_sz.h - 4) as i32 {break;}
					
					// modified time start column
					let mut mod_col_start = (x-1)+  w as i32  - 2 - 3 - (s.modified.len() as i32);
					if n_rows_plot < save_files.len() as i32 { // move over to left for scrollbar
						mod_col_start -= 1;
					}
					
					// file name
					d.mv(row_start, x + 1);
					let button_start = cursor_pos(d);
					
					if i == *mode {d.attron(A_REVERSE());}
					for i in 0..min(mod_col_start-1-x, s.nm.chars().count() as i32) {
						d.addch(s.nm.chars().nth(i as usize).unwrap());
					}
					
					// print ... if file name too long
					if mod_col_start < (x+1+s.nm.chars().count() as i32) {
						d.mv(row_start, mod_col_start - 3);
						d.addstr("...");
					}else{
						for _ in 0..((mod_col_start-1-x) - (s.nm.chars().count() as i32)) {
							d.addch(' ');
						}
					}
					
					// modified
					d.mv(row_start, mod_col_start);
					d.addstr("   ");
					d.addstr(&s.modified);
					buttons.add(button_start, i, d);
					if i == *mode {d.attroff(A_REVERSE());}
				}
			// no files
			}else{
				let row_start = y + 2 - first_ln as i32;
				d.mv(row_start, x + 1);
				d.addstr("No files");
			}
			
			// instructions
			{
				let instructions_w = buttons.Esc_to_close.print_txt(l).len() + 
					"   ".len() + buttons.Open.print_txt(l).len();
				let gap = ((w - instructions_w)/2) as i32;
				
				let row_start = if save_files.len() as i32 > n_rows_plot {
					1 + (self.screen_sz.h - 4) as i32
				}else{
					y + save_files.len() as i32 + 3
				};
				d.mv(row_start, x - 1 + gap);
				buttons.Esc_to_close.print(None, l, d);
				d.addstr("   ");
				buttons.Open.print(None, l, d);
			}
			
			//////// print scroll bars
			if save_files.len() > n_rows_plot as usize {
				let h = h as i32;
				let w = x as i32 + w as i32 - 2;
				let scroll_track_h = n_rows_plot;
				let frac_covered = n_rows_plot as f32 / save_files.len() as f32;
				let scroll_bar_h = ((scroll_track_h as f32) * frac_covered).round() as i32;
				debug_assertq!(frac_covered <= 1.);
				
				let frac_at_numer = if *mode < n_rows_plot as usize {
					0
				} else {
					first_ln + 2//+ n_rows_plot as usize
				};
				
				let frac_at = frac_at_numer as f32 / save_files.len() as f32;
				let scroll_bar_start = ((scroll_track_h as f32) * frac_at).round() as i32;
				
				d.mv(LOG_START_ROW, w-1);
				d.attron(COLOR_PAIR(CLOGO));
				d.addch(disp_chars.hline_char);
				for row in 0..scroll_bar_h-1 {
					d.mv(row + 1 + scroll_bar_start + LOG_START_ROW, w-1);
					d.addch(disp_chars.vline_char);
					//d.addch('#' as chtype);
				}
				d.mv(h-LOG_STOP_ROW, w-1);
				d.addch(disp_chars.hline_char);
				d.attroff(COLOR_PAIR(CLOGO));
			}
		
		////////// world & battle history (full screen)
		}else if let UIMode::WorldHistoryWindow {..} |
				UIMode::BattleHistoryWindow {..} |
				UIMode::EconomicHistoryWindow {..} = &self.ui_mode {
			d.clear();
			let events = match self.ui_mode {
				UIMode::WorldHistoryWindow {..} => {
					center_txt("World History", w, title_c, d);
					world_history_events(self.cur_player as usize, relations, logs)
				}
				UIMode::BattleHistoryWindow {..} => {
					center_txt("Battle History", w, title_c, d);
					battle_history_events(self.cur_player as usize, relations, logs)
				}
				UIMode::EconomicHistoryWindow {..} => {
					center_txt("Economic History", w, title_c, d);
					economic_history_events(self.cur_player as usize, relations, logs)
				}

				_ => {panicq!("unhandled UI mode");}
			};
			
			if let UIMode::WorldHistoryWindow {ref mut scroll_first_line} |
					UIMode::BattleHistoryWindow {ref mut scroll_first_line} |
					UIMode::EconomicHistoryWindow {ref mut scroll_first_line} = self.ui_mode {
				d.mv(0,0); buttons.Esc_to_close.print(None, l, d);
				
				// no data
				if events.len() == 0 {
					d.mv(h/2, 0);
					center_txt(&format!("The world is still pre-historical and no logs exists yet. Check back later."), w, None, d);
					return;
				}
				
				if h <= (LOG_START_ROW + LOG_STOP_ROW) {return;} // screen not tall enough
				
				let n_rows_plot = h - LOG_START_ROW - LOG_STOP_ROW;
				
				// check to make sure scroll line is within range (ex. if screen sz chgs)
				if events.len() <= *scroll_first_line || (events.len() - *scroll_first_line) < n_rows_plot as usize {
					if events.len() > n_rows_plot as usize {
						*scroll_first_line = events.len() - n_rows_plot as usize;
					}else{
						*scroll_first_line = 0;
					}
				}
				
				// plot log
				{
					let mut row_counter = 0;
					
					for log in events[*scroll_first_line..].iter() {
						if row_counter >= n_rows_plot {break;}
						if !log.visible(self.cur_player as usize, relations) {continue;}
						
						d.mv(LOG_START_ROW + row_counter, 0);
						l.print_date_log(log.turn, d);
						print_log(&log.val, true, owners, doctrine_templates, l, d);
						
						row_counter += 1;
					}
				}
				
				/////// print scroll instructions
				{
					/*let center_txt = |txt: &str, w| {
						let g = (w as usize - txt.len())/2;
						let mut sp = String::new();
						for _ in 0..g {sp.push(' ');}
						d.attron(title_c.unwrap());
						d.addstr(&format!("{}{}", sp, txt));
						d.attroff(title_c.unwrap());
					};*/
					
					let row = h - LOG_START_ROW;
					let instructions_width = "<Arrow keys>: Scroll".len() as i32;
					let col = (w - instructions_width)/2;
					/*macro_rules! nl{() => {d.mv(row, col); row += 1;};
					($last:expr) => {d.mv(row,col);};};*/
					
					// instructions
					/*nl!();
					center_txt("Keys", instructions_width);
					nl!(); nl!();*/
					d.mv(row,col);
					d.attron(COLOR_PAIR(ESC_COLOR));
					d.addstr("<Arrow keys>");
					d.attroff(COLOR_PAIR(ESC_COLOR));
					d.addstr(": Scroll");
				}
				
				//////// print scroll bars
				if *scroll_first_line != 0 || (events.len() - *scroll_first_line) > n_rows_plot as usize {
					let scroll_track_h = n_rows_plot;
					let frac_covered = n_rows_plot as f32 / events.len() as f32;
					let scroll_bar_h = max(((scroll_track_h as f32) * frac_covered).round() as i32, 1);
					debug_assertq!(frac_covered <= 1.);
					
					let frac_at = *scroll_first_line as f32 / events.len() as f32;
					let scroll_bar_start = ((scroll_track_h as f32) * frac_at).round() as i32;
					
					d.mv(LOG_START_ROW, w-1);
					d.attron(COLOR_PAIR(CLOGO));
					d.addch(disp_chars.hline_char);
					for row in 0..=scroll_bar_h-1 {
						d.mv(row + 1 + scroll_bar_start + LOG_START_ROW, w-1);
						d.addch(disp_chars.vline_char);
						//d.addch('#' as chtype);
					}
					d.mv(h-LOG_STOP_ROW, w-1);
					d.addch(disp_chars.hline_char);
					d.attroff(COLOR_PAIR(CLOGO));
				}
			}else{panicq!("unhandled UI mode");}
		
		////////// encyclopedia
		}else if let UIMode::EncyclopediaWindow {state} = &self.ui_mode {
			match state {
				///////////////////////////////////////// first page shown when window is first created, prompting for category type
				EncyclopediaState::CategorySelection {mode} => {
					let mut category_options = OptionsUI {options: Vec::with_capacity(ENCYCLOPEDIA_CATEGORY_NMS.len()), max_strlen: 0};
					
					register_shortcuts(ENCYCLOPEDIA_CATEGORY_NMS, &mut category_options);
					print_list_window(*mode, &l.What_would_you_like_to_learn_about, category_options, self, disp_chars, Some(l.What_would_you_like_to_learn_about.len()+4), None, 0, None, l, buttons, d);
				
				} EncyclopediaState::ExemplarSelection {selection_mode, category, mode} => {
					//////////////////////////////////////////////////// show names of unit templates etc
					if *selection_mode {
						// set exemplar names
						macro_rules! set_exemplar_nms{($templates: ident, $txt: expr) => {
							let mut exemplar_nms = Vec::with_capacity($templates.len());
							for template in $templates.iter() {exemplar_nms.push(template.nm[l.lang_ind].as_str());}
							
							let mut exemplar_options = OptionsUI {options: Vec::with_capacity(exemplar_nms.len()), max_strlen: 0};
							
							register_shortcuts(exemplar_nms.as_slice(), &mut exemplar_options);
							print_list_window(*mode, &format!("Select a {}:", $txt), exemplar_options, self, disp_chars, Some(30), None, 0, None, l, buttons, d);
						};};
						
						match category {
							EncyclopediaCategory::Unit => {set_exemplar_nms!(unit_templates, "unit");}
							EncyclopediaCategory::Bldg => {
								let exemplar_options = encyclopedia_bldg_list(bldg_templates, l);
								print_list_window(*mode, &l.Select_a_building, exemplar_options, self, disp_chars, Some(30), None, 0, None, l, buttons, d);
							}
							EncyclopediaCategory::Tech => {set_exemplar_nms!(tech_templates, "technology");}
							EncyclopediaCategory::Doctrine => {set_exemplar_nms!(doctrine_templates, "doctrine");}
							EncyclopediaCategory::Resource => {set_exemplar_nms!(resource_templates, "resource");}
						}
						
					///////////////////////////////////// show info for the selected unit
					}else{
						let exemplar_ind = if *category == EncyclopediaCategory::Bldg {
							let exemplar_options = encyclopedia_bldg_list(bldg_templates, l);
							if let ArgOptionUI::Ind(Some(ind)) = exemplar_options.options[*mode].arg {
								ind
							}else{panicq!("invalid option argument");}
						}else{
							*mode
						};
						self.show_exemplar_info(exemplar_ind, *category, OffsetOrPos::Offset(0), None, OffsetOrPos::Offset(0), InfoLevel::Full {buttons}, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, &stats[self.cur_player as usize], disp_chars, l, d);
					}
				} // either showing exemplar selection (unit, bldg, tech) or showing specific unit information
			} // match state (EncyclopediaState)
		
		/////////// prevailing doctrine changed
		}else if let UIMode::PrevailingDoctrineChangedWindow = &self.ui_mode {
			let doc = stats[self.cur_player as usize].doctrine_template;
			let title = l.Adopted_doctrine.replace("[]", &doc.nm[l.lang_ind]);
			let lens = vec![title.len(), l.doctrine_changed_line1.len(), l.doctrine_changed_line2.len()];
			let max_len = *lens.iter().max().unwrap();
			let window_sz = ScreenSz {h: 8, w: max_len + 4, sz: 0};
			let pos = print_window(window_sz, self.screen_sz, disp_chars, d);
			
			// title
			{
				d.mv(pos.y as i32 + 1, pos.x as i32 + ((window_sz.w - title.len())/2) as i32);
				d.attron(COLOR_PAIR(TITLE_COLOR));
				d.addstr(&title);
				d.attroff(COLOR_PAIR(TITLE_COLOR));
			}
			
			d.mv(pos.y as i32 + 3, pos.x as i32 + 2);
			d.addstr(&l.doctrine_changed_line1);
			
			d.mv(pos.y as i32 + 4, pos.x as i32 + 2);
			d.addstr(&l.doctrine_changed_line2);
			
			// esc to close window
			{
				let button = &mut buttons.Esc_to_close;
				d.mv(pos.y as i32 + 6, pos.x as i32 + ((window_sz.w - button.print_txt(l).len()) / 2) as i32);
				button.print(None, l, d);
			}
			
		///////////// show owned units
		}else if let UIMode::UnitsWindow {mode} = self.ui_mode {
			let mut w = 0;
			let mut label_txt_opt = None;
			let map_sz = *map_data.map_szs.last().unwrap();
			let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
			
			let pstats = &stats[self.cur_player as usize];
			let owned_units = owned_unit_list(units, self.cur_player, cursor_coord, pstats, &mut w, &mut label_txt_opt, map_sz, l);
			
			let top_right = print_list_window(mode, &l.Select_battalion, owned_units.clone(), self, disp_chars, Some(w), label_txt_opt, 0, None, l, buttons, d).1;
			
			// show info box
			if owned_units.options.len() > 0 {
				let pstats = &stats[self.cur_player as usize];
				if let ArgOptionUI::UnitInd(unit_ind) = owned_units.options[mode].arg {
					self.show_exemplar_info(units[unit_ind].template.id as usize, EncyclopediaCategory::Unit, OffsetOrPos::Pos(top_right.x as usize - 1), None, OffsetOrPos::Pos(top_right.y as usize + mode + 4), InfoLevel::AbbrevNoCostNoProdTime, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
				}else{panicq!("invalid UI setting");}
			}
		
		///////////// show brigades
		}else if let UIMode::BrigadesWindow {mode, brigade_action} = &self.ui_mode {
			let mut w = 0;
			let mut label_txt_opt = None;
			
			let pstats = &stats[self.cur_player as usize];
			
			let (entries, txt, n_gap_lines, has_buildable_actions) = match brigade_action {
				BrigadeAction::Join {..} | BrigadeAction::ViewBrigades => {
					(brigades_list(pstats, &mut w, &mut label_txt_opt, l),
					 l.Select_brigade.clone(), 0, false)
				} BrigadeAction::ViewBrigadeUnits {brigade_nm} => {
					let map_sz = *map_data.map_szs.last().unwrap();
					let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
					
					let has_buildable_actions = if let BrigadeAction::ViewBrigadeUnits {brigade_nm} = brigade_action {
							pstats.brigade_frm_nm(brigade_nm).has_buildable_actions(units)
					}else{panicq!("invalid brigade action");};
						
					let n_gap_lines = 4 + if has_buildable_actions {4} else {0};
					
					(brigade_unit_list(brigade_nm, pstats, units, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l),
					 String::new(), n_gap_lines, has_buildable_actions)
				}
			};
			
			let (top_left, top_right) = print_list_window(*mode, &txt, entries.clone(), self, disp_chars, Some(w), label_txt_opt, n_gap_lines, None, l, buttons, d);
			
			// print instructions & show info box
			if entries.options.len() > 0 {
				match entries.options[*mode].arg {
					////////// case where brigade_action = Join or ViewBrigades
					ArgOptionUI::BrigadeInd(brigade_ind) => {
						if let Some(brigade) = pstats.brigades.get(brigade_ind) {
							self.show_brigade_units(brigade, Coord {x: top_right.x - 1, y: top_right.y + *mode as isize + 5}, units, disp_chars, l, d);
						}
					///////// case where brigade_action = ViewBrigadeUnits
					} ArgOptionUI::UnitInd(unit_ind) => {
						const OPTION_STR: &str = "   * ";
						
						let mut roff = top_left.y as i32 + 2;
						macro_rules! mvl{() => {d.mv(roff, top_left.x as i32 + 2); roff += 1;};
							($fin: expr) => {d.mv(roff, top_left.x as i32 + 2);};};
						
						mvl!();
						
						let brigade_nm = if let BrigadeAction::ViewBrigadeUnits {brigade_nm} = brigade_action {brigade_nm} else {panicq!("invalid brigade action");};
						
						// title
						{
							let title = l.The_NM_Brigade.replace("[]", brigade_nm);
							for _ in 0..((w-title.len() - 4)/2) {d.addch(' ');}
							
							addstr_c(&title, TITLE_COLOR, d);
							mvl!();
						}
						
						// Build list:   X actions (/: view)
						// Automatic repair behavior: Repair walls in Sector ABC
						if has_buildable_actions {
							mvl!();
							d.addstr(&l.Build_list);
							
							let brigade = pstats.brigade_frm_nm(brigade_nm);
							
							let n_brigade_units = brigade.build_list.len();
							
							let action_txt = if n_brigade_units != 1 {&l.actions} else {&l.action};
							let txt = format!("{} {} ", n_brigade_units, action_txt);
							
							let txt_added = l.Build_list.len() + txt.len() + buttons.view_brigade.print_txt(l).len() + 4 + 2;
							if txt_added <= w {
								let gap = w - txt_added;
								for _ in 0..gap {d.addch(' ');}
							}
							d.addstr(&txt); d.addch('(');
							buttons.view_brigade.print(None, l, d);
							d.addch(')'); mvl!();
							
							// auto-repair behavior
							d.addstr(&l.Automatic_repair_behavior);
							let (txt, button) = if let Some(sector_nm) = &brigade.repair_sector_walls {
								(l.Repair_damaged_walls.replace("[]", sector_nm), &mut buttons.clear_brigade_repair)
							}else{(l.None.clone(), &mut buttons.change_brigade_repair)};
							
							let txt_added = l.Automatic_repair_behavior.len() + txt.len() + button.print_txt(l).len() + 5;
							if txt_added <= w {
								let gap = w - txt_added;
								for _ in 0..gap{d.addch(' ');}
							}
							d.addstr(&txt); d.addch(' ');
							button.print(None, l, d);
							mvl!();
						}
						
						{ // Assign an action to: 
							mvl!(); d.addstr(&l.Assign_an_action_to);
							
							mvl!(); d.addstr(OPTION_STR);
							buttons.assign_action_to_all_in_brigade.print(None, l, d);
							
							if has_buildable_actions {
								mvl!(); d.addstr(OPTION_STR);
								buttons.add_action_to_brigade_build_list.print(None, l, d);
							}
							
							mvl!(1); d.addstr(OPTION_STR);
							d.addstr(&l.an_individual_battalion);
						}
						
						// print infobox
						self.show_exemplar_info(units[unit_ind].template.id as usize, EncyclopediaCategory::Unit, OffsetOrPos::Pos(top_right.x as usize - 1), None, OffsetOrPos::Pos(n_gap_lines + top_right.y as usize + mode + 4), InfoLevel::AbbrevNoCostNoProdTime, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
					} _ => {panicq!("invalid UI setting");}
				}
			}
		
		//////////// brigade build list
		}else if let UIMode::BrigadeBuildList {mode, brigade_nm} = &self.ui_mode {
			let w = 1 + l.Or_select_an_action_and_press.len() + 5 + l.to_remove.len() + 1 + 4 + 4;
			let label_txt_opt = None;
			let pstats = &stats[self.cur_player as usize];
			let entries = brigade_build_list(brigade_nm, pstats, l);
			
			let n_gap_lines = 1;
			let top_left = print_list_window(*mode, &l.Select_battalion, entries, self, disp_chars, Some(w), label_txt_opt, n_gap_lines, None, l, buttons, d).0;
			
			let mut roff = top_left.y as i32 + 2;
			macro_rules! mvl{() => {d.mv(roff, top_left.x as i32 + 2); roff += 1;};
				($fin: expr) => {d.mv(roff, top_left.x as i32 + 2);};};
			
			// Press 'a' to add an action
			// Or select an action and press <Delete> to remove:			
			
			mvl!();
			buttons.Press_to_add_action_to_brigade_build_list.print(None, l, d);
			
			mvl!(1);
			d.addstr(&l.Or_select_an_action_and_press); d.addch(' ');
			print_key_always_active(KEY_DC, l, d);
			d.addch(' '); d.addstr(&l.to_remove);
			
		///////////// show sectors
		}else if let UIMode::SectorsWindow {mode, ..} = self.ui_mode {
			let mut w = 0;
			let mut label_txt_opt = None;
			let map_sz = *map_data.map_szs.last().unwrap();
			let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
			
			let sectors = sector_list(&stats[self.cur_player as usize], cursor_coord, &mut w, &mut label_txt_opt, map_sz, l);
			
			print_list_window(mode, &l.Select_map_sector, sectors, self, disp_chars, Some(w), label_txt_opt, 0, None, l, buttons, d);
		
		///////////// sector automation: get sector name (step 1)
		}else if let UIMode::CreateSectorAutomation {sector_nm: None, mode, ..} = self.ui_mode {
			let pstats = &stats[self.cur_player as usize];
			
			// alert player that sector should be created
			if pstats.sectors.len() == 0 {
				let txt = &l.No_map_sectors_found;
				let w = txt.len() + 4;
				let w_pos = print_window(ScreenSz{w, h: 2+4, sz:0}, self.screen_sz, disp_chars, d);
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let mut row = 0;
				macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
							     ($final: expr) => {d.mv(row + y, x);}};
				
				mvl!();
				buttons.Esc_to_close.print(None, l, d);
				
				mvl!(); mvl!(1);
				d.addstr(&txt);
				
			// ask player which sector to use
			}else{
				let mut w = 0;
				let mut label_txt_opt = None;
				let map_sz = *map_data.map_szs.last().unwrap();
				let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
				
				let sectors = sector_list(pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l);
				
				let txt = &l.In_which_sector_do_you_want_to_automate;
				w = txt.len() + 4;
				print_list_window(mode, txt, sectors, self, disp_chars, Some(w), label_txt_opt, 0, None, l, buttons, d);
			}
		
		//////////// sector automation: get action to perform when unit enters sector (step 2)
		}else if let UIMode::CreateSectorAutomation {sector_nm: Some(_), unit_enter_action: None, mode, ..} = self.ui_mode {
			let unit_enter_actions = [l.Assault_desc.as_str(), l.Defense_desc.as_str(), l.Report_desc.as_str()];
			let mut category_options = OptionsUI {options: Vec::with_capacity(unit_enter_actions.len()), max_strlen: 0};
			
			register_shortcuts(&unit_enter_actions, &mut category_options);
			let txt = &l.Select_unit_enter_action;
			print_list_window(mode, txt, category_options, self, disp_chars, Some(txt.len()+4), None, 0, None, l, buttons, d);
			
		//////////// sector automation: what to do when the unit is idle (step 3)
		}else if let UIMode::CreateSectorAutomation {sector_nm: Some(_), unit_enter_action: Some(_), idle_action: None, mode, ..} = self.ui_mode {
			let idle_actions = [l.Sentry_desc.as_str(), l.Patrol_desc.as_str()];
			let mut idle_options = OptionsUI {options: Vec::with_capacity(idle_actions.len()), max_strlen: 0};
			
			register_shortcuts(&idle_actions, &mut idle_options);
			let txt = &l.When_not_engaged_what_action;
			print_list_window(mode, txt, idle_options, self, disp_chars, Some(txt.len()+4), None, 0, None, l, buttons, d);
			
		//////////////////// sector automation: get distance to respond to threats (step 4)
		}else if let UIMode::CreateSectorAutomation {sector_nm: Some(_), unit_enter_action: Some(_), idle_action: Some(_), txt, curs_col, ..} = &self.ui_mode {
			let title_txt = &l.At_what_distance;
			let w = min(title_txt.len() + 4, self.screen_sz.w);
			let h = 7;
			
			let w_pos = print_window(ScreenSz{w,h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			d.mv(y,x);
			center_txt(title_txt, w, title_c, d);
			
			// print entered txt
			d.mv(y+2,x+1);
			d.addstr(&txt);
						
			// instructions
			{
				let instructions_w = format!("{}  {}", buttons.Esc_to_close.print_txt(l), buttons.Confirm.print_txt(l)).len() as i32;
				let gap = ((w - instructions_w)/2) as i32;
				d.mv(y + 4, x - 1 + gap);
				buttons.Esc_to_close.print(None, l, d); d.addstr("  ");
				buttons.Confirm.print(None, l, d);
			}
			
			// mv to cursor location
			d.mv(y + 2, x + 1 + *curs_col as i32);
		
		///////////// show owned improvement or military buildings
		}else if let UIMode::ImprovementBldgsWindow {mode} | UIMode::MilitaryBldgsWindow {mode} = self.ui_mode {
			let mut w = 0;
			let mut label_txt_opt = None;
			let map_sz = *map_data.map_szs.last().unwrap();
			let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
			
			let owned_bldgs = match self.ui_mode {
				UIMode::ImprovementBldgsWindow {..} => {owned_improvement_bldgs_list(bldgs, doctrine_templates, self.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l)}
				UIMode::MilitaryBldgsWindow {..} => {owned_military_bldgs_list(bldgs, self.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, l)}
				_ => {panicq!("ui mode match condition not met");}
			};
			
			let top_right = print_list_window(mode, &l.Select_a_building, owned_bldgs.clone(), self, disp_chars, Some(w), label_txt_opt, 0, None, l, buttons, d).1;
			
			// show info box
			if owned_bldgs.options.len() > 0 {
				let pstats = &stats[self.cur_player as usize];
				if let ArgOptionUI::BldgInd(bldg_ind) = owned_bldgs.options[mode].arg {
					self.show_exemplar_info(bldgs[bldg_ind].template.id as usize, EncyclopediaCategory::Bldg, OffsetOrPos::Pos(top_right.x as usize - 1), None, OffsetOrPos::Pos(top_right.y as usize + mode + 4), InfoLevel::AbbrevNoCostNoProdTime, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
				}else{panicq!("invalid UI setting");}
			}
		
		/////////////// end game
		}else if let UIMode::EndGameWindow  = self.ui_mode {
			let owner = &owners[self.cur_player as usize];
			let window_w = (max(format!("Watch as {} becomes an arcane footnote", owner.nm).len(),
						 "civilizations, you can now explore the remaining world.".len())
						+ 2 + 2) as i32; // 2 for the |, 2 for the spaces
			let window_h = 9;
			
			let mut row = (self.screen_sz.h as i32 - window_h)/2;
			let col = (self.screen_sz.w as i32 - window_w)/2;
			
			let mut y = 0;
			let mut x = 0;
			
			macro_rules! bln{() => {
				d.mv(row, col); row += 1;
				d.addch(disp_chars.vline_char);
				for _ in 0..(window_w-2) {d.addch(' ');}
				d.addch(disp_chars.vline_char);
			};};
			
			macro_rules! pl{() => {
				d.mv(row, col); row += 1;
				d.addch(disp_chars.vline_char);
				d.addch(' ');
			};};
			
			macro_rules! pr{() => {d.addch(' '); d.addch(disp_chars.vline_char);};};
			
			// clear to end of line
			macro_rules! clr{() => {
				d.getyx(stdscr(), &mut y, &mut x);
				for _ in x..(col + window_w-2) {d.addch(' ');}
				pr!();
			};};
			
			/////// top ln
			{
				d.mv(row, col); row += 1;
				d.addch(disp_chars.ulcorner_char);
				for _ in 0..(window_w-2) {d.addch(disp_chars.hline_char);}
				d.addch(disp_chars.urcorner_char);
			}
			
			//////// print title: {} has been destroyed!
			{
				pl!();
				let txt_len = format!("{} has been destroyed!", owner.nm).len() as i32;
				for _ in 0..((window_w - txt_len)/2) {d.addch(' ');}
				print_civ_nm(owner, d);
				d.attron(COLOR_PAIR(CYELLOW));
				d.addstr(" has been destroyed!");
				d.attroff(COLOR_PAIR(CYELLOW));
				clr!();
			}
			
			bln!();
			
			pl!();
			d.addstr("While you no longer have a place here in the realm of");
			clr!();
			
			pl!();
			d.addstr("civilizations, you can now explore the remaining world.");
			clr!();
			
			bln!();
			
			pl!();
			d.addstr("Watch as ");
			print_civ_nm(owner, d);
			d.addstr(" becomes an arcane footnote");
			clr!();
			
			pl!();
			d.addstr("in the historical abyss.");
			clr!();
			
			bln!();
			
			//////// esc
			{
				pl!();
				let button = &mut buttons.Esc_to_close;
				for _ in 0..((window_w - button.print_txt(l).len() as i32)/2) {d.addch(' ');}
				button.print(None, l, d);
				clr!();
			}
			
			////// bottom ln
			{
				d.mv(row, col);
				d.addch(disp_chars.llcorner_char);
				for _ in 0..(window_w-2) {d.addch(disp_chars.hline_char);}
				d.addch(disp_chars.lrcorner_char);
			}
		
		/////////////// initial game window
		}else if let UIMode::InitialGameWindow = self.ui_mode {
			const INTRO_TXT: &[&str] = &[
			"For as long as anyone can remember, the people of your tribe have subsisted on the land.",
			"Living and dying by what it has to offer. Recently, some members of the tribe have taken",
			"greater control of nature and have found that they can tend to, support and cultivate it,",
			"leading to small farms.",
			" ",
			"All creation is not without havoc. Your people have suffered increasingly",
			"organized attacks by local barbarian tribes and have taken it upon themselves",
			"to organize and mount a resistance to expand beyond and counter the forces of mankind",
			"and nature itself.",
			" ",
			"So, you have taken it upon yourself to lead this movement. What you make of it is",
			"entirely up to you."];
			
			let owner = &owners[self.cur_player as usize];
			let mut window_w = format!("{} has been founded!", owner.nm).len() as i32;
			
			// determine max width of window
			{
				for txt in INTRO_TXT.iter() {
					if window_w < (txt.len() as i32) {window_w = txt.len() as i32;}
				}
				
				window_w += 2 + 2; // 2 for the |, 2 for the spaces
			}
			
			let window_h = INTRO_TXT.len() as i32 + 2 + 2 + 2; // 2 for the top and bottom lines, 2 for txt, 2 blank lns
			
			let mut row = (self.screen_sz.h as i32 - window_h)/2;
			let col = (self.screen_sz.w as i32 - window_w)/2;
			
			let mut y = 0;
			let mut x = 0;
			
			macro_rules! bln{() => {
				d.mv(row, col); row += 1;
				d.addch(disp_chars.vline_char);
				for _ in 0..(window_w-2) {d.addch(' ');}
				d.addch(disp_chars.vline_char);
			};};
			
			macro_rules! pl{() => {
				d.mv(row, col); row += 1;
				d.addch(disp_chars.vline_char);
				d.addch(' ');
			};};
			
			macro_rules! pr{() => {d.addch(' '); d.addch(disp_chars.vline_char);};};
			
			// clear to end of line
			macro_rules! clr{() => {
				d.getyx(stdscr(), &mut y, &mut x);
				for _ in x..(col + window_w-2) {d.addch(' ');}
				pr!();
			};};
			
			/////// top ln
			{
				d.mv(row, col); row += 1;
				d.addch(disp_chars.ulcorner_char);
				for _ in 0..(window_w-2) {d.addch(disp_chars.hline_char);}
				d.addch(disp_chars.urcorner_char);
			}
			
			//////// print title: {} has been founded!
			{
				pl!();
				let txt_len = format!("{} has been founded!", owner.nm).len() as i32;
				for _ in 0..((window_w - txt_len)/2) {d.addch(' ');}
				print_civ_nm(owner, d);
				d.attron(COLOR_PAIR(CYELLOW));
				d.addstr(" has been founded!");
				d.attroff(COLOR_PAIR(CYELLOW));
				clr!();
			}
			
			bln!();
			
			////////// print intro text		
			for txt in INTRO_TXT.iter() {
				pl!();
				d.addstr(txt);
				clr!();
			}
						
			bln!();
			
			//////// esc
			{
				pl!();
				let button = &mut buttons.Esc_to_close;
				for _ in 0..((window_w - button.print_txt(l).len() as i32)/2) {d.addch(' ');}
				button.print(None, l, d);
				clr!();
			}
			
			////// bottom ln
			{
				d.mv(row, col);
				d.addch(disp_chars.llcorner_char);
				for _ in 0..(window_w-2) {d.addch(disp_chars.hline_char);}
				d.addch(disp_chars.lrcorner_char);
			}
		
		//////////// war status window
		}else if let UIMode::WarStatusWindow = self.ui_mode {
			self.show_war_status_window(relations, stats, owners, title_c, disp_chars, l, buttons, d);
			
		/////////////// about
		}else if let UIMode::AboutWindow = self.ui_mode {
			show_version_status(self.screen_sz, disp_chars, buttons, l, d);
		
		////////////// show owned cities
		}else if let UIMode::CitiesWindow {mode} = self.ui_mode {
			let mut w = 0;
			let mut label_txt_opt = None;
			let map_sz = *map_data.map_szs.last().unwrap();
			let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
			
			let owned_cities = owned_city_list(bldgs, self.cur_player, cursor_coord, &mut w, &mut label_txt_opt, map_sz, logs, l);
			
			print_list_window(mode, &l.Select_city, owned_cities, self, disp_chars, Some(w), label_txt_opt, 0, None, l, buttons, d);
		
		/////////////// contact embassy
		}else if let UIMode::ContactEmbassyWindow {state} = &mut self.ui_mode {
			match state {
				EmbassyState::CivSelection{mode} => {
					let contacted_civs = contacted_civilizations_list(relations, stats, owners, self.cur_player, turn);
					print_list_window(*mode, &l.Select_civilization, contacted_civs, self, disp_chars, None, None, 0, Some(owners), l, buttons, d);
				} EmbassyState::DialogSelection{mode, owner_id, ref mut quote_printer} => {
					let o = &owners[*owner_id];
					let w = 70;
					let w_pos = print_window(ScreenSz{w, h: 5+2+3+2+2, sz:0}, self.screen_sz, disp_chars, d);
					
					let y = w_pos.y as i32 + 1;
					let x = w_pos.x as i32 + 2;
					
					let w = (w - 2) as i32;
					
					let mut row = 0;
					macro_rules! mvl{() => {d.mv(row + y, x); row += 1;}};
					
					///////////// title -- country name
					{
						mvl!();
						let txt_len = format!("The {} Embassy", o.nm).len();
						let g = (w as usize - txt_len) / 2;
						let mut sp = String::with_capacity(g);
						for _ in 0..g {sp.push(' ');}
						
						d.addstr(&sp);
						d.addstr("The ");
						
						set_player_color(o, true, d);
						d.addstr(&o.nm);
						set_player_color(o, false, d);
						
						d.addstr(" Embassy");
						
						mvl!();
					}
					
					//////////// ruler
					{
						mvl!();
						d.addstr(&format!("Their leader, {} {}, ", o.ruler_nm.first, o.ruler_nm.last));
						relations.print_mood_action(*owner_id, self.cur_player as usize, owners, d);
						d.addstr(" says:");
						
						mvl!();
						d.addstr(&format!("   \"{}\"", quote_printer.gen()));
						
						mvl!();mvl!();
						center_txt("How do you respond?", w, None, d);
					}
					
					/////////// options
					{
						mvl!();mvl!();
						// todo: add Trade technology
						center_txt(if *mode == 0 {"* Threaten *"} else {"Threaten"}, w, None, d);
						mvl!();
						//center_txt("Demand tribute", w, None);
						//mvl!();
						
						// at war
						if relations.at_war(*owner_id, self.cur_player as usize) {
							center_txt(if *mode == 1 {"* Suggest a peace treaty *"} else {"Suggest a peace treaty"}, w, None, d);
						// peace treaty in effect
						}else if let Some(expiry) = relations.peace_treaty_turns_remaining(*owner_id, self.cur_player as usize, turn) {
							center_txt(&format!("(Peace treaty expires in {})", l.date_interval_str(expiry as f32)), w, Some(COLOR_PAIR(CGREEN3)), d);
						// at peace but no peace treaty in effect
						}else{
							center_txt(if *mode == 1 {"* Declare war *"} else {"Declare war"}, w, None, d);
						}
						
						mvl!();
					}
					
					// print key instructions
					{
						let txt_len = "<Left arrow> go back    <Enter> Perform action".len();
						let g = (w as usize - txt_len) / 2;
						let mut sp = String::with_capacity(g);
						for _ in 0..g {sp.push(' ');}
						
						let col2 = x + g as i32 + "<Left arrow> go back".len() as i32 + 3;
						
						mvl!();
						d.addstr(&sp);
						buttons.Esc_to_close.print(None, l, d);
						
						d.mv(row-1 + y, col2);
						d.attron(COLOR_PAIR(ESC_COLOR));
						d.addstr("<Up/Down arrows>");
						d.attroff(COLOR_PAIR(ESC_COLOR));
						d.addstr(" select"); 
						
						mvl!();
						d.addstr(&sp);
						d.attron(COLOR_PAIR(ESC_COLOR));
						d.addstr("<Left arrow>");
						d.attroff(COLOR_PAIR(ESC_COLOR));
						d.addstr(" go back"); 
						
						d.mv(row-1 + y, col2);
						d.attron(COLOR_PAIR(ESC_COLOR));
						d.addstr("<Enter>");
						d.attroff(COLOR_PAIR(ESC_COLOR));
						d.addstr(" perform action"); 
					}
				} EmbassyState::Threaten {owner_id, ref mut quote_printer} |
				  EmbassyState::DeclareWar {owner_id, ref mut quote_printer} |
				  EmbassyState::DeclarePeace {owner_id, ref mut quote_printer} |
				  EmbassyState::DeclaredWarOn {owner_id, ref mut quote_printer} => {
					let quote_txt = quote_printer.gen();
					let owner_id = *owner_id;
				  	
					let o = &owners[owner_id];
					let w = 70;
					let w_pos = print_window(ScreenSz{w, h: 5+4, sz:0}, self.screen_sz, disp_chars, d);
					
					let y = w_pos.y as i32 + 1;
					let x = w_pos.x as i32 + 2;
					
					let w = (w - 2) as i32;
					
					let mut row = 0;
					macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
							     ($final: expr) => {d.mv(row + y, x);}};
					
					///////////// title -- country name
					{
						// AI declares war on the human player
						if let EmbassyState::DeclaredWarOn {..} = state {
							let txt_len = format!("The {} civilization has declared war on you!", o.nm).len();
							let g = (w as usize - txt_len) / 2;
							let mut sp = String::with_capacity(g);
							for _ in 0..g {sp.push(' ');}
							
							mvl!();
							d.addstr(&sp);
							d.addstr("The ");
							
							set_player_color(o, true, d);
							d.addstr(&o.nm);
							set_player_color(o, false, d);
							
							d.addstr(" civilization has declared war on you!");
						}else{
							let title_txt = match state {
								EmbassyState::Threaten {..} => {"You have threatened "}
								EmbassyState::DeclareWar {..} => {"You have declared war on "}
								EmbassyState::DeclarePeace {..} => {"You have made peace with "}
								_ => {panicq!{"match condition shouldn't be possible"}}
							};
							
							let txt_len = format!("{}{}!", title_txt, o.nm).len();
							let g = (w as usize - txt_len) / 2;
							let mut sp = String::with_capacity(g);
							for _ in 0..g {sp.push(' ');}
							
							mvl!();
							d.addstr(&sp);
							d.addstr(title_txt);
							
							set_player_color(o, true, d);
							d.addstr(&o.nm);
							set_player_color(o, false, d);
						}
						
						d.addstr("!");
						mvl!();
					}
					
					//////////// ruler
					{
						mvl!();
						d.addstr(&format!("{} {}, ", o.ruler_nm.first, o.ruler_nm.last));
						relations.print_mood_action(owner_id, self.cur_player as usize, owners, d);
						d.addstr(if let EmbassyState::DeclaredWarOn {..} = state {" says:"} else {" responds:"});
						
						mvl!();
						d.addstr(&format!("   \"{}\"", quote_txt));
						
						mvl!();mvl!();
					}
					
					///////// esc to close
					{
						let txt_len = "<Esc> ".len() + l.to_close.len();
						let g = (w as usize - txt_len) / 2;
						let mut sp = String::with_capacity(g);
						for _ in 0..g {sp.push(' ');}
						
						mvl!(true);
						d.addstr(&sp);
						buttons.Esc_to_close.print(None, l, d);
					}
				} EmbassyState::DeclarePeaceTreaty {owner_id, ref mut quote_printer, gold_offering, curs_col, treaty_rejected} => {
					let o = &owners[*owner_id];
					let w = 80;
					let mut h = 5+5+4;
					if *treaty_rejected {h += 2};
					let w_pos = print_window(ScreenSz{w, h, sz:0}, self.screen_sz, disp_chars, d);
					
					let y = w_pos.y as i32 + 1;
					let x = w_pos.x as i32 + 2;
					
					let w = (w - 2) as i32;
					
					let mut row = 0;
					macro_rules! mvl{() => {d.mv(row + y, x); row += 1;}};
					
					///////////// title -- country name
					{
						const TITLE_TXT: &str = "Peace treaty with ";
						
						let txt_len = format!("{}{}!", TITLE_TXT, o.nm).len();
						let g = (w as usize - txt_len) / 2;
						let mut sp = String::with_capacity(g);
						for _ in 0..g {sp.push(' ');}
						
						mvl!();
						d.addstr(&sp);
						d.addstr(TITLE_TXT);
						
						set_player_color(o, true, d);
						d.addstr(&o.nm);
						set_player_color(o, false, d);
						
						mvl!();
					}
					
					const INPUT_OFFSET_COL: i32 = 4;
					
					////////// print gold input and quote
					{
						mvl!();
						d.addstr("How much gold would you like to offer in the treaty?");
						
						// gold offering
						d.mv(y+INPUT_OFFSET_COL,x);
						d.addstr(&gold_offering);
						
						mvl!();mvl!();mvl!();mvl!();
						d.addstr(&format!("( Negative values indicate {} will instead pay and not you;", o.nm));
						mvl!();
						d.addstr(&format!("  Peace treaties cannot be terminated for {} years after signing. )", relations.config.peace_treaty_min_years));
						
						
						// print notice that treaty rejected and give a quote from the ruler
						if *treaty_rejected {
							mvl!();mvl!();
							d.addstr(&format!("{} {} rejects your offer, ", o.ruler_nm.first, o.ruler_nm.last));
							relations.print_mood_action(*owner_id, self.cur_player as usize, owners, d);
							d.addstr(" saying:");
							
							mvl!();
							d.addstr(&format!("   \"{}\"", quote_printer.gen()));
						}
					}
					
					// instructions
					{
						let instructions_w = "<Esc>: Cancel   <Enter>: Propose treaty".len() as i32;
						let gap = ((w - instructions_w)/2) as i32;
						d.mv(y + row + 1, x - 1 + gap);
						buttons.Esc_to_close.print(None, l, d); d.addstr("   ");
						d.attron(COLOR_PAIR(ESC_COLOR));
						d.addstr("<Enter>");
						d.attroff(COLOR_PAIR(ESC_COLOR));
						d.addstr(": ");
						d.addstr(&l.Propose_treaty);
					}
					
					// mv to cursor location
					d.mv(y + INPUT_OFFSET_COL, x + *curs_col as i32);
				}
			}
		
		///////////////////////// citizen demand
		}else if let UIMode::CitizenDemandAlert {reason} = &self.ui_mode {
			let log_type = LogType::CitizenDemand {
						owner_id: self.cur_player as usize,
						reason: reason.clone()
					};
			let advisor_caution = &l.Advisor_caution;
			let w = {
				let log_len = print_log(&log_type, false, owners, doctrine_templates, l, d);
				max(advisor_caution.len(), log_len) + 4
			};
			let w_pos = print_window(ScreenSz{w, h: 2+4+3, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d);
			mvl!(); mvl!();
			print_log(&log_type, true, owners, doctrine_templates, l, d);
			mvl!(); mvl!(1);
			d.addstr(advisor_caution);
		
		///////////////////////// rioting alert
		}else if let UIMode::RiotingAlert {city_nm} = &self.ui_mode {
			let txt = l.Rioting_has_broken_out.replace("[city_nm]", city_nm);
			let w = txt.len() + 4;
			let w_pos = print_window(ScreenSz{w, h: 2+4, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d);
			mvl!(); mvl!(1);
			d.addstr(&txt);
		
		///////////////////////// generic alert
		}else if let UIMode::GenericAlert {txt} = &self.ui_mode {
			const MAX_W: usize = 70;
			let wrapped_txt = wrap_txt(txt, MAX_W);
			let w = min(MAX_W, txt.len()) + 4;
			let h = wrapped_txt.len() + 4;
			let w_pos = print_window(ScreenSz{w, h, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d); mvl!();
			
			for line in wrapped_txt.iter() {
				mvl!();
				d.addstr(line);
			}

		///////////////////////// civic advisors
		}else if let UIMode::CivicAdvisorsWindow = &self.ui_mode {
			// Cultural advisor (doctrinality_sum)
			// Chief of Police (crime_sum)
			// Public Counsel of Foreign Affairs / Public Counsel of Peace (pacifism_sum)
			// Health advisor (health_sum)
			// Economic advisor (unemployment_sum)
			struct Advisor {
				nm: String,
				txt: String,
				grade: String
			}
			
			let advisors = { // Vec[(advisor_nm, advice)]
				let pstats = &stats[self.cur_player as usize];
				//printlnq!("{:#?}", pstats.locally_logged.contrib);
				
				const N_LVLS: f32 = 5.;
				const SPACING: f32 = 1./N_LVLS;
				
				let advisors_advice_txts = vec![
					vec![&l.Cultural_Advisor_vlow, &l.Cultural_Advisor_low, &l.Cultural_Advisor_neutral, &l.Cultural_Advisor_high, &l.Cultural_Advisor_vhigh],
					vec![&l.Chief_of_Police_vlow, &l.Chief_of_Police_low, &l.Chief_of_Police_neutral, &l.Chief_of_Police_high, &l.Chief_of_Police_vhigh],
					vec![&l.Public_Counsel_of_FA_vlow, &l.Public_Counsel_of_FA_low, &l.Public_Counsel_of_FA_neutral, &l.Public_Counsel_of_FA_high, &l.Public_Counsel_of_FA_vhigh],
					vec![&l.Health_Advisor_vlow, &l.Health_Advisor_low, &l.Health_Advisor_neutral, &l.Health_Advisor_high, &l.Health_Advisor_vhigh],
					vec![&l.Economic_Advisor_vlow, &l.Economic_Advisor_low, &l.Economic_Advisor_neutral, &l.Economic_Advisor_high, &l.Economic_Advisor_vhigh]
				];
				
				let advice = |nm: &PersonName, title, mut val, sum, invert, advice_txts: &Vec<&String>| {
					if sum == 0. {
						val = 0.5;
					}else{
						val /= sum;
					}
					
					if invert {val = 1. - val;}
					
					let (txt, grade) = if val < SPACING {(advice_txts[0].clone(), "F")}
					else if val < (2.*SPACING) {(advice_txts[1].clone(), "C")}
					else if val < (3.*SPACING) {(advice_txts[2].clone(), "B")}
					else if val < (4.*SPACING) {(advice_txts[3].clone(), "A-")}
					else {(advice_txts[4].clone(), "A+")};
					
					Advisor {nm: format!("{}, {}:", nm.txt(), title), txt, grade: String::from(grade)}
				};
				
				let contrib = &pstats.locally_logged.contrib;
				let pos_sum = contrib.doctrine + contrib.pacifism;
				let neg_sum = contrib.health + contrib.unemployment + contrib.crime;
				
				let militarism_pacifism = if pstats.locally_logged.pacifism_sum < 0. {
					"miliatarism"
				}else{
					"pacifism"
				};
				
				let o = &owners[self.cur_player as usize];
				
				let mut pacifism_advice = advice(&o.pacifism_advisor_nm, &l.Public_Counsel_of_Foreign_Affairs, contrib.pacifism, pos_sum, false, &advisors_advice_txts[2]);
				pacifism_advice.txt = pacifism_advice.txt.replace("[]", militarism_pacifism);
				
				vec![
					advice(&o.doctrine_advisor_nm, &l.Cultural_Advisor, contrib.doctrine, pos_sum, false, &advisors_advice_txts[0]),
					advice(&o.crime_advisor_nm, &l.Chief_of_Police, contrib.crime, neg_sum, true, &advisors_advice_txts[1]),
					pacifism_advice,
					advice(&o.health_advisor_nm, &l.Health_Advisor, contrib.health, neg_sum, true, &advisors_advice_txts[3]),
					advice(&o.unemployment_advisor_nm, &l.Economic_Advisor, contrib.unemployment, neg_sum, true, &advisors_advice_txts[4])
				]
			};
			
			let w = 70;
			
			let mut n_lines = 2 + 2;
			let mut wrapped_quotes = Vec::with_capacity(advisors.len());
			for advisor in advisors.iter() {
				let wrapped = wrap_txt(&advisor.txt, w - 4);
				n_lines += wrapped.len() + 3;
				wrapped_quotes.push(wrapped);
			}
			
			let w_pos = print_window(ScreenSz{w, h: n_lines, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			// print key instructions
			mvl!(); buttons.Esc_to_close.print(None, l, d);
			
			{ // print title
				d.mv(row + y, x + ((w - l.Civic_Advisors.len())/2) as i32); row += 1;
				addstr_c(&l.Civic_Advisors, TITLE_COLOR, d); mvl!();
			}
			
			for (advisor, wrapped_quote) in advisors.iter().zip(wrapped_quotes.iter()) {
				mvl!(); addstr_c(&advisor.nm, TITLE_COLOR, d); mvl!();
				
				for quote_line in wrapped_quote.iter() {
					d.addstr(quote_line);
					mvl!();
				}
				
				{ // print grade
					let color = COLOR_PAIR(match advisor.grade.as_str() {
						"F" => CRED,
						"C" => CLOGO,
						"B" => CYELLOW,
						"A-" => CSAND1,
						"A+" => CGREEN,
						_ => {panicq!("unknown grade: {}", advisor.grade);}
					});
					
					d.addstr("Grade: ");
					addstr_attr(&format!("{}", advisor.grade), color, d);
				}
				mvl!();
			}

		///////////////////////// public polling
		}else if let UIMode::PublicPollingWindow = &self.ui_mode {
			let pstats = &stats[self.cur_player as usize];
			let contrib = &pstats.locally_logged.contrib;
			let pos_sum = contrib.doctrine + contrib.pacifism;
			let neg_sum = contrib.health + contrib.unemployment + contrib.crime;
			
			///// debug
			/*printlnq!("doctrine {} pacifism {}", contrib.doctrine/pos_sum, contrib.pacifism/pos_sum);
			printlnq!("health {} unemployment {} crime {}", contrib.health/neg_sum, contrib.unemployment/neg_sum, contrib.crime/neg_sum);
			printlnq!("{} {}", pos_sum, neg_sum);*/
			//////
					
			let w = 90;
			let n_lines = 30;
			
			let w_pos = print_window(ScreenSz{w, h: n_lines, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			// print key instructions
			mvl!(); buttons.Esc_to_close.print(None, l, d);
			
			{ // print title
				d.mv(row + y, x + ((w - l.Public_Polling.len())/2) as i32); row += 1;
				addstr_c(&l.Public_Polling, TITLE_COLOR, d); mvl!();
			}
			
			const RADIUS: i32 = 8;
			let mut pos = Coord {y: (row + y + 1) as isize, x: (x + 2) as isize};
			
			macro_rules! legend{($y_off:expr, $entry:expr, $txt:expr) => {
				d.mv(pos.y as i32 + RADIUS*2 + 1 + $y_off, pos.x as i32);
				d.attron($entry.color);
				d.addch(disp_chars.land_char);
				d.attroff($entry.color);
				d.addstr($txt);
				d.addstr(&format!(" ({}%)", ($entry.frac*100.).round() as usize));
			};};
			
			{
				d.mv(pos.y as i32 - 1, pos.x as i32);
				d.addstr(&l.What_are_you_most_satisfied_with);
				let ratios = vec![RatioEntry {frac: contrib.doctrine/pos_sum, color: COLOR_PAIR(CRED)},
							RatioEntry {frac: contrib.pacifism/pos_sum, color: COLOR_PAIR(CGREEN4)}];
				
				legend!(1, ratios[0], &l.doctrines_surrounding);
				if pstats.locally_logged.pacifism_sum >= 0. {
					legend!(2, ratios[1], &l.pacifism_surrounding);
				}else{
					legend!(2, ratios[1], &l.militarism_surrounding);
				}
				
				print_circle_plot(RADIUS, pos, &ratios, disp_chars, d);
			}
			
			////////////////
			{
				pos.x += (4*RADIUS + 15) as isize;
				d.mv(pos.y as i32 - 1, pos.x as i32);
				d.addstr(&l.What_are_you_least_satisfied_with);
				let ratios = vec![RatioEntry {frac: contrib.health/neg_sum, color: COLOR_PAIR(CRED)},
							RatioEntry {frac: contrib.unemployment/neg_sum, color: COLOR_PAIR(CGREEN4)},
							RatioEntry {frac: contrib.crime/neg_sum, color: COLOR_PAIR(CLOGO)}];
				
				print_circle_plot(RADIUS, pos, &ratios, disp_chars, d);
				
				pos.x += 5;
				legend!(1, ratios[0], &l.sickness_surrounding);
				legend!(2, ratios[1], &l.unemployment_surrounding);
				legend!(3, ratios[2], &l.crime_surrounding);
			}
			
			{ // fraction of greater to lesser pos or neg
				d.mv(pos.y as i32 + RADIUS*2 + 6, x + 2);
				let (frac, greater_label, lesser_label) = if contrib.pos_sum > (-contrib.neg_sum) {
					(-contrib.pos_sum/contrib.neg_sum, &l.positive, &l.negative)
				}else{
					(-contrib.neg_sum/contrib.pos_sum, &l.negative, &l.positive)
				};
				//printlnq!("{} {}", contrib.pos_sum, contrib.neg_sum);
				d.addstr(&l.on_average_responses.replace("[pos_neg1]", greater_label)
						.replace("[pos_neg2]", lesser_label)
						.replace("[frac]", &format!("{}", frac.round() as usize)));
			}
			
		///////////////////////// no actions remain
		}else if let UIMode::MvWithCursorNoActionsRemainAlert {unit_ind} = &self.ui_mode {
			let u = &units[*unit_ind];
			let txt = l.No_actions_remain.replace("[battalion_nm]", &u.nm)
								.replace("[unit_type]", &u.template.nm[l.lang_ind]);
			let w = txt.len() + 4;
			let w_pos = print_window(ScreenSz{w, h: 2+4, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d); mvl!(); mvl!();
			d.addstr(&txt); d.mv(row + y, x);

		///////////////////////// unmoved units notification
		}else if let UIMode::UnmovedUnitsNotification = self.ui_mode {
			let w = l.Assign_actions_to_them.len() + 4;
			let w_pos = print_window(ScreenSz{w, h: 5+3, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let w = (w - 2) as i32;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d); mvl!();
			center_txt(&l.You_have_unmoved_units, w, title_c, d);
			mvl!();mvl!();
			d.addstr(&l.Assign_actions_to_them); d.mv(row + y, x);
			center_txt(&l.Fortify_them_if, w, None, d);
		
		///////////////////////// foreign units in sector alert
		}else if let UIMode::ForeignUnitInSectorAlert {sector_nm, battalion_nm} = &self.ui_mode {
			let txt = l.The_X_Battalion_reports_activity.replace("[battalion_nm]", battalion_nm)
										  .replace("[sector_nm]", sector_nm);
			
			let w = txt.len() + 4;
			let w_pos = print_window(ScreenSz{w, h: 2+1+3, sz:0}, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let mut row = 0;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			mvl!(); buttons.Esc_to_close.print(None, l, d);
			mvl!();mvl!(1);
			d.addstr(&txt);
		
		////////////////////////// civilization intel
		}else if let UIMode::CivilizationIntelWindow {mode, selection_phase} = self.ui_mode {
			let contacted_civs = contacted_civilizations_list(relations, stats, owners, self.cur_player, turn);
			
			// select civilization
			if selection_phase {
				print_list_window(mode, &l.Select_civilization, contacted_civs, self, disp_chars, None, None, 0, Some(owners), l, buttons, d);
			// show information for civilization
			}else{
				let owner_id = if let ArgOptionUI::OwnerInd(owner_id) = contacted_civs.options[mode].arg {
					owner_id }else{ panicq!("owner id not in menu options"); };
				
				let o = &owners[owner_id];
				let motto_txt = format!("{} \"{}\"", l.National_motto, o.motto);
				let w = max("Our intelligence tells us this is an aggressive and mythological culture.".len(),
						motto_txt.len()) + 4;
				let w_pos = print_window(ScreenSz{w, h: 8+3, sz:0}, self.screen_sz, disp_chars, d);
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let w = (w - 2) as i32;
				
				let mut row = 0;
				macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
							     ($final: expr) => {d.mv(row + y, x);}};
				
				// print key instructions
				{
					mvl!();
					buttons.Esc_to_close.print(None, l, d);
					
					mvl!();
					d.attron(COLOR_PAIR(ESC_COLOR));
					d.addstr("<Left arrow>");
					d.attroff(COLOR_PAIR(ESC_COLOR));
					d.addstr(" go back"); 
				}
				
				///////////// title -- country name
				{
					mvl!();
					let txt_len = format!("Our Intel on {}", o.nm).len();
					let g = (w as usize - txt_len) / 2;
					let mut sp = String::with_capacity(g);
					for _ in 0..g {sp.push(' ');}
					
					d.addstr(&sp);
					d.addstr("Our Intel on ");
					
					set_player_color(o, true, d);
					d.addstr(&o.nm);
					set_player_color(o, false, d);
					
					mvl!();
				}
				
				//////////// ruler
				mvl!();
				d.addstr(&format!("{} {} {}", l.Ruler, o.ruler_nm.first, o.ruler_nm.last));
				
				////////////// motto
				mvl!();
				d.addstr(&motto_txt);
				
				////////// prevailing doctrine
				{
					mvl!();
					let pstats = &stats[owner_id];
					d.addstr(&l.Prevailing_doctrine);
					if pstats.doctrine_template.id == 0 {
						d.addstr(&format!(" {}", l.None));
					}else{
						d.addstr(&format!(" {}", pstats.doctrine_template.nm[l.lang_ind]));
					}
				}
				
				//////////// personality
				if let PlayerType::AI(personality) = &o.player_type {
					mvl!();mvl!(true);
					
					d.addstr("Our intelligence tells us this is a");
					
					const SPACE: f32 = 2./3.;
					if personality.friendliness < (SPACE - 1.) {
						d.addch('n');
						d.attron(COLOR_PAIR(CRED));
						d.addstr(" aggressive");
						d.attroff(COLOR_PAIR(CRED));
					}else if personality.friendliness < (2.*SPACE - 1.) {
						d.addstr(" reserved");
					}else{
						d.attron(COLOR_PAIR(CGREEN1));
						d.addstr(" friendly");
						d.attroff(COLOR_PAIR(CGREEN1));
					}
					
					d.addstr(" and ");
					
					d.addstr(if personality.spirituality < (SPACE - 1.) {
						"scientific"
					}else if personality.spirituality < (2.*SPACE - 1.) {
						"pragmatic"
					}else if personality.spirituality < (2.5*SPACE - 1.) {
						"religious"
					}else{
						"mythological"
					});
					d.addstr(" culture.");
				}
			}
			
		/////////////////////////// switch to player
		}else if let UIMode::SwitchToPlayerWindow {mode} = self.ui_mode {
			let all_civs = all_civilizations_list(owners);
			
			print_list_window(mode, &l.Select_civilization, all_civs, self, disp_chars, None, None, 0, Some(owners), l, buttons, d);

		////////////////////////// discover technology
		}else if let UIMode::DiscoverTechWindow {mode} = self.ui_mode {
			let techs = undiscovered_tech_list(&stats[self.cur_player as usize], tech_templates, l);
			
			print_list_window(mode, &l.Select_technology, techs, self, disp_chars, None, None, 0, None, l, buttons, d);
		
		////////////////////////// obtain resource
		}else if let UIMode::ObtainResourceWindow {mode} = self.ui_mode {
			let resources = all_resources_list(resource_templates, l);
			
			print_list_window(mode, &l.Select_resource, resources, self, disp_chars, None, None, 0, None, l, buttons, d);
		
		////////////////////////// select doctrine dedication
		}else if let UIMode::SelectBldgDoctrine {mode, ..} = self.ui_mode {
			let list = doctrines_available_list(&stats[self.cur_player as usize], doctrine_templates, l);
			
			print_list_window(mode, &l.Dedicate_to, list.clone(), self, disp_chars, None, None, 0, None, l, buttons, d);
			
			// print details for selected bldg
			if let ArgOptionUI::DoctrineTemplate(Some(doc)) = list.options[mode].arg {
				self.show_exemplar_info(doc.id, EncyclopediaCategory::Doctrine, OffsetOrPos::Offset(26), Some(25), OffsetOrPos::Offset(mode+4), InfoLevel::Abbrev, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, &stats[self.cur_player as usize], disp_chars, l, d);
			}else{panicq!("could not find doctrine template {}", mode);}
		
		////////////////////////// select auto-explore type
		}else if let UIMode::SelectExploreType {mode, ..} = self.ui_mode {
			print_list_window(mode, &l.Select_exploration_type, explore_types_list(l), self, disp_chars, None, None, 0, None, l, buttons, d);
		
		////////////////////////// set difficulty
		}else if let UIMode::SetDifficultyWindow {mode} = self.ui_mode {
			let list = game_difficulty_list(game_difficulties);
			
			print_list_window(mode, &l.Select_difficulty, list, self, disp_chars, None, None, 0, None, l, buttons, d);

		////////////////////////// place unit
		}else if let UIMode::PlaceUnitWindow {mode} = self.ui_mode {
			let unit_opts = discovered_units_list(&stats[self.cur_player as usize], unit_templates, l);
			
			print_list_window(mode, &l.Select_a_unit, unit_opts, self, disp_chars, None, None, 0, None, l, buttons, d);

		////////////// show resources available
		}else if let UIMode::ResourcesAvailableWindow = self.ui_mode {
			let pstats = &stats[self.cur_player as usize];
			let n_resources_avail = pstats.resources_avail.iter().filter(|&&r| r != 0).count();
			let w = 30;
			let window_sz = if n_resources_avail != 0 {
				ScreenSz{w, h: n_resources_avail + 6, sz: 0}
			}else{
				ScreenSz{w, h: 7, sz: 0}
			};
			let w_pos = print_window(window_sz, self.screen_sz, disp_chars, d);
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 1;
			
			let w = (w - 2) as i32;
			
			d.mv(y,x);
			center_txt(&l.Resources_available, w, title_c, d);
			
			// display resources
			{
				if n_resources_avail == 0 {
					d.mv(y + 2, x);
					center_txt(&l.None, w, None, d);
				}else{
					let mut n_shown = 0;
					for (avail, resource) in pstats.resources_avail.iter().zip(resource_templates) {
						if *avail == 0 || !pstats.resource_discov(resource) {continue;}
					
						d.mv(y + 2 + n_shown, x + 1);
						d.addstr(&format!("{}:", resource.nm[l.lang_ind]));
						
						let n_avail = format!("{}", avail);
						d.mv(y + 2 + n_shown, w_pos.x as i32 + window_sz.w as i32 - n_avail.len() as i32 - 2);
						d.addstr(&n_avail);
						
						n_shown += 1;
					}
					debug_assertq!(n_shown == n_resources_avail as i32);
				}
			}
			
			// instructions
			{
				let button = &mut buttons.Esc_to_close;
				let instructions_w = button.print_txt(l).len() as i32;
				let gap = ((w - instructions_w)/2) as i32;
				if n_resources_avail != 0 {
					d.mv(y + 3 + n_resources_avail as i32, x - 1 + gap);
				}else{
					d.mv(y + 4, x - 1 + gap);
				}
				button.print(None, l, d);
			}
			
		////////////////////// resource locations discovered
		}else if let UIMode::ResourcesDiscoveredWindow {mode} = self.ui_mode {
			let cursor_coord = self.cursor_to_map_coord_zoomed_in(map_data);
			
			let resource_opts = discovered_resources_list(&stats[self.cur_player as usize], cursor_coord, resource_templates, *map_data.map_szs.last().unwrap());
			
			let row = print_list_window(mode, &l.Go_to_resource, resource_opts.clone(), self, disp_chars, None, None, 0, None, l, buttons, d).0.y as usize;
			
			// show info box
			if resource_opts.options.len() > 0 {
				let pstats = &stats[self.cur_player as usize];
				if let ArgOptionUI::ResourceWCoord {rt, ..} = resource_opts.options[mode].arg {
					let w = 29 + 3;
					self.show_exemplar_info(rt.id as usize, EncyclopediaCategory::Resource, OffsetOrPos::Offset(w), None, OffsetOrPos::Pos(row + mode + 4), InfoLevel::Abbrev, unit_templates, bldg_templates, tech_templates, resource_templates, doctrine_templates, pstats, disp_chars, l, d);
				}else{panicq!("invalid UI setting");}
			}
		}
	}
}

