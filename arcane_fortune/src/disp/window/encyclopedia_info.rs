use std::cmp::Ordering;
use crate::gcore::Brigade;
use crate::disp::*;
use crate::units::{UnitTemplate, MovementType, WORKER_NM, Unit};
use crate::map::{METERS_PER_TILE, ZoneType, ArabilityType};
use crate::player::Stats;
use super::*;

pub enum OffsetOrPos {
	Offset(usize),
	Pos(usize)
}

#[derive(PartialEq)]
pub enum InfoLevel {
	Full,
	Abbrev,
	AbbrevNoCostNoProdTime
}

impl InfoLevel {
	pub fn is_not_full(&self) -> bool {
		match self {
			InfoLevel::Full {..} => false,
			InfoLevel::Abbrev | InfoLevel::AbbrevNoCostNoProdTime => true
		}
	}
}

// show tech requirements, movement, unit bonuses for a given unit, bldg, or tech
// `mode` is the index into the template vector
// `w_offset` = Offset(_) is the column offset to start the window at (relative to the center of the screen)
// `info_level` when Abbrev or AbbrevNo..., used as infobox (for showing unit & bldg production options), else as main, full encyclopedia view
impl DispState<'_,'_,'_,'_,'_,'_> {
	pub fn show_exemplar_info(&mut self, mode: usize, category: EncyclopediaCategory, w_offset_or_pos: OffsetOrPos, window_w: Option<i32>,
			h_offset_or_pos: OffsetOrPos, info_level: InfoLevel, temps: &Templates, pstats: &Stats) {
		let d = &mut self.renderer;
		let w = self.iface_settings.screen_sz.w as i32;
		let l = &self.local;
		
		let window_w = if let Some(window_w) = window_w {
			window_w
		}else{38};
		
		if w < window_w {return;} // screen not wide enough
		
		let w_offset = match w_offset_or_pos {
			OffsetOrPos::Offset(w_offset) => {w_offset}
			OffsetOrPos::Pos(pos) => {(pos as i32 - (w - window_w) / 2) as usize}
		};
		
		let col = (w - window_w) / 2 + w_offset as i32;
		
		let mut row = {
			let h = self.iface_settings.screen_sz.h as i32;
			
			// get number of rows of window for centering on screen
			let window_h = match category {
				EncyclopediaCategory::Unit => {
					let ut = &temps.units[mode];
					let mut n = 5 + 2 + 4;
					if info_level.is_not_full() {n -= 3;}
					if ut.carry_capac != 0 {n += 1;}
					if let Some(_) = ut.attack_per_turn {n += 2;}
					n
				} EncyclopediaCategory::Bldg => {
					let bt = &temps.bldgs[mode];
					let mut n = 3 + 2;
					if info_level.is_not_full() {n -= 3;}
					if bt.research_prod != 0 {n += 1;}
					if bt.resident_max != 0 {n += 3;}
					n
				} EncyclopediaCategory::Tech => {
					let mut n = 3;
					
					// if tech discovers any units, add a line
					for ut in temps.units.iter() {
						if let Some(tech_req) = &ut.tech_req {
							if tech_req.contains(&mode) {
								n += 1;
								break;
							}
						}
					}
					
					// if tech discovers any bldgs, add a line
					for bt in temps.bldgs.iter() {
						if let Some(tech_req) = &bt.tech_req {
							if tech_req.contains(&mode) {
								n += 1;
								break;
							}
						}
					}
					n
				} EncyclopediaCategory::Doctrine => {
					let mut n = 3;
					let d = &temps.doctrines[mode];
					if !d.pre_req_ind.is_none() {n += 1;}
					if d.bldg_req != 0. {n += 1;}
					if d.health_bonus != 0. {n += 1;}
					if d.crime_bonus != 0. {n += 1;}
					if d.pacifism_bonus != 0. {n += 1;}
					if d.happiness_bonus != 0. {n += 1;}
					if d.tax_aversion != 0. {n += 1;}
					n += d.bldgs_unlocks(temps.bldgs).len() as i32;
					n
				} EncyclopediaCategory::Resource => {
					let n = 3 + 2;
					n
				}
			} + 2;
			
			if h < window_h {return;} // screen not tall enough
			
			match h_offset_or_pos {
				OffsetOrPos::Offset(h_offset) => {
					//printlnq!("{}", h_offset);
					(h - window_h) / 2 + h_offset as i32
				}
				OffsetOrPos::Pos(h_pos) => {
					if h_pos <= (h - window_h + 3) as usize {
						h_pos as i32
					}else{
						h - window_h + 3
					}
				}
			}
		};
		
		macro_rules! clr_ln{() => {
				d.mv(row,col);
				d.addch(self.chars.vline_char);
				for _ in 0..window_w {d.addch(' ');}
				d.addch(self.chars.vline_char);
			};
			($i: expr) => {
				d.mv(row-$i, col);
				d.addch(self.chars.vline_char);
				for _ in 0..window_w {d.addch(' ');}
				d.addch(self.chars.vline_char);
			};
		}
		
		macro_rules! ctr_txt{($txt: expr) => {
			if ($txt.len() as i32) < w { // text too long
				clr_ln!();
				d.mv(row, (w - $txt.len() as i32)/2 + w_offset as i32);
				d.addstr($txt);
				row += 1;
			}
		};}
		
		macro_rules! lr_txt{
			($l_txt: expr, $r_txt: expr) => {
				clr_ln!();
				d.mv(row,col+2);
				d.addstr($l_txt);
				d.mv(row, col + window_w - $r_txt.len() as i32);
				d.addstr($r_txt);
				row += 1;
			};
			($l_txt: expr, $r_txt: expr, $row: expr) => {
				// clear line
				d.mv($row,col);
				d.addch(self.chars.vline_char);
				for _ in 0..window_w {d.addch(' ');}
				d.addch(self.chars.vline_char);
				
				// print txt
				d.mv($row,col+2);
				d.addstr($l_txt);
				d.mv($row, col + window_w - $r_txt.len() as i32);
				d.addstr($r_txt);
				$row += 1;
		};}
		
		// tech reqs to produce unit or bldg, `row` is the row to start printing on
		macro_rules! print_tech_req{($tech_req_opt: expr, $row: expr) => {
			if let Some(tech_req) = $tech_req_opt {
				for (i, tech) in tech_req.iter().enumerate() {
					lr_txt!(if i != 0 {""} else {&l.Required_technology},
							&temps.techs[*tech as usize].nm[l.lang_ind], *$row);
				}
			}else {lr_txt!(&l.Required_technology, &l.None, *$row);}
		};}
		
		macro_rules! print_resources_req{($resources_req: expr, $row: expr) => {
			for (i, rt) in $resources_req.iter().enumerate() {
				lr_txt!(if i != 0 {""} else {&l.Required_resources}, &rt.nm[l.lang_ind], *$row);
			}
		};}
		
		{ // print window top line
			d.mv(row- if let InfoLevel::Full {..} = info_level {4} else {1}, col);
			d.addch(self.chars.ulcorner_char);
			for _ in 0..window_w {d.addch(self.chars.hline_char);}
			d.addch(self.chars.urcorner_char);
		}
		
		// print key instructions
		if let InfoLevel::Full = info_level {
			clr_ln!(3);
			
			d.mv(row-3, col+2);
			self.buttons.Esc_to_close.print(None, l, d);
			
			clr_ln!(2);
			
			d.mv(row-2, col+2);
			self.buttons.to_go_back.print(None, l, d);
			
			clr_ln!(1);
		}
		
		//////////////// print text of window
		match category {
			EncyclopediaCategory::Unit => {
				let ut = &temps.units[mode];
				
				if let InfoLevel::Full {..} = info_level {
					ctr_txt!(&ut.nm[l.lang_ind]);
					ctr_txt!("");
					lr_txt!(&l.Symbol, &format!("{}", ut.char_disp));
					print_tech_req!(&ut.tech_req, &mut row);
					print_resources_req!(&ut.resources_req, &mut row);
					
					for bt in temps.bldgs.iter() {
						if let Some(units_producable) = &bt.units_producable {
							if units_producable.contains(&ut) {
								lr_txt!(&format!("{}:", l.Produced_by), &format!("{}", bt.nm[l.lang_ind]));
								break;
							}
						}
					}
				}
				lr_txt!(&l.Movement_speed, &format!("{:.1} {}", (METERS_PER_TILE*ut.actions_per_turn as f32)/1000., l.km_day));
				
				if ut.carry_capac != 0 {
					let mut carry_capac_string = format!("{} unit", ut.carry_capac);
					if ut.carry_capac > 1 {carry_capac_string.push('s');}
					lr_txt!(&l.Carrying_capacity, &carry_capac_string);
				}
				
				lr_txt!(&format!("{}:", l.Health), &format!("{}", ut.max_health));
				
				if let Some(attack_per_turn) = ut.attack_per_turn {
					lr_txt!(&l.Attack_strength, &format!("{}", attack_per_turn));
					if let Some(siege_bonus_per_turn) = ut.siege_bonus_per_turn {
						lr_txt!(&l.Siege_bonus, &format!("{}", siege_bonus_per_turn));
					}
					lr_txt!(&l.Attack_range, &format!("{} m", (METERS_PER_TILE as usize)*ut.attack_range.unwrap()));
				}
				
				// terrain
				{
					let terrain_desc = match ut.movement_type {
						MovementType::AllWater => {&l.All_water}
						MovementType::ShallowWater => {&l.Shallow_water_only}
						MovementType::Land => {&l.Land_only}
						MovementType::LandAndOwnedWalls => {&l.Land_and_owned_walls}
						MovementType::Air => {&l.Air}
						MovementType::AllMapTypes | MovementType::N => {panicq!("Invalid terrain description");}
					};
					
					lr_txt!(&format!("{}:", l.Terrain), terrain_desc);
				}
				
				// cost & production time
				if info_level != InfoLevel::AbbrevNoCostNoProdTime {
					lr_txt!(&format!("{}:", l.Time_required_to_produce), &l.date_interval_str(ut.production_req));
					lr_txt!(&l.Cost_per_day, &format!("{} gold", ut.upkeep));
				}
				
				if let InfoLevel::Full {..} = info_level {
					ctr_txt!("");
					ctr_txt!(&format!("(Note: 1 tile = {} m)", METERS_PER_TILE));
				}
			} EncyclopediaCategory::Bldg => {
				let bt = &temps.bldgs[mode];
				
				if let InfoLevel::Full {..} = info_level {
					ctr_txt!(&bt.nm[l.lang_ind]);
					ctr_txt!("");
					lr_txt!(&l.Symbol, &format!("{}", bt.plot_zoomed));
					print_tech_req!(&bt.tech_req, &mut row);
					if let Some(doctrine_req) = bt.doctrine_req {
						lr_txt!(&l.Req_prevailing_doctrine, &doctrine_req.nm[l.lang_ind]);
					}
				}
				
				if bt.research_prod != 0 {
					lr_txt!(&l.Research_output_per_day, &format!("{}", bt.research_prod));
				}
				
				if let BldgType::Taxable(zone) = bt.bldg_type {
					lr_txt!(&format!("{}:", l.Zone), zone.ztype.to_str(l));
				}
				
				// cost & production time
				if info_level != InfoLevel::AbbrevNoCostNoProdTime {
					if bt.upkeep < 0. {
						lr_txt!("Maximal tax payments:", &format!("{} gold", -bt.upkeep));
					}else{
						lr_txt!(&l.Cost_per_day, &format!("{} gold", bt.upkeep));
					}
					
					if bt.construction_req != 0. {
						let worker = UnitTemplate::frm_str(WORKER_NM, temps.units);
						//lr_txt!("Construction required:", &format!("{} days", bt.construction_req / worker.actions_per_turn));
						lr_txt!(&format!("{}:", l.Time_required_to_construct), &l.date_interval_str(bt.construction_req / worker.actions_per_turn));
					}
				}
				
				// zone bonuses
				if let BldgType::Gov(zone_bonuses) = &bt.bldg_type {
					for (zt_ind, zone_bonus_opt) in zone_bonuses.iter().enumerate() {
						if let Some(zone_bonus) = zone_bonus_opt {
							lr_txt!(&format!("{} bonus:", ZoneType::from(zt_ind).to_str(l)),
								  &format!("{}", zone_bonus));
						}
					}
				}
				
				{ // crime, happiness, doctrinality, pacifism, health, job search bonus
					if bt.crime_bonus < 0. {
						lr_txt!(&l.Crime_bonus, &format!("{}", bt.crime_bonus));
					}else if bt.crime_bonus != 0. {
						lr_txt!(&format!("{}:", l.Crime), &format!("{}", bt.crime_bonus));
					}
					/*else if bt.crime_bonus < 0. {
						lr_txt!(&l.Safety, &format!("{}", -bt.crime_bonus));
					}*/
					
					if bt.happiness_bonus != 0. {
						lr_txt!(&format!("{}:", l.Happiness), &format!("{}", bt.happiness_bonus));
					}
					
					if bt.doctrinality_bonus > 0. {
						lr_txt!(&l.Doctrinality, &format!("{}", bt.doctrinality_bonus));
					}else if bt.doctrinality_bonus < 0. {
						lr_txt!(&l.Methodicalism, &format!("{}", -bt.doctrinality_bonus));
					}
					
					if bt.pacifism_bonus > 0. {
						lr_txt!(&l.Pacifism, &format!("{}", bt.pacifism_bonus));
					}else if bt.pacifism_bonus < 0. {
						lr_txt!(&l.Militarism, &format!("{}", -bt.pacifism_bonus));
					}
					
					if bt.health_bonus > 0. {
						lr_txt!(&l.Health_bonus, &format!("{}", bt.health_bonus));
					}
					
					if bt.job_search_bonus > 0. {
						lr_txt!(&l.Economic_bonus, &format!("{}", bt.job_search_bonus));
					}
				}
				
				if let Some(units_producable) = &bt.units_producable {
					for (i, ut) in units_producable.iter().enumerate() {
						let can_create = format!("{}:", l.Can_create);
						lr_txt!(if i != 0 {""} else {&can_create}, &ut.nm[l.lang_ind]);
					}
				}
				
				if bt.resident_max != 0 {
					lr_txt!(&l.Max_residents, &format!("{}", bt.resident_max));
					lr_txt!(&l.Max_consumption, &format!("{}", bt.cons_max));
					lr_txt!(&l.Max_production, &format!("{}", bt.prod_max));
				}
			} EncyclopediaCategory::Tech => {
				let t = &temps.techs[mode];
				
				ctr_txt!(&t.nm[l.lang_ind]);
				ctr_txt!("");
				
				// req. tech
				if let Some(tech_req) = &t.tech_req {
					for (i, tech) in tech_req.iter().enumerate() {
						lr_txt!(if i != 0 {""} else {&l.Required_technology},
							&temps.techs[*tech as usize].nm[l.lang_ind]);
					}
				}else {lr_txt!(&l.Required_technology, &l.None);}
				
				if pstats.research_per_turn == 0 {
					lr_txt!(&l.Research_required, &format!("{}", t.research_req));
				}else{
					lr_txt!(&l.Time_to_discover, &l.date_interval_str(t.research_req as f32 / pstats.research_per_turn as f32));
				}
				
				// find units that require this technology
				{
					let mut units_discov = false;
					for ut in temps.units.iter() {
						if let Some(tech_req) = &ut.tech_req {
							if tech_req.contains(&mode) {
								lr_txt!(if units_discov {""} else {&l.Req_for_creating}, &ut.nm[l.lang_ind]);
								units_discov = true;
							}
						}
					}
				}
				
				// find buildings that require this technology
				{
					let mut bldgs_discov = false;
					for bt in temps.bldgs.iter() {
						if let Some(tech_req) = &bt.tech_req {
							if tech_req.contains(&mode) {
								lr_txt!(if bldgs_discov {""} else {&l.Req_for_building}, &bt.nm[l.lang_ind]);
								bldgs_discov = true;
							}
						}
					}
				}
				
				// find buildings that require this technology
				{
					let mut resources_discov = false;
					for rt in temps.resources.iter() {
						if rt.tech_req.contains(&mode) {
							lr_txt!(if resources_discov {""} else {&l.Req_for_discovering}, &rt.nm[l.lang_ind]);
							resources_discov = true;
						}
					}
				}
			} EncyclopediaCategory::Doctrine => {
				let d = &temps.doctrines[mode];
				
				if let InfoLevel::Full {..} = info_level {
					ctr_txt!(&d.nm[l.lang_ind]);
					ctr_txt!("");
					
					// req. doctrine
					if let Some(pre_req_ind) = &d.pre_req_ind {
						lr_txt!(&l.Pre_req, &temps.doctrines[*pre_req_ind].nm[l.lang_ind]);
					}
				}
				
				if d.bldg_req != 0. {
					lr_txt!(&l.Bldg_pts_req, &format!("{}", d.bldg_req));
				}
				
				// bonuses and bldgs discovered by this doctrine
				{
					if d.health_bonus != 0. {
						lr_txt!(&l.Health_bonus, &format!("{}", d.health_bonus));
					}
					
					if d.crime_bonus != 0. {
						lr_txt!(&l.Crime_bonus, &format!("{}", d.crime_bonus));
					}
					
					if d.pacifism_bonus > 0. {
						lr_txt!(&l.Pacifism_bonus, &format!("{}", d.pacifism_bonus));
					}else if d.pacifism_bonus < 0. {
						lr_txt!(&l.Militarism_bonus, &format!("{}", -d.pacifism_bonus));
					}
					
					if d.happiness_bonus != 0. {
						lr_txt!(&l.Happiness_bonus, &format!("{}", d.happiness_bonus));
					}
					
					if d.tax_aversion != 0. {
						lr_txt!(&l.Tax_aversion, &format!("{}", d.tax_aversion));
					}
					
					let bldgs_unlocks = d.bldgs_unlocks(temps.bldgs);
					if let Some(bldg_unlocked) = bldgs_unlocks.first() {
						lr_txt!(&l.Discovers, &bldg_unlocked.nm[l.lang_ind]);
						for bldg_unlocked in bldgs_unlocks.iter().skip(1) {
							lr_txt!("", &bldg_unlocked.nm[l.lang_ind]);
						}
					}
				}
			} EncyclopediaCategory::Resource => {
				let rt = &temps.resources[mode];
				
				if let InfoLevel::Full {..} = &info_level {
					ctr_txt!(&rt.nm[l.lang_ind]);
					ctr_txt!("");
					lr_txt!(&l.Symbol, &format!("{}", rt.plot_zoomed));
					print_tech_req!(&Some(rt.tech_req.clone()), &mut row);
				}
				
				lr_txt!(&l.Zoning_req_to_use, rt.zone.to_str(l));
				
				// zone bonuses
				for (zone_ind, zone_bonus) in rt.zone_bonuses.iter().enumerate() {
					if let Some(bonus) = zone_bonus {
						if *bonus != 0 {
							lr_txt!(&format!("{} bonus:", ZoneType::from(zone_ind).to_str(l)),
								  &format!("{}", bonus));
						}
					}
				}
				
				{ // units requiring this resource
					let mut found = false;
					for ut in temps.units.iter() {
						for resource_req in ut.resources_req.iter() {
							if *resource_req == rt {
								if found {
									lr_txt!("", &format!("{}", ut.nm[l.lang_ind]));
								}else{
									lr_txt!(&l.Req_to_create, &format!("{}", ut.nm[l.lang_ind]));
									found = true;
								}
							}
						}
					}
				}
				
				// arability probs
				if let InfoLevel::Full {..} = info_level {
					struct ArabilityProb {ind: usize, prob: f32}
					let mut arability_probs = Vec::with_capacity(rt.arability_probs.len());
					
					for (ind, arability_prob) in rt.arability_probs.iter().enumerate() {
						if let Some(prob) = arability_prob {
							arability_probs.push(ArabilityProb {ind, prob: *prob});
						}
					}
					
					// sort from greatest to least
					arability_probs.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(Ordering::Less));
					
					if arability_probs.len() > 1 {
						lr_txt!(&l.Found_in_most_to_least, "");
					}else{
						lr_txt!(&l.Found_in, "");
					}
					
					for arability_prob in arability_probs {
						let arability = ArabilityType::from(arability_prob.ind);
						let arability_color = COLOR_PAIR(arability.to_color(false));
						let arability_txt = arability.to_str(l);
						
						clr_ln!();
						d.mv(row, col + window_w - arability_txt.len() as i32);
						d.attron(arability_color);
						d.addstr(&arability_txt);
						d.attroff(arability_color);
						row += 1;
					}
				}
			}
		} // match category
		
		// print window bottom line
		{
			d.mv(row,col);
			d.addch(self.chars.llcorner_char);
			for _ in 0..window_w {d.addch(self.chars.hline_char);}
			d.addch(self.chars.lrcorner_char);
		}
	}
	
	pub fn show_brigade_units(&mut self, brigade: &Brigade, pos: Coord, units: &Vec<Unit>) {
		let d = &mut self.renderer;
		let l = &self.local;
		
		// create text to show
		let mut disp_txt = Vec::with_capacity(brigade.unit_inds.len());
		for unit_ind in brigade.unit_inds.iter() {
			if let Some(unit) = units.get(*unit_ind) {
				disp_txt.push(format!("{} ({})", unit.nm, unit.template.nm[l.lang_ind]));
			}else{panicq!("could not get unit ind {} units len {}", unit_ind, units.len());}
		}
		
		let window_w = if let Some(txt_len) = disp_txt.iter().map(|txt| txt.len()).max() {txt_len as i32} else {return;} + 2;
		let w = self.iface_settings.screen_sz.w as i32;
		if w < window_w {return;} // screen not wide enough
		
		let w_offset = (pos.x as i32 - (w - window_w) / 2) as usize;
		
		let col = (w - window_w) / 2 + w_offset as i32;
		
		let mut row = {
			let h = self.iface_settings.screen_sz.h as i32;
			
			// get number of rows of window for centering on screen
			let window_h = (2 + brigade.unit_inds.len()) as i32;
					
			if h < window_h {return;} // screen not tall enough
			
			if pos.y <= (h - window_h) as isize {
				pos.y as i32
			}else{
				h - window_h
			}
		};
		
		macro_rules! clr_ln{() => {
				d.mv(row,col);
				d.addch(self.chars.vline_char);
				for _ in 0..window_w {d.addch(' ');}
				d.addch(self.chars.vline_char);
			};}
		
		{ // print window top line
			d.mv(row-1, col);
			d.addch(self.chars.ulcorner_char);
			for _ in 0..window_w {d.addch(self.chars.hline_char);}
			d.addch(self.chars.urcorner_char);
		}
		
		//////////////// print text of window
		for txt in disp_txt.iter() {
			clr_ln!();
			d.mv(row, col + 2); row += 1;
			d.addstr(txt);
		}
			
		{ // print window bottom line
			d.mv(row,col);
			d.addch(self.chars.llcorner_char);
			for _ in 0..window_w {d.addch(self.chars.hline_char);}
			d.addch(self.chars.lrcorner_char);
		}
	}
}

