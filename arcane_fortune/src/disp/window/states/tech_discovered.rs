use super::*;
// when a new tech is discovered
pub struct TechDiscoveredWindowState {
	pub tech_ind: usize,
	pub prev_auto_turn: AutoTurn
}

impl TechDiscoveredWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let tech_discov = &temps.techs[self.tech_ind];
		let tech_id = tech_discov.id as usize;
		
		// find units that require this technology
		let units_discov = {
			let mut units_discov = Vec::with_capacity(temps.units.len());
			for ut in temps.units.iter() {
				if let Some(tech_req) = &ut.tech_req {
					if tech_req.contains(&tech_id) {
						units_discov.push(ut.nm[dstate.local.lang_ind].clone());
					}
				}
			}
			units_discov
		};
		
		// find buildings that require this technology
		let bldgs_discov = {
			let mut bldgs_discov = Vec::with_capacity(temps.bldgs.len());
			for bt in temps.bldgs.iter() {
				if let Some(tech_req) = &bt.tech_req {
					if tech_req.contains(&tech_id) {
						bldgs_discov.push(bt.nm[dstate.local.lang_ind].clone());
					}
				}
			}
			bldgs_discov
		};
		
		// find buildings that require this technology
		let resources_discov = {
			let mut resources_discov = Vec::with_capacity(temps.resources.len());
			for rt in temps.resources.iter() {
				if rt.tech_req.contains(&tech_id) {
					resources_discov.push(rt.nm[dstate.local.lang_ind].clone());
				}
			}
			resources_discov
		};
		
		let additional_ln = |list: &Vec<String>| {
			if list.len() == 0 {0} else {1}
		};
		
		let window_sz = ScreenSz {
			h: 6 + units_discov.len() + bldgs_discov.len() + resources_discov.len() + 
				 additional_ln(&units_discov) + additional_ln(&bldgs_discov) +
				 additional_ln(&resources_discov),
			w: "You have discovered .".len() + tech_discov.nm.len() + 8,
			sz: 0
		};
		
		let pos = dstate.print_window(window_sz);
		
		let mut row = pos.y as i32 + 1;
		
		macro_rules! ctr_txt{($txt: expr) => {
			if $txt.len() < window_sz.w { // text too long
				dstate.mv(row, (window_sz.w - $txt.len()) as i32/2 + pos.x as i32);
				dstate.renderer.addstr($txt);
				row += 1;
			}
		};};
		
		macro_rules! l_txt{($r_txt: expr) => {
			dstate.mv(row, pos.x as i32 + 2);
			dstate.renderer.addstr($r_txt);
			row += 1;
		};};
		
		macro_rules! r_txt{($r_txt: expr) => {
			dstate.mv(row, pos.x as i32 + (window_sz.w - $r_txt.len() - 2) as i32);
			dstate.renderer.addstr($r_txt);
			row += 1;
		};};
		
		// esc to close
		{
			dstate.mv(row, pos.x as i32 + 2); row += 2;
			dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
		}
		
		// 'You have discovered ...'
		{
			dstate.attron(COLOR_PAIR(CGREEN));
			ctr_txt!(&format!("You have discovered {}!", tech_discov.nm[dstate.local.lang_ind]));
			dstate.attroff(COLOR_PAIR(CGREEN));
			row += 1;
		}
		
		let mut intro_printed = false;
		
		// if first list print "This technology allows us to"
		macro_rules! print_intro_txt{
			($txt:expr) => {
				if intro_printed {
					l_txt!(&format!("...and {}:", $txt));
				}else{
					l_txt!(&format!("You can now {}:", $txt));
					intro_printed = true;
				}
			};
			($txt:expr, $flag:expr) => {
				if intro_printed {
					l_txt!(&format!("...and {}:", $txt));
				}else{
					l_txt!(&format!("You can now {}:", $txt));
				}
		};};
		
		if units_discov.len() != 0 {
			print_intro_txt!(&dstate.local.create);
			for unit_discov in units_discov.iter() {
				r_txt!(unit_discov);
			}
		}
		
		if bldgs_discov.len() != 0 {
			print_intro_txt!(&dstate.local.build);
			for bldg_discov in bldgs_discov.iter() {
				r_txt!(&bldg_discov);
			}
		}
		
		if resources_discov.len() != 0 {
			print_intro_txt!(&dstate.local.locate_and_use, true);
			for resource_discov in resources_discov.iter() {
				r_txt!(resource_discov);
			}
		}
		
		UIModeControl::UnChgd
	}
}

