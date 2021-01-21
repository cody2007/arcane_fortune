use super::*;
use crate::units::*;
pub struct ProdListWindowState {pub mode: usize}

/////////// production by workers or buildings
impl ProdListWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, units: &Vec<Unit>, bldgs: &Vec<Bldg>,
			exf: &HashedMapEx, pstats: &Stats, temps: &Templates,
			map_data: &mut MapData, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		debug_assertq!(dstate.iface_settings.zoom_ind == map_data.max_zoom_ind());
		let w = 29;
		
		////////////////////// worker producing bldg
		if dstate.iface_settings.unit_inds_frm_sel(pstats, units, map_data, exf) != None {
			let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_production.clone(), dstate.production_options.worker.clone(), Some(w), None, 0, None);
			
			// print details for selected bldg
			if let ArgOptionUI::BldgTemplate(Some(bt)) = dstate.production_options.worker.options[self.mode].arg {
				let template_ind = temps.bldgs.iter().position(|r| r == bt).unwrap();
				dstate.show_exemplar_info(template_ind, EncyclopediaCategory::Bldg, OffsetOrPos::Offset(w), None, OffsetOrPos::Offset(self.mode-4), InfoLevel::Abbrev, temps, pstats);
			// zeroth entry only should be none
			}else if self.mode != 0 {
				panicq!("could not find building template {}", self.mode);
			}
			dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);

		////////////////////// building producing unit
		}else if let Some(bldg_ind) = dstate.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
			let b = &bldgs[bldg_ind];
			
			// look-up menu listings for the selected bldg:
			if let Some(options) = &dstate.production_options.bldgs[b.template.id as usize] {
				let options = options.clone();
				let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_production.clone(), options, None, None, 0, None);
				
				// print details for selected bldg
				if let ArgOptionUI::UnitTemplate(Some(ut)) = dstate.production_options.bldgs[b.template.id as usize].as_ref().unwrap().options[self.mode].arg {
					let template_ind = temps.units.iter().position(|r| r == ut).unwrap();
					dstate.show_exemplar_info(template_ind, EncyclopediaCategory::Unit, OffsetOrPos::Offset(w), None, OffsetOrPos::Offset(self.mode+4), InfoLevel::Abbrev, temps, pstats);
				// zeroth entry only should be none
				}else if self.mode != 0 {
					panicq!("could not find unit template {}", self.mode);
				}
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);

			}else {panicq!("could not find building production options");}

		}else{
			return UIModeControl::Closed;
		}//panicq!("window active but no printing available");}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, units: &Vec<Unit>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			pstats: &Stats, map_data: &mut MapData, exf: &HashedMapEx, 
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		
		// set bldg worker produces
		if let Some(unit_inds) = dstate.iface_settings.unit_inds_frm_sel(pstats, units, map_data, exf) {
			debug_assertq!(dstate.iface_settings.zoom_ind == map_data.max_zoom_ind());
			//debug_assertq!(units[unit_ind].template.nm[0] == WORKER_NM);
			
			dstate.production_options = init_bldg_prod_windows(&temps.bldgs, pstats, &dstate.local);
			
			let opt = &dstate.production_options.worker; // get production options for worker
			
			macro_rules! set_production {($ind: expr) => (
				// start production
				if $ind != 0 {
					if let ArgOptionUI::BldgTemplate(Some(bt)) = &opt.options[$ind].arg {
						// choose doctrine to dedicate building?
						if bt.doctrinality_bonus > 0. {
							return UIModeControl::New(UIMode::SelectBldgDoctrine(SelectBldgDoctrineState {
								mode: 0,
								bldg_template: bt
							}));
						}else{
							let act = ActionType::WorkerBuildBldg {
									valid_placement: false,
									doctrine_dedication: None,
									template: bt,
									bldg_coord: None 
							};
							dstate.iface_settings.start_build_mv_mode(act, &worker_inds(&unit_inds, units), units, map_data);
						}
					}else{panicq!("Option argument not set");}
				}
				return UIModeControl::Closed;
			);};
			
			// shortcut key pressed?
			for (new_menu_ind, option) in opt.options.iter().enumerate() {
				// match found
				if option.key == Some(dstate.key_pressed as u8 as char) {
					set_production!(new_menu_ind);
				}
			}
			if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {set_production!(ind);}

			// generic keys
			match dstate.key_pressed {
				// down
				k if dstate.kbd.down(k) => {
					if (self.mode + 1) <= (opt.options.len()-1) {
						self.mode += 1;
					}else{
						self.mode = 0;
					}
					
				// up
				} k if dstate.kbd.up(k) => {
					if self.mode > 0 {
						self.mode -= 1;
					}else{
						self.mode = opt.options.len() - 1;
					}
					
				// enter
				} k if k == dstate.kbd.enter => {
					set_production!(self.mode);
				} _ => {}
			} // end key match
			
			return UIModeControl::UnChgd;
		
		// select unit bldg produces
		}else if let Some(bldg_ind) = dstate.iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
			debug_assertq!(dstate.iface_settings.zoom_ind == map_data.max_zoom_ind());
			let b = &mut bldgs[bldg_ind];
			dstate.production_options = init_bldg_prod_windows(&temps.bldgs, pstats, &dstate.local);
			
			if let Some(opt) = &dstate.production_options.bldgs[b.template.id as usize] { // get production options for current bldg
				macro_rules! set_production {($ind: expr) => (
					match b.args { // unwrap bldg arguments:
						BldgArgs::PopulationCenter {ref mut production, ..} |
						BldgArgs::GenericProducable {ref mut production} => {	
							// start production
							if $ind != 0 {
								if let ArgOptionUI::UnitTemplate(Some(ut)) = &opt.options[$ind].arg {
									production.push(ProductionEntry {
										production: ut,
										progress: 0
									});
								}	
							}/*else{
								*production_progress = None;
								*production = None;
							}*/
						} BldgArgs::None | BldgArgs::PublicEvent {..} => {panicq!("bldg arguments do not store production");}	
					}
					return UIModeControl::Closed;
				);};
				if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	set_production!(ind);}
				
				// shortcut key pressed?
				for (new_menu_ind, option) in opt.options.iter().enumerate() {
					// match found
					if option.key == Some(dstate.key_pressed as u8 as char) {
						set_production!(new_menu_ind);
					}
				}
				
				// generic keys
				match dstate.key_pressed {
					// down
					k if dstate.kbd.down(k) => {
						if (self.mode + 1) <= (opt.options.len()-1) {
							self.mode += 1;
						}else{
							self.mode = 0;
						}
						
					// up
					} k if dstate.kbd.up(k) => {
						if self.mode > 0 {
							self.mode -= 1;
						}else{
							self.mode = opt.options.len() - 1;
						}
						
					// enter
					} k if k == dstate.kbd.enter => {
						set_production!(self.mode);
					} _ => {}
				}
				return UIModeControl::UnChgd;
			} // unwrapping of production_options for selected bldg
		}
		UIModeControl::Closed
	}
}

