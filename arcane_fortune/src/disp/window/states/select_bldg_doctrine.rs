use super::*;
use crate::units::*;

pub struct SelectBldgDoctrineState<'bt,'ut,'rt,'dt> {
	pub mode: usize,
	pub bldg_template: &'bt BldgTemplate<'ut,'rt,'dt>
}

////////////////////////// select doctrine dedication
impl <'bt,'ut,'rt,'dt>SelectBldgDoctrineState<'bt,'ut,'rt,'dt> {
	pub fn print(&self, pstats: &Stats, temps: &Templates, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = doctrines_available_list(&pstats, temps.doctrines, &dstate.local);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Dedicate_to.clone(), list.clone(), None, None, 0, None);
		
		// print details for selected bldg
		if let ArgOptionUI::DoctrineTemplate(Some(doc)) = list.options[self.mode].arg {
			dstate.show_exemplar_info(doc.id, EncyclopediaCategory::Doctrine, OffsetOrPos::Offset(26), Some(25), OffsetOrPos::Offset(self.mode+4), InfoLevel::Abbrev, temps, pstats);
		}else{panicq!("could not find doctrine template {}", self.mode);}
		
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys(&mut self, units: &Vec<Unit<'bt,'ut,'rt,'dt>>, 
			pstats: &Stats, map_data: &mut MapData, exf: &HashedMapEx, 
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) 
				-> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(unit_inds) = dstate.iface_settings.unit_inds_frm_sel(pstats, units, map_data, exf) {
			let list = doctrines_available_list(pstats, temps.doctrines, &dstate.local);
			macro_rules! enter_action{($mode:expr) => {
				if let ArgOptionUI::DoctrineTemplate(Some(doc)) = list.options[$mode].arg {
					let act = ActionType::WorkerBuildBldg {
							valid_placement: false,
							doctrine_dedication: Some(doc),
							template: self.bldg_template,
							bldg_coord: None 
					};
					dstate.iface_settings.start_build_mv_mode(act, &worker_inds(&unit_inds, units), units, map_data);
					
					return UIModeControl::Closed;
				}else{panicq!("invalid UI option argument");}
			};};
			if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
			
			match dstate.key_pressed {
				// down
				k if dstate.kbd.down(k) => {
					if (self.mode + 1) <= (list.options.len()-1) {
						self.mode += 1;
					}else{
						self.mode = 0;
					}
				
				// up
				} k if dstate.kbd.up(k) => {
					if self.mode > 0 {
						self.mode -= 1;
					}else{
						self.mode = list.options.len() - 1;
					}
					
				// enter
				} k if k == dstate.kbd.enter => {
					enter_action!(self.mode);
				} _ => {}
			}
			return UIModeControl::UnChgd;
		}
		UIModeControl::Closed
	}
}
