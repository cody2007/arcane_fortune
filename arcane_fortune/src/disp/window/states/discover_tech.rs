use super::*;
pub struct DiscoverTechWindowState {pub mode: usize}

impl DiscoverTechWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let techs = undiscovered_tech_list(&pstats, temps.techs, &dstate.local);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_technology.clone(), techs, None, None, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, pstats: &mut Stats, temps: &Templates<'bt,'ut,'rt,'dt,'_>, 
			dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = undiscovered_tech_list(&pstats, temps.techs, &dstate.local);
		macro_rules! enter_action{($mode: expr) => {
			if let ArgOptionUI::TechInd(tech_ind) = list.options[$mode].arg {
				pstats.force_discover_undiscov_tech(tech_ind as SmSvType, temps, dstate);
			}else{panicq!("invalid UI setting");}
			
			return UIModeControl::Closed;
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
		UIModeControl::UnChgd
	}
}
