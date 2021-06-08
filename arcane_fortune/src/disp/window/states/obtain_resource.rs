use super::*;
pub struct ObtainResourceWindowState {pub mode: usize}

impl ObtainResourceWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let resources = all_resources_list(temps.resources, &dstate.local);
				
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_resource.clone(), resources, None, None, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, pstats: &mut Stats<'bt,'ut,'rt,'dt>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = all_resources_list(temps.resources, &dstate.local);
		
		if list.list_mode_update_and_action(&mut self.mode, dstate) {
			if let ArgOptionUI::ResourceInd(resource_ind) = list.options[self.mode].arg {
				for tech_req in temps.resources[resource_ind].tech_req.iter() {
					pstats.force_discover_undiscov_tech((*tech_req) as SmSvType, temps, dstate);
				}
				
				pstats.resources_avail[resource_ind] += 1;
				dstate.production_options = init_bldg_prod_windows(temps.bldgs, pstats, &dstate.local);
			}else{panicq!("invalid UI setting");}
			
			return UIModeControl::Closed;
		}
		UIModeControl::UnChgd
	}
}

