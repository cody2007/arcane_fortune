use super::*;
pub struct ResourcesDiscoveredWindowState {pub mode: usize}

////////////////////// resource locations discovered
impl ResourcesDiscoveredWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, map_data: &mut MapData, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let resource_opts = discovered_resources_list(pstats, cursor_coord, temps.resources, *map_data.map_szs.last().unwrap());
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Go_to_resource.clone(), resource_opts.clone(), None, None, 0, None);
		let row = list_pos.top_left.y as usize;
		
		// show info box
		if resource_opts.options.len() > 0 {
			if let ArgOptionUI::ResourceWCoord {rt, ..} = resource_opts.options[self.mode].arg {
				let w = 29 + 3;
				dstate.show_exemplar_info(rt.id as usize, EncyclopediaCategory::Resource, OffsetOrPos::Offset(w), None, OffsetOrPos::Pos(row + self.mode + 4), InfoLevel::Abbrev, temps, pstats);
			}else{panicq!("invalid UI setting");}
		}
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, map_data: &mut MapData, temps: &Templates<'bt,'ut,'rt,'dt,'_>, 
			pstats: &Stats, dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord(map_data);
		
		let list = discovered_resources_list(&pstats, cursor_coord,	temps.resources, *map_data.map_szs.last().unwrap());
		
		macro_rules! enter_action{($mode: expr) => {
			return UIModeControl::CloseAndGoTo(
				if let ArgOptionUI::ResourceWCoord {coord, ..} = list.options[$mode].arg
					{coord} else {panicq!("invalid UI setting");}
			);
		};}
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {	enter_action!(ind);}
		
		const SCROLL_FASTER_SPEED: usize = 3;
		
		let kbd = &dstate.kbd;
		match dstate.key_pressed {
			// down
			k if kbd.down_normal(k) => {
				if (self.mode + 1) <= (list.options.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if kbd.up_normal(k) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = list.options.len() - 1;
				}
			
			// down
			} k if k == kbd.fast_down as i32 => {
				if (self.mode + SCROLL_FASTER_SPEED) <= (list.options.len()-1) {
					self.mode += SCROLL_FASTER_SPEED;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if k == kbd.fast_up as i32 => {
				if self.mode >= SCROLL_FASTER_SPEED {
					self.mode -= SCROLL_FASTER_SPEED;
				}else{
					self.mode = list.options.len() - 1;
				}
			
			// enter
			} k if k == kbd.enter => {
				enter_action!(self.mode);
			} _ => {}
		}
		
		UIModeControl::UnChgd
	}
}
