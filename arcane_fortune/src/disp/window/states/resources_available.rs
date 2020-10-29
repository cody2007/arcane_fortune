use super::*;
pub struct ResourcesAvailableWindowState {}

////////////// show resources available
impl ResourcesAvailableWindowState {
	pub fn print<'bt,'ut,'rt,'dt> (&self, pstats: &Stats, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let n_resources_avail = pstats.resources_avail.iter().filter(|&&r| r != 0).count();
		let w = 30;
		let window_sz = if n_resources_avail != 0 {
			ScreenSz{w, h: n_resources_avail + 6, sz: 0}
		}else{
			ScreenSz{w, h: 7, sz: 0}
		};
		let w_pos = dstate.print_window(window_sz);
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 1;
		
		let w = (w - 2) as i32;
		
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		d.mv(y,x);
		center_txt(&l.Resources_available, w, Some(COLOR_PAIR(TITLE_COLOR)), d);
		
		// display resources
		{
			if n_resources_avail == 0 {
				d.mv(y + 2, x);
				center_txt(&l.None, w, None, d);
			}else{
				let mut n_shown = 0;
				for (avail, resource) in pstats.resources_avail.iter().zip(temps.resources) {
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
			let button = &mut dstate.buttons.Esc_to_close;
			let instructions_w = button.print_txt(l).len() as i32;
			let gap = ((w - instructions_w)/2) as i32;
			if n_resources_avail != 0 {
				d.mv(y + 3 + n_resources_avail as i32, x - 1 + gap);
			}else{
				d.mv(y + 4, x - 1 + gap);
			}
			button.print(None, l, d);
		}
		UIModeControl::UnChgd
	}
}
