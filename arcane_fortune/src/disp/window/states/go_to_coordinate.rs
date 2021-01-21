use super::*;
use std::convert::TryFrom;

pub struct GoToCoordinateWindowState {
	pub coordinate: String,
	pub curs_col: isize
}

impl GoToCoordinateWindowState {
	pub fn new(map_data: &MapData, dstate: &mut DispState) -> Self {
		dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		
		let coordinate = dstate.iface_settings.cursor_to_map_string(map_data);
		Self {
			curs_col: coordinate.len() as isize,
			coordinate
		}
	}
	
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let w = min(30, dstate.iface_settings.screen_sz.w);
		let h = 7;
		
		let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 1;
		
		let w = (w - 2) as i32;
		
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		d.mv(y,x);
		center_txt(&l.Go_to_coordinate, w, Some(COLOR_PAIR(TITLE_COLOR)), d);
		
		// print current location
		d.mv(y+2,x+1);
		d.addstr(&self.coordinate);
					
		{ // instructions
			let instructions_w = "<Esc>: Cancel  <Enter>: Go".len() as i32;
			let gap = ((w - instructions_w)/2) as i32;
			d.mv(y + 4, x - 1 + gap);
			dstate.buttons.Esc_to_close.print(None, l, d);
			d.addstr("  ");
			d.attron(COLOR_PAIR(ESC_COLOR));
			d.addstr(&l.Enter_key);
			d.attroff(COLOR_PAIR(ESC_COLOR));
			d.addstr(&format!(": {}", l.Go));
		}
		
		// mv to cursor location
		d.mv(y + 2, x + 1 + self.curs_col as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, map_data: &mut MapData, dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// enter pressed
		if dstate.key_pressed == dstate.kbd.enter && self.coordinate.len() > 0 {
			let coordinates: Vec<&str> = self.coordinate.split(",").collect();
			if let Result::Ok(y) = coordinates[0].trim().parse() {
				if let Result::Ok(x) = coordinates[1].trim().parse() {
					let map_sz = map_data.map_szs[dstate.iface_settings.zoom_ind];
					if y < map_sz.h as isize {
						return UIModeControl::CloseAndGoTo(Coord {y, x}.to_ind(map_sz) as u64);
					}
				}
			}
		// any key except enter was pressed
		}else{
			do_txt_entry_keys!(dstate.key_pressed, self.curs_col, self.coordinate, Printable::Coordinate, dstate);
		}
		UIModeControl::UnChgd
	}
}

