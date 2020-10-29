use super::*;

pub struct RiotingAlertState {pub city_nm: String}

///////////////////////// rioting alert
impl RiotingAlertState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let txt = dstate.local.Rioting_has_broken_out.replace("[city_nm]", &self.city_nm);
		let w = txt.len() + 4;
		let w_pos = dstate.print_window(ScreenSz{w, h: 2+4, sz:0});
		let d = &mut dstate.renderer;
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let mut row = 0;
		macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
					     ($final: expr) => {d.mv(row + y, x);}};
		
		mvl!(); dstate.buttons.Esc_to_close.print(None, &dstate.local, d);
		mvl!(); mvl!(1);
		d.addstr(&txt);
		UIModeControl::UnChgd
	}
}

