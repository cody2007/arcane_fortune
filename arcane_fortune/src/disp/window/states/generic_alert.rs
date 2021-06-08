use super::*;
pub struct GenericAlertState {pub txt: String}

///////////////////////// generic alert

impl GenericAlertState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const MAX_W: usize = 70;
		let wrapped_txt = wrap_txt(&self.txt, MAX_W);
		let w = min(MAX_W, self.txt.len()) + 4;
		let h = wrapped_txt.len() + 4;
		let w_pos = dstate.print_window(ScreenSz{w, h, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		let l = &dstate.local;
		let d = &mut dstate.renderer;
		
		let mut row = 0;
		macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
					     ($final: expr) => {d.mv(row + y, x);}}
		
		mvl!(); dstate.buttons.Esc_to_close.print(None, l, d); mvl!();
		
		for line in wrapped_txt.iter() {
			mvl!();
			d.addstr(line);
		}
		UIModeControl::UnChgd
	}
}
