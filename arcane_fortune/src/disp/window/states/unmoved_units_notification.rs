use super::*;
pub struct UnmovedUnitsNotificationState {}

///////////////////////// unmoved units notification
impl UnmovedUnitsNotificationState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let w = dstate.local.Assign_actions_to_them.len() + 4;
		let w_pos = dstate.print_window(ScreenSz{w, h: 5+3, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let w = (w - 2) as i32;
		
		let mut row = 0;
		let l = &dstate.local;
		let d = &mut dstate.renderer;
		macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
					     ($final: expr) => {d.mv(row + y, x);}};
		
		mvl!(); dstate.buttons.Esc_to_close.print(None, l, d); mvl!();
		center_txt(&l.You_have_unmoved_units, w, Some(COLOR_PAIR(TITLE_COLOR)), d);
		mvl!();mvl!();
		d.addstr(&l.Assign_actions_to_them); d.mv(row + y, x);
		center_txt(&l.Fortify_them_if, w, None, d);
		UIModeControl::UnChgd
	}
}
