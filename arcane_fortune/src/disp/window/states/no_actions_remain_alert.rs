use super::*;
pub struct MvWithCursorNoActionsRemainAlertState {pub unit_ind: usize}

///////////////////////// no actions remain
impl MvWithCursorNoActionsRemainAlertState {
	pub fn print<'bt,'ut,'rt,'dt> (&self, units: &Vec<Unit>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt>  {
		let u = &units[self.unit_ind];
		let l = &dstate.local;
		let txt = l.No_actions_remain.replace("[battalion_nm]", &u.nm)
							.replace("[unit_type]", &u.template.nm[l.lang_ind]);
		let w = txt.len() + 4;
		let w_pos = dstate.print_window(ScreenSz{w, h: 2+4, sz:0});
		
		let l = &dstate.local;
		let d = &mut dstate.renderer;
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let mut row = 0;
		macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
					     ($final: expr) => {d.mv(row + y, x);}}
		
		mvl!(); dstate.buttons.Esc_to_close.print(None, l, d); mvl!(); mvl!();
		d.addstr(&txt); d.mv(row + y, x);
		UIModeControl::UnChgd
	}
}

