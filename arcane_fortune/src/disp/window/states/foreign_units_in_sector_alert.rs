use super::*;
pub struct ForeignUnitInSectorAlertState {
	pub sector_nm: String,
	pub battalion_nm: String
}

///////////////////////// foreign units in sector alert
impl ForeignUnitInSectorAlertState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let txt = dstate.local.The_X_Battalion_reports_activity.replace("[battalion_nm]", &self.battalion_nm)
									  .replace("[sector_nm]", &self.sector_nm);
		
		let w = txt.len() + 4;
		let w_pos = dstate.print_window(ScreenSz{w, h: 2+1+3, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let mut row = 0;
		let d = &mut dstate.renderer;
		macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
					     ($final: expr) => {d.mv(row + y, x);}};
		
		mvl!(); dstate.buttons.Esc_to_close.print(None, &dstate.local, d);
		mvl!();mvl!(1);
		d.addstr(&txt);
		UIModeControl::UnChgd
	}
}

