use super::*;
use crate::zones::HappinessCategory;
pub struct CitizenDemandAlertState {pub reason: HappinessCategory}
///////////////////////// citizen demand

impl CitizenDemandAlertState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let log_type = LogType::CitizenDemand {
				owner_id: dstate.iface_settings.cur_player as usize,
				reason: self.reason.clone()
			};
		let w = {
			let log_len = log_type.print(false, players, temps.doctrines, dstate);
			max(dstate.local.Advisor_caution.len(), log_len) + 4
		};
		let w_pos = dstate.print_window(ScreenSz{w, h: 2+4+3, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let mut row = 0;
		macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;};
					     ($final: expr) => {dstate.mv(row + y, x);}}
		
		mvl!(); dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
		mvl!(); mvl!();
		log_type.print(true, players, temps.doctrines, dstate);
		mvl!(); mvl!(1);
		dstate.renderer.addstr(&dstate.local.Advisor_caution); // update the width calculation above if this changes
		UIModeControl::UnChgd
	}
}
