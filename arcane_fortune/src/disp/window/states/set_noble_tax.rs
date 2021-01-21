// set the noble's tax dues to the parent empire
use super::*;

pub struct SetNobleTaxState {
	get_text_window_state: GetTextWindowState,
	noble_id: usize
}

impl SetNobleTaxState {
	pub fn new(noble_id: usize, relations: &Relations, dstate: &mut DispState) -> Self {
		dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
		let current_noble_tax_rate = relations.tax_rate(noble_id, dstate.iface_settings.cur_player as usize).unwrap();
		
		Self {
			get_text_window_state: 
				GetTextWindowState::new(
					TxtType::CustomPrintNm(
						dstate.local.Set_noble_tax_rate.clone()
					),
					current_noble_tax_rate.to_string()
				),
			noble_id
		}
	}
	
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		self.get_text_window_state.print(dstate)
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, relations: &mut Relations,
			dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if let Some(txt) = self.get_text_window_state.keys_ret_txt(dstate) {
			if let Result::Ok(tax_rate) = txt.parse() {
				if tax_rate <= 100 {
					let cur_player = dstate.iface_settings.cur_player as usize;
					relations.set_tax_rate(self.noble_id, cur_player, tax_rate);
					return UIModeControl::Closed;
				}
			}
		}
		UIModeControl::UnChgd
	}
}

