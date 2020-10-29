use super::*;
use crate::units::*;
use std::convert::TryFrom;

pub struct GetTextWindowState {
	pub txt: String,
	pub curs_col: isize,
	pub txt_type: TxtType
}

pub enum TxtType {
	BrigadeNm,
	SectorNm
}

impl GetTextWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let w = min(40, dstate.iface_settings.screen_sz.w);
		let h = 7;
		
		let w_pos = dstate.print_window(ScreenSz{w,h, sz:0});
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		let title_c = Some(COLOR_PAIR(TITLE_COLOR));
		let buttons = &mut dstate.buttons;
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 1;
		
		let w = (w - 2) as i32;
		
		d.mv(y,x);
		match self.txt_type {
			TxtType::BrigadeNm => {center_txt(&l.Choose_a_name_for_the_brigade, w, title_c, d);}
			TxtType::SectorNm => {center_txt(&l.Choose_a_name_for_the_sector, w, title_c, d);}
		}
		
		// print entered txt
		d.mv(y+2,x+1);
		d.addstr(&self.txt);
					
		// instructions
		{
			let instructions_w = format!("{}  {}", dstate.buttons.Esc_to_close.print_txt(l), dstate.buttons.Confirm.print_txt(l)).len() as i32;
			let gap = ((w - instructions_w)/2) as i32;
			d.mv(y + 4, x - 1 + gap);
			dstate.buttons.Esc_to_close.print(None, l, d); d.addstr("  ");
			dstate.buttons.Confirm.print(None, l, d);
		}
		
		// mv to cursor location
		d.mv(y + 2, x + 1 + self.curs_col as i32);
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		if dstate.buttons.Confirm.activated(dstate.key_pressed, &dstate.mouse_event) {
			if self.txt.len() > 0 {
				let action_type = match self.txt_type {
					TxtType::BrigadeNm => {
						ActionType::BrigadeCreation {
							nm: self.txt.clone(),
							start_coord: None,
							end_coord: None
						}
					} TxtType::SectorNm => {
						ActionType::SectorCreation {
							nm: self.txt.clone(),
							creation_type: SectorCreationType::New,
							start_coord: None,
							end_coord: None
						}
					}
				};
				
				dstate.iface_settings.add_action_to = AddActionTo::NoUnit {
					action: ActionMeta::new(action_type),
				};
				
				return UIModeControl::Closed;
			}
		}
		
		match dstate.key_pressed {
			KEY_LEFT => {if self.curs_col != 0 {self.curs_col -= 1;}}
			KEY_RIGHT => {
				if self.curs_col < (self.txt.len() as isize) {
					self.curs_col += 1;
				}
			}
			
			KEY_HOME | KEY_UP => {self.curs_col = 0;}
			
			// end key
			KEY_DOWN | 0x166 | 0602 => {self.curs_col = self.txt.len() as isize;}
			
			// backspace
			KEY_BACKSPACE | 127 | 0x8  => {
				if self.curs_col != 0 {
					self.curs_col -= 1;
					self.txt.remove(self.curs_col as usize);
				}
			}
			
			// delete
			KEY_DC => {
				if self.curs_col != self.txt.len() as isize {
					self.txt.remove(self.curs_col as usize);
				}
			}
			_ => { // insert character
				if self.txt.len() < (min(MAX_SAVE_AS_W, dstate.iface_settings.screen_sz.w)-5) {
					if let Result::Ok(c) = u8::try_from(dstate.key_pressed) {
						if let Result::Ok(ch) = char::try_from(c) {
							self.txt.insert(self.curs_col as usize, ch);
							self.curs_col += 1;
						}
					}
				}
			}
		}
		UIModeControl::UnChgd
	}
}
