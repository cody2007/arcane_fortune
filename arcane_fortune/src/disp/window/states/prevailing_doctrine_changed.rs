use super::*;
pub struct PrevailingDoctrineChangedWindowState {}

impl PrevailingDoctrineChangedWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let doc = players[dstate.iface_settings.cur_player as usize].stats.doctrine_template;
		let title = dstate.local.Adopted_doctrine.replace("[]", &doc.nm[dstate.local.lang_ind]);
		let lens = vec![title.len(), dstate.local.doctrine_changed_line1.len(), dstate.local.doctrine_changed_line2.len()];
		let max_len = *lens.iter().max().unwrap();
		let window_sz = ScreenSz {h: 8, w: max_len + 4, sz: 0};
		let pos = dstate.print_window(window_sz);
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		
		// title
		{
			d.mv(pos.y as i32 + 1, pos.x as i32 + ((window_sz.w - title.len())/2) as i32);
			addstr_c(&title, TITLE_COLOR, d);
		}
		
		d.mv(pos.y as i32 + 3, pos.x as i32 + 2);
		d.addstr(&l.doctrine_changed_line1);
		
		d.mv(pos.y as i32 + 4, pos.x as i32 + 2);
		d.addstr(&l.doctrine_changed_line2);
		
		// esc to close window
		{
			let button = &mut dstate.buttons.Esc_to_close;
			d.mv(pos.y as i32 + 6, pos.x as i32 + ((window_sz.w - button.print_txt(l).len()) / 2) as i32);
			button.print(None, l, d);
		}
		
		UIModeControl::UnChgd
	}
}
