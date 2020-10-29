use super::*;
pub struct EndGameWindowState {}

impl EndGameWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let d = &mut dstate.renderer;
		let owner = &players[dstate.iface_settings.cur_player as usize];
		let window_w = (max(format!("Watch as {} becomes an arcane footnote", owner.personalization.nm).len(),
					 "civilizations, you can now explore the remaining world.".len())
					+ 2 + 2) as i32; // 2 for the |, 2 for the spaces
		let window_h = 9;
		
		let mut row = (dstate.iface_settings.screen_sz.h as i32 - window_h)/2;
		let col = (dstate.iface_settings.screen_sz.w as i32 - window_w)/2;
		
		let mut y = 0;
		let mut x = 0;
		
		macro_rules! bln{() => {
			d.mv(row, col); row += 1;
			d.addch(dstate.chars.vline_char);
			for _ in 0..(window_w-2) {d.addch(' ');}
			d.addch(dstate.chars.vline_char);
		};};
		
		macro_rules! pl{() => {
			d.mv(row, col); row += 1;
			d.addch(dstate.chars.vline_char);
			d.addch(' ');
		};};
		
		macro_rules! pr{() => {d.addch(' '); d.addch(dstate.chars.vline_char);};};
		
		// clear to end of line
		macro_rules! clr{() => {
			d.getyx(stdscr(), &mut y, &mut x);
			for _ in x..(col + window_w-2) {d.addch(' ');}
			pr!();
		};};
		
		/////// top ln
		{
			d.mv(row, col); row += 1;
			d.addch(dstate.chars.ulcorner_char);
			for _ in 0..(window_w-2) {d.addch(dstate.chars.hline_char);}
			d.addch(dstate.chars.urcorner_char);
		}
		
		//////// print title: {} has been destroyed!
		{
			pl!();
			let txt_len = format!("{} has been destroyed!", owner.personalization.nm).len() as i32;
			for _ in 0..((window_w - txt_len)/2) {d.addch(' ');}
			print_civ_nm(owner, d);
			d.attron(COLOR_PAIR(CYELLOW));
			d.addstr(" has been destroyed!");
			d.attroff(COLOR_PAIR(CYELLOW));
			clr!();
		}
		
		bln!();
		
		pl!();
		d.addstr("While you no longer have a place here in the realm of");
		clr!();
		
		pl!();
		d.addstr("civilizations, you can now explore the remaining world.");
		clr!();
		
		bln!();
		
		pl!();
		d.addstr("Watch as ");
		print_civ_nm(owner, d);
		d.addstr(" becomes an arcane footnote");
		clr!();
		
		pl!();
		d.addstr("in the historical abyss.");
		clr!();
		
		bln!();
		
		//////// esc
		{
			pl!();
			let button = &mut dstate.buttons.Esc_to_close;
			for _ in 0..((window_w - button.print_txt(&dstate.local).len() as i32)/2) {d.addch(' ');}
			button.print(None, &dstate.local, d);
			clr!();
		}
		
		////// bottom ln
		{
			d.mv(row, col);
			d.addch(dstate.chars.llcorner_char);
			for _ in 0..(window_w-2) {d.addch(dstate.chars.hline_char);}
			d.addch(dstate.chars.lrcorner_char);
		}
		
		UIModeControl::UnChgd
	}
}

