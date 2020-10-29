use super::*;
pub struct InitialGameWindowState {}

impl InitialGameWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let d = &mut dstate.renderer;
		let l = &dstate.local;
		const INTRO_TXT: &[&str] = &[
			"For as long as anyone can remember, the people of your tribe have subsisted on the land.",
			"Living and dying by what it has to offer. Recently, some members of the tribe have taken",
			"greater control of nature and have found that they can tend to, support and cultivate it,",
			"leading to small farms.",
			" ",
			"All creation is not without havoc. Your people have suffered increasingly",
			"organized attacks by local barbarian tribes and have taken it upon themselves",
			"to organize and mount a resistance to expand beyond and counter the forces of mankind",
			"and nature itself.",
			" ",
			"So, you have taken it upon yourself to lead this movement. What you make of it is",
			"entirely up to you."];
		
		let owner = &players[dstate.iface_settings.cur_player as usize];
		let mut window_w = format!("{} has been founded!", owner.personalization.nm).len() as i32;
		
		// determine max width of window
		{
			for txt in INTRO_TXT.iter() {
				if window_w < (txt.len() as i32) {window_w = txt.len() as i32;}
			}
			
			window_w += 2 + 2; // 2 for the |, 2 for the spaces
		}
		
		let window_h = INTRO_TXT.len() as i32 + 2 + 2 + 2; // 2 for the top and bottom lines, 2 for txt, 2 blank lns
		
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
		
		//////// print title: {} has been founded!
		{
			pl!();
			let txt_len = format!("{} has been founded!", owner.personalization.nm).len() as i32;
			for _ in 0..((window_w - txt_len)/2) {d.addch(' ');}
			print_civ_nm(owner, d);
			d.attron(COLOR_PAIR(CYELLOW));
			d.addstr(" has been founded!");
			d.attroff(COLOR_PAIR(CYELLOW));
			clr!();
		}
		
		bln!();
		
		////////// print intro text		
		for txt in INTRO_TXT.iter() {
			pl!();
			d.addstr(txt);
			clr!();
		}
					
		bln!();
		
		//////// esc
		{
			pl!();
			let button = &mut dstate.buttons.Esc_to_close;
			for _ in 0..((window_w - button.print_txt(l).len() as i32)/2) {d.addch(' ');}
			button.print(None, l, d);
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
