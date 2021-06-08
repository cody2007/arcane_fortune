use super::*;
pub struct PublicPollingWindowState {}

///////////////////////// public polling
impl PublicPollingWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let contrib = &pstats.locally_logged.contrib;
		let pos_sum = contrib.doctrine + contrib.pacifism;
		let neg_sum = contrib.health + contrib.unemployment + contrib.crime;
		
		///// debug
		/*printlnq!("doctrine {} pacifism {}", contrib.doctrine/pos_sum, contrib.pacifism/pos_sum);
		printlnq!("health {} unemployment {} crime {}", contrib.health/neg_sum, contrib.unemployment/neg_sum, contrib.crime/neg_sum);
		printlnq!("{} {}", pos_sum, neg_sum);*/
		//////
				
		let w = 90;
		let n_lines = 30;
		
		let w_pos = dstate.print_window(ScreenSz{w, h: n_lines, sz:0});
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let mut row = 0;
		macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;};
					     ($final: expr) => {dstate.mv(row + y, x);}}
		
		// print key instructions
		mvl!(); dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
		
		{ // print title
			dstate.mv(row + y, x + ((w - dstate.local.Public_Polling.len())/2) as i32); row += 1;
			addstr_c(&dstate.local.Public_Polling, TITLE_COLOR, &mut dstate.renderer); mvl!();
		}
		
		const RADIUS: i32 = 8;
		let mut pos = Coord {y: (row + y + 1) as isize, x: (x + 2) as isize};
		
		macro_rules! legend{($y_off:expr, $entry:expr, $txt:expr) => {
			dstate.mv(pos.y as i32 + RADIUS*2 + 1 + $y_off, pos.x as i32);
			dstate.attron($entry.color);
			dstate.addch(dstate.chars.land_char);
			dstate.attroff($entry.color);
			dstate.renderer.addstr($txt);
			dstate.addstr(&format!(" ({}%)", ($entry.frac*100.).round() as usize));
		};}
		
		{
			dstate.mv(pos.y as i32 - 1, pos.x as i32);
			dstate.renderer.addstr(&dstate.local.What_are_you_most_satisfied_with);
			let ratios = vec![RatioEntry {frac: contrib.doctrine/pos_sum, color: COLOR_PAIR(CRED)},
						RatioEntry {frac: contrib.pacifism/pos_sum, color: COLOR_PAIR(CGREEN4)}];
			
			legend!(1, ratios[0], &dstate.local.doctrines_surrounding);
			if pstats.locally_logged.pacifism_sum >= 0. {
				legend!(2, ratios[1], &dstate.local.pacifism_surrounding);
			}else{
				legend!(2, ratios[1], &dstate.local.militarism_surrounding);
			}
			
			print_circle_plot(RADIUS, pos, &ratios, dstate);
		}
		
		////////////////
		{
			pos.x += (4*RADIUS + 15) as isize;
			dstate.mv(pos.y as i32 - 1, pos.x as i32);
			dstate.renderer.addstr(&dstate.local.What_are_you_least_satisfied_with);
			let ratios = vec![RatioEntry {frac: contrib.health/neg_sum, color: COLOR_PAIR(CRED)},
						RatioEntry {frac: contrib.unemployment/neg_sum, color: COLOR_PAIR(CGREEN4)},
						RatioEntry {frac: contrib.crime/neg_sum, color: COLOR_PAIR(CLOGO)}];
			
			print_circle_plot(RADIUS, pos, &ratios, dstate);
			
			pos.x += 5;
			legend!(1, ratios[0], &dstate.local.sickness_surrounding);
			legend!(2, ratios[1], &dstate.local.unemployment_surrounding);
			legend!(3, ratios[2], &dstate.local.crime_surrounding);
		}
		
		{ // fraction of greater to lesser pos or neg
			dstate.mv(pos.y as i32 + RADIUS*2 + 6, x + 2);
			let (frac, greater_label, lesser_label) = if contrib.pos_sum > (-contrib.neg_sum) {
				(-contrib.pos_sum/contrib.neg_sum, &dstate.local.positive, &dstate.local.negative)
			}else{
				(-contrib.neg_sum/contrib.pos_sum, &dstate.local.negative, &dstate.local.positive)
			};
			//printlnq!("{} {}", contrib.pos_sum, contrib.neg_sum);
			dstate.renderer.addstr(&dstate.local.on_average_responses.replace("[pos_neg1]", greater_label)
					.replace("[pos_neg2]", lesser_label)
					.replace("[frac]", &format!("{}", frac.round() as usize)));
		}
		UIModeControl::UnChgd
	}
}

