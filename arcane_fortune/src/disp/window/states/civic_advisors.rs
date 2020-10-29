use super::*;
pub struct CivicAdvisorsWindowState {}

///////////////////////// civic advisors
impl CivicAdvisorsWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, player: &Player, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// Cultural advisor (doctrinality_sum)
		// Chief of Police (crime_sum)
		// Public Counsel of Foreign Affairs / Public Counsel of Peace (pacifism_sum)
		// Health advisor (health_sum)
		// Economic advisor (unemployment_sum)
		struct Advisor {
			nm: String,
			txt: String,
			grade: String
		}
		
		let l = &dstate.local;
		
		let advisors = { // Vec[(advisor_nm, advice)]
			//printlnq!("{:#?}", pstats.locally_logged.contrib);
			
			const N_LVLS: f32 = 5.;
			const SPACING: f32 = 1./N_LVLS;
			
			let advisors_advice_txts = vec![
				vec![&l.Cultural_Advisor_vlow, &l.Cultural_Advisor_low, &l.Cultural_Advisor_neutral, &l.Cultural_Advisor_high, &l.Cultural_Advisor_vhigh],
				vec![&l.Chief_of_Police_vlow, &l.Chief_of_Police_low, &l.Chief_of_Police_neutral, &l.Chief_of_Police_high, &l.Chief_of_Police_vhigh],
				vec![&l.Public_Counsel_of_FA_vlow, &l.Public_Counsel_of_FA_low, &l.Public_Counsel_of_FA_neutral, &l.Public_Counsel_of_FA_high, &l.Public_Counsel_of_FA_vhigh],
				vec![&l.Health_Advisor_vlow, &l.Health_Advisor_low, &l.Health_Advisor_neutral, &l.Health_Advisor_high, &l.Health_Advisor_vhigh],
				vec![&l.Economic_Advisor_vlow, &l.Economic_Advisor_low, &l.Economic_Advisor_neutral, &l.Economic_Advisor_high, &l.Economic_Advisor_vhigh]
			];
			
			let advice = |nm: &PersonName, title, mut val, sum, invert, advice_txts: &Vec<&String>| {
				if sum == 0. {
					val = 0.5;
				}else{
					val /= sum;
				}
				
				if invert {val = 1. - val;}
				
				let (txt, grade) = if val < SPACING {(advice_txts[0].clone(), "F")}
				else if val < (2.*SPACING) {(advice_txts[1].clone(), "C")}
				else if val < (3.*SPACING) {(advice_txts[2].clone(), "B")}
				else if val < (4.*SPACING) {(advice_txts[3].clone(), "A-")}
				else {(advice_txts[4].clone(), "A+")};
				
				Advisor {nm: format!("{}, {}:", nm.txt(), title), txt, grade: String::from(grade)}
			};
			
			let contrib = &player.stats.locally_logged.contrib;
			let pos_sum = contrib.doctrine + contrib.pacifism;
			let neg_sum = contrib.health + contrib.unemployment + contrib.crime;
			
			let militarism_pacifism = if player.stats.locally_logged.pacifism_sum < 0. {
				"miliatarism"
			}else{
				"pacifism"
			};
			
			let o = &player.personalization;
			
			let mut pacifism_advice = advice(&o.pacifism_advisor_nm, &dstate.local.Public_Counsel_of_Foreign_Affairs, contrib.pacifism, pos_sum, false, &advisors_advice_txts[2]);
			pacifism_advice.txt = pacifism_advice.txt.replace("[]", militarism_pacifism);
			
			vec![
				advice(&o.doctrine_advisor_nm, &l.Cultural_Advisor, contrib.doctrine, pos_sum, false, &advisors_advice_txts[0]),
				advice(&o.crime_advisor_nm, &l.Chief_of_Police, contrib.crime, neg_sum, true, &advisors_advice_txts[1]),
				pacifism_advice,
				advice(&o.health_advisor_nm, &l.Health_Advisor, contrib.health, neg_sum, true, &advisors_advice_txts[3]),
				advice(&o.unemployment_advisor_nm, &l.Economic_Advisor, contrib.unemployment, neg_sum, true, &advisors_advice_txts[4])
			]
		};
		
		let w = 70;
		
		let mut n_lines = 2 + 2;
		let mut wrapped_quotes = Vec::with_capacity(advisors.len());
		for advisor in advisors.iter() {
			let wrapped = wrap_txt(&advisor.txt, w - 4);
			n_lines += wrapped.len() + 3;
			wrapped_quotes.push(wrapped);
		}
		
		let w_pos = dstate.print_window(ScreenSz{w, h: n_lines, sz:0}); 
		
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		
		let mut row = 0;
		macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;};
					     ($final: expr) => {dstate.mv(row + y, x);}};
		
		// print key instructions
		mvl!(); dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
		
		{ // print title
			dstate.mv(row + y, x + ((w - dstate.local.Civic_Advisors.len())/2) as i32); row += 1;
			addstr_c(&dstate.local.Civic_Advisors, TITLE_COLOR, &mut dstate.renderer); mvl!();
		}
		
		for (advisor, wrapped_quote) in advisors.iter().zip(wrapped_quotes.iter()) {
			mvl!(); addstr_c(&advisor.nm, TITLE_COLOR, &mut dstate.renderer); mvl!();
			
			for quote_line in wrapped_quote.iter() {
				dstate.renderer.addstr(quote_line);
				mvl!();
			}
			
			{ // print grade
				let color = COLOR_PAIR(match advisor.grade.as_str() {
					"F" => CRED,
					"C" => CLOGO,
					"B" => CYELLOW,
					"A-" => CSAND1,
					"A+" => CGREEN,
					_ => {panicq!("unknown grade: {}", advisor.grade);}
				});
				
				dstate.renderer.addstr("Grade: ");
				addstr_attr(&format!("{}", advisor.grade), color, &mut dstate.renderer);
			}
			mvl!();
		}
		UIModeControl::UnChgd
	}
}

