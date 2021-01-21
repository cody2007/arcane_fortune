use super::*;

pub struct IntroNobilityJoinOptionsState<'bt,'ut,'rt,'dt> {
	pub mode: usize,
	pub candidate_houses: Vec<NobilityState<'bt,'ut,'rt,'dt>>
}

fn get_paragraph_lines(mut w: usize, l: &Localization) -> Vec<&str> {
	w -= 4;
	let mut lines = wrap_txt(&l.intro_add_nobility_p1, w);
	lines.append(&mut vec![""]);
	lines.append(&mut wrap_txt(&l.intro_add_nobility_p2, w));
	lines.append(&mut vec![""]);
	lines.append(&mut wrap_txt(&l.intro_add_nobility_p3, w));
	lines
}

impl <'bt,'ut,'rt,'dt>IntroNobilityJoinOptionsState<'bt,'ut,'rt,'dt> {
	pub fn new(player_coord: u64, players: &Vec<Player>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData<'rt>, exf: &HashedMapEx<'bt,'ut,'rt,'dt>,
			map_sz: MapSz, gstate: &mut GameState) -> Option<Self> {
		const N_CANDIDATE_HOUSES: usize = 3;
		let mut candidate_houses = Vec::with_capacity(N_CANDIDATE_HOUSES);
		
		const N_ATTEMPTS: usize = 100;
		for _ in 0..N_ATTEMPTS {
			if let Some(new_house_coord) = new_noble_house_coord_near_coord(player_coord, players, map_sz, &mut gstate.rng) {
				if let Some(mut nobility_state) = NobilityState::new(new_house_coord, temps, map_data, exf, map_sz, gstate) {
					nobility_state.house.has_req_to_join = true; // do not ask to join the empire later
					candidate_houses.push(nobility_state);
				}
			}
			if candidate_houses.len() == N_CANDIDATE_HOUSES {break;}
		}
		
		if candidate_houses.len() == 0 {return None;}
		
		Some(Self {
			mode: 0,
			candidate_houses
		})
	}
	
	pub fn print(&self, turn: usize, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let (options, label_txt_opt) = OptionsUI::candidate_houses(&self.candidate_houses, &dstate.local, turn);
		let w = options.max_strlen + 4;
		
		let n_gap_lines = get_paragraph_lines(w, &dstate.local).len();
		let list_pos = dstate.print_list_window(self.mode, String::new(), options, Some(w), label_txt_opt, n_gap_lines, None);
		
		{ // title
			dstate.renderer.mv(list_pos.top_left.y as i32 + 1, list_pos.top_left.x as i32 + 1);
			center_txt(&dstate.local.Natural_born_leaders, w as i32, Some(COLOR_PAIR(CGREEN)), &mut dstate.renderer);
		}
		
		// paragraph txt
		for (line_num, line) in get_paragraph_lines(w, &dstate.local).iter().enumerate() {
			dstate.renderer.mv(list_pos.top_left.y as i32 + line_num as i32 + 3, list_pos.top_left.x as i32 + 2);
			dstate.renderer.addstr(line);
		}
		
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &Vec<Bldg>, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			gstate: &mut GameState, temps: &Templates<'bt,'ut,'rt,'dt,'_>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = OptionsUI::candidate_houses(&self.candidate_houses, &dstate.local, gstate.turn).0;
		
		if list_mode_update_and_action(&mut self.mode, list.options.len(), dstate) {
			// add nobility
			if let Some(nobility_state) = self.candidate_houses.get(self.mode) {
				nobility_state.add_to_players(players, units, bldgs, map_data, exs, gstate, temps, 0);
				gstate.relations.join_as_fiefdom(players.len()-1, dstate.iface_settings.cur_player as usize, players, &mut gstate.logs, gstate.turn, &mut gstate.rng, dstate);
			}
			
			return UIModeControl::Closed;
		}
		
		UIModeControl::UnChgd
	}
}

impl <'bt,'ut,'rt,'dt>OptionsUI<'bt,'ut,'rt,'dt> {
	// returns OptionsUI and label_txt_opt
	fn candidate_houses(candidate_houses: &Vec<NobilityState<'bt,'ut,'rt,'dt>>, l: &Localization,
			turn: usize) -> (Self, Option<String>) {
		struct NobleTxt {
			name: String,
			age: String,
			personality: String,
			doctrine: String
		}
		
		// text for each nobility
		let noble_txts = {
			let mut noble_txts = Vec::with_capacity(candidate_houses.len());
			for candidate_house in candidate_houses.iter() {
				let head_noble = candidate_house.house.head_noble();
				let personality_tags = head_noble.personality.color_tags(l);
				
				noble_txts.push(NobleTxt {
					name: 
						head_noble.name.txt(),
					age: 
						l.age_gender
							.replace("[age]", &format!("{}", head_noble.age(turn).round()))
							.replace("[mf]", if head_noble.gender_female {&l.female} else {&l.male}),
					personality: 
						color_tags_txt(&l.personality_desc, &personality_tags),
					doctrine:
						candidate_house.ai_state.doctrine_txt(l)
				});
			}
			noble_txts
		};
		
		let max_name = noble_txts.iter().map(|nt| nt.name.len()).max().unwrap();
		let max_age = noble_txts.iter().map(|nt| nt.age.len()).max().unwrap();
		let mut max_personality = noble_txts.iter().map(|nt| nt.personality.len()).max().unwrap();
		let mut max_doctrine = noble_txts.iter().map(|nt| nt.doctrine.len()).max().unwrap();
		
		max_personality = max(max_personality, l.Known_to_be.len());
		max_doctrine = max(max_doctrine, l.Follower_of.len());
		
		let gap = |n| {
			let mut txt = String::with_capacity(n);
			for _ in 0..n {
				txt.push(' ');
			}
			txt
		};
		
		// space out columns to be equal across rows
		let mut txt_entries = Vec::with_capacity(candidate_houses.len());
		for noble_txt in noble_txts.iter() {
			txt_entries.push(format!("{}{}    {}{}    {}{}    {}{}",
				noble_txt.name, gap(max_name - noble_txt.name.len()),
				noble_txt.age, gap(max_age - noble_txt.age.len()),
				noble_txt.personality, gap(max_personality - noble_txt.personality.len()),
				gap(max_doctrine - noble_txt.doctrine.len()), noble_txt.doctrine
			));
		}
		
		// last entry for selecting no nobility
		const COL_GAP_LEN: usize = 4;
		txt_entries.push(format!("{}{}",
				gap((max_name + max_age + max_personality + max_doctrine + COL_GAP_LEN*3 - l.No_one_let_us_proceed.len())/2),
				l.No_one_let_us_proceed));
		
		// top row of labels
		let label_txt_opt = {
			if noble_txts.len() != 0 {
				Some(format!("{}    {}    {}{}    {}{}",
					gap(max_name),
					gap(max_age),
					l.Known_to_be, gap(max_personality - l.Known_to_be.len()),
					gap(max_doctrine - l.Follower_of.len()), l.Follower_of
				))
			}else {None}
		};
		
		// register_shortcuts takes [&str]s, so take references of all the strings
		let mut nms = Vec::with_capacity(txt_entries.len());
		for entry in txt_entries.iter() {
			nms.push(entry.as_str());
		}
		
		(OptionsUI::new(&nms), label_txt_opt)
	}
}
