use super::*;
pub struct CivilizationIntelWindowState {
	pub mode: usize,
	pub selection_phase: bool
}

////////////////////////// civilization intel
impl CivilizationIntelWindowState {
	pub fn print<'bt,'ut,'rt,'dt> (&self, players: &Vec<Player>, gstate: &GameState, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt>  {
		let contacted_civs = contacted_civilizations_list(gstate, players, dstate.iface_settings.cur_player, &dstate.local);
		
		// select civilization
		if self.selection_phase {
			let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_civilization.clone(), contacted_civs, None, None, 0, Some(players));
			dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		// show information for civilization
		}else{
			let owner_id = if let ArgOptionUI::OwnerInd(owner_id) = contacted_civs.options[self.mode].arg {
				owner_id }else{ panicq!("owner id not in menu options"); };
			
			let player = &players[owner_id];
			let o = player;
			let motto_txt = format!("{} \"{}\"", dstate.local.National_motto, o.personalization.motto);
			let w = max("Our intelligence tells us this is an aggressive and mythological culture.".len(),
					motto_txt.len()) + 4;
			let w_pos = dstate.print_window(ScreenSz{w, h: 8+3, sz:0});
			
			let y = w_pos.y as i32 + 1;
			let x = w_pos.x as i32 + 2;
			
			let w = (w - 2) as i32;
			
			let mut row = 0;
			let d = &mut dstate.renderer;
			let l = &dstate.local;
			macro_rules! mvl{() => {d.mv(row + y, x); row += 1;};
						     ($final: expr) => {d.mv(row + y, x);}};
			
			// print key instructions
			{
				mvl!();
				dstate.buttons.Esc_to_close.print(None, l, d);
				
				mvl!();
				addstr_c(&l.Left_arrow, ESC_COLOR, d);
				d.addch(' ');
				d.addstr(&l.to_go_back);
			}
			
			///////////// title -- country name
			{
				mvl!();
				let txt_len = format!("Our Intel on {}", o.personalization.nm).len();
				let g = (w as usize - txt_len) / 2;
				let mut sp = String::with_capacity(g);
				for _ in 0..g {sp.push(' ');}
				
				d.addstr(&sp);
				d.addstr("Our Intel on ");
				
				set_player_color(o, true, d);
				d.addstr(&o.personalization.nm);
				set_player_color(o, false, d);
				
				mvl!();
			}
			
			//////////// ruler
			mvl!();
			d.addstr(&format!("{} {} {}", l.Ruler, o.personalization.ruler_nm.first, o.personalization.ruler_nm.last));
			
			////////////// motto
			mvl!();
			d.addstr(&motto_txt);
			
			////////// prevailing doctrine
			{
				mvl!();
				let pstats = &players[owner_id].stats;
				d.addstr(&l.Prevailing_doctrine);
				if pstats.doctrine_template.id == 0 {
					d.addstr(&format!(" {}", l.None));
				}else{
					d.addstr(&format!(" {}", pstats.doctrine_template.nm[l.lang_ind]));
				}
			}
			
			//////////// personality
			if let PlayerType::Empire(EmpireState {personality, ..}) = &player.ptype {
				mvl!();mvl!(true);
				
				color_tags_print(&dstate.local.Our_intelligence_tells_us,
					&personality.color_tags(&dstate.local), None, &mut dstate.renderer);
			}
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &GameState, players: &Vec<Player>,
			dstate: &DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// civilization selection window
		if self.selection_phase {
			let list = contacted_civilizations_list(gstate, players, dstate.iface_settings.cur_player, &dstate.local);
			if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {
				self.mode = ind;
				self.selection_phase = false;
				return UIModeControl::UnChgd;
			}
			
			match dstate.key_pressed {
				// down
				k if dstate.kbd.down(k) => {
					if (self.mode + 1) <= (list.options.len()-1) {
						self.mode += 1;
					}else{
						self.mode = 0;
					}
				
				// up
				} k if dstate.kbd.up(k) => {
					if self.mode > 0 {
						self.mode -= 1;
					}else{
						self.mode = list.options.len() - 1;
					}
					
				// enter
				} k if k == dstate.kbd.enter => {
					self.selection_phase = false;
					
				} _ => {}
			}
		// display civ info
		}else{
			if dstate.key_pressed == KEY_LEFT {
				self.selection_phase = true; // go back to selection page
			}
		}
		UIModeControl::UnChgd
	}
}
			
