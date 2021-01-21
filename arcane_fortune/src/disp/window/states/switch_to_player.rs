use super::*;
pub struct SwitchToPlayerWindowState {pub mode: usize}

impl SwitchToPlayerWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let all_civs = OptionsUI::all_civs_and_nobility(players);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_civilization.clone(), all_civs, None, None, 0, Some(players));
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list = OptionsUI::all_civs_and_nobility(players);
		if list_mode_update_and_action(&mut self.mode, list.options.len(), dstate) {
			if let ArgOptionUI::OwnerInd(owner_ind) = list.options[self.mode].arg {
				dstate.iface_settings.cur_player = owner_ind as SmSvType;
				
			}else{panicq!("invalid UI setting");}
			
			dstate.iface_settings.unit_subsel = 0;
			dstate.iface_settings.add_action_to = AddActionTo::None;
			
			let pstats = &mut players[dstate.iface_settings.cur_player as usize].stats;
			dstate.production_options = init_bldg_prod_windows(temps.bldgs, pstats, &dstate.local);
			dstate.update_menu_indicators(dstate.iface_settings.cur_player_paused(players));
			compute_zoomed_out_discoveries(map_data, &mut players[dstate.iface_settings.cur_player as usize].stats);
			
			return UIModeControl::Closed;
		}
		UIModeControl::UnChgd
	}
}

// for creating list of all players
impl <'bt,'ut,'rt,'dt>OptionsUI<'bt,'ut,'rt,'dt> {
	pub fn all_civs_and_nobility(players: &Vec<Player>) -> Self {
		let mut nms_string = Vec::with_capacity(players.len());
		let mut owner_ids = Vec::with_capacity(players.len());
		
		for (owner_id, player) in players.iter().enumerate() {
			match player.ptype {
				PlayerType::Barbarian(_) => {}
				PlayerType::Nobility(_) | PlayerType::Empire(_) | PlayerType::Human(_) => {
					nms_string.push(player.personalization.nm.clone());
					owner_ids.push(owner_id);
				}
			}
		}
		
		// register_shortcuts takes [&str]s, so take references of all the strings
		let mut nms = Vec::with_capacity(nms_string.len());
		
		for nm_string in nms_string.iter() {
			nms.push(nm_string.as_str());
		}
		
		let mut civs = OptionsUI::new(&nms);
		
		// associate owner_id w/ each menu entry
		for (i, opt) in civs.options.iter_mut().enumerate() {
			opt.arg = ArgOptionUI::OwnerInd(owner_ids[i]);
		}

		civs
	}
}

