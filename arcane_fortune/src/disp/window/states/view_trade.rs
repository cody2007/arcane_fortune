// for canceling and viewing current trades
use super::*;

pub struct ViewTradeState {
	player_id: usize, // between this player and dstate.iface_settings.cur_player we show the trades
	pub mode: usize, // index of trade to view or show
	view: bool // true: show the trade details, false: select this trade among others in a list
}

impl ViewTradeState {
	// player_id = player to trade with
	pub fn new(player_id: usize) -> Self {
		Self {
			player_id,
			mode: 0,
			view: false
		}
	}
	
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, gstate: &GameState, temps: &Templates,
			players: &Vec<Player>, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// show specific trade deal
		if self.view {
			let cur_player = dstate.iface_settings.cur_player as usize;
			
			if let Some(trade_deals) = gstate.relations.status(self.player_id, cur_player).trade_deals() {
				if let Some(trade_deal) = trade_deals.get(self.mode) {
					let expires = trade_deal.expires(gstate.turn);
					let h = 10 + TRADE_LIST_HEIGHT as usize;
					let w_pos = dstate.print_window(ScreenSz{w: TRADE_LIST_WIDTH as usize, h, sz:0});
					
					let mut row = w_pos.y + 1;
					
					dstate.mv(row, w_pos.x + 1);
					center_txt(&trade_deal.nm(&dstate.local), TRADE_LIST_WIDTH, Some(COLOR_PAIR(TITLE_COLOR)), &mut dstate.renderer);
					row += 2;
					
					trade_deal.print(row, w_pos.x, self.player_id, temps, players, dstate);
					row += 1;
					
					// expiration of trade
					if let Some(expires) = expires {
						dstate.mv(row + TRADE_LIST_HEIGHT, w_pos.x + 1);
						let cancel_txt = dstate.local.Trade_cannot_be_canceled_until.replace("[]", &dstate.local.date_str(expires));
						center_txt(&cancel_txt, TRADE_LIST_WIDTH, None, &mut dstate.renderer);
					}
					
					{ // print key instructions
						let mut row = (w_pos.y + TRADE_LIST_HEIGHT + 6) as i32;
						dstate.buttons.to_go_back.print_centered(row, &dstate.iface_settings, &dstate.local, &mut dstate.renderer);
						if expires.is_none() {
							dstate.buttons.Cancel_trade.print_centered(row + 1, &dstate.iface_settings, &dstate.local, &mut dstate.renderer);
							row += 1;
						}
						dstate.buttons.Esc_to_close.print_centered(row + 1, &dstate.iface_settings, &dstate.local, &mut dstate.renderer);
					}
					
					return UIModeControl::UnChgd;
				}
			}
			// trade deal not found
			self.view = false;
	
		// list current trade deals
		}else{
			let entries = OptionsUI::current_trade_deals(self.player_id, &gstate.relations, dstate);
			let w = 35;
			let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_a_trade_to_view.clone(), entries, Some(w), None, 0, None);
			dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32); // text mode cursor for screen readers
		}
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, gstate: &mut GameState, players: &mut Vec<Player>,
			dstate: &DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		// show specific trade deal
		if self.view {
			if dstate.buttons.to_go_back.activated(dstate.key_pressed, &dstate.mouse_event) {self.view = false;}
			
			// cancel trade
			if dstate.buttons.Cancel_trade.activated(dstate.key_pressed, &dstate.mouse_event) {
				let cur_player = dstate.iface_settings.cur_player as usize;
				gstate.relations.rm_trade(self.mode, self.player_id, cur_player, players, gstate.turn);
				self.mode = 0;
				self.view = false;
			}
		// list current trade deals
		}else{
			if OptionsUI::current_trade_deals(self.player_id, &gstate.relations, dstate).list_mode_update_and_action(&mut self.mode, dstate) {
				self.view = true;
			}
		}
		UIModeControl::UnChgd
	}
}

impl <'bt,'ut,'rt,'dt>OptionsUI<'bt,'ut,'rt,'dt> {
	fn current_trade_deals(player_id: usize, relations: &Relations, dstate: &DispState) -> Self {
		let nms_string = if let Some(trade_deals) = relations.status(player_id, dstate.iface_settings.cur_player as usize).trade_deals() {
			let mut nms_string = Vec::with_capacity(trade_deals.len());
			for trade_deal in trade_deals.iter() {
				nms_string.push(trade_deal.nm(&dstate.local));
			}
			nms_string
		// no current trade deals
		}else{Vec::new()};
		
		// register_shortcuts takes [&str]s, so take references of all the strings
		let mut nms = Vec::with_capacity(nms_string.len());
		
		for nm_string in nms_string.iter() {
			nms.push(nm_string.as_str());
		}
		
		OptionsUI::new(&nms)
	}
}

