// for proposing a new trade
use super::*;

pub struct TradeState<'bt,'ut,'rt,'dt> {
	player_id: usize,
	
	pub add_trade_item: Option<AddTradeItemUI<'bt,'ut,'rt,'dt>>, // Some value when the player is selecting which item to add
	trade_rejected: bool,
	trade_deal: TradeDeal
}

pub struct AddTradeItemUI<'bt,'ut,'rt,'dt> {
	cur_player_offers: bool, // if true, current player offers the item, else the `player_id` in TradeState offers
	pub state: AddTradeItemStateUI<'bt,'ut,'rt,'dt>
}

impl AddTradeItemUI<'_,'_,'_,'_> {
	fn new(cur_player_offers: bool) -> Self {
		Self {
			cur_player_offers,
			state: AddTradeItemStateUI::SelItemType {mode: 0}
		}
	}
}

///////////////////////
// two step process to add a trade item-- get item type, then specifics
pub enum AddTradeItemStateUI<'bt,'ut,'rt,'dt> {
	SelItemType {mode: usize},
	GetItemVals(TradeItemValsUI<'bt,'ut,'rt,'dt>) 
}

// specifics for item type
pub enum TradeItemValsUI<'bt,'ut,'rt,'dt> {
	LumpGold(GetTextWindowState),
	GoldPerTurn(GetTextWindowState),
	Resource {mode: usize, opts: OptionsUI<'bt,'ut,'rt,'dt>},
	Tech {mode: usize, opts: OptionsUI<'bt,'ut,'rt,'dt>}
}

pub const TRADE_LIST_HEIGHT: isize = 10;
pub const TRADE_LIST_WIDTH: i32 = 70;

impl TradeDeal {
	pub fn print(&self, row: isize, col: isize, other_player_id: usize, 
			temps: &Templates, players: &Vec<Player>, dstate: &mut DispState) {
		let cur_player = dstate.iface_settings.cur_player as usize;
		
		{ //// player offers
			dstate.mv(row, col + 1);
			center_txt(&dstate.local.You_offer, TRADE_LIST_WIDTH/2, None, &mut dstate.renderer);
			
			let offers = self.player_gives(cur_player, other_player_id);
			
			// nothing offered
			if offers.len() == 0 {
				dstate.mv(row + 2, col + 1);
				center_txt(&dstate.local.Nothing, TRADE_LIST_WIDTH/2, Some(COLOR_PAIR(CGRAY)), &mut dstate.renderer);
			// print offers
			}else{
				for (offset, offer) in offers.iter().enumerate() {
					dstate.mv(row + 2 + offset as isize, col + 2);
					offer.print(temps, dstate);
				}
			}
		}
		
		{ //// they offer
			{ // "[they] offer:" -- colorize
				let personalization = &players[other_player_id].personalization;
				let tags = vec![KeyValColor {
					key: String::from("[they]"),
					val: personalization.nm.clone(),
					attr: COLOR_PAIR(personalization.color)
				}];
				let len = color_tags_txt(&dstate.local.They_offer, &tags).len() as i32;
				dstate.mv(row, col as i32 + TRADE_LIST_WIDTH/2 + (TRADE_LIST_WIDTH/2 - len)/2); // center
				color_tags_print(&dstate.local.They_offer, &tags, None, &mut dstate.renderer);
			}
			
			let offers = self.player_gives(other_player_id, cur_player);
			
			// nothing offered
			if offers.len() == 0 {
				dstate.mv(row + 2, col as i32 + TRADE_LIST_WIDTH/2);
				center_txt(&dstate.local.Nothing, TRADE_LIST_WIDTH/2, Some(COLOR_PAIR(CGRAY)), &mut dstate.renderer);
			// print offers
			}else{
				for (offset, offer) in offers.iter().enumerate() {
					dstate.mv(row + 2 + offset as isize, col as i32 + 2 + TRADE_LIST_WIDTH/2);
					offer.print(temps, dstate);
				}
			}
		}
		
		// line between players
		for r_off in 0..TRADE_LIST_HEIGHT {
			dstate.mv(row + r_off, col as i32 + TRADE_LIST_WIDTH/2);
			dstate.addch(dstate.chars.vline_char);
		}
	}
}

//////////////////
// TradeState is embedded in UIMode
impl <'bt,'ut,'rt,'dt>TradeState<'bt,'ut,'rt,'dt> {
	// player_id = player to trade with
	pub fn new(player_id: usize, turn: usize) -> Self {
		Self {
			player_id,
			add_trade_item: None,
			trade_rejected: false,
			trade_deal: TradeDeal::new(turn)
		}
	}
	
	pub fn print(&self, players: &Vec<Player>, gstate: &GameState,
			temps: &Templates, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		let h = if self.trade_rejected {2} else {0} + 10 + TRADE_LIST_HEIGHT as usize;
		let w_pos = dstate.print_window(ScreenSz{w: TRADE_LIST_WIDTH as usize, h, sz:0});
		
		let mut row = w_pos.y + 1;
		
		dstate.mv(row, w_pos.x + 1);
		center_txt(&dstate.local.Trade_proposal, TRADE_LIST_WIDTH, Some(COLOR_PAIR(TITLE_COLOR)), &mut dstate.renderer);
		row += 2;
		
		if self.trade_rejected {
			dstate.mv(row, w_pos.x + 1);
			center_txt(&dstate.local.Trade_rejected, TRADE_LIST_WIDTH, Some(COLOR_PAIR(CRED)), &mut dstate.renderer);
			row += 2;
		}
		
		self.trade_deal.print(row, w_pos.x, self.player_id, temps, players, dstate);
		
		{ ///////// print mood
			let loc = ScreenCoord {y: w_pos.y + 3, x: w_pos.x + TRADE_LIST_WIDTH as isize};
			gstate.relations.print_mood_factors(self.player_id, dstate.iface_settings.cur_player as usize, players, loc, dstate);
		}
		
		{ // print key instructions
			let row = (w_pos.y + TRADE_LIST_HEIGHT + 5) as i32;
			
			dstate.buttons.Offer_trade_item.print_centered_at((row, w_pos.x as i32 + 1), TRADE_LIST_WIDTH/2, &dstate.local, &mut dstate.renderer);
			dstate.buttons.Request_trade_item.print_centered_at((row, w_pos.x as i32 + TRADE_LIST_WIDTH/2), TRADE_LIST_WIDTH/2, &dstate.local, &mut dstate.renderer);
			
			dstate.buttons.Propose_trade.print_centered(row + 2, &dstate.iface_settings, &dstate.local, &mut dstate.renderer); 
			dstate.buttons.Esc_to_close.print_centered(row + 3, &dstate.iface_settings, &dstate.local, &mut dstate.renderer);
		}
		
		// player is adding a new trade item (draw window on top of main window)
		if let Some(add_trade_item) = &self.add_trade_item {
			match &add_trade_item.state {
				// get type of item, ex gold or resource
				AddTradeItemStateUI::SelItemType {mode} => {
					let w = 35;
					let entries = OptionsUI::trade_item_types(gstate.relations.defensive_pact(cur_player, self.player_id), &dstate.local);
					let list_pos = dstate.print_list_window(*mode, dstate.local.Select_an_item.clone(), entries, Some(w), None, 0, None);
					dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32); // text mode cursor for screen readers
					
				// get specific value for a given item, like the amount of gold or type of resource
				} AddTradeItemStateUI::GetItemVals(item_vals) => {
					match item_vals {
						TradeItemValsUI::LumpGold(get_text_window_state) |
						TradeItemValsUI::GoldPerTurn(get_text_window_state) => {
							return get_text_window_state.print(dstate);
						} TradeItemValsUI::Resource {mode, opts} => {
							let w = 35;
							let list_pos = dstate.print_list_window(*mode, dstate.local.Select_an_item.clone(), opts.clone(), Some(w), None, 0, None);
							dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);	
						} TradeItemValsUI::Tech {mode, opts} => {
							let w = 35;
							let list_pos = dstate.print_list_window(*mode, dstate.local.Select_an_item.clone(), opts.clone(), Some(w), None, 0, None);
							dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
						}
					}
				}
			}
		}
		
		UIModeControl::UnChgd
	}
	
	pub fn keys(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		
		//////////////////////////////////////
		// the player is adding a new trade item (a separate window is created)
		if let Some(add_trade_item) = &mut self.add_trade_item {
			// close sub-window
			if dstate.buttons.Esc_to_close.activated(dstate.key_pressed, &dstate.mouse_event) {
				self.add_trade_item = None;
				dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				return UIModeControl::UnChgd;
			}
			
			// owner1 offers owner2
			let (owner1, owner2) = get_owners(add_trade_item.cur_player_offers, self.player_id, dstate);
			
			match &mut add_trade_item.state {
				// get type of item, ex gold or resource
				AddTradeItemStateUI::SelItemType {ref mut mode} => {
					let entries = OptionsUI::trade_item_types(gstate.relations.defensive_pact(owner1, owner2), &dstate.local);
					if list_mode_update_and_action(mode, entries.options.len(), dstate) {
						add_trade_item.state = AddTradeItemStateUI::GetItemVals(
							match *mode {
								0 => {
									dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
									
									TradeItemValsUI::LumpGold(
										GetTextWindowState::new(
											TxtType::CustomPrintNm(
												dstate.local.Gold.clone()
											),
											String::new()
										)
									)
								} 1 => {
									dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
									
									TradeItemValsUI::GoldPerTurn(
										GetTextWindowState::new(
											TxtType::CustomPrintNm(
												dstate.local.Gold_per_turn.clone()
											),
											String::new()
										)
									)
								} 2 => TradeItemValsUI::Resource {
										mode: 0,
										opts: OptionsUI::tradeable_resources(&players[owner1].stats, &players[owner2].stats, temps, dstate)
									},
								3 => TradeItemValsUI::Tech {
										mode: 0,
										opts: OptionsUI::tradeable_techs(&players[owner1].stats, &players[owner2].stats, temps, dstate)
									},
								// world map
								4 => {
									self.trade_deal.add_item(TradeItem::WorldMap, owner1, owner2);
									self.add_trade_item = None;
									return UIModeControl::UnChgd;
								// defensive pact
								} 5 => {
									self.trade_deal.add_item(TradeItem::DefensivePact, owner1, owner2);
									self.add_trade_item = None;
									return UIModeControl::UnChgd;
								}
								_ => {panicq!("invalid list item");}
							}
						);
					}
				// get specific value for a given item, like the amount of gold or type of resource
				} AddTradeItemStateUI::GetItemVals(item_vals) => {
					match item_vals {
						TradeItemValsUI::LumpGold(get_txt_window_state) => {
							if let Some(txt) = get_txt_window_state.keys_ret_txt(dstate) {
								if let Result::Ok(val) = txt.parse() {
									if val > 0. && val <= players[owner1].stats.gold {
										self.trade_deal.add_item(TradeItem::LumpGold(val), owner1, owner2);
										dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
										self.add_trade_item = None;
									}
								}
							}
						} TradeItemValsUI::GoldPerTurn(get_txt_window_state) => {
							if let Some(txt) = get_txt_window_state.keys_ret_txt(dstate) {
								if let Result::Ok(val) = txt.parse() {
									if val > 0. && val <= players[owner1].stats.net_income(players, &gstate.relations) {
										self.trade_deal.add_item(TradeItem::GoldPerTurn(val), owner1, owner2);
										dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
										self.add_trade_item = None;
									}
								}
							}
						} TradeItemValsUI::Resource {mode, opts} => {
							// mouse hovering
							if let Some(ind) = dstate.buttons.list_item_hovered(&dstate.mouse_event) {*mode = ind;}
							
							if list_mode_update_and_action(mode, opts.options.len(), dstate) {
								if let ArgOptionUI::ResourceInd(res_ind) = &opts.options[*mode].arg {
									self.trade_deal.add_item(TradeItem::Resource(*res_ind), owner1, owner2);
									self.add_trade_item = None;
								}else{panicq!("unknown argument option");}
							}
						} TradeItemValsUI::Tech {mode, opts} => {
							// mouse hovering
							if let Some(ind) = dstate.buttons.list_item_hovered(&dstate.mouse_event) {*mode = ind;}
							
							if list_mode_update_and_action(mode, opts.options.len(), dstate) {
								if let ArgOptionUI::TechInd(tech_ind) = &opts.options[*mode].arg {
									self.trade_deal.add_item(TradeItem::Tech(*tech_ind), owner1, owner2);
									self.add_trade_item = None;
								}else{panicq!("unknown argument option (tech)");}
							}
						}
					}
				}
			}
		
		/////////////////////////////////////////
		// main trading screen actions
		}else{
			let b = &dstate.buttons;
			if b.Esc_to_close.activated(dstate.key_pressed, &dstate.mouse_event) {
				return UIModeControl::Closed;
			}else if b.Offer_trade_item.activated(dstate.key_pressed, &dstate.mouse_event) {
				self.add_trade_item = Some(AddTradeItemUI::new(true));
				self.trade_rejected = false;
			}else if b.Request_trade_item.activated(dstate.key_pressed, &dstate.mouse_event) {
				self.add_trade_item = Some(AddTradeItemUI::new(false));
				self.trade_rejected = false;
			}else if b.Propose_trade.activated(dstate.key_pressed, &dstate.mouse_event) {
				let cur_player = dstate.iface_settings.cur_player as usize;
				
				// trade accepted
				if gstate.relations.propose_trade(&self.trade_deal, cur_player, self.player_id, players, temps, map_data, dstate, gstate.turn) {
					return UIModeControl::New(UIMode::GenericAlert(GenericAlertState {
						txt: dstate.local.Trade_accepted.clone()
					}));
				// trade rejected
				}else{
					self.trade_rejected = true;
				}
			}
		}
		UIModeControl::UnChgd
	}
}

// returns indices: (offerer, receiver)
fn get_owners(cur_player_offers: bool, other_player: usize, dstate: &DispState) -> (usize, usize) {
	if cur_player_offers {
		(dstate.iface_settings.cur_player as usize, other_player)
	}else{
		(other_player, dstate.iface_settings.cur_player as usize)
	}
}

// list selections
impl <'bt,'ut,'rt,'dt>OptionsUI<'bt,'ut,'rt,'dt> {
	fn trade_item_types(current_defensive_pact: bool, l: &Localization) -> Self {
		let mut nms_string = vec![
			l.Gold.clone(), l.Gold_per_turn.clone(), l.Resource.clone(), l.Technology.clone(), l.World_map.clone()
		];
		
		if !current_defensive_pact {
			nms_string.push(l.Defensive_pact.clone());
		}
		
		// register_shortcuts takes [&str]s, so take references of all the strings
		let mut nms = Vec::with_capacity(nms_string.len());
		
		for nm_string in nms_string.iter() {
			nms.push(nm_string.as_str());
		}
		
		OptionsUI::new(&nms)
	}
	
	// what resources could player1 give to player2?
	fn tradeable_resources(player1: &Stats, player2: &Stats,
			temps: &Templates, dstate: &DispState) -> Self {
		let mut offerable_resources = Vec::with_capacity(player1.resources_avail.len());
		let mut offerable_res_inds = Vec::with_capacity(player1.resources_avail.len());
		for (resource_ind, _) in player1.resources_avail.iter().enumerate()
				.filter(|(resource_ind, n_avail)|
						**n_avail != 0 &&
						player2.resources_avail[*resource_ind] == 0) {
			offerable_resources.push(temps.resources[resource_ind].nm[dstate.local.lang_ind].as_str());
			offerable_res_inds.push(resource_ind);
		}
		
		let mut opts = OptionsUI::new(&offerable_resources);
		
		// associate indices with each menu entry
		for (offerable_res_ind, opt) in offerable_res_inds.iter().zip(opts.options.iter_mut()) {
			opt.arg = ArgOptionUI::ResourceInd(*offerable_res_ind);
		}
	
		opts
	}
	
	// what techs could player1 give to player2?
	fn tradeable_techs(player1: &Stats, player2: &Stats,
			temps: &Templates, dstate: &DispState) -> Self {
		let mut offerable_techs = Vec::with_capacity(player1.techs_progress.len()); // string
		let mut offerable_tech_inds = Vec::with_capacity(player1.techs_progress.len());
		for (tech_ind, _) in player1.techs_progress.iter().enumerate()
				.filter(|(tech_ind, tech)|
						**tech == TechProg::Finished &&
						!player2.tech_met(&Some(vec![*tech_ind]))) {
			offerable_techs.push(temps.techs[tech_ind].nm[dstate.local.lang_ind].as_str());
			offerable_tech_inds.push(tech_ind);
		}
		
		let mut opts = OptionsUI::new(&offerable_techs);
		
		// associate indices with each menu entry
		for (offerable_tech_ind, opt) in offerable_tech_inds.iter().zip(opts.options.iter_mut()) {
			opt.arg = ArgOptionUI::TechInd(*offerable_tech_ind);
		}
		opts
	}
}

// embeded in TradeState
impl TradeItem {
	fn print(&self, temps: &Templates, dstate: &mut DispState) {
		match self {
			Self::LumpGold(gold) => {
				dstate.renderer.addstr(&format!("{}: {}", dstate.local.Gold, gold));
			} Self::GoldPerTurn(gpt) => {
				dstate.renderer.addstr(&format!("{}: {}", dstate.local.Gold_per_turn, gpt));
			} Self::Resource(ind) => {
				dstate.renderer.addstr(&temps.resources[*ind].nm[dstate.local.lang_ind]);
			} Self::Tech(ind) => {
				dstate.renderer.addstr(&temps.techs[*ind].nm[dstate.local.lang_ind]);
			} Self::WorldMap => {
				dstate.renderer.addstr(&dstate.local.World_map);
			} Self::DefensivePact => {
				dstate.renderer.addstr(&dstate.local.Defensive_pact);
			}
		}
	}
}

