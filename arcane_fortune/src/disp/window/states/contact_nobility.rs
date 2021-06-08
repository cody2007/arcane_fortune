use super::*;
use crate::nn::{TxtPrinter, TxtCategory};

pub enum ContactNobilityState {
	NobilitySelection {mode: usize}, // selection of houses in empire
	
	// \/ owner_id is the noble house
	DialogSelection {mode: usize, owner_id: usize, quote_printer: TxtPrinter},
	PopulationTargetSelection {mode: usize, owner_id: usize}
}

impl ContactNobilityState {
	pub fn new() -> Self {Self::NobilitySelection {mode: 0}}
	
	pub fn print<'bt,'ut,'rt,'dt> (&mut self, gstate: &GameState, players: &mut Vec<Player>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		match self {
			// ask which nobility to contact
			ContactNobilityState::NobilitySelection {mode} => {
				let options = noble_houses_list(cur_player, &gstate.relations, players, &dstate.local);
				let list_pos = dstate.print_list_window(*mode, dstate.local.Select_a_noble_house.clone(), options, None, None, 0, None);
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			} ContactNobilityState::DialogSelection {mode, owner_id, quote_printer} => {
				let o = &players[*owner_id];
				
				let h = if o.ptype.house().unwrap().target_city_coord.is_none() {17} else {17+2};
				
				let w = 70;
				let w_pos = dstate.print_window(ScreenSz{w, h, sz:0});
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let w = (w-2) as i32;
				
				let mut row = 0;
				macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;}}
				
				{ ///////////// title -- house name
					mvl!();
					let txt = dstate.local.house_nm.replace("[]", &o.personalization.nm);
					let g = (w as usize - txt.len()) / 2;
					let mut sp = String::with_capacity(g);
					for _ in 0..g {sp.push(' ');}
					
					dstate.addstr(&sp);
					
					set_player_color(o, true, &mut dstate.renderer);
					dstate.addstr(&txt);
					set_player_color(o, false, &mut dstate.renderer);
					
					mvl!();
				}
				
				{ ////// ruler
					mvl!();
					{ // House_greeting: "[name] [affinity] greets you saying:"
						let (mood_txt, mood_color) = gstate.relations.mood_action_txt(*owner_id, cur_player, players, &dstate.local);
						let txt = dstate.local.House_greeting.replace("[name]", &format!("{} {}", o.personalization.ruler_nm.first, o.personalization.ruler_nm.last));
						let tags = vec![KeyValColor {
							key: "[affinity]".to_string(),
							val: mood_txt,
							attr: mood_color
						}];
						color_tags_print(&txt, &tags, None, &mut dstate.renderer);
					}
					
					mvl!();
					dstate.addstr(&format!("   \"{}\"", quote_printer.gen()));
				}
				
				// show attack target
				(|| {
				 	let house = players[*owner_id].ptype.house_mut().unwrap();
				 	if let Some(target_city_coord) = house.target_city_coord {
						if let Some(ex) = exf.get(&target_city_coord) {
							if let Some(bldg_ind) = ex.bldg_ind {
								let b = &bldgs[bldg_ind];
								if let BldgArgs::PopulationCenter {nm: city_nm, ..} = &b.args {
									// Ordered to attack: city name (civ name)
									let player = &players[b.owner_id as usize];
									row += 1;
									dstate.mv(row + y, x as usize + (w as usize - (dstate.local.Ordered_to_attack.len() + city_nm.len() + 
											player.personalization.nm.len() + 4))/2);
									row += 1;
									dstate.attron(COLOR_PAIR(CGRAY));
									dstate.renderer.addstr(&dstate.local.Ordered_to_attack);
									dstate.addch(' ');
									dstate.attroff(COLOR_PAIR(CGRAY));
									dstate.renderer.addstr(city_nm);
									dstate.addstr(" (");
									print_civ_nm(&player, &mut dstate.renderer);
									dstate.addch(')');
									return;
								}
							}
						}
						// target set but no city found, clear it
						house.target_city_coord = None;
					}
				})();
				
				mvl!();mvl!();
				center_txt(&dstate.local.How_do_you_respond, w, None, &mut dstate.renderer);
				
				let mut screen_reader_cur_loc = (0,0);
				{ /////////// options
					mvl!();
					
					let screen_w = dstate.iface_settings.screen_sz.w as i32;
					
					macro_rules! print_button {($button: ident, $mode: expr) => {
						dstate.buttons.$button.print_centered_selection(row + y, *mode == $mode, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
						row += 1;
					};}
					
					print_button!(Suggest_a_trade, 0);
					print_button!(Lets_review_our_current_trade_deals, 1);
					print_button!(Lets_discuss_your_tax_obligations, 2);
					print_button!(Threaten, 3);
					print_button!(Send_your_troops_to_attack, 4);
					//center_txt("Demand tribute", w, None);
					
					mvl!();
				}
				
				print_leader_mood_and_key_instructions(row, w, w_pos, *owner_id, players, &gstate.relations, dstate);
				
				dstate.renderer.mv(screen_reader_cur_loc.0, screen_reader_cur_loc.1);
			// select target for the nobility to attack
			} ContactNobilityState::PopulationTargetSelection {mode, ..} => {
				let options = OptionsUI::target_cities(cur_player as SmSvType, bldgs, players, &gstate.relations, &dstate.local).0;
				let w = Some(45);
				let list_pos = dstate.print_list_window(*mode, dstate.local.Select_a_target.clone(), options, w, None, 0, Some(players));
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			}
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player>, gstate: &mut GameState, exf: &HashedMapEx,
			bldgs: &Vec<Bldg>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let cur_player = dstate.iface_settings.cur_player as usize;
		match self {
			// ask which nobility to contact
			ContactNobilityState::NobilitySelection {ref mut mode} => {
				let list = noble_houses_list(cur_player, &gstate.relations, players, &dstate.local);
				
				if list.list_mode_update_and_action(mode, dstate) {
					let owner_id = if let ArgOptionUI::OwnerInd(owner_ind) = list.options[*mode].arg {
						owner_ind
					}else{panicq!("list argument option not properly set");};
					
					let quote_category = TxtCategory::from_relations(&gstate.relations, owner_id, cur_player, players);
					
					*self = Self::DialogSelection {
						mode: 0,
						owner_id,
						quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen())
					};
				}
			} ContactNobilityState::DialogSelection {mode, owner_id, ..} => {
				if let Some(mode) = button_mode_update_and_action(mode, vec![
							&mut dstate.buttons.Suggest_a_trade,
							&mut dstate.buttons.Lets_review_our_current_trade_deals,
							&mut dstate.buttons.Lets_discuss_your_tax_obligations,
							&mut dstate.buttons.Threaten,
							&mut dstate.buttons.Send_your_troops_to_attack
						], dstate.key_pressed, &dstate.mouse_event, &dstate.kbd) {
					return match mode {
						// suggest a trade
						0 => {UIModeControl::New(UIMode::Trade(TradeState::new(*owner_id, gstate.turn)))
						
						// review trade
						} 1 => {UIModeControl::New(UIMode::ViewTrade(ViewTradeState::new(*owner_id)))
						
						// tax obligations
						} 2 => {UIModeControl::New(UIMode::SetNobleTax(SetNobleTaxState::new(*owner_id, &gstate.relations, dstate)))
							
						// threaten
						} 3 => {
							let cur_player = dstate.iface_settings.cur_player as usize;
							gstate.relations.threaten(cur_player, *owner_id, players, gstate.turn);
							let quote_category = TxtCategory::from_relations(&gstate.relations, *owner_id, cur_player, players);
							
							UIModeControl::New(UIMode::ContactEmbassyWindow(
								ContactEmbassyWindowState::Threaten {
									owner_id: *owner_id,
									quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen())
								}
							))
						
						// send your troops to
						} 4 => {
							*self = Self::PopulationTargetSelection {mode: 0, owner_id: *owner_id};
							UIModeControl::UnChgd
							
						} _ => {panicq!("invalid return");}
					};
				}
				
				if dstate.key_pressed == KEY_LEFT {
					*self = Self::NobilitySelection {mode: *owner_id};
				}
			// instruct noble house to attack city
			} ContactNobilityState::PopulationTargetSelection {mode, owner_id} => {
				let (options, city_coords) = OptionsUI::target_cities(cur_player as SmSvType, bldgs, players, &gstate.relations, &dstate.local);
				if options.list_mode_update_and_action(mode, dstate) {
					let city_coord = city_coords[*mode];
					if let Some(ex) = exf.get(&city_coord) {
						if let Some(bldg_ind) = ex.bldg_ind {
							let attackee_ind = bldgs[bldg_ind].owner_id as usize;
							
							gstate.relations.declare_war_ret_ui_mode(cur_player, attackee_ind, &mut gstate.logs, players, gstate.turn, None, &mut gstate.rng, dstate);
							
							// noble house will send troops, set nobility target
							//printlnq!("mood {}", gstate.relations.friendliness_toward(*owner_id, cur_player, players));
							let mut response_txt = if gstate.relations.friendliness_toward(*owner_id, cur_player, players) > 0. {
								// set nobility target
								if let Some(house) = players[*owner_id].ptype.house_mut() {
									house.target_city_coord = Some(city_coord);
								}else{panicq!("owner is not a noble house");}
								
								dstate.local.We_will_begin_preparing.clone()
							// no troops will be sent
							}else{
								dstate.local.We_will_not_begin_preparing.clone()
							};
							
							// The House of X responds: "..."
							let house_nm = dstate.local.house_nm.replace("[]", &players[*owner_id].personalization.nm);
							response_txt = response_txt.replace("[house_nm]", &house_nm);
							
							return UIModeControl::New(UIMode::GenericAlert(GenericAlertState {txt: response_txt}))
						}
					}
				}
			}
		}
		UIModeControl::UnChgd
	}
}

// list selection -- cities to attack
impl <'bt,'ut,'rt,'dt>OptionsUI<'bt,'ut,'rt,'dt> {
	// returns OptionsUI & city coords
	// city_coords are not put in opt.arg because this is used for coloring of the list entries when printing
	fn target_cities(cur_player: SmSvType, bldgs: &Vec<Bldg>, players: &Vec<Player>,
			relations: &Relations, l: &Localization) -> (Self, Vec<u64>) {
		struct TargetEntry {
			owner_ind: usize,
			txt: String
		}
		
		// get relevant population centers
		let mut targets = Vec::new();
		let mut city_coords = Vec::new();
		for b in bldgs.iter()
			.filter(|b| b.owner_id != cur_player &&
				!relations.fiefdom(cur_player as usize, b.owner_id as usize) &&
				relations.discovered(cur_player as usize, b.owner_id as usize)) {
			if let BldgArgs::PopulationCenter {nm, population, ..} = &b.args {
				let o = &players[b.owner_id as usize];
				targets.push(TargetEntry {
					owner_ind: b.owner_id as usize,
					// [city name], Rsdnts: [residents] ([empire])
					txt: format!("{}, {}: {} ({})", nm, l.Population, population.iter().sum::<u32>(), o.personalization.nm)
				});
				city_coords.push(b.coord);
			}
		}
		
		// register_shortcuts takes [&str]s, so take references of all the strings
		let mut nms = Vec::with_capacity(targets.len());
		for entry in targets.iter() {
			nms.push(entry.txt.as_str());
		}
		
		let mut options = OptionsUI::new(&nms);
		
		// associate owner_id w/ each menu entry
		for (opt, target) in options.options.iter_mut().zip(targets.iter()) {
			opt.arg = ArgOptionUI::OwnerInd(target.owner_ind);
		}
		
		(options, city_coords)
	}
}

