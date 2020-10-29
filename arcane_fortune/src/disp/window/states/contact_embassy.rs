use crate::nn::*;
use super::*;
use std::convert::TryFrom;

pub enum ContactEmbassyWindowState {
	CivSelection {mode: usize}, // of the civs
	DialogSelection {mode: usize, owner_id: usize, quote_printer: TxtPrinter},
	
	// just show a quote and let the user close the window
	Threaten {owner_id: usize, quote_printer: TxtPrinter},
	DeclareWar {owner_id: usize, quote_printer: TxtPrinter},
	DeclarePeace {owner_id: usize, quote_printer: TxtPrinter},
	DeclaredWarOn {owner_id: usize, quote_printer: TxtPrinter}, // AI declares war on the human player
	
	DeclarePeaceTreaty {owner_id: usize, quote_printer: TxtPrinter,
		        gold_offering: String, curs_col: isize,
		        treaty_rejected: bool
	}
}

impl ContactEmbassyWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&mut self, players: &Vec<Player>, gstate: &GameState, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		match self {
			Self::CivSelection{mode} => {
				let contacted_civs = contacted_civilizations_list(gstate, players, dstate.iface_settings.cur_player, &dstate.local);
				let list_pos = dstate.print_list_window(*mode, dstate.local.Select_civilization.clone(), contacted_civs, None, None, 0, Some(players)); 
				dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
			} Self::DialogSelection{mode, owner_id, ref mut quote_printer} => {
				let o = &players[*owner_id];
				let w = 70;
				let w_pos = dstate.print_window(ScreenSz{w, h: 5+2+3+2+2, sz:0});
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let w = (w - 2) as i32;
				
				let mut row = 0;
				macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;}};
				
				let mut screen_reader_cur_loc = (0,0);
				let cur_player = dstate.iface_settings.cur_player as usize;
				
				{ ///////////// title -- country name
					mvl!();
					let txt_len = format!("The {} Embassy", o.personalization.nm).len();
					let g = (w as usize - txt_len) / 2;
					let mut sp = String::with_capacity(g);
					for _ in 0..g {sp.push(' ');}
					
					dstate.addstr(&sp);
					dstate.addstr("The ");
					
					set_player_color(o, true, &mut dstate.renderer);
					dstate.addstr(&o.personalization.nm);
					set_player_color(o, false, &mut dstate.renderer);
					
					dstate.addstr(" Embassy");
					
					mvl!();
				}
				
				{ //////////// ruler
					mvl!();
					dstate.addstr(&format!("Their leader, {} {}, ", o.personalization.ruler_nm.first, o.personalization.ruler_nm.last));
					gstate.relations.print_mood_action(*owner_id, cur_player, players, dstate);
					dstate.addstr(" says:");
					
					mvl!();
					dstate.addstr(&format!("   \"{}\"", quote_printer.gen()));
					
					mvl!();mvl!();
					center_txt("How do you respond?", w, None, &mut dstate.renderer);
				}
				
				{ /////////// options
					mvl!();
					
					let screen_w = dstate.iface_settings.screen_sz.w as i32;
					
					// todo: add Trade technology
					dstate.buttons.Threaten.print_centered_selection(row + y, *mode == 0, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
					row += 1;
					
					//center_txt("Demand tribute", w, None);
					//mvl!();
					
					// at war
					if gstate.relations.at_war(*owner_id, cur_player) {
						dstate.buttons.Suggest_a_peace_treaty.print_centered_selection(row + y, *mode == 1, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
					// peace treaty in effect
					}else if let Some(expiry) = gstate.relations.peace_treaty_turns_remaining(*owner_id, cur_player, gstate.turn) {
						center_txt(&dstate.local.Peace_treaty_expires_in.replace("[]", &dstate.local.date_interval_str(expiry as f32)), w, Some(COLOR_PAIR(CGREEN3)), &mut dstate.renderer);
					// at peace but no peace treaty in effect
					}else{
						dstate.buttons.Declare_war.print_centered_selection(row + y, *mode == 1, &mut screen_reader_cur_loc, screen_w, &mut dstate.renderer);
					}
					
					mvl!(); mvl!();
				}
				
				////////// print mood
				{
					let loc = ScreenCoord {y: w_pos.y + 3, x: w_pos.x + 1 + w as isize};
					gstate.relations.print_mood_factors(*owner_id, cur_player, loc, dstate);
				}
				
				// print key instructions
				{
					let txt_len = "<Left arrow> go back    <Enter> Perform action".len();
					let g = (w as usize - txt_len) / 2;
					let mut sp = String::with_capacity(g);
					for _ in 0..g {sp.push(' ');}
					
					let col2 = x + g as i32 + "<Left arrow> go back".len() as i32 + 3;
					
					mvl!();
					dstate.addstr(&sp);
					dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
					
					dstate.mv(row-1 + y, col2);
					addstr_c(&dstate.local.Up_Down_arrows, ESC_COLOR, &mut dstate.renderer);
					dstate.addch(' ');
					dstate.renderer.addstr(&dstate.local.select);
					
					mvl!();
					dstate.addstr(&sp);
					addstr_c(&dstate.local.Left_arrow, ESC_COLOR, &mut dstate.renderer);
					dstate.addch(' ');
					dstate.renderer.addstr(&dstate.local.to_go_back); 
					
					dstate.mv(row-1 + y, col2);
					addstr_c(&dstate.local.Enter_key, ESC_COLOR, &mut dstate.renderer);
					dstate.addch(' ');
					dstate.renderer.addstr(&dstate.local.perform_action);
				}
				
				dstate.renderer.mv(screen_reader_cur_loc.0, screen_reader_cur_loc.1);
				
			} Self::Threaten {owner_id, ref mut quote_printer} |
			  Self::DeclareWar {owner_id, ref mut quote_printer} |
			  Self::DeclarePeace {owner_id, ref mut quote_printer} |
			  Self::DeclaredWarOn {owner_id, ref mut quote_printer} => {
				let quote_txt = quote_printer.gen();
				let owner_id = *owner_id;
				
				let o = &players[owner_id];
				let w = 70;
				let w_pos = dstate.print_window(ScreenSz{w, h: 5+4, sz:0});
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let w = (w - 2) as i32;
				
				let mut row = 0;
				macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;};
						     ($final: expr) => {dstate.mv(row + y, x);}};
				
				///////////// title -- country name
				{
					// AI declares war on the human player
					if let Self::DeclaredWarOn {..} = self {
						let txt_len = format!("The {} civilization has declared war on you!", o.personalization.nm).len();
						let g = (w as usize - txt_len) / 2;
						let mut sp = String::with_capacity(g);
						for _ in 0..g {sp.push(' ');}
						
						mvl!();
						dstate.addstr(&sp);
						dstate.addstr("The ");
						
						set_player_color(o, true, &mut dstate.renderer);
						dstate.addstr(&o.personalization.nm);
						set_player_color(o, false, &mut dstate.renderer);
						
						dstate.addstr(" civilization has declared war on you!");
					}else{
						let title_txt = match self {
							Self::Threaten {..} => {"You have threatened "}
							Self::DeclareWar {..} => {"You have declared war on "}
							Self::DeclarePeace {..} => {"You have made peace with "}
							_ => {panicq!{"match condition shouldn't be possible"}}
						};
						
						let txt_len = format!("{}{}!", title_txt, o.personalization.nm).len();
						let g = (w as usize - txt_len) / 2;
						let mut sp = String::with_capacity(g);
						for _ in 0..g {sp.push(' ');}
						
						mvl!();
						dstate.addstr(&sp);
						dstate.addstr(title_txt);
						
						set_player_color(o, true, &mut dstate.renderer);
						dstate.addstr(&o.personalization.nm);
						set_player_color(o, false, &mut dstate.renderer);
					}
					
					dstate.addstr("!");
					mvl!();
				}
				
				//////////// ruler
				{
					mvl!();
					dstate.addstr(&format!("{} {}, ", o.personalization.ruler_nm.first, o.personalization.ruler_nm.last));
					gstate.relations.print_mood_action(owner_id, dstate.iface_settings.cur_player as usize, players, dstate);
					dstate.addstr(if let Self::DeclaredWarOn {..} = self {" says:"} else {" responds:"});
					
					mvl!();
					dstate.addstr(&format!("   \"{}\"", quote_txt));
					
					mvl!();mvl!();
				}
				
				///////// esc to close
				{
					let txt_len = "<Esc> ".len() + dstate.local.to_close.len();
					let g = (w as usize - txt_len) / 2;
					let mut sp = String::with_capacity(g);
					for _ in 0..g {sp.push(' ');}
					
					mvl!(true);
					dstate.addstr(&sp);
					dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
				}
			} Self::DeclarePeaceTreaty {owner_id, ref mut quote_printer, gold_offering, curs_col, treaty_rejected} => {
				let o = &players[*owner_id];
				let w = 80;
				let mut h = 5+5+4;
				if *treaty_rejected {h += 2};
				let w_pos = dstate.print_window(ScreenSz{w, h, sz:0});
				
				let y = w_pos.y as i32 + 1;
				let x = w_pos.x as i32 + 2;
				
				let w = (w - 2) as i32;
				
				let mut row = 0;
				macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;}};
				
				///////////// title -- country name
				{
					const TITLE_TXT: &str = "Peace treaty with ";
					
					let txt_len = format!("{}{}!", TITLE_TXT, o.personalization.nm).len();
					let g = (w as usize - txt_len) / 2;
					let mut sp = String::with_capacity(g);
					for _ in 0..g {sp.push(' ');}
					
					mvl!();
					dstate.addstr(&sp);
					dstate.addstr(TITLE_TXT);
					
					set_player_color(o, true, &mut dstate.renderer);
					dstate.addstr(&o.personalization.nm);
					set_player_color(o, false, &mut dstate.renderer);
					
					mvl!();
				}
				
				const INPUT_OFFSET_COL: i32 = 4;
				
				////////// print gold input and quote
				{
					mvl!();
					dstate.addstr("How much gold would you like to offer in the treaty?");
					
					// gold offering
					dstate.mv(y+INPUT_OFFSET_COL,x);
					dstate.addstr(&gold_offering);
					
					mvl!();mvl!();mvl!();mvl!();
					dstate.addstr(&format!("( Negative values indicate {} will instead pay and not you;", o.personalization.nm));
					mvl!();
					dstate.addstr(&format!("  Peace treaties cannot be terminated for {} years after signing. )", gstate.relations.config.peace_treaty_min_years));
					
					
					// print notice that treaty rejected and give a quote from the ruler
					if *treaty_rejected {
						mvl!();mvl!();
						dstate.addstr(&format!("{} {} rejects your offer, ", o.personalization.ruler_nm.first, o.personalization.ruler_nm.last));
						gstate.relations.print_mood_action(*owner_id, dstate.iface_settings.cur_player as usize, players, dstate);
						dstate.addstr(" saying:");
						
						mvl!();
						dstate.addstr(&format!("   \"{}\"", quote_printer.gen()));
					}
				}
				
				// instructions
				{
					const SPACER: &str = "   ";
					let instructions_w = (dstate.buttons.Esc_to_close.print_txt(&dstate.local).len() + SPACER.len()
								  + dstate.buttons.Propose_treaty.print_txt(&dstate.local).len()) as i32;
					let gap = ((w - instructions_w)/2) as i32;
					dstate.mv(y + row + 1, x - 1 + gap);
					dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer); dstate.addstr(SPACER);
					dstate.buttons.Propose_treaty.print(None, &dstate.local, &mut dstate.renderer);
				}
				
				// mv to cursor location
				dstate.mv(y + INPUT_OFFSET_COL, x + *curs_col as i32);
			}
		}
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			gstate: &mut GameState, dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		match self {
			Self::CivSelection {ref mut mode} => {
				let list = contacted_civilizations_list(gstate, players, dstate.iface_settings.cur_player, &dstate.local);
				
				macro_rules! enter_action{($mode: expr) => {
					let owner_id = if let ArgOptionUI::OwnerInd(owner_ind) = list.options[$mode].arg {
						owner_ind
					}else{panicq!("list argument option not properly set");};
					
					let quote_category = TxtCategory::from_relations(&gstate.relations, owner_id, dstate.iface_settings.cur_player as usize, players);
					
					*self= Self::DialogSelection {
						mode: 0,
						owner_id,
						quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen())
					};
					return UIModeControl::UnChgd;
				};};
				if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
				
				let kbd = &dstate.kbd;
				match dstate.key_pressed {
					// down
					k if kbd.down(k) => {
						if (*mode + 1) <= (list.options.len()-1) {
							*mode += 1;
						}else{
							*mode = 0;
						}
					
					// up
					} k if kbd.up(k) => {
						if *mode > 0 {
							*mode -= 1;
						}else{
							*mode = list.options.len() - 1;
						}
						
					// enter
					} k if k == kbd.enter => {
						enter_action!(*mode);
					} _ => {}
				}
			} Self::DialogSelection {ref mut mode, owner_id, ..} => {
				let cur_player = dstate.iface_settings.cur_player as usize;
				let n_options = if gstate.relations.at_war(cur_player, *owner_id) {
						2 // threaten, declare peace
					}else if None == gstate.relations.peace_treaty_turns_remaining(cur_player, *owner_id, gstate.turn) {
						2 // threaten, declare war
					}else {1 // threaten (active peace treaty prevents war)
				};
				
				macro_rules! threaten{() => {
					gstate.relations.threaten(cur_player, *owner_id, gstate.turn);
					let quote_category = TxtCategory::from_relations(&gstate.relations, *owner_id, cur_player, players);
					
					*self = Self::Threaten {
						owner_id: *owner_id,
						quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen())
					};
					return UIModeControl::UnChgd;
				};};
				
				// declare peace (move to treaty proposal page)
				macro_rules! declare_peace{() => {
					let quote_category = TxtCategory::from_relations(&gstate.relations, *owner_id, cur_player, players);
					dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_VISIBLE);
					
					const DEFAULT_GOLD: &str = "0";
					
					*self = Self::DeclarePeaceTreaty {
						owner_id: *owner_id,
						quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen()),
						curs_col: DEFAULT_GOLD.len() as isize,
						gold_offering: String::from(DEFAULT_GOLD),
						treaty_rejected: false
					};
					return UIModeControl::UnChgd;
				};};
				
				macro_rules! declare_war{() => {
					let owner_id = *owner_id;
					let quote_category = TxtCategory::from_relations(&gstate.relations, owner_id, cur_player, players);
					
					*self = Self::DeclareWar {
						owner_id,
						quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen())
					};
					if let Some(new_ui_mode) = gstate.relations.declare_war_ret_ui_mode(cur_player, owner_id, &mut gstate.logs, players, gstate.turn, dstate.iface_settings.cur_player_paused(players), &mut gstate.rng, dstate) {
						return UIModeControl::New(new_ui_mode);
					}
					return UIModeControl::UnChgd;
				};};
					
				if dstate.buttons.Threaten.activated(0, &dstate.mouse_event) {threaten!();} else
				if dstate.buttons.Suggest_a_peace_treaty.activated(0, &dstate.mouse_event) {declare_peace!();} else
				if dstate.buttons.Declare_war.activated(0, &dstate.mouse_event) {declare_war!();}
				
				if dstate.buttons.Threaten.hovered(&dstate.mouse_event) {*mode = 0;} else
				if dstate.buttons.Suggest_a_peace_treaty.hovered(&dstate.mouse_event) {*mode = 1;} else
				if dstate.buttons.Declare_war.hovered(&dstate.mouse_event) {*mode = 1;}
				
				let kbd = &dstate.kbd;
				match dstate.key_pressed {
					k if kbd.up(k) => {if *mode > 0 {*mode -= 1;} else {*mode = n_options-1;}
					} k if kbd.down(k) => {if *mode < (n_options-1) {*mode += 1;} else {*mode = 0;}
					// enter
					} k if k == kbd.enter => {
						match *mode {
							0 => {threaten!();
								
							// declare peace or war
							} 1 => {
								if gstate.relations.at_war(*owner_id, cur_player)
									{declare_peace!();} else {declare_war!();}
							} _ => {panicq!("unknown menu selection for embassy dialog selection ({})", *mode);}
						}
					} KEY_LEFT => {
						*self = Self::CivSelection {mode: *owner_id};
					} _ => {}
				}
			} Self::Threaten {..} | Self::DeclareWar {..} | Self::DeclarePeace {..} | Self::DeclaredWarOn {..} => {}
			
			
			Self::DeclarePeaceTreaty {owner_id, ref mut gold_offering, ref mut curs_col,
					ref mut treaty_rejected, ref mut quote_printer} => {
				// enter key pressed
				if dstate.buttons.Propose_treaty.activated(dstate.key_pressed, &dstate.mouse_event) && gold_offering.len() > 0 {
					if let Result::Ok(gold_offering) = gold_offering.parse() {
						let treaty_accepted = (|| {
							if let Some(empire_state) = players[*owner_id].ptype.empire_state() {
								// the relevant player has sufficient gold for the treaty
								let cur_player = dstate.iface_settings.cur_player as usize;
								let gold_sufficient = (gold_offering >= 0. && players[cur_player].stats.gold >= gold_offering) ||
											    (gold_offering < 0. && players[*owner_id].stats.gold >= (-gold_offering));
								
								// AI accepts treaty?
								return gold_sufficient && empire_state.accept_peace_treaty(*owner_id, cur_player, gold_offering, gstate, players);
							}
							
							// should occur only for the human player -- nobility & barbarians shouldn't appear as UI options
							debug_assertq!(players[*owner_id].ptype.is_human());
							true
						})();
						
						// update player states & UI
						if treaty_accepted {
							let cur_player = dstate.iface_settings.cur_player as usize;
							gstate.relations.declare_peace(cur_player, *owner_id, &mut gstate.logs, gstate.turn);
							
							players[cur_player].stats.gold -= gold_offering;
							players[*owner_id].stats.gold += gold_offering;
							
							let quote_category = TxtCategory::from_relations(&gstate.relations, *owner_id, cur_player, players);
							
							*self = Self::DeclarePeace {
								owner_id: *owner_id,
								quote_printer: TxtPrinter::new(quote_category, gstate.rng.gen())
							};
							dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
						}else{
							*treaty_rejected = true;
							quote_printer.new_seq();
						}
					}
					
				// enter key not pressed or no gold offered
				}else{
					do_txt_entry_keys!(dstate.key_pressed, *curs_col, gold_offering, Printable::Numeric, dstate);
				}	
			}
		}
		UIModeControl::UnChgd
	}
}
