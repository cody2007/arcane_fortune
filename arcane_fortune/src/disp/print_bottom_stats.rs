use crate::disp_lib::*;
use crate::map::*;
use crate::units::{ActionType, ActionMetaCont, WORKER_NM, worker_can_continue_bldg};//MAX_UNITS_PER_PLOT, WORKER_NM};
use crate::gcore::hashing::*;
use crate::gcore::{Relations, Log, LogType};
use crate::player::{Player, PlayerType, Stats};
use crate::movement::{manhattan_dist_components, manhattan_dist};
use crate::zones::return_zone_coord;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;

use super::*;
use super::vars::*;
use super::color::*;

const LAND_STATS_COL: i32 = (SUB_MAP_WIDTH + 4) as i32;
const UNIT_STATS_COL: i32 = (SUB_MAP_WIDTH + 19) as i32;

impl ActionType<'_,'_,'_,'_> {
	pub fn print(&self, roff: &mut i32, path_valid: bool, actions_req: f32,
			turns_est: bool, ut_opt: Option<&UnitTemplate>, bldgs: &Vec<Bldg>, exf: &HashedMapEx,
			map_data: &mut MapData, iface_settings: &IfaceSettings,
			l: &Localization, kbd: &KeyboardMap, disp_chars: &DispChars, txt_list: &mut TxtList, d: &mut DispState) {
		macro_rules! mvl {() => (d.mv(*roff, UNIT_STATS_COL); *roff += 1;);}
			
		macro_rules! action_key {
			($nm: expr, $color: expr) => {mvl!(); d.attron(COLOR_PAIR($color)); d.addstr($nm); d.attroff(COLOR_PAIR($color))};
			($nm: expr, $color: expr, $last: expr) => {d.mv(*roff, UNIT_STATS_COL); d.attron(COLOR_PAIR($color)); d.addstr($nm); d.attroff(COLOR_PAIR($color))};};
		
		let cur_player = iface_settings.cur_player;
		let map_sz = *map_data.map_szs.last().unwrap();
		let full_zoom = iface_settings.zoom_ind == map_data.max_zoom_ind();
		
		macro_rules! mv_cursor_to{($txt: expr) => {
			action_key!(&format!("<{}>", l.click_and_drag), CGREEN);
			d.addstr(&format!(" {} ", l.or));
			d.attron(COLOR_PAIR(CGREEN));
			d.addstr(&l.arrow_keys);
			d.attroff(COLOR_PAIR(CGREEN));
			d.addch(' ');
			d.addstr($txt);
		};};
		
		match self {
			ActionType::MvWithCursor => {
				mv_cursor_to!(&l.to_move_unit);
				
				if !full_zoom {
					action_key!(&format!("<{}>", key_txt(kbd.zoom_in, l)), CGREEN);
					d.addch(' '); d.addstr(&l.to_zoom_in);
				}
				
				action_key!("<Esc>", ESC_COLOR, true);
				d.addch(' ');
				d.addstr(&l.to_stop_moving_unit);
			
			} ActionType::WorkerRmZonesAndBldgs {start_coord: None, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::True, None, roff,
						&l.Select_an_area,
						(&l.Change_location, &l.move_mouse),
						(&l.Start_selecting, &l.click_and_drag),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			} ActionType::WorkerRmZonesAndBldgs {..} => {
				print_mv_to_sel2(full_zoom, PathValid::True, None, roff,
						&l.Select_an_area,
						(&l.Change_location, &l.move_mouse),
						(&l.Confirm_selection, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);

			} ActionType::WorkerZone {valid_placement: false, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::False, None, roff,
						&l.Zoning_instructions,
						(&l.Change_location, &l.move_mouse),
						(&l.Finish_zoning, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			
			} ActionType::WorkerBuildBldg {valid_placement: false, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::False, None, roff,
						&l.Building_instructions,
						(&l.Change_location, &l.drag_the_X),
						(&l.Build, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			
			} ActionType::WorkerBuildBldg {template, valid_placement: true, ..} => {
				let actions_req = if let Some(ut) = ut_opt {
					((actions_req + template.construction_req) / ut.actions_per_turn).ceil()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid), Some((actions_req, turns_est)), roff,
						&l.Building_instructions,
						(&l.Change_location, &l.drag_the_X),
						(&l.Build, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			
			} ActionType::WorkerContinueBuildBldg {..} => {
				// valid placement of cursor
				if let Some(bldg_ind) = worker_can_continue_bldg(true, bldgs, map_data, exf, iface_settings) {
					let actions_req = if let Some(ut) = ut_opt {
						let template = &bldgs[bldg_ind].template;
						((actions_req + template.construction_req) / ut.actions_per_turn).ceil()
					}else{0.};
					
					print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Building_instructions,
						(&l.Change_location, &l.drag_the_X),
						(&l.Resume_construction, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
					return;

				// invalid placement
				}else{
					print_mv_to_sel2(full_zoom,
						PathValid::FalseCustomMsg(&l.Select_edge_to_continue_construction),
						None, roff,
						&l.Building_instructions,
						(&l.Change_location, &l.drag_the_X),
						(&l.Resume_construction, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
				}
			
			} ActionType::WorkerZone {start_coord: Some(start_coord), ..} => {
				let end_coord = iface_settings.cursor_to_map_coord(map_data);
				let start_coord = Coord::frm_ind(*start_coord, map_sz);
				let zone_dim_szs = manhattan_dist_components(end_coord, start_coord, map_sz);
				let zone_sz = ((zone_dim_szs.w + 1) * (zone_dim_szs.h + 1)) as f32;
				
				let actions_req = if let Some(ut) = ut_opt {
					((actions_req + zone_sz) / ut.actions_per_turn).ceil()
				}else{0.};
				
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Zoning_instructions,
						(&l.Change_location, &l.move_mouse),
						(&l.Finish_zoning, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
				
			} ActionType::WorkerZone {..} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Zoning_instructions,
						(&l.Change_location, &l.move_mouse),
						(&l.Start_zoning, &l.click_and_drag),
						kbd, l, disp_chars, iface_settings, txt_list, d);

			} ActionType::WorkerRepairWall {wall_coord: coord_opt, ..} => {
				let actions_req = if let Some(ut) = ut_opt {
					let get_repair_time = || {
						if let Some(coord) = &coord_opt {
							if let Some(ex) = exf.get(coord) {
								if let Some(owner_id) = ex.actual.owner_id {
									if owner_id == cur_player {
										if let Some(s) = &ex.actual.structure {
											if s.structure_type == StructureType::Wall && s.health != std::u8::MAX {
												return (std::u8::MAX - s.health) as f32 /
														ut.repair_wall_per_turn.unwrap() as f32;
											}
										}
									}
								}
							}
						}
						0.
					};
						
					(actions_req / ut.actions_per_turn).ceil() + get_repair_time()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Repair_wall_instructions,
						(&l.Change_dest, &l.drag_the_X),
						(&l.Repair_wall, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);

			} ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Build_gate_instructions,
						(&l.Change_dest, &l.drag_the_X),
						(&l.Build_gate_no_pre_colon, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			
			} ActionType::WorkerBuildStructure {structure_type: StructureType::Road, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Building_instructions,
						(&l.Change_dest, &l.drag_the_X),
						(&l.Build_road, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);

			} ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Building_instructions,
						(&l.Change_dest, &l.drag_the_X),
						(&l.Build_wall, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);

			} ActionType::Attack {..} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						&l.Attacking_instructions,
						(&l.Change_dest, &l.drag_the_X),
						(&format!("{}:", l.Attack), &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			
			} ActionType::Mv | ActionType::MvIgnoreWalls | ActionType::MvIgnoreOwnWalls |
			ActionType::GroupMv {start_coord: Some(_), end_coord: Some(_)} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff, 
						&l.Move_unit_instructions,
						(&l.Change_dest, &l.drag_the_X),
						(&l.Move_unit, &l.stop_dragging),
						kbd, l, disp_chars, iface_settings, txt_list, d);
			
			} ActionType::BrigadeCreation {start_coord: Some(_), end_coord: Some(_), ..} => {
			  
			} ActionType::WorkerBuildStructure {structure_type: StructureType::N, ..} =>
				panicq!("invalid structure in action interface"),
			
			ActionType::CivilianMv {..} | ActionType::Fortify {..} | ActionType::AutoExplore {..} | ActionType::WorkerZoneCoords {..} |
			ActionType::UIWorkerAutomateCity | ActionType::BrigadeCreation {..} |
			ActionType::SectorCreation {..} | ActionType::GroupMv {..} |
			ActionType::BurnBuilding {..} | ActionType::SectorAutomation {..} => 
				panicq!("action shouldn't have action interface")
		}
	}
}

fn print_city_hist(city_nm_show: &str, owner_id: usize,
		stats_row: i32, players: &Vec<Player>, logs: &Vec<Log>,
		l: &Localization, txt_list: &mut TxtList, d: &mut DispState) {
	let mut roff = stats_row; // row offset for printing
	
	macro_rules! mvl {() => (d.mv(roff, UNIT_STATS_COL); roff += 1;);}
	
	macro_rules! print_owner{($id: expr) => {
		let player = &players[$id];
		txt_list.add_b(d);
		set_player_color(player, true, d);
		d.addstr(&player.personalization.nm);
		set_player_color(player, false, d);
	};};
	
	let mut max_width = 0;
	
	for log in logs.iter() {
		match &log.val {
			LogType::CityFounded {owner_id, city_nm} => {
				if city_nm == city_nm_show {
					mvl!();
					txt_list.add_b(d);
					l.print_date_log(log.turn, d);
					d.addstr("Founded by the ");
					print_owner!(*owner_id);
					d.addstr(" civilization.");
					
					let width = l.date_str(log.turn).len() +
						"Founded by the ".len() +
						" civilization.".len() +
						players[*owner_id].personalization.nm.len();
					
					if width > max_width {max_width = width;}
				}
			}
			LogType::CityCaptured {city_attackee_nm, owner_attacker_id, ..} => {
				if city_attackee_nm == city_nm_show {
					mvl!();
					txt_list.add_b(d);
					l.print_date_log(log.turn, d);
					d.addstr("Captured by the ");
					print_owner!(*owner_attacker_id);
					d.addstr(" civilization.");
					
					let width = "Captured by the ".len() +
						" civilization.".len() +
						players[*owner_attacker_id].personalization.nm.len();
					
					if width > max_width {max_width = width;}
				}
			}
			LogType::CityDestroyed {city_attackee_nm, owner_attacker_id, ..} => {
				if city_attackee_nm == city_nm_show {
					mvl!();
					txt_list.add_b(d);
					l.print_date_log(log.turn, d);
					d.addstr("Destroyed by the ");
					print_owner!(*owner_attacker_id);
					d.addstr(" civilization.");
					
					let width = "Destroyed by the ".len() +
						" civilization.".len() +
						players[*owner_attacker_id].personalization.nm.len();
					
					if width > max_width {max_width = width;}
				}
			}
			LogType::NobleHouseJoinedEmpire {house_id, empire_id} => {
				// house_joined_empire_abbrev: "Joined the [empire_nm] empire."
				if owner_id == *house_id {
					mvl!();
					txt_list.add_b(d);
					l.print_date_log(log.turn, d);
					
					let empire = &players[*empire_id].personalization;
					
					let tags = vec![KeyValColor {
						key: String::from("[empire_nm]"),
						val: empire.nm.clone(),
						attr: COLOR_PAIR(empire.color)
					}];
					
					color_tags_print(&l.house_joined_empire_abbrev, &tags, None, d);
					let width = color_tags_txt(&l.house_joined_empire_abbrev, &tags).len();
					
					if width > max_width {max_width = width;}
				}
			}
			
			LogType::CivCollapsed {..} | LogType::CivDestroyed {..} |
			LogType::UnitDestroyed {..} | LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} | LogType::CityDisbanded {..} |
			LogType::ICBMDetonation {..} | LogType::PrevailingDoctrineChanged {..} |
			LogType::CivDiscov {..} | LogType::UnitAttacked {..} | 
			LogType::StructureAttacked {..} | LogType::WarDeclaration {..} |
			LogType::Rioting {..} | LogType::RiotersAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::PeaceDeclaration {..} | LogType::Debug {..} => {}
		}
	}
	
	// print name of city (centered)
	d.mv(stats_row - 2, UNIT_STATS_COL + if max_width >= city_nm_show.len()
			{(max_width - city_nm_show.len())/2} else {0} as i32);
	txt_list.add_b(d);
	d.addstr(&format!("{}", city_nm_show));
}

enum PathValid<'l> {
	False,
	FalseCustomMsg(&'l str),
	True,
}

impl <'l>PathValid<'l> {
	fn from(path_valid: bool) -> Self {
		if path_valid {PathValid::True} else {PathValid::False}
	}
	
	fn is_true(&self) -> bool {
		match self {
			PathValid::True => true,
			PathValid::False | PathValid::FalseCustomMsg(_) => false
		}
	}
}

// `turns_req`: (turns_req, est)
// `change` and `confirm` tuples: (label on left most column, mouse instructions on next column over)
fn print_mv_to_sel2<'l>(full_zoom: bool, path_valid: PathValid<'l>, turns_req: Option<(f32, bool)>, roff: &mut i32,
		title: &str, change: (&str, &str), mut confirm: (&'l str, &'l str), kbd: &KeyboardMap, l: &'l Localization,
		disp_chars: &DispChars, iface_settings: &IfaceSettings, txt_list: &mut TxtList, d: &mut DispState) {
	macro_rules! color_txt {($nm: expr, $color: expr) => {
		d.attron(COLOR_PAIR($color)); d.addstr($nm); d.attroff(COLOR_PAIR($color))}};
	
	let esc_txt = key_txt(kbd.esc, l);
	let mut confirm_key = '\n' as u8 as i32;
	
	// check if we should show `Zoom in` instead of confirmation text
	if !full_zoom {
		confirm = (&l.Zoom_in, &l.mouse_wheel);
		confirm_key = kbd.zoom_in;
	}
	
	let confirm_kbd_len = key_txt(confirm_key, l).len();
	
	/*
							title
							
					 |     mouse    |   keyboard    (titles)   row 0
	change.0                 |   change.1   | a, s, d, w    (change)   row 1
	confirm.0                |  confirm.1   |   <Enter>     (confirm)  row 2 -- this row may be `Zoom in:`
	Cancel:                  |  right click |    <Esc>      (cancel)   row 3
	
	*/
	
	// now we define the columns (to take the max down the rows to get the needed printing width)
	//              (titles)            (change)   (confirm)          (cancel)
	let lbls =      ["",                change.0,  confirm.0,         &l.Cancel_colon];
	let mouse_txt = [&l.mouse,          change.1,  confirm.1,         &l.right_click];
	let kbd_lens =  [l.keyboard.len(),  3*3+3,     confirm_kbd_len,   esc_txt.len()];
	
	let lbl_w = lbls.iter().max_by_key(|l| l.len()).unwrap().len() as i32 + 2;
	let mouse_w = mouse_txt.iter().max_by_key(|m| m.len()).unwrap().len() as i32 + 3;
	let kbd_w = *kbd_lens.iter().max().unwrap() as i32 + 2;
	
	// title
	d.mv(*roff-2, UNIT_STATS_COL + (lbl_w + mouse_w + kbd_w - title.len() as i32)/2);
	txt_list.add_b(d);
	color_txt!(title, CGREEN);
	
	// path blocked
	if !path_valid.is_true() && full_zoom {
		d.mv(*roff, UNIT_STATS_COL);
		if let PathValid::FalseCustomMsg(txt) = path_valid {
			color_txt!(txt, CRED);
		}else{
			color_txt!(&l.Current_location_blocked, CRED);
		}
		*roff += 1;
	}
	
	macro_rules! center{($txt: expr, $w: expr) => {
		let gap = ($w - $txt.len() as i32)/2;
		for _ in 0..gap {d.addch(' ');}
		d.addstr($txt);
	};};
	
	// adds to txt list for screen readers
	macro_rules! center_log{($txt: expr, $w: expr) => {
		let gap = ($w - $txt.len() as i32)/2;
		for _ in 0..gap {d.addch(' ');}
		txt_list.add_b(d);
		d.addstr($txt);
	};};

	
	let v = disp_chars.vline_char;
	
	let col2 = UNIT_STATS_COL + lbl_w;
	let col3 = col2 + mouse_w;
	
	// line 1 lbls (mouse, keyboard)
	{
		d.mv(*roff, col2);
		//d.addch(v);
		d.attron(COLOR_PAIR(CGRAY));
		center!(&l.mouse, mouse_w);
		d.attroff(COLOR_PAIR(CGRAY));
		
		d.mv(*roff, col3);
		//d.addch(v);
		d.attron(COLOR_PAIR(CGRAY));
		center!(&l.keyboard, kbd_w);
		d.attroff(COLOR_PAIR(CGRAY));
		*roff += 1;
	}
	
	let colors = [CWHITE, CWHITE, CGREEN, ESC_COLOR]; // for mouse instructions
	let show_confirm_row = path_valid.is_true() || !full_zoom;
	
	macro_rules! vline{($col: expr) => {
		d.mv(*roff, $col);
		d.attron(COLOR_PAIR(CGRAY));
		d.addch(v);
		d.attroff(COLOR_PAIR(CGRAY));
	};};
	
	// columns 1 & 2 for all remaining rows (left-most column and mouse instructions)
	for (row_ind, ((lbl, mouse), color)) in lbls.iter()
					.zip(mouse_txt.iter())
					.zip(colors.iter()).enumerate().skip(1) {
		// if invalid path and full zoom, don't show confirmation action
		if row_ind == 2 && !show_confirm_row {continue;}
		
		// label
		d.mv(*roff, UNIT_STATS_COL + lbl_w - lbl.len() as i32 - 1);
		txt_list.add_b(d);
		d.addstr(lbl);
		vline!(col2);
		
		// mouse instructions
		d.attron(COLOR_PAIR(*color));
		center!(mouse, mouse_w);
		d.attroff(COLOR_PAIR(*color));
		vline!(col3);
		
		// last column (keyboard instructions)
		match row_ind {
			// a,s,d,w
			1 => {
				d.addch(' ');
				txt_list.add_b(d);
				for k in &[kbd.left, kbd.down, kbd.right] {
					iface_settings.print_key(*k, l, d);
					d.addstr(", ");
				}
				iface_settings.print_key(kbd.up, l, d);
			
			// <Enter> or i
			} 2 => {
				if confirm_key == '\n' as u8 as i32 {
					d.attron(COLOR_PAIR(CGREEN));
					center_log!(&l.Enter_key, kbd_w);
					d.attroff(COLOR_PAIR(CGREEN));
				}else{
					d.mv(*roff, col3 + (kbd_w - 1)/2);
					txt_list.add_b(d);
					iface_settings.print_key(confirm_key, l, d);
				}
			// <Esc>
			} 3 => {
				d.attron(COLOR_PAIR(ESC_COLOR));
				center_log!("<Esc>", kbd_w);
				d.attroff(COLOR_PAIR(ESC_COLOR));
			} _ => {panicq!("unsupported # of rows");}
		}
		
		*roff += 1;
	}
	
	// turns req
	if let Some((turns_req, est)) = turns_req {
		if turns_req != 0. {
			let len = (l.Days_required.len() + if est {l.est.len()} else {0}) as i32;
			d.mv(*roff, UNIT_STATS_COL + (lbl_w + mouse_w + kbd_w - len)/2);
			d.attron(COLOR_PAIR(CGRAY));
			d.addstr(&format!("{} {}", l.Days_required, turns_req));
			if est {
				d.addstr(&format!(" {}", l.est));
			}
			d.attroff(COLOR_PAIR(CGRAY));
		}
	}
}

/*fn print_mv_to_sel(full_zoom: bool, path_valid: bool, turns_req: f32, turns_est: bool, nm: &str, nm_cancel: &str, chg_location: Option<&str>, roff: &mut i32, kbd: &KeyboardMap, l: &Localization, d: &mut DispState) {
	macro_rules! mvl {() => (d.mv(*roff, UNIT_STATS_COL); *roff += 1;);}
	
	macro_rules! action_key {
		  ($nm: expr, $color: expr) => {mvl!(); d.attron(COLOR_PAIR($color)); d.addstr($nm); d.attroff(COLOR_PAIR($color))};
		  ($nm: expr, $color: expr, $last: expr) => {d.mv(*roff, UNIT_STATS_COL); d.attron(COLOR_PAIR($color)); d.addstr($nm); d.attroff(COLOR_PAIR($color))};};
	
	// path blocked
	if !path_valid && full_zoom {
		d.attron(COLOR_PAIR(CRED)); mvl!();
		d.addstr(&l.Current_location_blocked);
		d.attroff(COLOR_PAIR(CRED));
	}
	
	// mv cursor to change dest
	if let Some(chg_location) = chg_location {
		action_key!(&l.Move_cursor, CGREEN);
		if chg_location.len() != 0 {
			if chg_location == "selection" {
				d.addstr(" to change selection");
			}else if chg_location == "destination" {
				d.addstr(" to change destination");
			}else{
				d.addstr(&format!(" to change {} location", chg_location));
			}
		}else{
			d.addstr(" to change location");
		}
	}
	
	// enter to confirm
	if full_zoom && path_valid {
		action_key!("<Enter>", CGREEN);
		d.addstr(&format!(" to {}", nm));
	// <i> to zoom
	}else if !full_zoom {
		action_key!(&format!("<{}>", key_txt(kbd.zoom_in, l)), CGREEN);
		d.addstr(" to zoom in");
	}
	
	action_key!("<Esc>", ESC_COLOR);
	d.addstr(&format!(" to cancel {}", nm_cancel));
	mvl!(); mvl!();
	
	if turns_req != 0. {
		d.addstr(&format!("{} {}", l.Days_required, turns_req));
		if turns_est {
			d.addstr(&format!(" {}", l.est));
		}
	}
}*/

impl IfaceSettings<'_,'_,'_,'_,'_> {
	// cmds: Vec<(key, txt)>
	fn print_cmds(&self, mut cmds: Vec<&mut Button>, roff: i32, l: &Localization, txt_list: &mut TxtList, d: &mut DispState) {
		const ROWS_PER_COL: usize = 5;
		let mut col_offset = UNIT_STATS_COL;
		for col_cmds in cmds.chunks_mut(ROWS_PER_COL) {
			let mut max_w = 0;
			for (cmd_ind, button) in col_cmds.iter_mut().enumerate() {
				d.mv(roff + cmd_ind as i32, col_offset);
				txt_list.add_b(d);
				let w = button.print(Some(self), l, d) + 1;
				if w > max_w {max_w = w;}
			}
			
			// shift over offset for next column
			col_offset += (max_w + 1) as i32;
		}
	}

	fn print_owner(&self, mut roff: i32, player: &Player, relations: &Relations,
			txt_list: &mut TxtList, l: &Localization, d: &mut DispState) {
		match player.ptype {
			PlayerType::Human(_) | PlayerType::Empire(_) => {
				d.mv(roff, LAND_STATS_COL); roff += 1;
				txt_list.add_b(d);
				d.addstr(&l.Country);
				d.addstr(": ");
			}
			PlayerType::Barbarian(_) => {}
			PlayerType::Nobility(_) => {
				d.mv(roff, LAND_STATS_COL); roff += 1;
				txt_list.add_b(d);
				set_player_color(player, true, d);
				d.addstr(&l.only_House_of);
				set_player_color(player, false, d);
			}
		}
		
		d.mv(roff, LAND_STATS_COL);
		txt_list.add_b(d);
		set_player_color(player, true, d);
		d.addstr(if player.id == self.cur_player {"You"} else {&player.personalization.nm});
		set_player_color(player, false, d);
		
		if self.cur_player != player.id && relations.at_war(self.cur_player as usize, player.id as usize) {
			d.mv(roff + 1, LAND_STATS_COL);
			txt_list.add_b(d);
			d.addch('(');
			d.attron(COLOR_PAIR(CRED));
			d.addstr("WAR");
			d.attroff(COLOR_PAIR(CRED));
			d.addch(')');
		}
	}
	
	fn zoom_in_to_change_actions(&self, roff: i32, kbd: &KeyboardMap, l: &Localization, txt_list: &mut TxtList, d: &mut DispState) {
		d.mv(roff, UNIT_STATS_COL);
		txt_list.add_b(d);
		d.addstr("Zoom ");
		
		let combine_w_word = kbd.zoom_in == 'i' as i32;
		
		if combine_w_word {
			self.print_key(kbd.zoom_in, l, d);
		}else{
			d.addch('i');
		}
		d.addstr("n to change unit's actions");
		if !combine_w_word {
			d.addch('(');
			self.print_key(kbd.zoom_in, l, d);
			d.addch(')');
		}
	}
	
	fn print_unit_nm_health(&self, roff: &mut i32, unit_ind: usize, u: &Unit, pstats: &Stats, players: &Vec<Player>,
			l: &Localization, buttons: &mut Buttons, txt_list: &mut TxtList, d: &mut DispState) {
		macro_rules! mvl {() => (d.mv(*roff, UNIT_STATS_COL); *roff += 1;);}
		
		// print nm
		{
			mvl!();
			if u.template.nm[0] != RIOTER_NM {
				d.mv(*roff-2, UNIT_STATS_COL);
				txt_list.add_b(d);
				d.addstr(&l.The_Battalion.replace("[nm]", &u.nm).replace("[type]", &u.template.nm[l.lang_ind]));				
				
				if self.add_action_to.is_none() {
					if let Some(brigade_nm) = pstats.unit_brigade_nm(unit_ind) {
						d.mv(*roff-1, UNIT_STATS_COL);
						txt_list.add_b(d);
						d.addstr(&l.Member_of_the_Brigade.replace("[]", brigade_nm));
						d.addstr(" (");
						buttons.view_brigade.print(Some(self), l, d);
						d.addstr(", ");
						buttons.leave_brigade.print(Some(self), l, d);
						d.addstr(")");
					}
				}
				
			}else{
				txt_list.add_b(d);
				d.addstr(&u.template.nm[0]);
			}
		}
		
		// print health
		{
			mvl!(); txt_list.add_b(d); d.addstr(&format!("{}: ", l.Health));
			let health_frac = u.health();
			colorize(health_frac, true, d);
			if health_frac > 1. {
				d.addstr(&format!("{}%", health_frac.round()));
			}else{
				d.addstr(&format!("{:.2}%", health_frac));
			}
			colorize(health_frac, false, d);
		}
		
		//// show boarded units
		if let Some(units_carried) = &u.units_carried {
			if units_carried.len() > 0 && self.add_action_to.is_none() {
				debug_assertq!(u.template.carry_capac >= units_carried.len());
				
				let col2 = ("Action: Fortified   ".chars().count() as i32) + UNIT_STATS_COL;
				
				d.mv(*roff, col2); txt_list.add_b(d);
				d.addstr(&l.Carrying); d.addch(' ');
				for (i, c) in units_carried.iter().enumerate() {
					plot_unit(c.template, &players[c.owner_id as usize], d);
					if i == (units_carried.len()-1) {break;}
					d.addstr(", ");
				}
			}
		}
		
		d.mv(*roff, UNIT_STATS_COL); *roff += 2;
	}
	
	// if in mv mode
	fn print_unit_action(&self, mut roff: i32, action_iface: &ActionInterfaceMeta, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx, map_data: &mut MapData, pstats: &Stats, players: &Vec<Player>, kbd: &KeyboardMap, l: &Localization,
			buttons: &mut Buttons, disp_chars: &DispChars, txt_list: &mut TxtList, d: &mut DispState) {
		if let Some(unit_ind) = action_iface.unit_ind {
			let u = &units[unit_ind];
			
			self.print_unit_nm_health(&mut roff, unit_ind, u, pstats, players, l, buttons, txt_list, d);
			
			let actions_req = if let Some(ActionMetaCont {final_end_coord, ..}) = &action_iface.action.action_meta_cont {
				let map_sz = *map_data.map_szs.last().unwrap();
				manhattan_dist(*final_end_coord, Coord::frm_ind(u.return_coord(), map_sz), map_sz) as f32
			}else{
				action_iface.action.actions_req
			};
			
			let path_valid = action_iface.action.path_coords.len() != 0 ||
					u.return_coord() == self.cursor_to_map_ind(map_data);
			
			let turns_est = !action_iface.action.action_meta_cont.is_none();
			
			action_iface.action.action_type.print(&mut roff, path_valid, actions_req, turns_est, Some(u.template), bldgs, exf, map_data, self, l, kbd, disp_chars, txt_list, d);
		}
	}
	
	// possible actions
	fn print_broadcastable_actions(&self, brigade_nm: &String, pstats: &Stats, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx,
			stats_row: i32, map_data: &mut MapData, kbd: &KeyboardMap, l: &Localization,
			buttons: &mut Buttons, disp_chars: &DispChars, txt_list: &mut TxtList, d: &mut DispState) {
		let brigade = pstats.brigade_frm_nm(brigade_nm);
		let mut roff = stats_row;
		macro_rules! mvl {() => (d.mv(roff, UNIT_STATS_COL); roff += 1;);}
		d.mv(roff-1, UNIT_STATS_COL);
		buttons.Cancel.print(Some(self), l, d);
		
		mvl!(); txt_list.add_b(d);
		d.addstr(&l.Choose_an_action_for_all_brigade_units.replace("[]", brigade_nm));
		mvl!();
		
		// show current active action
		if let AddActionTo::AllInBrigade {action_ifaces: Some(action_ifaces), ..} = &self.add_action_to {
			if let Some(action_iface) = action_ifaces.first() {
				let path_valid = action_ifaces.iter().any(|af| af.action.path_coords.len() != 0);
				let actions_req = 0.;
				let turns_est = false;
				let ut_opt = None;
				mvl!();
				action_iface.action.action_type.print(&mut roff, path_valid, actions_req, turns_est, ut_opt, bldgs, exf, map_data, self, l, kbd, disp_chars, txt_list, d);
			}else{panicq!("no active action_ifaces in AllInBrigade move");}
		
		// show possible actions
		}else if self.zoom_ind == map_data.max_zoom_ind() {
			let mut cmds = Vec::with_capacity(30);	
			macro_rules! add{($($button_nm: ident),*) => {$(cmds.push(&mut buttons.$button_nm);)*};};
			
			add!(move_unit, fortify, pass_move, auto_explore, disband);
			
			// soldier actions
			if brigade.unit_inds.iter().any(|&ind| !units[ind].template.attack_per_turn.is_none()) {
				add!(attack, soldier_automate);
			}
			
			// worker actions
			if brigade.unit_inds.iter().any(|&ind| units[ind].template.nm[0] == WORKER_NM) {
				add!(automate_zone_creation, continue_bldg_construction);//, build_bldg);
			}
			
			// repair wall
			if brigade.unit_inds.iter().any(|&ind| units[ind].template.repair_wall_per_turn != None) {
				add!(repair_wall);
			}
			
			// unload boat
			if brigade.unit_inds.iter().any(|&ind| {
				if let Unboard::Loc {..} = unboard_land_adj(ind, units, bldgs, map_data, exf)
					{true} else {false}
			}) {add!(unload_boat);}
			
			self.print_cmds(cmds, roff, l, txt_list, d);
		// zoom in to change actions
		}else{
			self.zoom_in_to_change_actions(roff, kbd, l, txt_list, d);
		}
	}
	
	// possible actions
	fn print_build_list_actions(&self, brigade_nm: &String, pstats: &Stats, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx,
			stats_row: i32, map_data: &mut MapData, kbd: &KeyboardMap, l: &Localization,
			buttons: &mut Buttons, disp_chars: &DispChars, txt_list: &mut TxtList, d: &mut DispState) {
		let brigade = pstats.brigade_frm_nm(brigade_nm);
		let mut roff = stats_row;
		macro_rules! mvl {() => (d.mv(roff, UNIT_STATS_COL); roff += 1;);}
		d.mv(roff-1, UNIT_STATS_COL);
		buttons.Cancel.print(Some(self), l, d);
		
		mvl!(); txt_list.add_b(d);
		d.addstr(&l.Choose_an_action_to_add_to_the_brigade_build_list.replace("[]", brigade_nm));
		mvl!();
		
		// show current active action
		if let AddActionTo::BrigadeBuildList {action: Some(action), ..} = &self.add_action_to {
			let path_valid = true;
			let actions_req = 0.;
			let turns_est = false;
			let ut_opt = None;
			mvl!();
			action.action_type.print(&mut roff, path_valid, actions_req, turns_est, ut_opt, bldgs, exf, map_data, self, l, kbd, disp_chars, txt_list, d);
		// show possible actions
		}else if self.zoom_ind == map_data.max_zoom_ind() {
			let mut cmds = Vec::with_capacity(30);
			macro_rules! add{($($button_nm: ident),*) => {$(cmds.push(&mut buttons.$button_nm);)*};};
			
			add!(zone_agricultural, zone_residential, zone_business, zone_industrial, build_bldg);
					
			// repair wall
			if brigade.unit_inds.iter().any(|&ind| units[ind].template.repair_wall_per_turn != None) {
				add!(repair_wall);
			}
			
			add!(build_gate);
			
			self.print_cmds(cmds, roff, l, txt_list, d);
		// zoom in to change actions
		}else{
			self.zoom_in_to_change_actions(roff, kbd, l, txt_list, d);
		}
	}
	
	// health, nm, possible actions
	fn print_unit_stats(&self, unit_inds: &Vec<usize>, stats_row: i32, lside_row: i32,
			show_land: bool, player: &Player, players: &Vec<Player>, units: &Vec<Unit>, 
			bldgs: &Vec<Bldg>, map_data: &mut MapData, exf: &HashedMapEx, 
			relations: &Relations, kbd: &KeyboardMap, l: &Localization,
			buttons: &mut Buttons, disp_chars: &DispChars, txt_list: &mut TxtList, d: &mut DispState){
		
		//debug_assertq!(self.zoom_ind == map_data.max_zoom_ind());
		//debug_assertq!(unit_inds.len() <= MAX_UNITS_PER_PLOT);
		debug_assertq!(unit_inds.len() > 0);
		
		let mut roff = stats_row;
		
		/////////
		// multi-unit display
		if show_land && unit_inds.len() != 1 {
			d.mv(roff - 2, UNIT_STATS_COL);
			txt_list.add_b(d);
			d.addstr(&format!("{}: ", l.Units));
			for unit_ind_ind in 0..unit_inds.len() {
				if unit_ind_ind > 0 {d.addstr(", ");}
			
				let u = &units[unit_inds[unit_ind_ind]];
				if unit_ind_ind == self.unit_subsel {d.attron(A_UNDERLINE());}
				plot_unit(u.template, &players[u.owner_id as usize], d);
				if unit_ind_ind == self.unit_subsel {d.attroff(A_UNDERLINE());}
			}
			
			if self.ui_mode.is_none() && self.add_action_to.is_none() {
				d.addch(' ');
				buttons.tab.print_key_only(Some(self), l, d);
			}
		} // end multi-unit disp
		
		let unit_ind = *unit_inds.get(self.unit_subsel).unwrap_or_else(|| &unit_inds[0]);
		let u = &units[unit_ind];
			
		// in action mode
		if let AddActionTo::IndividualUnit {action_iface} = &self.add_action_to {
			self.print_unit_action(roff, action_iface, units, bldgs, exf, map_data, &player.stats, players, kbd, l, buttons, disp_chars, txt_list, d);
		
		// not interactively moving or building anything -- show actions unit could perform
		}else if show_land {
			self.print_unit_nm_health(&mut roff, unit_ind, u, &player.stats, players, l, buttons, txt_list, d);
			
			// player's unit
			if u.owner_id == self.cur_player || self.show_actions {
				txt_list.add_b(d);
				d.addstr(&format!("{}: ", l.Action));
				if let Some(action) = u.action.last() {
					d.addstr(&action.action_type.nm(l));
					/*d.addstr(&format!("{}", action.path_coords.len()));
					if let Some(action_meta_cont) = &action.action_meta_cont {
						d.addstr(&format!(" checkpoint {}", action_meta_cont.checkpoint_path_coords.len()));
					}*/
				}else{
					d.addstr(&l.Idle);
				}
				
				if let Some(actions_used) = u.actions_used {
					if actions_used != 0. {
						d.addstr(&format!(" ({}/{})", actions_used, u.template.actions_per_turn));
					}
				}else{
					d.addch(' ');
					d.addstr(&l.no_actions_remain);
				}
				
				// show possible actions or instructions to zoom in
				if self.ui_mode.is_none() && (u.template.nm[0] != RIOTER_NM || self.show_actions) {
					// show possible actions
					if self.zoom_ind == map_data.max_zoom_ind() && !u.actions_used.is_none() {
						let mut cmds = Vec::with_capacity(30);
						macro_rules! add{($($button_nm: ident),*) => {$(cmds.push(&mut buttons.$button_nm);)*};};
						
						add!(move_unit, move_with_cursor);
						
						{ // fortify
							if let Some(action) = u.action.last() {
								if let ActionType::Fortify {..} = action.action_type {} else {
									add!(fortify);
								}
							}else{add!(fortify);}
						}
						
						add!(pass_move, auto_explore, disband);
						
						// leave brigade
						if player.stats.unit_brigade_nm(unit_ind).is_none() && player.stats.brigades.len() != 0 {
							add!(join_brigade);
						}
						
						add!(group_move);
						
						// worker
						if u.template.nm[0] == WORKER_NM {
							add!(zone_agricultural, zone_residential, zone_business, zone_industrial, 
								automate_zone_creation, rm_bldgs_and_zones, continue_bldg_construction, build_bldg, build_road);
							
							// can only build wall if this worker is the only
							// one on the tile
							if unit_inds.len() == 1 {add!(build_wall);}
							add!(build_gate);
							
							// repair wall
							if u.template.repair_wall_per_turn != None {add!(repair_wall);}
						
						// soldier actions
						}else if let Some(_) = u.template.attack_per_turn {
							add!(attack, soldier_automate);
						}
						
						// show option to unboard
						if let Some(_) = u.units_carried {
							if let Unboard::Loc {..} = unboard_land_adj(unit_ind, units, bldgs, map_data, exf) {
								add!(unload_boat);
							}
						}
						
						self.print_cmds(cmds, roff, l, txt_list, d);
						
					// zoom in to change actions
					}else if self.zoom_ind != map_data.max_zoom_ind() {
						self.zoom_in_to_change_actions(roff, kbd, l, txt_list, d);
					}
				} // menu/windows not active
			// currently owned unit
			}/*else{ // someone else's unit
				let o = &owners[u.owner_id as usize];
				if o.player_type != PlayerType::Barbarian {
					d.addstr("Country: ");
				}
				set_player_color(o, true);
				d.addstr(&format!("{}", o.nm));
				set_player_color(o, false);
			}*/
		} // unit stats
		
		// Country owner
		if show_land && u.owner_id != self.cur_player {
			self.print_owner(lside_row, &players[u.owner_id as usize], relations, txt_list, l, d);
		}
	}
	
	fn print_unit_bldg_stats(&self, map_cur_coord: u64, stats_row: i32, lside_row: i32, show_land: bool,
			map_data: &mut MapData, exs: &Vec<HashedMapEx>, player: &Player, players: &Vec<Player>,
			units: &Vec<Unit>, bldg_config: &BldgConfig, bldgs: &Vec<Bldg>, relations: &Relations,
			logs: &Vec<Log>, kbd: &KeyboardMap, l: &Localization, buttons: &mut Buttons, disp_chars: &DispChars,
			txt_list: &mut TxtList, d: &mut DispState){
		let pstats = &player.stats;
		//if self.zoom_ind != map_data.max_zoom_ind() {return;} // only show at full zoom
		
		// if in IndividualUnit move mode:
		//	show stats of unit proposed to make the move (rather than showing the units at the current cursor coord)
		let get_cursor_or_sel_coord = || {
			if let AddActionTo::IndividualUnit {action_iface} = &self.add_action_to {
				let start_coord = action_iface.start_coord;
				((start_coord.y as usize)*map_data.map_szs[self.zoom_ind].w + (start_coord.x as usize)) as u64
			}else{
				map_cur_coord
			}
		};
		
		let exz = &exs[self.zoom_ind];
		let exf = exs.last().unwrap();
		
		// show brigade
		if let AddActionTo::AllInBrigade {brigade_nm, ..} = &self.add_action_to {
			self.print_broadcastable_actions(brigade_nm, pstats, units, bldgs, exf, stats_row, map_data, kbd, l, buttons, disp_chars, txt_list, d);
		}else if let AddActionTo::BrigadeBuildList {brigade_nm, ..} = &self.add_action_to {
			self.print_build_list_actions(brigade_nm, pstats, units, bldgs, exf, stats_row, map_data, kbd, l, buttons, disp_chars, txt_list, d);

		// ex data
		}else if let Some(ex) = exz.get(&get_cursor_or_sel_coord()) {
			// show zoomed out city history
			if show_land {
				if let Some(fog) = self.get_fog_or_actual(get_cursor_or_sel_coord(), ex, pstats) {
					if let Some(max_city_nm) = &fog.max_city_nm {
						if let Some(owner_id) = fog.owner_id {
							print_city_hist(max_city_nm, owner_id as usize, stats_row, players, logs, l, txt_list, d);
						}
						return;
					}
				}
			}
			
			// show unit
			if let Some(unit_inds) = &ex.unit_inds {
				self.print_unit_stats(unit_inds, stats_row, lside_row, show_land, player, players, units, bldgs, map_data, exz, relations, kbd, l, buttons, disp_chars, txt_list, d);
				return;
				
			// show bldg (full zoom)
			}else if let Some(bldg_ind) = ex.bldg_ind {
				if show_land {
					let b = &bldgs[bldg_ind];
					let bt = b.template;
					//let ex = exz.get(&b.coord).unwrap();
					
					let mut roff = stats_row - 2; // row offset for printing
					
					macro_rules! mvl {() => (d.mv(roff, UNIT_STATS_COL); roff += 1;);}	
					d.mv(roff, UNIT_STATS_COL); roff += 2;
					
					// print name of bldg
					// if city hall & we're not showing all player's actions, show history and return
					if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
						if b.owner_id == self.cur_player || self.show_actions {
							txt_list.add_b(d);
							d.addstr(&format!("{} ({})", nm, bt.nm[l.lang_ind]));
						}else{
							print_city_hist(nm, b.owner_id as usize, stats_row, players, logs, l, txt_list, d);
							return;
						}
					}else{
						txt_list.add_b(d);
						d.addstr(&bt.nm[l.lang_ind]);
						
						// print "(abandoned)"?
						if let BldgType::Taxable(_) = b.template.bldg_type {
							if b.n_residents() == 0 {addstr_c(&l.abandoned, CGRAY, d);}
						}
					}
					
					// print damage
					if let Some(damage) = b.damage {
						d.mv(roff - 1, UNIT_STATS_COL);
						txt_list.add_b(d);
						d.addstr(&l.Damage);
						let damage_frac = (damage as f32 / bldg_config.max_bldg_damage as f32) * 100.;
						colorize(100.-damage_frac, true, d);
						d.addstr(&format!(" {:.1}%", damage_frac));
						colorize(100.-damage_frac, false, d);
					}
					
					mvl!();
					
					//////////////////////////////////
					// gov & taxable bldg specific printing
					match b.template.bldg_type {
						BldgType::Gov(_) => {
							// under construction?
							if let Some(prog) = b.construction_done {
								txt_list.add_b(d);
								d.addstr(&format!("{} {}%", l.Construction_progress,
										((100.*prog as f32 / bt.construction_req as f32).round() as usize)));
								return;
							}
							
							// print production and taxes
							if !b.template.units_producable.is_none() && (b.owner_id == self.cur_player || self.show_actions) {
								// constructed
								if let BldgArgs::PopulationCenter {tax_rates, ..} = &b.args {
									txt_list.add_b(d);
									d.addstr(&l.Taxes);
									
									macro_rules! tax_ln{($shortcut_key: expr, $nm: expr, $zt: path) => (
										mvl!();
										if let UIMode::SetTaxes($zt) = self.ui_mode {
											d.addstr(&format!("   {}", $nm));
										}else{
											$shortcut_key.print(Some(self), l, d);
										}
										d.addstr(&format!(" : {}% ", tax_rates[$zt as usize]));
										
										if let UIMode::SetTaxes($zt) = self.ui_mode {
											buttons.increase_tax.print_key_only(Some(self), l, d);
											
											buttons.increase_tax_alt.pos = buttons.increase_tax.pos.clone();
											// ^ so that it's also on screen and it's key will be potentially
											//   seen as active when checked in gcore/non_menu_keys.rs
											
											d.addstr(" | ");
											buttons.decrease_tax.print_key_only(Some(self), l, d);
										}
									);}
									tax_ln!(buttons.tax_agricultural, &l.Agricultural, ZoneType::Agricultural);
									tax_ln!(buttons.tax_residential, &l.Residential, ZoneType::Residential);
									tax_ln!(buttons.tax_business, &l.Business, ZoneType::Business);
									tax_ln!(buttons.tax_industrial, &l.Industrial, ZoneType::Industrial);
									roff += 1;
								}
								
								// production
								mvl!();
								txt_list.add_b(d);
								d.addstr(&l.Production); d.mv(roff, UNIT_STATS_COL); /////!!!!!!!!!!!!!! (roff not incremented or else you get a compiler warning)
								buttons.change_bldg_production.print(Some(self), l, d);
								
								if let BldgArgs::PopulationCenter {production, ..} | BldgArgs::GenericProducable {production} = &b.args {
									if let Some(production_entry) = production.last() {
										let ut = production_entry.production;
										txt_list.add_b(d);
										d.addstr(&format!("{}  ({}/{})", &ut.nm[l.lang_ind],
											production_entry.progress, ut.production_req.round() as usize));
										if production.len() > 1 {
											d.addstr(" (");
											buttons.view_production.print(Some(self), l, d);
											d.addch(' ');
											d.addstr(&l.and_x_more.replace("[]", &format!("{}", production.len()-1)));
											d.addch(')');
										}
									}else{ 
										d.addstr(&l.None);
									}
								}
							}else if b.template.doctrinality_bonus > 0. {
								d.mv(roff, UNIT_STATS_COL);
								txt_list.add_b(d);
								d.addstr(&format!("{}: ", l.Dedication));
								d.addstr(&b.doctrine_dedication.nm[l.lang_ind]);
							}
						/////////////////////////////////////
						// taxable bldg
						} BldgType::Taxable(zone_type) => {
							txt_list.add_b(d);
							d.addstr(&l.City_Hall_dist); d.addch(' ');
							
							let zi = players[b.owner_id as usize].zone_exs.get(&return_zone_coord(b.coord, *map_data.map_szs.last().unwrap())).unwrap();
							match zi.ret_city_hall_dist() {
								Dist::Is {dist, ..} | Dist::ForceRecompute {dist, ..} => {
									d.addstr(&format!("{}", dist)); mvl!();
									
									let taxable_upkeep = b.return_taxable_upkeep();
									if taxable_upkeep != 0. {
										let effective_tax_rate = -100. * taxable_upkeep / (b.template.upkeep * b.operating_frac());
										txt_list.add_b(d);
										d.addstr(&l.Enforced_tax);
										d.addstr(&format!(" {:.1}%", effective_tax_rate));
										mvl!();
									}
									txt_list.add_b(d);
									d.addstr(&l.Tax_payments); d.addch(' ');
									if taxable_upkeep == 0. {
										d.addch('0');
									}else{
										d.addstr(&format!("{:.2}", taxable_upkeep));
									}
								}
								Dist::NotPossible {..} => {
									d.addstr(&l.No_route);
									d.addch(' ');
									d.attron(COLOR_PAIR(CRED));
									d.addstr(&l.build_roads);
									d.attroff(COLOR_PAIR(CRED)); mvl!();
									d.addstr(&l.Enforced_tax);
									d.addstr(" 0%");
								} Dist::NotInit => {d.addstr(&l.Not_yet_determined);}
							}
							
							if b.n_residents() != 0 {
								roff += 1;
								let resident_start_line = roff;
								mvl!();
								debug_assertq!(Some(zone_type) == ex.actual.ret_zone_type());
								let is_residential_zone = ex.actual.ret_zone_type() == Some(ZoneType::Residential);
								
								txt_list.add_b(d);
								d.addstr(if is_residential_zone {&l.Residents} else {&l.Employees});
								d.addch(' ');
								
								d.addstr(&format!("{}/{}", b.n_residents(), bt.resident_max));
								
								mvl!();
								txt_list.add_b(d);
								if is_residential_zone {
									d.addstr(&l.Employed);
									d.addstr(&format!(" {}/{}", b.n_sold(), b.n_residents() ));
								}else{
									d.addstr(&l.Products_sold);
									d.addstr(&format!(" {}/{}", b.n_sold(), b.prod_capac() ));
								}
								
								d.mv(roff, UNIT_STATS_COL);
								txt_list.add_b(d);
								d.addstr(&l.Consumption);
								d.addstr(&format!(" {}/{}", b.cons(), b.cons_capac() ));
								
								if zone_type == ZoneType::Residential {
									let zs = &zi.zone_agnostic_stats;
									{ // Dispositions: doctrine_sum
										d.mv(resident_start_line, UNIT_STATS_COL + 20);
										txt_list.add_b(d);
										d.addstr(&l.Dispositions);
										
										const RANGE: f32 = 0.5*2.;
										const N_STEPS: f32 = 6.;
										const STEP: f32 = RANGE / N_STEPS;
										let val = zs.locally_logged.doctrinality_sum.iter().sum::<f32>();
										let desc = if val < (-RANGE/2.) {&l.Scholar
										}else if val < (-RANGE/2. + STEP) {&l.Artisan
										}else if val < (-RANGE/2. + 2.*STEP) {&l.Literate
										}else if val < (-RANGE/2. + 3.*STEP) {&l.Illiterate
										}else if val < (-RANGE/2. + 4.*STEP) {&l.Adherant
										}else if val < (-RANGE/2. + 5.*STEP) {&l.Reverant
										}else{&l.Devout};
										d.addstr(&format!(" {}", desc));
									}
									
									{ // Politics: pacifism
										d.mv(resident_start_line+1, UNIT_STATS_COL + 20);
										txt_list.add_b(d);
										d.addstr(&l.Politics);
										
										const POL_RANGE: f32 = 0.5*2.;
										const N_STEPS: f32 = 5.;
										const STEP: f32 = POL_RANGE / N_STEPS;
										let val = zs.locally_logged.pacifism_sum;
										let desc = if val < (-POL_RANGE/2.) {&l.Militarist
										}else if val < (-POL_RANGE/2. + STEP) {&l.Interventionist
										}else if val < (-POL_RANGE/2. + 2.*STEP) {&l.Pragmatist
										}else if val < (-POL_RANGE/2. + 3.*STEP) {&l.Peace_minded
										}else{&l.Pacifist};
										d.addstr(&format!(" {}", desc));// {} {} {}", desc, val, POL_RANGE, STEP));
									}
									
									{ // Moods: happiness
										d.mv(resident_start_line+2, UNIT_STATS_COL + 20);
										txt_list.add_b(d);
										d.addstr(&l.Moods);
										
										const RANGE: f32 = 200.*2.;
										const N_STEPS: f32 = 5.;
										const STEP: f32 = RANGE / N_STEPS;
										let val = zs.locally_logged.happiness_sum;
										let desc = if val < (-RANGE/2.) {&l.Treasonous
										}else if val < (-RANGE/2. + STEP) {&l.Rebellious
										}else if val < (-RANGE/2. + 2.*STEP) {&l.Doubtful
										}else if val < (-RANGE/2. + 3.*STEP) {&l.Hopeful
										}else if val < (-RANGE/2. + 4.*STEP) {&l.Content
										}else if val < (-RANGE/2. + 5.*STEP) {&l.Joyful
										}else{&l.Euphoric};
										d.addstr(&format!(" {}", desc));
									}
								}
							}
						} // taxable bldg
					} // match bldg type
				}
			}
		
		// show action (can occur if we are zoomed out and `ex` is empty)
		}else if let AddActionTo::IndividualUnit {action_iface} = &self.add_action_to {
			self.print_unit_action(stats_row as i32, action_iface, units, bldgs, exs.last().unwrap(), map_data, pstats, players, kbd, l, buttons, disp_chars, txt_list, d);
		}
	}
	
	pub fn print_bottom_stats(&self, map_data: &mut MapData, exs: &Vec<HashedMapEx>, player: &Player, players: &Vec<Player>, units: &Vec<Unit>,
			bldg_config: &BldgConfig, bldgs: &Vec<Bldg>, relations: &Relations, logs: &Vec<Log>,
			kbd: &KeyboardMap, l: &Localization, buttons: &mut Buttons, txt_list: &mut TxtList, disp_chars: &DispChars, d: &mut DispState){
		////////////////////////////////////////////////
		// land stats
		let stats_row = (self.screen_sz.h - MAP_ROW_STOP_SZ + 2) as i32;
		
		d.mv(stats_row, LAND_STATS_COL);
		let map_cur_coord = self.cursor_to_map_ind(map_data);
		let mzc = map_data.get(ZoomInd::Val(self.zoom_ind), map_cur_coord);
		
		//d.addstr(&format!("Coord: ({}, {})", (map_cur_coord_unr.x as f32*z), map_cur_coord_unr.y as f32*z));
		d.mv(stats_row + 1, LAND_STATS_COL);
		let mut r_off = stats_row + 4;

		// land is undiscovered
		let show_land = !self.show_fog || player.stats.land_discov[self.zoom_ind].map_coord_ind_discovered(map_cur_coord);
		if !show_land {
			txt_list.add_b(d);
			d.addstr(&l.Undiscovered);
			//return;
		
		// land is discovered
		}else{
			//d.addstr(&format!("Elevation: {}", mzc.elevation));
			
			let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
			let ex_wrapped = exs[self.zoom_ind].get(&map_cur_coord);
			let resource_wrapped = map_data.get(ZoomInd::Full, map_cur_coord).get_resource(map_cur_coord, map_data, map_sz); 
			
			let space_empty_ign_zone = (|| {
				if !self.add_action_to.is_none() {return false;}
				if let Some(ex) = &ex_wrapped {
					ex.bldg_ind.is_none() && ex.actual.max_city_nm.is_none() && ex.actual.max_bldg_template.is_none() && ex.unit_inds.is_none()
				}else {true}
			})();
			
			// show sector
			if self.show_sectors && self.zoom_ind == map_data.max_zoom_ind() {
				if let Some(sector_nm) = player.stats.sector_nm_frm_coord(map_cur_coord, map_sz) {
					d.mv(stats_row-1, LAND_STATS_COL);
					txt_list.add_b(d);
					d.addstr(&l.Sector); d.addch(' ');
					if space_empty_ign_zone {
						d.addstr(&sector_nm);
					// crop sector name
					}else if (UNIT_STATS_COL - LAND_STATS_COL) > (l.Sector.len()+2) as i32 {
						let len = (UNIT_STATS_COL - LAND_STATS_COL) as usize - l.Sector.len() - 2;
						d.addstr(&crop_txt(sector_nm, len));
					}
				}
			}
			
			///// arability
			{
				d.mv(stats_row + 1, LAND_STATS_COL);
				let arability_str = ArabilityType::frm_arability(mzc.arability, mzc.map_type, mzc.show_snow).to_str(l);
				let arability_str_lns: Vec<&str> = arability_str.split(" ").collect();
				
				txt_list.add_b(d);
				d.addstr(arability_str_lns[0]);
				
				// print second line of arability name
				if arability_str_lns.len() > 1 {
					// print on same line (no ex data shown and not moving)
					if ex_wrapped.is_none() && self.add_action_to.is_none() && resource_wrapped.is_none() { 
						d.addch(' ');
					}else{
						txt_list.add_b(d);
						d.mv(stats_row + 2, LAND_STATS_COL);
					}
					d.addstr(&arability_str_lns[1]);
				}
			}
			
			// show resource, road, zone, owner if no bldgs or units
			if space_empty_ign_zone {
				///// resource
				if let Some(resource) = resource_wrapped {
					if player.stats.resource_discov(resource) {
						d.mv(r_off, LAND_STATS_COL); r_off += 1;
						txt_list.add_b(d);
						d.attron(COLOR_PAIR(resource.zone.to_color()));
						d.addstr(&format!("{}", resource.nm[l.lang_ind]));
						d.attroff(COLOR_PAIR(resource.zone.to_color()));
						
						let window_w = 40;
						
						macro_rules! lr_txt{($row: expr, $l_txt: expr, $r_txt: expr) => {
							d.mv($row, UNIT_STATS_COL+2);
							txt_list.add_b(d);
							d.addstr($l_txt);
							d.mv($row, UNIT_STATS_COL + window_w - $r_txt.len() as i32);
							txt_list.add_b(d);
							d.addstr($r_txt);
							$row += 1;
						};};
						
						macro_rules! print_extended_resource_info{() => {
							let mut row = stats_row - 2;
							
							// center title
							d.mv(row, (window_w - resource.nm.len() as i32)/2 + UNIT_STATS_COL as i32); row += 2;
							txt_list.add_b(d);
							d.addstr(&resource.nm[l.lang_ind]);
							
							lr_txt!(row, &l.Zoning_req_to_use, resource.zone.to_str());
							
							// zone bonuses
							for (zone_ind, zone_bonus) in resource.zone_bonuses.iter().enumerate() {
								if let Some(bonus) = zone_bonus {
									if *bonus != 0 {
										lr_txt!(row, &format!("{} bonus:", ZoneType::from(zone_ind).to_str()),
												&format!("{}", bonus));
									}
								}
							}
						};};
						
						if let Some(ex) = &ex_wrapped {
							if ex.unit_inds == None && ex.bldg_ind == None {
								print_extended_resource_info!();
							}
						}else{
							print_extended_resource_info!();
						}
					}
				}
				
				////// road, zone, owner
				d.mv(r_off, LAND_STATS_COL); r_off += 1;
				if let Some(ex) = ex_wrapped {
					// structure
					if let Some(s) = ex.actual.structure {
						txt_list.add_b(d);
						d.addstr(match s.structure_type {
								StructureType::Road => {&l.Road}
								StructureType::Wall => {&l.Wall}
								StructureType::Gate => {&l.Gate}
								StructureType::N => {panicq!("invalid structure type")}
						});
						
						// damaged
						if s.health != std::u8::MAX {
							txt_list.add_b(d);
							d.mv(r_off, LAND_STATS_COL); r_off += 1;
							let health_frac = 100.*(s.health as f32) / (std::u8::MAX as f32); 
							colorize(health_frac, true, d);
							d.addstr(&format!("{:.1}%", 100.-health_frac));
							colorize(health_frac, false, d);
							d.addstr(&l.damaged);
						}
					
					// Zone
					}else if self.zoom_ind == map_data.max_zoom_ind() && !ex.actual.ret_zone_type().is_none() && ex.actual.owner_id == Some(self.cur_player) {
						let zt = ex.actual.ret_zone_type().unwrap();
						txt_list.add_b(d);
						d.addstr(zt.to_str());
					}
					
					/////////////////////////////////////// zone debug info
					#[cfg(any(feature="opt_debug", debug_assertions))]
					if self.zoom_ind == map_data.max_zoom_ind() && !ex.actual.ret_zone_type().is_none() {
						let zt = ex.actual.ret_zone_type().unwrap();

						if let Some(zone_ex) = players[ex.actual.owner_id.unwrap() as usize].zone_exs.get(&return_zone_coord(map_cur_coord, *map_data.map_szs.last().unwrap())) {
							/////////////////////////////
							// print zone demands
							if let Some(demand_weighted_sum) = zone_ex.demand_weighted_sum[zt as usize] {
								d.mv(1,0);
								d.addstr(&format!("Demand weighted sum: {}", demand_weighted_sum));
							}
							
							/////////////////////////////////////////// 
							// ZoneDemandRaw
							let mut row = 2;
							for (zone_ind, demand_raw) in zone_ex.demand_raw.iter().enumerate() { // indexed by ZoneType
								if let Some(zdr) = demand_raw {
									d.mv(row, 0); row += 1;
									d.addstr(&format!("{} date: {}", ZoneType::from(zone_ind).to_str(), l.date_str(zdr.turn_computed)));
									
									// loop over ZoneDemandType
									for (zone_demand_ind, demand) in zdr.demand.iter().enumerate() {
										d.mv(row, 0); row += 1;
										d.addstr(&format!("   {}: {}", ZoneDemandType::from(zone_demand_ind).to_str(), demand));
									}
									row += 1;
								}
							}
							
							// city hall dist
							d.mv(row, 0);
							match zone_ex.ret_city_hall_dist() {
								Dist::NotInit => {d.addstr("Not init");}
								Dist::NotPossible {turn_computed} => {d.addstr(&format!("Not possible, date: {}", l.date_str(turn_computed)));}
								Dist::Is {dist, bldg_ind} | Dist::ForceRecompute {dist, bldg_ind} => {d.addstr(&format!("Dist: {}, bldg_ind: {}", dist, bldg_ind));}
							}
							///////////////////////////////////////////////////////
						}
					}
					
					// Country owner
					if let Some(owner_id) = ex.actual.owner_id {
						self.print_owner(r_off+1, &players[owner_id as usize], relations, txt_list, l, d);
					}	
				} // extended data
			}
		
			//d.addstr(&format!("{} {}", mzc.elevation, mzc.arability));
		}
		/////////////////////////////// end land stats
		
		//////////////////////////
		// show group movement (select rectangle)
		if let Some(action) = self.add_action_to.first_action() {
			let full_zoom = self.zoom_ind == map_data.max_zoom_ind();
			let mut roff = r_off - 1;
			
			match action.action_type {
				// rectangle started
				ActionType::GroupMv {start_coord: Some(_), end_coord: None} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						&l.Select_a_rectangular_group_of_units,
						(&l.Move_corner, &l.drag_the_X),
						(&l.Finish, &l.stop_dragging),
						kbd, l, disp_chars, self, txt_list, d);
					return;
				}
				// rectangle started
				ActionType::BrigadeCreation {start_coord: Some(_), end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						&l.Create_brigade_by_drawing_a_rectangle,
						(&l.Change_location, &l.move_mouse),
						(&l.Finish, &l.stop_dragging),
						kbd, l, disp_chars, self, txt_list, d);
					return;
				}
				// rectangle started
				ActionType::SectorCreation {start_coord: Some(_), end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						&l.Create_sector_by_drawing_a_rectangle,
						(&l.Change_location, &l.move_mouse),
						(&l.Finish, &l.stop_dragging),
						kbd, l, disp_chars, self, txt_list, d);
					return;
				}
				// enter to start selecting
				ActionType::BrigadeCreation {start_coord: None, end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						&l.Create_brigade_by_drawing_a_rectangle,
						(&l.Change_location, &l.move_mouse),
						(&l.Start_selecting, &l.click_and_drag),
						kbd, l, disp_chars, self, txt_list, d);
					return;
				}
				// enter to start selecting
				ActionType::SectorCreation {start_coord: None, end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						&l.Create_sector_by_drawing_a_rectangle,
						(&l.Change_location, &l.move_mouse),
						(&l.Start_selecting, &l.click_and_drag),
						kbd, l, disp_chars, self, txt_list, d);
					return;
				}
				_ => {}
			}
		}
		
		self.print_unit_bldg_stats(map_cur_coord, stats_row, r_off+1, show_land, map_data, exs, player, players, units, bldg_config,
				bldgs, relations, logs, kbd, l, buttons, disp_chars, txt_list, d);
	}
}

