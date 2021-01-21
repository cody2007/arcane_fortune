use crate::renderer::*;
use crate::map::*;
use crate::units::{ActionType, ActionMetaCont, WORKER_NM, worker_can_continue_bldg};//MAX_UNITS_PER_PLOT, WORKER_NM};
use crate::gcore::hashing::*;
use crate::gcore::*;
use crate::player::{Player, PlayerType, Stats};
use crate::movement::{manhattan_dist_components, manhattan_dist};
use crate::zones::return_zone_coord;
use crate::localization::Localization;

use super::*;
use super::vars::*;
use super::color::*;

const LAND_STATS_COL: i32 = (SUB_MAP_WIDTH + 4) as i32;
const UNIT_STATS_COL: i32 = (SUB_MAP_WIDTH + 19) as i32;

impl ActionType<'_,'_,'_,'_> {
	pub fn print(&self, roff: &mut i32, path_valid: bool, actions_req: f32,
			turns_est: bool, ut_opt: Option<&UnitTemplate>, bldgs: &Vec<Bldg>, exf: &HashedMapEx,
			map_data: &mut MapData, disp: &mut Disp) {
		macro_rules! mvl {() => (disp.state.renderer.mv(*roff, UNIT_STATS_COL); *roff += 1;);}
			
		macro_rules! action_key {
			($nm: expr, $color: expr) => {mvl!(); disp.attron(COLOR_PAIR($color)); disp.state.renderer.addstr($nm); disp.attroff(COLOR_PAIR($color))};
			($nm: expr, $color: expr, $last: expr) => {disp.mv(*roff, UNIT_STATS_COL); disp.attron(COLOR_PAIR($color)); disp.state.renderer.addstr($nm); disp.attroff(COLOR_PAIR($color))};};
		
		let cur_player = disp.state.iface_settings.cur_player;
		let map_sz = *map_data.map_szs.last().unwrap();
		let full_zoom = disp.state.iface_settings.zoom_ind == map_data.max_zoom_ind();
		
		macro_rules! mv_cursor_to{($txt: expr) => {
			action_key!(&format!("<{}>", disp.state.local.click_and_drag), CGREEN);
			disp.addstr(&format!(" {} ", disp.state.local.or));
			disp.attron(COLOR_PAIR(CGREEN));
			disp.state.renderer.addstr(&disp.state.local.arrow_keys);
			disp.attroff(COLOR_PAIR(CGREEN));
			disp.addch(' ');
			disp.state.renderer.addstr($txt);
		};};
		
		match self {
			ActionType::MvWithCursor => {
				mv_cursor_to!(&disp.state.local.to_move_unit);
				
				if !full_zoom {
					action_key!(&format!("<{}>", key_txt(disp.state.kbd.zoom_in, &disp.state.local)), CGREEN);
					disp.addch(' '); disp.state.renderer.addstr(&disp.state.local.to_zoom_in);
				}
				
				action_key!("<Esc>", ESC_COLOR, true);
				disp.addch(' ');
				disp.state.renderer.addstr(&disp.state.local.to_stop_moving_unit);
			
			} ActionType::WorkerRmZonesAndBldgs {start_coord: None, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::True, None, roff,
						disp.state.local.Select_an_area.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.move_mouse.clone()),
						(disp.state.local.Start_selecting.clone(), disp.state.local.click_and_drag.clone()), disp);
			} ActionType::WorkerRmZonesAndBldgs {..} => {
				print_mv_to_sel2(full_zoom, PathValid::True, None, roff,
						disp.state.local.Select_an_area.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.move_mouse.clone()),
						(disp.state.local.Confirm_selection.clone(), disp.state.local.stop_dragging.clone()), disp);

			} ActionType::WorkerZone {valid_placement: false, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::False, None, roff,
						disp.state.local.Zoning_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.move_mouse.clone()),
						(disp.state.local.Finish_zoning.clone(), disp.state.local.stop_dragging.clone()), disp);
			
			} ActionType::WorkerBuildBldg {valid_placement: false, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::False, None, roff,
						disp.state.local.Building_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Build.clone(), disp.state.local.stop_dragging.clone()), disp);
			
			} ActionType::WorkerBuildBldg {template, valid_placement: true, ..} => {
				let actions_req = if let Some(ut) = ut_opt {
					((actions_req + template.construction_req) / ut.actions_per_turn).ceil()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid), Some((actions_req, turns_est)), roff,
						disp.state.local.Building_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Build.clone(), disp.state.local.stop_dragging.clone()), disp);
			
			} ActionType::WorkerContinueBuildBldg {..} => {
				// valid placement of cursor
				if let Some(bldg_ind) = worker_can_continue_bldg(true, bldgs, map_data, exf, &disp.state.iface_settings) {
					let actions_req = if let Some(ut) = ut_opt {
						let template = &bldgs[bldg_ind].template;
						((actions_req + template.construction_req) / ut.actions_per_turn).ceil()
					}else{0.};
					
					print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Building_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Resume_construction.clone(), disp.state.local.stop_dragging.clone()), disp);
					return;

				// invalid placement
				}else{
					print_mv_to_sel2(full_zoom,
						PathValid::FalseCustomMsg(disp.state.local.Select_edge_to_continue_construction.clone()),
						None, roff,
						disp.state.local.Building_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Resume_construction.clone(), disp.state.local.stop_dragging.clone()), disp);
				}
			
			} ActionType::WorkerZone {start_coord: Some(start_coord), ..} => {
				let end_coord = disp.state.iface_settings.cursor_to_map_coord(map_data);
				let start_coord = Coord::frm_ind(*start_coord, map_sz);
				let zone_dim_szs = manhattan_dist_components(end_coord, start_coord, map_sz);
				let zone_sz = ((zone_dim_szs.w + 1) * (zone_dim_szs.h + 1)) as f32;
				
				let actions_req = if let Some(ut) = ut_opt {
					((actions_req + zone_sz) / ut.actions_per_turn).ceil()
				}else{0.};
				
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Zoning_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.move_mouse.clone()),
						(disp.state.local.Finish_zoning.clone(), disp.state.local.stop_dragging.clone()), disp);
				
			} ActionType::WorkerZone {..} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Zoning_instructions.clone(),
						(disp.state.local.Change_location.clone(), disp.state.local.move_mouse.clone()),
						(disp.state.local.Start_zoning.clone(), disp.state.local.click_and_drag.clone()), disp);

			} ActionType::WorkerRepairWall {wall_coord: coord_opt, ..} => {
				let actions_req = if let Some(ut) = ut_opt {
					let get_repair_time = || {
						if let Some(coord) = &coord_opt {
						if let Some(ex) = exf.get(coord) {
						if let Some(owner_id) = ex.actual.owner_id {
						if owner_id == cur_player {
						if let Some(s) = &ex.actual.structure {
						if s.structure_type == StructureType::Wall && s.health != std::u8::MAX {
							return (std::u8::MAX - s.health) as f32 / ut.repair_wall_per_turn.unwrap() as f32;
						}}}}}}
						0.
					};
						
					(actions_req / ut.actions_per_turn).ceil() + get_repair_time()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Repair_wall_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Repair_wall.clone(), disp.state.local.stop_dragging.clone()), disp);

			} ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Build_gate_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Build_gate_no_pre_colon.clone(), disp.state.local.stop_dragging.clone()), disp);
			
			} ActionType::WorkerBuildStructure {structure_type: StructureType::Road, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Building_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Build_road.clone(), disp.state.local.stop_dragging.clone()), disp);

			} ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, ..} => {
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Building_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Build_wall.clone(), disp.state.local.stop_dragging.clone()), disp);

			} ActionType::Attack {..} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Attacking_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(format!("{}:", disp.state.local.Attack.clone()), disp.state.local.stop_dragging.clone()), disp);
			
			} ActionType::Assassinate {..} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff,
						disp.state.local.Assassination_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(format!("{}:", disp.state.local.Assassinate.clone()), disp.state.local.stop_dragging.clone()), disp);
	
			} ActionType::Mv | ActionType::MvIgnoreWallsAndOntoPopulationCenters | ActionType::MvIgnoreOwnWalls |
			ActionType::ScaleWalls | ActionType::GroupMv {start_coord: Some(_), end_coord: Some(_)} => {
				let actions_req = if let Some(ut) = ut_opt {
					(actions_req / ut.actions_per_turn).ceil()
				}else{0.};
				
				print_mv_to_sel2(full_zoom, PathValid::from(path_valid),
						Some((actions_req, turns_est)), roff, 
						disp.state.local.Move_unit_instructions.clone(),
						(disp.state.local.Change_dest.clone(), disp.state.local.drag_the_X.clone()),
						(disp.state.local.Move_unit.clone(), disp.state.local.stop_dragging.clone()), disp);
			
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

impl DispState<'_,'_,'_,'_,'_,'_> {
	fn print_city_hist(&mut self, city_nm_show: &str, owner_id: usize,
			stats_row: i32, players: &Vec<Player>, logs: &Vec<Log>, temps: &Templates) {
		let mut roff = stats_row; // row offset for printing
		
		macro_rules! mvl {() => (self.renderer.mv(roff, UNIT_STATS_COL); roff += 1;);}
		
		macro_rules! print_owner{($id: expr) => {
			let player = &players[$id];
			self.txt_list.add_b(&mut self.renderer);
			set_player_color(player, true, &mut self.renderer);
			self.addstr(&player.personalization.nm);
			set_player_color(player, false, &mut self.renderer);
		};};
		
		let mut max_width = 0;
		
		for log in logs.iter() {
			macro_rules! start_log {() => {
				mvl!();
				self.txt_list.add_b(&mut self.renderer);
				self.local.print_date_log(log.turn, &mut self.renderer);
			};};
	
			match &log.val {
				LogType::LeaderAssassinated {city_nm, ..} => {
					if city_nm == city_nm_show {
						start_log!();
						log.val.print(true, players, &temps.doctrines, self);
					}
				}
				LogType::CityFounded {owner_id, city_nm} => {
					if city_nm == city_nm_show {
						start_log!();
						
						let player = &players[*owner_id];
						let personalization = &player.personalization;
						let (txt, tags) = match player.ptype {
							PlayerType::Empire(_) | PlayerType::Human(_) | PlayerType::Barbarian(_) => {
								// Founded_by_the_civilization: "Founded by the [] civilization.", "#Founded by the [] civilization."
								(&self.local.Founded_by_the_civilization,
								 vec![KeyValColor {
									key: String::from("[]"),
									val: personalization.nm.clone(),
									attr: COLOR_PAIR(personalization.color)
								}])
							}
							PlayerType::Nobility(_) => {
								// house_nm: "House of []"
								// Founed_by_the_house_of: "Founded by the [house_nm]."
								(&self.local.Founded_by_the_house_of,
								 vec![KeyValColor {
									key: String::from("[house_nm]"),
									val: self.local.house_nm.replace("[]", &personalization.nm),
									attr: COLOR_PAIR(personalization.color)
								}])
							}
						};
						
						color_tags_print(txt, &tags, None, &mut self.renderer);
						
						let width = self.local.date_str(log.turn).len() + color_tags_txt(txt, &tags).len();
						
						if width > max_width {max_width = width;}
					}
				}
				LogType::CityCaptured {city_attackee_nm, owner_attacker_id, ..} => {
					if city_attackee_nm == city_nm_show {
						start_log!();
						self.addstr("Captured by the ");
						print_owner!(*owner_attacker_id);
						self.addstr(" civilization.");
						
						let width = "Captured by the ".len() +
							" civilization.".len() +
							players[*owner_attacker_id].personalization.nm.len();
						
						if width > max_width {max_width = width;}
					}
				}
				LogType::CityDestroyed {city_attackee_nm, owner_attacker_id, ..} => {
					if city_attackee_nm == city_nm_show {
						start_log!();
						self.addstr("Destroyed by the ");
						print_owner!(*owner_attacker_id);
						self.addstr(" civilization.");
						
						let width = "Destroyed by the ".len() +
							" civilization.".len() +
							players[*owner_attacker_id].personalization.nm.len();
						
						if width > max_width {max_width = width;}
					}
				}
				LogType::HouseDeclaresIndependence {house_id, ..} => {
					if owner_id == *house_id {
						start_log!();
						log.val.print(true, players, &temps.doctrines, self);
					}
				}
				LogType::KingdomJoinedEmpire {kingdom_id, ..} => {
					if owner_id == *kingdom_id {
						start_log!();
						log.val.print(true, players, &temps.doctrines, self);
					}
				}
	
				LogType::NobleHouseJoinedEmpire {house_id, empire_id} => {
					// house_joined_empire_abbrev: "Joined the [empire_nm] empire."
					if owner_id == *house_id {
						start_log!();
						let empire = &players[*empire_id].personalization;
						
						let tags = vec![KeyValColor {
							key: String::from("[empire_nm]"),
							val: empire.nm.clone(),
							attr: COLOR_PAIR(empire.color)
						}];
						
						color_tags_print(&self.local.house_joined_empire_abbrev, &tags, None, &mut self.renderer);
						let width = color_tags_txt(&self.local.house_joined_empire_abbrev, &tags).len();
						
						if width > max_width {max_width = width;}
					}
				}
				
				LogType::CivCollapsed {..} | LogType::CivDestroyed {..} | LogType::NoNobleSuccessor {..} |
				LogType::UnitDestroyed {..} | LogType::UnitDisbanded {..} |
				LogType::BldgDisbanded {..} | LogType::CityDisbanded {..} |
				LogType::ICBMDetonation {..} | LogType::PrevailingDoctrineChanged {..} |
				LogType::CivDiscov {..} | LogType::UnitAttacked {..} | 
				LogType::StructureAttacked {..} | LogType::WarDeclaration {..} |
				LogType::Rioting {..} | LogType::RiotersAttacked {..} |
				LogType::CitizenDemand {..} | LogType::GenericEvent {..} |
				LogType::PeaceDeclaration {..} | LogType::Debug {..} => {}
			}
		}
		
		// print name of city (centered)
		self.mv(stats_row - 2, UNIT_STATS_COL + if max_width >= city_nm_show.len()
				{(max_width - city_nm_show.len())/2} else {0} as i32);
		self.txt_list.add_b(&mut self.renderer);
		self.addstr(&format!("{}", city_nm_show));
	}
}

enum PathValid {
	False,
	FalseCustomMsg(String),
	True,
}

impl PathValid {
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
fn print_mv_to_sel2(full_zoom: bool, path_valid: PathValid, turns_req: Option<(f32, bool)>, roff: &mut i32,
		title: String, change: (String, String), mut confirm: (String, String), disp: &mut Disp) {
	macro_rules! color_txt {($nm: expr, $color: expr) => {
		disp.attron(COLOR_PAIR($color)); disp.state.renderer.addstr($nm); disp.attroff(COLOR_PAIR($color))}};
	
	let esc_txt = key_txt(disp.state.kbd.esc, &disp.state.local);
	let mut confirm_key = '\n' as u8 as i32;
	
	// check if we should show `Zoom in` instead of confirmation text
	if !full_zoom {
		confirm = (disp.state.local.Zoom_in.clone(), disp.state.local.mouse_wheel.clone());
		confirm_key = disp.state.kbd.zoom_in;
	}
	
	let confirm_kbd_len = key_txt(confirm_key, &disp.state.local).len();
	
	/*
							title
							
					 |     mouse    |   keyboard    (titles)   row 0
	change.0                 |   change.1   | a, s, d, w    (change)   row 1
	confirm.0                |  confirm.1   |   <Enter>     (confirm)  row 2 -- this row may be `Zoom in:`
	Cancel:                  |  right click |    <Esc>      (cancel)   row 3
	
	*/
	
	// now we define the columns (to take the max down the rows to get the needed printing width)
	//              (titles)            (change)   (confirm)          (cancel)
	let lbls =      [String::new(),                change.0.clone(),  confirm.0.clone(),         disp.state.local.Cancel_colon.clone()];
	let mouse_txt = [disp.state.local.mouse.clone(),          change.1.clone(), confirm.1.clone(),         disp.state.local.cancel_click.clone()];
	let kbd_lens =  [disp.state.local.keyboard.len(),  3*3+3,     confirm_kbd_len,   esc_txt.len()];
	
	let lbl_w = lbls.iter().map(|l| l.len()).max().unwrap() as i32 + 2;
	let mouse_w = mouse_txt.iter().map(|m| m.len()).max().unwrap() as i32 + 3;
	let kbd_w = *kbd_lens.iter().max().unwrap() as i32 + 2;
	
	// title
	disp.mv(*roff-2, UNIT_STATS_COL + (lbl_w + mouse_w + kbd_w - title.len() as i32)/2);
	disp.state.txt_list.add_b(&mut disp.state.renderer);
	color_txt!(&title, CGREEN);
	
	// path blocked
	if !path_valid.is_true() && full_zoom {
		disp.mv(*roff, UNIT_STATS_COL);
		if let PathValid::FalseCustomMsg(txt) = &path_valid {
			color_txt!(txt, CRED);
		}else{
			color_txt!(&disp.state.local.Current_location_blocked, CRED);
		}
		*roff += 1;
	}
	
	macro_rules! center{($txt: expr, $w: expr) => {
		let gap = ($w - $txt.len() as i32)/2;
		for _ in 0..gap {disp.addch(' ');}
		disp.state.renderer.addstr($txt);
	};};
	
	// adds to txt list for screen readers
	macro_rules! center_log{($txt: expr, $w: expr) => {
		let gap = ($w - $txt.len() as i32)/2;
		for _ in 0..gap {disp.addch(' ');}
		disp.state.txt_list.add_b(&mut disp.state.renderer);
		disp.state.renderer.addstr($txt);
	};};

	
	let v = disp.state.chars.vline_char;
	
	let col2 = UNIT_STATS_COL + lbl_w;
	let col3 = col2 + mouse_w;
	
	// line 1 lbls (mouse, keyboard)
	{
		disp.mv(*roff, col2);
		//disp.addch(v);
		disp.attron(COLOR_PAIR(CGRAY));
		center!(&disp.state.local.mouse, mouse_w);
		disp.attroff(COLOR_PAIR(CGRAY));
		
		disp.mv(*roff, col3);
		//disp.addch(v);
		disp.attron(COLOR_PAIR(CGRAY));
		center!(&disp.state.local.keyboard, kbd_w);
		disp.attroff(COLOR_PAIR(CGRAY));
		*roff += 1;
	}
	
	let colors = [CWHITE, CWHITE, CGREEN, ESC_COLOR]; // for mouse instructions
	let show_confirm_row = path_valid.is_true() || !full_zoom;
	
	macro_rules! vline{($col: expr) => {
		disp.mv(*roff, $col);
		disp.attron(COLOR_PAIR(CGRAY));
		disp.addch(v);
		disp.attroff(COLOR_PAIR(CGRAY));
	};};
	
	// columns 1 & 2 for all remaining rows (left-most column and mouse instructions)
	for (row_ind, ((lbl, mouse), color)) in lbls.iter()
					.zip(mouse_txt.iter())
					.zip(colors.iter()).enumerate().skip(1) {
		// if invalid path and full zoom, don't show confirmation action
		if row_ind == 2 && !show_confirm_row {continue;}
		
		// label
		disp.mv(*roff, UNIT_STATS_COL + lbl_w - lbl.len() as i32 - 1);
		disp.state.txt_list.add_b(&mut disp.state.renderer);
		disp.addstr(lbl);
		vline!(col2);
		
		// mouse instructions
		disp.attron(COLOR_PAIR(*color));
		center!(mouse, mouse_w);
		disp.attroff(COLOR_PAIR(*color));
		vline!(col3);
		
		// last column (keyboard instructions)
		match row_ind {
			// a,s,d,w
			1 => {
				disp.addch(' ');
				disp.state.txt_list.add_b(&mut disp.state.renderer);
				for k in &[disp.state.kbd.left, disp.state.kbd.down, disp.state.kbd.right] {
					disp.print_key(*k);
					disp.addstr(", ");
				}
				disp.print_key(disp.state.kbd.up);
			
			// <Enter> or i
			} 2 => {
				if confirm_key == '\n' as u8 as i32 {
					disp.attron(COLOR_PAIR(CGREEN));
					center_log!(&disp.state.local.Enter_key, kbd_w);
					disp.attroff(COLOR_PAIR(CGREEN));
				}else{
					disp.mv(*roff, col3 + (kbd_w - 1)/2);
					disp.state.txt_list.add_b(&mut disp.state.renderer);
					disp.print_key(confirm_key);
				}
			// <Esc>
			} 3 => {
				disp.attron(COLOR_PAIR(ESC_COLOR));
				center_log!("<Esc>", kbd_w);
				disp.attroff(COLOR_PAIR(ESC_COLOR));
			} _ => {panicq!("unsupported # of rows");}
		}
		
		*roff += 1;
	}
	
	// turns req
	if let Some((turns_req, est)) = turns_req {
		if turns_req != 0. {
			let len = (disp.state.local.Days_required.len() + if est {disp.state.local.est.len()} else {0}) as i32;
			disp.mv(*roff, UNIT_STATS_COL + (lbl_w + mouse_w + kbd_w - len)/2);
			disp.attron(COLOR_PAIR(CGRAY));
			disp.addstr(&format!("{} {}", disp.state.local.Days_required, turns_req));
			if est {
				disp.addstr(&format!(" {}", disp.state.local.est));
			}
			disp.attroff(COLOR_PAIR(CGRAY));
		}
	}
}

/*fn print_mv_to_sel(full_zoom: bool, path_valid: bool, turns_req: f32, turns_est: bool, nm: &str, nm_cancel: &str, chg_location: Option<&str>, roff: &mut i32, kbd: &KeyboardMap, l: &Localization, d: &mut DispState) {
	macro_rules! mvl {() => (self.mv(*roff, UNIT_STATS_COL); *roff += 1;);}
	
	macro_rules! action_key {
		  ($nm: expr, $color: expr) => {mvl!(); self.attron(COLOR_PAIR($color)); self.addstr($nm); self.attroff(COLOR_PAIR($color))};
		  ($nm: expr, $color: expr, $last: expr) => {self.mv(*roff, UNIT_STATS_COL); self.attron(COLOR_PAIR($color)); self.addstr($nm); self.attroff(COLOR_PAIR($color))};};
	
	// path blocked
	if !path_valid && full_zoom {
		self.attron(COLOR_PAIR(CRED)); mvl!();
		self.addstr(&self.state.local.Current_location_blocked);
		self.attroff(COLOR_PAIR(CRED));
	}
	
	// mv cursor to change dest
	if let Some(chg_location) = chg_location {
		action_key!(&self.state.local.Move_cursor, CGREEN);
		if chg_location.len() != 0 {
			if chg_location == "selection" {
				self.addstr(" to change selection");
			}else if chg_location == "destination" {
				self.addstr(" to change destination");
			}else{
				self.addstr(&format!(" to change {} location", chg_location));
			}
		}else{
			self.addstr(" to change location");
		}
	}
	
	// enter to confirm
	if full_zoom && path_valid {
		action_key!("<Enter>", CGREEN);
		self.addstr(&format!(" to {}", nm));
	// <i> to zoom
	}else if !full_zoom {
		action_key!(&format!("<{}>", key_txt(kbd.zoom_in, l)), CGREEN);
		self.addstr(" to zoom in");
	}
	
	action_key!("<Esc>", ESC_COLOR);
	self.addstr(&format!(" to cancel {}", nm_cancel));
	mvl!(); mvl!();
	
	if turns_req != 0. {
		self.addstr(&format!("{} {}", l.Days_required, turns_req));
		if turns_est {
			self.addstr(&format!(" {}", l.est));
		}
	}
}*/

// print the buttons in `cmds`
fn print_cmds(mut cmds: Vec<&mut Button>, roff: i32, l: &Localization, txt_list: &mut TxtList, d: &mut Renderer, ui_mode: &UIMode) {
	const ROWS_PER_COL: usize = 5;
	let mut col_offset = UNIT_STATS_COL;
	for col_cmds in cmds.chunks_mut(ROWS_PER_COL) {
		let mut max_w = 0;
		for (cmd_ind, button) in col_cmds.iter_mut().enumerate() {
			d.mv(roff + cmd_ind as i32, col_offset);
			txt_list.add_b(d);
			let w = button.print(Some(ui_mode), l, d) + 1;
			if w > max_w {max_w = w;}
		}
		
		// shift over offset for next column
		col_offset += (max_w + 1) as i32;
	}
}


impl Disp<'_,'_,'_,'_,'_,'_> {
	fn print_owner(&mut self, mut roff: i32, player: &Player, relations: &Relations) {
		let cur_player = self.state.iface_settings.cur_player;
		match player.ptype {
			PlayerType::Human(_) | PlayerType::Empire(_) => {
				self.mv(roff, LAND_STATS_COL); roff += 1;
				self.state.txt_list.add_b(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Country);
				self.addstr(": ");
			}
			PlayerType::Barbarian(_) => {}
			PlayerType::Nobility(_) => {
				self.mv(roff, LAND_STATS_COL); roff += 1;
				self.state.txt_list.add_b(&mut self.state.renderer);
				set_player_color(player, true, &mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.only_House_of);
				set_player_color(player, false, &mut self.state.renderer);
			}
		}
		
		self.mv(roff, LAND_STATS_COL);
		self.state.txt_list.add_b(&mut self.state.renderer);
		set_player_color(player, true, &mut self.state.renderer);
		self.addstr(if player.id == cur_player {"You"} else {&player.personalization.nm});
		set_player_color(player, false, &mut self.state.renderer);
		
		if cur_player != player.id && relations.at_war(cur_player as usize, player.id as usize) {
			self.mv(roff + 1, LAND_STATS_COL);
			self.state.txt_list.add_b(&mut self.state.renderer);
			self.addch('(');
			addstr_c("WAR", CRED, &mut self.state.renderer);
			self.addch(')');
		}
	}
	
	fn zoom_in_to_change_actions(&mut self, roff: i32) {
		self.state.renderer.mv(roff, UNIT_STATS_COL);
		self.state.txt_list.add_b(&mut self.state.renderer);
		self.state.renderer.addstr("Zoom ");
		
		let combine_w_word = self.state.kbd.zoom_in == 'i' as i32;
		
		if combine_w_word {
			self.print_key(self.state.kbd.zoom_in);
		}else{
			self.state.renderer.addch('i');
		}
		self.state.renderer.addstr("n to change unit's actions");
		if !combine_w_word {
			self.state.renderer.addch('(');
			self.print_key(self.state.kbd.zoom_in);
			self.state.renderer.addch(')');
		}
	}
	
	fn print_unit_nm_health(&mut self, roff: &mut i32, unit_ind: usize, u: &Unit, pstats: &Stats, players: &Vec<Player>) {
		macro_rules! mvl {() => (self.state.renderer.mv(*roff, UNIT_STATS_COL); *roff += 1;);}
		
		// print nm
		{
			mvl!();
			if u.template.nm[0] != RIOTER_NM {
				self.mv(*roff-2, UNIT_STATS_COL);
				self.state.txt_list.add_b(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.The_Battalion.replace("[nm]", &u.nm).replace("[type]", &u.template.nm[self.state.local.lang_ind]));
				
				if self.state.iface_settings.add_action_to.is_none() {
					if let Some(brigade_nm) = pstats.unit_brigade_nm(unit_ind) {
						self.mv(*roff-1, UNIT_STATS_COL);
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.addstr(&self.state.local.Member_of_the_Brigade.replace("[]", brigade_nm));
						self.addstr(" (");
						self.state.buttons.view_brigade.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
						self.addstr(", ");
						self.state.buttons.leave_brigade.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
						self.addstr(")");
					}
				}
				
			}else{
				self.state.txt_list.add_b(&mut self.state.renderer);
				self.addstr(&u.template.nm[0]);
			}
		}
		
		// print health
		{
			mvl!(); self.state.txt_list.add_b(&mut self.state.renderer); self.addstr(&format!("{}: ", self.state.local.Health));
			let health_frac = u.health();
			colorize(health_frac, true, &mut self.state.renderer);
			if health_frac > 1. {
				self.addstr(&format!("{}%", health_frac.round()));
			}else{
				self.addstr(&format!("{:.2}%", health_frac));
			}
			colorize(health_frac, false, &mut self.state.renderer);
		}
		
		//// show boarded units
		if let Some(units_carried) = &u.units_carried {
			if units_carried.len() > 0 && self.state.iface_settings.add_action_to.is_none() {
				debug_assertq!(u.template.carry_capac >= units_carried.len());
				
				let col2 = ("Action: Fortified   ".chars().count() as i32) + UNIT_STATS_COL;
				
				self.mv(*roff, col2); self.state.txt_list.add_b(&mut self.state.renderer);
				self.state.renderer.addstr(&self.state.local.Carrying); self.addch(' ');
				for (i, c) in units_carried.iter().enumerate() {
					plot_unit(c.template, &players[c.owner_id as usize], &mut self.state.renderer);
					if i == (units_carried.len()-1) {break;}
					self.addstr(", ");
				}
			}
		}
		
		self.mv(*roff, UNIT_STATS_COL); *roff += 2;
	}
	
	// if in mv mode
	fn print_unit_action(&mut self, mut roff: i32, action_iface: &ActionInterfaceMeta, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx, map_data: &mut MapData, pstats: &Stats, players: &Vec<Player>) {
		if let Some(unit_ind) = action_iface.unit_ind {
			let u = &units[unit_ind];
			
			self.print_unit_nm_health(&mut roff, unit_ind, u, pstats, players);
			
			let actions_req = if let Some(ActionMetaCont {final_end_coord, ..}) = &action_iface.action.action_meta_cont {
				let map_sz = *map_data.map_szs.last().unwrap();
				manhattan_dist(*final_end_coord, Coord::frm_ind(u.return_coord(), map_sz), map_sz) as f32
			}else{
				action_iface.action.actions_req
			};
			
			let path_valid = action_iface.action.path_coords.len() != 0 ||
					u.return_coord() == self.state.iface_settings.cursor_to_map_ind(map_data);
			
			let turns_est = !action_iface.action.action_meta_cont.is_none();
			
			action_iface.action.action_type.print(&mut roff, path_valid, actions_req, turns_est, Some(u.template), bldgs, exf, map_data, self);
		}
	}
	
	// possible actions
	fn print_broadcastable_actions(&mut self, brigade_nm: &String, pstats: &Stats, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx, stats_row: i32, map_data: &mut MapData) {
		let brigade = pstats.brigade_frm_nm(brigade_nm);
		let mut roff = stats_row;
		macro_rules! mvl {() => (self.state.renderer.mv(roff, UNIT_STATS_COL); roff += 1;);}
		self.mv(roff-1, UNIT_STATS_COL);
		self.state.buttons.Cancel.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
		
		mvl!(); self.state.txt_list.add_b(&mut self.state.renderer);
		self.state.renderer.addstr(&self.state.local.Choose_an_action_for_all_brigade_units.replace("[]", brigade_nm));
		mvl!();
		
		// show current active action
		if let AddActionTo::AllInBrigade {action_ifaces: Some(action_ifaces), ..} = &self.state.iface_settings.add_action_to {
			if let Some(action_iface) = action_ifaces.first() {
				let path_valid = action_ifaces.iter().any(|af| af.action.path_coords.len() != 0);
				let actions_req = 0.;
				let turns_est = false;
				let ut_opt = None;
				mvl!();
				action_iface.action.action_type.clone().print(&mut roff, path_valid, actions_req, turns_est, ut_opt, bldgs, exf, map_data, self);
			}else{panicq!("no active action_ifaces in AllInBrigade move");}
		
		// show possible actions
		}else if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
			let mut cmds = Vec::with_capacity(30);	
			macro_rules! add{($($button_nm: ident),*) => {$(cmds.push(&mut self.state.buttons.$button_nm);)*};};
			
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
			
			print_cmds(cmds, roff, &self.state.local, &mut self.state.txt_list, &mut self.state.renderer, &self.ui_mode);
		// zoom in to change actions
		}else{
			self.zoom_in_to_change_actions(roff);
		}
	}
	
	// possible actions
	fn print_build_list_actions(&mut self, brigade_nm: &String, pstats: &Stats, units: &Vec<Unit>,
			bldgs: &Vec<Bldg>, exf: &HashedMapEx, stats_row: i32, map_data: &mut MapData) {
		let brigade = pstats.brigade_frm_nm(brigade_nm);
		let mut roff = stats_row;
		macro_rules! mvl {() => (self.state.renderer.mv(roff, UNIT_STATS_COL); roff += 1;);}
		self.mv(roff-1, UNIT_STATS_COL);
		self.state.buttons.Cancel.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
		
		mvl!(); self.state.txt_list.add_b(&mut self.state.renderer);
		self.addstr(&self.state.local.Choose_an_action_to_add_to_the_brigade_build_list.replace("[]", brigade_nm));
		mvl!();
		
		// show current active action
		if let AddActionTo::BrigadeBuildList {action: Some(action), ..} = &self.state.iface_settings.add_action_to {
			let path_valid = true;
			let actions_req = 0.;
			let turns_est = false;
			let ut_opt = None;
			mvl!();
			action.action_type.clone().print(&mut roff, path_valid, actions_req, turns_est, ut_opt, bldgs, exf, map_data, self);
		// show possible actions
		}else if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
			let mut cmds = Vec::with_capacity(30);
			macro_rules! add{($($button_nm: ident),*) => {$(cmds.push(&mut self.state.buttons.$button_nm);)*};};
			
			add!(zone_agricultural, zone_residential, zone_business, zone_industrial, build_bldg);
					
			// repair wall
			if brigade.unit_inds.iter().any(|&ind| units[ind].template.repair_wall_per_turn != None) {
				add!(repair_wall);
			}
			
			add!(build_gate);
			
			print_cmds(cmds, roff, &self.state.local, &mut self.state.txt_list, &mut self.state.renderer, &self.ui_mode);
		// zoom in to change actions
		}else{
			self.zoom_in_to_change_actions(roff);
		}
	}
	
	// prints health, nm, possible actions for single unit
	fn print_unit_stats(&mut self, unit_inds: &Vec<usize>, stats_row: i32, lside_row: i32,
			show_land: bool, player: &Player, players: &Vec<Player>, units: &Vec<Unit>, 
			bldgs: &Vec<Bldg>, map_data: &mut MapData, exf: &HashedMapEx, relations: &Relations) {
		//debug_assertq!(self.zoom_ind == map_data.max_zoom_ind());
		//debug_assertq!(unit_inds.len() <= MAX_UNITS_PER_PLOT);
		debug_assertq!(unit_inds.len() > 0);
		
		let mut roff = stats_row;
		
		/////////
		// multi-unit display
		if show_land && unit_inds.len() != 1 {
			self.mv(roff - 2, UNIT_STATS_COL);
			self.state.txt_list.add_b(&mut self.state.renderer);
			self.addstr(&format!("{}: ", &self.state.local.Units));
			for unit_ind_ind in 0..unit_inds.len() {
				if unit_ind_ind > 0 {self.addstr(", ");}
			
				let u = &units[unit_inds[unit_ind_ind]];
				if unit_ind_ind == self.state.iface_settings.unit_subsel {self.attron(A_UNDERLINE());}
				plot_unit(u.template, &players[u.owner_id as usize], &mut self.state.renderer);
				if unit_ind_ind == self.state.iface_settings.unit_subsel {self.attroff(A_UNDERLINE());}
			}
			
			if self.ui_mode.is_none() && self.state.iface_settings.add_action_to.is_none() {
				self.addch(' ');
				self.state.buttons.tab.print_key_only(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
			}
		} // end multi-unit disp
		
		let unit_ind = *unit_inds.get(self.state.iface_settings.unit_subsel).unwrap_or_else(|| &unit_inds[0]);
		let u = &units[unit_ind];
			
		// in action mode
		if let AddActionTo::IndividualUnit {action_iface} = &self.state.iface_settings.add_action_to {
			let action_iface = action_iface.clone();
			self.print_unit_action(roff, &action_iface, units, bldgs, exf, map_data, &player.stats, players);
		
		// not interactively moving or building anything -- show actions unit could perform
		}else if show_land {
			self.print_unit_nm_health(&mut roff, unit_ind, u, &player.stats, players);
			
			// player's unit
			if u.owner_id == self.state.iface_settings.cur_player || self.state.iface_settings.show_actions {
				{ // print Action: [] (actions used/actions per turn)
					self.state.txt_list.add_b(&mut self.state.renderer);
					self.addstr(&format!("{}: ", self.state.local.Action));
					if let Some(action) = u.action.last() {
						self.addstr(&action.action_type.nm(&self.state.local));
						/*self.addstr(&format!("{}", action.path_coords.len()));
						if let Some(action_meta_cont) = &action.action_meta_cont {
							self.addstr(&format!(" checkpoint {}", action_meta_cont.checkpoint_path_coords.len()));
						}*/
					}else{
						self.state.renderer.addstr(&self.state.local.Idle);
					}
					
					if let Some(actions_used) = u.actions_used {
						if actions_used != 0. {
							self.addstr(&format!(" ({}/{})", actions_used, u.template.actions_per_turn));
						}
					}else{
						self.addch(' ');
						self.state.renderer.addstr(&self.state.local.no_actions_remain);
					}
				}
				
				// show possible actions or instructions to zoom in
				if self.ui_mode.is_none() && (u.template.nm[0] != RIOTER_NM || self.state.iface_settings.show_actions) {
					// show possible actions
					if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() && !u.actions_used.is_none() {
						let mut cmds = Vec::with_capacity(30);
						macro_rules! add{($($button_nm: ident),*) => {$(cmds.push(&mut self.state.buttons.$button_nm);)*};};
						
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
						
						{ // unit-specific actions
							let ut = &u.template;
							
							// assassin
							if !ut.assassin_per_turn.is_none() {
								add!(scale_walls, assassinate);
							}
							
							// worker
							if ut.nm[0] == WORKER_NM {
								add!(zone_agricultural, zone_residential, zone_business, zone_industrial, 
									automate_zone_creation, rm_bldgs_and_zones, continue_bldg_construction, build_bldg, build_road);
								
								// can only build wall if this worker is the only
								// one on the tile
								if unit_inds.len() == 1 {add!(build_wall);}
								add!(build_gate);
								
								// repair wall
								if u.template.repair_wall_per_turn != None {add!(repair_wall);}
							
							// soldier actions
							}else if !ut.attack_per_turn.is_none() {
								add!(attack, soldier_automate);
							}
						}
						
						// show option to unboard
						if let Some(_) = u.units_carried {
							if let Unboard::Loc {..} = unboard_land_adj(unit_ind, units, bldgs, map_data, exf) {
								add!(unload_boat);
							}
						}
						
						print_cmds(cmds, roff, &self.state.local, &mut self.state.txt_list, &mut self.state.renderer, &self.ui_mode);
						
					// zoom in to change actions
					}else if self.state.iface_settings.zoom_ind != map_data.max_zoom_ind() {
						self.zoom_in_to_change_actions(roff);
					}
				} // menu/windows not active
			// currently owned unit
			}/*else{ // someone else's unit
				let o = &owners[u.owner_id as usize];
				if o.player_type != PlayerType::Barbarian {
					self.addstr("Country: ");
				}
				set_player_color(o, true);
				self.addstr(&format!("{}", o.nm));
				set_player_color(o, false);
			}*/
		} // unit stats
		
		// Country owner
		if show_land && u.owner_id != self.state.iface_settings.cur_player {
			self.print_owner(lside_row, &players[u.owner_id as usize], relations);
		}
	}
}

impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'_,'bt,'ut,'rt,'dt> {
	fn print_unit_bldg_stats(&mut self, map_cur_coord: u64, stats_row: i32, lside_row: i32, show_land: bool,
			map_data: &mut MapData, exs: &Vec<HashedMapEx>, player: &Player, players: &Vec<Player>,
			units: &Vec<Unit>, bldg_config: &BldgConfig, bldgs: &Vec<Bldg>, gstate: &GameState,
			temps: &Templates) {
		let pstats = &player.stats;
		//if self.zoom_ind != map_data.max_zoom_ind() {return;} // only show at full zoom
		
		// if in IndividualUnit move mode:
		//	show stats of unit proposed to make the move (rather than showing the units at the current cursor coord)
		let get_cursor_or_sel_coord = || {
			if let AddActionTo::IndividualUnit {action_iface} = &self.state.iface_settings.add_action_to {
				let start_coord = action_iface.start_coord;
				((start_coord.y as usize)*map_data.map_szs[self.state.iface_settings.zoom_ind].w + (start_coord.x as usize)) as u64
			}else{
				map_cur_coord
			}
		};
		
		let exz = &exs[self.state.iface_settings.zoom_ind];
		let exf = exs.last().unwrap();
		
		// show brigade
		if let AddActionTo::AllInBrigade {brigade_nm, ..} = &self.state.iface_settings.add_action_to {
			let brigade_nm = brigade_nm.clone();
			self.print_broadcastable_actions(&brigade_nm, pstats, units, bldgs, exf, stats_row, map_data);
		}else if let AddActionTo::BrigadeBuildList {brigade_nm, ..} = &self.state.iface_settings.add_action_to {
			let brigade_nm = brigade_nm.clone();
			self.print_build_list_actions(&brigade_nm, pstats, units, bldgs, exf, stats_row, map_data);

		// ex data
		}else if let Some(ex) = exz.get(&get_cursor_or_sel_coord()) {
			// show zoomed out city history
			if show_land {
				if let Some(fog) = self.state.iface_settings.get_fog_or_actual(get_cursor_or_sel_coord(), self.state.iface_settings.zoom_ind, ex, players, &gstate.relations) {
					if let Some(max_city_nm) = &fog.max_city_nm {
						if let Some(owner_id) = fog.owner_id {
							self.state.print_city_hist(max_city_nm, owner_id as usize, stats_row, players, &gstate.logs, temps);
						}
						return;
					}
				}
			}
			
			// show unit
			if let Some(unit_inds) = &ex.unit_inds {
				self.print_unit_stats(unit_inds, stats_row, lside_row, show_land, player, players, units, bldgs, map_data, exz, &gstate.relations);
				return;
				
			// show bldg (full zoom)
			}else if let Some(bldg_ind) = ex.bldg_ind {
				if show_land {
					let b = &bldgs[bldg_ind];
					let bt = b.template;
					//let ex = exz.get(&b.coord).unwrap();
					
					let mut roff = stats_row - 2; // row offset for printing
					
					macro_rules! mvl {() => (self.mv(roff, UNIT_STATS_COL); roff += 1;);}	
					self.mv(roff, UNIT_STATS_COL); roff += 2;
					
					// print name of bldg
					// if city hall & we're not showing all player's actions, show history and return
					if let BldgArgs::PopulationCenter {nm, ..} = &b.args {
						if b.owner_id == self.state.iface_settings.cur_player || self.state.iface_settings.show_actions {
							self.state.txt_list.add_b(&mut self.state.renderer);
							self.addstr(&format!("{} ({})", nm, bt.nm[self.state.local.lang_ind]));
						}else{
							self.state.print_city_hist(nm, b.owner_id as usize, stats_row, players, &gstate.logs, temps);
							return;
						}
					// public event
					}else if let BldgArgs::PublicEvent {nm, public_event_type, ..} = &b.args {
						self.state.txt_list.add_b(&mut self.state.renderer);
						
						// capitialize first letter
						let mut nm_upper: Vec<char> = nm.chars().collect();
						nm_upper[0] = nm_upper[0].to_uppercase().nth(0).unwrap();
						let nm_upper: String = nm_upper.into_iter().collect();
						
						self.addstr(&nm_upper);
						
						// happiness bonus
						mvl!(); mvl!();
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.state.renderer.addstr(&self.state.local.Happiness_bonus);
						self.addstr(&format!(" {}", public_event_type.happiness_bonus(bldg_config)));
						
					}else{
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.addstr(&bt.nm[self.state.local.lang_ind]);
						
						// print "(abandoned)"?
						if let BldgType::Taxable(_) = b.template.bldg_type {
							if b.n_residents() == 0 {addstr_c(&self.state.local.abandoned, CGRAY, &mut self.state.renderer);}
						}
					}
					
					// print damage
					if let Some(damage) = b.damage {
						self.mv(roff - 1, UNIT_STATS_COL);
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.state.renderer.addstr(&self.state.local.Damage);
						let damage_frac = (damage as f32 / bldg_config.max_bldg_damage as f32) * 100.;
						colorize(100.-damage_frac, true, &mut self.state.renderer);
						self.addstr(&format!(" {:.1}%", damage_frac));
						colorize(100.-damage_frac, false, &mut self.state.renderer);
					}
					
					mvl!();
					
					//////////////////////////////////
					// gov & taxable bldg specific printing
					match b.template.bldg_type {
						BldgType::Gov(_) => {
							// under construction?
							if let Some(prog) = b.construction_done {
								self.state.txt_list.add_b(&mut self.state.renderer);
								self.addstr(&format!("{} {}%", &self.state.local.Construction_progress,
										((100.*prog as f32 / bt.construction_req as f32).round() as usize)));
								return;
							}
							
							// print production and taxes
							if !b.template.units_producable.is_none() && (b.owner_id == self.state.iface_settings.cur_player || self.state.iface_settings.show_actions) {
								// constructed
								if let BldgArgs::PopulationCenter {tax_rates, ..} = &b.args {
									self.state.txt_list.add_b(&mut self.state.renderer);
									self.state.renderer.addstr(&self.state.local.Taxes);
									
									macro_rules! tax_ln{($shortcut_key: expr, $nm: expr, $zt: path) => (
										mvl!();
										if let UIMode::SetTaxes($zt) = self.ui_mode {
											self.addstr(&format!("   {}", $nm));
										}else{
											$shortcut_key.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
										}
										self.addstr(&format!(" : {}% ", tax_rates[$zt as usize]));
										
										if let UIMode::SetTaxes($zt) = self.ui_mode {
											self.state.buttons.increase_tax.print_key_only(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
											
											self.state.buttons.increase_tax_alt.pos = self.state.buttons.increase_tax.pos.clone();
											// ^ so that it's also on screen and it's key will be potentially
											//   seen as active when checked in gcore/non_menu_keys.rs
											
											self.addstr(" | ");
											self.state.buttons.decrease_tax.print_key_only(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
										}
									);}
									tax_ln!(self.state.buttons.tax_agricultural, &self.state.local.Agricultural, ZoneType::Agricultural);
									tax_ln!(self.state.buttons.tax_residential, &self.state.local.Residential, ZoneType::Residential);
									tax_ln!(self.state.buttons.tax_business, &self.state.local.Business, ZoneType::Business);
									tax_ln!(self.state.buttons.tax_industrial, &self.state.local.Industrial, ZoneType::Industrial);
									roff += 1;
								}
								
								// production
								mvl!();
								self.state.txt_list.add_b(&mut self.state.renderer);
								self.state.renderer.addstr(&self.state.local.Production); self.mv(roff, UNIT_STATS_COL); /////!!!!!!!!!!!!!! (roff not incremented or else you get a compiler warning)
								self.state.buttons.change_bldg_production.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
								
								if let BldgArgs::PopulationCenter {production, ..} | BldgArgs::GenericProducable {production} = &b.args {
									if let Some(production_entry) = production.last() {
										let ut = production_entry.production;
										self.state.txt_list.add_b(&mut self.state.renderer);
										self.addstr(&format!("{}  ({}/{})", &ut.nm[self.state.local.lang_ind],
											production_entry.progress, ut.production_req.round() as usize));
										if production.len() > 1 {
											self.addstr(" (");
											self.state.buttons.view_production.print(Some(&self.ui_mode), &self.state.local, &mut self.state.renderer);
											self.addch(' ');
											self.addstr(&self.state.local.and_x_more.replace("[]", &format!("{}", production.len()-1)));
											self.addch(')');
										}
									}else{ 
										self.state.renderer.addstr(&self.state.local.None);
									}
								}
							}else if b.template.doctrinality_bonus > 0. {
								self.mv(roff, UNIT_STATS_COL);
								self.state.txt_list.add_b(&mut self.state.renderer);
								self.addstr(&format!("{}: ", self.state.local.Dedication));
								self.addstr(&b.doctrine_dedication.nm[self.state.local.lang_ind]);
							}
						/////////////////////////////////////
						// taxable bldg
						} BldgType::Taxable(zone_type) => {
							self.state.txt_list.add_b(&mut self.state.renderer);
							self.state.renderer.addstr(&self.state.local.City_Hall_dist); self.addch(' ');
							
							let zi = players[b.owner_id as usize].zone_exs.get(&return_zone_coord(b.coord, *map_data.map_szs.last().unwrap())).unwrap();
							match zi.ret_city_hall_dist() {
								Dist::Is {dist, ..} | Dist::ForceRecompute {dist, ..} => {
									self.addstr(&format!("{}", dist)); mvl!();
									
									let taxable_upkeep = b.return_taxable_upkeep();
									if taxable_upkeep != 0. {
										let effective_tax_rate = -100. * taxable_upkeep / (b.template.upkeep * b.operating_frac());
										self.state.txt_list.add_b(&mut self.state.renderer);
										self.state.renderer.addstr(&self.state.local.Enforced_tax);
										self.addstr(&format!(" {:.1}%", effective_tax_rate));
										mvl!();
									}
									self.state.txt_list.add_b(&mut self.state.renderer);
									self.state.renderer.addstr(&self.state.local.Tax_payments); self.addch(' ');
									if taxable_upkeep == 0. {
										self.addch('0');
									}else{
										self.addstr(&format!("{:.2}", taxable_upkeep));
									}
								}
								Dist::NotPossible {..} => {
									self.state.renderer.addstr(&self.state.local.No_route);
									self.addch(' ');
									self.attron(COLOR_PAIR(CRED));
									self.state.renderer.addstr(&self.state.local.build_roads);
									self.attroff(COLOR_PAIR(CRED)); mvl!();
									self.state.renderer.addstr(&self.state.local.Enforced_tax);
									self.addstr(" 0%");
								} Dist::NotInit => {self.state.renderer.addstr(&self.state.local.Not_yet_determined);}
							}
							
							if b.n_residents() != 0 {
								roff += 1;
								let resident_start_line = roff;
								mvl!();
								debug_assertq!(Some(zone_type) == ex.actual.ret_zone_type());
								let is_residential_zone = ex.actual.ret_zone_type() == Some(ZoneType::Residential);
								
								self.state.txt_list.add_b(&mut self.state.renderer);
								self.state.renderer.addstr(if is_residential_zone {&self.state.local.Residents} else {&self.state.local.Employees});
								self.addch(' ');
								
								self.addstr(&format!("{}/{}", b.n_residents(), bt.resident_max));
								
								mvl!();
								self.state.txt_list.add_b(&mut self.state.renderer);
								if is_residential_zone {
									self.state.renderer.addstr(&self.state.local.Employed);
									self.addstr(&format!(" {}/{}", b.n_sold(), b.n_residents() ));
								}else{
									self.state.renderer.addstr(&self.state.local.Products_sold);
									self.addstr(&format!(" {}/{}", b.n_sold(), b.prod_capac() ));
								}
								
								self.mv(roff, UNIT_STATS_COL);
								self.state.txt_list.add_b(&mut self.state.renderer);
								self.state.renderer.addstr(&self.state.local.Consumption);
								self.addstr(&format!(" {}/{}", b.cons(), b.cons_capac() ));
								
								if zone_type == ZoneType::Residential {
									let zs = &zi.zone_agnostic_stats;
									{ // Dispositions: doctrine_sum
										self.mv(resident_start_line, UNIT_STATS_COL + 20);
										self.state.txt_list.add_b(&mut self.state.renderer);
										self.state.renderer.addstr(&self.state.local.Dispositions);
										
										const RANGE: f32 = 0.5*2.;
										const N_STEPS: f32 = 6.;
										const STEP: f32 = RANGE / N_STEPS;
										let val = zs.locally_logged.doctrinality_sum.iter().sum::<f32>();
										let desc = if val < (-RANGE/2.) {&self.state.local.Scholar
										}else if val < (-RANGE/2. + STEP) {&self.state.local.Artisan
										}else if val < (-RANGE/2. + 2.*STEP) {&self.state.local.Literate
										}else if val < (-RANGE/2. + 3.*STEP) {&self.state.local.Illiterate
										}else if val < (-RANGE/2. + 4.*STEP) {&self.state.local.Adherant
										}else if val < (-RANGE/2. + 5.*STEP) {&self.state.local.Reverant
										}else{&self.state.local.Devout};
										self.state.renderer.addstr(&format!(" {}", desc));
									}
									
									{ // Politics: pacifism
										self.mv(resident_start_line+1, UNIT_STATS_COL + 20);
										self.state.txt_list.add_b(&mut self.state.renderer);
										self.state.renderer.addstr(&self.state.local.Politics);
										
										const POL_RANGE: f32 = 0.5*2.;
										const N_STEPS: f32 = 5.;
										const STEP: f32 = POL_RANGE / N_STEPS;
										let val = zs.locally_logged.pacifism_sum;
										let desc = if val < (-POL_RANGE/2.) {&self.state.local.Militarist
										}else if val < (-POL_RANGE/2. + STEP) {&self.state.local.Interventionist
										}else if val < (-POL_RANGE/2. + 2.*STEP) {&self.state.local.Pragmatist
										}else if val < (-POL_RANGE/2. + 3.*STEP) {&self.state.local.Peace_minded
										}else{&self.state.local.Pacifist};
										self.state.renderer.addstr(&format!(" {}", desc));// {} {} {}", desc, val, POL_RANGE, STEP));
									}
									
									{ // Moods: happiness
										self.mv(resident_start_line+2, UNIT_STATS_COL + 20);
										self.state.txt_list.add_b(&mut self.state.renderer);
										self.state.renderer.addstr(&self.state.local.Moods);
										
										const RANGE: f32 = 200.*2.;
										const N_STEPS: f32 = 5.;
										const STEP: f32 = RANGE / N_STEPS;
										let val = zs.locally_logged.happiness_sum;
										let desc = if val < (-RANGE/2.) {&self.state.local.Treasonous
										}else if val < (-RANGE/2. + STEP) {&self.state.local.Rebellious
										}else if val < (-RANGE/2. + 2.*STEP) {&self.state.local.Doubtful
										}else if val < (-RANGE/2. + 3.*STEP) {&self.state.local.Hopeful
										}else if val < (-RANGE/2. + 4.*STEP) {&self.state.local.Content
										}else if val < (-RANGE/2. + 5.*STEP) {&self.state.local.Joyful
										}else{&self.state.local.Euphoric};
										self.state.renderer.addstr(&format!(" {}", desc));
									}
								}
							}
						} // taxable bldg
					} // match bldg type
				}
			}
		
		// show action (can occur if we are zoomed out and `ex` is empty)
		}else if let AddActionTo::IndividualUnit {action_iface} = &self.state.iface_settings.add_action_to {
			let action_iface = action_iface.clone();
			self.print_unit_action(stats_row as i32, &action_iface, units, bldgs, exs.last().unwrap(), map_data, pstats, players);
		}
	}

	pub fn print_bottom_stats(&mut self, map_data: &mut MapData, exs: &Vec<HashedMapEx>, player: &Player, players: &Vec<Player>, units: &Vec<Unit>,
			bldg_config: &BldgConfig, bldgs: &Vec<Bldg>, gstate: &GameState, temps: &Templates){
		////////////////////////////////////////////////
		// land stats
		let stats_row = (self.state.iface_settings.screen_sz.h - MAP_ROW_STOP_SZ + 2) as i32;
		
		self.mv(stats_row, LAND_STATS_COL);
		let map_cur_coord = self.state.iface_settings.cursor_to_map_ind(map_data);
		let mzc = map_data.get(ZoomInd::Val(self.state.iface_settings.zoom_ind), map_cur_coord);
		
		//self.addstr(&format!("Coord: ({}, {})", (map_cur_coord_unr.x as f32*z), map_cur_coord_unr.y as f32*z));
		self.mv(stats_row + 1, LAND_STATS_COL);
		let mut r_off = stats_row + 4;

		// land is undiscovered
		let show_land = !self.state.iface_settings.show_fog || land_discovered(map_cur_coord, self.state.iface_settings.cur_player as usize, self.state.iface_settings.zoom_ind, players, &gstate.relations);
		if !show_land {
			self.state.txt_list.add_b(&mut self.state.renderer);
			self.state.renderer.addstr(&self.state.local.Undiscovered);
			//return;
		
		// land is discovered
		}else{
			//self.addstr(&format!("Elevation: {}", mzc.elevation));
			
			let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
			let ex_wrapped = exs[self.state.iface_settings.zoom_ind].get(&map_cur_coord);
			let resource_wrapped = map_data.get(ZoomInd::Full, map_cur_coord).get_resource(map_cur_coord, map_data, map_sz); 
			
			let space_empty_ign_zone = (|| {
				if !self.state.iface_settings.add_action_to.is_none() {return false;}
				if let Some(ex) = &ex_wrapped {
					ex.bldg_ind.is_none() && ex.actual.max_city_nm.is_none() && ex.actual.max_bldg_template.is_none() && ex.unit_inds.is_none()
				}else {true}
			})();
			
			// show sector
			if self.state.iface_settings.show_sectors && self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() {
				if let Some(sector_nm) = player.stats.sector_nm_frm_coord(map_cur_coord, map_sz) {
					self.mv(stats_row-1, LAND_STATS_COL);
					self.state.txt_list.add_b(&mut self.state.renderer);
					self.state.renderer.addstr(&self.state.local.Sector); self.addch(' ');
					if space_empty_ign_zone {
						self.addstr(&sector_nm);
					// crop sector name
					}else if (UNIT_STATS_COL - LAND_STATS_COL) > (self.state.local.Sector.len()+2) as i32 {
						let len = (UNIT_STATS_COL - LAND_STATS_COL) as usize - self.state.local.Sector.len() - 2;
						self.addstr(&crop_txt(sector_nm, len));
					}
				}
			}
			
			///// arability
			{
				self.mv(stats_row + 1, LAND_STATS_COL);
				let arability_str = ArabilityType::frm_arability(mzc.arability, mzc.map_type, mzc.show_snow).to_str(&self.state.local);
				let arability_str_lns: Vec<&str> = arability_str.split(" ").collect();
				
				self.state.txt_list.add_b(&mut self.state.renderer);
				self.addstr(arability_str_lns[0]);
				
				// print second line of arability name
				if arability_str_lns.len() > 1 {
					// print on same line (no ex data shown and not moving)
					if ex_wrapped.is_none() && self.state.iface_settings.add_action_to.is_none() && resource_wrapped.is_none() { 
						self.addch(' ');
					}else{
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.mv(stats_row + 2, LAND_STATS_COL);
					}
					self.addstr(&arability_str_lns[1]);
				}
			}
			
			// show resource, road, zone, owner if no bldgs or units
			if space_empty_ign_zone {
				///// resource
				if let Some(resource) = resource_wrapped {
					if player.stats.resource_discov(resource) {
						self.mv(r_off, LAND_STATS_COL); r_off += 1;
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.attron(COLOR_PAIR(resource.zone.to_color()));
						self.addstr(&format!("{}", resource.nm[self.state.local.lang_ind]));
						self.attroff(COLOR_PAIR(resource.zone.to_color()));
						
						let window_w = 40;
						
						macro_rules! lr_txt{($row: expr, $l_txt: expr, $r_txt: expr) => {
							self.mv($row, UNIT_STATS_COL+2);
							self.state.txt_list.add_b(&mut self.state.renderer);
							self.state.renderer.addstr($l_txt);
							self.mv($row, UNIT_STATS_COL + window_w - $r_txt.len() as i32);
							self.state.txt_list.add_b(&mut self.state.renderer);
							self.state.renderer.addstr($r_txt);
							$row += 1;
						};};
						
						macro_rules! print_extended_resource_info{() => {
							let mut row = stats_row - 2;
							
							// center title
							self.mv(row, (window_w - resource.nm.len() as i32)/2 + UNIT_STATS_COL as i32); row += 2;
							self.state.txt_list.add_b(&mut self.state.renderer);
							self.addstr(&resource.nm[self.state.local.lang_ind]);
							
							lr_txt!(row, &self.state.local.Zoning_req_to_use, resource.zone.to_str());
							
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
				self.mv(r_off, LAND_STATS_COL); r_off += 1;
				if let Some(ex) = ex_wrapped {
					// structure
					if let Some(s) = ex.actual.structure {
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.state.renderer.addstr(match s.structure_type {
								StructureType::Road => {&self.state.local.Road}
								StructureType::Wall => {&self.state.local.Wall}
								StructureType::Gate => {&self.state.local.Gate}
								StructureType::N => {panicq!("invalid structure type")}
						});
						
						// damaged
						if s.health != std::u8::MAX {
							self.state.txt_list.add_b(&mut self.state.renderer);
							self.mv(r_off, LAND_STATS_COL); r_off += 1;
							let health_frac = 100.*(s.health as f32) / (std::u8::MAX as f32); 
							colorize(health_frac, true, &mut self.state.renderer);
							self.addstr(&format!("{:.1}%", 100.-health_frac));
							colorize(health_frac, false, &mut self.state.renderer);
							self.state.renderer.addstr(&self.state.local.damaged);
						}
					
					// Zone
					}else if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() && !ex.actual.ret_zone_type().is_none() && ex.actual.owner_id == Some(self.state.iface_settings.cur_player) {
						let zt = ex.actual.ret_zone_type().unwrap();
						self.state.txt_list.add_b(&mut self.state.renderer);
						self.addstr(zt.to_str());
					}
					
					/////////////////////////////////////// zone debug info
					#[cfg(any(feature="opt_debug", debug_assertions))]
					if self.state.iface_settings.zoom_ind == map_data.max_zoom_ind() && !ex.actual.ret_zone_type().is_none() {
						let zt = ex.actual.ret_zone_type().unwrap();

						if let Some(zone_ex) = players[ex.actual.owner_id.unwrap() as usize].zone_exs.get(&return_zone_coord(map_cur_coord, *map_data.map_szs.last().unwrap())) {
							/////////////////////////////
							// print zone demands
							if let Some(demand_weighted_sum) = zone_ex.demand_weighted_sum[zt as usize] {
								self.mv(1,0);
								self.addstr(&format!("Demand weighted sum: {}", demand_weighted_sum));
							}
							
							/////////////////////////////////////////// 
							// ZoneDemandRaw
							let mut row = 2;
							for (zone_ind, demand_raw) in zone_ex.demand_raw.iter().enumerate() { // indexed by ZoneType
								if let Some(zdr) = demand_raw {
									self.mv(row, 0); row += 1;
									self.addstr(&format!("{} date: {}", ZoneType::from(zone_ind).to_str(), self.state.local.date_str(zdr.turn_computed)));
									
									// loop over ZoneDemandType
									for (zone_demand_ind, demand) in zdr.demand.iter().enumerate() {
										self.mv(row, 0); row += 1;
										self.addstr(&format!("   {}: {}", ZoneDemandType::from(zone_demand_ind).to_str(), demand));
									}
									row += 1;
								}
							}
							
							// city hall dist
							self.mv(row, 0);
							match zone_ex.ret_city_hall_dist() {
								Dist::NotInit => {self.addstr("Not init");}
								Dist::NotPossible {turn_computed} => {self.addstr(&format!("Not possible, date: {}", self.state.local.date_str(turn_computed)));}
								Dist::Is {dist, bldg_ind} | Dist::ForceRecompute {dist, bldg_ind} => {self.addstr(&format!("Dist: {}, bldg_ind: {}", dist, bldg_ind));}
							}
							///////////////////////////////////////////////////////
						}
					}
					
					// Country owner
					if let Some(owner_id) = ex.actual.owner_id {
						self.print_owner(r_off+1, &players[owner_id as usize], &gstate.relations);
					}	
				} // extended data
			}
		
			//self.addstr(&format!("{} {}", mzc.elevation, mzc.arability));
		}
		/////////////////////////////// end land stats
		
		//////////////////////////
		// show group movement (select rectangle)
		if let Some(action) = self.state.iface_settings.add_action_to.first_action() {
			let full_zoom = self.state.iface_settings.zoom_ind == map_data.max_zoom_ind();
			let mut roff = r_off - 1;
			
			match action.action_type {
				// rectangle started
				ActionType::GroupMv {start_coord: Some(_), end_coord: None} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						self.state.local.Select_a_rectangular_group_of_units.clone(),
						(self.state.local.Move_corner.clone(), self.state.local.drag_the_X.clone()),
						(self.state.local.Finish.clone(), self.state.local.stop_dragging.clone()), self);
					return;
				}
				// rectangle started
				ActionType::BrigadeCreation {start_coord: Some(_), end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						self.state.local.Create_brigade_by_drawing_a_rectangle.clone(),
						(self.state.local.Change_location.clone(), self.state.local.move_mouse.clone()),
						(self.state.local.Finish.clone(), self.state.local.stop_dragging.clone()), self);
					return;
				}
				// rectangle started
				ActionType::SectorCreation {start_coord: Some(_), end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						self.state.local.Create_sector_by_drawing_a_rectangle.clone(),
						(self.state.local.Change_location.clone(), self.state.local.move_mouse.clone()),
						(self.state.local.Finish.clone(), self.state.local.stop_dragging.clone()), self);
					return;
				}
				// enter to start selecting
				ActionType::BrigadeCreation {start_coord: None, end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						self.state.local.Create_brigade_by_drawing_a_rectangle.clone(),
						(self.state.local.Change_location.clone(), self.state.local.move_mouse.clone()),
						(self.state.local.Start_selecting.clone(), self.state.local.click_and_drag.clone()), self);
					return;
				}
				// enter to start selecting
				ActionType::SectorCreation {start_coord: None, end_coord: None, ..} => {
					print_mv_to_sel2(full_zoom, PathValid::True, None, &mut roff,
						self.state.local.Create_sector_by_drawing_a_rectangle.clone(),
						(self.state.local.Change_location.clone(), self.state.local.move_mouse.clone()),
						(self.state.local.Start_selecting.clone(), self.state.local.click_and_drag.clone()), self);
					return;
				}
				_ => {}
			}
		}
		
		self.print_unit_bldg_stats(map_cur_coord, stats_row, r_off+1, show_land, map_data, exs, player, players, units, bldg_config, bldgs, gstate, temps);
	}
}

