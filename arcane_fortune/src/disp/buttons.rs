#![allow(non_snake_case)]
use super::*;
use crate::KeyboardMap;

#[derive(Clone, PartialEq)]
pub struct ScreenCoord {
	pub y: isize,
	pub x: isize
}

// text cursor
pub fn cursor_pos(d: &DispState) -> ScreenCoord {
	let mut y = 0;
	let mut x = 0;
	d.getyx(stdscr(), &mut y, &mut x);
	ScreenCoord {y: y as isize, x: x as isize}
}

#[derive(Clone, PartialEq)]
pub struct ListButton {
	pos: ButtonPosition,
	id: usize // id in the original menu/window list
		    // (only entries shown on the screen are added to the `Buttons` struct)
	
	// while OptionsUI would've been a great place to put this information,
	// it would require: storing the OptionsUI in the UIMode enum, to pass between
	// printing and keybinding code. either general code or specific code for each
	// UIMode value would be needed to update which items are and aren't on the screen.
	// putting all this in Buttons allows generic key-checking/update routines to be 
	// used. in the future, pulling the OptionsUI out of UIMode and merging it w/ ListButton
	// may be preferred.
}

#[derive(Clone, PartialEq)]
pub struct ButtonPosition {
	start: ScreenCoord,
	end: ScreenCoord
}

#[derive(Clone, PartialEq)]
pub struct Button {
	pub pos: Option<ButtonPosition>, // can be none if off-screen
	key: i32,
	pub txt: String,
	pub tool_tip: String,
	pub hovered_over: bool // used for printing tool tips
}

macro_rules! create_buttons{($($entry:ident = $key: ident, $txt: ident, $tool_tip: ident)*) => (
	#[derive(PartialEq)]
	pub struct Buttons {
		pub list: Vec<ListButton>, // for window lists, menus
		$(pub $entry: Button),*
	}
	
	impl Buttons {
		pub fn new(kbd: &KeyboardMap, l: &Localization) -> Self {
			Self {
				list: Vec::new(),
				$($entry: Button {
					pos: None,
					key: kbd.$key,
					txt: l.$txt.clone(),
					tool_tip: l.$tool_tip.clone(),
					hovered_over: false
				}),*
			}
		}
		
		pub fn clear_positions(&mut self) {
			self.list.clear();
			$(self.$entry.pos = None;
			  self.$entry.hovered_over = false;)*
		}
		
		pub fn add(&mut self, start: ScreenCoord, id: usize, d: &DispState) {
			let end = cursor_pos(d);
			self.list.push(ListButton {
				id,
				pos: ButtonPosition {start, end}
			});
		}
		
		pub fn list_item_clicked(&self, mouse_event: &Option<MEVENT>) -> Option<usize> {
			if let Some(mouse) = &mouse_event {
				if !lbutton_released(mouse_event) && !lbutton_clicked(mouse_event) {return None;}
				
				for entry in self.list.iter() {
					if entry.pos.contains(mouse.y, mouse.x) {
						return Some(entry.id);
					}
				}
			}
			None
		}
		pub fn list_item_hovered(&self, mouse_event: &Option<MEVENT>) -> Option<usize> {
			if let Some(mouse) = &mouse_event {
				if lbutton_released(mouse_event) || lbutton_clicked(mouse_event) {return None;}
				
				for entry in self.list.iter() {
					if entry.pos.contains(mouse.y, mouse.x) {
						return Some(entry.id);
					}
				}
			}
			None
		}
		
		pub fn print_tool_tip(&self, disp_chars: &DispChars, d: &mut DispState) {
			$(if self.$entry.print_tool_tip(disp_chars, d) {return;})*
		}
	}
);}

// button.nm = kbd.nm, l.nm
create_buttons!(
	move_unit = move_unit, Move_to, Empty_txt
	group_move = group_move, Group_move, Group_move_tool_tip
	
	zone_residential = residential, Zone_residential, Empty_txt
	zone_industrial = industrial, Zone_industrial, Empty_txt
	zone_agricultural = agricultural, Zone_agricultural, Empty_txt
	zone_business = business, Zone_business, Empty_txt
	
	tax_residential = residential, Residential, Empty_txt
	tax_industrial = industrial, Industrial, Empty_txt
	tax_agricultural = agricultural, Agricultural, Empty_txt
	tax_business = business, Business, Empty_txt
	
	increase_tax = increase_tax, Empty_txt, Empty_txt
	increase_tax_alt = increase_tax_alt, Empty_txt, Empty_txt
	decrease_tax = decrease_tax, Empty_txt, Empty_txt
	
	fortify = fortify, Fortify, Fortify_tool_tip
	pass_move = pass_move, Pass_turn, Empty_txt
	auto_explore = auto_explore, Auto_explore, Empty_txt
	disband = disband, Disband, Disband_tool_tip
	attack = attack, Attack, Empty_txt
	soldier_automate = soldier_automate, Automate, Soldier_automate_tool_tip
	automate_zone_creation = automate_zone_creation, Automate_worker, Automate_worker_tool_tip
	continue_bldg_construction = continue_bldg_construction, Cont_bldg_construction, Cont_bldg_construction_tool_tip
	build_bldg = build_bldg, Create_bldg, Empty_txt
	repair_wall = repair_wall, Repair_wall_at, Empty_txt
	unload_boat = unload_boat, Unload_units, Empty_txt
	build_gate = build_gate, Build_gate, Build_gate_tool_tip
	build_wall = build_wall, Build_wall_to, Empty_txt
	move_with_cursor = move_with_cursor, Move_w_cursor, Move_w_cursor_tool_tip
	leave_brigade = join_or_leave_brigade, Leave_brigade, Empty_txt
	join_brigade = join_or_leave_brigade, Join_brigade, Empty_txt
	build_road = build_road, Build_road_to, Empty_txt
	view_brigade = view_brigade, view_brigade, Empty_txt
	view_production = view_production, view_production, Empty_txt
	change_bldg_production = change_bldg_production, Empty_txt, Empty_txt
	progress_day = progress_day, Next_day, Empty_txt
	progress_day_ign_unmoved_units = progress_day_ign_unmoved_units, Nxt_day, Empty_txt
	progress_month = progress_month, Next_month, Empty_txt
	finish_all_unit_actions = finish_all_unit_actions, Fin_all, Fin_all_tool_tip
	stop_fin_all_unit_actions = finish_all_unit_actions, Stop, Empty_txt
	assign_action_to_all_in_brigade = assign_action_to_all_in_brigade, all_battalions_in_brigade, all_battalions_in_brigade_tool_tip
	add_action_to_brigade_build_list = add_action_to_brigade_build_list, brigade_build_list, brigade_build_list_tool_tip
	show_expanded_submap = show_expanded_submap, expand, Empty_txt
	hide_submap = show_expanded_submap, hide, Empty_txt
	Press_to_add_action_to_brigade_build_list = add_action_to_brigade_build_list, Press_to_add_action_to_brigade_build_list, brigade_build_list_tool_tip
	Esc_to_close = esc, to_close, Empty_txt
	Cancel = esc, Cancel, Empty_txt
	Save = enter, Save, Empty_txt
	Confirm = enter, Confirm, Empty_txt
	change_brigade_repair = change_brigade_repair, change_brigade_repair, brigade_repair_tool_tip
	clear_brigade_repair = clear_brigade_repair, clear_brigade_repair, brigade_repair_tool_tip
	to_go_back = left, to_go_back, Empty_txt
	Open = enter, Open, Empty_txt
	Load_game = enter, Load_game, Empty_txt
	New_game = enter, New_game, Empty_txt
	tab = tab, Empty_txt, Empty_txt
	Exit = enter, Exit, Empty_txt
	rm_bldgs_and_zones = rm_bldgs_and_zones, rm_bldgs_and_zones, Empty_txt
);

impl Button {
	// should match button.print() ...
	pub fn print_txt(&self, l: &Localization) -> String {
		let mut key_txt = key_txt(self.key, l);
		
		// text contains custom keybinding printing location
		if let Some(_) = self.txt.find("[]") {
			self.txt.replace("[]", &key_txt)
		
		// text is format "k: some text"
		// inserts 'k' and optionally ':' if it's not present
		}else{
			let add_colon = self.txt.chars().nth(0) != Some(':');
			
			if add_colon {key_txt.push_str(": ");}
			key_txt.push_str(&self.txt);
			key_txt
		}
	}
	
	pub fn print_tool_tip(&self, disp_chars: &DispChars, d: &mut DispState) -> bool {
		if self.hovered_over && self.tool_tip.len() != 0 {
			const BOX_COLOR: CInd = CLOGO;
			let lines = wrap_txt(&self.tool_tip, 50);
			let len = lines.iter().max_by_key(|l| l.len()).unwrap().len();
			
			let (mut y,x) = {
				let pos = self.pos.as_ref().unwrap();
				let end = &pos.end;
				let screen_sz = getmaxyxu(d);
				
				// would wrap off the right of the page -- attempt to place on left
				if (end.x as i32 + len as i32 + 4 - 1) > screen_sz.w as i32 {
					let start = &pos.start;
					if start.x >= (len+5) as isize {
						(start.y as i32 - 1, start.x as i32 - len as i32 - 5)
					}else{return true;}
				}else{
					(end.y as i32 - 1, end.x as i32 + 1)
				}
			};
			
			{ // top line
				d.attron(COLOR_PAIR(BOX_COLOR));
				d.mv(y, x); y += 1;
				d.addch(disp_chars.ulcorner_char);
				for _ in 0..(len+2) {
					d.addch(disp_chars.hline_char);
				}
				d.addch(disp_chars.urcorner_char);
			}
			
			// text lines
			for (line_i, line) in lines.iter().enumerate() {
				d.mv(y, x); y += 1;
				d.addch(disp_chars.vline_char);
				d.attroff(COLOR_PAIR(BOX_COLOR));
				if line_i == 0 {d.addch(' ');}
				d.addstr(line);
				for _ in line.len()..(len+1) {
					d.addch(' ');
				}
				if line_i != 0 {d.addch(' ');}
				d.attron(COLOR_PAIR(BOX_COLOR));
				d.addch(disp_chars.vline_char);
			}
			
			{ // bottom line
				d.mv(y, x);
				d.addch(disp_chars.llcorner_char);
				for _ in 0..(len+2) {
					d.addch(disp_chars.hline_char);
				}
				d.addch(disp_chars.lrcorner_char);
				d.attroff(COLOR_PAIR(BOX_COLOR));
			}

			true
		}else{false}
	}
	
	// should match button.print_txt() ...
	// If iface_settings is provided: key highlighting is only shown when UIMode is none
	// If iface_settings is not provided: key highlighting is always shown (ex. if the highlighting is for entries in a window)
	// Returns the button width
	pub fn print(&mut self, iface_settings: Option<&IfaceSettings>, l: &Localization, d: &mut DispState) -> isize {
		let w = self.print_txt(l).len() as isize;
		
		let pos = {
			let start = cursor_pos(d);
			let end = ScreenCoord {y: start.y, x: start.x + w};
			ButtonPosition {start, end}
		};
		
		// set color and mouse cursor if mouse is over the button
		let non_key_txt_color = (|| {
			if let Some((y,x)) = d.mouse_pos() {
				// hovering
				if pos.contains(y,x) {
					d.set_mouse_to_hand();
					self.hovered_over = true; // used for printing tool tips
					return CGRAY;
				}
			}
			CWHITE
		})();
		
		// print
		{
			// "some text [] asdf" -> "some text k asdf" where k is the keybinding
			if let Some(_) = self.txt.find("[]") {
				// only show highlighting when no window is active (ex. for shortcut keys at the bottom of the screen)
				let key_attr = if let Some(iface_settings) = iface_settings {
					shortcut_or_default_show(&iface_settings.ui_mode, non_key_txt_color)
				// highlighting always, ex. for shortcuts in a window
				}else{shortcut_indicator()};
				
				let tag = KeyValColor {
					key: String::from("[]"),
					val: key_txt(self.key, l),
					attr: key_attr
				};
				color_tags_print(&self.txt, &vec![tag], Some(COLOR_PAIR(non_key_txt_color)), d);
			
			// text is format "k: some text"
			// inserts 'k' and optionally ':' if it's not present
			}else{
				// only show highlighting when no window is active (ex. for shortcut keys at the bottom of the screen)
				if let Some(iface_settings) = iface_settings {
					iface_settings.print_key(self.key, l, d);
				// highlighting always, ex. for shortcuts in a window
				}else{
					print_key_always_active(self.key, l, d);
				}
				
				let add_colon = self.txt.chars().nth(0) != Some(':');
				if add_colon {d.addstr(": ");}
				addstr_c(&self.txt, non_key_txt_color, d);
			}
		}
		
		self.pos = Some(pos);
		w
	}
	
	pub fn print_key_only(&mut self, iface_settings: Option<&IfaceSettings>, l: &Localization, d: &mut DispState) -> isize {
		let w = key_txt(self.key, l).len() as isize;
		
		let pos = {
			let start = cursor_pos(d);
			let end = ScreenCoord {y: start.y, x: start.x + w};
			ButtonPosition {start, end}
		};
		
		// print
		{
			// only show highlighting when no window is active (ex. for shortcut keys at the bottom of the screen)
			if let Some(iface_settings) = iface_settings {
				iface_settings.print_key(self.key, l, d);
			// highlighting always, ex. for shortcuts in a window
			}else{
				print_key_always_active(self.key, l, d);
			}
		}
		
		self.pos = Some(pos);
		w
	}
	
	pub fn print_without_parsing(&mut self, d: &mut DispState) {
		let pos = {
			let start = cursor_pos(d);
			let end = ScreenCoord {y: start.y, x: start.x + self.txt.len() as isize};
			ButtonPosition {start, end}
		};
		
		// set color and mouse cursor if mouse is over the button
		let color = (|| {
			if let Some((y,x)) = d.mouse_pos() {
				// hovering
				if pos.contains(y,x) {
					d.set_mouse_to_hand();
					return CGRAY;
				}
			}
			CWHITE
		})();
		
		addstr_c(&self.txt, color, d);
		
		self.pos = Some(pos);
	}
	
	pub fn activated(&self, key_pressed: i32, mouse_event: &Option<MEVENT>) -> bool {
		// if self.pos is not set, it means the button was not shown on the screen
		// in the past frame and the button should not be considered activatable
		// (returning true here could prevent other button handlers, ex. in
		//  gcore/non_menu_keys.rs w/ the same keybinding
		//  from running)
		if let Some(p) = &self.pos {
			if key_pressed == self.key {return true;}
			if !lbutton_released(mouse_event) && !lbutton_clicked(mouse_event) {return false;}
			
			if let Some(mouse) = mouse_event {
				return p.contains(mouse.y, mouse.x);
			}else{panicq!("mouse event should not be none");}
		}
		false
	}
	
	pub fn activated_ign_not_being_on_screen(&self, key_pressed: i32, mouse_event: &Option<MEVENT>) -> bool {
		if let Some(p) = &self.pos {
			if lbutton_released(mouse_event) || lbutton_clicked(mouse_event) {
				if let Some(mouse) = mouse_event {
					if p.contains(mouse.y, mouse.x) {return true;}
				}else{panicq!("mouse event should not be none");}
			}
		}
		key_pressed == self.key
	}
	
	pub fn hovered(&self, mouse_event: &Option<MEVENT>) -> bool {
		if let Some(p) = &self.pos {
			if lbutton_released(mouse_event) || lbutton_clicked(mouse_event) {return false;}
			
			if let Some(mouse) = mouse_event {
				return p.contains(mouse.y, mouse.x);
			}
		}
		false
	}
}

impl ButtonPosition {
	pub fn contains(&self, y: i32, x: i32) -> bool {
		y >= self.start.y as i32 && y <= self.end.y as i32 &&
		x >= self.start.x as i32 && x < self.end.x as i32
	}
}

