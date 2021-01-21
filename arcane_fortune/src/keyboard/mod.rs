use crate::renderer::*;
use crate::config_load::*;

#[derive(Clone)]
pub enum MouseClick {Left, Right, Middle}

impl MouseClick {
	pub fn released(&self, mouse_event: &Option<MEVENT>) -> bool {
		match self {
			MouseClick::Left => {lbutton_released(mouse_event)}
			MouseClick::Right => {rbutton_released(mouse_event)}
			MouseClick::Middle => {mbutton_released(mouse_event)}
		}
	}
	
	pub fn clicked(&self, mouse_event: &Option<MEVENT>) -> bool {
		match self {
			MouseClick::Left => {lbutton_clicked(mouse_event)}
			MouseClick::Right => {rbutton_clicked(mouse_event)}
			MouseClick::Middle => {mbutton_clicked(mouse_event)}
		}
	}
	
	pub fn pressed(&self, mouse_event: &Option<MEVENT>) -> bool {
		match self {
			MouseClick::Left => {lbutton_pressed(mouse_event)}
			MouseClick::Right => {rbutton_pressed(mouse_event)}
			MouseClick::Middle => {mbutton_pressed(mouse_event)}
		}
	}
	
	pub fn dragging(&self, mouse_event: &Option<MEVENT>) -> bool {
		match self {
			MouseClick::Left => {ldragging(mouse_event)}
			MouseClick::Right => {rdragging(mouse_event)}
			MouseClick::Middle => {mdragging(mouse_event)}
		}
	}
	
	pub fn released_clicked_or_dragging(&self, mouse_event: &Option<MEVENT>) -> bool {
		self.released(mouse_event) || self.clicked(mouse_event) || self.dragging(mouse_event)
	}
	
	pub fn released_clicked_pressed_or_dragging(&self, mouse_event: &Option<MEVENT>) -> bool {
		self.released(mouse_event) || self.clicked(mouse_event) || self.pressed(mouse_event) || self.dragging(mouse_event)
	}
}

macro_rules! create_keys{($($entry:ident = $nm: expr)*) => (
	// each field's value is the value we should actually
	// expect from the keyboard
	#[derive(Clone)]
	pub struct KeyboardMap {
		$(pub $entry: i32),*,
		pub map_drag: MouseClick,
		pub action_drag: MouseClick,
		pub action_cancel: MouseClick
	}
	
	impl KeyboardMap {
		pub fn new() -> Self {
			const KEYBOARD_CONFIG: &str = "config/keyboard.txt";
			let key_sets = config_parse(read_file(KEYBOARD_CONFIG));
			Self {
				$($entry: find_kbd_key($nm, &key_sets)),*,
				map_drag: find_mouse_click("map_drag", &key_sets),
				action_drag: find_mouse_click("action_drag", &key_sets),
				action_cancel: find_mouse_click("action_cancel", &key_sets)
			}
		}
	}
);}

create_keys!(
	left = "left"
	down = "down"
	right = "right"
	up = "up"
	
	fast_left = "fast_left"
	fast_down = "fast_down"
	fast_right = "fast_right"
	fast_up = "fast_up"
	
	diag_up_left = "diag_up_left"
	diag_up_right = "diag_up_right"
	diag_down_left = "diag_down_left"
	diag_down_right = "diag_down_right"
	
	fast_diag_up_left = "fast_diag_up_left"
	fast_diag_up_right = "fast_diag_up_right"
	fast_diag_down_left = "fast_diag_down_left"
	fast_diag_down_right = "fast_diag_down_right"
	
	//////// misc
	enter = "enter"
	tab = "tab"
	esc = "esc"
	zoom_in = "zoom_in"
	zoom_out = "zoom_out"
	show_expanded_submap = "show_expanded_submap"
	
	open_top_menu = "open_top_menu"
	
	toggle_cursor_mode = "toggle_cursor_mode"
	center_on_cursor = "center_on_cursor"
	
	progress_day = "progress_day"
	progress_day_ign_unmoved_units = "progress_day_ign_unmoved_units"
	progress_month = "progress_month"
	finish_all_unit_actions = "finish_all_unit_actions"
	change_brigade_repair = "change_brigade_repair"
	clear_brigade_repair = "clear_brigade_repair"
	center_on_next_unmoved_unit = "center_on_next_unmoved_unit"
	offer_trade_item = "offer_trade_item"
	request_trade_item = "request_trade_item"
	
	/////// unit actions
	disband = "disband"
	move_unit = "move_unit"
	pass_move = "pass_move"
	change_bldg_production = "change_bldg_production"
	auto_explore = "auto_explore"
	fortify = "fortify"
	build_bldg = "build_bldg"
	build_road = "build_road"
	attack = "attack"
	scale_walls = "scale_walls"
	assassinate = "assassinate"
	build_wall = "build_wall"
	automate_zone_creation = "automate_zone_creation"
	repair_wall = "repair_wall"
	continue_bldg_construction = "continue_bldg_construction"
	group_move = "group_move"
	unload_boat = "unload_boat"
	join_or_leave_brigade = "join_or_leave_brigade"
	move_with_cursor = "move_with_cursor"
	soldier_automate = "soldier_automate"
	view_brigade = "view_brigade"
	view_build_list = "view_build_list"
	view_production = "view_production"
	assign_action_to_all_in_brigade = "assign_action_to_all_in_brigade"
	add_action_to_brigade_build_list = "add_action_to_brigade_build_list"
	build_gate = "build_gate"
	rm_bldgs_and_zones = "rm_bldgs_and_zones"
	
	////////// zones (both creating and taxing)
	agricultural = "agricultural"
	residential = "residential"
	business = "business"
	industrial = "industrial"
	increase_tax = "increase_tax"
	increase_tax_alt = "increase_tax_alt"
	decrease_tax = "decrease_tax"
	
	//////// text mode tabbing (used with screen readers)
	start_tabbing_through_window_screen_mode = "start_tabbing_through_window_screen_mode"
	start_tabbing_through_bottom_screen_mode = "start_tabbing_through_bottom_screen_mode"
	start_tabbing_through_right_screen_mode = "start_tabbing_through_right_screen_mode"
	forward_tab = "forward_tab"
	backward_tab = "backward_tab"
);

use sdl2_lib::{KEY_UP, KEY_DOWN};
impl KeyboardMap {
	// fast or slow up
	pub fn up(&self, k: i32) -> bool {
		self.up_normal(k) || k == self.fast_up as i32
	}
	
	// fast or slow down
	pub fn down(&self, k: i32) -> bool {
		self.down_normal(k) || k == self.fast_up as i32
	}
	
	// i.e., not the fast up
	pub fn up_normal(&self, k: i32) -> bool {
		k == self.up as i32 ||
		k == KEY_UP
	}
	
	// i.e., not the fast down
	pub fn down_normal(&self, k: i32) -> bool {
		k == self.down as i32 ||
		k == KEY_DOWN
	}
}

