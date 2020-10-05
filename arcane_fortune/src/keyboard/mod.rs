use crate::config_load::*;

macro_rules! create_keys{($($entry:ident = $nm: expr)*) => (
	// each field's value is the value we should actually
	// expect from the keyboard
	#[derive(Clone)]
	pub struct KeyboardMap {$(pub $entry: i32),*}
	
	impl KeyboardMap {
		pub fn new() -> Self {
			const KEYBOARD_CONFIG: &str = "config/keyboard.txt";
			let key_sets = config_parse(read_file(KEYBOARD_CONFIG));
			Self {$($entry: find_kbd_key($nm, &key_sets)),*}
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
	start_tabbing_through_bottom_screen_mode = "start_tabbing_through_bottom_screen_mode"
	start_tabbing_through_right_screen_mode = "start_tabbing_through_right_screen_mode"
	forward_tab = "forward_tab"
	backward_tab = "backward_tab"
);

