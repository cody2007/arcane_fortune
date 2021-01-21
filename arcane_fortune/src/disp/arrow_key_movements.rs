use super::*;

enum_From!{ViewMvMode {Cursor, Screen, Float}}

#[derive(PartialEq)]
pub enum CoordSet {X, Y}

pub const SCROLL_ACCEL_INIT: f32 = 4.;
const SCROLL_ACCEL: f32 = 2.;
const SCROLL_MAX_V: f32 = 100.;

const FLOAT_SZ_Y: isize = 10;
const FLOAT_SZ_X: isize = FLOAT_SZ_Y*3;

impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn reset_unit_subsel_only(&mut self) {
		match self.state.iface_settings.add_action_to {
			AddActionTo::None |
			AddActionTo::NoUnit {..} |
			AddActionTo::BrigadeBuildList {..} |
			AddActionTo::AllInBrigade {..} => {
				self.state.iface_settings.unit_subsel = 0;
			} AddActionTo::IndividualUnit {..} => {}
		}
	}
	
	pub fn reset_unit_subsel(&mut self) {
		self.reset_unit_subsel_only();
		self.ui_mode = UIMode::None;
	}
	
	fn update_cursor_state(&mut self, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		self.reset_unit_subsel();
		self.state.iface_settings.chk_cursor_bounds(map_data);
		let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
		self.update_move_search_ui(map_data, exs, units, bldgs, gstate, players, map_sz);
	}
	
	// the user clicked somewhere on the map, set the text cursor position to that location
	// input is in screen coordinates
	pub fn set_text_coord(&mut self, coord: Coord, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState) {
		self.state.iface_settings.cur = coord;
		self.update_cursor_state(map_data, exs, units, bldgs, gstate, players);
	}
	
	// move the cursor when the arrow keys are pressed
	pub fn linear_update(&mut self, coord_set: CoordSet, sign: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		match coord_set {
			CoordSet::X => {self.state.iface_settings.cur.x += sign;}
			CoordSet::Y => {self.state.iface_settings.cur.y += sign;}
		}
		
		self.state.iface_settings.reset_cur_accel();
		self.update_cursor_state(map_data, exs, units, bldgs, gstate, players);
	}
	
	// move view when the arrow keys are pressed
	pub fn linear_update_screen(&mut self, coord_set: CoordSet, sign: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		match coord_set {
			CoordSet::X => {self.state.iface_settings.map_loc.x += sign;}
			CoordSet::Y => {self.state.iface_settings.map_loc.y += sign;}
		}
		
		self.state.iface_settings.reset_cur_accel();
		self.update_cursor_state(map_data, exs, units, bldgs, gstate, players);
	}
	
	pub fn linear_update_float(&mut self, coord_set: CoordSet, sign: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		let map_screen_ctr = self.state.iface_settings.map_screen_center();
		
		let in_bounds = match coord_set {
			CoordSet::X => is_update_in_float_bound(self.state.iface_settings.cur.x + sign, map_screen_ctr.x, FLOAT_SZ_X),
			CoordSet::Y => is_update_in_float_bound(self.state.iface_settings.cur.y + sign, map_screen_ctr.y, FLOAT_SZ_Y)
		};
		
		if in_bounds {self.linear_update(coord_set, sign, map_data, exs, units, bldgs, gstate, players);}
		else {self.linear_update_screen(coord_set, sign, map_data, exs, units, bldgs, gstate, players);}
	}
	
	pub fn accel_update(&mut self, coord_set: CoordSet, sign: f32, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		let iface_settings = &mut self.state.iface_settings;
		match coord_set {
			CoordSet::X => {
				iface_settings.cur.x = ((iface_settings.cur.x as f32) + (iface_settings.cur_v.x * sign)) as isize;
				iface_settings.cur_v.x = if iface_settings.cur_v.x > SCROLL_MAX_V {SCROLL_MAX_V}
						else {iface_settings.cur_v.x + SCROLL_ACCEL};
			} CoordSet::Y => {
				iface_settings.cur.y = ((iface_settings.cur.y as f32) + (iface_settings.cur_v.y * sign)) as isize;
				iface_settings.cur_v.y = if iface_settings.cur_v.y > SCROLL_MAX_V {SCROLL_MAX_V}
						else {iface_settings.cur_v.y + SCROLL_ACCEL};
			}
		}
		
		self.update_cursor_state(map_data, exs, units, bldgs, gstate, players);
	}
	
	pub fn accel_update_screen(&mut self, coord_set: CoordSet, sign: f32, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		let iface_settings = &mut self.state.iface_settings;
		match coord_set {
			CoordSet::X => {
				iface_settings.map_loc.x = ((iface_settings.map_loc.x as f32) + (iface_settings.map_loc_v.x * sign)) as isize;
				iface_settings.map_loc_v.x = if iface_settings.map_loc_v.x > SCROLL_MAX_V {SCROLL_MAX_V}
						else {iface_settings.map_loc_v.x + SCROLL_ACCEL};
			} CoordSet::Y => {
				iface_settings.map_loc.y = ((iface_settings.map_loc.y as f32) + (iface_settings.map_loc_v.y * sign)) as isize;
				iface_settings.map_loc_v.y = if iface_settings.map_loc_v.y > SCROLL_MAX_V {SCROLL_MAX_V}
						else {iface_settings.map_loc_v.y + SCROLL_ACCEL};
			}
		}
		
		self.update_cursor_state(map_data, exs, units, bldgs, gstate, players);
	}
	
	pub fn accel_update_float(&mut self, coord_set: CoordSet, sign: f32, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>) {
		let map_screen_ctr = self.state.iface_settings.map_screen_center();
		
		let in_bounds = match coord_set {
			CoordSet::X => is_update_in_float_bound(((self.state.iface_settings.cur.x as f32) + (self.state.iface_settings.cur_v.x * sign)) as isize, map_screen_ctr.x, FLOAT_SZ_X),
			CoordSet::Y => is_update_in_float_bound(((self.state.iface_settings.cur.y as f32) + (self.state.iface_settings.cur_v.y * sign)) as isize, map_screen_ctr.y, FLOAT_SZ_Y)
		};
		
		if in_bounds {self.accel_update(coord_set, sign, map_data, exs, units, bldgs, gstate, players);}
		else {self.accel_update_screen(coord_set, sign, map_data, exs, units, bldgs, gstate, players);}
	}
	
	pub fn chg_zoom(&mut self, inc: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, gstate: &mut GameState,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz) {
		let iface_settings = &mut self.state.iface_settings;
		if ((inc == 1) && (iface_settings.zoom_ind == map_data.max_zoom_ind())) ||
			((inc == -1) && (iface_settings.zoom_ind - 1) == ZOOM_IND_SUBMAP) { return; }
		
		let zoom_ind_prev = iface_settings.zoom_ind;
		if inc == 1 { iface_settings.zoom_ind += 1; } else { iface_settings.zoom_ind -= 1; }
		
		debug_assertq!(zoom_ind_prev <= map_data.max_zoom_ind());
		debug_assertq!(iface_settings.zoom_ind <= map_data.max_zoom_ind());
		
		let dy = (iface_settings.cur.y as f32) - (MAP_ROW_START as f32);
		let dx = iface_settings.cur.x as f32;
		
		let h_full = map_data.map_szs[ZOOM_IND_ROOT].h as f32;
		let w_full = map_data.map_szs[ZOOM_IND_ROOT].w as f32;
		
		let frac_prev_h = h_full / map_data.map_szs[zoom_ind_prev].h as f32;
		let frac_prev_w = w_full / map_data.map_szs[zoom_ind_prev].w as f32;
		
		let frac_h = h_full / map_data.map_szs[iface_settings.zoom_ind].h as f32;
		let frac_w = w_full / map_data.map_szs[iface_settings.zoom_ind].w as f32;
		
		iface_settings.map_loc.y = (((frac_prev_h*(iface_settings.map_loc.y as f32)) + ((frac_prev_h - frac_h)*dy))/frac_h).round() as isize;
		iface_settings.map_loc.x = (((frac_prev_w*(iface_settings.map_loc.x as f32)) + ((frac_prev_w - frac_w)*dx))/frac_w).round() as isize;
		iface_settings.start_map_drag = None;
		
		iface_settings.chk_cursor_bounds(map_data);
		if iface_settings.zoom_ind == map_data.max_zoom_ind() {
			self.update_move_search_ui(map_data, exs, units, bldgs, gstate, players, map_sz);
		}
	}
}

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn chk_cursor_bounds(&mut self, map_data: &MapData){
		let map_sz = map_data.map_szs[self.zoom_ind];
		
		// update map_loc_ when scrolling and set cursor within screen boundaries
		if self.cur.y < (MAP_ROW_START as isize) {
			self.map_loc.y -= (MAP_ROW_START as isize) - self.cur.y + 1;
			self.cur.y = MAP_ROW_START as isize;
		}
		if self.cur.x < 0 {
			self.map_loc.x -= 1 - self.cur.x;
			self.cur.x = 0;
		}
		
		let d = (self.screen_sz.w as isize) - (MAP_COL_STOP_SZ as isize);
		if self.cur.x >= d {
			self.map_loc.x += self.cur.x - d + 1;
			self.cur.x = d - 1;
		}
		
		let d = (self.screen_sz.h as isize) - (MAP_ROW_STOP_SZ as isize);
		if self.cur.y >= d {
			self.map_loc.y += self.cur.y - d + 1;
			self.cur.y = d - 1;
		}

		let d = map_sz.h as isize;
		if self.cur.y > d {self.cur.y = d;}

		// dont let map_loc_y get out of bounds
		let d = (map_sz.h as isize) - (self.map_screen_sz.h as isize);
		if self.map_loc.y > d {self.map_loc.y = d;}

		if self.map_loc.y < 0 {self.map_loc.y = 0;}
		
		// wrap map_loc_x around map
		let d = map_sz.w as isize;
		if self.map_loc.x >= d {self.map_loc.x %= d;};
		if self.map_loc.x < 0 {self.map_loc.x = d + (self.map_loc.x % d);}
	}

	pub fn reset_cur_accel(&mut self){
		self.cur_v = VelocCoord {x: SCROLL_ACCEL_INIT, y: SCROLL_ACCEL_INIT};	
	}
	
	// center position (in screen coordinates) of the map
	fn map_screen_center(&self) -> Coord {
		Coord {
			y: ((self.map_screen_sz.h/2) + MAP_ROW_START) as isize,
			x: (self.map_screen_sz.w/2) as isize
		}
	}
	
	pub fn ctr_on_cur(&mut self, map_data: &MapData) {
		let map_screen_ctr = self.map_screen_center();
		
		self.map_loc.y -= map_screen_ctr.y - self.cur.y;
		self.map_loc.x -= map_screen_ctr.x - self.cur.x;
		
		self.cur = map_screen_ctr;
		
		self.chk_cursor_bounds(map_data);
	}
}

// returns true if the candidate cursor update is still within the float window
// cand_coord is the candidate coordinate for the update
// ctr_coord is the center of the map (in screen coordinates)
fn is_update_in_float_bound(cand_coord: isize, ctr_coord: isize, float_sz: isize) -> bool {
	cand_coord > (ctr_coord - float_sz) &&
	cand_coord < (ctr_coord + float_sz)
}

impl ViewMvMode {
	pub fn toggle(&mut self) {
		*self = match self {
			ViewMvMode::Screen => ViewMvMode::Cursor,
			ViewMvMode::Cursor => ViewMvMode::Float,
			ViewMvMode::Float => ViewMvMode::Screen,
			ViewMvMode::N => {panicq!("invalid view setting")}
		};
	}
}

