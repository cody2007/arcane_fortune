use crate::disp_lib::*;
use crate::map::*;
use crate::units::vars::{UnitTemplate, Unit, RIOTER_NM};
use crate::buildings::*;
//use crate::saving::SmSvType;
use crate::gcore::{Relations, Log};
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::zones::{FogVars, return_zone_coord, StructureData};
use crate::ai::{AIState, BarbarianState};
use crate::player::{Player, PlayerType, Stats, PersonName};
use crate::localization::Localization;

mod vars;
mod fns;
mod color;
mod logo_vars;
mod logo_utils;
mod print_submap;
mod print_bottom_stats;
pub mod print_rside_stats;
pub mod menus;
pub mod plot_window;
pub mod window; pub use window::*;
mod version_status;
mod init_display;
mod new_game_options;
pub mod print_map;
pub mod profiling;
pub mod tree;
pub mod fire;
pub mod assign_action_iface; pub use assign_action_iface::*;
pub mod draw_selection; pub use draw_selection::*;
pub mod buttons; pub use buttons::*;
pub mod pie_plot; pub use pie_plot::*;
pub mod screen_reader; pub use screen_reader::*;

pub use vars::*;
pub use fns::*;
pub use color::*;
pub use logo_vars::*;
pub use logo_utils::*;
pub use plot_window::*;
pub use version_status::*;
pub use init_display::*;
pub use new_game_options::*;
pub use print_map::*;
pub use profiling::*;
pub use tree::*;
pub use fire::*;

#[derive(PartialEq)]
pub enum CoordSet {X, Y}

pub const SCROLL_ACCEL_INIT: f32 = 4.;
const SCROLL_ACCEL: f32 = 2.;
const SCROLL_MAX_V: f32 = 100.;

// Interface settings
impl <'f,'bt,'ut,'rt,'dt>IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	// center cursor on screen
	/*pub fn center_cursor(&mut self, map_data: &mut MapData) {
		self.cur = Coord { 
			y: ((self.map_screen_sz.h / 2) + MAP_ROW_START) as isize,
			x: (self.map_screen_sz.w / 2) as isize };
		self.reset_unit_subsel();
		self.chk_cursor_bounds(map_data);
	}*/
	
	pub fn set_screen_sz(&mut self, d: &mut DispState){
		self.screen_sz = getmaxyxu(d);

		self.map_screen_sz.h = self.screen_sz.h - MAP_ROW_STOP_SZ - MAP_ROW_START;
		self.map_screen_sz.w = self.screen_sz.w - MAP_COL_STOP_SZ;
	}
	
	pub fn reset_unit_subsel(&mut self){
		match self.add_action_to {
			AddActionTo::None |
			AddActionTo::NoUnit {..} |
			AddActionTo::BrigadeBuildList {..} |
			AddActionTo::AllInBrigade {..} => {
				self.unit_subsel = 0;
			} AddActionTo::IndividualUnit {..} => {}
		}
		self.ui_mode = UIMode::None;
	}
	
	pub fn chk_cursor_bounds(&mut self, map_data: &MapData){
		let screen_sz = self.screen_sz;
		let map_sz = map_data.map_szs[self.zoom_ind];
		let map_screen_sz = self.map_screen_sz;
		
		// update map_loc_ when scrolling and set cursor within screen boundaries
		if self.cur.y < (MAP_ROW_START as isize) {
			self.map_loc.y -= (MAP_ROW_START as isize) - self.cur.y + 1;
			self.cur.y = MAP_ROW_START as isize;
		}
		if self.cur.x < 0 {
			self.map_loc.x -= 1 - self.cur.x;
			self.cur.x = 0;
		}
		
		let d = (screen_sz.w as isize) - (MAP_COL_STOP_SZ as isize);
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
		let d = (map_sz.h as isize) - (map_screen_sz.h as isize);
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
	
	// for the cursor
	pub fn linear_update(&mut self, coord_set: CoordSet, sign: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, relations: &mut Relations, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			map_sz: MapSz, logs: &mut Vec<Log>, turn: usize, d: &mut DispState) {
		if coord_set == CoordSet::X {
			self.cur.x += sign;
		}else{
			self.cur.y += sign;
		}
		
		self.reset_cur_accel();
		self.reset_unit_subsel();
		self.chk_cursor_bounds(map_data);
		self.update_move_search_ui(map_data, exs, units, bldgs, relations, players, logs, map_sz, turn, d);
	}
	
	// input is in screen coordinates
	pub fn set_text_coord(&mut self, coord: Coord, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>,
			exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, relations: &mut Relations, map_data: &mut MapData<'rt>,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz, logs: &mut Vec<Log>, turn: usize, d: &mut DispState) {
		
		self.cur = coord;
		self.reset_unit_subsel();
		self.chk_cursor_bounds(map_data);
		self.update_move_search_ui(map_data, exs, units, bldgs, relations, players, logs, map_sz, turn, d);
	}
	
	// for the view
	pub fn linear_update_screen(&mut self, coord_set: CoordSet, sign: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz, logs: &mut Vec<Log>, turn: usize, d: &mut DispState) {
		if coord_set == CoordSet::X {
			self.map_loc.x += sign;
		}else{
			self.map_loc.y += sign;
		}
		
		self.reset_cur_accel();
		self.reset_unit_subsel();
		self.chk_cursor_bounds(map_data);
		self.update_move_search_ui(map_data, exs, units, bldgs, relations, players, logs, map_sz, turn, d);
	}
	
	pub fn accel_update(&mut self, coord_set: CoordSet, sign: f32, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz, logs: &mut Vec<Log>, turn: usize, d: &mut DispState) {
		if coord_set == CoordSet::X {
			self.cur.x = ((self.cur.x as f32) + (self.cur_v.x * sign)) as isize;
			self.cur_v.x = if self.cur_v.x > SCROLL_MAX_V {SCROLL_MAX_V}
					else {self.cur_v.x + SCROLL_ACCEL};
				
		}else{
			self.cur.y = ((self.cur.y as f32) + (self.cur_v.y * sign)) as isize;
			self.cur_v.y = if self.cur_v.y > SCROLL_MAX_V {SCROLL_MAX_V}
					else {self.cur_v.y + SCROLL_ACCEL};
		}
		
		self.reset_unit_subsel();
		self.chk_cursor_bounds(map_data);
		self.update_move_search_ui(map_data, exs, units, bldgs, relations, players, logs, map_sz, turn, d);
	}
	
	pub fn accel_update_screen(&mut self, coord_set: CoordSet, sign: f32, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz, logs: &mut Vec<Log>, turn: usize, d: &mut DispState) {
		if coord_set == CoordSet::X {
			self.map_loc.x = ((self.map_loc.x as f32) + (self.map_loc_v.x * sign)) as isize;
			self.map_loc_v.x = if self.map_loc_v.x > SCROLL_MAX_V {SCROLL_MAX_V}
					else {self.map_loc_v.x + SCROLL_ACCEL};
		}else{
			self.map_loc.y = ((self.map_loc.y as f32) + (self.map_loc_v.y * sign)) as isize;
			self.map_loc_v.y = if self.map_loc_v.y > SCROLL_MAX_V {SCROLL_MAX_V}
					else {self.map_loc_v.y + SCROLL_ACCEL};
		}
		
		self.reset_unit_subsel();
		self.chk_cursor_bounds(map_data);
		self.update_move_search_ui(map_data, exs, units, bldgs, relations, players, logs, map_sz, turn, d);
	}

	pub fn chg_zoom(&mut self, inc: isize, map_data: &mut MapData<'rt>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
			units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, bldgs: &mut Vec<Bldg<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
			players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, map_sz: MapSz, logs: &mut Vec<Log>, turn: usize, d: &mut DispState) {
		if ((inc == 1) && (self.zoom_ind == map_data.max_zoom_ind())) ||
			((inc == -1) && (self.zoom_ind - 1) == ZOOM_IND_SUBMAP) { return; }
		
		let zoom_ind_prev = self.zoom_ind;
		if inc == 1 { self.zoom_ind += 1; } else { self.zoom_ind -= 1; }
		
		debug_assertq!(zoom_ind_prev <= map_data.max_zoom_ind());
		debug_assertq!(self.zoom_ind <= map_data.max_zoom_ind());
		
		let dy = (self.cur.y as f32) - (MAP_ROW_START as f32);
		let dx = self.cur.x as f32;
		
		let h_full = map_data.map_szs[ZOOM_IND_ROOT].h as f32;
		let w_full = map_data.map_szs[ZOOM_IND_ROOT].w as f32;
		
		let frac_prev_h = h_full / map_data.map_szs[zoom_ind_prev].h as f32;
		let frac_prev_w = w_full / map_data.map_szs[zoom_ind_prev].w as f32;
		
		let frac_h = h_full / map_data.map_szs[self.zoom_ind].h as f32;
		let frac_w = w_full / map_data.map_szs[self.zoom_ind].w as f32;
		
		self.map_loc.y = (((frac_prev_h*(self.map_loc.y as f32)) + ((frac_prev_h - frac_h)*dy))/frac_h).round() as isize;
		self.map_loc.x = (((frac_prev_w*(self.map_loc.x as f32)) + ((frac_prev_w - frac_w)*dx))/frac_w).round() as isize;
		self.start_map_drag = None;
		
		self.chk_cursor_bounds(map_data);
		if self.zoom_ind == map_data.max_zoom_ind() {
			self.update_move_search_ui(map_data, exs, units, bldgs, relations, players, logs, map_sz, turn, d);
		}
	}
}

fn plot_unit(unit_template: &UnitTemplate, player: &Player, d: &mut DispState){
	if unit_template.nm[0] != RIOTER_NM {
		set_player_color(player, true, d);
	}else{d.attron(COLOR_PAIR(CWHITE));}
	
	d.addch(unit_template.char_disp as chtype);
	
	if unit_template.nm[0] != RIOTER_NM {
		set_player_color(player, false, d);
	}else{d.attroff(COLOR_PAIR(CWHITE));}
}

impl DispChars {
	pub fn convert_to_line(&self, pchar: char) -> chtype {
		if pchar == (self.urcorner_char as u8 as char) {
			self.urcorner_char as chtype
		}else if pchar == (self.ulcorner_char as u8 as char) {
			self.ulcorner_char as chtype
		}else if pchar == (self.lrcorner_char as u8 as char) {
			self.lrcorner_char as chtype
		}else if pchar == (self.llcorner_char as u8 as char) {
			self.llcorner_char as chtype
		}else if pchar == (self.hline_char as u8 as char) {
			self.hline_char as chtype
		}else if pchar == (self.vline_char as u8 as char) {
			self.vline_char as chtype
		}else{
			pchar as chtype
		}
	}
}

// c_plot, c_bldg are the coordinates to plot and of the bldg
pub fn print_bldg_char(mut c_plot: Coord, mut c_bldg: Coord, bt: &BldgTemplate,
		fire: &Option<Fire>, disp_chars: &DispChars, map_sz: MapSz, d: &mut DispState){
	let h = bt.sz.h as isize;
	let w = bt.sz.w as isize;
	
	// wrap
	if (c_plot.x - c_bldg.x) >= w {
		c_bldg.x += map_sz.w as isize;
	}else if c_plot.x < c_bldg.x {
		c_plot.x += map_sz.w as isize;
	}
	
	let ind = ((c_plot.y - c_bldg.y)*w + (c_plot.x - c_bldg.x)) as usize;
	debug_assertq!(ind < (h*w) as usize);
	
	// print fire
	if let Some(fire) = &fire {
		match fire.layout[ind] {
			FireTile::Smoke => {
				d.attron(COLOR_PAIR(CGRAY));
				d.addch('#');
				d.attroff(COLOR_PAIR(CGRAY));
				return;
			}
			FireTile::Fire {color} => {
				d.attron(COLOR_PAIR(white_fg(color)));
				d.addch(' ');
				d.attroff(COLOR_PAIR(white_fg(color)));
				return;
			}
			FireTile::None => {}
		}
	}
	
	// print regular building character
	let pchar = if let Some(pchar) = bt.print_str.chars().nth(ind) {
		pchar
	}else{panicq!("Failed to parse: {}, character {}", bt.print_str, ind);};
	
	//let at_edge = (c_plot.y == c_bldg.y) || (c_plot.x == c_bldg.x) ||
	//	(c_plot.y - c_bldg.y) == (h-1) || (c_plot.x - c_bldg.x) == (w-1);
	
	//d.addch(if at_edge {disp_chars.convert_to_line(pchar)} else{pchar as chtype});

	
	d.addch(disp_chars.convert_to_line(pchar));	
}

pub fn ret_bldg_color(at_edge: bool, bldg_ind: usize, b: &Bldg, bldgs: &Vec<Bldg>, player: &Player,
		iface_settings: &IfaceSettings, map_data: &MapData, exf: &HashedMapEx, map_sz: MapSz) -> chtype {
	if b.construction_done != None && !at_edge {
		return COLOR_PAIR(CYELLOW);
	}
	
	// show unconnected bldgs
	if iface_settings.show_unconnected_bldgs && iface_settings.cur_player == b.owner_id {
		// only show in unconnected color if not a city hall
		if let BldgArgs::CityHall {..} = &b.args {} else {
			if let Some(zone_ex) = player.zone_exs.get(&return_zone_coord(b.coord, map_sz)) {
				match zone_ex.ret_city_hall_dist() {
					Dist::NotInit | Dist::NotPossible {..} => {
						return COLOR_PAIR(CDARKRED);
					}
					Dist::Is {..} | Dist::ForceRecompute {..}=> {}
				}
			}
		}
	}
	
	// show unoccupied bldgs
	if iface_settings.show_unoccupied_bldgs {
		if let BldgType::Taxable(_) = b.template.bldg_type {
			if b.n_residents() == 0 {
				return COLOR_PAIR(CDARKGRAY);
			}
		}
	}
	
	// connected to bldg selected by cursor?
	if let Some(bldg_ind_sel) = iface_settings.bldg_ind_frm_cursor(bldgs, map_data, exf) {
		return match bldgs[bldg_ind_sel].connected(bldg_ind) {
			CommuteType::To => {COLOR_PAIR(CGREENWHITE)}
			CommuteType::Frm => {COLOR_PAIR(CBLUEWHITE)}
			CommuteType::None => {COLOR_PAIR(player.personalization.color)}
		};
	}
	COLOR_PAIR(player.personalization.color)
}

// plot_coord is in map coordinates, not screen coordinates
fn plot_bldg(plot_coord: u64, bldgs: &Vec<Bldg>, ex: &MapEx, players: &Vec<Player>, map_sz: MapSz,
		disp_chars: &DispChars, fog: &FogVars, iface_settings: &IfaceSettings, map_data: &MapData, exf: &HashedMapEx, d: &mut DispState){
	
	debug_assertq!(ex.bldg_ind.is_none() != fog.max_bldg_template.is_none());
	
	// at most zoomed in lvl, show actual bldg
	if let Some(bldg_ind) = ex.bldg_ind {
		let b = &bldgs[bldg_ind];

		let c_bldg = Coord::frm_ind(b.coord, map_sz);
		let c_plot = Coord::frm_ind(plot_coord, map_sz);

		let h = b.template.sz.h as isize;
		let w = b.template.sz.w as isize;

		let at_edge = (c_plot.y == c_bldg.y) || (c_plot.x == c_bldg.x) ||
			(c_plot.y - c_bldg.y) == (h-1) || (c_plot.x - c_bldg.x) == (w-1);
		
		let bldg_color = ret_bldg_color(at_edge, bldg_ind, b, bldgs, &players[b.owner_id as usize],
				iface_settings, map_data, exf, map_sz);
		
		d.attron(bldg_color);
		print_bldg_char(c_plot, c_bldg, b.template, &b.fire, disp_chars, map_sz, d);
		d.attroff(bldg_color);
	// zoomed out
	}else{
		let o = &players[fog.owner_id.unwrap() as usize];
		set_player_color(o, true, d);
		d.addch(fog.max_bldg_template.unwrap().plot_zoomed as chtype);
		set_player_color(o, false, d);
	}
}

pub fn plot_zone(zone: ZoneType, disp_chars: &DispChars, d: &mut DispState){
	d.addch(disp_chars.land_char as chtype | COLOR_PAIR(zone.to_color()));
}

pub const ROAD_CHAR: chtype = ' ' as chtype;
pub const GATE_CHAR: chtype = '=' as chtype;

impl IfaceSettings<'_,'_,'_,'_,'_> {
	// `coord` is in map coordinates, not screen coordinates
	pub fn plot_land(&self, zoom_ind: usize, coord: u64, map_data: &mut MapData, units: &Vec<Unit>, bldgs: &Vec<Bldg>,
			exs: &Vec<HashedMapEx>, players: &Vec<Player>, 
			disp_chars: &DispChars, sel: bool, alt_ind: usize, d: &mut DispState) {
		
		let map_sz = map_data.map_szs[zoom_ind];
		let mfc = map_data.get(ZoomInd::Val(zoom_ind), coord);
		let ex_wrapped = exs[zoom_ind].get(&coord);
		let pstats = &players[self.cur_player as usize].stats;
		
		let land_discovered = pstats.land_discov[zoom_ind].map_coord_ind_discovered(coord);
		
		debug_assertq!(coord < (map_sz.h*map_sz.w) as u64);
		
		let land_color = mfc.land_color(self.underlay, sel);
		
		// get resource if present, and set alt_ind_offset to 1 if present, so unit display alternation
		// will include the resource in the display flashing
		let mut get_resource = || {
			if zoom_ind == map_data.max_zoom_ind() && self.show_resources && (!self.show_fog || land_discovered){
				if let Some(resource) = mfc.get_resource(coord, map_data, map_sz) {
					if pstats.resource_discov(resource) { // technologically
						return (1, mfc.resource_char(coord, map_data, map_sz, disp_chars));
					} // resource discovered
				} // resource present
			} // not showing resources or not zoomed in
			(0, None)
		};
		
		let (mut alt_ind_offset, resource_char_opt) = get_resource();
		
		macro_rules! show_resource{
			() => {
				if let Some(resource_char) = resource_char_opt {
					if resource_char != ' ' as chtype {
						d.addch(resource_char | COLOR_PAIR(white_fg(land_color)));
						return;
					}else{
						alt_ind_offset = 0;
					}
				}
			};
			($final: expr) => {
				if let Some(resource_char) = resource_char_opt {
					if resource_char != ' ' as chtype {
						d.addch(resource_char | COLOR_PAIR(white_fg(land_color)));
						return;
					}
				}
		}};
		
		// show resource if alt_ind aligns
		if (alt_ind % 3) == 0 {show_resource!();}
		
		// plot unit, bldg, structure, or zone
		if let Some(ex) = ex_wrapped {
			// show unit
			if (!self.show_fog || land_discovered) && self.show_units {
				if let Some(unit_inds) = &ex.unit_inds {
					let sel_show = (alt_ind - alt_ind_offset) % unit_inds.len();
					let u = &units[unit_inds[sel_show]];
					plot_unit(u.template, &players[u.owner_id as usize], d);
					return;
				}
			}
			
			// show zone overlays (ex. demand, happiness, crime), bldg, road/wall, zones
			if let Some(fog) = self.get_fog_or_actual(coord, ex, pstats) {
				// bldg
				if self.show_bldgs && (!fog.max_bldg_template.is_none() || !ex.bldg_ind.is_none()) {
					plot_bldg(coord, bldgs, ex, players, map_sz, disp_chars, &fog, self, map_data, exs.last().unwrap(), d);
					return;
				}
				
				// zone overlays (demand, happiness, or crime)
				if self.zone_overlay_map != ZoneOverlayMap::None {
					if let Some(zt) = ex.actual.ret_zone_type() {
						let owner_id = ex.actual.owner_id.unwrap() as usize;
						if let Some(zone_ex) = players[owner_id].zone_exs.get(&return_zone_coord(coord, map_sz)) {
							const N_STEPS: f32 = 6.; // 9.;
							macro_rules! plot_val{($val: expr, $step: expr, $offset: expr) => {
								let c = 
									/*if $val <= ($step + $offset) {CSAND4} else
									if $val <= (2.*$step + $offset) {CSAND3} else
									if $val <= (3.*$step + $offset) {CSAND2} else
									if $val <= (4.*$step + $offset) {CSAND1} else
									if $val <= (5.*$step + $offset) {CGREEN1} else
									if $val <= (6.*$step + $offset) {CGREEN2} else
									if $val <= (7.*$step + $offset) {CGREEN3} else
									if $val <= (8.*$step + $offset) {CGREEN4} else {CGREEN5};*/
									if $val <= ($step + $offset) {CBLUERED5} else // red
									if $val <= (2.*$step + $offset) {CBLUERED4} else
									if $val <= (3.*$step + $offset) {CBLUERED3} else
									if $val <= (4.*$step + $offset) {CBLUERED2} else
									if $val <= (5.*$step + $offset) {CBLUERED1} else {CBLUERED0}; // blue
								
								d.addch(disp_chars.land_char as chtype | COLOR_PAIR(c));
								return;
							};};
							
							match self.zone_overlay_map {
								ZoneOverlayMap::ZoneDemands => {
									if let Some(zdws) = zone_ex.demand_weighted_sum[zt as usize] {
										let val = zdws + 0.5;
										const ZONE_DEMAND_STEP: f32 = 1.0/N_STEPS;
										
										plot_val!(val, ZONE_DEMAND_STEP, 0.);
									}
								} ZoneOverlayMap::Happiness => {
									let happiness_step = (pstats.locally_logged.happiness_sum/N_STEPS).abs();
									plot_val!(zone_ex.zone_agnostic_stats.locally_logged.happiness_sum, happiness_step, -(pstats.locally_logged.happiness_sum/2.).abs());
								} ZoneOverlayMap::Crime => {
									const CRIME_STEP: f32 = 5./N_STEPS;
									plot_val!(-zone_ex.zone_agnostic_stats.crime_sum, CRIME_STEP, -5.);
								} ZoneOverlayMap::None | ZoneOverlayMap::N => {panicq!("should not be possible");}
							}
						}
					}
				}
				
				// road / wall
				if !sel && self.show_structures && !fog.structure.is_none() {
					if let Some(structure) = fog.structure {
						d.addch (match structure.structure_type {
								StructureType::Road => {ROAD_CHAR
								} StructureType::Gate => {GATE_CHAR
								} StructureType::Wall => {
									let c = Coord::frm_ind(coord, map_sz);
									
									// damaged
									if structure.health < (0.9 * std::u8::MAX as f32).round() as u8 {
										'#' as chtype
									// at full-enough health, display line
									}else{
										let chk_loc = |i,j| -> bool {
											if let Some(coord_chk) = map_sz.coord_wrap(c.y + i, c.x + j) {
												if let Some(ex) = exs[zoom_ind].get(&coord_chk) {
													if let Some(StructureData {structure_type: StructureType::Wall, ..}) = ex.actual.structure {
														return true;
													}
												}
											}
											false
										};
										
										let top = chk_loc(-1,0);
										let bottom = chk_loc(1,0);
										let left = chk_loc(0,-1);
										let right = chk_loc(0,1);
										
										if top && bottom && left && right {
											'+' as chtype
										}else if top && bottom {
											disp_chars.vline_char as chtype
										}else if left && right {
											disp_chars.hline_char as chtype
										}else if top && right {
											disp_chars.llcorner_char as chtype
										}else if top && left {
											disp_chars.lrcorner_char as chtype
										}else if bottom && left {
											disp_chars.urcorner_char as chtype
										}else if bottom && right {
											disp_chars.ulcorner_char as chtype
										}else{
											match structure.orientation {
												'-' => disp_chars.vline_char as chtype,
												'|' => disp_chars.hline_char as chtype,
												_ => (structure.orientation as chtype)
											}
										}
									}
								} StructureType::N => {panicq!("unknown structure");}
						});
						return;
					}
				}
				
				// zones
				if !sel && self.show_zones {
					if let Some(zt) = fog.ret_zone_type() {
						plot_zone(zt, disp_chars, d);
						return;
					}
				} // zns
			}
		}
		
		// plot resource if unit, bldg, structure or zone was not plotted
		show_resource!(true);
		
		// plot land or sector
		if !self.show_fog || land_discovered {
			if self.show_sectors && self.zoom_ind == map_data.max_zoom_ind() && !pstats.sector_nm_frm_coord(coord, map_sz).is_none() {
				d.addch('.' as chtype | COLOR_PAIR(white_fg(land_color)));
				return;
			}
			match mfc.map_type {
			   MapType::Land | MapType::ShallowWater | MapType::DeepWater => {
				   	d.addch(' ' as chtype | COLOR_PAIR(white_fg(land_color))); //land_char_use
			   } MapType::Mountain => {
			   		d.addch('^' as chtype);
			   } MapType::N => {
			   		panicq!("unknown map type");
			   	}
			}
		// show fog
		}else{
			d.addch(' ' as chtype);
		}
	}
}

impl <'rt>Map<'rt> {
	pub fn land_color(&self, underlay: Underlay, sel: bool) -> CInt {
		// space selected (sub-map)
		if sel {
			match self.map_type {
			   MapType::Land => {
				match underlay {
				   Underlay::Arability => {
					ArabilityType::frm_arability(self.arability, self.map_type, self.show_snow).to_color(sel)
				   } Underlay::Elevation => {
					let elevation = self.elevation as isize;
					
					if elevation <= ELEVATION_STEP {CREDGREEN5} else
					if elevation <= (2*ELEVATION_STEP) {CREDGREEN4} else
					if elevation <= (3*ELEVATION_STEP) {CREDGREEN3} else
					if elevation <= (4*ELEVATION_STEP) {CREDGREEN2} else
					if elevation <= (5*ELEVATION_STEP) {CREDGREEN1} else
					if elevation <= (6*ELEVATION_STEP) {CREDSAND1} else
					if elevation <= (7*ELEVATION_STEP) {CREDSAND2} else
					if elevation <= (8*ELEVATION_STEP) {CREDSAND3} else {CREDSAND4}
				   } Underlay::WaterMountains => {CBLACK
				   } Underlay::N => {panicq!("invalid underlay")}}
			   } MapType::Mountain => {CREDGRAY
			   } MapType::ShallowWater => {CREDSHALLOW_WATER
			   } MapType::DeepWater => {CREDDEEP_WATER
			   } MapType::N => {panicq!("unknown map type")}
			}
			
		//////////////////////////////// not selected
		}else{
			match self.map_type {
			   MapType::Land => {
				match underlay {
				   Underlay::Arability => {
					ArabilityType::frm_arability(self.arability, self.map_type, self.show_snow).to_color(sel)
				   } Underlay::Elevation => {
					let elevation = self.elevation as isize;
					
					if elevation <= ELEVATION_STEP {CGREEN5} else
					if elevation <= (2*ELEVATION_STEP) {CGREEN4} else
					if elevation <= (3*ELEVATION_STEP) {CGREEN3} else
					if elevation <= (4*ELEVATION_STEP) {CGREEN2} else
					if elevation <= (5*ELEVATION_STEP) {CGREEN1} else
					if elevation <= (6*ELEVATION_STEP) {CSAND1} else
					if elevation <= (7*ELEVATION_STEP) {CSAND2} else
					if elevation <= (8*ELEVATION_STEP) {CSAND3} else {CSAND4}
					
				   } Underlay::WaterMountains => {CBLACK
				   } Underlay::N => {panicq!("invalid underlay")}}
			   } MapType::Mountain => {CWHITE
			   } MapType::DeepWater => {CDEEP_WATER
			   } MapType::ShallowWater => {CSHALLOW_WATER
			   } MapType::N => {panicq!("unknown map type")}
			}
		}
	}
}

pub fn shortcut_indicator() -> chtype {
	COLOR_PAIR(CCYAN) | A_UNDERLINE()
}

pub fn shortcut_or_default_show(ui_mode: &UIMode, def_color: i32) -> chtype {
	match *ui_mode {
		UIMode::TextTab {..} | UIMode::None | UIMode::SetTaxes(_) => {shortcut_indicator()},
		_ => {COLOR_PAIR(def_color)}
	}
}

pub fn shortcut_show(ui_mode: &UIMode) -> chtype {
	shortcut_or_default_show(ui_mode, CWHITE)
}

// red to green gradient for example hp
const PERC_STEP: f32 = 100./5.;

pub fn colorize(perc: f32, on: bool, d: &mut DispState){
	let v = if perc <= PERC_STEP {CRED
		}else if perc <= (2.*PERC_STEP) {CREDGREEN4
		}else if perc <= (3.*PERC_STEP) {CREDGREEN3
		}else if perc <= (4.*PERC_STEP) {CREDGREEN2
		}else {CGREEN1};
	
	let v = COLOR_PAIR(v);
	if on {d.attron(v);} else {d.attroff(v);}
}

// add commas to number. ex: convert 12345 to 12,345
/*pub fn num_format<T: std::fmt::Display>(n: T) -> String {
	let mut n_str = format!("{}", n);
	
	// number needs commas inserted
	if n_str.len() > 3 {
		// start from back and work to higher decimal places
		let mut insert_pos = n_str.len() - 3;
		loop {
			n_str.insert(insert_pos, ',');
			if insert_pos <= 3 {break;}
			insert_pos -= 3;
		}
	}

	n_str
}*/

pub fn float_string(gold: f32) -> String {
	if gold < 1000. {
		format!("{:.1}", gold)
	}else if gold < 1_000_000. {
		format!("{:.1}K", gold/1000.)
	}else if gold < 1_000_000_000. {
		format!("{:.1}M", gold/1_000_000.)
	}else if gold < 1_000_000_000_000. {
		format!("{:.1}B", gold/1_000_000_000.)
	}else if gold < 1_000_000_000_000_000. {
		format!("{:.1}T", gold/1_000_000_000_000.)
	}else{
		format!("{:.1}Q", gold/1_000_000_000_000_000.)
	}
}

pub const TURNS_PER_YEAR: usize = 360;

impl Localization {
	fn month_str(&self, month: usize) -> String {
		match month {
			0 => self.Jan.clone(),
			1 => self.Feb.clone(),
			2 => self.Mar.clone(),
			3 => self.Apr.clone(),
			4 => self.May.clone(),
			5 => self.Jun.clone(),
			6 => self.Jul.clone(),
			7 => self.Aug.clone(),
			8 => self.Sep.clone(),
			9 => self.Oct.clone(),
			10 => self.Nov.clone(),
			11 => self.Dec.clone(),
			_ => {panicq!("Invalid month {}", month);}
		}
	}
	
	pub fn date_str_underscores(&self, turn: usize) -> String {
		let year = turn / TURNS_PER_YEAR;
		let remainder = turn % TURNS_PER_YEAR;
		let month = remainder / 30;
		let day = (remainder % 30) + 1;
		format!("{}_{}_{}", self.month_str(month), day, year)
	}
	
	pub fn date_str(&self, turn: usize) -> String {
		let year = turn / TURNS_PER_YEAR;
		let remainder = turn % TURNS_PER_YEAR;
		let month = remainder / 30;
		let day = (remainder % 30) + 1;
		format!("{} {}, {}", self.month_str(month), day, year)
	}
	
	pub fn print_date_log(&self, turn: usize, d: &mut DispState) {
		d.attron(COLOR_PAIR(CYELLOW));
		d.addstr(&self.date_str(turn));
		d.attroff(COLOR_PAIR(CYELLOW));
		
		d.addstr(": ");
	}
	
	pub fn date_interval_str(&self, turn_duration: f32) -> String {
		if turn_duration <= 1.1 {
			format!("1 {}", self.day)
		}else if turn_duration < (2.*12.*30.) {
			format!("{} {}", turn_duration.round() as usize, self.days)
		}else if turn_duration < (4.*12.*30.) {
			format!("{:.1} {}", turn_duration / 30., self.mons)
		}else{
			format!("{:.1} {}", turn_duration / (12.*30.), self.yrs)
		}
	}
}

pub fn print_civ_nm (player: &Player, d: &mut DispState) {
	set_player_color(player, true, d);
	d.addstr(&player.personalization.nm);
	set_player_color(player, false, d);
}

pub enum Printable { FileNm, Numeric, Coordinate }

pub fn is_printable(c: char, printable_type: Printable) -> bool {
	match printable_type {
		Printable::FileNm => {
			const PRINTABLE: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890!@#$%^&*()-=_+[{]}|;:'\",<.>? ";
			PRINTABLE.contains(c)
		} Printable::Numeric => {
			const PRINTABLE: &str = "01234567890,.-";
			PRINTABLE.contains(c)
		} Printable::Coordinate => {
			const PRINTABLE: &str = "01234567890, ";
			PRINTABLE.contains(c)
		}
	}
}

impl IfaceSettings<'_,'_,'_,'_,'_> {
	pub fn update_cursor(&self, pstats: &Stats, map_data: &mut MapData, disp_chars: &DispChars,
			txt_list: &TxtList, d: &mut DispState) {
		
		if screen_reader_mode() {
			d.curs_set(CURSOR_VISIBILITY::CURSOR_VERY_VISIBLE);
			
			if let UIMode::Menu {sel_loc, ..} = &self.ui_mode {
				d.mv(sel_loc.0, sel_loc.1);
				return;
			}else if let UIMode::TextTab {mode, loc} = &self.ui_mode {
				match loc {
					TextTabLoc::BottomStats => {
						if let Some(loc) = txt_list.bottom.get(*mode) {
							d.mv(loc.0, loc.1);
						}
					}
					TextTabLoc::RightSide => {
						if let Some(loc) = txt_list.right.get(*mode) {
							d.mv(loc.0, loc.1);
						}
					}
				}
				return;
			}
		}
		
		match &self.ui_mode {
			// these windows need to show the cursor in another location (input)
			UIMode::ContactEmbassyWindow{..} | UIMode::SaveAsWindow {..} |
			UIMode::SaveAutoFreqWindow {..} | UIMode::GoToCoordinateWindow {..} |
			UIMode::CreateSectorAutomation {sector_nm: Some(_), unit_enter_action: Some(_),
				idle_action: Some(_), ..} | 
			UIMode::GetTextWindow {..} => {}
			
			// show cursor at selection location (on map) -- cursor is hidden for options != None
			UIMode::None |
			UIMode::TextTab {..} |
			UIMode::SetTaxes(_) |
			UIMode::Menu {..} |
			UIMode::ProdListWindow {..} |
			UIMode::CurrentBldgProd {..} |
			UIMode::SelectBldgDoctrine {..} |
			UIMode::CitizenDemandAlert {..} |
			UIMode::PublicPollingWindow |
			UIMode::SelectExploreType {..} |
			UIMode::OpenWindow {..} |
			UIMode::TechWindow {..} |
			UIMode::CreateSectorAutomation {..} |
			UIMode::DoctrineWindow {..} |
			UIMode::PlotWindow {..} |
			UIMode::UnitsWindow {..} |
			UIMode::GenericAlert {..} |
			UIMode::NoblePedigree {..} |
			UIMode::NobilityReqToJoin {..} |
			
			UIMode::BrigadesWindow {..} |
			UIMode::BrigadeBuildList {..} |
			
			UIMode::SectorsWindow {..} |
			UIMode::MilitaryBldgsWindow {..} |
			UIMode::ImprovementBldgsWindow {..} |
			UIMode::CitiesWindow {..} |
			//UIMode::ContactEmbassyWindow {..} |
			UIMode::CivilizationIntelWindow {..} |
			UIMode::SwitchToPlayerWindow {..} |
			UIMode::SetDifficultyWindow {..} |
			UIMode::RiotingAlert {..} |
			UIMode::TechDiscoveredWindow {..} |
			UIMode::DiscoverTechWindow {..} |
			UIMode::ObtainResourceWindow {..} |
			UIMode::PlaceUnitWindow {..} |
			UIMode::ResourcesAvailableWindow {..} |
			UIMode::ResourcesDiscoveredWindow {..} |
			UIMode::WorldHistoryWindow {..} |
			UIMode::BattleHistoryWindow {..} |
			UIMode::EconomicHistoryWindow {..} |
			UIMode::WarStatusWindow {..} |
			UIMode::EncyclopediaWindow {..} |
			UIMode::InitialGameWindow |
			UIMode::EndGameWindow |
			UIMode::UnmovedUnitsNotification |
			UIMode::PrevailingDoctrineChangedWindow |
			UIMode::MvWithCursorNoActionsRemainAlert {..} |
			UIMode::CivicAdvisorsWindow |
			UIMode::ForeignUnitInSectorAlert {..} |
			UIMode::AboutWindow => {
				macro_rules! mv_to_cur {()=>(d.mv(self.cur.y as i32, self.cur.x as i32););}
				mv_to_cur!();
				
				let cursor_coord = self.cursor_to_map_ind(map_data);
				let char_shown = d.inch() & A_CHARTEXT();
				
				//  make cursor black (because it would otherwise not be visible)
				//	if cursor is on land & discovered/no fog & on tundra or pine forest
				if (char_shown == ' ' as chtype || char_shown == (disp_chars.land_char as chtype & A_CHARTEXT()) || char_shown == ('^' as chtype)) && 
						self.ui_mode.is_none() && 
						(!self.show_fog || pstats.land_discov[self.zoom_ind].map_coord_ind_discovered(cursor_coord)) {
					
					let mfc = map_data.get(ZoomInd::Val(self.zoom_ind), cursor_coord);
					if let ArabilityType::Tundra | ArabilityType::PineForest =
							ArabilityType::frm_arability(mfc.arability, mfc.map_type, mfc.show_snow) {
						d.attron(COLOR_PAIR(CBLACK));
						d.addch(' ' as chtype);
						d.attroff(COLOR_PAIR(CBLACK));
						mv_to_cur!();
					}else{
						d.addch(' ' as chtype);
						mv_to_cur!();
					}
		}}}
	}
	
	// purpose: if player has any units or bldgs nearby, return actual map information
	//
	// if `show_fog`:
	//		land discovered:
	//			if fog entry present: return fog
	//			fog entry not present: return `actual` from `ex`
	//		land undiscovered:
	//			None
	//
	// `show_fog` = false: returns `actual` variables from `ex`
	pub fn get_fog_or_actual<'r,'bt,'ut,'rt,'dt>(&self, coord: u64, ex: &'r MapEx<'bt,'ut,'rt,'dt>,
				pstats: &'r Stats<'bt,'ut,'rt,'dt>) -> Option<&'r FogVars<'bt,'ut,'rt,'dt>> {
		if self.show_fog {
			return if pstats.land_discov[self.zoom_ind].map_coord_ind_discovered(coord) {
				// unit is not present, show what was last visible
				Some(if let Some(fog) = pstats.fog[self.zoom_ind].get(&coord) {
					fog
				// unit is present, show `actual` ex
				}else{
					&ex.actual
				})
			}else{
				None
			};
		}
		
		Some(&ex.actual)
	}
	
	// used for updating menu indicators (none when current player is not an AI)
	pub fn cur_player_paused(&self, players: &Vec<Player>) -> Option<bool> {
		if let Some(player) = players.get(self.cur_player as usize) {
			if let PlayerType::AI {ai_state, ..} = &player.ptype {
				return Some(ai_state.paused);
			}
		}
		None
	}
}

use std::f32::consts::PI;
use crate::movement::manhattan_dist;
pub fn direction_string(cur_coord: Coord, test_coord: Coord, map_sz: MapSz) -> String {
	let direction = {
		// tan(theta) = dy/dx
		
		let dy = -(test_coord.y - cur_coord.y); // up is positive -- cartesian
		let dx = (test_coord.x - cur_coord.x) / 2; // right is positive -- compress
		
		if dx == 0 || dy == 0 {
			""
		}else{
			let angle = (dy as f32 / dx as f32).atan();
			
			// quad 1
			if dy > 0 && dx > 0 {
				// should range from 0:pi/2
				if angle <= (PI/8.) {
					" E"
				}else if angle <= (3.*PI/8.) {
					" NE"
				}else{
					" N"
				}
			// quad 2
			}else if dy > 0 && dx < 0 {
				// should range from 0:-pi/2
				if angle >= -(PI/8.) {
					" W"
				}else if angle >= -(3.*PI/8.) {
					" NW"
				}else{
					" N"
				}
			// quad 3
			}else if dx < 0 && dx < 0 {
				// should range from 0:pi/2
				if angle <= (PI/8.) {
					" W"
				}else if angle <= (3.*PI/8.) {
					" SW"
				}else{
					" S"
				}
			// quad 4
			}else{
				// should range from 0:-pi/2
				if angle >= -(PI/8.) {
					" E"
				}else if angle >= -(3.*PI/8.) {
					" SE"
				}else{
					" S"
				}
			}
		}
	};
	
	let dist = (METERS_PER_TILE as usize) * manhattan_dist(cur_coord, test_coord, map_sz);
	if dist < 1000 {
		format!("{} m{}", dist, direction)
	}else{
		format!("{:.1} km{}", dist as f32 / 1000., direction)
	}
}

pub fn addstr_c(txt: &str, color: i32, d: &mut DispState) {
	d.attron(COLOR_PAIR(color));
	d.addstr(txt);
	d.attroff(COLOR_PAIR(color));
}

pub fn addstr_attr(txt: &str, attr: chtype, d: &mut DispState) {
	d.attron(attr);
	d.addstr(txt);
	d.attroff(attr);
}

//////////////////////////////////////////
// keyboard shortcut printing
use crate::config_load::UNSET_KEY;

// note: iface_settings.print_key() should match this...
pub fn key_txt(key: i32, l: &Localization) -> String {
	if key == KEY_ESC {String::from("<Esc>")
	}else if key == KEY_DC {String::from("<Del>")
	}else if key == '\t' as i32 {String::from("<Tab>")
	}else if key == '\n' as i32 {l.Enter_key.clone()
	}else if key == UNSET_KEY {String::from("Unset key")
	// regular letter key
	}else{
		let k = key as u8 as char;
		if k.is_ascii_uppercase() {
			format!("Shft {}", k)
		}else{
			format!("{}", k)
		}
	}
}

pub fn print_key_always_active(key: i32, l: &Localization, d: &mut DispState) {
	if key == KEY_ESC {addstr_c("<Esc>", ESC_COLOR, d);
	}else if key == KEY_DC {addstr_c("<Del>", ESC_COLOR, d);
	}else if key == '\t' as i32 {addstr_c("<Tab>", ESC_COLOR, d);
	}else if key == '\n' as i32 {addstr_c(&l.Enter_key, ESC_COLOR, d);
	}else if key == UNSET_KEY {d.addstr("Unset key");
	
	// regular ASCII key
	}else{
		if (key as u8 as char).is_ascii_uppercase() {
			addstr_attr("Shft ", shortcut_indicator(), d);
		}
		d.addch(key as chtype | shortcut_indicator());
	}
}

impl IfaceSettings<'_,'_,'_,'_,'_> {
	// note: key_txt() should match this...
	pub fn print_key(&self, key: i32, l: &Localization, d: &mut DispState) {
		if key == KEY_ESC {addstr_c("<Esc>", ESC_COLOR, d);
		}else if key == KEY_DC {addstr_c("<Del>", ESC_COLOR, d);
		}else if key == '\t' as i32 {addstr_c("<Tab>", ESC_COLOR, d);
		}else if key == '\n' as i32 {addstr_c(&l.Enter_key, ESC_COLOR, d);
		}else if key == UNSET_KEY {d.addstr("Unset key");
		
		// regular letter key
		}else{
			if (key as u8 as char).is_ascii_uppercase() {
				d.attron(shortcut_show(&self.ui_mode));
				d.addstr("Shft ");
				d.attroff(shortcut_show(&self.ui_mode));
			}
			d.addch(key as chtype | shortcut_show(&self.ui_mode));
		}
	}
}

pub fn crop_txt(txt: &str, len: usize) -> String {
	if txt.len() <= len {return txt.to_string();}
	let mut cropped = String::with_capacity(len);
	for c in txt.chars().take(len-3) {
		cropped.push(c);
	}
	cropped.push_str("...");
	cropped
}

pub fn wrap_txt(mut txt: &str, w: usize) -> Vec<&str> {
	let mut lines = Vec::new();
	
	loop {
		if txt.len() > w {
			let space_inds = {
				let mut space_inds = Vec::with_capacity(txt.len());
				for (ind, _v) in txt.chars().enumerate().filter(|(_,v)| *v == ' ') {
					space_inds.push(ind);
				}
				space_inds
			};
			
			if let Some(wrap_ind) = space_inds.iter().rev().
					find(|&&ind| ind < w) {
				let (before, after) = txt.split_at(*wrap_ind);
				lines.push(before);
				txt = after;
			}else{panicq!("could not wrap text: {}", txt);}
		}else{
			lines.push(txt);
			return lines;
		}
	}
}

impl PersonName {
	pub fn txt(&self) -> String {
		format!("{} {}", self.first, self.last)
	}
}

