use std::fmt;
use std::cmp::min;
use std::time::Instant;
use crate::disp::SCROLL_ACCEL_INIT;
use crate::map::*;
use crate::saving::*;
use crate::gcore::hashing::HashedMapEx;
use crate::gcore::Relations;
use super::{TreeSelMv};
use crate::resources::ResourceTemplate;
use crate::tech::TechWindowState;
use crate::doctrine::DoctrineWindowState;
use crate::doctrine::DoctrineTemplate;
use crate::movement::{Dest, MvVarsAtZoom};
use crate::renderer::*;
//use super::num_format;

enum_From!{Underlay {Arability, Elevation, WaterMountains}}

#[derive(Clone)]
pub struct ActionInterfaceMeta<'f,'bt,'ut,'rt,'dt> {
	pub action: ActionMeta<'bt,'ut,'rt,'dt>,
	pub unit_ind: Option<usize>, // can be none for civil moves
	pub max_search_depth: usize,
	pub start_coord: Coord,
	pub movement_type: MovementType,
	
	pub movable_to: &'f dyn Fn(u64, u64, &Map, &HashedMapEx, MvVarsAtZoom, &Vec<Bldg>, &Dest, MovementType) -> bool
}

enum_From!{AutoTurn {Off, On, FinishAllActions}}
enum_From!{ZoneOverlayMap {None, ZoneDemands, Happiness, Crime}}

pub enum AddActionTo<'f,'bt,'ut,'rt,'dt> {
	None,
	/////////////////////////////////////////////////
	// no paths are computed (no particular unit/source is specified)
	NoUnit {
		action: ActionMeta<'bt,'ut,'rt,'dt>
	},
	// ^ for creating sectors and groups
	
	BrigadeBuildList {
		brigade_nm: String,
		action: Option<ActionMeta<'bt,'ut,'rt,'dt>>
			// ^ can be none if the player has not indicated which action to perform yet, but they entered
			//	the mode to add the action to the brigade build list
	},
	// ^ assigned to workers
	
	////////////////////////////////////////////////////
	// paths are computed when cursor moves:
	
	IndividualUnit {
		action_iface: ActionInterfaceMeta<'f,'bt,'ut,'rt,'dt>
	},
	// ^ for giving one action to an individual unit or for doing things like
	//   creating a brigade or sector, where no unit in particular is being targeted
	
	AllInBrigade {
		brigade_nm: String,
		action_ifaces: Option<Vec<ActionInterfaceMeta<'f,'bt,'ut,'rt,'dt>>>
			// ^ can be none if the player has not indicated which action to perform yet, but they entered
			//	the mode to assign the action to all units in the brigade
	}
	// ^ example is movement for all in the brigade
}

impl <'f,'bt,'ut,'rt,'dt>AddActionTo<'f,'bt,'ut,'rt,'dt> {
	pub fn is_none(&self) -> bool {
		match self {
			AddActionTo::None => {true}
			AddActionTo::NoUnit {..} |
			AddActionTo::BrigadeBuildList {..} |
			AddActionTo::IndividualUnit {..} |
			AddActionTo::AllInBrigade {..} => {false}
		}
	}
	
	pub fn first_action(&self) -> Option<&ActionMeta<'bt,'ut,'rt,'dt>> {
		match self {
			AddActionTo::None => None,
			AddActionTo::NoUnit {action} => {Some(action)}
			AddActionTo::BrigadeBuildList {action, ..} => {
				action.as_ref()
			
			} AddActionTo::IndividualUnit {action_iface} => {
				Some(&action_iface.action)
			} AddActionTo::AllInBrigade {action_ifaces: Some(action_ifaces), ..} => {
				if let Some(action_iface) = action_ifaces.first() {
					Some(&action_iface.action)
				}else{
					None
				}
			} AddActionTo::AllInBrigade {action_ifaces: None, ..} => None
		}
	}
	
	pub fn first_action_mut(&mut self) -> Option<&mut ActionMeta<'bt,'ut,'rt,'dt>> {
		match self {
			AddActionTo::None => None,
			AddActionTo::NoUnit {action} => {Some(action)}
			AddActionTo::BrigadeBuildList {action, ..} => {
				action.as_mut()
			
			} AddActionTo::IndividualUnit {action_iface} => {
				Some(&mut action_iface.action)
			} AddActionTo::AllInBrigade {action_ifaces: Some(action_ifaces), ..} => {
				if let Some(action_iface) = action_ifaces.first_mut() {
					Some(&mut action_iface.action)
				}else{
					None
				}
			} AddActionTo::AllInBrigade {action_ifaces: None, ..} => None
		}
	}
	
	pub fn brigade_sel_nm(&self) -> Option<&String> {
		match self {
			AddActionTo::None |
			AddActionTo::NoUnit {..} |
			AddActionTo::IndividualUnit {..} => {
				None
			}
			AddActionTo::BrigadeBuildList {brigade_nm, ..} |
			AddActionTo::AllInBrigade {brigade_nm, ..} => {
				Some(brigade_nm)
			}
		}
	}
	
	pub fn actions(&self) -> Vec<&ActionMeta<'bt,'ut,'rt,'dt>> {
		match self {
			AddActionTo::None => {Vec::new()}
			AddActionTo::NoUnit {action} => {vec![action]}
			AddActionTo::BrigadeBuildList {action, ..} => {
				if let Some(action) = action {
					vec![action]
				}else{Vec::new()}
			}
			AddActionTo::IndividualUnit {action_iface} => {
				vec![&action_iface.action]
			}
			AddActionTo::AllInBrigade {action_ifaces, ..} => {
				if let Some(action_ifaces) = action_ifaces {
					let mut actions = Vec::with_capacity(action_ifaces.len());
					for action_iface in action_ifaces.iter() {
						actions.push(&action_iface.action);
					}
					actions
				}else{Vec::new()}
			}
		}
	}
	
	// `Some` value when the move mode is in AddActionTo::BrigadeBuildList or AddActionTo::IndividualUnit
	// returns None otherwise (including when the movement mode is 
	pub fn build_action(&self) -> Option<&ActionMeta<'bt,'ut,'rt,'dt>> {
		match self {
			AddActionTo::None | AddActionTo::NoUnit {..} | AddActionTo::AllInBrigade {..} => None,
			AddActionTo::BrigadeBuildList {action, ..} => {
				action.as_ref()
			} AddActionTo::IndividualUnit {action_iface, ..} => {
				Some(&action_iface.action)
			}
		}
	}
	
	pub fn is_build_list(&self) -> bool {
		match self {
			AddActionTo::BrigadeBuildList {..} => true,
			AddActionTo::None | AddActionTo::NoUnit {..} |
			AddActionTo::IndividualUnit {..} | AddActionTo::AllInBrigade {..} => false
		}
	}
	
	pub fn is_individual_unit(&self) -> bool {
		match self {
			AddActionTo::IndividualUnit {..} => true,
			AddActionTo::None | AddActionTo::NoUnit {..} |
			AddActionTo::BrigadeBuildList {..} | AddActionTo::AllInBrigade {..} => false
		}
	}
	
	// i.e., there is an action_iface
	pub fn specific_unit_or_units(&self) -> bool {
		match self {
			AddActionTo::None | AddActionTo::NoUnit {..} |
			AddActionTo::BrigadeBuildList {..} => {false}
			AddActionTo::IndividualUnit {..} | AddActionTo::AllInBrigade {..} => {true}
		}
	}
}

pub enum ShowExpandedSubmap {
	Open,
	Closed(Instant)
}

const DELAY_SUBMAP_OPEN: u64 = 3000; // in milliseconds. delay in re-opening submap
impl ShowExpandedSubmap {
	// only open if it has been closed for longer than DELAY_SUBMAP_OPEN
	pub fn open(&mut self) {
		if let Self::Closed(close_time) = &self {
			if close_time.elapsed().as_millis() as u64 > DELAY_SUBMAP_OPEN {
				*self = Self::Open;
			}
		}
	}
	
	pub fn is_open(&self) -> bool {
		match self {
			Self::Open => true,
			Self::Closed(_) => false
		}
	}
	
	pub fn close(&mut self) {
		*self = Self::Closed(Instant::now());
	}
	
	pub fn toggle_immediately(&mut self) {
		match self {
			Self::Open => {self.close();}
			Self::Closed(_) => {*self = Self::Open;}
		}
	}
}

pub struct IfaceSettings<'f,'bt,'ut,'rt,'dt> {
	// movement, worker zoning, building, ...
	pub add_action_to: AddActionTo<'f,'bt,'ut,'rt,'dt>,
	
	// underlay
	pub underlay: Underlay, // map underlay
	
	// map overlays
	pub show_structures: bool,
	pub show_units: bool,
	pub show_bldgs: bool,
	pub show_zones: bool,
	pub show_resources: bool,
	pub zone_overlay_map: ZoneOverlayMap,
	pub show_unconnected_bldgs: bool,
	pub show_unoccupied_bldgs: bool,
	pub show_sectors: bool,
	pub show_pipes: bool,
	
	pub cur_player: SmSvType,
	
	// display
	pub zoom_ind: usize, // >= N_EXPLICITLY_STORED_ZOOM_LVLS implies we are into maps that are not explicitly stored in memory
	
	pub unit_subsel: usize, // used when more than one unit is on a plot, to denote which one to show or assign actions to
	
	pub show_expanded_submap: ShowExpandedSubmap,
	
	pub map_screen_sz: ScreenSz, // size available to display map
	pub screen_sz: ScreenSz,
	
	pub map_loc: Coord, // in map coordinates (upper left-most part of map shown on the screen)
	pub map_loc_v: VelocCoord,
	
	pub cur: Coord, // in screen coordinates
	pub cur_v: VelocCoord,
	
	pub start_map_drag: Option<Coord>, // screen coordinates of start of drag
	
	pub view_mv_mode: ViewMvMode, // whether asdwx are moving the cursor or the screen view
	
	pub all_player_pieces_mvd: bool,
	
	pub workers_create_city_sectors: bool,
	pub show_fog: bool,
	pub show_actions: bool, // of AI players
	pub show_all_zone_information: bool,
	
	pub auto_turn: AutoTurn, // Off, On, FinishAllActions
	pub interrupt_auto_turn: bool,
	
	pub save_nm: String,
	
	pub checkpoint_freq: SmSvType // in years
}

impl_saving!{IfaceSettings<'f,'bt,'ut,'rt,'dt> {add_action_to, underlay, show_structures, show_units, show_bldgs,
	     show_zones, show_resources, zone_overlay_map, show_unconnected_bldgs, show_unoccupied_bldgs, show_sectors, show_pipes,
	     cur_player, zoom_ind, unit_subsel, show_expanded_submap,
	     map_screen_sz, screen_sz, view_mv_mode,
	     map_loc, map_loc_v, cur, cur_v, start_map_drag, all_player_pieces_mvd, workers_create_city_sectors, show_fog, show_actions,
	     show_all_zone_information, 
	     auto_turn, interrupt_auto_turn,
	     save_nm, checkpoint_freq}}

impl <'f,'bt,'ut,'rt,'dt> IfaceSettings<'f,'bt,'ut,'rt,'dt>{
	pub fn default(save_nm: String, cur_player: SmSvType) -> IfaceSettings<'f,'bt,'ut,'rt,'dt>{
		Self {
			add_action_to: AddActionTo::None,
			
			underlay: if !screen_reader_mode() {Underlay::Arability} else {Underlay::WaterMountains},
			
			show_structures: true, show_units: true,
			show_bldgs: true, show_zones: true, show_resources: true,
			zone_overlay_map: ZoneOverlayMap::None,
			show_unconnected_bldgs: true, show_unoccupied_bldgs: true,
			show_sectors: true,
			show_pipes: false, // use iface_settings.show_pipes() to read
			
			cur_player,
			zoom_ind: ZOOM_IND_ROOT,
			unit_subsel: 0,
			
			show_expanded_submap: ShowExpandedSubmap::default(),
			
			map_screen_sz: ScreenSz { h: 0, w: 0, sz: 0},
			screen_sz: ScreenSz { h: 0, w: 0, sz: 0},
			
			view_mv_mode: ViewMvMode::Screen,
			
			map_loc: Coord { y: 10, x: 10},
			map_loc_v: VelocCoord { y: SCROLL_ACCEL_INIT, x: SCROLL_ACCEL_INIT},
			
			cur: Coord { y: 0, x: 0},
			cur_v: VelocCoord { y: SCROLL_ACCEL_INIT, x: SCROLL_ACCEL_INIT},
			
			start_map_drag: None,
			
			all_player_pieces_mvd: false,
			
			workers_create_city_sectors: true,
			show_fog: true,//false,
			show_actions: false,//true,
			show_all_zone_information: false,
			
			auto_turn: AutoTurn::Off,
			interrupt_auto_turn: true,
			
			save_nm,
			checkpoint_freq: 5 // save every so many years
		}
	}
	
	// return coordinates in maps[current_zoom] from `screen_coord`
	pub fn screen_coord_to_map_coord(&self, screen_coord: Coord, map_data: &MapData<'rt>) -> Coord {
		let z = map_data.zoom_spacing(self.zoom_ind);
		let map_sz_f = map_data.map_szs[ZOOM_IND_ROOT];
		let sz_j = (map_sz_f.w as f32 / z) as isize;
		
		let c = Coord { y: self.map_loc.y + screen_coord.y - (MAP_ROW_START as isize),
			x: (self.map_loc.x + screen_coord.x) % sz_j };
		debug_assertq!(c.y < (map_sz_f.h as f32 / z) as isize);
		c
	}

	// return coordinates in maps[current_zoom] from cursor position
	pub fn cursor_to_map_coord(&self, map_data: &MapData<'rt>) -> Coord {
		let z = map_data.zoom_spacing(self.zoom_ind);
		let map_sz_f = map_data.map_szs[ZOOM_IND_ROOT];
		let sz_j = (map_sz_f.w as f32 / z) as isize;
		
		let c = Coord { y: self.map_loc.y + self.cur.y - (MAP_ROW_START as isize),
			x: (self.map_loc.x + self.cur.x) % sz_j };
		debug_assertq!(c.y < (map_sz_f.h as f32 / z) as isize);
		c
	}
	
	pub fn cursor_to_map_string(&self, map_data: &MapData<'rt>) -> String {
		let c = self.cursor_to_map_coord(map_data);
		format!("{}, {}", c.y, c.x)
	}
	
	// return coordinates fully zoomed in cursor coordinate from cursor position
	pub fn cursor_to_map_coord_zoomed_in(&self, map_data: &MapData<'rt>) -> Coord {
		let mut cursor_coord = self.cursor_to_map_coord(map_data);
		if self.zoom_ind != map_data.max_zoom_ind() {
			cursor_coord = cursor_coord.to_zoom(self.zoom_ind, map_data.max_zoom_ind(), &map_data.map_szs);
		}
		cursor_coord
	}

	// return coordinates in maps[current_zoom] from cursor position
	pub fn cursor_to_map_ind(&self, map_data: &MapData<'rt>) -> u64 {
		let c = self.cursor_to_map_coord(map_data);
		(c.y *(map_data.map_szs[self.zoom_ind].w as isize)+ c.x) as u64
	}
	
	// return coordinates in maps[current_zoom] from `screen_coord`
	pub fn screen_coord_to_map_ind(&self, screen_coord: Coord, map_data: &MapData<'rt>) -> u64 {
		let c = self.screen_coord_to_map_coord(screen_coord, map_data);
		(c.y *(map_data.map_szs[self.zoom_ind].w as isize)+ c.x) as u64
	}
}

pub fn set_auto_turn_sep(state: AutoTurn, iface_settings: &mut IfaceSettings, renderer: &mut Renderer) {
	iface_settings.auto_turn = state;
	renderer.timeout(match iface_settings.auto_turn {
		AutoTurn::On | AutoTurn::FinishAllActions => 1,
		AutoTurn::Off => MAX_DELAY_FRAMES,
		AutoTurn::N => {panicq!("invalid auto turn");}
	});
}

impl <'f,'bt,'ut,'rt,'dt>DispState<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn set_auto_turn(&mut self, state: AutoTurn) {
		set_auto_turn_sep(state, &mut self.iface_settings, &mut self.renderer);
	}
}

impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn player_end_game(&mut self, relations: &mut Relations) {
		self.state.iface_settings.show_fog = false;
		self.state.iface_settings.show_actions = true;
		self.ui_mode = UIMode::EndGameWindow(EndGameWindowState{});
		self.state.set_auto_turn(AutoTurn::Off);
		
		relations.discover_all_civs(self.state.iface_settings.cur_player as usize);
		
		self.state.renderer.timeout(MAX_DELAY_FRAMES); // if it was chgd from auto_turn_increment
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
	}
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Coord { pub y: isize, pub x: isize }

impl_saving! {Coord {y,x}}

impl Coord {
	#[inline]
	pub fn frm_ind(ind: u64, map_sz: MapSz) -> Coord {
		let w = map_sz.w as u64;

		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			let h = map_sz.h as u64;
		
			let y = ind / w;
			debug_assertq!(y < h);
		}
		Coord {y: (ind / w) as isize,
			 x: (ind % w) as isize}
	}
	
	#[inline]
	pub fn to_ind(&self, map_sz: MapSz) -> usize {
		debug_assertq!(self.y >= 0);
		//debug_assertq!(self.x >= 0);
		debug_assertq!(self.y < map_sz.h as isize);
		//debug_assertq!(self.x < map_sz.w as isize);
		
		map_sz.coord_wrap(self.y, self.x).unwrap_or_else(||
				panicq!("invalid coordinate: {} map_sz {}", self, map_sz)) as usize
		//(self.y as usize)*map_sz.w + (self.x as usize)
	}
	
	pub fn to_screen_coords(&self, map_loc: Coord, map_screen_sz: ScreenSz) -> Option<Coord> {
		let c = Coord { y: self.y - map_loc.y + (MAP_ROW_START as isize),
		                x: self.x - map_loc.x };
		
		return if c.y < (MAP_ROW_START as isize) || c.x < 0 || 
			    c.y >= (map_screen_sz.h as isize) || c.x >= (map_screen_sz.w as isize) {
			None
		}else{
			Some(c)
		};
		
		/*if let Some(ind) = map_sz.coord_wrap(c.y, c.x) {
			Some(Coord::frm_ind(ind, map_sz))
		}else {None}*/
	}
	
	pub fn to_screen_coords_unchecked(&self, map_loc: Coord) -> Coord {
		Coord {
			y: self.y - map_loc.y + (MAP_ROW_START as isize),
			x: self.x - map_loc.x
		}
	}

	pub fn to_zoom(&self, zoom_ind_frm: usize, zoom_ind_to: usize, map_szs: &Vec<MapSz>) -> Self {
		let map_sz = map_szs[zoom_ind_to];
		let map_sz_p1 = map_szs[zoom_ind_frm];
		
		/*debug_assertq!(map_sz_p1.h > map_sz.h && map_sz_p1.w > map_sz.w,
				"map_sz_p1 {} {} map_sz {} {}", 
				map_sz_p1.h, map_sz_p1.w, map_sz.h, map_sz.w);*/
		// ^ `map_sz_p1` should be zoomed in relative to `map_sz`
		
		let frac_i = (map_sz.h as f32) / (map_sz_p1.h as f32);
		let frac_j = (map_sz.w as f32) / (map_sz_p1.w as f32);
		
		// compute coordinates (i, j) from zoomed in coordinates (i_p1, j_p1)
		let coord = Coord {y: min(map_sz.h as isize -1, (self.y as f32*frac_i).round() as isize),
					 x: min(map_sz.w as isize -1, (self.x as f32*frac_j).round() as isize)};
		
		debug_assertq!(coord.y < map_sz.h as isize, "coord.y: {} map_sz.h: {} zoom_ind: {}", coord.y, map_sz.h, zoom_ind_to);
		debug_assertq!(coord.x < map_sz.w as isize, "coord.x: {} map_sz.w: {} zoom_ind: {}", coord.x, map_sz.w, zoom_ind_to);
		coord
	}
	
	pub fn wrap(&self, map_sz: MapSz) -> Self {
		debug_assertq!(self.y >= 0 && self.y < map_sz.h as isize,
				"could not wrap invalid coordinates: {}", self);
		
		let j = self.x;
		let w = map_sz.w as isize;
		
		Coord {
			y: self.y,
			x: match j {
				j if j < 0 => 
					w + (j % w),
				j if j >= w => 
					j % w,
				_ =>
					j
			}
		}
	}
}

impl fmt::Display for Coord {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		//write!(f, "{}, {}", num_format(self.y), num_format(self.x))
		write!(f, "{}, {}", self.y, self.x)
	}
}

// choose smaller coord (upper left corner) to use as start -- returns also the size of the region
pub fn start_coord_use(start_coord: Coord, end_coord: Coord, map_sz: MapSz) -> (Coord, ScreenSz) {
	let mut c = Coord {
		y: if start_coord.y < end_coord.y {start_coord.y} else {end_coord.y},
		x: if start_coord.x < end_coord.x {start_coord.x} else {end_coord.x}};
	
	let h = (start_coord.y - end_coord.y).abs() + 1;
	let mut w = (start_coord.x - end_coord.x).abs() + 1;
	
	// wrap horizontally?
	let mszw = map_sz.w as isize;
	let w2 = ((start_coord.x + mszw) - end_coord.x).abs() + 1;
	let w3 = (start_coord.x - (end_coord.x + mszw)).abs() + 1;
	
	if w2 < w && w2 < w3 {
		w = w2;
		c.x = if (start_coord.x + mszw) < end_coord.x {
			start_coord.x + mszw
		}else{
			end_coord.x
		};
	}
	
	if w3 < w && w3 < w2{
		w = w3;
		c.x = if start_coord.x < (end_coord.x + mszw) {
			start_coord.x
		}else{
			end_coord.x + mszw
		};
	}
	
	(c, ScreenSz {h: h as usize, w: w as usize, sz: (h*w) as usize})
}

#[derive(Clone, Copy, PartialEq)]
pub struct VelocCoord { pub y: f32, pub x: f32 }
impl_saving!{VelocCoord {y,x}}

pub type ScreenSz = MapSz;

pub const MAP_ROW_START: usize = 1;
pub const MAP_ROW_STOP_SZ: usize = 11; // before edge
pub const MAP_COL_STOP_SZ: usize = 16; // before edge. ideally should be even for the rci bars

pub const SUB_MAP_HEIGHT: usize = MAP_ROW_STOP_SZ - 3;
pub const SUB_MAP_WIDTH: usize = SUB_MAP_HEIGHT * 4;
//pub const SUB_MAP_SZ: usize = (SUB_MAP_HEIGHT * SUB_MAP_WIDTH);

pub const TURN_ROW: i32 = MAP_ROW_START as i32;

pub const FAST_TURN_INC: usize = 30;

impl <'f,'bt,'ut,'rt,'dt>Disp<'f,'_,'bt,'ut,'rt,'dt> {
	pub fn create_tech_window(&mut self, prompt_tech: bool) {
		self.reset_auto_turn();
		
		self.ui_mode = UIMode::TechWindow(TechWindowState {
				sel: None, // tech selection index
				sel_mv: TreeSelMv::None,
				tree_offsets: None,
				prompt_tech,
				prev_auto_turn: self.state.iface_settings.auto_turn
		});
		
		self.state.set_auto_turn(AutoTurn::Off);
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
	}
	
	pub fn create_pedigree_window(&mut self) {
		self.reset_auto_turn();
		self.ui_mode = UIMode::NoblePedigree(NoblePedigreeState {
				mode: 0,
				house_nm: None
		});
		
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
	}
	
	pub fn create_spirituality_window(&mut self) {
		self.reset_auto_turn();
		
		self.ui_mode = UIMode::DoctrineWindow(DoctrineWindowState {
				sel: None, // spirituality selection index
				sel_mv: TreeSelMv::None,
				tree_offsets: None,
				prev_auto_turn: self.state.iface_settings.auto_turn
		});
		
		self.state.set_auto_turn(AutoTurn::Off);
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
	}
	
	pub fn create_tech_discovered_window(&mut self, tech_ind: usize) {
		self.reset_auto_turn();
		
		self.ui_mode = UIMode::TechDiscoveredWindow(TechDiscoveredWindowState {
			tech_ind,
			prev_auto_turn: self.state.iface_settings.auto_turn
		});
		
		self.state.set_auto_turn(AutoTurn::Off);
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
	}
	
	// only if settings indicate auto turn should be interrupted
	// or the player isn't in the auto-turn mode
	pub fn create_interrupt_window(&mut self, ui_mode: UIMode<'bt,'ut,'rt,'dt>) {
		if self.state.iface_settings.interrupt_auto_turn || self.state.iface_settings.auto_turn == AutoTurn::Off {
			self.state.set_auto_turn(AutoTurn::Off);
			self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
			self.ui_mode = ui_mode;
		}
	}
	
	// regardless of auto-turn settings
	pub fn create_window(&mut self, ui_mode: UIMode<'bt,'ut,'rt,'dt>) {
		self.state.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
		self.ui_mode = ui_mode;
	}
	
	pub fn create_alert_window(&mut self, txt: String) {
		self.create_window(UIMode::GenericAlert(GenericAlertState {txt}));
	}
}

use crate::movement::manhattan_dist_components;
// inferred line between coordinates, returns only the last part of the 
// line which should be visible on the screen
pub fn line_to(coord_frm: u64, coord_to: u64, map_sz: MapSz, screen_sz: ScreenSz) -> Vec<u64> {
	let coord_frm = Coord::frm_ind(coord_frm, map_sz);
	let coord_to = Coord::frm_ind(coord_to, map_sz);
	
	let dist_comps = manhattan_dist_components(coord_to, coord_frm, map_sz);
	
	let line_fn = |x, slope, b| {
		slope*x as f32 + b
	};
	
	// compute y(x)
	if dist_comps.w > dist_comps.h {
		let mut path = Vec::with_capacity(dist_comps.w);
		
		let slope = (coord_frm.y - coord_to.y) as f32 / (coord_frm.x - coord_to.x) as f32;
		let b = coord_to.y as f32 - slope*coord_to.x as f32;
		
		let max_dist_compute = min(screen_sz.w, dist_comps.w) as isize;
		
		if coord_frm.x < coord_to.x {
			for x in (coord_to.x - max_dist_compute)..coord_to.x {
				if let Some(coord) = map_sz.coord_wrap(line_fn(x, slope, b) as isize, x) {
					path.push(coord);
				}
			}
		}else{
			for x in coord_to.x..(coord_to.x + max_dist_compute) {
				if let Some(coord) = map_sz.coord_wrap(line_fn(x, slope, b) as isize, x) {
					path.push(coord);
				}
			}
		}
		
		path
	// compute x(y)
	}else{
		let mut path = Vec::with_capacity(dist_comps.h);
		
		let slope = (coord_frm.x - coord_to.x) as f32 / (coord_frm.y - coord_to.y) as f32;
		let b = coord_to.x as f32 - slope*coord_to.y as f32;
		
		let max_dist_compute = min(screen_sz.h, dist_comps.h) as isize;
		
		if coord_frm.y < coord_to.y {
			for y in (coord_to.y - max_dist_compute)..coord_to.y {
				if let Some(coord) = map_sz.coord_wrap(y, line_fn(y, slope, b) as isize) {
					path.push(coord);
				}
			}
		}else{
			for y in coord_to.y..(coord_to.y + max_dist_compute) {
				if let Some(coord) = map_sz.coord_wrap(y, line_fn(y, slope, b) as isize) {
					path.push(coord);
				}
			}
		}
		
		path
	}
}

pub struct ScreenFrac {
	pub y: f32,
	pub x: f32
}

// screen reader functions
use super::*;
impl UIMode<'_,'_,'_,'_> {
	pub fn right_side_tabbing(&self) -> bool {
		if let UIMode::TextTab {loc: TextTabLoc::RightSide, ..} = self {
			true
		}else{
			false
		}
	}
	
	// ex. if a window is displayed, don't show the map
	pub fn hide_map(&self) -> bool {
		if !screen_reader_mode() {return false;}
		
		match self {
			// do not hide the map
			UIMode::None |
			UIMode::TextTab {..} |
			UIMode::SetTaxes(_) => false,
				
			// hide the map when these are shown (when in screen reader mode)
			UIMode::Trade(_) |
			UIMode::Budget(_) |
			UIMode::ViewTrade(_) |
			UIMode::ZoneLand(_) |
			UIMode::ContactEmbassyWindow(_) |
			UIMode::ContactNobilityWindow(_) |
			UIMode::NobilityRequestWindow(_) |
			UIMode::SaveAsWindow(_) |
			UIMode::SaveAutoFreqWindow(_) |
			UIMode::GoToCoordinateWindow(_) |
			UIMode::CreateSectorAutomation(_) |
			UIMode::GetTextWindow(_) |
			UIMode::Menu {..} |
			UIMode::ProdListWindow(_) |
			UIMode::CurrentBldgProd(_) |
			UIMode::SelectBldgDoctrine(_) |
			UIMode::CitizenDemandAlert(_) |
			UIMode::PublicPollingWindow(_) |
			UIMode::SelectExploreType(_) |
			UIMode::OpenWindow(_) |
			UIMode::TechWindow(_) |
			UIMode::DoctrineWindow(_) |
			UIMode::PlotWindow(_) |
			UIMode::UnitsWindow(_) |
			UIMode::NobilityDeclaresIndependenceWindow(_) |
			UIMode::NobleUnitsWindow(_) |
			UIMode::SetNobleTax(_) |
			UIMode::GenericAlert(_) |
			UIMode::NoblePedigree(_) |
			UIMode::BrigadesWindow(_) |
			UIMode::BrigadeBuildList(_) |
			UIMode::SectorsWindow(_) |
			UIMode::BldgsWindow(_) |
			UIMode::CitiesWindow(_) |
			UIMode::ManorsWindow(_) |
			//UIMode::ContactEmbassyWindow(_) |
			UIMode::CivilizationIntelWindow(_) |
			UIMode::SwitchToPlayerWindow(_) |
			UIMode::SetDifficultyWindow(_) |
			UIMode::RiotingAlert(_) |
			UIMode::TechDiscoveredWindow(_) |
			UIMode::DiscoverTechWindow(_) |
			UIMode::ObtainResourceWindow(_) |
			UIMode::PlaceUnitWindow(_) |
			UIMode::ResourcesAvailableWindow(_) |
			UIMode::ResourcesDiscoveredWindow(_) |
			UIMode::HistoryWindow(_) |
			UIMode::WarStatusWindow(_) | UIMode::FriendsAndFoesWindow(_) |
			UIMode::EncyclopediaWindow(_) |
			UIMode::InitialGameWindow(_) | UIMode::IntroNobilityJoinOptions(_) |
			UIMode::EndGameWindow(_) |
			UIMode::UnmovedUnitsNotification(_) |
			UIMode::PrevailingDoctrineChangedWindow(_) |
			UIMode::MvWithCursorNoActionsRemainAlert(_) |
			UIMode::CivicAdvisorsWindow(_) |
			UIMode::AcceptNobilityIntoEmpire(_) |
			UIMode::ForeignUnitInSectorAlert(_) |
			UIMode::AboutWindow(_) => true
		}
	}
	
	pub fn show_menu(&self) -> bool {
		if !screen_reader_mode() {return true;}
		if let UIMode::Menu {..} | UIMode::None |
		   UIMode::TextTab {..} | UIMode::SetTaxes(_) = self {
			true
		}else{
			false
		}
	}
}

