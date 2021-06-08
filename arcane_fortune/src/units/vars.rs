use std::num::ParseIntError;
use std::process::exit;
use std::fmt;

use crate::map::*;
use crate::resources::*;
use crate::buildings::*;
use crate::saving::*;
use crate::doctrine::DoctrineTemplate;
use crate::gcore::hashing::{HashedMapEx};
use crate::movement::{MvVarsAtZoom, Dest};
use crate::disp::Coord;
use crate::localization::Localization;
use crate::player::*;
use crate::containers::*;
use crate::zones::Zone;

pub const MAX_UNITS_PER_PLOT: usize = 2;

enum_From!{MovementType {
	AllWater,
	ShallowWater,
	Land,
	LandAndOwnedWalls, // allows ending path on the moving civ's own wall
	Air,
	AllMapTypes /* unlike `Air`, this does not ignore structures
				^ used for path finding when the destination is far and
				  we want to ignore the fact that some checkpoints could be in water
				  (for land based units) while *not* ignoring structures like walls
				  (ignoring walls could result in units arriving to places that
				  should otherwise be inaccessible like within walled-off cities) */
}}

// required to parse movement_type from text file configurations [ find_req_key_parse() ]
impl std::str::FromStr for MovementType {
	type Err = ParseIntError;
	
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Result::Ok (match s {
			"ShallowWater" => MovementType::ShallowWater,
			"AllWater" => MovementType::AllWater,
			"Land" => MovementType::Land,
			"LandAndOwnedWalls" => MovementType::LandAndOwnedWalls,
			"Air" => MovementType::Air,
			_ => {q!(format!("Failed to parse \"{}\" into movement_type. Valid options are: ShallowWater, Land, LandAndOwnedWalls, Air", s));}
		})
	}
}

pub const WORKER_NM: &str = "Worker";
pub const WARRIOR_NM: &str = "Wood Clubber";
pub const ARCHER_NM: &str = "Archer";
pub const ICBM_NM: &str = "ICBM";
pub const EXPLORER_NM: &str = "Explorer";
pub const RIOTER_NM: &str = "Rioter";

pub const WORKER_WALL_CONSTRUCTION_TURNS: usize = 2;

#[derive(PartialEq, Clone, Default)]
pub struct UnitTemplate<'rt> {
	pub id: SmSvType,
	pub nm: Vec<String>,
	
	pub tech_req: Option<Vec<usize>>, // index into tech_templates
	pub resources_req: Vec<&'rt ResourceTemplate>,
	
	pub movement_type: MovementType,
	
	pub carry_capac: usize, // number of units ship can carry
	
	pub actions_per_turn: f32,
	pub attack_per_turn: Option<usize>, // health drain on attacked
	pub siege_bonus_per_turn: Option<usize>, // health drain on walls in addition to attack_per_turn
	pub repair_wall_per_turn: Option<usize>,
	pub assassin_per_turn: Option<usize>,
	pub attack_range: Option<usize>,
	pub max_health: usize,
	
	pub production_req: f32, // to action req. to produce
	pub char_disp: char, // to show on map
	pub upkeep: f32, // should be positive to indicate a negative expense
}

impl_saving_template!{UnitTemplate<'rt>{id, nm, tech_req, resources_req,
			movement_type, carry_capac, actions_per_turn, attack_per_turn, siege_bonus_per_turn,
			repair_wall_per_turn, assassin_per_turn,
			attack_range, max_health, production_req, char_disp, upkeep}}

impl <'rt>UnitTemplate<'rt> {
	pub fn frm_str<'ut>(txt: &str, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
		for ut in unit_templates.iter() {
			if txt == ut.nm[0] {
				return ut;
			}
		}
		panicq!("Could not find unit type \"{}\"", txt);
	}
}

enum_From!{ExploreType {SpiralOut, Random}} // also update explore_types_list if anything else is added

//////// sector automation
enum_From!{SectorUnitEnterAction {
	Assault, // attack and declare war on anyone that enters
	Defense, // attack only if already at war with
	Report // pause game and report enemy
}}

#[derive(Copy, Clone, PartialEq)]
pub enum SectorIdleAction { // for the SectorAutomation ActionType
	Sentry, // do not move
	Patrol {
		dist_monitor: usize,
		perim_coord_ind: usize,
		perim_coord_turn_computed: usize, // if this doesn't match what's in pstats.sector,
		// then the perimeter has changed and should be reloaded (i.e., `perim_coord_ind` is stale)
	} // unit walks perimeter, and will respond to threats up to dist_monitor away
}
//////////

enum_From!{SectorCreationType {New, AddTo}}

#[derive(Clone, PartialEq)]
pub enum ActionType<'bt,'ut,'rt,'dt> {
	Mv,
	MvWithCursor, // only used with the player AI -- shouldn't be left at this state once the player exits this move mode
	MvIgnoreWallsAndOntoPopulationCenters,
	// ^ ignores walls for AI attack planning (into the city).
	//   it is used to get the path coords for a standard movement `Mv` operation
	//   where walls are not ignored. (due to the planning in ai/attack_fronts)
	MvIgnoreOwnWalls,
	// ^ for AI movements, because the AI can get stuck in its own cities (todo, this could be better handled)
	CivilianMv,
	AutoExplore {start_coord: u64, explore_type: ExploreType}, // moves explorer back to starting coord and tries again if we run out of places to explore
	
	WorkerBuildStructure {structure_type: StructureType, turns_expended: usize},
	WorkerBuildPipe,
	WorkerRepairWall {wall_coord: Option<u64>, turns_expended: usize},
	// ^ coordinate wall is at (not set until after path coords are computed--but should be set before the end of the turn)
	
	WorkerBuildBldg { 
		valid_placement: bool,
		template: &'bt BldgTemplate<'ut,'rt,'dt>, 
		doctrine_dedication: Option<&'dt DoctrineTemplate>,
		bldg_coord: Option<u64> // after traversing to bldg site this is set (set in end_turn, set to None when creating new action)
	},
	Attack { // these values are allowed to be None because in UI mode an attack can be set to an empty plot of land
		attack_coord: Option<u64>, // coordinate to attack after traversing all of U.path_coords (end of the path will not contain final dest.)
		attackee: Option<u32>, // owner to attack--abandon attack if owner at the final attack coordinate is different
		ignore_own_walls: bool // useful for AI defense so AI does not get stuck in defending city
	},
	
	Fortify { turn: usize  }, // turn fortified
	
	WorkerZone { // for UI placement with human player 
		valid_placement: bool,
		zone: Zone, // to create
		start_coord: Option<u64>, // set to start coord of zone (at full zoom)
		end_coord: Option<u64>  }, // set to end of zone (second location chosen by cursor, converted to map coords at full zoom)
	
	GroupMv { // for UI movement of multiple units with human player (unit should be set to `Mv` once enter is pressed)
		start_coord: Option<u64>, // set to start coord of rectangle (at full zoom)
		end_coord: Option<u64>  }, // set to end of rectangle (second location chosen by cursor, converted to map coords at full zoom)
	
	BrigadeCreation { // for UI (stored in pstats once selected)
		nm: String, // index into nms.brigrades[] -- should really be a string but we might be relying on the `Copy` trait somewhere for this enum(?)
		start_coord: Option<u64>,
		end_coord: Option<u64>
	},
	
	SectorCreation { // for UI (stored in pstats once selected)
		nm: String, // index into nms.sectors[] -- should really be a string but we might be relying on the `Copy` trait somewhere for this enum(?)
		creation_type: SectorCreationType, // New or AddTo
		start_coord: Option<u64>,
		end_coord: Option<u64>
	},
	
	WorkerZoneCoords { zone: Zone }, // the zone is placed at path_coords, for AI
	
	UIWorkerAutomateCity,
	
	BurnBuilding {coord: u64},
	
	WorkerContinueBuildBldg,  // for UI bldg selection. (unit should be set to `WorkerBuildBldg` once enter is pressed)
		// ^ the bldg_ind that the unit should continue building can be obtained from units/worker_can_continue_bldg()
		
	SectorAutomation {
		unit_enter_action: SectorUnitEnterAction,
		idle_action: SectorIdleAction,
		sector_nm: String
	},
	// ^ unit guards sector and will perform `unit_enter_action` when any threats enter. performs `idle_action` when no threats around
	
	WorkerRmZonesAndBldgs {
		start_coord: Option<u64>,
		end_coord: Option<u64>
	},
	
	ScaleWalls, // used for assassains -- chance of discovery
	Assassinate {
		attack_coord: Option<u64> // coordinate to attack after traversing all of U.path_coords (end of the path will not contain final dest.)
	}
}

impl fmt::Display for ActionType<'_,'_,'_,'_> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			ActionType::Mv => String::from("Mv"),
			ActionType::WorkerRmZonesAndBldgs {..} => String::from("WorkerRmZonesAndBldgs"),
			ActionType::MvWithCursor => String::from("MvWithCursor"),
			ActionType::MvIgnoreWallsAndOntoPopulationCenters => String::from("MvIgnoreWallsAndOntoPopulationCenters"),
			ActionType::MvIgnoreOwnWalls => String::from("MvIgnoreOwnWalls"),
			ActionType::CivilianMv => String::from("CivilianMv"),
			ActionType::WorkerBuildPipe => String::from("BuildPipe"),
			ActionType::AutoExplore {start_coord, ..} => format!("AutoExplore {}", start_coord),
			ActionType::WorkerBuildStructure {..} => String::from("WorkerBuildStructure"),
			ActionType::WorkerRepairWall {wall_coord: Some(coord), turns_expended} => {
				format!("WorkerRepairWall at coordinate {}, turns expended {}", coord, turns_expended)}
			ActionType::WorkerRepairWall {wall_coord: None, turns_expended} => {
				format!("WorkerRepairWall at unset coordinate, turns_expended {}", turns_expended)}
			ActionType::WorkerBuildBldg {..} => String::from("WorkerBuildBldg"),
			ActionType::Attack {attack_coord, attackee, ignore_own_walls} => {
				let attack_coord_txt = if let Some(attack_coord) = attack_coord {format!("{}", attack_coord)} else {String::from("None")};
				let attackee_txt = if let Some(attackee) = attackee {format!("{}", attackee)} else {String::from("None")};
				format!("Attack (coord: {}, attackee: {}, ignore_own_walls: {})", attack_coord_txt, attackee_txt, ignore_own_walls)
			}
			ActionType::Assassinate {..} => format!("Assassinate"),
			ActionType::Fortify {turn} => format!("Fortify (turn: {})", turn),
			ActionType::WorkerZone {..} => String::from("WorkerZone"),
			ActionType::GroupMv{..} => String::from("GroupMv"),
			ActionType::BrigadeCreation{..} => String::from("BrigadeCreation"),
			ActionType::SectorCreation{..} => String::from("SectorCreation"),
			ActionType::WorkerZoneCoords {..} => String::from("WorkerZoneCoords"),
			ActionType::UIWorkerAutomateCity => String::from("UIWorkerAutomateCity"),
			ActionType::BurnBuilding {..} => String::from("BurnBuilding"),
			ActionType::WorkerContinueBuildBldg {..} => String::from("WorkerContinueBuildBldg"),
			ActionType::SectorAutomation {..} => String::from("SectorAutomation"),
			ActionType::ScaleWalls => String::from("ScaleWalls")
		})
	}
}

impl <'bt,'ut,'rt,'dt> ActionType<'bt,'ut,'rt,'dt> {
	pub fn nm(&self, l: &Localization) -> String {
		match self {
			ActionType::WorkerRmZonesAndBldgs {..} => l.Removing_zones_and_bldgs.clone(),
			ActionType::Mv | ActionType::MvWithCursor | ActionType::ScaleWalls |
			ActionType::MvIgnoreWallsAndOntoPopulationCenters | ActionType::MvIgnoreOwnWalls |
			ActionType::GroupMv {..} => l.Moving.clone(),
			ActionType::BrigadeCreation {..} => {panicq!("unit action shouldn't be set to brigade creation");}
			ActionType::SectorCreation {..} => {panicq!("unit action shouldn't be set to sector creation");}
			ActionType::CivilianMv {..} => l.Moving_civilian.clone(),
			ActionType::AutoExplore {..} => l.Auto_exploring.clone(),
			ActionType::WorkerBuildStructure {structure_type: StructureType::Road, ..} => l.Building_road.clone(),
			ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, ..} => l.Building_gate.clone(),
			ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, ..} => l.Building_wall.clone(),
			ActionType::WorkerBuildPipe => l.Building_pipe.clone(),
			ActionType::WorkerRepairWall {..} => l.Repairing_wall.clone(),
			ActionType::WorkerBuildBldg {template, ..} => format!("{} {}", l.Building, template.nm[l.lang_ind]),
			ActionType::WorkerContinueBuildBldg {..} => l.Building.clone(),
			ActionType::Attack {..} => l.Attacking.clone(),
			ActionType::Assassinate {..} => l.Assassinate.clone(),
			ActionType::Fortify {..} => l.Fortified.clone(),
			ActionType::WorkerZone {zone, ..} => format!("{} {}", l.Zoning, zone.ztype.to_str(l)),
		      ActionType::WorkerZoneCoords {zone, ..} => format!("{} {}", l.Zoning, zone.ztype.to_str(l)),
		      ActionType::BurnBuilding {..} => l.Burn_building.clone(),
		      ActionType::UIWorkerAutomateCity => l.Automated.clone(),
		      ActionType::SectorAutomation {sector_nm, idle_action, unit_enter_action} => {
		      	let idle_action_nm = match idle_action {
					SectorIdleAction::Sentry => &l.Sentry,
					SectorIdleAction::Patrol {..} => &l.Patrol
				};
				
				let enter_action_nm = match unit_enter_action {
					SectorUnitEnterAction::Assault => &l.Assault,
					SectorUnitEnterAction::Defense => &l.Defense,
					SectorUnitEnterAction::Report => &l.Report,
					SectorUnitEnterAction::N => {panicq!("invalid unit enter action");}
				};
		      	format!("{} {} ({} {})", idle_action_nm, enter_action_nm, l.Sector, sector_nm)
			}
		      ActionType::WorkerBuildStructure {structure_type: StructureType::N, ..} => panicq!("invalid structure")
		}
	}
}

// continuation of path if we can't compute it all at once &
// zoomed out approximate trajectory
#[derive(Clone, PartialEq)]
pub struct ActionMetaCont {
	pub final_end_coord: Coord,
	pub checkpoint_path_coords: Vec<u64>
	// ^ these are converted and saved as coordinates at ZoomInd::Full
	//   (although they are computed from the ZOOM_IND_ROOT map)
}

impl_saving!{ActionMetaCont {final_end_coord, checkpoint_path_coords}}

// used w/ each unit, plus staging for iface_settings
#[derive(Clone, PartialEq)]
pub struct ActionMeta<'bt,'ut,'rt,'dt> {
	pub action_type: ActionType<'bt,'ut,'rt,'dt>,
	pub actions_req: f32,
	
	pub path_coords: Vec<u64>,
	pub action_meta_cont: Option<ActionMetaCont>
	// if it is too far to compute the full
	// path and save it in path_coords, then we save the final_end_coord and
	// compute the full path on a zoomed out map
}

impl_saving!{ActionMeta<'bt,'ut,'rt,'dt> {action_type, actions_req, path_coords, action_meta_cont}}

impl <'bt,'ut,'rt,'dt>ActionMeta<'bt,'ut,'rt,'dt> {
	pub fn new(action_type: ActionType<'bt,'ut,'rt,'dt>) -> Self {
		ActionMeta {
			action_type,
			actions_req: 0.,
			path_coords: Vec::new(),
			action_meta_cont: None
		}
	}
	
	pub fn with_capacity(action_type: ActionType<'bt,'ut,'rt,'dt>, len: usize) -> Self {
		ActionMeta {
			action_type,
			actions_req: 0.,
			path_coords: Vec::with_capacity(len),
			action_meta_cont: None
		}
	}
}

#[derive(Clone, PartialEq)]
pub struct Unit<'bt,'ut,'rt,'dt> {
	pub nm: String,
	pub health: usize,
	pub owner_id: SmSvType,
	pub creation_turn: SmSvType,
	pub template: &'ut UnitTemplate<'rt>,
	coord: u64,
	
	pub units_carried: Option<Vec<Unit<'bt,'ut,'rt,'dt>>>,
	
	pub actions_used: Option<f32>, // once actions_used >= actions_per_turn, this should switch to none
	
	// current actions -- they are popped off, so first entry is execute last
	pub action: Vec<ActionMeta<'bt,'ut,'rt,'dt>>
}

impl_saving!{Unit<'bt,'ut,'rt,'dt> {nm, health, owner_id, creation_turn, template, coord, units_carried, actions_used, action}}

impl <'bt,'ut,'rt,'dt> Unit<'bt,'ut,'rt,'dt> {
	pub fn default(unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> Self {
		Unit {
			nm: " ".to_string(), health: 0, owner_id: 0, creation_turn: 0, template: &unit_templates[0],
			coord: 0, units_carried: None, actions_used: None, action: Vec::new()
		}
	}
	
	#[inline]
	pub fn return_coord(&self) -> u64 {self.coord}	
}

// a variant of compute_zooms_coord could be made that works only w/ a single player..., making this
// function not require all player data...
pub fn rm_unit_coord_frm_map<'bt,'ut,'rt,'dt>(unit_ind: usize, is_cur_player: bool, coord: u64,
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		pstats: &mut Stats<'bt,'ut,'rt,'dt>, map_sz: MapSz, gstate: &mut GameState) {
	
	// update exs at all zoom lvls
	compute_zooms_coord_unit(coord, unit_ind, RecompType::RmUnit, map_data, exs);
	
	// update fog of war for old location
	compute_active_window(coord, is_cur_player, PresenceAction::SetAbsent, map_data, exs, pstats, map_sz, gstate, units);
}

pub fn set_coord<'bt,'ut,'rt,'dt>(coord: u64, unit_ind: usize, is_cur_player: bool, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, pstats: &mut Stats<'bt,'ut,'rt,'dt>, map_sz: MapSz, gstate: &mut GameState) {
	
	let u = &units[unit_ind];
	
	rm_unit_coord_frm_map(unit_ind, is_cur_player, u.coord, units, map_data, exs, pstats, map_sz, gstate);
	units[unit_ind].coord = coord;
	
	// update exs at all zoom lvls
	compute_zooms_coord_unit(coord, unit_ind, RecompType::AddUnit, map_data, exs);
	
	// indicate unit at new location
	compute_active_window(coord, is_cur_player, PresenceAction::SetPresentAndDiscover, map_data, exs, pstats, map_sz, gstate, units);
}

pub fn unboard_unit<'bt,'ut,'rt,'dt>(coord: u64, mut u: Unit<'bt,'ut,'rt,'dt>, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>) {
	u.coord = coord;
	u.action.clear();
	u.actions_used = Some(0.);
	
	units.push(u);
	let unit_ind = units.len() - 1;
	
	// update exs at all zoom lvls
	compute_zooms_coord_unit(coord, unit_ind, RecompType::AddUnit, map_data, exs);
}

use crate::movement::movable_to;
use crate::renderer::endwin;

pub fn add_unit<'bt,'ut,'rt,'dt>(coord: u64, is_cur_player: bool, unit_template: &'ut UnitTemplate<'rt>, 
		units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>,
		bldgs: &Vec<Bldg>, player: &mut Player<'bt,'ut,'rt,'dt>, gstate: &mut GameState, temps: &Templates<'bt,'ut,'rt,'dt,'_>){
	
	let exf = exs.last_mut().unwrap();
	// set the first param (current location of unit) to some other coord than the destination so movable_to does not always return true
	debug_assertq!(movable_to(coord+1, coord, &map_data.get(ZoomInd::Full, coord), exf, MvVarsAtZoom::NonCivil{units, start_owner: player.id, blind_undiscov: None},
				bldgs, &Dest::NoAttack, unit_template.movement_type));
	
	units.push(Unit {
			nm: temps.nms.units[gstate.rng.usize_range(0, temps.nms.units.len())].clone(),
			health: unit_template.max_health,
			owner_id: player.id,
			creation_turn: gstate.turn as SmSvType,
			template: unit_template,
			coord,
			units_carried: None,
			actions_used: Some(0.),
			action: Vec::new()
		});
	
	let unit_ind = units.len() - 1;
	
	let map_sz = map_data.map_szs[map_data.max_zoom_ind()];
	
	// update exs at all zoom lvls
	compute_zooms_coord_unit(coord, unit_ind, RecompType::AddUnit, map_data, exs);
	compute_active_window(coord, is_cur_player, PresenceAction::SetPresentAndDiscover, map_data, exs, &mut player.stats, map_sz, gstate, units);
	
	player.stats.unit_expenses += unit_template.upkeep;
	
	// record-keeping
	player.add_unit(unit_ind, coord, unit_template, temps.units, map_sz);
}

pub fn disband_unit<'bt,'ut,'rt,'dt>(unit_ind: usize, is_cur_player: bool, units: &mut Vec<Unit<'bt,'ut,'rt,'dt>>, map_data: &mut MapData, 
		exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, gstate: &mut GameState, map_sz: MapSz) {
	
	let u = &units[unit_ind];
	let owner_id = u.owner_id as usize;
	let player = &mut players[owner_id];
	
	player.rm_unit(unit_ind, u.template);
	
	rm_unit_coord_frm_map(unit_ind, is_cur_player, u.coord, units, map_data, exs, &mut player.stats, map_sz, gstate);
		
	let u = &units[unit_ind];
	player.stats.unit_expenses -= u.template.upkeep;
	
	// swap_remove [move units.last() to units[unit_ind]]... (if we removed the last unit [unit_ind == units.len()] then nothing needs to be done]
	if let Some(last_unit) = units.pop() {
		if unit_ind != units.len() {
			//printlnq!("moving unit {} ({}) to {} (owner {} owner prev {})",
			//		last_unit.template.nm, units.len(), unit_ind, last_unit.owner_id, owner_id);
			debug_assertq!(unit_ind < units.len());
			units[unit_ind] = last_unit;
			
			let u = &units[unit_ind];
			let owner_id = u.owner_id as usize;
			let player = &mut players[owner_id];
			
			player.chg_unit_ind(units.len(), unit_ind, u.template);
			
			rm_unit_coord_frm_map(units.len(), is_cur_player, u.coord, units, map_data, exs, &mut player.stats, map_sz, gstate); //////////// ??????????
			compute_zooms_coord_unit(units[unit_ind].coord, unit_ind, RecompType::AddUnit, map_data, exs);
		}
	}
}

