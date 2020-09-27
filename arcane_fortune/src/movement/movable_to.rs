use crate::map::*;
use crate::buildings::*;
use crate::units::*;
use crate::saving::SmSvType;
use crate::gcore::hashing::HashedMapEx;
use crate::disp_lib::endwin;
use crate::zones::StructureData;

// given to update_move_search
// (supplies all land discov because we may need to zoom out to compute the full path)
#[derive(Clone, Copy, PartialEq)]
pub enum MvVars<'bt,'ut,'rt,'st,'u,'l> {
	NonCivil {
		units: &'u Vec<Unit<'bt,'ut,'rt,'st>>,
		start_owner: SmSvType, // owner of unit that's moving
		blind_undiscov: Option<&'l Vec<LandDiscov>>
		// ^ when not None (should be the land_discov for the relevant player and zoom):
		//		act as if undiscovered land is always movable to
		//	(should be None for AI and barbarians)
	}, 
	None
}

// land discov only at the current zoom, used for fns in this file
#[derive(Clone, Copy, PartialEq)]
pub enum MvVarsAtZoom<'bt,'ut,'rt,'st,'u,'l> {
	NonCivil {
		units: &'u Vec<Unit<'bt,'ut,'rt,'st>>,
		start_owner: SmSvType, // owner of unit that's moving
		blind_undiscov: Option<&'l LandDiscov>
		// ^ when not None (should be the land_discov for the relevant player and zoom):
		//		act as if undiscovered land is always movable to
		//	(should be None for AI and barbarians)
	}, 
	None
}

impl <'bt,'ut,'rt,'st,'u,'l>MvVars<'bt,'ut,'rt,'st,'u,'l> {
	pub fn to_zoom(&self, zoom_ind: usize) -> MvVarsAtZoom {
		match *self {
			MvVars::NonCivil {units, start_owner, blind_undiscov: None, ..} => {
				MvVarsAtZoom::NonCivil {units, start_owner, blind_undiscov: None}
			}
			MvVars::NonCivil {units, start_owner, blind_undiscov: Some(land_discov)} => {
				MvVarsAtZoom::NonCivil {units, start_owner, 
					blind_undiscov: Some(&land_discov[zoom_ind])
				}
			}
			MvVars::None => {
				MvVarsAtZoom::None
			}
		}
	}
}

// map type consistent with movement type (land based units should only move to land unless
// they are boarding a boat) -- MvVarsAtZoom::NonCivil supplied w/ update_move_search_ui for units, but not 
// civilians (ex. commerce, residents)
#[inline]
pub fn map_type_consistent_w_mvment(coord: u64, mfc: &Map, exf: &HashedMapEx, movement_type: MovementType, mv_vars: MvVarsAtZoom) -> bool {
	let ex_wrapped = exf.get(&coord); 
	
	// movement onto undiscovered land? it should act as if all movement as possible to not reveal map
	if let MvVarsAtZoom::NonCivil {blind_undiscov: Some(land_discov), ..} = &mv_vars {
		if !land_discov.map_coord_ind_discovered(coord) {
			return true;
		}
	}
	
	// not movable over resources
	//if let Some(_) = mfc.resource {return false;}
	//if let Some(_) = mfc.resource_cont {return false;}
	
	match movement_type {
		MovementType::Land | MovementType::LandAndOwnedWalls => {
			if mfc.map_type == MapType::Land {
				return true;
			
			// boarding a boat -- it has room and is the correct owner?
			}else if mfc.map_type == MapType::ShallowWater {
				if let Some(ex) = ex_wrapped {
					if let Some(unit_inds) = &ex.unit_inds {
						if let MvVarsAtZoom::NonCivil {units, start_owner, ..} = mv_vars {
							// boat is owned by current owner and it has room?
							for unit_ind in unit_inds {
								let u = &units[*unit_ind];
								if u.template.carry_capac == 0 || u.owner_id != start_owner {continue;}
								if let Some(units_carried) = &u.units_carried {
									if (units_carried.len() + 1) > u.template.carry_capac {
										continue;
									}
								}
								return true;
							}
						}else{panicq!("mv_vars non civil must be supplied");}
					}
				}
			}
			
			if movement_type == MovementType::LandAndOwnedWalls {
				if let Some(ex) = ex_wrapped {
					if let Some(StructureData {structure_type: StructureType::Wall, ..}) = ex.actual.structure {
						if let MvVarsAtZoom::NonCivil {start_owner, ..} = &mv_vars {
							if ex.actual.owner_id.unwrap() == *start_owner {
								return true;
							}
						}
					}
				}
			}
		} MovementType::Air | MovementType::AllMapTypes => {return true;
		} MovementType::ShallowWater => {
			if mfc.map_type == MapType::ShallowWater {return true;}
		} MovementType::AllWater => {
			if mfc.map_type == MapType::DeepWater || mfc.map_type == MapType::ShallowWater {return true;}
		} MovementType::N => {panicq!("Invalid movement type");}
	}
	
	false
}

#[derive(PartialEq, Clone)]
pub enum Dest {
	NoAttack,
	Attack {start_owner: SmSvType, dest_coord: u64, ignore_own_walls: bool}, // ex. city defenders needing to exit city (ai/ai_actions.rs)
	RepairWall {dest_coord: u64}, // destination can be a wall
	IgnoreWalls {dest_coord: u64}, // allow movement through walls, and onto city hall (as terminal location)
	// ^ IgnoreWalls allows for computation of direct paths to city hall -- any walls in the way are cleared (in ai/attack_fronts.rs)
	// 	(not used to actually move units, but only get the direct path to a city hall)
	
	IgnoreOwnWalls {start_owner: SmSvType, dest_coord: u64}, // used for AI movements--navigation out of city walls
	//	ex. assembly of attack fronts in ai/attack_fronts.rs
	
	RepairBldg {dest_coord: u64}, // destination can be a bldg
}

impl Dest {
	pub fn from(action_type: &ActionType, end_coord_ind: u64, start_owner: Option<SmSvType>) -> Self {
		match action_type {
			ActionType::MvIgnoreWalls => Dest::IgnoreWalls {dest_coord: end_coord_ind},
			ActionType::Attack {ignore_own_walls, ..} => Dest::Attack {
				start_owner: start_owner.unwrap(),
				dest_coord: end_coord_ind,
				ignore_own_walls: *ignore_own_walls
			},
			// ActionType::Attack {attack_coord} may not be set(?) when calling
			// this function from movement/mod.rs
			
			ActionType::WorkerBuildStructure {structure_type: StructureType::Gate, ..} |
			ActionType::WorkerRepairWall {..} => Dest::RepairWall {dest_coord: end_coord_ind},
			
			ActionType::WorkerContinueBuildBldg {..} |
			ActionType::BurnBuilding {..} => Dest::RepairBldg {dest_coord: end_coord_ind},
			
			ActionType::MvIgnoreOwnWalls => Dest::IgnoreOwnWalls {
				start_owner: start_owner.unwrap(),
				dest_coord: end_coord_ind
			},
			
			ActionType::Mv |
			ActionType::MvWithCursor |
			ActionType::CivilianMv |
			ActionType::AutoExplore {..} |
			ActionType::WorkerBuildStructure {..} |
			ActionType::WorkerBuildBldg {..} | 
			ActionType::Fortify {..} |
			ActionType::WorkerZone {..} |
			ActionType::WorkerRmZonesAndBldgs {..} |
			ActionType::WorkerZoneCoords {..} |
			ActionType::UIWorkerAutomateCity |
			ActionType::BrigadeCreation {..} |
			ActionType::SectorCreation {..} |
			ActionType::SectorAutomation {..} |
			ActionType::GroupMv {..} => Dest::NoAttack
		}
	}
}

// is the coordinate movable to for a unit?
pub fn movable_to(src_coord: u64, dest_coord_chk: u64, mfc: &Map, exf: &HashedMapEx, mv_vars: MvVarsAtZoom, bldgs: &Vec<Bldg>, 
		dest: &Dest, movement_type: MovementType) -> bool {
	if src_coord == dest_coord_chk {return true;} // if this weren't here, the fn could return false if there are too many units on the plot
	///////// **** note: ^ do not rely on src_coord to be valid, it can sometimes be set to dest_coord_chk+1 to skip this check (such as when
	/////////   we are creating a unit or bldg and simply want to see if the space is available
	if movement_type == MovementType::Air {return true;}
	
	// if we are supposed to move only on land, check that this is land, same for water
	if !map_type_consistent_w_mvment(dest_coord_chk, mfc, exf, movement_type, mv_vars) {return false;}
	
	// check movability related to other units, bldgs, and structures 
	if let Some(ex) = exf.get(&dest_coord_chk) {
		// movement onto undiscovered land? it should act as if all movement as possible to not reveal map
		if let MvVarsAtZoom::NonCivil {blind_undiscov: Some(land_discov), ..} = &mv_vars {
			if !land_discov.map_coord_ind_discovered(dest_coord_chk) {
				return true;
			}
		}
		
		macro_rules! non_attack_chk{() => {
			// too many units per plot
			// (don't check in the case where a unit is boarding a boat)
			if (mfc.map_type == MapType::Land && movement_type == MovementType::Land) ||
			   (mfc.map_type == MapType::Land && movement_type == MovementType::LandAndOwnedWalls) ||
			   (mfc.map_type == MapType::ShallowWater && movement_type == MovementType::ShallowWater) {
				if let Some(unit_inds) = &ex.unit_inds {
					if (unit_inds.len()+1) > MAX_UNITS_PER_PLOT {
						return false;
					}
				}
			}
			
			// not possible to travel on bldgs
			if ex.bldg_ind != None {return false;}
		};};
		
		// at attack destination
		match dest {
			Dest::Attack {dest_coord, ..} |
			Dest::IgnoreWalls {dest_coord} => {
				if *dest_coord == dest_coord_chk {
					if let Some(bldg_ind) = ex.bldg_ind {
						let bt = &bldgs[bldg_ind].template;
						
						// if not at city hall
						if bt.nm[0] != CITY_HALL_NM && bt.nm[0] != BARBARIAN_CAMP_NM {
							return false;
						}
						
						// if at city hall, check at entrance
						/*let bldg_coord = Coord::frm_ind(bldgs[bldg_ind].coord, map_sz);
						
						let sz = bldgs[bldg_ind].template.sz;
						
						let w = sz.w as isize;
						let h = sz.h as isize;
						
						if map_sz.coord_wrap(bldg_coord.y + h-1, bldg_coord.x + w/2).unwrap() != dest_coord_chk {
							return false;
						}*/
					}
				}else {non_attack_chk!();}
			} Dest::RepairBldg {dest_coord} => {
				if *dest_coord != dest_coord_chk {
					non_attack_chk!();
				}
			} Dest::NoAttack | Dest::RepairWall {..} | Dest::IgnoreOwnWalls {..} => {
				non_attack_chk!();
			}
		}
		
		if let Some(structure) = &ex.actual.structure {
			match structure.structure_type {
				// conditions we allow moving on or through a wall
				StructureType::Wall => {
					return match &dest {
						Dest::NoAttack | Dest::RepairBldg {..} => {false}
						Dest::Attack {start_owner, dest_coord, ignore_own_walls} => {
							(*dest_coord == dest_coord_chk) || // <-- attacking wall
								(movement_type == MovementType::LandAndOwnedWalls || // <-- ex. archers
								*ignore_own_walls) && // <-- AI (for moving out of own cities, perhaps do this without violating wall movement in the future)
									Some(*start_owner) == ex.actual.owner_id
						}
						
						// only allow if final destination is a wall
						Dest::RepairWall {dest_coord} => {*dest_coord == dest_coord_chk}
						Dest::IgnoreWalls {..} => {true}
						
						// as long as we are not ending on a wall, moving through our own is ok
						// (if movement type is LandAndOwnedWalls we DO allow ending on a wall)
						Dest::IgnoreOwnWalls {start_owner, dest_coord} => {
							(*dest_coord != dest_coord_chk ||
							 movement_type == MovementType::LandAndOwnedWalls) &&
							 Some(*start_owner) == ex.actual.owner_id
						}
					};
				} StructureType::Gate => {
					// if attacking, allow moving to gate if at destination
					if let Dest::Attack {dest_coord, ..} = &dest {
						if *dest_coord == dest_coord_chk {return true;}
					}
					
					// if attacker owns the gate, allow moving through it
					if let MvVarsAtZoom::NonCivil {start_owner, ..} = &mv_vars {
						return *start_owner == ex.actual.owner_id.unwrap();
					}
				} StructureType::Road => {
				} StructureType::N => {panicq!("unknown structure")}
			}
		}
	}
	true
}

// is the coordinate movable to for a civilian? (they only move on roads)
// exf is exs at the full zoom
pub fn civil_movable_to(src_coord: u64, dest_coord: u64, mfc: &Map, exf: &HashedMapEx, _: MvVarsAtZoom, bldgs: &Vec<Bldg>,
		_dest: &Dest, _: MovementType) -> bool {
	debug_assertq!(src_coord != dest_coord); // civil src to dest are the same
	if mfc.map_type != MapType::Land {return false;}
	
	if let Some(ex) = exf.get(&dest_coord) {
		if let Some(structure) = ex.actual.structure {
			return match structure.structure_type {
				StructureType::Road | StructureType::Gate => true,
				StructureType::Wall => false,
				StructureType::N => {panicq!("unknown structure")}
			};
		}
		
		if let Some(bldg_ind) = ex.bldg_ind {
			if bldgs[bldg_ind].template.nm[0] == CITY_HALL_NM {
				return true;
			}
		}
	} // ex set
	false
}

