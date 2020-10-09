use std::time::Instant;
use std::collections::VecDeque;
use crate::map::{MapEx, ZoneType, Map, MapSz, ZoneDemandType, MapType, TechProg, ResourceCont,
	StructureType, LandDiscov
};
use crate::player::{Nms, AIPersonality, Stats, PlayerType, PersonName, Personalization, Player};
use crate::zones::{ZoneDemandRaw, StructureData,
	ZoneAgnosticStats, ZoneAgnosticContribFracs, ZoneAgnosticLocallyLogged};
use crate::disp::{Coord, ScreenSz, Fire, FireTile};
use crate::doctrine::DoctrineTemplate;
use crate::buildings::*;
use crate::units::*;
use crate::tech::*;
use crate::ai::{AttackFront, AttackFronts, AttackFrontState, AIState,
	BarbarianState, CityState, Neighbors, Defender, AIConfig};
use crate::gcore::{LogType, Log, Relations, RelationsConfig,
	Sector, Brigade, Rectangle, PerimCoords};
use crate::resources::ResourceTemplate;
use crate::gcore::{RelationStatus, Bonuses};//, HashStruct64};
use crate::nobility::*;

pub type SmSvType = u32; // ex, to cast down from usize

impl Default for ZoneAgnosticStats {
	fn default() -> Self {
		ZoneAgnosticStats {
			turn_computed: 0,
			gov_bldg_happiness_sum: 0.,
			
			locally_logged: ZoneAgnosticLocallyLogged::default(),
			
			crime_sum: 0.,
			health_sum: 0.,
			unemployment_sum: 0.
		}
	}
}

impl ZoneAgnosticStats {
	pub fn default_init(turn_computed: usize,
			doctrine_templates: &Vec<DoctrineTemplate>) -> Self {
		ZoneAgnosticStats {
			turn_computed,
			gov_bldg_happiness_sum: 0.,
			
			locally_logged: ZoneAgnosticLocallyLogged::default_init(doctrine_templates),
			
			crime_sum: 0.,
			health_sum: 0.,
			unemployment_sum: 0.
		}
	}
}

impl ZoneAgnosticLocallyLogged {
	pub fn default_init(doctrine_templates: &Vec<DoctrineTemplate>) -> Self {
		ZoneAgnosticLocallyLogged {
			happiness_sum: 0.,
			doctrinality_sum: vec![0.; doctrine_templates.len()],
			pacifism_sum: 0.,
			contrib: ZoneAgnosticContribFracs::default()
		}
	}
}

impl Default for AIConfig<'_> {
	fn default() -> Self {
		AIConfig {
			city_creation_resource_bonus: Vec::new(),
			strategic_resources: Vec::new()
		}
}}

impl ZoneDemandRaw {
	pub fn default_init(turn_computed: usize) -> Self {
		ZoneDemandRaw {turn_computed,
			demand: vec!{0; ZoneDemandType::N as usize}.into_boxed_slice() }
	}
}

impl Default for ZoneDemandRaw {
	fn default() -> Self {
		ZoneDemandRaw {turn_computed: 0,
			       demand: vec!{0; ZoneDemandType::N as usize}.into_boxed_slice() }
	}
}

impl Default for Map<'_> {
	fn default() -> Self { 
		Map {
			arability: 0.,
			show_snow: false,
			elevation: 0.,
			map_type: MapType::Land,
			resource: None,
			resource_cont: None
		}
	}
}

impl Default for ResourceCont {
	fn default() -> Self {
		ResourceCont {offset_i: 0, offset_j: 0}
	}
}

impl Default for AttackFront {
	fn default() -> Self {
		AttackFront {
			siegers: Vec::new(),
			attackers: Vec::new(),
			state: AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()}
		}
}}

impl Default for AttackFronts {
	fn default() -> Self {
		AttackFronts {vals: Vec::new()}
	}
}

impl Default for AIState<'_,'_,'_,'_> {
	fn default() -> Self {
		AIState {
			city_states: Vec::new(),
			attack_fronts: AttackFronts::default(),
			icbm_inds: Vec::new(),
			damaged_wall_coords: Vec::new(),
			next_bonus_bldg: None,
			goal_doctrine: None,
			paused: false
		}}}

impl Default for Defender {
	fn default() -> Self {
		Defender {unit_ind: 0, failed_mv_attempts: 0}
	}
}

impl Default for CityState<'_,'_,'_,'_> {
	fn default() -> Self {
		CityState {
			coord: 0,
			gate_loc: 0,
			wall_coords: Vec::new(),
			damaged_wall_coords: Vec::new(),
			
			city_ul: Coord {y: 0, x: 0},
			city_lr: Coord {y: 0, x: 0},
			
			ch_ind: None,
			boot_camp_ind: None,
			academy_ind: None,
			bonus_bldg_inds: Vec::new(),
			
			worker_actions: Vec::new(),
			worker_inds: Vec::new(),
			
			explorer_inds: Vec::new(),
			
			defenders: Vec::new(),
			defense_positions: Vec::new(),
			
			neighbors_possible: Neighbors::N
}}}

impl Default for BarbarianState {
	fn default() -> Self{
		BarbarianState {
			attacker_inds: Vec::new(),
			defender_inds: Vec::new(),
			camp_ind: 0
}}}

impl Default for MapEx<'_,'_,'_,'_> {
	fn default() -> Self {
		MapEx {
			actual: Default::default(),
			unit_inds: None,
			bldg_ind: None,
}}}

impl Default for ActionMeta<'_,'_,'_,'_> {
	fn default() -> Self {
		ActionMeta {
			action_type: ActionType::Mv,
			actions_req: -1.,
			path_coords: Vec::new(),
			action_meta_cont: None
}}}

impl Default for ActionMetaCont {
	fn default() -> Self {
		ActionMetaCont {
			final_end_coord: Coord {x: 0, y: 0},
			checkpoint_path_coords: Vec::new()
}}}

impl Default for PersonName {
	fn default() -> Self {
		PersonName {first: String::new(), last: String::new()}
}}

impl Default for Personalization {
	fn default() -> Self {
		Self {
			color: 0,
			nm: String::new(),
			gender_female: false,
			ruler_nm: PersonName::default(),
			doctrine_advisor_nm: PersonName::default(),
			crime_advisor_nm: PersonName::default(),
			pacifism_advisor_nm: PersonName::default(),
			health_advisor_nm: PersonName::default(),
			unemployment_advisor_nm: PersonName::default(),
			motto: String::new(),
			city_nm_theme: 0,
}}}

impl Default for TechTemplate {
	fn default() -> Self {
		TechTemplate {
			id: 0,
			nm: Vec::new(),
			tech_req: None,
			research_req: 0
}}}

impl Default for TechProg {
	fn default() -> Self {TechProg::Prog(0)}
}

impl Default for LandDiscov {
	fn default() -> Self {
		LandDiscov {
			map_sz_discov: MapSz {h: 0, w: 0, sz: 0},
			map_sz: MapSz {h: 0, w: 0, sz: 0},
			frac_i: 0., frac_j: 0.,
			discovered: Vec::new()
}}}

impl Default for Bonuses {
	fn default() -> Self {
		Bonuses {
			combat_factor: 1.,
			production_factor: 1,
			gold_per_day: 0.
		}
	}
}

impl <'bt,'ut,'rt,'st>Stats<'bt,'ut,'rt,'st> {
	pub fn default(doctrine_templates: &'st Vec<DoctrineTemplate>) -> Self {
		Stats {
			id: 0,
			alive: true,
			population: 0,
			gold: 0.,
			employed: 0,
			
			doctrine_template: &doctrine_templates[0],
			locally_logged: ZoneAgnosticLocallyLogged::default_init(doctrine_templates),
			crime: 0.,
			health: 0.,
			
			resources_avail: Vec::new(),
			resources_discov_coords: Vec::new(),
			
			bonuses: Default::default(),
			
			alive_log: Vec::new(),
			population_log: Vec::new(),
			unemployed_log: Vec::new(),
			gold_log: Vec::new(),
			net_income_log: Vec::new(),
			defense_power_log: Vec::new(),
			offense_power_log: Vec::new(),
			zone_demand_log: Vec::new(),
			
			happiness_log: Vec::new(),
			crime_log: Vec::new(),
			doctrinality_log: Vec::new(),
			pacifism_log: Vec::new(),
			health_log: Vec::new(),
			
			research_per_turn_log: Vec::new(),
			research_completed_log: Vec::new(),
			
			mpd_log: Vec::new(),
			
			zone_demand_sum_map: vec!{Default::default(); ZoneType::N as usize}.into_boxed_slice(),
			
			tax_income: 0., unit_expenses: 0., bldg_expenses: 0.,
			
			techs_progress: Vec::new(),
			techs_scheduled: vec!{0;1},
			research_per_turn: 0,
			
			brigades: Vec::new(),
			sectors: Vec::new(),
			
			land_discov: Vec::new(),
			fog: Vec::new()
}}}

impl Default for StructureData {
	fn default() -> Self {
		StructureData {
			structure_type: StructureType::N,
			health: 0,
			orientation: '!'
		}
	}
}

impl Default for UnitTemplate<'_> {
	fn default() -> Self {
		UnitTemplate {id: 0, nm: Vec::new(), tech_req: None,
			resources_req: Vec::new(),
			movement_type: MovementType::Land, carry_capac: 0, 
			actions_per_turn: 0., attack_per_turn: None, 
			siege_bonus_per_turn: None, 
			repair_wall_per_turn: None, attack_range: None,
			max_health: 0, production_req: 0., char_disp: '!', upkeep: 0.
}}}

impl Default for BldgTemplate<'_,'_,'_> {
	fn default() -> Self {
		BldgTemplate {id: 0, nm: Vec::new(), 
			menu_txt: None, tech_req: None, doctrine_req: None, research_prod: 0,
			sz: ScreenSz{h: 0, w: 0, sz: 0}, print_str: String::new(),
			plot_zoomed: '!', bldg_type: BldgType::Taxable(ZoneType::N), units_producable: None,
			units_producable_txt: None, unit_production_rate: 0,
			construction_req: 0., upkeep: 0.,
			resident_max: 0, cons_max: 0, prod_max: 0,
			crime_bonus: 0., happiness_bonus: 0.,
			doctrinality_bonus: 0., pacifism_bonus: 0., health_bonus: 0.,
			job_search_bonus: 0.,
			barbarian_only: false
}}}

impl Default for ResourceTemplate {
	fn default() -> Self {
		Self {id: 0, nm: Vec::new(), tech_req: Vec::new(),
			sz: ScreenSz {h: 0, w: 0, sz: 0}, print_str: String::new(),
			plot_zoomed: '!', zone: ZoneType::N,
			zone_bonuses: Vec::new(), arability_probs: Vec::new(),
}}}

impl Default for MapSz {
	fn default() -> Self { MapSz {h: 0, w: 0, sz: 0} }
}

impl Default for Coord {
	fn default() -> Self { Self {y: 0, x: 0} } }

impl Default for Nms {
	fn default() -> Self { Self {
		cities: Vec::new(),
		units: Vec::new(),
		brigades: Vec::new(),
		sectors: Vec::new(),
		noble_houses: Vec::new(),
		females: Vec::new(),
		males: Vec::new()
}}}

impl Default for Log {
	fn default() -> Self {Self {turn: 0, val: LogType::CivCollapsed {owner_id: 0}}}}

impl Default for RelationsConfig {
	fn default() -> Self {
		Self {
			peace_treaty_min_years: 0,
			declare_war_mood_drop: 0.,
			threaten_mood_drop: 0.
		}
	}
}

impl Default for AIPersonality {
	fn default() -> Self {
		AIPersonality {
			friendliness: 0.,
			spirituality: 0.,
		}
	}
}

impl Default for RelationStatus {
	fn default() -> Self {Self::Peace {turn_started: 0}}
}

impl Default for DoctrineTemplate {
	fn default() -> Self {
		Self {
			id: 0,
			nm: Vec::new(),
			
			pre_req_ind: None,
			bldg_req: 0.,
			
			health_bonus: 0.,
			crime_bonus: 0.,
			pacifism_bonus: 0.,
			happiness_bonus: 0.,
			tax_aversion: 0.
		}
	}
}

impl Default for Fire {
	fn default() -> Self {
		Fire {
			t_updated: Instant::now(),
			layout: Vec::new()
		}
	}
}

impl Default for FireTile {
	fn default() -> Self {
		FireTile::None
	}
}

impl Default for Brigade<'_,'_,'_,'_> {
	fn default() -> Self {
		Self {
			nm: String::new(),
			unit_inds: Vec::new(),
			build_list: VecDeque::new(),
			repair_sector_walls: None
		}
	}
}

impl Default for Sector {
	fn default() -> Self {
		Self {
			nm: String::new(),
			segments: Vec::new(),
			perim_coords: PerimCoords::default()
		}
	}
}

impl Default for PerimCoords {
	fn default() -> Self {
		Self {coords: Vec::new(), turn_computed: 0}}
}

impl Default for BldgConfig {
	fn default() -> Self {
		Self {
			fire_damage_rate: 0,
			fire_repair_rate: 0,
			max_bldg_damage: 0,
			job_search_bonus_dist: 0
		}
	}
}

impl Default for ZoneAgnosticContribFracs {
	fn default() -> Self {
		Self {doctrine: 0., pacifism: 0., health: 0., unemployment: 0., crime: 0., pos_sum: 0., neg_sum: 0.}
	}
}

impl Default for ZoneAgnosticLocallyLogged {
	fn default() -> Self {
		Self {happiness_sum: 0., doctrinality_sum: Vec::new(),
			pacifism_sum: 0., contrib: ZoneAgnosticContribFracs::default()}
	}
}

impl Default for Rectangle {
	fn default() -> Self {
		Rectangle {start: Coord::default(), end: Coord::default()}
	}
}

/*impl Default for Player<'_,'_,'_,'_> {
	fn default() -> Self {
		Self {
			id: 0,
			ptype: Default::default(),
			personalization: Default::default(),
			stats: Default::default(), // requires reference, does not have default
			zone_exs: Default::default()
		}
	}
}*/

impl Default for House<'_,'_,'_,'_> {
	fn default() -> Self {
		Self {
			head_noble_pair_ind: 0,
			noble_pairs: Vec::new(),
			
			personality: Default::default(),
			
			city_state: Default::default(),
			attack_fronts: Default::default(),
			has_req_to_join: false
		}
	}
}

impl Default for Noble {
	fn default() -> Self {
		Self {
			name: Default::default(),
			personality: Default::default(),
			born_turn: 0,
			gender_female: false
		}
	}
}

impl Default for Marriage {
	fn default() -> Self {
		Self {
			partner: Default::default(),
			children: Vec::new()
		}
	}
}

impl Default for NoblePair {
	fn default() -> Self {
		Self {
			noble: Default::default(),
			marriage: None
		}
	}
}

