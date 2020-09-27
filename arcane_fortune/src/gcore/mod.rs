use crate::map::*;
use crate::units::*;
use crate::buildings::*;
use crate::movement::*;
use crate::disp::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::zones::{HappinessCategory, PacifismMilitarism};
use crate::disp_lib::DispState;
use crate::saving::*;
use crate::disp_lib::endwin;
use crate::ai::AIState;
use crate::localization::Localization;

pub mod profiling;
pub mod hashing;
pub mod rand;
pub mod sectors; pub use sectors::*;
pub mod brigades; pub use brigades::*;
mod end_turn;
mod non_menu_keys;
mod relations;
mod economy;
mod difficulty;
mod testing;

pub use profiling::*;
pub use hashing::*;
pub use rand::*;
pub use end_turn::*;
pub use non_menu_keys::*;
pub use relations::*;
pub use economy::*;
pub use difficulty::*;
pub use testing::*;

#[derive(Clone, PartialEq)]
pub enum LogType {
	CivCollapsed { owner_id: usize },
	CivDestroyed {
		owner_attackee_id: usize,
		owner_attacker_id: usize
	},
	UnitDestroyed { // i.e. attacked
		unit_attackee_nm: String,
		unit_attacker_nm: String,
		
		unit_attackee_type_nm: String,
		unit_attacker_type_nm: String,
		
		owner_attackee_id: usize,
		owner_attacker_id: usize,
	},
	CityCaptured { // i.e. attacked
		city_attackee_nm: String,
		
		owner_attackee_id: usize,
		owner_attacker_id: usize
	},
	UnitDisbanded { // due to budget
		owner_id: usize,
		unit_nm: String,
		unit_type_nm: String
	},
	BldgDisbanded { // due to budget
		owner_id: usize,
		bldg_nm: String
	},
	CityDisbanded { // due to budget
		owner_id: usize,
		city_nm: String
	},
	Rioting {
		owner_id: usize,
		city_nm: String
	},
	CityDestroyed { // due to bomb, possibly other causes
		city_attackee_nm: String,
		
		owner_attackee_id: usize,
		owner_attacker_id: usize
	},
	CityFounded {
		owner_id: usize,
		city_nm: String
	},
	CivDiscov {
		discover_id: usize,
		discovee_id: usize
	},
	UnitAttacked { // right now only logged for current player
		unit_attackee_nm: String,
		unit_attacker_nm: String,
		
		unit_attackee_type_nm: String,
		unit_attacker_type_nm: String,

		owner_attackee_id: usize,
		owner_attacker_id: usize
	},
	StructureAttacked { // right now only logged for current player
		structure_coord: u64,
		
		unit_attacker_nm: String,
		unit_attacker_type_nm: String,
		
		structure_type: StructureType,
		
		owner_attackee_id: usize,
		owner_attacker_id: usize
	},
	WarDeclaration {
		owner_attackee_id: usize,
		owner_attacker_id: usize
	},
	PeaceDeclaration {
		owner1_id: usize,
		owner2_id: usize
	},
	ICBMDetonation {
		owner_id: usize
	},
	PrevailingDoctrineChanged {
		owner_id: usize,
		doctrine_frm_id: usize,
		doctrine_to_id: usize
	},
	RiotersAttacked {
		owner_id: usize
	},
	CitizenDemand {
		owner_id: usize,
		reason: HappinessCategory
	},
	Debug {
		owner_id: Option<usize>,
		txt: String
	}
}

#[derive(Clone, PartialEq)]
pub struct Log {
	pub turn: usize,
	pub val: LogType
}

impl_saving!{Log {turn, val}}

impl Log {
	pub fn visible(&self, player_id: usize, relations: &Relations) -> bool {
		macro_rules! ret_false {($id: expr) => {
			if !relations.discovered(player_id, $id) {return false;}
		};};
		
		match self.val {
			LogType::CivCollapsed {owner_id} |
			LogType::UnitDisbanded {owner_id, ..} |
			LogType::BldgDisbanded {owner_id, ..} |
			LogType::CityDisbanded {owner_id, ..} |
			LogType::Rioting {owner_id, ..} |
			LogType::RiotersAttacked {owner_id} |
			LogType::ICBMDetonation {owner_id, ..} |
			LogType::PrevailingDoctrineChanged {owner_id, ..} |
			LogType::CitizenDemand {owner_id, ..} |
			LogType::CityFounded {owner_id, ..} => {
				ret_false!(owner_id);
			} LogType::CityCaptured {owner_attackee_id, owner_attacker_id, ..} |
			  LogType::CityDestroyed {owner_attackee_id, owner_attacker_id, ..} |
			  LogType::CivDestroyed {owner_attackee_id, owner_attacker_id} |
			  LogType::UnitDestroyed {owner_attackee_id, owner_attacker_id, ..} |
			  LogType::UnitAttacked {owner_attackee_id, owner_attacker_id, ..} |
			  LogType::StructureAttacked {owner_attackee_id, owner_attacker_id, ..} |
			  LogType::WarDeclaration {owner_attackee_id, owner_attacker_id} => {
				ret_false!(owner_attackee_id);
				ret_false!(owner_attacker_id);
			} LogType::PeaceDeclaration {owner1_id, owner2_id} => {
				ret_false!(owner1_id);
				ret_false!(owner2_id);
			} LogType::CivDiscov {discover_id, discovee_id} => {
				ret_false!(discover_id);
				ret_false!(discovee_id);
			} LogType::Debug {..} => {return true;}
		}
		true
	}
}

pub fn dbg_log(txt: &str, owner_id: SmSvType, logs: &mut Vec<Log>, turn: usize) {
	logs.push(Log {turn, val: LogType::Debug{
			txt: txt.to_string(),
			owner_id: Some(owner_id as usize)}
	});
}

pub fn print_log(log: &LogType, print: bool, owners: &Vec<Owner>, doctrine_templates: &Vec<DoctrineTemplate>,
		l: &Localization, d: &mut DispState) -> usize {
	macro_rules! print_civ_txt{($owner: expr, $txt: expr) => {
		set_player_color($owner, true, d);
		d.addstr($txt);
		set_player_color($owner, false, d);
	};};
	
	let txt = match log {
		LogType::CivCollapsed {owner_id} => {
			if print {
				d.addstr("The ");
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" civilization has collapsed.");
			}
			format!("The {} civilization has collapsed.", owners[*owner_id].nm)}
		LogType::CivDestroyed {owner_attackee_id, owner_attacker_id} => {
			if owners[*owner_attackee_id].player_type == PlayerType::Barbarian {
				if print {
					d.addstr("A local ");
					print_civ_txt!(&owners[*owner_attackee_id], "Barbarian");
					d.addstr(" tribe has been conquered by the ");
					print_civ_nm(&owners[*owner_attacker_id], d);
					d.addstr(" civilization!");
				}
				format!("A local Barbarian tribe has been conquered by the {} civilization!", owners[*owner_attacker_id].nm)
			}else{
				if print {
					d.addstr("The ");
					print_civ_nm(&owners[*owner_attackee_id], d);
					d.addstr(" civilization has been conquered by the ");
					print_civ_nm(&owners[*owner_attacker_id], d);
					d.addstr(" civilization!");
				}
				format!("The {} civilization has been conquered by the {} civilization!", 
						owners[*owner_attackee_id].nm, owners[*owner_attacker_id].nm)
			}}
		LogType::UnitDestroyed {unit_attackee_nm, unit_attacker_nm, 
				unit_attackee_type_nm, unit_attacker_type_nm,
				owner_attackee_id, owner_attacker_id} => {
			if print {
				d.addstr("The ");
				print_civ_txt!(&owners[*owner_attackee_id], unit_attackee_nm);
				d.addstr(&format!(" {} Battalion has been destroyed by The ", unit_attackee_type_nm));
				print_civ_txt!(&owners[*owner_attacker_id], unit_attacker_nm);
				d.addstr(&format!(" {} Battalion.", unit_attacker_type_nm));
			}
			format!("The {} {} Battalion has been destroyed by The {} {} Battalion.", unit_attackee_nm, 
				unit_attackee_type_nm, unit_attacker_nm, unit_attacker_type_nm)}
		LogType::CityCaptured {city_attackee_nm, owner_attackee_id, owner_attacker_id} => {
			if print {
				print_civ_txt!(&owners[*owner_attackee_id], city_attackee_nm);
				d.addstr(" has been captured by the ");
				print_civ_nm(&owners[*owner_attacker_id], d);
				d.addstr(" civilization!");
			}
			format!("{} has been captured by the {} civilization!", city_attackee_nm, owners[*owner_attacker_id].nm)}
		LogType::UnitDisbanded {owner_id, unit_nm, unit_type_nm} => {
			if print {
				d.addstr("The ");
				print_civ_txt!(&owners[*owner_id], unit_nm);
				
				// ex. keep `ICBM` uppercase
				if unit_type_nm.to_uppercase() == *unit_type_nm {
					d.addstr(&format!(" {} battalion of the ", unit_type_nm));
				}else{
					d.addstr(&format!(" {} battalion of the ", unit_type_nm.to_lowercase()));
				}
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" civilization has been disbanded due to budgetary incompetence.");
			}
			format!("The {} {} battalion of the {} civilization has been disbanded due to budgetary incompetence.",
					unit_nm, unit_type_nm.to_lowercase(), owners[*owner_id].nm)}
		LogType::BldgDisbanded {owner_id, bldg_nm} => {
			if print {
				d.addstr(&format!("A {} run by the ", bldg_nm.to_lowercase()));
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" civilization has been demolished due to severe financial mismanagement.");
			}
			format!("A {} run by the {} civilization has been demolished due to severe financial mismanagement.",
					bldg_nm.to_lowercase(), owners[*owner_id].nm)}
		LogType::CityDisbanded {owner_id, city_nm} => {
			if print {
				print_civ_txt!(&owners[*owner_id], city_nm);
				d.addstr(" of the ");
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" civilization has been ruined due to gross budgetary incompetence.");
			}
			format!("{} of the {} civilization has been ruined due to gross budgetary incompetence.", city_nm, owners[*owner_id].nm)}
		LogType::CityDestroyed {city_attackee_nm, owner_attackee_id, owner_attacker_id} => {
			if print {
				print_civ_txt!(&owners[*owner_attackee_id], city_attackee_nm);
				d.addstr(" has been destroyed by the ");
				print_civ_nm(&owners[*owner_attacker_id], d);
				d.addstr(" civilization!");
			}
			format!("{} has been destroyed by the {} civilization!", city_attackee_nm, owners[*owner_attacker_id].nm)}
		LogType::Rioting {city_nm, owner_id} => {
			if print {
				d.addstr("Rioting has broken out in ");
				print_civ_txt!(&owners[*owner_id], city_nm);
				d.addch('.');
			}
			format!("Rioting has broken out in {}.", city_nm)}
		LogType::RiotersAttacked {owner_id} => {
			if print {
				d.addstr("A massacre has occured in ");
				print_civ_nm(&owners[*owner_id], d);
				d.addch('.');
			}
			format!("A massacre has occured in {}.", owners[*owner_id].nm)}

		LogType::CityFounded {owner_id, city_nm} => {
			if print {
				print_civ_txt!(&owners[*owner_id], city_nm);
				d.addstr(" has been founded by the ");
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" civilization.");
			}
			format!("{} has been founded by the {} civilization.", city_nm, owners[*owner_id].nm)}
		LogType::CivDiscov {discover_id, discovee_id} => {
			if owners[*discover_id].player_type == PlayerType::Barbarian {
				if print {
					d.addstr("Local ");
					print_civ_txt!(&owners[*discover_id], "barbarians");
					d.addstr(" have discovered the ");
					print_civ_nm(&owners[*discovee_id], d);
					d.addstr(" civilization.");
				}
				format!("Local barbarians have discovered the {} civilization.", owners[*discovee_id].nm)
			
			}else if owners[*discovee_id].player_type == PlayerType::Barbarian {
				if print {
					d.addstr("The ");
					print_civ_nm(&owners[*discover_id], d);
					d.addstr(" civilization has discovered local ");
					print_civ_txt!(&owners[*discovee_id], "barbarians");
					d.addstr(".");
				}
				format!("The {} civilization has discovered local barbarians.", owners[*discover_id].nm)
			}else{
				if print {
					d.addstr("The ");
					print_civ_nm(&owners[*discover_id], d);
					d.addstr(" civilization has discovered the ");
					print_civ_nm(&owners[*discovee_id], d);
					d.addstr(" civilization.");
				}
				format!("The {} civilization has discovered the {} civilization.", owners[*discover_id].nm, owners[*discovee_id].nm)
			}}
		LogType::UnitAttacked {unit_attackee_nm, unit_attacker_nm, 
				unit_attackee_type_nm, unit_attacker_type_nm,
				owner_attackee_id, owner_attacker_id} => {
			if print {
				d.addstr("The ");
				print_civ_txt!(&owners[*owner_attackee_id], unit_attackee_nm);
				d.addstr(&format!(" {} Battalion has been attacked by The ", unit_attackee_type_nm));
				print_civ_txt!(&owners[*owner_attacker_id], unit_attacker_nm);
				d.addstr(&format!(" {} Battalion.", unit_attacker_type_nm));
			}
			format!("The {} {} Battalion has been attacked by The {} {} Battalion.", unit_attackee_nm, 
					unit_attackee_type_nm, unit_attacker_nm, unit_attacker_type_nm)}
		LogType::StructureAttacked {unit_attacker_nm, 
				unit_attacker_type_nm, structure_type,
				owner_attackee_id, owner_attacker_id, structure_coord: _} => {
			let structure_nm = match structure_type {
				StructureType::Wall => "wall",
				StructureType::Road => "road",
				StructureType::Gate => "gate",
				StructureType::N => {panicq!("unknown structure");}
			};
			
			if print {
				d.addstr(&format!("A {} owned by the ", structure_nm));
				print_civ_nm(&owners[*owner_attackee_id], d);
				d.addstr(" has been attacked by The ");
				print_civ_txt!(&owners[*owner_attacker_id], unit_attacker_nm);
				d.addstr(&format!(" {} Battalion.", unit_attacker_type_nm));
			}
			format!("A {} owned by the {} has been attacked by The {} {} Battalion.", structure_nm, 
					owners[*owner_attackee_id].nm, unit_attacker_nm, unit_attacker_type_nm)}

		LogType::WarDeclaration {owner_attackee_id, owner_attacker_id} => {
			if print {
				d.addstr("The ");
				print_civ_nm(&owners[*owner_attacker_id], d);
				d.addstr(" civilization has declared war on the ");
				print_civ_nm(&owners[*owner_attackee_id], d);
				d.addstr(" civilization!");
			}
			format!("The {} civilization has declared war on the {} civilization!",
					owners[*owner_attacker_id].nm, owners[*owner_attackee_id].nm)}
		LogType::PeaceDeclaration {owner1_id, owner2_id} => {
			if print {
				d.addstr("A peace treaty between the ");
				print_civ_nm(&owners[*owner1_id], d);
				d.addstr(" and the ");
				print_civ_nm(&owners[*owner2_id], d);
				d.addstr(" civilizations has been signed.");
			}
			format!("A peace treaty between the {} and the {} civilizations has been signed.",
					owners[*owner1_id].nm, owners[*owner2_id].nm)}
		LogType::ICBMDetonation {owner_id} => {
			if print {
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" has detonated an ICBM!");
			}
			format!("{} has detonated an ICBM!", owners[*owner_id].nm)}
		LogType::PrevailingDoctrineChanged {owner_id, doctrine_to_id, ..} => {
			if print {
				let owner = &owners[*owner_id];
				print_civ_txt!(owner, &format!("{}'s", owner.nm));
				d.addstr(&format!(" people now embrace a new doctrine: {}", doctrine_templates[*doctrine_to_id].nm[l.lang_ind]));
			}
			format!("{}'s people now embrace a new doctrine: {}", owners[*owner_id].nm,
					doctrine_templates[*doctrine_to_id].nm[l.lang_ind])}
		LogType::CitizenDemand {owner_id, reason} => {
			// The people of [] demand [X]
			let x = match reason {
				HappinessCategory::Doctrine => {"to have their empire run with a higher doctrine in mind"}
				HappinessCategory::PacifismOrMilitarism(PacifismMilitarism::Pacifism) => {
					"to have more pacifism in the empire"
				}
				HappinessCategory::PacifismOrMilitarism(PacifismMilitarism::Militarism) => {
					"to have more militarism in the empire"
				}
				HappinessCategory::PacifismOrMilitarism(PacifismMilitarism::N) => {panicq!("invalid value");}
				HappinessCategory::Health => {"the egregious health conditions in which they live fixed"}
				HappinessCategory::Unemployment => {"the economy, specifically unemployment, be improved"}
				HappinessCategory::Crime => {"crime be reigned in"}
			};
			
			if print {
				d.addstr("The people of ");
				print_civ_nm(&owners[*owner_id], d);
				d.addstr(" demand ");
				d.addstr(x);
				d.addch('.');
			}
			format!("The people of {} demand {}.", owners[*owner_id].nm, x)}
		LogType::Debug {txt, owner_id} => {
			if print {
				if let Some(id) = owner_id {
					print_civ_txt!(&owners[*id], &format!("{}: {}", owners[*id].nm, txt));
				}else{
					d.addstr(&txt);
				}
			}
			if let Some(id) = owner_id {
				format!("{}: {}", owners[*id].nm, txt)
			}else{
				txt.clone()
			}}
	};
	
	txt.len()
}

/*#[inline]
pub fn approx_eq(a: f32, b: f32) -> bool {
	((a + std::f32::EPSILON) >= b) && ((a - std::f32::EPSILON) <= b)
}*/

#[inline]
pub fn approx_eq_tol(a: f32, b: f32, tol: f32) -> bool {
	((a + tol) >= b) && ((a - tol) <= b)
}

fn in_debt(pstats: &Stats) -> bool {
	let assets = pstats.gold + pstats.net_income();
	assets < 0. && !approx_eq_tol(assets, 0., 0.001)
}

impl Stats<'_,'_,'_,'_> {
	pub fn net_income(&self) -> f32 {
		self.bonuses.gold_per_day + self.tax_income - self.unit_expenses - self.bldg_expenses
	}
}

// either from collapse or take-over
pub fn civ_destroyed(owner_id: usize, stats: &mut Vec<Stats>, ai_states: &mut Vec<Option<AIState>>,
		relations: &mut Relations, iface_settings: &mut IfaceSettings, turn: usize, d: &mut DispState) {
	let pstats = &mut stats[owner_id];
	pstats.alive = false;
	pstats.gold = 0.;
	
	for war_enemy in relations.at_war_with(owner_id) {
		relations.declare_peace_wo_logging(war_enemy, owner_id, turn);
	}
	
	// human player end game
	if owner_id == iface_settings.cur_player as usize {
		iface_settings.player_end_game(relations, d);
	
	// ai end game -- clear attack fronts
	}else if let Some(ai_state) = &mut ai_states[owner_id] {
		//debug_assertq!(ai_state.city_states.len() == 0, "n_cities: {}", ai_state.city_states.len());
		// ^ a city_state could exist, but \/ none should have a city hall
		debug_assertq!(!ai_state.city_states.iter().any(|cs| !cs.ch_ind.is_none()));
		
		ai_state.city_states = Vec::new();
		ai_state.attack_fronts.vals = Vec::new();
	}
}

// for destroying a civ
pub fn rm_player_zones<'bt,'ut,'rt,'dt>(owner_id: usize, bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>,
		stats: &mut Vec<Stats>, doctrine_templates: &Vec<DoctrineTemplate>, exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, map_data: &mut MapData,
		zone_exs_owners: &mut Vec<HashedMapZoneEx>, owners: &Vec<Owner>, map_sz: MapSz) {
	// gather ex coords
	let exf = exs.last().unwrap();
	let mut coords = Vec::with_capacity(exf.len());
	for coord in exf.keys() {
		coords.push(*coord);
	}
	
	// rm zones
	let mut coords_rmd = Vec::with_capacity(exf.len());
	for coord in coords {
		let ex = exs.last_mut().unwrap().get_mut(&coord).unwrap();
		if ex.actual.owner_id == Some(owner_id as SmSvType) {
			ex.actual.rm_zone(coord, zone_exs_owners, stats, doctrine_templates, map_sz);
			coords_rmd.push(coord);
		}
	}
	
	// update map
	for coord in coords_rmd {
		compute_zooms_coord(coord, RecompType::Bldgs(bldgs, bldg_templates, zone_exs_owners), map_data, exs, owners);
	}
}

pub fn worker_inds(unit_inds: &Vec<usize>, units: &Vec<Unit>) -> Vec<usize> {
	let mut unit_inds_keep = Vec::with_capacity(unit_inds.len());
	for unit_ind in unit_inds {
		if units[*unit_ind].template.nm[0] == WORKER_NM {
			unit_inds_keep.push(*unit_ind);
		}
	}
	unit_inds_keep
}

