use super::*;
use sdl2_lib::COLOR_PAIR;

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
	NobleHouseJoinedEmpire {
		house_id: usize,
		empire_id: usize
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
			} LogType::NobleHouseJoinedEmpire {house_id, empire_id} => {
				ret_false!(house_id);
				ret_false!(empire_id);
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

pub fn print_log(log: &LogType, print: bool, players: &Vec<Player>, doctrine_templates: &Vec<DoctrineTemplate>,
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
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" civilization has collapsed.");
			}
			format!("The {} civilization has collapsed.", players[*owner_id].personalization.nm)}
		LogType::CivDestroyed {owner_attackee_id, owner_attacker_id} => {
			if players[*owner_attackee_id].ptype.is_barbarian() {
				if print {
					d.addstr("A local ");
					print_civ_txt!(&players[*owner_attackee_id], "Barbarian");
					d.addstr(" tribe has been conquered by the ");
					print_civ_nm(&players[*owner_attacker_id], d);
					d.addstr(" civilization!");
				}
				format!("A local Barbarian tribe has been conquered by the {} civilization!", players[*owner_attacker_id].personalization.nm)
			}else{
				if print {
					d.addstr("The ");
					print_civ_nm(&players[*owner_attackee_id], d);
					d.addstr(" civilization has been conquered by the ");
					print_civ_nm(&players[*owner_attacker_id], d);
					d.addstr(" civilization!");
				}
				format!("The {} civilization has been conquered by the {} civilization!", 
						players[*owner_attackee_id].personalization.nm, players[*owner_attacker_id].personalization.nm)
			}}
		LogType::UnitDestroyed {unit_attackee_nm, unit_attacker_nm, 
				unit_attackee_type_nm, unit_attacker_type_nm,
				owner_attackee_id, owner_attacker_id} => {
			if print {
				d.addstr("The ");
				print_civ_txt!(&players[*owner_attackee_id], unit_attackee_nm);
				d.addstr(&format!(" {} Battalion has been destroyed by The ", unit_attackee_type_nm));
				print_civ_txt!(&players[*owner_attacker_id], unit_attacker_nm);
				d.addstr(&format!(" {} Battalion.", unit_attacker_type_nm));
			}
			format!("The {} {} Battalion has been destroyed by The {} {} Battalion.", unit_attackee_nm, 
				unit_attackee_type_nm, unit_attacker_nm, unit_attacker_type_nm)}
		LogType::CityCaptured {city_attackee_nm, owner_attackee_id, owner_attacker_id} => {
			if print {
				print_civ_txt!(&players[*owner_attackee_id], city_attackee_nm);
				d.addstr(" has been captured by the ");
				print_civ_nm(&players[*owner_attacker_id], d);
				d.addstr(" civilization!");
			}
			format!("{} has been captured by the {} civilization!", city_attackee_nm, players[*owner_attacker_id].personalization.nm)}
		LogType::UnitDisbanded {owner_id, unit_nm, unit_type_nm} => {
			if print {
				d.addstr("The ");
				print_civ_txt!(&players[*owner_id], unit_nm);
				
				// ex. keep `ICBM` uppercase
				if unit_type_nm.to_uppercase() == *unit_type_nm {
					d.addstr(&format!(" {} battalion of the ", unit_type_nm));
				}else{
					d.addstr(&format!(" {} battalion of the ", unit_type_nm.to_lowercase()));
				}
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" civilization has been disbanded due to budgetary incompetence.");
			}
			format!("The {} {} battalion of the {} civilization has been disbanded due to budgetary incompetence.",
					unit_nm, unit_type_nm.to_lowercase(), players[*owner_id].personalization.nm)}
		LogType::BldgDisbanded {owner_id, bldg_nm} => {
			if print {
				d.addstr(&format!("A {} run by the ", bldg_nm.to_lowercase()));
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" civilization has been demolished due to severe financial mismanagement.");
			}
			format!("A {} run by the {} civilization has been demolished due to severe financial mismanagement.",
					bldg_nm.to_lowercase(), players[*owner_id].personalization.nm)}
		LogType::CityDisbanded {owner_id, city_nm} => {
			if print {
				print_civ_txt!(&players[*owner_id], city_nm);
				d.addstr(" of the ");
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" civilization has been ruined due to gross budgetary incompetence.");
			}
			format!("{} of the {} civilization has been ruined due to gross budgetary incompetence.", city_nm, players[*owner_id].personalization.nm)}
		LogType::CityDestroyed {city_attackee_nm, owner_attackee_id, owner_attacker_id} => {
			if print {
				print_civ_txt!(&players[*owner_attackee_id], city_attackee_nm);
				d.addstr(" has been destroyed by the ");
				print_civ_nm(&players[*owner_attacker_id], d);
				d.addstr(" civilization!");
			}
			format!("{} has been destroyed by the {} civilization!", city_attackee_nm, players[*owner_attacker_id].personalization.nm)}
		LogType::Rioting {city_nm, owner_id} => {
			if print {
				d.addstr("Rioting has broken out in ");
				print_civ_txt!(&players[*owner_id], city_nm);
				d.addch('.');
			}
			format!("Rioting has broken out in {}.", city_nm)}
		LogType::RiotersAttacked {owner_id} => {
			if print {
				d.addstr("A massacre has occured in ");
				print_civ_nm(&players[*owner_id], d);
				d.addch('.');
			}
			format!("A massacre has occured in {}.", players[*owner_id].personalization.nm)}

		LogType::CityFounded {owner_id, city_nm} => {
			if print {
				print_civ_txt!(&players[*owner_id], city_nm);
				d.addstr(" has been founded by the ");
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" civilization.");
			}
			format!("{} has been founded by the {} civilization.", city_nm, players[*owner_id].personalization.nm)}
		LogType::CivDiscov {discover_id, discovee_id} => {
			if players[*discover_id].ptype.is_barbarian() {
				if print {
					d.addstr("Local ");
					print_civ_txt!(&players[*discover_id], "barbarians");
					d.addstr(" have discovered the ");
					print_civ_nm(&players[*discovee_id], d);
					d.addstr(" civilization.");
				}
				format!("Local barbarians have discovered the {} civilization.", players[*discovee_id].personalization.nm)
			
			}else if players[*discovee_id].ptype.is_barbarian() {
				if print {
					d.addstr("The ");
					print_civ_nm(&players[*discover_id], d);
					d.addstr(" civilization has discovered local ");
					print_civ_txt!(&players[*discovee_id], "barbarians");
					d.addstr(".");
				}
				format!("The {} civilization has discovered local barbarians.", players[*discover_id].personalization.nm)
			}else{
				if print {
					d.addstr("The ");
					print_civ_nm(&players[*discover_id], d);
					d.addstr(" civilization has discovered the ");
					print_civ_nm(&players[*discovee_id], d);
					d.addstr(" civilization.");
				}
				format!("The {} civilization has discovered the {} civilization.", players[*discover_id].personalization.nm, players[*discovee_id].personalization.nm)
			}}
		LogType::UnitAttacked {unit_attackee_nm, unit_attacker_nm, 
				unit_attackee_type_nm, unit_attacker_type_nm,
				owner_attackee_id, owner_attacker_id} => {
			if print {
				d.addstr("The ");
				print_civ_txt!(&players[*owner_attackee_id], unit_attackee_nm);
				d.addstr(&format!(" {} Battalion has been attacked by The ", unit_attackee_type_nm));
				print_civ_txt!(&players[*owner_attacker_id], unit_attacker_nm);
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
				print_civ_nm(&players[*owner_attackee_id], d);
				d.addstr(" has been attacked by The ");
				print_civ_txt!(&players[*owner_attacker_id], unit_attacker_nm);
				d.addstr(&format!(" {} Battalion.", unit_attacker_type_nm));
			}
			format!("A {} owned by the {} has been attacked by The {} {} Battalion.", structure_nm, 
					players[*owner_attackee_id].personalization.nm, unit_attacker_nm, unit_attacker_type_nm)}

		LogType::WarDeclaration {owner_attackee_id, owner_attacker_id} => {
			if print {
				d.addstr("The ");
				print_civ_nm(&players[*owner_attacker_id], d);
				d.addstr(" civilization has declared war on the ");
				print_civ_nm(&players[*owner_attackee_id], d);
				d.addstr(" civilization!");
			}
			format!("The {} civilization has declared war on the {} civilization!",
					players[*owner_attacker_id].personalization.nm, players[*owner_attackee_id].personalization.nm)}
		LogType::PeaceDeclaration {owner1_id, owner2_id} => {
			if print {
				d.addstr("A peace treaty between the ");
				print_civ_nm(&players[*owner1_id], d);
				d.addstr(" and the ");
				print_civ_nm(&players[*owner2_id], d);
				d.addstr(" civilizations has been signed.");
			}
			format!("A peace treaty between the {} and the {} civilizations has been signed.",
					players[*owner1_id].personalization.nm, players[*owner2_id].personalization.nm)}
		LogType::ICBMDetonation {owner_id} => {
			if print {
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" has detonated an ICBM!");
			}
			format!("{} has detonated an ICBM!", players[*owner_id].personalization.nm)}
		LogType::PrevailingDoctrineChanged {owner_id, doctrine_to_id, ..} => {
			if print {
				let owner = &players[*owner_id];
				print_civ_txt!(owner, &format!("{}'s", owner.personalization.nm));
				d.addstr(&format!(" people now embrace a new doctrine: {}", doctrine_templates[*doctrine_to_id].nm[l.lang_ind]));
			}
			format!("{}'s people now embrace a new doctrine: {}", players[*owner_id].personalization.nm,
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
				print_civ_nm(&players[*owner_id], d);
				d.addstr(" demand ");
				d.addstr(x);
				d.addch('.');
			}
			format!("The people of {} demand {}.", players[*owner_id].personalization.nm, x)}
		LogType::NobleHouseJoinedEmpire {house_id, empire_id} => {
			let house = &players[*house_id].personalization;
			let empire = &players[*empire_id].personalization;
			
			// house_nm: "House of []"
			// house_joined_empire: "The [house_nm] has joined the [empire_nm] empire"
			let tags = vec![
				KeyValColor {
					key: String::from("[house_nm]"),
					val: l.house_nm.replace("[]", &house.nm),
					attr: COLOR_PAIR(house.color)
				},
				KeyValColor {
					key: String::from("[empire_nm]"),
					val: empire.nm.clone(),
					attr: COLOR_PAIR(empire.color)
				}];
			if print {
				color_tags_print(&l.house_joined_empire, &tags, None, d);
			}
			color_tags_txt(&l.house_joined_empire, &tags)}
		LogType::Debug {txt, owner_id} => {
			if print {
				if let Some(id) = owner_id {
					print_civ_txt!(&players[*id], &format!("{}: {}", players[*id].personalization.nm, txt));
				}else{
					d.addstr(&txt);
				}
			}
			if let Some(id) = owner_id {
				format!("{}: {}", players[*id].personalization.nm, txt)
			}else{
				txt.clone()
			}}
	};
	
	txt.len()
}

