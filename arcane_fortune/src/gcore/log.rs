use super::*;

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
	GenericEvent { // ex. funeral, birth, marriage
		location: String,
		event_type: String,
		owner_id: usize
	},
	NoNobleSuccessor {owner_id: usize}, // house collapses because the head noble dies and does not have any children
	LeaderAssassinated {
		owner_id: usize,
		city_nm: String
	},
	HouseDeclaresIndependence {
		house_id: usize,
		empire_id: usize
	},
	KingdomJoinedEmpire {
		kingdom_id: usize,
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
			LogType::LeaderAssassinated {owner_id, ..} |
			LogType::CivCollapsed {owner_id} |
			LogType::NoNobleSuccessor {owner_id} |
			LogType::UnitDisbanded {owner_id, ..} |
			LogType::BldgDisbanded {owner_id, ..} |
			LogType::CityDisbanded {owner_id, ..} |
			LogType::Rioting {owner_id, ..} |
			LogType::RiotersAttacked {owner_id} |
			LogType::ICBMDetonation {owner_id, ..} |
			LogType::PrevailingDoctrineChanged {owner_id, ..} |
			LogType::CitizenDemand {owner_id, ..} |
			LogType::GenericEvent {owner_id, ..} |
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
			} LogType::NobleHouseJoinedEmpire {house_id, empire_id} |
			  LogType::HouseDeclaresIndependence {house_id, empire_id} => {
				ret_false!(house_id);
				ret_false!(empire_id);
			} LogType::KingdomJoinedEmpire {kingdom_id, empire_id} => {
				ret_false!(kingdom_id);
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

impl LogType {
	pub fn print(&self, print: bool, players: &Vec<Player>, doctrine_templates: &Vec<DoctrineTemplate>,
			dstate: &mut DispState) -> usize {
		let l = &dstate.local;
		let d = &mut dstate.renderer;
		macro_rules! print_civ_txt{($owner: expr, $txt: expr) => {
			set_player_color($owner, true, d);
			d.addstr($txt);
			set_player_color($owner, false, d);
		};};
		
		let txt = match self {
			LogType::CivCollapsed {owner_id} => {
				let player = &players[*owner_id];
				let personalization = &player.personalization;
				let (txt, tags) = match player.ptype {
					PlayerType::Human(_) | PlayerType::Empire(_) | PlayerType::Barbarian(_) => {
						// Civilization_collapsed: "The [] civilization has collapsed."
						(&l.Civilization_collapsed,
						 vec![KeyValColorInput {
							key: String::from("[]"),
							val: personalization.nm.clone(),
							color: personalization.color
						}])
					}
					PlayerType::Nobility(_) => {
						// house_nm: "House of []"
						// House_collapsed: "The [house_nm] has collapsed."
						
						(&l.House_collapsed,
						 vec![KeyValColorInput {
							key: String::from("[house_nm]"),
							val: l.house_nm.replace("[]", &personalization.nm),
							color: personalization.color
						}])
					}
				};
				
				if print {color_input_tags_print(txt, &tags, None, d);}
				color_input_tags_txt(txt, &tags)}
			LogType::NoNobleSuccessor {owner_id} => {
				let player = &players[*owner_id];
				let personalization = &player.personalization;
				let (txt, tags) = match player.ptype {
					PlayerType::Human(_) | PlayerType::Empire(_) | PlayerType::Barbarian(_) => {
						panic!("invalid player type for log entry");
					}
					PlayerType::Nobility(_) => {
						// house_nm: "House of []"
						// House_collapsed: "The [house_nm] has collapsed because there exists no successor."
						
						(&l.House_no_successor,
						 vec![KeyValColorInput {
							key: String::from("[house_nm]"),
							val: l.house_nm.replace("[]", &personalization.nm),
							color: personalization.color
						}])
					}
				};
				
				if print {color_input_tags_print(txt, &tags, None, d);}
				color_input_tags_txt(txt, &tags)}
			LogType::LeaderAssassinated {owner_id, city_nm} => {
				// Assassination_log_txt: "An assassination has occured near [location] by an unknown assailant!"
				let tags = vec![KeyValColorInput {
					key: String::from("[location]"),
					val: city_nm.clone(),
					color: players[*owner_id].personalization.color
				}];
				
				if print {color_input_tags_print(&l.Assassination_log_txt, &tags, None, d);}
				color_input_tags_txt(&l.Assassination_log_txt, &tags)}
			LogType::HouseDeclaresIndependence {house_id, empire_id} => {
				let house = &players[*house_id].personalization;
				let empire = &players[*empire_id].personalization;
				
				// House_declares_independence_txt: "The House of [house_nm] has declared independence from the [empire] empire!"
				let tags = vec![
					KeyValColorInput {
						key: String::from("[house_nm]"),
						val: house.nm.clone(),
						color: house.color
					},
					KeyValColorInput {
						key: String::from("[empire]"),
						val: empire.nm.clone(),
						color: empire.color
					}
				];
				
				if print {color_input_tags_print(&l.House_declares_independence_txt, &tags, None, d);}
				color_input_tags_txt(&l.House_declares_independence_txt, &tags)}
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
			LogType::KingdomJoinedEmpire {kingdom_id, empire_id} => {
				let kingdom = &players[*kingdom_id].personalization;
				let empire = &players[*empire_id].personalization;
				
				// kingdom_joined_empire: "[kingdom_nm] has joined as kingdom of the [empire_nm] empire."
				let tags = vec![
					KeyValColorInput {
						key: String::from("[kingdom_nm]"),
						val: kingdom.nm.clone(),
						color: kingdom.color
					},
					KeyValColorInput {
						key: String::from("[empire_nm]"),
						val: empire.nm.clone(),
						color: empire.color
					}];
				if print {
					color_input_tags_print(&l.kingdom_joined_empire, &tags, None, d);
				}
				
				color_input_tags_txt(&l.kingdom_joined_empire, &tags)}
			LogType::NobleHouseJoinedEmpire {house_id, empire_id} => {
				let house = &players[*house_id].personalization;
				let empire = &players[*empire_id].personalization;
				
				// house_nm: "House of []"
				// house_joined_empire: "The [house_nm] has joined the [empire_nm] empire"
				let tags = vec![
					KeyValColorInput {
						key: String::from("[house_nm]"),
						val: l.house_nm.replace("[]", &house.nm),
						color: house.color
					},
					KeyValColorInput {
						key: String::from("[empire_nm]"),
						val: empire.nm.clone(),
						color: empire.color
					}];
				if print {
					color_input_tags_print(&l.house_joined_empire, &tags, None, d);
					/*d.mv(2,2);
					
					let house = &players[*house_id];
					set_player_color(house, true, d);
					d.addstr(" test2");
					set_player_color(house, false, d);
					
					//set_color(house.personalization.color,d);
					set_attr(COLOR_PAIR(house.personalization.color), d);
					//d.attron(COLOR_PAIR(house.personalization.color as i32));
					d.addstr("test");
					d.attroff(COLOR_PAIR(house.personalization.color));
					
					let house = &players[*empire_id];
					
					d.attron(COLOR_PAIR(house.personalization.color));
					//set_player_color(&players[*house_id], true, d);
					d.addstr("test");
					//set_player_color(&players[*house_id], false, d);
					d.attroff(COLOR_PAIR(house.personalization.color));
					
					set_player_color(&players[*house_id], true, d);
					d.addstr(" test2");
					set_player_color(&players[*house_id], false, d);
					d.refresh();
					loop {}*/
				}
				
				color_input_tags_txt(&l.house_joined_empire, &tags)}
			LogType::GenericEvent {owner_id, location, event_type} => {
				let owner = &players[*owner_id].personalization;
				
				// An_event_is_being_held_near: "A [event_type] is being held near [location]."
				let tags = vec![
					KeyValColorInput {
						key: String::from("[location]"),
						val: location.clone(),
						color: owner.color
					}
				];
				
				let txt = l.An_event_is_being_held_near.replace("[event_type]", event_type);
				
				if print {
					color_input_tags_print(&txt, &tags, None, d);
				}
				color_input_tags_txt(&txt, &tags)}
		
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
}

impl GameState {
	pub fn log_event(&mut self, log_type: LogType) {
		self.logs.push(Log {turn: self.turn, val: log_type});
	}
}
