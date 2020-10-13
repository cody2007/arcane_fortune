use super::{Log, LogType};
use crate::saving::*;
use crate::resources::*;
use crate::units::*;
use crate::buildings::*;
use crate::disp_lib::*;
use crate::disp::*;
use crate::disp::menus::{update_menu_indicators, OptionsUI};
use crate::config_load::{config_parse, find_req_key_parse, read_file};
use crate::nn::{TxtCategory, TxtPrinter};
use crate::ai::*;
use crate::player::{PlayerType, Player};
use crate::gcore::XorState;
use crate::doctrine::DoctrineTemplate;

#[derive(Clone, PartialEq)]
pub struct RelationsConfig {
	pub peace_treaty_min_years: usize,
	pub declare_war_mood_drop: f32,
	pub threaten_mood_drop: f32
}

impl_saving!{RelationsConfig {peace_treaty_min_years, declare_war_mood_drop, threaten_mood_drop}}

pub fn init_relations_config() -> RelationsConfig {
	const RELATIONS_CONFIG_FILE: &str = "config/relations.txt";
	let key_sets = config_parse(read_file(RELATIONS_CONFIG_FILE));
	assertq!(key_sets.len() > 0, "no entries found in {}", RELATIONS_CONFIG_FILE);
	let key_sets_f = &key_sets[0];
	
	RelationsConfig {
		peace_treaty_min_years: find_req_key_parse("peace_treaty_min_years", key_sets_f),
		declare_war_mood_drop: find_req_key_parse("declare_war_mood_drop", key_sets_f),
		threaten_mood_drop: find_req_key_parse("threaten_mood_drop", key_sets_f)
	}
}

#[derive(Clone, PartialEq)]
pub enum RelationStatus {
	Peace {turn_started: usize},
	War {turn_started: usize},
	Fiefdom {turn_joined: usize}
}

#[derive(Clone, PartialEq)]
pub struct Relations {
	// square matrices only the upper triangle contains relevant values
	// i.e., it should be indexed (i,j) where i < j
	relation_status: Vec<RelationStatus>, // use relations.peace_treaty_expiry() to check if war can be declared
	discov: Vec<bool>,
	
	mood_toward: Vec<f32>, // (i,j) owner i's view toward owner j. this matrix does *not* need to be symmetric, unlike above
	
	n_owners: usize,
	
	pub config: RelationsConfig
}

impl_saving!{Relations{relation_status, discov, mood_toward, n_owners, config}}

impl RelationStatus {
	pub fn is_peace(&self) -> bool {
		match self {
			RelationStatus::Peace {..} => true,
			RelationStatus::War {..} => false,
			RelationStatus::Fiefdom {..} => true
		}
	}
}

fn order_indices(owner1: usize, owner2: usize) -> (usize, usize) {
	debug_assertq!(owner1 != owner2);
	if owner1 < owner2 {
		(owner1, owner2)
	}else{
		(owner2, owner1)
	}
}

impl Relations {
	pub fn war_lengths(&self, owner1: usize, turn: usize) -> usize {
		let mut lengths = 0;
		for owner2 in self.at_war_with(owner1) {
			lengths += turn - self.turn_war_started(owner1, owner2);
		}
		lengths
	}
	
	pub fn at_war(&self, owner1: usize, owner2: usize) -> bool {
		let (omin, omax) = order_indices(owner1, owner2);
		!self.relation_status[omin*self.n_owners + omax].is_peace()
	}
	
	pub fn fiefdom(&self, owner1: usize, owner2: usize) -> bool {
		let (omin, omax) = order_indices(owner1, owner2);
		match self.relation_status[omin*self.n_owners + omax] {
			RelationStatus::Fiefdom {..} => true,
			RelationStatus::War {..} | RelationStatus::Peace {..} => false
		}
	}

	fn peace_treaty_expiry_turn(&self, turn_started: usize) -> usize {
		self.config.peace_treaty_min_years * TURNS_PER_YEAR + turn_started
	}
	
	// returns number of years until expiration, none if no peace treaty
	pub fn peace_treaty_turns_remaining(&self, owner1: usize, owner2: usize, turn: usize) -> Option<usize> {
		let (omin, omax) = order_indices(owner1, owner2);
		
		match self.relation_status[omin*self.n_owners + omax] {
			RelationStatus::War {..} | RelationStatus::Fiefdom {..} => None,
			RelationStatus::Peace {turn_started} => {
				// peace treaty in effect; see declare_war() for similar computation
				let expiry_turn = self.peace_treaty_expiry_turn(turn_started);
				if turn_started != 0 && expiry_turn > turn {
					Some(expiry_turn - turn)
				}else{
					None
				}
			}
		}
	}
	
	pub fn turn_war_started(&self, owner1: usize, owner2: usize) -> usize {
		let (omin, omax) = order_indices(owner1, owner2);
		
		match self.relation_status[omin*self.n_owners + omax] {
			RelationStatus::War {turn_started} => {turn_started}
			RelationStatus::Peace {..} | RelationStatus::Fiefdom {..} => {panicq!("war not active")}
		}
	}
	
	// all owners owner1 is at war with
	pub fn at_war_with(&self, owner1: usize) -> Vec<usize> {
		let mut wars = Vec::with_capacity(self.n_owners);
		for owner_id in (0..self.n_owners).filter(|&o| o != owner1) {
			if self.at_war(owner1, owner_id) {
				wars.push(owner_id);
			}
		}
		wars
	}
	
	pub fn noble_houses(&self, owner1: usize) -> Vec<usize> {
		let mut houses = Vec::with_capacity(self.n_owners);
		for owner_id in (0..self.n_owners).filter(|&o| o != owner1) {
			if self.fiefdom(owner1, owner_id) {
				houses.push(owner_id);
			}
		}
		houses
	}
	
	// owner 1 threatens owner2
	// note: using an ICBM calls this multiple times to lower mood
	pub fn threaten(&mut self, owner1: usize, owner2: usize) {
		// owner2's mood toward owner1's drops
		self.mood_toward[owner2*self.n_owners + owner1] -= self.config.threaten_mood_drop;
	}
	
	// owner 1 declares war on owner2
	// returns true if already at war or war can be declared, false if war cannot be declared
	pub fn declare_war(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, players: &Vec<Player>, turn: usize,
			iface_settings: &mut IfaceSettings, cur_ai_player_is_paused: Option<bool>,
			disp_settings: &DispSettings, menu_options: &mut OptionsUI, rng: &mut XorState, d: &mut DispState) -> bool {
		self.discover_civ(owner1, owner2, logs, turn);
		
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_ind = omin*self.n_owners + omax;
		
		// log & disable auto-turn
		if let RelationStatus::Peace {turn_started} = &self.relation_status[relation_ind] {
			// peace treaty in effect; see peace_treaty_expiry() for similar computation
			if *turn_started != 0 && self.peace_treaty_expiry_turn(*turn_started) > turn {
				return false;
			}
			
			self.relation_status[relation_ind] = RelationStatus::War {turn_started: turn};
			self.discov[relation_ind] = true;
			
			// owner2's mood toward owner1's drops
			self.mood_toward[owner2*self.n_owners + owner1] -= self.config.declare_war_mood_drop;
			
			logs.push(Log {turn,
					   val: LogType::WarDeclaration {
					   	owner_attacker_id: owner1,
					   	owner_attackee_id: owner2
					}
			});
			
			// disable auto-turn
			if iface_settings.interrupt_auto_turn && owner2 as SmSvType == iface_settings.cur_player {
				iface_settings.set_auto_turn(AutoTurn::Off, d);
				update_menu_indicators(menu_options, iface_settings, cur_ai_player_is_paused, disp_settings);
				d.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				
				let quote_category = TxtCategory::from_relations(self, owner1, owner2, players);
				
				iface_settings.ui_mode = UIMode::ContactEmbassyWindow {
					state: EmbassyState::DeclaredWarOn {
							owner_id: owner1,
							quote_printer: TxtPrinter::new(quote_category, rng.gen())
						 }
				};
			}
		}
		
		true
	}
	
	// owner 1 joints as noble house of owner2
	pub fn join_as_fiefdom(&mut self, owner1: usize, owner2: usize, players: &mut Vec<Player>,
			logs: &mut Vec<Log>, turn: usize) {
		self.discover_civ(owner1, owner2, logs, turn);
		
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_ind = omin*self.n_owners + omax;
		self.relation_status[relation_ind] = RelationStatus::Fiefdom {turn_joined: turn};
		
		let empire_color = players[owner2].personalization.color;
		players[owner1].personalization.color = empire_color;
		
		logs.push(Log {turn,
			val: LogType::NobleHouseJoinedEmpire {
				house_id: owner1,
				empire_id: owner2
			}
		});
	}
	
	pub fn declare_peace_wo_logging(&mut self, owner1: usize, owner2: usize, turn: usize) {
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_status = &mut self.relation_status[omin*self.n_owners + omax];
		
		debug_assertq!(!relation_status.is_peace());
		*relation_status = RelationStatus::Peace {turn_started: turn};
	}

	pub fn declare_peace(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, turn: usize) {
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_status = &mut self.relation_status[omin*self.n_owners + omax];
		
		debug_assertq!(!relation_status.is_peace());
		*relation_status = RelationStatus::Peace {turn_started: turn};
		
		logs.push(Log {turn,
				   val: LogType::PeaceDeclaration {
				   owner1_id: owner1,
				   owner2_id: owner2
				}
		});
	}
	
	pub fn discovered(&self, owner1: usize, owner2: usize) -> bool {
		if owner1 == owner2 {return true;}
		let (omin, omax) = order_indices(owner1, owner2);
		self.discov[omin*self.n_owners + omax]
	}
	
	// owner1 discovers owner2
	pub fn discover_civ(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, turn: usize) {
		if owner1 == owner2 {return;}
		let (omin, omax) = order_indices(owner1, owner2);
		let discov = &mut self.discov[omin*self.n_owners + omax];
		if !*discov {
			*discov = true;
			
			logs.push(Log {turn,
					val: LogType::CivDiscov {
						discover_id: owner1,
						discovee_id: owner2
			}});
		}
	}
	
	pub fn discover_all_civs(&mut self, owner1: usize) {
		for owner_id in (0..self.n_owners).filter(|&owner_id| owner_id != owner1) {
			let (omin, omax) = order_indices(owner1, owner_id);
			self.discov[omin*self.n_owners + omax] = true;
		}
	}
	
	pub fn new(n_owners: usize) -> Self {
		Self {
			relation_status: vec!{RelationStatus::Peace {turn_started: 0}; n_owners*n_owners},
			discov: vec!{false; n_owners*n_owners},
			mood_toward: vec!{0.; n_owners*n_owners},
			n_owners,
			config: init_relations_config()
		}
	}
	
	// how owner1 sees owner2
	pub fn print_mood_action(&self, owner1: usize, owner2: usize, players: &Vec<Player>, d: &mut DispState) {
		let friendliness = self.friendliness_toward(owner1, owner2, players);
		
		macro_rules! c{($txt: expr, $color: expr) => {
			d.attron(COLOR_PAIR($color));
			d.addstr($txt);
			d.attroff(COLOR_PAIR($color));
		}};
		
		if friendliness < -0.75 {
			c!("furiously", CRED);
		}else if friendliness < -0.5 {
			c!("angrily", CRED);
		}else if friendliness < -0.1 {
			c!("irritatedly", CRED);
		}else if friendliness < -0.25 {
			d.addstr("bluntly");
		}else if friendliness < 0.1 {
			d.addstr("neutrally");
		}else if friendliness < 0.5 {
			c!("pleasently", CGREEN1);
		}else if friendliness < 0.75 {
			c!("warmly", CGREEN1);
		}else{
			c!("enthusiastically", CGREEN1);
		}
	}
	
	// how friendly is owner1 to owner2
	pub fn friendliness_toward(&self, owner1: usize, owner2: usize, players: &Vec<Player>) -> f32 {
		// intrinsic friendliness
		let friendliness = match &players[owner1].ptype {
			PlayerType::Empire(EmpireState {personality, ..}) => {
				personality.friendliness
			} PlayerType::Nobility(NobilityState {house, ..}) => {
				house.head_personality().friendliness
			} PlayerType::Human {..} | PlayerType::Barbarian {..} => {0.}
		};
		
		// from relations
		debug_assertq!(self.mood_toward.len() == (self.n_owners*self.n_owners));
		
		friendliness + self.mood_toward[owner1*self.n_owners + owner2]
	}
}

impl TxtCategory {
	// how does owner1 see owner2?
	pub fn from_relations(relations: &Relations, owner1: usize, owner2: usize, players: &Vec<Player>) -> Self {
		let friendliness = relations.friendliness_toward(owner1, owner2, players);
		
		const SPACE: f32 = 2./3.;
		if friendliness < (SPACE - 1.) {
			TxtCategory::Negative
		}else if friendliness < (2.*SPACE - 1.) {
			TxtCategory::Neutral
		}else{
			TxtCategory::Positive
		}
	}
}

impl Default for Relations {
	fn default() -> Self {
		Self {
			relation_status: Vec::new(),
			discov: Vec::new(),
			mood_toward: Vec::new(),
			n_owners: 0,
			config: Default::default()
}}}

