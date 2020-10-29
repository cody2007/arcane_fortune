use super::{Log, LogType};
use crate::saving::*;
use crate::resources::*;
use crate::units::*;
use crate::buildings::*;
use crate::renderer::*;
use crate::disp::*;
use crate::config_load::{config_parse, find_req_key_parse, read_file};
use crate::nn::{TxtCategory, TxtPrinter};
use crate::ai::*;
use crate::player::{PlayerType, Player};
use crate::gcore::XorState;
use crate::doctrine::DoctrineTemplate;
use crate::containers::*;
use crate::localization::*;

#[derive(Clone, PartialEq, Default)]
pub struct RelationsConfig {
	pub peace_treaty_min_years: usize,
	pub friendship_threshold: f32,
	pub enemy_threshold: f32,
	
	pub mood_you_declared_war_on_us: f32,
	pub mood_we_declared_war_on_you: f32,
	pub mood_we_are_sympathetic_to_the_unjust_war_against_you: f32,
	pub mood_you_declared_war_on_our_friend: f32,
	pub mood_you_declared_war_on_our_enemy: f32,
	pub mood_you_threatened_us: f32,
	
	// multiplied by AIPersonality.spirituality (provided that it is > 0.)
	pub mood_common_doctrine: f32,
	pub mood_differing_doctrine: f32,
	
	// only relevant for noble houses wrt the overlord empire
	pub mood_fiefdom: f32,
	pub mood_fiefdom_we_do_not_support_the_war: f32,
	pub mood_fiefdom_we_demand_war: f32
}

impl_saving!{RelationsConfig {peace_treaty_min_years,
	friendship_threshold, enemy_threshold,
	mood_you_declared_war_on_us, mood_we_declared_war_on_you,
	mood_we_are_sympathetic_to_the_unjust_war_against_you,
	mood_you_declared_war_on_our_friend,
	mood_you_declared_war_on_our_enemy,
	mood_you_threatened_us, mood_common_doctrine, mood_differing_doctrine, mood_fiefdom,
	mood_fiefdom_we_do_not_support_the_war, mood_fiefdom_we_demand_war
}}

pub fn init_relations_config() -> RelationsConfig {
	const RELATIONS_CONFIG_FILE: &str = "config/relations.txt";
	let key_sets = config_parse(read_file(RELATIONS_CONFIG_FILE));
	assertq!(key_sets.len() > 0, "no entries found in {}", RELATIONS_CONFIG_FILE);
	let key_sets_f = &key_sets[0];
	
	RelationsConfig {
		peace_treaty_min_years: find_req_key_parse("peace_treaty_min_years", key_sets_f),
		friendship_threshold: find_req_key_parse("friendship_threshold", key_sets_f),
		enemy_threshold: find_req_key_parse("enemy_threshold", key_sets_f),
		mood_you_declared_war_on_us: find_req_key_parse("mood_you_declared_war_on_us", key_sets_f),
		mood_we_declared_war_on_you: find_req_key_parse("mood_we_declared_war_on_you", key_sets_f),
		mood_we_are_sympathetic_to_the_unjust_war_against_you: find_req_key_parse("mood_we_are_sympathetic_to_the_unjust_war_against_you", key_sets_f),
		mood_you_declared_war_on_our_friend: find_req_key_parse("mood_you_declared_war_on_our_friend", key_sets_f),
		mood_you_declared_war_on_our_enemy: find_req_key_parse("mood_you_declared_war_on_our_enemy", key_sets_f),
		mood_you_threatened_us: find_req_key_parse("mood_you_threatened_us", key_sets_f),
		mood_common_doctrine: find_req_key_parse("mood_common_doctrine", key_sets_f),
		mood_differing_doctrine: find_req_key_parse("mood_differing_doctrine", key_sets_f),
		mood_fiefdom: find_req_key_parse("mood_fiefdom", key_sets_f),
		mood_fiefdom_we_do_not_support_the_war: find_req_key_parse("mood_fiefdom_we_do_not_support_the_war", key_sets_f),
		mood_fiefdom_we_demand_war: find_req_key_parse("mood_fiefdom_we_demand_war", key_sets_f),
	}
}

#[derive(Clone, PartialEq)]
pub enum RelationStatus {
	Undiscovered,
	Peace {turn_started: usize},
	War {turn_started: usize},
	Fiefdom {turn_joined: usize}
}

enum_From!{MoodType {
	YouDeclaredWarOnUs,
	WeDeclaredWarOnYou,
	WeAreSympatheticToTheUnjustWarAgainstYou,
	YouDeclaredWarOnOurFriend,
	YouDeclaredWarOnOurEnemy,
	YouThreatenedUs,
	CommonDoctrine,
	DifferingDoctrine,
	
	Fiefdom,
	FiefdomWeDoNotSupportTheWar,
	FiefdomWeDemandWar
}}

#[derive(Clone, PartialEq, Default)]
pub struct MoodFactor {
	pub mtype: MoodType,
	pub turn: usize
}

impl_saving!{MoodFactor {mtype, turn}}

#[derive(Clone, PartialEq, Default)]
pub struct MoodToward {
	pub val: f32,
	pub factors: Vec<MoodFactor>
}

impl_saving!{MoodToward {val, factors}}

#[derive(Clone, PartialEq)]
pub struct Relations {
	// square matrices only the upper triangle contains relevant values
	// i.e., it should be indexed (i,j) where i < j
	relation_status: Vec<RelationStatus>, // use relations.peace_treaty_expiry() to check if war can be declared
	
	mood_toward: Vec<MoodToward>, // (i,j) owner i's view toward owner j. this matrix does *not* need to be symmetric, unlike above
	
	n_owners: usize,
	
	pub config: RelationsConfig
}

impl_saving!{Relations{relation_status, mood_toward, n_owners, config}}

impl MoodType {
	pub fn weight(&self, config: &RelationsConfig) -> f32 {
		match self {
			MoodType::YouDeclaredWarOnUs => config.mood_you_declared_war_on_us,
			MoodType::WeDeclaredWarOnYou => config.mood_we_declared_war_on_you,
			MoodType::WeAreSympatheticToTheUnjustWarAgainstYou => config.mood_we_are_sympathetic_to_the_unjust_war_against_you,
			MoodType::YouDeclaredWarOnOurFriend => config.mood_you_declared_war_on_our_friend,
			MoodType::YouDeclaredWarOnOurEnemy => config.mood_you_declared_war_on_our_enemy,
			MoodType::YouThreatenedUs => config.mood_you_threatened_us,
			MoodType::CommonDoctrine => config.mood_common_doctrine,
			MoodType::DifferingDoctrine => config.mood_differing_doctrine,
			MoodType::Fiefdom => config.mood_fiefdom,
			MoodType::FiefdomWeDoNotSupportTheWar => config.mood_fiefdom_we_do_not_support_the_war,
			MoodType::FiefdomWeDemandWar => config.mood_fiefdom_we_demand_war,
			MoodType::N => {panicq!("invalid mood type");}
		}
	}
	
	pub fn txt(&self, l: &Localization) -> String {
		match self {
			MoodType::YouDeclaredWarOnUs => l.mood_you_declared_war_on_us.clone(),
			MoodType::WeDeclaredWarOnYou => l.mood_we_declared_war_on_you.clone(),
			MoodType::WeAreSympatheticToTheUnjustWarAgainstYou => l.mood_we_are_sympathetic_to_the_unjust_war_against_you.clone(),
			MoodType::YouDeclaredWarOnOurFriend => l.mood_you_declared_war_on_our_friend.clone(),
			MoodType::YouDeclaredWarOnOurEnemy => l.mood_you_declared_war_on_our_enemy.clone(),
			MoodType::YouThreatenedUs => l.mood_you_threatened_us.clone(),
			MoodType::CommonDoctrine => l.mood_common_doctrine.clone(),
			MoodType::DifferingDoctrine => l.mood_differing_doctrine.clone(),
			MoodType::Fiefdom => l.mood_fiefdom.clone(),
			MoodType::FiefdomWeDoNotSupportTheWar => l.mood_fiefdom_we_do_not_support_the_war.clone(),
			MoodType::FiefdomWeDemandWar => l.mood_fiefdom_we_demand_war.clone(),
			MoodType::N => {panicq!("invalid mood type");}
		}
	}
}

impl RelationStatus {
	pub fn is_peace(&self) -> bool {
		match self {
			RelationStatus::Fiefdom {..} |
			RelationStatus::Undiscovered |
			RelationStatus::Peace {..} => true,
			RelationStatus::War {..} => false
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
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Peace {..} => false
		}
	}
	
	fn peace_treaty_expiry_turn(&self, turn_started: usize) -> usize {
		self.config.peace_treaty_min_years * TURNS_PER_YEAR + turn_started
	}
	
	// returns number of years until expiration, none if no peace treaty
	pub fn peace_treaty_turns_remaining(&self, owner1: usize, owner2: usize, turn: usize) -> Option<usize> {
		let (omin, omax) = order_indices(owner1, owner2);
		
		match self.relation_status[omin*self.n_owners + omax] {
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Fiefdom {..} => None,
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
			RelationStatus::Undiscovered |
			RelationStatus::Peace {..} |
			RelationStatus::Fiefdom {..} => {panicq!("war not active")}
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
	pub fn threaten(&mut self, owner1: usize, owner2: usize, turn: usize) {
		self.add_mood_factor(owner1, owner2, MoodType::YouThreatenedUs, turn);
	}
	
	// add owner2's mood factor towards owner1
	fn add_mood_factor(&mut self, owner1: usize, owner2: usize, mtype: MoodType, turn: usize) {
		self.mood_toward[owner2*self.n_owners + owner1].factors.push(
			MoodFactor {mtype, turn}
		);
	}
	
	// owner 1 declares war on owner2
	// returns UIMode that should be set after this call
	pub fn declare_war_ret_ui_mode<'bt,'ut,'rt,'dt>(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, players: &Vec<Player>, turn: usize,
			cur_ai_player_is_paused: Option<bool>, rng: &mut XorState,
			dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> Option<UIMode<'bt,'ut,'rt,'dt>> {
		self.discover_civ(owner1, owner2, logs, turn);
		
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_ind = omin*self.n_owners + omax;
		
		// log & disable auto-turn
		if let RelationStatus::Peace {turn_started} = &self.relation_status[relation_ind] {
			// peace treaty in effect; see peace_treaty_expiry() for similar computation
			if *turn_started != 0 && self.peace_treaty_expiry_turn(*turn_started) > turn {
				return None;
			}
			
			self.relation_status[relation_ind] = RelationStatus::War {turn_started: turn};
			
			self.add_mood_factor(owner1, owner2, MoodType::YouDeclaredWarOnUs, turn);
			self.add_mood_factor(owner2, owner1, MoodType::WeDeclaredWarOnYou, turn);
			
			// how do others see the war? (if they see owner2, the attackee positively,
			//	then their mood will be lowered for owner1. if they see owner2 negatively,
			//	then their mood will be raised for owner1.
			for owner_i in (0..self.n_owners).filter(|&i| i != owner1 && i != owner2) {
				let friendliness_to_owner2 = self.friendliness_toward(owner_i, owner2, players);
				if friendliness_to_owner2 >= self.config.friendship_threshold {
					self.add_mood_factor(owner1, owner_i, MoodType::YouDeclaredWarOnOurFriend, turn);
					self.add_mood_factor(owner2, owner_i, MoodType::WeAreSympatheticToTheUnjustWarAgainstYou, turn);
				}else if friendliness_to_owner2 <= self.config.enemy_threshold {
					self.add_mood_factor(owner1, owner_i, MoodType::YouDeclaredWarOnOurEnemy, turn);
				}
			}
			
			logs.push(Log {turn,
					   val: LogType::WarDeclaration {
					   	owner_attacker_id: owner1,
					   	owner_attackee_id: owner2
					}
			});
			
			// disable auto-turn
			if dstate.iface_settings.interrupt_auto_turn && owner2 as SmSvType == dstate.iface_settings.cur_player {
				dstate.set_auto_turn(AutoTurn::Off);
				dstate.update_menu_indicators(cur_ai_player_is_paused);
				dstate.renderer.curs_set(CURSOR_VISIBILITY::CURSOR_INVISIBLE);
				
				let quote_category = TxtCategory::from_relations(self, owner1, owner2, players);
				
				return Some(UIMode::ContactEmbassyWindow(
					ContactEmbassyWindowState::DeclaredWarOn {
						owner_id: owner1,
						quote_printer: TxtPrinter::new(quote_category, rng.gen())
					 }
				));
			}
		}
		None
	}
	
	// owner 1 declares war on owner2
	// (same functionality as relations.declare_war_ret_ui_mode except it sets a new UIMode...
	//  this is the preferred function to use in most cases)
	pub fn declare_war(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, players: &Vec<Player>, turn: usize,
			cur_ai_player_is_paused: Option<bool>, rng: &mut XorState, disp: &mut Disp) {
		if let Some(ui_mode) = self.declare_war_ret_ui_mode(owner1, owner2, logs, players, turn, cur_ai_player_is_paused, rng, &mut disp.state) {
			disp.ui_mode = ui_mode;
		}
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
	
	pub fn fiefdom_of(&self, owner1: usize) -> Option<usize> {
		for owner2 in (0..self.n_owners).filter(|&owner2| owner2 != owner1) {
			let (omin, omax) = order_indices(owner1, owner2);
			let relation_ind = omin*self.n_owners + omax;
			
			if let RelationStatus::Fiefdom {..} = &self.relation_status[relation_ind] {
				return Some(owner2);
			}
		}
		None
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
		if let RelationStatus::Undiscovered = self.relation_status[omin*self.n_owners + omax] {
			false
		}else{
			true
		}
	}
	
	// owner1 discovers owner2
	pub fn discover_civ(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, turn: usize) {
		if owner1 == owner2 {return;}
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_status = &mut self.relation_status[omin*self.n_owners + omax];
		if let RelationStatus::Undiscovered = relation_status {
			*relation_status = RelationStatus::Peace {turn_started: 0};
			
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
			let relation_status = &mut self.relation_status[omin*self.n_owners + omax];
			if let RelationStatus::Undiscovered = relation_status {
				*relation_status = RelationStatus::Peace {turn_started: 0};
			}
		}
	}
	
	pub fn new(n_owners: usize) -> Self {
		// note, the defaults should be the same as self.add_player()
		Self {
			relation_status: vec!{Default::default(); n_owners*n_owners},
			mood_toward: vec![MoodToward {val: 0., factors: Vec::new()}; n_owners*n_owners],
			n_owners,
			config: init_relations_config()
		}
	}
	
	pub fn add_player(&mut self) {
		// note, the defaults should be the same as Relations::new()
		self.n_owners += 1;
		let n_owners_sq = self.n_owners * self.n_owners;
		
		let mut relation_status = vec!{Default::default(); n_owners_sq};
		let mut mood_toward = vec!{MoodToward {val: 0., factors: Vec::new()}; n_owners_sq};
		
		for i in 0..(self.n_owners-1) {
			for j in 0..(self.n_owners-1) {
				let ind_new = i*self.n_owners + j;
				let ind_old = i*(self.n_owners-1) + j;
				
				relation_status[ind_new] = self.relation_status[ind_old].clone();
				mood_toward[ind_new] = self.mood_toward[ind_old].clone();
			}
		}
		
		self.relation_status = relation_status;
		self.mood_toward = mood_toward;
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
		
		let mood_toward = &self.mood_toward[owner1*self.n_owners + owner2];
		let factor_sum = mood_toward.factors.iter()
			.map(|f| f.mtype.weight(&self.config)).sum::<f32>();
		
		friendliness + factor_sum + mood_toward.val
	}
	
	// what are the factors influencing owner1's mood toward owner2?
	pub fn print_mood_factors(&self, owner1: usize, owner2: usize, loc: ScreenCoord, dstate: &mut DispState) {
		let factors = &self.mood_toward[owner1*self.n_owners + owner2].factors;
		if factors.len() == 0 {return;}
		
		let max_w = {
			let mut max_w = dstate.local.Mood.len();
			for factor in factors.iter() {
				let weight = factor.mtype.weight(&self.config);
				let w = if weight >= 0. {
					format!("+{} ", weight)
				}else{
					format!("{}", weight)
				}.len() + factor.mtype.txt(&dstate.local).len();
				if w > max_w {max_w = w;}
			}
			max_w + 4
		};
		
		let window_sz = ScreenSz {h: 2 + 2 + factors.len(), w: max_w + 4, sz: 0};
		dstate.print_window_at(window_sz, loc);
		
		dstate.mv(loc.y as i32 + 1, loc.x as i32 + ((max_w - dstate.local.Mood.len())/2) as i32);
		dstate.txt_list.add_w(&mut dstate.renderer);
		dstate.renderer.addstr(&dstate.local.Mood);
		
		for (row_off, factor) in factors.iter().enumerate() {
			dstate.mv(row_off as i32 + loc.y as i32 + 3, loc.x as i32 + 2);
			dstate.txt_list.add_w(&mut dstate.renderer);
			
			let weight = factor.mtype.weight(&self.config);
			if weight >= 0. {
				addstr_c(&format!("+{} ", weight), CGREEN, &mut dstate.renderer);
			}else{
				addstr_c(&format!("{} ", weight), CRED, &mut dstate.renderer);
			}
			dstate.renderer.addstr(&factor.mtype.txt(&dstate.local));
		}
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
			mood_toward: Vec::new(),
			n_owners: 0,
			config: Default::default()
}}}

