use super::{Log, LogType, TURNS_PER_YEAR};
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
use crate::tech::transfer_undiscov_tech;
use crate::containers::*;
use crate::localization::*;

pub mod trade; pub use trade::*;

#[derive(Clone, Default)]
pub struct Relations {
	// square matrices only the upper triangle contains relevant values
	// i.e., it should be indexed (i,j) where i < j
	relation_status: Vec<RelationStatus>, // use relations.peace_treaty_expiry() to check if war can be declared
	
	mood_toward: Vec<MoodToward>, // (i,j) owner i's view toward owner j. this matrix does *not* need to be symmetric, unlike above
	
	n_owners: usize,
	
	federations: Vec<Federation>,
	
	pub config: RelationsConfig
}

impl_saving!{Relations {relation_status, mood_toward, n_owners, federations, config}}

#[derive(Clone, Default)]
pub struct Federation {
	turn_founded: usize,
	name: String
}
impl_saving!{Federation {turn_founded, name}}

#[derive(Clone, Default)]
pub struct RelationsConfig {
	pub peace_treaty_min_years: usize,
	pub trade_reasonableness_threshold: f32, // used to give a mood bonus or penalty if an offer is outside of this
	friendship_threshold: f32,
	enemy_threshold: f32,
	
	mood_you_declared_war_on_us: f32,
	mood_we_declared_war_on_you: f32,
	mood_failed_assassination: f32,
	mood_successful_assassination: f32,
	mood_we_are_sympathetic_to_the_unjust_war_against_you: f32,
	mood_you_declared_war_on_our_friend: f32,
	mood_you_declared_war_on_our_enemy: f32,
	mood_your_offer_insulted_us: f32, // if an offer is outside of the trade_reasonableness_threshold
	mood_you_are_easily_taken_advantage_of: f32, // ^
	mood_you_threatened_us: f32,
	mood_we_fear_you: f32,
	mood_you_helped_us: f32,
	mood_you_refused_to_help_us: f32,
	mood_your_empires_nobility_are_dignified: f32, // when a noble with high friendliness joins an empire, other empires will have this mood towards the empire the nobility was added to
	mood_your_empires_nobility_are_brutish: f32, // when a noble with low friendliness joins an empire, other empires will have this mood towards the empire the nobility was added to
	
	mood_independence_handled: f32, // a noble attempts to leave an empire and the player responds w/ war
	mood_independence_not_handled: f32, // the same as the above but the player does nothing
	
	// multiplied by AIPersonality.spirituality (provided that it is > 0.)
	mood_common_doctrine: f32,
	mood_differing_doctrine: f32,
	
	// only relevant for noble houses wrt the overlord empire
	mood_fiefdom: f32,
	mood_fiefdom_we_do_not_support_the_war: f32,
	mood_fiefdom_we_demand_war: f32,
	mood_fiefdom_taxes_are_too_high: f32,
	mood_fiefdom_taxes_are_low: f32
}

impl_saving!{RelationsConfig {
	peace_treaty_min_years, trade_reasonableness_threshold,
	friendship_threshold, enemy_threshold,
	
	mood_you_declared_war_on_us, mood_we_declared_war_on_you,
	mood_failed_assassination, mood_successful_assassination,
	mood_we_are_sympathetic_to_the_unjust_war_against_you,
	mood_you_declared_war_on_our_friend,
	mood_you_declared_war_on_our_enemy,
	mood_your_offer_insulted_us,
	mood_you_are_easily_taken_advantage_of,
	mood_you_threatened_us, 
	mood_we_fear_you,
	mood_you_helped_us,
	mood_you_refused_to_help_us,
	mood_your_empires_nobility_are_dignified,
	mood_your_empires_nobility_are_brutish,
	
	mood_independence_handled, mood_independence_not_handled,
	
	mood_common_doctrine, mood_differing_doctrine, mood_fiefdom,
	
	mood_fiefdom_we_do_not_support_the_war, mood_fiefdom_we_demand_war,
	mood_fiefdom_taxes_are_too_high, mood_fiefdom_taxes_are_low
}}

pub fn init_relations_config() -> RelationsConfig {
	const RELATIONS_CONFIG_FILE: &str = "config/relations.txt";
	let key_sets = config_parse(read_file(RELATIONS_CONFIG_FILE));
	assertq!(key_sets.len() > 0, "no entries found in {}", RELATIONS_CONFIG_FILE);
	let key_sets_f = &key_sets[0];
	
	RelationsConfig {
		peace_treaty_min_years: find_req_key_parse("peace_treaty_min_years", key_sets_f),
		trade_reasonableness_threshold: find_req_key_parse("trade_reasonableness_threshold", key_sets_f),
		friendship_threshold: find_req_key_parse("friendship_threshold", key_sets_f),
		enemy_threshold: find_req_key_parse("enemy_threshold", key_sets_f),
		mood_you_declared_war_on_us: find_req_key_parse("mood_you_declared_war_on_us", key_sets_f),
		mood_we_declared_war_on_you: find_req_key_parse("mood_we_declared_war_on_you", key_sets_f),
		mood_failed_assassination: find_req_key_parse("mood_failed_assassination", key_sets_f),
		mood_successful_assassination: find_req_key_parse("mood_successful_assassination", key_sets_f),
	
		mood_we_are_sympathetic_to_the_unjust_war_against_you: find_req_key_parse("mood_we_are_sympathetic_to_the_unjust_war_against_you", key_sets_f),
		mood_you_declared_war_on_our_friend: find_req_key_parse("mood_you_declared_war_on_our_friend", key_sets_f),
		mood_you_declared_war_on_our_enemy: find_req_key_parse("mood_you_declared_war_on_our_enemy", key_sets_f),
		mood_your_offer_insulted_us: find_req_key_parse("mood_your_offer_insulted_us", key_sets_f),
		mood_you_are_easily_taken_advantage_of: find_req_key_parse("mood_you_are_easily_taken_advantage_of", key_sets_f),
		mood_you_threatened_us: find_req_key_parse("mood_you_threatened_us", key_sets_f),
		mood_we_fear_you: find_req_key_parse("mood_we_fear_you", key_sets_f),
		mood_you_helped_us: find_req_key_parse("mood_you_helped_us", key_sets_f),
		mood_you_refused_to_help_us: find_req_key_parse("mood_you_refused_to_help_us", key_sets_f),
		mood_your_empires_nobility_are_dignified: find_req_key_parse("mood_your_empires_nobility_are_dignified", key_sets_f),
		mood_your_empires_nobility_are_brutish: find_req_key_parse("mood_your_empires_nobility_are_brutish", key_sets_f),
		
		mood_independence_handled: find_req_key_parse("mood_independence_handled", key_sets_f),
		mood_independence_not_handled: find_req_key_parse("mood_independence_not_handled", key_sets_f),
		
		mood_common_doctrine: find_req_key_parse("mood_common_doctrine", key_sets_f),
		mood_differing_doctrine: find_req_key_parse("mood_differing_doctrine", key_sets_f),
		
		mood_fiefdom: find_req_key_parse("mood_fiefdom", key_sets_f),
		mood_fiefdom_we_do_not_support_the_war: find_req_key_parse("mood_fiefdom_we_do_not_support_the_war", key_sets_f),
		mood_fiefdom_we_demand_war: find_req_key_parse("mood_fiefdom_we_demand_war", key_sets_f),
		mood_fiefdom_taxes_are_too_high: find_req_key_parse("mood_fiefdom_taxes_are_too_high", key_sets_f),
		mood_fiefdom_taxes_are_low: find_req_key_parse("mood_fiefdom_taxes_are_low", key_sets_f)
	}
}

#[derive(Clone)]
pub enum RelationStatus {
	Undiscovered,
	Peace(TradeStatus), 
	War {turn_started: usize},
	Fiefdom {
		tax_rate: u8,
		trade_status: TradeStatus,
	},
	Kingdom {
		kingdom_id: usize, // relationstatus is in a symmetric structure. this it the subordinate empire
		trade_status: TradeStatus
	}
	
	//Vassal(TradeStatus),
	//DefensivePact(TradeStatus),
	//Federation {turn_joined: usize, federation_ind: usize},
	//TradeLeague {turn_joined: usize, league_ind: usize}, // (should be within TradeDeals instead)
}

#[derive(Clone, Default)]
pub struct TradeStatus {
	turn_started: usize,
	trade_deals: Vec<TradeDeal>
}

impl_saving!{TradeStatus {turn_started, trade_deals}}

impl TradeStatus {
	fn new(turn_started: usize) -> Self {
		Self {
			turn_started,
			trade_deals: Vec::new()
		}
	}
	
	fn deals(&self) -> &Vec<TradeDeal> {
		&self.trade_deals
	}
	
	fn deals_mut(&mut self) -> &mut Vec<TradeDeal> {
		&mut self.trade_deals
	}
}

#[derive(Clone, PartialEq, Default)]
pub struct MoodToward {
	pub val: f32,
	factors: Vec<MoodFactor>
	// see relations.friendliness_toward()
	// which returns val + sum(factors.weight())
}

impl_saving!{MoodToward {val, factors}}

impl MoodToward {
	// transient moods (ex. due to nobility, being well liked or hated)
	// of owner1 toward owner2
	// i.e. moods that can be removed if the condition changes, such as the nobility leaving the empire
	pub fn transient_and_permanent_factors(&self, owner1: usize,
			owner2: usize, relations: &Relations, players: &Vec<Player>) -> Vec<MoodFactor> {
		let mut factors = self.factors.clone();
		
		let mut add_factor = |mtype| {
			factors.push(MoodFactor {mtype, turn: 0});
		};
		
		let judger = &players[owner1];
		let judgee = &players[owner2];
		
		{ // doctrines
			let judger_doctrine = judger.stats.doctrine_template.id;
			let judgee_doctrine = judgee.stats.doctrine_template.id;
			
			// common doctrine
			if judger_doctrine != 0 && judger_doctrine == judgee_doctrine {
				add_factor(MoodType::CommonDoctrine);
			
			// differing doctrine
			}else if let Some(personality) = judger.ptype.personality() {
				if personality.concerned_about_uncommon_doctrines() {
					add_factor(MoodType::DifferingDoctrine);
				}
			}
		}
		
		{ // nobles - likable and unlikable
			for house_ind in relations.noble_houses(owner2) {
				if let Some(house) = players[house_ind].ptype.house() {
					let noble_personality = house.head_personality();
					
					if noble_personality.diplomatic_friendliness_bonus() {
						add_factor(MoodType::YourEmpiresNobilityAreDignified);
					}else if noble_personality.diplomatic_friendliness_penalty() {
						add_factor(MoodType::YourEmpiresNobilityAreBrutish);
					}
				}//else{panicq!("house {} of owner {} is not a house (owner1 {})", house_ind, owner2, owner1);}
			}
		}
		
		// fiefdom-specific moods
		if relations.fiefdom(owner1, owner2) {
			// fiefdom does not support the wars the parent is involved with
			if let Some(judger_personality) = judger.ptype.personality() {
				if judger_personality.diplomatic_friendliness_bonus() {
					for _ in 0..relations.at_war_with(owner2).len() {
						add_factor(MoodType::FiefdomWeDoNotSupportTheWar);
					}
				}
			}
			
			// taxes too high / too low
			let tax_rate = relations.tax_rate(owner1, owner2).unwrap();
			if tax_rate > 30 {
				add_factor(MoodType::FiefdomTaxesAreTooHigh);
			}else if tax_rate < 10 {
				add_factor(MoodType::FiefdomTaxesAreLow);
			}
		}
		
		factors
	}
}

#[derive(Clone, PartialEq, Default)]
pub struct MoodFactor {
	pub mtype: MoodType,
	pub turn: usize
}

impl_saving!{MoodFactor {mtype, turn}}

enum_From!{MoodType {
	YouDeclaredWarOnUs,
	WeDeclaredWarOnYou,
	WeAreSympatheticToTheUnjustWarAgainstYou,
	YouDeclaredWarOnOurFriend,
	YouDeclaredWarOnOurEnemy,
	FailedAssassination,
	SuccessfulAssassination,
	YourOfferInsultedUs,
	YouAreEasilyTakenAdvantageOf,
	YouThreatenedUs,
	WeFearYou,
	YouHelpedUs,
	YouRefusedToHelpUs,
	YourEmpiresNobilityAreDignified,
	YourEmpiresNobilityAreBrutish,
	
	IndependenceHandled,
	IndependenceNotHandled,
	
	CommonDoctrine,
	DifferingDoctrine,
	
	Fiefdom,
	FiefdomWeDoNotSupportTheWar,
	FiefdomWeDemandWar,
	FiefdomTaxesAreTooHigh,
	FiefdomTaxesAreLow
}}


impl MoodType {
	pub fn weight(&self, config: &RelationsConfig) -> f32 {
		match self {
			MoodType::YouDeclaredWarOnUs => config.mood_you_declared_war_on_us,
			MoodType::WeDeclaredWarOnYou => config.mood_we_declared_war_on_you,
			MoodType::WeAreSympatheticToTheUnjustWarAgainstYou => config.mood_we_are_sympathetic_to_the_unjust_war_against_you,
			MoodType::YouDeclaredWarOnOurFriend => config.mood_you_declared_war_on_our_friend,
			MoodType::YouDeclaredWarOnOurEnemy => config.mood_you_declared_war_on_our_enemy,
			MoodType::FailedAssassination => config.mood_failed_assassination,
			MoodType::SuccessfulAssassination => config.mood_successful_assassination,
			MoodType::YourOfferInsultedUs => config.mood_your_offer_insulted_us,
			MoodType::YouAreEasilyTakenAdvantageOf => config.mood_you_are_easily_taken_advantage_of,
			MoodType::YouThreatenedUs => config.mood_you_threatened_us,
			MoodType::WeFearYou => config.mood_we_fear_you,
			MoodType::YouHelpedUs => config.mood_you_helped_us,
			MoodType::YouRefusedToHelpUs => config.mood_you_refused_to_help_us,
			MoodType::YourEmpiresNobilityAreDignified => config.mood_your_empires_nobility_are_dignified,
			MoodType::YourEmpiresNobilityAreBrutish => config.mood_your_empires_nobility_are_brutish,
			
			MoodType::IndependenceHandled => config.mood_independence_handled,
			MoodType::IndependenceNotHandled => config.mood_independence_not_handled,
			
			MoodType::CommonDoctrine => config.mood_common_doctrine,
			MoodType::DifferingDoctrine => config.mood_differing_doctrine,
			
			MoodType::Fiefdom => config.mood_fiefdom,
			MoodType::FiefdomWeDoNotSupportTheWar => config.mood_fiefdom_we_do_not_support_the_war,
			MoodType::FiefdomWeDemandWar => config.mood_fiefdom_we_demand_war,
			MoodType::FiefdomTaxesAreTooHigh => config.mood_fiefdom_taxes_are_too_high,
			MoodType::FiefdomTaxesAreLow => config.mood_fiefdom_taxes_are_low,
			MoodType::N => {panicq!("invalid mood type");}
		}
	}
	
	pub fn txt(&self, l: &Localization) -> String {
		match self {
			MoodType::YouDeclaredWarOnUs => l.mood_you_declared_war_on_us.clone(),
			MoodType::WeDeclaredWarOnYou => l.mood_we_declared_war_on_you.clone(),
			MoodType::FailedAssassination => l.mood_failed_assassination.clone(),
			MoodType::SuccessfulAssassination => l.mood_successful_assassination.clone(),
			MoodType::WeAreSympatheticToTheUnjustWarAgainstYou => l.mood_we_are_sympathetic_to_the_unjust_war_against_you.clone(),
			MoodType::YouDeclaredWarOnOurFriend => l.mood_you_declared_war_on_our_friend.clone(),
			MoodType::YouDeclaredWarOnOurEnemy => l.mood_you_declared_war_on_our_enemy.clone(),
			MoodType::YourOfferInsultedUs => l.mood_your_offer_insulted_us.clone(),
			MoodType::YouAreEasilyTakenAdvantageOf => l.mood_you_are_easily_taken_advantage_of.clone(),
			MoodType::YouThreatenedUs => l.mood_you_threatened_us.clone(),
			MoodType::WeFearYou => l.mood_we_fear_you.clone(),
			MoodType::YouHelpedUs => l.mood_you_helped_us.clone(),
			MoodType::YouRefusedToHelpUs => l.mood_you_refused_to_help_us.clone(),
			MoodType::YourEmpiresNobilityAreDignified => l.mood_your_empires_nobility_are_dignified.clone(),
			MoodType::YourEmpiresNobilityAreBrutish => l.mood_your_empires_nobility_are_brutish.clone(),
			
			MoodType::IndependenceHandled => l.mood_independence_handled.clone(),
			MoodType::IndependenceNotHandled => l.mood_independence_not_handled.clone(),
			
			MoodType::CommonDoctrine => l.mood_common_doctrine.clone(),
			MoodType::DifferingDoctrine => l.mood_differing_doctrine.clone(),
			
			MoodType::Fiefdom => l.mood_fiefdom.clone(),
			MoodType::FiefdomWeDoNotSupportTheWar => l.mood_fiefdom_we_do_not_support_the_war.clone(),
			MoodType::FiefdomWeDemandWar => l.mood_fiefdom_we_demand_war.clone(),
			MoodType::FiefdomTaxesAreTooHigh => l.mood_fiefdom_taxes_are_too_high.clone(),
			MoodType::FiefdomTaxesAreLow => l.mood_fiefdom_taxes_are_low.clone(),
			MoodType::N => {panicq!("invalid mood type");}
		}
	}
}

impl RelationStatus {
	pub fn is_peace(&self) -> bool {
		match self {
			RelationStatus::Fiefdom {..} |
			RelationStatus::Undiscovered |
			RelationStatus::Kingdom {..} |
			RelationStatus::Peace(_) => true,
			RelationStatus::War {..} => false
		}
	}
	
	pub fn trade_deals(&self) -> Option<&Vec<TradeDeal>> {
		match self {
			RelationStatus::Peace(trade_status) |
			RelationStatus::Kingdom {trade_status, ..} |
			RelationStatus::Fiefdom {trade_status, ..} => {
				Some(trade_status.deals())
			}
			RelationStatus::Undiscovered |
			RelationStatus::War {..} => None
		}
	}
	
	fn trade_status_cloned(&self) -> Option<TradeStatus> {
		match self {
			RelationStatus::Peace(trade_status) |
			RelationStatus::Kingdom {trade_status, ..} |
			RelationStatus::Fiefdom {trade_status, ..} => {
				Some(trade_status.clone())
			}
			RelationStatus::Undiscovered |
			RelationStatus::War {..} => None
		}
	}
	
	fn trade_deals_mut(&mut self) -> Option<&mut Vec<TradeDeal>> {
		match self {
			RelationStatus::Peace(trade_status) |
			RelationStatus::Kingdom {trade_status, ..} |
			RelationStatus::Fiefdom {trade_status, ..} => {
				Some(trade_status.deals_mut())
			}
			RelationStatus::Undiscovered |
			RelationStatus::War {..} => None
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
		!self.status(owner1, owner2).is_peace()
	}
	
	// ordering doesn't matter
	pub fn status(&self, owner1: usize, owner2: usize) -> &RelationStatus {
		let (omin, omax) = order_indices(owner1, owner2);
		&self.relation_status[omin*self.n_owners + omax]
	}
	
	// ordering doesn't matter
	pub fn status_mut(&mut self, owner1: usize, owner2: usize) -> &mut RelationStatus {
		let (omin, omax) = order_indices(owner1, owner2);
		&mut self.relation_status[omin*self.n_owners + omax]
	}
	
	// ordering of owner input parameters does not matter
	pub fn fiefdom(&self, owner1: usize, owner2: usize) -> bool {
		match self.status(owner1, owner2) {
			RelationStatus::Fiefdom {..} => true,
			RelationStatus::Kingdom {..} |
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Peace(_) => false
		}
	}
	
	// ordering of owner input parameters does not matter
	// returns true if owner1 is a kingdom or owner2 OR owner2 is a kingdom of owner1
	pub fn kingdom(&self, owner1: usize, owner2: usize) -> bool {
		match self.status(owner1, owner2) {
			RelationStatus::Kingdom {..} => true,
			RelationStatus::Fiefdom {..} |
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Peace(_) => false
		}
	}
	
	// ordering does not matter
	// the tax rate one of the owners pays to the other
	pub fn tax_rate(&self, owner1: usize, owner2: usize) -> Option<u8> {
		match self.status(owner1, owner2) {
			RelationStatus::Fiefdom {tax_rate, ..} => Some(*tax_rate),
			RelationStatus::Kingdom {..} |
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Peace(_) => None
		}
	}
	
	// ordering does not matter
	// the tax rate one of the owners pays to the other
	pub fn set_tax_rate(&mut self, owner1: usize, owner2: usize, new_tax_rate: u8) {
		match self.status_mut(owner1, owner2) {
			RelationStatus::Fiefdom {tax_rate, ..} => {
				*tax_rate = new_tax_rate;
			}
			RelationStatus::Kingdom {..} |
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Peace(_) => {panicq!("expected fiefdom, found another relation status");}
		}
	}
	
	fn peace_treaty_expiry_turn(&self, turn_started: usize) -> usize {
		self.config.peace_treaty_min_years * TURNS_PER_YEAR + turn_started
	}
	
	// returns number of years until expiration, none if no peace treaty
	pub fn peace_treaty_turns_remaining(&self, owner1: usize, owner2: usize, turn: usize) -> Option<usize> {
		match self.status(owner1, owner2) {
			RelationStatus::Undiscovered |
			RelationStatus::War {..} |
			RelationStatus::Fiefdom {..} |
			RelationStatus::Kingdom {..} => None,
			RelationStatus::Peace(TradeStatus {turn_started, ..}) => {
				// peace treaty in effect; see declare_war() for similar computation
				let expiry_turn = self.peace_treaty_expiry_turn(*turn_started);
				if *turn_started != 0 && expiry_turn > turn {
					Some(expiry_turn - turn)
				}else{
					None
				}
			}
		}
	}
	
	pub fn turn_war_started(&self, owner1: usize, owner2: usize) -> usize {
		match self.status(owner1, owner2) {
			RelationStatus::War {turn_started} => {*turn_started}
			RelationStatus::Undiscovered |
			RelationStatus::Peace(_) |
			RelationStatus::Kingdom {..} |
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
	pub fn threaten(&mut self, owner1: usize, owner2: usize, players: &Vec<Player>, turn: usize) {
		// owner1 is stronger than owner2 and has not threatened owner2 yet
		if let Some(adv) = current_war_advantage(owner1, players, self) {
			if adv > 2**players[owner2].stats.defense_power_log.last().unwrap() as isize && // owner1 is stronger than owner2
				!self.mood_toward[owner2*self.n_owners + owner1].factors.iter() // owner1 has not threatened owner2 yet
					.any(|factor| factor.mtype == MoodType::YouThreatenedUs) {
				self.add_mood_factor(owner1, owner2, MoodType::WeFearYou, turn);
				return;
			}
		}
		
		self.add_mood_factor(owner1, owner2, MoodType::YouThreatenedUs, turn);
	}
	
	// add owner2's mood factor towards owner1
	pub fn add_mood_factor(&mut self, owner1: usize, owner2: usize, mtype: MoodType, turn: usize) {
		self.mood_toward[owner2*self.n_owners + owner1].factors.push(
			MoodFactor {mtype, turn}
		);
	}
	
	// owner 1 declares war on owner2
	// returns UIMode that should be set after this call
	pub fn declare_war_ret_ui_mode<'bt,'ut,'rt,'dt>(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, players: &Vec<Player>, turn: usize,
			cur_ai_player_is_paused: Option<bool>, rng: &mut XorState,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> Option<UIMode<'bt,'ut,'rt,'dt>> {
		self.discover_civ(owner1, owner2, logs, turn);
		
		let (omin, omax) = order_indices(owner1, owner2);
		let relation_ind = omin*self.n_owners + omax;
		
		// log & disable auto-turn
		if let RelationStatus::Peace(TradeStatus {turn_started, ..}) = &self.relation_status[relation_ind] {
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
			
			logs.push(
				Log {turn,
				   val: LogType::WarDeclaration {
					owner_attacker_id: owner1,
					owner_attackee_id: owner2
				}
			});
			
			// everyone who is a fiefdom of owner1 declares war on owner2
			for owner in self.noble_houses(owner1) {
				self.declare_war_ret_ui_mode(owner, owner2, logs, players, turn, None, rng, dstate);
			}
			
			// everyone in a defensive pact (or fiefdom) of owner2 declares war on owner1
			for owner in self.defensive_pact_owners(owner2) {
				self.declare_war_ret_ui_mode(owner, owner1, logs, players, turn, None, rng, dstate);
			}
			
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
	
	// ordering of owner input vars does not matter
	fn synchronize_wars(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, players: &Vec<Player>, turn: usize,
			rng: &mut XorState, dstate: &mut DispState) {
		let owner1_wars = self.at_war_with(owner1);
		let owner2_wars = self.at_war_with(owner2);
		
		// owner2 declares war on everyone that owner1 is at war with that owner2 is not at war with
		for owner1_war in owner1_wars.iter().filter(|owner1_war| !owner2_wars.contains(owner1_war)) {
			self.declare_war_ret_ui_mode(owner2, *owner1_war, logs, players, turn, None, rng, dstate);
		}
		
		// owner1 declares war on everyone that owner2 is at war with that owner1 is not at war with
		for owner2_war in owner2_wars.iter().filter(|owner2_war| !owner1_wars.contains(owner2_war)) {
			self.declare_war_ret_ui_mode(owner1, *owner2_war, logs, players, turn, None, rng, dstate);
		}
	}
	
	// owner 1 declares war on owner2
	// (same functionality as relations.declare_war_ret_ui_mode except it sets a new UIMode...
	//  this is the preferred function to use in most cases)
	pub fn declare_war(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, players: &Vec<Player>, turn: usize,
			rng: &mut XorState, disp: &mut Disp) {
		if let Some(ui_mode) = self.declare_war_ret_ui_mode(owner1, owner2, logs, players, turn, disp.state.iface_settings.cur_player_paused(players), rng, &mut disp.state) {
			disp.ui_mode = ui_mode;
		}
	}
	
	// owner 1 joints as noble house of owner2
	pub fn join_as_fiefdom(&mut self, noble_owner: usize, parent_owner: usize, players: &mut Vec<Player>,
			logs: &mut Vec<Log>, turn: usize, rng: &mut XorState, dstate: &mut DispState) {
		self.discover_civ(noble_owner, parent_owner, logs, turn);
		
		*self.status_mut(noble_owner, parent_owner) = RelationStatus::Fiefdom {
			tax_rate: 20,
			trade_status: TradeStatus::new(turn)
		};
		
		let empire_color = players[parent_owner].personalization.color;
		let noble_player = &mut players[noble_owner];
		noble_player.personalization.color = empire_color;
		
		// nobility sees the parent empire positively
		self.add_mood_factor(parent_owner, noble_owner, MoodType::Fiefdom, turn);
		
		logs.push(Log {turn,
			val: LogType::NobleHouseJoinedEmpire {
				house_id: noble_owner,
				empire_id: parent_owner
			}
		});
		
		self.synchronize_wars(noble_owner, parent_owner, logs, players, turn, rng, dstate);
	}
	
	// kingdom_id joins as kingdom of empire_id
	pub fn join_as_kingdom<'bt,'ut,'rt,'dt>(&mut self, kingdom_id: usize, empire_id: usize, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			logs: &mut Vec<Log>, turn: usize, rng: &mut XorState, temps: &Templates<'bt,'ut,'rt,'dt,'_>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) {
		self.discover_civ(kingdom_id, empire_id, logs, turn);
		
		{ // set status
			let status = self.status_mut(kingdom_id, empire_id);
			
			let trade_status = status.trade_status_cloned().unwrap_or(TradeStatus::new(turn));
			
			*status = RelationStatus::Kingdom {
				kingdom_id,
				trade_status
			};
		}
		
		logs.push(Log {turn,
			val: LogType::KingdomJoinedEmpire {
				kingdom_id,
				empire_id
			}
		});
		
		self.synchronize_wars(kingdom_id, empire_id, logs, players, turn, rng, dstate);
		
		// add all the kingdom's tech to the parent empire
		transfer_undiscov_tech(kingdom_id, empire_id, players, temps, dstate);
	}
	
	pub fn fiefdom_of(&self, owner1: usize) -> Option<usize> {
		for owner2 in (0..self.n_owners).filter(|&owner2| owner2 != owner1) {
			if let RelationStatus::Fiefdom {..} = self.status(owner1, owner2) {
				return Some(owner2);
			}
		}
		None
	}
	
	pub fn kingdom_of(&self, owner1: usize) -> Option<usize> {
		for owner2 in (0..self.n_owners).filter(|&owner2| owner2 != owner1) {
			if let RelationStatus::Kingdom {kingdom_id, ..} = self.status(owner1, owner2) {
				// owner1 is a kingdom of owner2
				if *kingdom_id == owner1 {
					return Some(owner2);
				}
			}
		}
		None
	}
	
	pub fn declare_peace_wo_logging(&mut self, owner1: usize, owner2: usize, turn: usize) {
		let relation_status = self.status_mut(owner1, owner2);
		debug_assertq!(!relation_status.is_peace());
		*relation_status = RelationStatus::Peace(TradeStatus::new(turn));
	}
	
	pub fn declare_peace(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, turn: usize) {
		self.declare_peace_wo_logging(owner1, owner2, turn);
		
		logs.push(Log {turn,
				   val: LogType::PeaceDeclaration {
				   owner1_id: owner1,
				   owner2_id: owner2
				}
		});
	}
	
	pub fn discovered(&self, owner1: usize, owner2: usize) -> bool {
		if owner1 == owner2 {return true;}
		if let RelationStatus::Undiscovered = self.status(owner1, owner2) {
			false
		}else{
			true
		}
	}
	
	// owner1 discovers owner2
	pub fn discover_civ(&mut self, owner1: usize, owner2: usize, logs: &mut Vec<Log>, turn: usize) {
		if owner1 == owner2 {return;}
		let relation_status = self.status_mut(owner1, owner2);
		if let RelationStatus::Undiscovered = relation_status {
			*relation_status = RelationStatus::Peace(TradeStatus::new(0));
			
			logs.push(Log {turn,
					val: LogType::CivDiscov {
						discover_id: owner1,
						discovee_id: owner2
			}});
		}
	}
	
	pub fn discover_all_civs(&mut self, owner1: usize) {
		for owner_id in (0..self.n_owners).filter(|&owner_id| owner_id != owner1) {
			let relation_status = self.status_mut(owner1, owner_id);
			if let RelationStatus::Undiscovered = relation_status {
				*relation_status = RelationStatus::Peace(TradeStatus::new(0));
			}
		}
	}
	
	pub fn new(n_owners: usize) -> Self {
		// note, the defaults should be the same as self.add_player()
		Self {
			relation_status: vec!{Default::default(); n_owners*n_owners},
			mood_toward: vec![MoodToward::default(); n_owners*n_owners],
			n_owners,
			federations: Vec::new(),
			config: init_relations_config()
		}
	}
	
	pub fn add_player(&mut self) {
		// note, the defaults should be the same as Relations::new()
		self.n_owners += 1;
		let n_owners_sq = self.n_owners * self.n_owners;
		
		let mut relation_status = vec!{Default::default(); n_owners_sq};
		let mut mood_toward = vec!{MoodToward::default(); n_owners_sq};
		
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
	
	// (txt, color_pair)
	pub fn mood_action_txt(&self, owner1: usize, owner2: usize, players: &Vec<Player>, l: &Localization) -> (String, chtype) {
		let friendliness = self.friendliness_toward(owner1, owner2, players);
		
		if friendliness < -0.75 {
			(l.furiously.clone(), COLOR_PAIR(CRED))
		}else if friendliness < -0.5 {
			(l.angrily.clone(), COLOR_PAIR(CRED))
		}else if friendliness < -0.1 {
			(l.irritatedly.clone(), COLOR_PAIR(CRED))
		}else if friendliness < -0.25 {
			(l.bluntly.clone(), COLOR_PAIR(CWHITE))
		}else if friendliness < 0.1 {
			(l.neutrally.clone(), COLOR_PAIR(CWHITE))
		}else if friendliness < 0.5 {
			(l.pleasently.clone(), COLOR_PAIR(CGREEN1))
		}else if friendliness < 0.75 {
			(l.warmly.clone(), COLOR_PAIR(CGREEN1))
		}else{
			(l.enthusiastically.clone(), COLOR_PAIR(CGREEN1))
		}
	}
	
	// how owner1 sees owner2
	pub fn print_mood_action(&self, owner1: usize, owner2: usize, players: &Vec<Player>, dstate: &mut DispState) {
		let (txt, color) = self.mood_action_txt(owner1, owner2, players, &dstate.local);
		
		dstate.attron(color);
		dstate.addstr(&txt);
		dstate.attroff(color);
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
		let factor_sum = mood_toward.transient_and_permanent_factors(owner1, owner2, self, players).iter()
			.map(|f| f.mtype.weight(&self.config)).sum::<f32>();
		
		friendliness + factor_sum + mood_toward.val
	}
	
	// what are the factors influencing owner1's mood toward owner2?
	pub fn print_mood_factors(&self, owner1: usize, owner2: usize, players: &Vec<Player>,
			loc: ScreenCoord, dstate: &mut DispState) {
		let factors = &self.mood_toward[owner1*self.n_owners + owner2].transient_and_permanent_factors(owner1, owner2, self, players);
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

pub fn friendliness_thresh(level: f32) -> f32 {
	const FRIENDLINESS_VALS_SPACING: f32 = 2./3.;
	level*FRIENDLINESS_VALS_SPACING - 1.
}

impl TxtCategory {
	// how does owner1 see owner2?
	pub fn from_relations(relations: &Relations, owner1: usize, owner2: usize, players: &Vec<Player>) -> Self {
		let friendliness = relations.friendliness_toward(owner1, owner2, players);
		
		if friendliness < friendliness_thresh(1.) {
			TxtCategory::Negative
		}else if friendliness < friendliness_thresh(2.) {
			TxtCategory::Neutral
		}else{
			TxtCategory::Positive
		}
	}
}

