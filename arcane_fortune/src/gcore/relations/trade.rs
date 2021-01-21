use super::*;
use crate::map::*;

// where player_i's owner_index is less than player_j's
#[derive(Clone, Default)]
pub struct TradeDeal {
	player_i_gives: Vec<TradeItem>,
	player_j_gives: Vec<TradeItem>,
	pub turn_started: usize
}

impl_saving!{TradeDeal {player_i_gives, player_j_gives, turn_started}}

#[derive(Clone, PartialEq)]
pub enum TradeItem {
	LumpGold(f32),
	GoldPerTurn(f32),
	Resource(usize),
	Tech(usize),
	DefensivePact,
	WorldMap
}

impl TradeItem {
	// execute the trade
	pub fn give_item<'bt,'ut,'rt,'dt>(&self, sender_ind: usize, receiver_ind: usize, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
			temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) {
		match self {
			TradeItem::LumpGold(gold) => {
				players[receiver_ind].stats.gold += gold;
				players[sender_ind].stats.gold -= gold;
			}
			TradeItem::GoldPerTurn(_) |
			TradeItem::DefensivePact => {}
			TradeItem::Resource(resource) => {
				players[receiver_ind].stats.resources_avail[*resource] += 1;
				players[sender_ind].stats.resources_avail[*resource] -= 1;
			}
			TradeItem::Tech(tech_ind) => {
				players[receiver_ind].stats.discover_tech(*tech_ind, temps, dstate);
			}
			TradeItem::WorldMap => {
				let land_discov_full = players[sender_ind].stats.land_discov.last().unwrap().clone();
				let pstats = &mut players[receiver_ind].stats;
				for coord in LandDiscovIter::from(&land_discov_full) {
					compute_zooms_discover(coord, pstats, map_data);
				}
			}
		}
	}
	
	// occurs over time ex. gold per turn or resources
	fn ongoing_item(&self) -> bool {
		match self {
			TradeItem::GoldPerTurn(_) |
			TradeItem::Resource(_) |
			TradeItem::DefensivePact => true,
			TradeItem::LumpGold(_) |
			TradeItem::Tech(_) |
			TradeItem::WorldMap => {false}
		}
	}
	
	// restore resources
	pub fn rescend_item(&self, sender_ind: usize, receiver_ind: usize, players: &mut Vec<Player>) {
		match self {
			TradeItem::LumpGold(_) |
			TradeItem::GoldPerTurn(_) |
			TradeItem::Tech(_) |
			TradeItem::DefensivePact |
			TradeItem::WorldMap => {}
			TradeItem::Resource(resource_ind) => {
				players[sender_ind].stats.resources_avail[*resource_ind] += 1;
				players[receiver_ind].stats.resources_avail[*resource_ind] -= 1;
			}
		}
	}
	
	fn gold_per_turn(&self) -> Option<f32> {
		match self {
			TradeItem::GoldPerTurn(gold) => Some(*gold),
			TradeItem::DefensivePact |
			TradeItem::LumpGold(_) |
			TradeItem::Resource(_) |
			TradeItem::Tech(_) |
			TradeItem::WorldMap => None
		}
	}
	
	fn defensive_pact(&self) -> bool {
		match self {
			TradeItem::LumpGold(_) |
			TradeItem::GoldPerTurn(_) |
			TradeItem::Resource(_) |
			TradeItem::Tech(_) |
			TradeItem::WorldMap => false,
			TradeItem::DefensivePact => true
		}
	}
}

impl TradeDeal {
	pub fn new(turn: usize) -> Self {
		Self {
			player_i_gives: Vec::new(),
			player_j_gives: Vec::new(),
			turn_started: turn
		}
	}
	
	// owner1 gives to owner2
	pub fn player_gives(&self, owner1: usize, owner2: usize) -> &Vec<TradeItem> {
		debug_assertq!(owner1 != owner2);
		if owner1 < owner2 {
			&self.player_i_gives
		}else{
			&self.player_j_gives
		}
	}
	
	// owner1 gives to owner2
	pub fn add_item(&mut self, item: TradeItem, owner1: usize, owner2: usize) {
		debug_assertq!(owner1 != owner2);
		
		// add to both players
		if item == TradeItem::DefensivePact {
			self.player_i_gives.push(item.clone());
			self.player_j_gives.push(item);
		// asymmetric trade item
		}else{
			if owner1 < owner2 {
				self.player_i_gives.push(item);
			}else{
				self.player_j_gives.push(item);
			}
		}
	}
	
	pub fn nm(&self, l: &Localization) -> String {
		l.Trade_agreement_of.replace("[]", &l.date_str(self.turn_started))
	}
	
	// none indicates the deal can be canceled at any time
	pub fn expires(&self, turn: usize) -> Option<usize> { //, config: &RelationsConfig) -> Option<usize> {
		let expires = MIN_TREATY_DAYS as usize + self.turn_started;
		if expires < turn {
			None
		}else{
			Some(expires)
		}
	}
	
	// trade is ongoing and not a one time deal (ex. gold per turn or resources are exchanged)
	fn ongoing(&self) -> bool {
		self.player_i_gives.iter().any(|item| item.ongoing_item()) ||
		self.player_j_gives.iter().any(|item| item.ongoing_item())
	}
	
	fn contains_defensive_pact(&self) -> bool {
		self.player_i_gives.iter().any(|trade_item| trade_item.defensive_pact()) ||
		self.player_j_gives.iter().any(|trade_item| trade_item.defensive_pact())
	}
}

impl Relations {
	// player1 & player2 have a defensive pact together. ordering of the players doesn't matter
	pub fn defensive_pact(&self, player1: usize, player2: usize) -> bool {
		{ // return true if fiefdom or kingdom
			match self.status(player1, player2) {
				RelationStatus::Kingdom {..} |
				RelationStatus::Fiefdom {..} => {return true;}
				RelationStatus::Undiscovered |
				RelationStatus::War {..} |
				RelationStatus::Peace(_) => {}
			}
		}
		
		if let Some(trade_deals) = self.status(player1, player2).trade_deals() {
			return trade_deals.iter().any(|trade_deal| trade_deal.contains_defensive_pact());
		}
		false
	}
	
	// players which have a defensive pact w/ player1 (or are a fiefdom of the player)
	pub fn defensive_pact_owners(&self, player1: usize) -> Vec<usize> {
		let mut defensive_pact_players = Vec::with_capacity(self.n_owners);
		
		for player_chk in (0..self.n_owners).filter(|&player_chk| player_chk != player1) {
			// fiefdom
			if self.fiefdom(player1, player_chk) {
				defensive_pact_players.push(player_chk);
			// defensive pact
			}else if let Some(trade_deals) = self.status(player1, player_chk).trade_deals() {
				if trade_deals.iter().any(|trade_deal| trade_deal.contains_defensive_pact()) {
					defensive_pact_players.push(player_chk);
				}
			}
		}
		defensive_pact_players
	}
	
	// the oredering of the players doesn't matter
	pub fn rm_trade(&mut self, trade_ind: usize, player1: usize, player2: usize, 
			players: &mut Vec<Player>, turn: usize) {
		let (omin, omax) = order_indices(player1, player2);
		let pair_ind = omin*self.n_owners + omax;
		if let Some(trade_deals) = self.relation_status[pair_ind].trade_deals_mut() {
			if let Some(trade_deal) = trade_deals.get(trade_ind) {
				if trade_deal.expires(turn).is_none() {
					{ // restore items (resources)
						for trade_item in trade_deal.player_i_gives.iter() {
							trade_item.rescend_item(omin, omax, players);
						}
						for trade_item in trade_deal.player_j_gives.iter() {
							trade_item.rescend_item(omax, omin, players);
						}
					}
					
					trade_deals.swap_remove(trade_ind);
				}
			}
		}
	}
	
	// does not check if the AI wants the trade or not (use propose_trade for that)
	pub fn add_trade<'bt,'ut,'rt,'dt>(&mut self, trade_deal: &TradeDeal, proposer_ind: usize,
			other_player_ind: usize, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>, turn: usize) {
		let mut trade_deal_copy = trade_deal.clone();
		trade_deal_copy.turn_started = turn; // if auto-turn is enabled the turn stored here could be out of date
		
		if let Some(trade_deals) = self.status_mut(other_player_ind, proposer_ind).trade_deals_mut() {
			// proposer gives other player
			for trade_item in trade_deal_copy.player_gives(proposer_ind, other_player_ind) {
				trade_item.give_item(proposer_ind, other_player_ind, players, temps, map_data, dstate);
			}
			
			// other player gives proposer
			for trade_item in trade_deal_copy.player_gives(other_player_ind, proposer_ind) {
				trade_item.give_item(other_player_ind, proposer_ind, players, temps, map_data, dstate);
			}
			
			if trade_deal_copy.ongoing() {
				trade_deals.push(trade_deal_copy);
			}
		}else{panicq!("unexpected relation status when proposing treaty");}
	}
	
	// returns true if the trade was accepted
	// adds trade if accepted
	pub fn propose_trade<'bt,'ut,'rt,'dt>(&mut self, trade_deal: &TradeDeal, proposer_ind: usize,
			other_player_ind: usize, players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>, turn: usize) -> bool {
		let (proposer_give_val, other_player_give_val) = {
			let proposer_gives = trade_deal.player_gives(proposer_ind, other_player_ind);
			let other_player_gives = trade_deal.player_gives(other_player_ind, proposer_ind);
			
			(trade_value(proposer_gives, proposer_ind, other_player_ind, players, temps),
			 trade_value(other_player_gives, other_player_ind, proposer_ind, players, temps))
		};
		
		// accept trade
		if proposer_give_val > other_player_give_val {
			self.add_trade(trade_deal, proposer_ind, other_player_ind, players, temps, map_data, dstate, turn);
			
			// the trade is trade too good
			if (proposer_give_val / other_player_give_val) > self.config.trade_reasonableness_threshold {
				self.add_mood_factor(proposer_ind, other_player_ind, MoodType::YouAreEasilyTakenAdvantageOf, turn);
			}
			
			true
		// reject trade
		}else{
			// the trade is insultingly bad
			if (other_player_give_val / proposer_give_val) > self.config.trade_reasonableness_threshold {
				self.add_mood_factor(proposer_ind, other_player_ind, MoodType::YourOfferInsultedUs, turn);
			}
			
			false
		}
	}
	
	// returns gold received, negative if gives
	pub fn trade_gold_per_turn(&self, player_i: usize) -> f32 {
		let mut net_recv = 0.;
		for player_j in player_i..self.n_owners {
			if let Some(trade_deals) = self.status(player_i, player_j).trade_deals() {
				for trade_deal in trade_deals.iter() {
					// player_i gives player_j
					for trade_item in trade_deal.player_gives(player_i, player_j) {
						if let Some(gold) = trade_item.gold_per_turn() {
							net_recv -= gold;
						}
					}
					
					// player_j gives_player_i
					for trade_item in trade_deal.player_gives(player_j, player_i) {
						if let Some(gold) = trade_item.gold_per_turn() {
							net_recv += gold;
						}
					}
				}
			}
		}
		net_recv
	}
}

const MIN_TREATY_DAYS: f32 = 5.*(TURNS_PER_YEAR as f32);
const GOLD_PER_TECH_POINT: f32 = 50.;
const GOLD_PER_WORLD_MAP_TILE: f32 = 10.;
const GOLD_PER_RESOURCE_VALUE: f32 = 100.;
const GOLD_PER_DEFENSE_PACT_VALUE: f32 = 5000.;

fn trade_value(offers: &Vec<TradeItem>, proposer_ind: usize, receiver_ind: usize,
		players: &Vec<Player>, temps: &Templates) -> f32 {
	offers.iter().fold(0., |sum, offer|
		sum + match offer {
			TradeItem::LumpGold(gold) => {*gold}
			TradeItem::GoldPerTurn(gpt) => {*gpt*MIN_TREATY_DAYS}
			TradeItem::Resource(resource_ind) => {
				temps.resources[*resource_ind].ai_valuation * GOLD_PER_RESOURCE_VALUE
			} TradeItem::Tech(tech_ind) => {
				temps.techs[*tech_ind].research_req as f32 * GOLD_PER_TECH_POINT
			} TradeItem::WorldMap => {
				players[proposer_ind].stats.land_discov.last().unwrap().n_discovered as f32 * GOLD_PER_WORLD_MAP_TILE
			} TradeItem::DefensivePact => {
				if let Some(proposer_defense) = players[proposer_ind].stats.defense_power_log.last() {
				if let Some(receiver_defense) = players[receiver_ind].stats.defense_power_log.last() {
					return GOLD_PER_DEFENSE_PACT_VALUE * (*proposer_defense as f32 / *receiver_defense as f32);
				}}
				0.
			}
		}
	)
}

