use std::fmt;
use crate::saving::*;
use crate::gcore::{XorState};
use crate::nobility::House;
use crate::ai::{BarbarianState, AIState};
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::tech::TechTemplate;

#[derive(Clone, PartialEq)]
pub enum PlayerType<'bt,'ut,'rt,'dt> {
	Human {ai_state: AIState<'bt,'ut,'rt,'dt>},
	AI {
		ai_state: AIState<'bt,'ut,'rt,'dt>,
		personality: AIPersonality
	},
	Barbarian {barbarian_state: BarbarianState},
	Nobility {house: House<'bt,'ut,'rt,'dt>}
}

#[derive(Clone, PartialEq, Copy)]
pub struct AIPersonality {
	pub friendliness: f32, // negative friendliness is agression; range [-1:1]
	// ^ (1 - friendliness) -> proportionate to war declaration probability
	pub spirituality: f32, // negative spirituality is scientific; range [-1:1]
	// ^ (1 - spirituality)/2 -> probability of scientific buildings
}

impl_saving!{AIPersonality {friendliness, spirituality} }

///////////////////////// PlayerType
impl PlayerType<'_,'_,'_,'_> {
	pub fn is_ai(&self) -> bool {
		if let PlayerType::AI {..} = &self {true} else {false}
	}
	
	pub fn is_barbarian(&self) -> bool {
		if let PlayerType::Barbarian {..} = &self {true} else {false}
	}

	pub fn is_nobility(&self) -> bool {
		if let PlayerType::Nobility {..} = &self {true} else {false}
	}
	
	pub fn is_human(&self) -> bool {
		if let PlayerType::Human {..} = &self {true} else {false}
	}
}

impl <'bt,'ut,'rt,'dt>PlayerType<'bt,'ut,'rt,'dt> {
	pub fn ai_state(&self) -> Option<&AIState<'bt,'ut,'rt,'dt>> {
		match &self {
			PlayerType::AI {ai_state, ..} | PlayerType::Human {ai_state} => {
				Some(ai_state)
			}
			PlayerType::Barbarian {..} | PlayerType::Nobility {..} => {
				None
			}
		}
	}
	
	pub fn house(&self) -> Option<&House<'bt,'ut,'rt,'dt>> {
		match &self {
			PlayerType::Nobility {house} => {Some(house)}
			PlayerType::AI {..} | PlayerType::Human {..} |
			PlayerType::Barbarian {..} => {None}
		}
	}
	
	pub fn ai_state_mut<'s>(&'s mut self) -> Option<&'s mut AIState<'bt,'ut,'rt,'dt>> {
		match self {
			PlayerType::AI {ai_state, ..} | PlayerType::Human {ai_state} => {
				Some(ai_state)
			}
			PlayerType::Barbarian {..} | PlayerType::Nobility {..} => {
				None
			}
		}
	}
}

impl fmt::Display for PlayerType<'_,'_,'_,'_> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			PlayerType::Human {..} => {"Human"}
			PlayerType::AI {..} => {"AI"}
			PlayerType::Barbarian {..} => {"Barbarian"}
			PlayerType::Nobility {..} => {"Nobility"}
		})
	}
}

//////////////////////// AIPersonality
impl AIPersonality {
	pub fn new(rng: &mut XorState) -> Self {
		Self {
			friendliness: 2.*rng.gen_f32b() - 1., // between -1:1
			spirituality: 2.*rng.gen_f32b() - 1.
		}
	}
}

