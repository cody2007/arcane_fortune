use super::*;

#[derive(Clone, PartialEq)]
pub enum PlayerType<'bt,'ut,'rt,'dt> {
	Human(AIState<'bt,'ut,'rt,'dt>),
	Empire(EmpireState<'bt,'ut,'rt,'dt>),
	Barbarian(BarbarianState),
	Nobility(NobilityState<'bt,'ut,'rt,'dt>)
}

///////////////////////// PlayerType
impl PlayerType<'_,'_,'_,'_> {
	pub fn is_empire(&self) -> bool {
		if let PlayerType::Empire(_) = &self {true} else {false}
	}
	
	pub fn is_barbarian(&self) -> bool {
		if let PlayerType::Barbarian(_) = &self {true} else {false}
	}
	
	pub fn is_nobility(&self) -> bool {
		if let PlayerType::Nobility(_) = &self {true} else {false}
	}
	
	pub fn is_human(&self) -> bool {
		if let PlayerType::Human(_) = &self {true} else {false}
	}
	
	pub fn personality(&self) -> Option<AIPersonality> {
		if let Some(house) = self.house() {
			Some(house.head_personality())
		}else if let Some(empire_state) = self.empire_state() {
			Some(empire_state.personality)
		}else{
			None
		}
	}
	
	/*pub fn nm(&self) -> &str {
		match self {
			Self::Human(_) => "Human",
			Self::Empire(_) => "Empire",
			Self::Barbarian(_) => "Barbarian",
			Self::Nobility(_) => "Nobility"
		}
	}*/
}

impl Player<'_,'_,'_,'_> {
	pub fn req_population_center(&self, turn: usize) -> bool {
		const NO_CITY_HALL_GRACE_TURNS: usize = 100 + GAME_START_TURN;
		const NO_MANOR_GRACE_TURNS: usize = NO_CITY_HALL_GRACE_TURNS + NOBILITY_TURN_DELAY;
		match &self.ptype {
			PlayerType::Human(_) | PlayerType::Empire(_) => {
				turn > NO_CITY_HALL_GRACE_TURNS
			}
			PlayerType::Nobility(_) => {
				turn > (self.personalization.founded + NO_MANOR_GRACE_TURNS)
			}
			PlayerType::Barbarian(_) => false
		}
	}
}

impl <'bt,'ut,'rt,'dt>PlayerType<'bt,'ut,'rt,'dt> {
	pub fn empire_or_human_ai_state(&self) -> Option<&AIState<'bt,'ut,'rt,'dt>> {
		match &self {
			PlayerType::Empire(EmpireState {ai_state, ..}) | PlayerType::Human(ai_state) => {
				Some(ai_state)
			}
			PlayerType::Barbarian(_) | PlayerType::Nobility(_) => {
				None
			}
		}
	}
	
	pub fn house(&self) -> Option<&House> {
		match &self {
			PlayerType::Nobility(NobilityState {house, ..}) => {Some(house)}
			PlayerType::Empire(_) | PlayerType::Human(_) | PlayerType::Barbarian(_) => {None}
		}
	}
	
	pub fn house_mut(&mut self) -> Option<&mut House> {
		match self {
			PlayerType::Nobility(NobilityState {house, ..}) => {Some(house)}
			PlayerType::Empire(_) | PlayerType::Human(_) |
			PlayerType::Barbarian(_) => {None}
		}
	}
	
	pub fn empire_state(&self) -> Option<&EmpireState<'bt,'ut,'rt,'dt>> {
		match self {
			PlayerType::Empire(empire_state) => {
				Some(empire_state)
			}
			PlayerType::Nobility(_) | PlayerType::Human(_) | PlayerType::Barbarian(_) => {
				None
			}
		}
	}
	
	pub fn nobility_state(&self) -> Option<&NobilityState<'bt,'ut,'rt,'dt>> {
		match self {
			PlayerType::Nobility(nobility_state) => {
				Some(nobility_state)
			}
			PlayerType::Empire(_) | PlayerType::Human(_) | PlayerType::Barbarian(_) => {
				None
			}
		}
	}
	
	pub fn any_ai_state(&self) -> Option<&AIState<'bt,'ut,'rt,'dt>> {
		match self {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) |
			PlayerType::Human(ai_state) => {
				Some(ai_state)
			}
			PlayerType::Barbarian(_) => {
				None
			}
		}
	}
	
	pub fn any_ai_state_mut<'s>(&'s mut self) -> Option<&'s mut AIState<'bt,'ut,'rt,'dt>> {
		match self {
			PlayerType::Empire(EmpireState {ai_state, ..}) |
			PlayerType::Nobility(NobilityState {ai_state, ..}) |
			PlayerType::Human(ai_state) => {
				Some(ai_state)
			}
			PlayerType::Barbarian(_) => {
				None
			}
		}
	}
}

/*impl fmt::Display for PlayerType<'_,'_,'_,'_> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			PlayerType::Human {..} => {"Human"}
			PlayerType::AI {..} => {"AI"}
			PlayerType::Barbarian {..} => {"Barbarian"}
			PlayerType::Nobility {..} => {"Nobility"}
		})
	}
}*/

//////////////////////// AIPersonality
impl AIPersonality {
	pub fn new(rng: &mut XorState) -> Self {
		Self {
			friendliness: 2.*rng.gen_f32b() - 1., // between -1:1
			spirituality: 2.*rng.gen_f32b() - 1.
		}
	}
}

