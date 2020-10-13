use super::*;

pub const CITY_GRID_HEIGHT: usize = 10;
pub const CITY_GRID_WIDTH: usize = 2*CITY_GRID_HEIGHT; // should always be twice the height (assumed by the construction code in new_city_plan() & create_city_grid_actions())
// ^ if the height to width relationship is changed for some reason, so should the FIEFDOM_ variables

pub const CITY_HEIGHT: usize = CITY_GRID_HEIGHT * GRID_SZ;
pub const CITY_WIDTH: usize = CITY_GRID_WIDTH * GRID_SZ;

#[derive(PartialEq, Clone, Default)]
pub struct EmpireState<'bt,'ut,'rt,'dt> {
	pub ai_state: AIState<'bt,'ut,'rt,'dt>,
	pub personality: AIPersonality
}

impl_saving!{EmpireState<'bt,'ut,'rt,'dt> {ai_state, personality}}

