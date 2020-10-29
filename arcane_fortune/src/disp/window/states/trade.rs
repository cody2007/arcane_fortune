use super::*;

pub struct TradeState {
	player_id: usize,
	cur_player_lump_gold_offered: usize,
	cur_player_gold_per_turn_offered: usize,
	cur_player_offerings: TradeOfferings,
	other_player_offerings: TradeOfferings
}

struct TradeOfferings {
	resources: Vec<usize>, // index into resource_templates
	techs: Vec<usize>, // index into tech_templates
	world_map: bool
}

impl TradeState {
	pub fn new(player_id: usize) -> Self {
		Self {
			player_id,
			cur_player_lump_gold_offered: 0,
			cur_player_gold_per_turn_offered: 0,
			cur_player_offerings: Default::default(),
			other_player_offerings: Default::default()
		}
	}
	
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let w = 70;
		let w_pos = dstate.print_window(ScreenSz{w, h: 5+2+3+2+2, sz:0});
			
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self) -> UIModeControl<'bt,'ut,'rt,'dt> {
		UIModeControl::UnChgd
	}
}

impl Default for TradeOfferings {
	fn default() -> Self {
		Self {
			resources: Vec::new(),
			techs: Vec::new(),
			world_map: false
		}
	}
}

