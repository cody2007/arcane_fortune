use super::*;
use crate::nn::TxtPrinter;

pub enum ContactNobilityState {
	NobilitySelection {mode: usize}, // selection of houses in empire
	DialogSelection {mode: usize, owner_id: usize, quote_printer: TxtPrinter}
}

impl ContactNobilityState {
	pub fn print<'bt,'ut,'rt,'dt> (&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt>  {
		panicq!("todo");
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, dstate: &mut DispState<'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		panicq!("todo");
	}
}
