use super::*;
pub struct BudgetState {}

impl BudgetState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		const WINDOW_W: i32 = 50;
		let w_pos = dstate.print_window(ScreenSz{w: WINDOW_W as usize, h: 18, sz:0});
		
		let mut row = 0;
		let y = w_pos.y as i32 + 1;
		let x = w_pos.x as i32 + 2;
		macro_rules! mvl{() => {dstate.mv(row + y, x); row += 1;}}
		
		{ // title and exit instructions
			mvl!();
			dstate.buttons.Esc_to_close.print(None, &dstate.local, &mut dstate.renderer);
			
			dstate.mv(row + y, x + (WINDOW_W - dstate.local.Budget.len() as i32)/2);
			dstate.attron(COLOR_PAIR(CGREEN));
			dstate.renderer.addstr(&dstate.local.Budget);
			dstate.attroff(COLOR_PAIR(CGREEN));
		}
		
		row += 2; mvl!();
		dstate.renderer.addstr("Tax income:");
		
		mvl!();
		dstate.renderer.addstr("From your cities:");
		mvl!();
		dstate.renderer.addstr("From noble houses:");
		
		row += 1; mvl!();
		dstate.renderer.addstr("Unit costs:");
		
		row += 1; mvl!();
		dstate.renderer.addstr("Government bldg upkeep:");
		
		row += 1; mvl!();
		dstate.renderer.addstr("Discretionary infrastructure costs:");
		mvl!();
		dstate.renderer.addstr("Walls");
		mvl!();
		dstate.renderer.addstr("Roads");
		mvl!();
		dstate.renderer.addstr("Pipes");
	
		/*
			Tax income:
				From your cities:
				From noble houses:
				
			Units costs: 
			
			Government building upkeep:
			
			Infrastructure costs:
				Walls
				Roads
				Pipes
		*/
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self) -> UIModeControl<'bt,'ut,'rt,'dt> {
		UIModeControl::UnChgd
	}
}

