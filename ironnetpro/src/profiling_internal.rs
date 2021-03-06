use std::time::Instant;

pub struct StackElapsed {
	stack_nm: Vec<&'static str>,
	elapsed: u64
}

pub struct StackStart {
	stack_nm: Vec<&'static str>,
	start_instant: Instant
}

static mut PREV_STACKS_ELAPSED: Vec<StackElapsed> = Vec::new();
static mut CUR_STACKS: Vec<StackStart> = Vec::new();

pub struct Guard {}

impl Guard {
	pub fn new(nm: &'static str) -> Self {
		unsafe {
			let mut stack_nm = if let Some(cur_stack) = CUR_STACKS.last() {
				cur_stack.stack_nm.clone()
			}else{
				Vec::new()
			};
			
			stack_nm.push(nm);
			
			CUR_STACKS.push(StackStart {
					stack_nm,
					start_instant: Instant::now()
					});
		}
		Guard {}
	}
}

impl Drop for Guard {
	fn drop(&mut self) {
		unsafe {
			let cur_stack = CUR_STACKS.pop().unwrap();
			let t = cur_stack.start_instant.elapsed().as_nanos() as u64;
			
			if let Some(stack_timer) = PREV_STACKS_ELAPSED.iter_mut().rev().
				find(|st| st.stack_nm == cur_stack.stack_nm) {
					stack_timer.elapsed += t;
			}else{
				PREV_STACKS_ELAPSED.push(StackElapsed {
						stack_nm: cur_stack.stack_nm.clone(),
						elapsed: t
				});
			}
		}
	}
}

pub fn write_prof() {
	unsafe {
		for prev_stack in PREV_STACKS_ELAPSED.iter() {
			let mut nm = String::new();
			for p in prev_stack.stack_nm.iter() {
				nm.push_str(" -> ");
				nm.push_str(p);
			}
			println!("{:.1} {}", prev_stack.elapsed as f32/1_000_000., nm);
		}
	}
}

