use super::*;
pub struct RatioEntry {
	pub frac: f32,
	pub color: chtype
}

use std::f32::consts::PI;
pub fn print_circle_plot(radius: i32, pos: Coord, ratios: &Vec<RatioEntry>,
		disp_chars: &DispChars, d: &mut DispState) {
	debug_assertq!(ratios.iter().fold(0., |sum, i| sum + i.frac) == 1.);
	
	let quad = |y: f32, x: f32| {
		if y >= 0. && x >= 0. {1} else
		if y >= 0. && x <= 0. {2} else
		if y <= 0. && x <= 0. {3} else {4}
	};
	
	let mut theta_sum = 0.;
	for ratio in ratios.iter() {
		let theta_sum_prev = theta_sum;
		theta_sum += 2.*ratio.frac*PI;
		
		// line constraints
		let y_constraint_prev = |x: f32| {x * (theta_sum_prev.tan())};
		let y_constraint = |x: f32| {x * (theta_sum.tan())};
		
		let q_prev = quad(theta_sum_prev.sin(), theta_sum_prev.cos());
		let q = if theta_sum >= (2.*PI) {4} else {quad(theta_sum.sin(), theta_sum.cos())};
		
		d.attron(ratio.color);
		for y in (-radius)..radius {
			// r^2 = y^2 + x^2
			const X_EXPANSION: f32 = 2.;
			let x_max = (((radius*radius - y*y) as f32).sqrt() * X_EXPANSION).round() as i32;
			for x in (-x_max)..x_max {
				let x_pre_expansion = x as f32 / X_EXPANSION;
				let current_quad = quad(y as f32, x_pre_expansion);
				
				#[derive(PartialEq)]
				enum Bound {Val(f32), None, All};
				let quad_bound_prev = 
					if current_quad < q_prev {
						Bound::None
					}else if current_quad == q_prev {
						Bound::Val(y_constraint_prev(x_pre_expansion))
					}else{
						Bound::All
					};
				
				let quad_bound = 
					if current_quad < q {
						Bound::All
					}else if current_quad == q {
						Bound::Val(y_constraint(x_pre_expansion))
					}else{
						Bound::None
					};
				
				if !match quad_bound_prev {
					Bound::None => {false}
					Bound::All => {true}
					Bound::Val(y_bound) => {
						if q_prev == 1 || q_prev == 4 {
							y_bound <= y as f32
						}else{
							y as f32 <= y_bound
						}
					}
				} {continue;}
				
				if !match quad_bound {
					Bound::None => {false}
					Bound::All => {true}
					Bound::Val(y_bound) => {
						if q == 1 || q == 4 {
							y as f32 <= y_bound
						}else{
							y_bound <= y as f32
						}
					}
				} {continue;}
				
				d.mv(pos.y as i32 + y + radius, pos.x as i32 + radius*X_EXPANSION as i32+ x);
				d.addch(disp_chars.land_char);
			} // x
		} // y
		d.attroff(ratio.color);
		//break;
	} // ratio
}

