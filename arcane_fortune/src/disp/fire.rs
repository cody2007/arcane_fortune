use crate::buildings::{Bldg, BldgTemplate};
use crate::units::UnitTemplate;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::saving::*;
use crate::renderer::{CInt};//, endwin};
use std::time::Instant;
use crate::disp::color::*;
use crate::disp::ScreenSz;
use crate::gcore::XorState;

const FIRE_UPDATE_MS: usize = 100;
const UPDATE_PROB: f32 = 0.35;

#[derive(Clone, PartialEq)]
pub enum FireTile {
	Smoke,
	Fire {color: CInt},
	None
}

#[derive(Clone, PartialEq)]
pub struct Fire {
	pub t_updated: Instant,
	pub layout: Vec<FireTile>
}

impl_saving!{Fire {t_updated, layout}}

impl FireTile {
	pub fn new(row: usize, col: usize, bldg_sz: ScreenSz, rng: &mut XorState) -> Self {
		let half_w = (bldg_sz.w as f32 - 1.) / 2.;
		
		let hm1 = bldg_sz.h as f32 - 1.;
		
		// for flame prob
		let mut row_prob_factor = row as f32 / hm1;
		if row_prob_factor < 0.1 {row_prob_factor = 0.1;}
		
		// for smoke prob
		let mut smoke_prob = (hm1 - row as f32) / hm1;
		if smoke_prob > 0.15 {smoke_prob = 0.15;}
		
		let mut dist_prob = 1. - (col as f32 - half_w).abs() / half_w;
		// ^ zeros on sides, 1 at absolute center, drops off linearly from center
		
		const MAX_VAL: f32 = 0.95;
		const MIN_VAL: f32 = 0.6;
		
		if dist_prob > MAX_VAL {dist_prob = MAX_VAL;}
		if dist_prob < MIN_VAL {dist_prob = MIN_VAL;}
		
		// do not add fire, add smoke instead?
		if rng.gen_f32b() > (row_prob_factor*dist_prob) {
			if rng.gen_f32b() < smoke_prob {
				FireTile::Smoke
			}else{
				FireTile::None
			}
		}else{
			FireTile::Fire {color: CRED}
		}
	}
}

impl Fire {
	// colorize the fire based on # of fire neighbors
	pub fn colorize(&mut self, bldg_sz: ScreenSz) {
		for row in 0..bldg_sz.h {
		for col in 0..bldg_sz.w {
			let ind = row * bldg_sz.w + col;
			match self.layout[ind] {
				FireTile::Smoke | FireTile::None => {continue;}
				
				FireTile::Fire {..} => {
					let neighbor_present = |i, j| {
						let row2 = row as i32 + i;
						let col2 = col as i32 + j;
						// valid coord
						if row2 >= 0 && row2 < bldg_sz.h as i32 &&
						   col2 >= 0 && col2 < bldg_sz.w as i32 {
							let ind2 = row2 as usize*bldg_sz.w + col2 as usize;
							if let FireTile::Fire {..} = self.layout[ind2] {
								true
							   }else {false}
						}else{
							// if close to bottom, count edges
							// as also being a neighbor (fire)
							row2 >= bldg_sz.h as i32 - 2
						}
					};
					
					let mut n_neighbors = 0;
					
					for i in -1..=1 {
					for j in -1..=1 {
						if i == 0 && j == 0 {continue;}
						if neighbor_present(i,j) {
							n_neighbors += 1;
						}
					}}
					
					let color = if n_neighbors < 3 {
						CSAND1
					}else if n_neighbors < 4 {
						CSAND3
					}else if n_neighbors < 6 {
						CSAND4
					}else{
						CRED
					};
					
					self.layout[ind] = FireTile::Fire {color};
				}
			}
		}}
	}
}

impl Bldg<'_,'_,'_,'_> {
	pub fn create_fire(&mut self, rng: &mut XorState) {
		let sz = self.template.sz;
		let mut layout = Vec::with_capacity(sz.h*sz.w);
		
		// add fire and smoke
		for row in 0..sz.h {
		for col in 0..sz.w {
			layout.push(FireTile::new(row, col, sz, rng));
		}}
		
		let mut fire = Fire {
			t_updated: Instant::now(),
			layout
		};
		
		fire.colorize(sz);
		
		self.fire = Some(fire);
	}
	
	pub fn update_fire(&mut self, rng: &mut XorState) {
		if let Some(ref mut fire) = self.fire {
			if (fire.t_updated.elapsed().as_millis() as usize) < FIRE_UPDATE_MS {return;}
			
			let sz = self.template.sz;
			let hm1 = sz.h as f32 - 1.;
			
			// update random tiles only
			for row in 0..sz.h {
				let mut row_prob_factor = (hm1 - row as f32) / hm1;
				if row_prob_factor < 0.2 {row_prob_factor = 0.2;}
				
				for col in 0..sz.w {
					if rng.gen_f32b() > (row_prob_factor*UPDATE_PROB) {continue;}
					let ind = row * sz.w + col;
					fire.layout[ind] = FireTile::new(row, col, sz, rng);
				}
			}
			
			fire.colorize(sz);
			fire.t_updated = Instant::now();
			
		}//else{panicq!("no fire to update");}
	}
}

