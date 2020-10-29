use std::fmt;
use std::cmp::{min,max};
use crate::gcore::rand::*;
use crate::map::MapType;
use crate::resources::ResourceTemplate;
use crate::renderer::endwin;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

use super::ZOOM_IND_ROOT;
use super::vars::*;

pub const REGION_SZ: usize = 9;

// fill_point() will set a value for region[y,x] based on taking the average of points up to 
// a DIST_ADD points away (using the manhattan distance)
#[inline]
fn fill_point(region: &mut Vec<f32>, y: isize, x: isize, rng: &mut XorState){
	const SIM: f32 = 0.85; // higher values -> more smoothing; lower values -> more random
	const DIST_ADD: isize = 5;

	let mut sum = 0.;
	let mut n_sum = 0;
	
	let j_start = max(0, x-DIST_ADD) as usize;
	let j_diff = min((REGION_SZ-1) as isize, x+DIST_ADD) as usize - j_start + 1;
	
	for i in max(0, y-DIST_ADD)..=min((REGION_SZ-1) as isize, y+DIST_ADD) {
		let i_off = (i as usize)*REGION_SZ;
		
		for val in region.iter().skip(i_off + j_start).take(j_diff).filter(|&&val| val != 0.) {
			sum += val;
			n_sum += 1;
		}
	}
	
	region[y as usize*REGION_SZ + x as usize] = SIM*sum/(n_sum as f32) + (1.-SIM)*rng.gen_f32b();
}

// fill around (center_y, center_x) with manhattan distance `dist`
// each point filled, is computed based on the values of points up to `DIST_ADD` away
// from the to-be-filled point.
// higher `SIM` values indicate more smoothing and less random noise
#[inline]
fn fill_points(region: &mut Vec<f32>, dist: usize, center_y: usize, center_x: usize, rng: &mut XorState){
	let mut perim = Vec::with_capacity(8*dist);
	// ^ indices of [y,x] for each point along the perimieter, at a distance of `dist` from [center_y, center_x]
	
	macro_rules! add_perim{($y: expr, $x: expr) => {
		if $y < REGION_SZ && $x < REGION_SZ  && // in bounds
		   region[$y*REGION_SZ + $x] == 0. { // and location unitialized
			perim.push([$y, $x]);
		}
	};};
	
	// corners
	add_perim!(center_y + dist,   center_x + dist);
	add_perim!(center_y + dist,   center_x - dist);
	add_perim!(center_y - dist,   center_x + dist);
	add_perim!(center_y - dist,   center_x - dist);
	
	for i in 1..(2*dist) {
		// left and right columns
		add_perim!(center_y - dist + i,   center_x - dist);
		add_perim!(center_y - dist + i,   center_x + dist);
		
		// top and bottom rows
		add_perim!(center_y - dist,   center_x - dist + i);
		add_perim!(center_y + dist,   center_x - dist + i);
	}
	
	/*if perim.len() == 0 {
		println!("dist: {}   y,x: {},{}", dist, center_y, center_x);
	}*/
	debug_assertq!(perim.len() <= (8*dist), "{} {}", perim.len(), 8*dist);
	
	rng.shuffle(&mut perim);
	
	// generate each perimeter point:
	for ind in perim {
		
		// for each neighbor of the perimeter point, sum points up to `DIST_ADD` away
		fill_point(region, ind[0] as isize, ind[1] as isize, rng);
		
	}
}

//           ct
//     ul   top    ur
// cl left center right cr
//     ll  bottom  lr
//           cb
struct RegionSkeleton {
	center: f32,
	left: f32, right: f32, top: f32, bottom: f32,
	ul: f32, ll: f32, ur: f32, lr: f32,
	cl: f32, ct: f32, cr: f32, cb: f32
}

// gen region of size REGION_SZ x REGION_SZ. the corners and mid-points of the perimeter are from `vals`
// `row` and `col` are the locations within `vals` to use (denoating the center). `w` is the width of `vals`
// having dimensions: [h, w]
impl RegionSkeleton {
	#[inline]
	fn gen_region(&self, row: usize, col: usize, rand_off: u64) -> Vec<f32>{
		let mut region = vec!{0.; REGION_SZ*REGION_SZ};
		let mut perimeter_sum = self.ul + self.ll + self.ur + self.right + self.lr + self.top + self.bottom;
		
		/////// set corners & mid-points from amounts stored from `vals`
		region[0] = self.ul;
		region[(REGION_SZ/2)*REGION_SZ] = self.left; // mid-way down column
		region[REGION_SZ-1] = self.ll;
		
		region[REGION_SZ-1] = self.ur;
		region[(REGION_SZ/2)*REGION_SZ + REGION_SZ-1] = self.right; // mid-way down column
		region[(REGION_SZ-1)*REGION_SZ + REGION_SZ/2] = self.lr;
		
		region[REGION_SZ/2] = self.top;
		region[(REGION_SZ-1)*REGION_SZ + REGION_SZ/2] = self.bottom;
		
		// set lines along edges. skip corners and mid-points (they are defined by `vals`)
		// set values to be mean of mid point and the corners of the line, plus a random f32.
		{
			const SIM_EDGES: f32 = 0.9;//6;
			
			macro_rules! set_edge{($seed:expr, $orth:expr, $mid:expr, $left:expr, $right:expr,
				$row:expr, $col:expr, $roff:expr, $coff:expr) => {
			
			let mut rng = XorState::init(rand_off + $seed as u64);
			
			// left of mid-point
			for i in 1..(REGION_SZ/2) {
				let dist = ((REGION_SZ/2) - i) as f32 / (REGION_SZ/2) as f32;
				let l_and_m = $left*dist + $mid*(1.-dist);
				let val = SIM_EDGES * ($orth+self.center+2.*l_and_m)/4. + (1.-SIM_EDGES)*rng.gen_f32b();
				
				//if i == 1 {println!("{} {} {}", val, center, l_and_m);}
				
				region[i*$row*REGION_SZ + $roff*REGION_SZ + i*$col + $coff] = val;
				perimeter_sum += val;
			}
			// right of mid-point
			for i in (1+(REGION_SZ/2))..REGION_SZ {
				let dist = (REGION_SZ - i) as f32 / REGION_SZ as f32;
				let r_and_m = $right*dist + $mid*(1.-dist);
				let val = SIM_EDGES * ($orth+self.center+2.*r_and_m)/4. + (1.-SIM_EDGES)*rng.gen_f32b();
				
				region[i*$row*REGION_SZ + $roff*REGION_SZ + i*$col + $coff] = val;
				perimeter_sum += val;
			}
			};};
			
			set_edge!((row-1)*TRACT_IN_SZ + col, self.ct, self.top, self.ul, self.ur, 0, 1, 0, 0); // top row
			set_edge!((row+1)*TRACT_IN_SZ + col, self.cb, self.bottom, self.ll, self.lr, 0, 1, REGION_SZ-1, 0); // bottom row
			set_edge!(row*TRACT_IN_SZ + col-1, self.cl, self.left, self.ul, self.ll, 1, 0, 0, 0); // left col
			set_edge!(row*TRACT_IN_SZ + col+1, self.cr, self.right, self.ur, self.lr, 1, 0, 0, REGION_SZ-1); // right col
		}
		
		// set center
		{
			let mut rng = XorState::init(rand_off + (row*TRACT_IN_SZ + col) as u64);
			const PAD: usize = 1;
			let y = rng.usize_range(PAD, REGION_SZ-PAD-1);
			let x = rng.usize_range(PAD, REGION_SZ-PAD-1);
			region[y*REGION_SZ + x] = self.center;

			//////////////////////////
			// fill around sides of center (border of edge)
			for dist in (1..(REGION_SZ/2)).rev() {
				fill_points(&mut region, dist, REGION_SZ/2, REGION_SZ/2, &mut rng);
			}
			
			fill_point(&mut region, (REGION_SZ/2) as isize, (REGION_SZ/2) as isize, &mut rng);
		}
		
		//////////////////////////
		// smooth interior
		let mut interior_sum = 0.;
		{
			const SMOOTH_DIST: isize = 1;
			const SMOOTH_INTERIOR: f32 = 0.1;
			for i in 1..(REGION_SZ-1) {
				for j in 1..(REGION_SZ-1) {
					let mut sum = 0.;
					let mut n_sum = 0;
					
					for i2 in -SMOOTH_DIST..=SMOOTH_DIST {
						let i_use = i2 + i as isize;
						if i_use < 0 || i_use >= REGION_SZ as isize {continue;} // out of bounds
						for j2 in -SMOOTH_DIST..=SMOOTH_DIST {
							let j_use = j2 + j as isize;
							if j_use < 0 || j_use >= REGION_SZ as isize {continue;} // out of bounds
							
							sum += region[(i_use as usize)*REGION_SZ + (j_use as usize)];
							n_sum += 1;
						} // j2
					} // i2
					
					let val = (1.-SMOOTH_INTERIOR) * region[i*REGION_SZ + j] + 
								  SMOOTH_INTERIOR * sum/(n_sum as f32);
					
					//region[i*REGION_SZ + j] = val;
					interior_sum += val;
				} // j
			} // smooth interior i
		}
		
		///////////////////
		// subtract mean of perimiter from interior
		const PERIMETER_SZ: usize = (REGION_SZ)*4 - 4;
		let perimeter_mean = perimeter_sum / PERIMETER_SZ as f32;
		let interior_mean = interior_sum / ((REGION_SZ*REGION_SZ) - PERIMETER_SZ) as f32;
		for (ind, b) in region.iter_mut().enumerate() {
			let i = ind / REGION_SZ;
			let j = ind % REGION_SZ;
			if i == 0 || i == (REGION_SZ-1) || j == 0 || j == (REGION_SZ) {
				continue;
			}
			*b += perimeter_mean - interior_mean;
		}
		
		region
	}
}

const N_REGIONS_PER_TRACT: usize = 12;
const TRACT_OVERLAP: usize = 4; // on the parent, how much do tracts overlap when generating the child 
				// (TRACT_IN_SZ - TRACT_OVERLAP) is the spacing between tract positions on the parent

const TRACT_IN_SZ: usize = N_REGIONS_PER_TRACT*2 + TRACT_OVERLAP;
			// size of one dimension (of two)
			// why N_REGIONS_PER_TRACT*2? because:
			// gen_region() [called by gen_tract()] expects a 3x3 input.
			// see gen_region(), a 'top', 'center' and 'bottom' (among other) values are taken from the input.
			// it is desirable for the 'bottom' of one region to be similar to the 'top' of the one below.
			// for this reason, the spacing is regions are spaced apart by 2 on the parent instead of every 3.
			// (ex. REGION_IN_SZ - REGION_OVERLAP) = (3 - 1) = 2

const TRACT_OUT_SZ: usize = N_REGIONS_PER_TRACT*(REGION_SZ-1) + 1; // size of one dimension (of two)

pub const H_ROOT: usize = 19*(TRACT_IN_SZ - TRACT_OVERLAP) + TRACT_IN_SZ; //484; //500;
pub const W_ROOT: usize = 80*(TRACT_IN_SZ - TRACT_OVERLAP) + TRACT_IN_SZ; //H_ROOT*4;//1992; //H_ROOT*4;

#[inline]
pub fn upsample_dim(h: usize) -> (usize, usize) {
	let mut n_tracts = 1;
	
	let denom_val = |n_tracts| -> usize {
		if n_tracts != 0 {
			(n_tracts-1)*(TRACT_IN_SZ - TRACT_OVERLAP) + TRACT_IN_SZ
		}else{
			0
		}
	};
	
	loop {
		if denom_val(n_tracts) > h {break;}
		n_tracts += 1;
	}
	n_tracts -= 1;
	let n_round_off = if denom_val(n_tracts) != 0 {h % denom_val(n_tracts)} else {0};
	//endwin();
	//println!("n_tracts {} h {} round off {}", n_tracts, h, h % (TRACT_IN_SZ - TRACT_OVERLAP));
	(n_tracts * TRACT_OUT_SZ, n_round_off)
}

// each tract is composed of multiple regions [gen_region()], which are then mean matched locally
// thresholding is done to determine land/water, and then the elevation is 
// mean and range matched globally across the tract.
fn gen_tract<'rt>(mut map_in: Vec<Map<'rt>>, rand_off: u64, resource_templates: &'rt Vec<ResourceTemplate>) -> Vec<Map<'rt>> {
	debug_assertq!(map_in.len() == (TRACT_IN_SZ*TRACT_IN_SZ));
	let mut map_out = vec!{Map {
			arability: 0.,
			elevation: 0.,
			show_snow: false,
			map_type: MapType::Land,
			resource: None,
			resource_cont: None
		}; TRACT_OUT_SZ*TRACT_OUT_SZ};
	
	struct Range {max: f32, min: f32};
	
	// generates map_out[:].$vals_out, and fills $range_in (instance of the Range struct)
	macro_rules! gen_floats{($vals_out: ident, $range_in: ident) => {
		//////////// normalize (mu = 0, std = 1)
		$range_in = Range {
			max: map_in[0].$vals_out, 
			min: map_in[0].$vals_out
		};
		for mi in map_in[1..].iter() {
			if $range_in.min > mi.$vals_out {$range_in.min = mi.$vals_out;}
			if $range_in.max < mi.$vals_out {$range_in.max = mi.$vals_out;}
		}
		
		for mi in map_in.iter_mut() {
			mi.$vals_out -= $range_in.min;
			mi.$vals_out /= $range_in.max - $range_in.min;
		}

		// gen regions. space apart so we can feed "+" as points and have the tops
		// and bottoms of each region be the same at neighboring regions
		for region_i in 0..N_REGIONS_PER_TRACT { let row = region_i*2 + 2; let roff = region_i*(REGION_SZ-1);
		for region_j in 0..N_REGIONS_PER_TRACT { let col = region_j*2 + 2; let coff = region_j*(REGION_SZ-1);
			
			//           ct
			//     ul   top    ur
			// cl left center right cr
			//     ll  bottom  lr
			//           cb

			let skeleton = RegionSkeleton {
				center: map_in[row*TRACT_IN_SZ + col].$vals_out,

				left: map_in[row*TRACT_IN_SZ + col-1].$vals_out,
				right: map_in[row*TRACT_IN_SZ + col+1].$vals_out,
				top: map_in[(row-1)*TRACT_IN_SZ + col].$vals_out,
				bottom: map_in[(row+1)*TRACT_IN_SZ + col].$vals_out,

				ul: map_in[(row-1)*TRACT_IN_SZ + col-1].$vals_out,
				ll: map_in[(row+1)*TRACT_IN_SZ + col-1].$vals_out,
				ur: map_in[(row-1)*TRACT_IN_SZ + col+1].$vals_out,
				lr: map_in[(row+1)*TRACT_IN_SZ + col+1].$vals_out,
								
				cl: map_in[row*TRACT_IN_SZ + col-2].$vals_out,
				ct: map_in[(row-2)*TRACT_IN_SZ + col].$vals_out,
				cr: map_in[row*TRACT_IN_SZ + col+2].$vals_out,
				cb: map_in[(row+2)*TRACT_IN_SZ + col].$vals_out
			};	
			
			let region = skeleton.gen_region(row, col, rand_off);
			
			// copy over
			for i in 0..REGION_SZ {
			for j in 0..REGION_SZ {
				map_out[(i + roff)*TRACT_OUT_SZ + j + coff].$vals_out = region[i*REGION_SZ + j]; // todo optimization: j loop could be implicit...
			}}
		}} // row, col
	};};
	
	let mut elevation_range_in;
	let mut arability_range_in;

	gen_floats!(elevation, elevation_range_in);
	gen_floats!(arability, arability_range_in);
	
	////////////////////////// upsample show_snow
	{
		// gen region show_snow based on simple upsampling of 3x3 grid of the map-to-be-upsampled
		for region_i in 0..N_REGIONS_PER_TRACT { let row = region_i*2 + 2; let roff = region_i*(REGION_SZ-1);
		for region_j in 0..N_REGIONS_PER_TRACT { let col = region_j*2 + 2; let coff = region_j*(REGION_SZ-1);
			
			//  ul   top    ur
			// left center right
			//  ll  bottom  lr

			let center = map_in[row*TRACT_IN_SZ + col].show_snow;
			
			if center {
				for i in 0..REGION_SZ {
				for j in 0..REGION_SZ {
					map_out[(i + roff)*TRACT_OUT_SZ + j + coff].show_snow = true;
				}}
				continue;
			}
			
			let left = map_in[row*TRACT_IN_SZ + col-1].show_snow;
			let right = map_in[row*TRACT_IN_SZ + col+1].show_snow;
			let top = map_in[(row-1)*TRACT_IN_SZ + col].show_snow;
			let bottom = map_in[(row+1)*TRACT_IN_SZ + col].show_snow;

			let ul = map_in[(row-1)*TRACT_IN_SZ + col-1].show_snow;
			let ll = map_in[(row+1)*TRACT_IN_SZ + col-1].show_snow;
			let ur = map_in[(row-1)*TRACT_IN_SZ + col+1].show_snow;
			let lr = map_in[(row+1)*TRACT_IN_SZ + col+1].show_snow;
			
			// copy over
			for i in 0..REGION_SZ {
			for j in 0..REGION_SZ {
				map_out[(i + roff)*TRACT_OUT_SZ + j + coff].show_snow = 
					// top rows
					if i < (REGION_SZ/3) {
						if j < (REGION_SZ/3) {ul}
						else if j < 2*(REGION_SZ/3) {top}
						else {ur}
						
					// middle rows
					}else if i <= 2*(REGION_SZ/3) {
						if j < (REGION_SZ/3) {left}
						else if j <= 2*(REGION_SZ/3) {center}
						else {right}
					
					// bottom rows
					}else if j < (REGION_SZ/3) {ll}
					else if j < 2*(REGION_SZ/3) {bottom}
					else {lr};
			}}
		}} // row, col
	}
	
	/////////////// smooth over tracts of regions
	/*{
		const SMOOTH_FRAC: f32 = 0.;//0.3;//0.6;
		const SMOOTH_W: isize = 13;
		
		for i in 0..TRACT_OUT_SZ {
		for j in 0..TRACT_OUT_SZ {
			let mut sum = 0.;
			let mut n_sum = 0;
			for i_off in -SMOOTH_W..=SMOOTH_W {
				
				let ii_off = (i as isize) + i_off;
				if ii_off < 0 || ii_off >= TRACT_OUT_SZ as isize {continue;}
				let ind_r = (ii_off) as usize;
				
				for j_off in -SMOOTH_W..=SMOOTH_W {
					
					let jj_off = (j as isize) + j_off;
					if jj_off < 0 || jj_off >= TRACT_OUT_SZ as isize {continue;}
					let ind_c = (jj_off) as usize;
					
					sum += map_out[ind_r*TRACT_OUT_SZ + ind_c].elevation;
					n_sum += 1;
				} // col_off
			} // row_off
			
			let ind = i*TRACT_OUT_SZ + j;
			map_out[ind].elevation = (1.-SMOOTH_FRAC)*map_out[ind].elevation + SMOOTH_FRAC*sum/(n_sum as f32)
		}} // i,j
	}*/
	
	//////////////// find threshold
	{
		let mut mountain_lower = 100.;
		let mut land_upper = -10.;
		
		let mut land_lower = 10.;
		let mut water_upper = -10.;
		
		let mut shallow_water_lower = 10.;
		let mut deep_water_upper = -10.;
		
		for mi in map_in.iter() {
			if mi.map_type == MapType::Mountain && mountain_lower > mi.elevation {
				mountain_lower = mi.elevation;
			}else if mi.map_type == MapType::Land && land_upper < mi.elevation {
				land_upper = mi.elevation;
			}	
			
			if mi.map_type == MapType::Land && land_lower > mi.elevation {
				land_lower = mi.elevation;
			}else if (mi.map_type == MapType::DeepWater || mi.map_type == MapType::ShallowWater) && water_upper < mi.elevation {
				water_upper = mi.elevation;
				
				if mi.map_type == MapType::ShallowWater && shallow_water_lower > mi.elevation {
					shallow_water_lower = mi.elevation;
				}else if mi.map_type == MapType::DeepWater && deep_water_upper < mi.elevation {
					deep_water_upper = mi.elevation;
				}
			}
		}
		
		// !!!!!!!!!!!!!!!!!! debug_assertq!(water_upper <= land_lower, "{} {}", water_upper, land_lower);
		let thresh_mountain = (mountain_lower + land_upper)*0.5;
		let thresh_land = (land_lower + water_upper)*0.5;
		let thresh_water = (shallow_water_lower + deep_water_upper)*0.5;
		
		for mo in map_out.iter_mut() {
			mo.map_type = if mo.elevation > thresh_mountain {
						MapType::Mountain
					}else if mo.elevation > thresh_land {
						MapType::Land
					} else if mo.elevation > thresh_water {
						MapType::ShallowWater
					} else {
						MapType::DeepWater
					};
		}
	}
	
	//////////////// 1. match mean of 2x2 output regions, 2. match range of outputs to inputs
	macro_rules! match_mean_and_range{($vals_out: ident, $range_in: ident) => {	
		////////////////////////// 1. match mean output 2x2 region_sz
		{
			const SMOOTH_SZ: usize = 2*REGION_SZ;
			const SMOOTH_FRAC: f32 = 0.45;
			
			let dim_use = (TRACT_OUT_SZ/SMOOTH_SZ)*SMOOTH_SZ;
			
			for row in (0..dim_use).step_by(SMOOTH_SZ) {
				for col in (0..dim_use).step_by(SMOOTH_SZ) {
					let mut sum = 0.;
					let mut n_sum = 0;
					
					/////// sum
					for i in 0..SMOOTH_SZ {
						let ind_off = (row + i)*TRACT_OUT_SZ + col;
						for mo in map_out[ind_off..(ind_off+SMOOTH_SZ)].iter() {
							sum += mo.$vals_out;
							n_sum += 1;
						}
					}
					
					///// match mean
					let mean = sum / n_sum as f32;
					for i in 0..SMOOTH_SZ {
						let ind_off = (row + i)*TRACT_OUT_SZ + col;
						for mo in map_out[ind_off..(ind_off+SMOOTH_SZ)].iter_mut() {
							mo.$vals_out = mo.$vals_out*(1.-SMOOTH_FRAC) + mean*SMOOTH_FRAC;
						}
					}
				} // col
			} // row
		} // 2x2 region mean matching		

		
		/////////////////////////////// 2. match range of outputs to be the range of the inputs
		let mut min_out = map_out[0].$vals_out;
		let mut max_out = map_out[0].$vals_out;
		for mo in map_out[1..].iter() {
			if min_out > mo.$vals_out {min_out = mo.$vals_out;}
			if max_out < mo.$vals_out {max_out = mo.$vals_out;}
		}
		
		for mo in map_out.iter_mut() {
			mo.$vals_out -= min_out;
			mo.$vals_out /= max_out - min_out;
			
			mo.$vals_out *= $range_in.max - $range_in.min;
			mo.$vals_out += $range_in.min;
		}
	};};
	
	match_mean_and_range!(elevation, elevation_range_in);
	match_mean_and_range!(arability, arability_range_in);
	
	//////////////////// add resources
	{
		let mut rng = XorState::init(rand_off + 0xBADF00D);
		
		struct ResourceLoc {y: usize, x: usize};
		let mut coords_lr = Vec::new(); // lower right extent of each resource added
		
		// loop over map
		/*'map_loop: for (ind, m) in map_out.iter_mut().enumerate() {
			let arability_type = ArabilityType::frm_arability(m.arability, m.map_type, m.show_snow);
			let arability_ind = arability_type as usize;
			
			// loop over resources that have probabilities for this arability type
			'resource_loop: for r in resource_templates.iter() {
				if let Some(prob) = r.arability_probs[arability_ind] {
					if rng.gen_f32b() < prob {
						// convert to coordinates of candidate location to add resource
						let cand_ul = ResourceLoc {y: ind / TRACT_OUT_SZ,
							                        x: ind % TRACT_OUT_SZ};
						
						// check proposed resource is not partially outside region
						if cand_ul.y >= TRACT_OUT_SZ || cand_ul.x >= TRACT_OUT_SZ
								{continue 'resource_loop;}
						
						let cand_lr = ResourceLoc {y: cand_ul.y + r.sz.h,
										   x: cand_ul.x + r.sz.w};
						
						// check if proposed resource is not overlapping previously added resources
						if coords_lr.iter().rev().
								 any(|added_lr: &ResourceLoc| added_lr.y >= cand_ul.y && added_lr.x >= cand_ul.x)
									{continue 'resource_loop;}
						
						coords_lr.push(cand_lr);
						m.resource = Some(r);
						//endwin();
						//println!("{} {} {}", r.nm, r.sz.h, r.sz.w);
						continue 'map_loop;
					} // add
				} // prob exists for map location's arability type
			} // resource loop
		} // map loop*/
		'map_loop: for ind in 0..(TRACT_OUT_SZ*TRACT_OUT_SZ) {
			let m = &mut map_out[ind];
			let arability_type = ArabilityType::frm_arability(m.arability, m.map_type, m.show_snow);
			let arability_ind = arability_type as usize;
			
			// loop over resources that have probabilities for this arability type
			'resource_loop: for r in resource_templates.iter() {
				if let Some(prob) = r.arability_probs[arability_ind] {
					if rng.gen_f32b() < prob {
						// convert to coordinates of candidate location to add resource
						let cand_ul = ResourceLoc {y: ind / TRACT_OUT_SZ,
							                        x: ind % TRACT_OUT_SZ};						
						let cand_lr = ResourceLoc {y: cand_ul.y + r.sz.h,
										   x: cand_ul.x + r.sz.w};
						
						// check proposed resource is not partially outside region
						if cand_lr.y >= TRACT_OUT_SZ || cand_lr.x >= TRACT_OUT_SZ
								{continue 'resource_loop;}

						// check if proposed resource is not overlapping previously added resources
						if coords_lr.iter().rev().
								 any(|added_lr: &ResourceLoc| added_lr.y >= cand_ul.y && added_lr.x >= cand_ul.x)
									{continue 'resource_loop;}
						
						coords_lr.push(cand_lr);
						m.resource = Some(r);
						for i in 0..r.sz.h {
						for j in 0..r.sz.w {
							let ind = (cand_ul.y + i)*TRACT_OUT_SZ + cand_ul.x + j;
							map_out[ind].resource_cont = Some(ResourceCont {offset_i: i as u8, offset_j: j as u8});
						}}
						//endwin();
						//println!("{} {} {}", r.nm, r.sz.h, r.sz.w);
						continue 'map_loop;
					} // add
				} // prob exists for map location's arability type
			} // resource loop
		} // map loop
	}
	
	map_out
}

#[derive(Copy,Clone,PartialEq)]
pub enum ZoomInd {
	Full,
	Val(usize)
}

impl fmt::Display for ZoomInd {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
				ZoomInd::Full => String::from("Full"),
				ZoomInd::Val(ind) => format!("{}", ind)
			}
		)
	}
}

impl <'rt>MapData<'rt> {
	// get it from either the buffer or generate it
	#[inline]
	pub fn get(&mut self, zoom_ind: ZoomInd, coord: u64) -> Map<'rt> {
		//#[cfg(feature="profile")]
		//let _g = Guard::new("map_data.get");
		
		macro_rules! gen_or_ld_cached_non_explicit_coord{($zoom_ind_val: expr) => {
			// map not stored explicitly:
			debug_assertq!($zoom_ind_val > ZOOM_IND_ROOT); // the full (root) map should be returned from self.zoom_out ^
			debug_assertq!(coord < (self.map_szs[$zoom_ind_val].h*self.map_szs[$zoom_ind_val].w) as u64, 
					"coord {} {}, {} {}, z: {} max: {}",
					coord / self.map_szs[$zoom_ind_val].w as u64, coord % self.map_szs[$zoom_ind_val].w as u64,
					self.map_szs[$zoom_ind_val].h, self.map_szs[$zoom_ind_val].w, $zoom_ind_val, self.max_zoom_ind());
			
			// requested location:
			let zoom_in_ind = $zoom_ind_val - N_EXPLICITLY_STORED_ZOOM_LVLS; 
			let mzc = self.zoom_in[zoom_in_ind].get(&coord);
			
			// coordinate is already cached
			if let Some(mczu) = mzc {
				return *mczu;
			
			// must generate coordinate
			}else{
				//endwin();
				let mut tract_parent = Vec::with_capacity(TRACT_IN_SZ*TRACT_IN_SZ);
				let w_parent = self.map_szs[$zoom_ind_val-1].w;
				let w = self.map_szs[$zoom_ind_val].w;
				
				let n_tracts = (w / TRACT_OUT_SZ) as u64;
				
				// on current zoom
				let i_ret = coord / (w as u64);
				let j_ret = coord % (w as u64);
				
				debug_assertq!(i_ret < self.map_szs[$zoom_ind_val].h as u64);
				
				// on parent
				let tract_i = i_ret / TRACT_OUT_SZ as u64;
				let tract_j = j_ret / TRACT_OUT_SZ as u64;
				
				// tract parent is the original generated map
				if ($zoom_ind_val-1) == ZOOM_IND_ROOT {
					for i in 0..TRACT_IN_SZ {
					for j in 0..TRACT_IN_SZ {
						let ti = tract_i*(TRACT_IN_SZ - TRACT_OVERLAP) as u64 + i as u64;
						let tj = tract_j*(TRACT_IN_SZ - TRACT_OVERLAP) as u64 + j as u64;
						let coord = ti*w_parent as u64 + tj;
						
						debug_assertq!(ti < self.map_szs[$zoom_ind_val-1].h as u64, "i {} ti {} h {} zoom_ind {}",
								i, ti, self.map_szs[$zoom_ind_val-1].h, $zoom_ind_val);
						debug_assertq!(tj < w_parent as u64);
						
						debug_assertq!((coord as usize) < self.zoom_out[ZOOM_IND_ROOT].len());
						tract_parent.push(self.zoom_out[ZOOM_IND_ROOT][coord as usize]);
					}}
					
				// retrieve tract parent from cache or also re-generate
				}else{
					for i in 0..TRACT_IN_SZ {
					for j in 0..TRACT_IN_SZ {
						let ti = tract_i*(TRACT_IN_SZ-4) as u64 + i as u64;
						let tj = tract_j*(TRACT_IN_SZ-4) as u64 + j as u64;
						let coord = ti*w_parent as u64 + tj;
						
						debug_assertq!(ti < self.map_szs[$zoom_ind_val-1].h as u64,
								"i {} TRACT_IN_SZ {} ti {} h {} zoom_ind_val {}", 
								i, TRACT_IN_SZ, ti, self.map_szs[$zoom_ind_val-1].h, $zoom_ind_val);
						tract_parent.push(self.get(ZoomInd::Val($zoom_ind_val-1), coord));
					}}
				}
				
				let rand_off = tract_i*n_tracts + tract_j;
				let tract = gen_tract(tract_parent, rand_off, self.rt); 
				
				// put values in cache	
				for (tract_coord, t) in tract.iter().enumerate() {
					let i = tract_coord / TRACT_OUT_SZ;
					let j = tract_coord % TRACT_OUT_SZ;
					
					let coord = (tract_i*TRACT_OUT_SZ as u64 + i as u64)*w as u64 + 
						tract_j*TRACT_OUT_SZ as u64 + j as u64;
					
					self.zoom_in[zoom_in_ind].insert(coord, *t);
					
					self.deque_zoom_in.push_back(DequeZoomIn {zoom_in_ind, coord});
					
					// buffer must be shortened
					if self.deque_zoom_in.len() > self.max_zoom_in_buffer_sz {
						let entry_rm = self.deque_zoom_in.pop_front().unwrap();
						self.zoom_in[entry_rm.zoom_in_ind].remove(&entry_rm.coord);
						self.zoom_in[entry_rm.zoom_in_ind].shrink_to_fit();
					}
				}
				
				let tract_sub_i = (i_ret % TRACT_OUT_SZ as u64) as usize;
				let tract_sub_j = (j_ret % TRACT_OUT_SZ as u64) as usize;
				
				let ind = tract_sub_i*TRACT_OUT_SZ + tract_sub_j;
				debug_assertq!(ind < tract.len());
				return tract[ind];
			}
		};};
		
		match zoom_ind {
		   ZoomInd::Full => {
			 // the most zoomed in scale (not explicitly stored)
			 gen_or_ld_cached_non_explicit_coord!(self.max_zoom_ind());
		   } ZoomInd::Val(zoom_ind_val) => {
			// stored explicitly
			if zoom_ind_val <= ZOOM_IND_ROOT {	
				return self.zoom_out[zoom_ind_val][coord as usize];
			}
			
			gen_or_ld_cached_non_explicit_coord!(zoom_ind_val);
		}} // matching the zoom lvl
	} // end get() def
}
