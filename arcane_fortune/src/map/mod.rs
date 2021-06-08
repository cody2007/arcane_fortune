/*use std::fs::File;
use std::process::exit;
use std::io::prelude::*;
use std::path::Path;*/

use crate::gcore::XorState;
use crate::renderer::*;

#[macro_use]
pub mod vars;
pub mod zoom_in;
pub mod zoom_out;
pub mod utils;
pub mod gen_utils;

pub use vars::*;
pub use zoom_in::*;
pub use zoom_out::*;
pub use utils::*;
use gen_utils::*;

use crate::disp::*;
use crate::containers::*;

const MAX_VAL: f32 = 50000.;
const ARABILITY_RANGE: isize = 255;
const N_ARABILITY_LVLS: isize = 9;

pub const ARABILITY_STEP: isize = ARABILITY_RANGE / (2+N_ARABILITY_LVLS);
const SNOW_SPREAD_THRESH: u8 = 2*(ARABILITY_STEP as u8);

const ELEVATION_RANGE:isize = 24;//32 ;//+ 8; //(32+16);
pub const ELEVATION_STEP:isize = ELEVATION_RANGE / (2+N_ARABILITY_LVLS);

pub fn map_gen<'rt>(map_sz: MapSz, rng: &mut XorState, dstate: &mut DispState) -> Vec<Map<'rt>> {
	dstate.renderer.clear();
	let mut screen_sz = getmaxyxu(&dstate.renderer);
	
	let map_szg = MapSz {h: map_sz.h/2, w: map_sz.w/2, sz: map_sz.sz/4};
	
	//////////////////// generate elevation
	let mut elevation_g = {
		let mut elevation_g = vec![0. as f32; map_szg.sz].into_boxed_slice();
		
		const SIM: f32 = 2.075; // weighting of neighbor similarity
		const N_INIT: usize = 4; // number of random initial points to start w/
		
		let mut added: Box<[bool]> = vec![false; map_szg.sz].into_boxed_slice();
		let mut ind_stack: Box<[usize]> = vec![0; map_szg.sz].into_boxed_slice();
		let mut ind_stack_sz = N_INIT;

		for i in 0..N_INIT {
			ind_stack[i] = (rng.gen() as usize) % map_szg.sz;
			added[ind_stack[i]] = true;
		}
		
		for ind in 0..map_szg.sz {
			print_map_status(Some(ind), None, None, None, None, &mut screen_sz, map_szg, ind, dstate);
			
			debug_assertq!(ind < ind_stack_sz);
			let rand_ind = ind_stack[ind];
			
			debug_assertq!(rand_ind < map_szg.sz);
			debug_assertq!(added[rand_ind] != false);
			debug_assertq!(elevation_g[rand_ind] == 0.);
			
			let i = rand_ind / map_szg.w;
			let j = rand_ind % map_szg.w;
			debug_assertq!(i < map_szg.h);
			
			let mut sum_neighbors = 0.;
			let mut n_neighbors = 0;
			
			let r: isize = 16;
			let ur = (r as usize)*2;
			let rand_rng = (r as usize)*2 + 1;
			let i_inds = rng.inds(rand_rng);
			let j_inds = rng.inds(rand_rng);
			
			for i2 in 0..=ur {
				let i_off: isize = (i_inds[i2] as isize) - r;
				debug_assertq!(i_off >= -r);
				debug_assertq!(i_off <= r);
				let internal = (r*r) - (i_off*i_off);
				let y = (internal as f32).sqrt().round() as isize;
			
				for j2 in 0..=ur {
					let j_off = (j_inds[j2] as isize) - r;
					debug_assertq!(j_off >= -r);
					debug_assertq!(j_off <= r);
					
					// circle constraint
					if j_off < (-y) || j_off > y { continue; }
					
					add_neighbor_stack((i as isize) + i_off, (j as isize) + j_off, 
							&mut sum_neighbors, &mut n_neighbors, 
							&elevation_g, &mut ind_stack, &mut ind_stack_sz, 
							&mut added, map_szg);
				}
			}
			
			if n_neighbors == 0 {
				elevation_g[rand_ind] = rng.gen_norm();
				continue;
			}
			
			elevation_g[rand_ind] = rng.gen_norm() + ((SIM*sum_neighbors) as f32) / (n_neighbors as f32);
			
			// clamp
			if elevation_g[rand_ind] > MAX_VAL {
				elevation_g[rand_ind] = MAX_VAL;
			}else if elevation_g[rand_ind] < -MAX_VAL {
				elevation_g[rand_ind] = -MAX_VAL;
			}
		}
		elevation_g
	};
	
	smooth_map(&mut elevation_g, map_szg, rng, None, &mut screen_sz, dstate);
	
	///////////// threshold land and mountains
	let type_g = {
		let mut type_g = vec![MapType::DeepWater; map_szg.sz];
		let mut elevation_sorted = elevation_g.clone();
		elevation_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
		
		let szg = map_szg.sz as f32;
		let thresh_shallowwater = elevation_sorted[ (szg*0.505) as usize];
		let thresh_land = elevation_sorted[ (szg*0.6) as usize];
		let thresh_mountain = elevation_sorted[ (szg*0.99) as usize];
		
		for (i, type_g) in type_g.iter_mut().enumerate() {
			*type_g = match elevation_g[i] {
				v if v > thresh_mountain => MapType::Mountain,
			        v if v > thresh_land => MapType::Land,
				v if v > thresh_shallowwater => MapType::ShallowWater,
				_ => MapType::DeepWater
			};
		}
		type_g
	};
	
	/////////////// arability
	let arability_g = {
		let mut arability_g = vec![0_u8; map_szg.sz];
		let mut arability_f = vec![0.; map_szg.sz].into_boxed_slice();
		
		for i in 0..map_szg.h {
			for j in 0..map_szg.w {
				let ind = i*map_szg.w + j;
				debug_assertq!(ind < map_szg.sz);
				
				print_map_status(Some(map_szg.sz), Some(map_szg.sz), Some(ind), None, None, &mut screen_sz, map_szg, ind, dstate);
				
				if type_g[ind] != MapType::Land { continue; }
				
				const DIST_SEARCH: isize = 255;
				
				let mut dist = (DIST_SEARCH as f32)*(2. as f32).sqrt();
				let mut found_path_to_water = false;
				
				// start at disp = 1 and step farther out until
				// we find a plot of land with water
				// ex. disp = 1: we check at all "c" locations
				//
				// c c c
				// c . c
				// c c c

				// find min displacement from water
				'outer: for disp in 1..DIST_SEARCH {
					for i2 in -disp..=disp {
						let i_sum = (i as isize) + i2;
						if i_sum < 0 || i_sum >= (map_szg.h as isize) { continue; }
						
						///////////////////////////////////
						macro_rules! check_if_water_update_dist {
 							($jo:expr) => {{
								let j_sum = (j as isize) + ($jo);
								if j_sum >= 0 && j_sum < (map_szg.w as isize) {
									let indc = i_sum*(map_szg.w as isize) + j_sum;
									debug_assertq!(indc >= 0 && indc < (map_szg.sz as isize));
									
									if type_g[indc as usize] == MapType::ShallowWater ||
									   type_g[indc as usize] == MapType::DeepWater {
										let internal = i2*i2 + ($jo)*($jo);
										let dist_new = (internal as f32).sqrt();
										if dist_new < dist { dist = dist_new; }
										
										found_path_to_water = true;
										break 'outer;
									}
								}
							}};
						}
						////////////////////////////////
						
						check_if_water_update_dist!(-disp); // left column
						check_if_water_update_dist!(disp); // right column
						
						// top/bottom of column
						if i2 == (-disp) || i2 == disp {
							for j2 in (-disp+1)..disp {
								check_if_water_update_dist!(j2);	
							} // rows
						}
						
					} // i2
				} // displacement from water
				
				if found_path_to_water {
					let map_quart = (map_szg.h as isize)/4;
					
					let mut nsdist = ((i as isize) - map_quart).abs(); // northern pole dist
					let sdist = ((i as isize) - 3*map_quart).abs(); // souther pole dist
					if nsdist > sdist { nsdist = sdist; } // min n/s dist
					
					let mut dist_use = 255. - (3.*dist + 200.*((nsdist as f32)/(map_quart as f32))) + 50.*rng.gen_norm();
					
					// clip
					if dist_use < 0. {
						dist_use = 0.;
					}else if dist_use > ((ARABILITY_RANGE as f32) - 1.) {
						dist_use = (ARABILITY_RANGE as f32) - 1.;
					}
					arability_f[ind] = dist_use;
				}
			} // j
		} // i
		
		smooth_map(&mut arability_f, map_szg, rng, Some(&type_g), &mut screen_sz, dstate);
		
		for (ind, ag) in arability_g.iter_mut().enumerate() {
			*ag = arability_f[ind].round() as u8;
		}
		arability_g
	};
	
	////////////////////////////////////////////////////////// determine where to show snow
	// snow should extend all continous areas of low arability starting from FRAC_HEIGHT_SNOW
	let show_snow_g = {
		let mut show_snow_g = vec![false; map_szg.sz];
		const FRAC_HEIGHT_SNOW: f32 = 0.1;
		
		for _repeat in 0..2 {
			macro_rules! chk_neighbor_snow {
				($ind: expr, $above_or_below:expr) => {{
					debug_assertq!($ind < map_szg.sz);
					
					if type_g[$ind] == MapType::ShallowWater || type_g[$ind] == MapType::DeepWater { continue; }
					if show_snow_g[$ind] { continue; }
					
					let i = ($ind / map_szg.w) as isize;
					let frac_height = (i as f32) / (map_szg.h as f32);
					if frac_height <= FRAC_HEIGHT_SNOW || frac_height >= (1.-FRAC_HEIGHT_SNOW) {
						show_snow_g[$ind] = true;
						continue;
					}
					
					if arability_g[$ind] > SNOW_SPREAD_THRESH { continue; }
					debug_assertq!(i != 0 && i < ((map_szg.h as isize) - 1));
					
					let j = ($ind % map_szg.w) as isize;
					let mut found = false;
					for j2 in (-1)..=(1 as isize) {
						let ind2 = map_szg.coord_wrap(i+($above_or_below), j+j2);
						if ind2.is_none() { continue; }
						if show_snow_g[ind2.unwrap() as usize] == false { continue; }
						found = true;
						break;
					}
					show_snow_g[$ind] = found;
				}};
			}
			
			// scan from above
			for ind in 0..map_szg.sz {
				chk_neighbor_snow!(ind, -1);
			}
		
			// scan from below
			for ind in (0..map_szg.sz).rev() {
				chk_neighbor_snow!(ind, 1);
			}
		}
		show_snow_g
	};
	
	// scale elevation between -128 and 127
	for e in elevation_g.iter_mut() {
		*e = match *e {
			v if v < -4. => -4.,
			v if v > 4. => 4.,
			_ => *e
		};
		
		*e = ((*e + 4.) * (254./8.)) - 127.;
		debug_assertq!(*e <= (i8::max_value() as f32));
		debug_assertq!(*e >= (i8::min_value() as f32));
	}
	
	/////////////////////// upsample
	let mut map: Vec<Map> = vec![Default::default(); map_sz.sz];
	{
		debug_assertq!(map_szg.w == (map_sz.w/2));
		debug_assertq!(map_szg.h == (map_sz.h/2));
		
		///////////////////////////////////////////
		// upsample vals_in by 2x in both dimensions; store in vals_out
		
		macro_rules! upsample {($vals_out: ident, $vals_in: ident, $converter_type: ty, $final_type: ident) => {{
			//////////////
			// create all even coordinates (direct copy / init all values from input)
			for i in 0..map_szg.h {
			for j in 0..map_szg.w {
				let map_coord_g = i*map_szg.w + j;
				let map_coord = (i*2)*map_sz.w + j*2;
				
				map[map_coord].$vals_out = $vals_in[map_coord_g] as $final_type;
			}}
			
			/////////////
			// create all odd coordinates 
			for i in (1..(map_sz.h as isize)).step_by(2) {
			for j in (1..(map_sz.w as isize)).step_by(2) {
				let mut n_summed = 0;
				let mut val_sum = 0.;
				
				let icoords = [-1, 1 as isize];
				
				for i2 in &icoords {
				for j2 in &icoords {
					let map_coord_surround = map_sz.coord_wrap(i+i2, j+j2);
					if map_coord_surround.is_none()  {continue;}
					val_sum += map[map_coord_surround.unwrap() as usize].$vals_out as $converter_type as f32;
					///// ^ conversion (1)
					n_summed += 1;
				}}
				if n_summed != 0  {val_sum /= n_summed as f32;}
				val_sum = val_sum.round();
				
				let map_coord = (i as usize)*map_sz.w + (j as usize);
				convert_to_T!(map[map_coord].$vals_out, val_sum, $final_type);
				///// ^ conversion (2)
			}}
			
			// (1) converting T to floats:
			//
			// ^ `as` type conversion only goes from enum to int, so we then have to go a second time from int to float
			// no methods/function definitions can change the definition of `as`.
			
			// (2) converting floats to T:
			// T::From is not defined from floats to ints, so we have first have to use `as` to convert to an int 
			// before we use T::From. Why use T::from at all then? because it's the only way to generically convert
			// to MapType. (again, we can't alter the `as` keyword--it's only for primitives and won't work for enums)
			
			// but** bool::From is not defined even from ints... so it must be handled independently.. and we
			// might as well just deal with every single one individually.........................
			
			/////////////////////
			// fill in all (odd,even) and (even,odd) coordinates
			for i in 0..(map_sz.h as isize) {
			for j in 0..(map_sz.w as isize) {
				let io = i % 2;
				let jo = j % 2;
				if (io == 0 && jo == 0) || (io == 1 && jo == 1)  {continue;}
				
				let mut n_summed = 0;
				let mut val_sum = 0.;
				
				let ex_coords_i = [-1, 0, 0, 1].iter();
				let ex_coords_j = [0, -1, 1, 0].iter();

				for (i2, j2) in ex_coords_i.zip(ex_coords_j) {
					if let Some(map_coord_surround) = map_sz.coord_wrap(i+i2, j+j2) {
						val_sum += map[map_coord_surround as usize].$vals_out as $converter_type as f32;
						///// ^ conversion (1)
						n_summed += 1;
					}
				}
				
				if n_summed != 0  {val_sum /= n_summed as f32;}
				let map_coord = (i as usize)*map_sz.w + (j as usize);
				convert_to_T!(map[map_coord].$vals_out, val_sum, $final_type);
				/////// ^ conversion 2
			}}
		}}; }
		/////////////////////////////////////////////
		
		upsample!(arability, arability_g, u8, f32);
		upsample!(show_snow, show_snow_g, isize, bool);
		upsample!(elevation, elevation_g, f32, f32);
		upsample!(map_type, type_g, usize, MapType);
		
		//////// tmp write to csv
		/*let mut buf = String::new();
		for mfc in map.iter() {
			buf.push_str(&format!("{},", mfc.elevation));
		}
		
		if let Result::Ok(ref mut file) = File::create(Path::new("/tmp/test.csv").as_os_str()) {
			if let Result::Err(_) = file.write_all(&buf.into_bytes()) {
				q!(format!("failed writing file"));
			}
		}else{
			q!(format!("failed opening file for writing"));
		}*/

		// ensure deepwater is not adjacent to any land
		'map_type_loop: for i in 0..map_sz.sz {
			if map[i].map_type != MapType::DeepWater {continue;}
			
			let c = Coord::frm_ind(i as u64, map_sz);
			for i_off in -1..=1 {
			for j_off in -1..=1 {
				if let Some(coord) = map_sz.coord_wrap(c.y + i_off, c.x + j_off) {
					if map[coord as usize].map_type == MapType::Land {
						map[i].map_type = MapType::ShallowWater;
						continue 'map_type_loop;
					}
				}
			}}
		}
	} // upsample
	
	map
}

