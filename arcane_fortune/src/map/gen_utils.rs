use crate::renderer::*;
use crate::gcore::rand::XorState;
use crate::disp::*;
use super::vars::*;
use crate::containers::*;

pub fn print_map_status(elevation_prog: Option<usize>, elevation_smooth_prog: Option<usize>,
		arability_prog: Option<usize>, arability_smooth_prog: Option<usize>,
		unit_placement_prog: Option<usize>, screen_sz: &mut ScreenSz, 
		map_szg: MapSz, cur_ind: usize, dstate: &mut DispState){
	
	if (cur_ind % 500) != 0 {return;}
	
	let screen_sz_new = getmaxyxu(&dstate.renderer);
	if screen_sz_new != *screen_sz {
		*screen_sz = screen_sz_new;
		dstate.renderer.clear();
	}
	
	let mut txt_under_logo_row_off = dstate.print_centered_logo(screen_sz_new, 0) as i32;
	txt_under_logo_row_off += (LOGO_HEIGHT + 1) as i32;
	
	const OFFSET_TXT: &str = "Generating land arability... ";
	let col = ((screen_sz_new.w - OFFSET_TXT.len())/2) as i32;
	
	macro_rules! print_progress{($txt: expr, $var: expr) => {
		if let Some(var) = $var {
			dstate.renderer.mv(txt_under_logo_row_off, col as i32);
			dstate.renderer.addstr($txt);
			dstate.renderer.addstr("...");
			dstate.renderer.mv(txt_under_logo_row_off, col + OFFSET_TXT.len() as i32);
			
			if var != map_szg.sz {
				let var_print = 100. * (var as f32) / (map_szg.sz as f32);
				addstr_c(&format!("{:.0}%", var_print), CYELLOW, &mut dstate.renderer);
			}else{
				addstr_c(&dstate.local.Done, CGREEN, &mut dstate.renderer);
			}
			txt_under_logo_row_off += 1;
		}
	};};
	
	print_progress!(&dstate.local.Generating_elevation_map, elevation_prog);
	print_progress!(&dstate.local.Smoothing_elevation_map, elevation_smooth_prog);
	print_progress!(&dstate.local.Generating_arability_map, arability_prog);
	print_progress!(&dstate.local.Smoothing_arability_map, arability_smooth_prog);
	print_progress!(&dstate.local.Placing_humankind, unit_placement_prog);

	dstate.renderer.refresh();
}

pub fn add_neighbor_stack(i: isize, j: isize, sum_neighbors: &mut f32, n_neighbors: &mut usize, 
		elevation_g: &Box<[f32]>, ind_stack: &mut Box<[usize]>, ind_stack_sz: &mut usize, 
		added: &mut Box<[bool]>, map_sz: MapSz) {
	
	if let Some(ind) = map_sz.coord_wrap(i, j) {
		let ind = ind as usize;
		if added[ind] == false {
			ind_stack[*ind_stack_sz] = ind;
			*ind_stack_sz += 1;
			added[ind] = true;
			return;
		}
		
		*n_neighbors += 1;
		*sum_neighbors += elevation_g[ind];
	}
}

////////////// called in smooth_map()
// if type_g != None, ignore water
pub fn add_neighbor(i: isize, j: isize, sum_neighbors: &mut f32, n_neighbors: &mut usize, 
		vals: &Box<[f32]>, map_sz: MapSz, type_g: Option<&Vec<MapType>>) {
	if let Some(ind) = map_sz.coord_wrap(i, j) {
		let ind = ind as usize;
		if let Some(type_g) = type_g {
			if type_g[ind] == MapType::DeepWater || type_g[ind] == MapType::ShallowWater {
				return;
			}
		}
	
		*n_neighbors += 1;
		*sum_neighbors += vals[ind];
	}
}

// if type_g != None, ignore water
pub fn smooth_map(vals: &mut Box<[f32]>, map_szg: MapSz, rng: &mut XorState, type_g: Option<&Vec<MapType>>, screen_sz: &mut ScreenSz, dstate: &mut DispState) {
	let rand_map_inds = rng.inds(map_szg.sz);
	
	for (ind, rand_ind) in (*rand_map_inds).iter().enumerate() {
		match type_g {
			None => {print_map_status(Some(map_szg.sz), Some(ind), None, None, None, screen_sz, map_szg, ind, dstate);}
			Some(_) => {print_map_status(Some(map_szg.sz), Some(map_szg.sz), Some(map_szg.sz), Some(ind), None, screen_sz, map_szg, ind, dstate);}
		}
		let rand_ind = *rand_ind;
		debug_assertq!(rand_ind < map_szg.sz);
		
		if let Some(type_g) = type_g {
			if type_g[rand_ind] == MapType::DeepWater || type_g[rand_ind] == MapType::ShallowWater {
				continue;
			}
		}
		
		let i = rand_ind / map_szg.w;
		let j = rand_ind % map_szg.w;
		debug_assertq!(i < map_szg.h);
		
		let mut sum_neighbors = 0.;
		let mut n_neighbors = 0;
	
		let r: isize = 10;
		let ur = (r as usize)*2;
		let rand_rng = (r as usize)*2 + 1;
		let i_inds = rng.inds(rand_rng);
		let j_inds = rng.inds(rand_rng);
		
		for i2 in 0..ur {
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
				
				add_neighbor((i as isize) + i_off, (j as isize) + j_off, 
						&mut sum_neighbors, &mut n_neighbors, 
						&vals, map_szg, type_g);
			}
		}
		
		vals[rand_ind] = sum_neighbors / (n_neighbors as f32);
	}
}

