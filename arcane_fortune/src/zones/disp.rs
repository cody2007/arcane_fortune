use crate::renderer::*;
use crate::disp::*;
use crate::gcore::hashing::HashedMapEx;
use crate::player::Stats;
use crate::containers::*;
use super::*;

const MAX_ROAD_DIST: usize = 9;

pub fn new_zone_w_roads(coord_start: u64, coord_end: u64, map_sz: MapSz, exf: &HashedMapEx) -> Box<[bool]>{
	let c1 = Coord::frm_ind(coord_start, map_sz);
	let c2 = Coord::frm_ind(coord_end, map_sz);
	
	let (c, rect_sz) = disp::start_coord_use(c1, c2, map_sz);
	let h = rect_sz.h as usize;
	let w = rect_sz.w as usize;
	
	let mut roads = vec!(false; h*w).into_boxed_slice();
	
	// grid
	for i_off in 0..h {
	for j_off in 0..w {
		if (i_off % MAX_ROAD_DIST) == 0 || (j_off % (2*MAX_ROAD_DIST)) == 0 {
			roads[i_off*w + j_off] = true;
		}
	}}
	
	// can we add roads on the periphery to align with existing roads oustide the zone?
	
	// check bottom row
	if (c.y as usize + h) < map_sz.h {
		for j_off in 0..w {
			let coord = map_sz.coord_wrap(c.y + h as isize, c.x + j_off as isize);
			
			// not a road...
			let ex_wrapped = exf.get(&coord.unwrap());
			if ex_wrapped.is_none() || ex_wrapped.unwrap().actual.structure.is_none() ||
				ex_wrapped.unwrap().actual.ret_structure() != Some(StructureType::Road) {
					continue;
			}
			
			// already at road
			if j_off % (2*MAX_ROAD_DIST) == 0 {continue;}
			
			// there's a road above, connect to it
			if h >= 2 && roads[(h-2)*w + j_off] {
				roads[(h-1)*w + j_off] = true;
				continue;
			}
			
			// column roads
			let d = 2*MAX_ROAD_DIST;
			let j_road1 = (j_off / d) * d;
			let j_road2 = j_road1 + d;
			
			// closer to road1 or road2 is not in zone
			if j_road2 >= w || (j_road1 as isize - j_off as isize).abs() < (j_road2 as isize - j_off as isize).abs() {
				debug_assertq!(j_road1 < w && j_off < w);
				for j_use in j_road1..=j_off {
					roads[(h-1)*w + j_use] = true;
				}
			}else{
				debug_assertq!(j_off < w && j_road2 < w);
				for j_use in j_off..=j_road2 {
					roads[(h-1)*w + j_use] = true;
				}
			}			
		} // j
	}
	
	// check far right col
	for i_off in 0..h {
		let coord = map_sz.coord_wrap(c.y + i_off as isize, c.x + w as isize);
		
		// not a road...
		let ex_wrapped = exf.get(&coord.unwrap());
		if ex_wrapped.is_none() || ex_wrapped.unwrap().actual.structure.is_none() ||
			ex_wrapped.unwrap().actual.ret_structure() != Some(StructureType::Road) {
				continue;
		}
		
		// already at road
		if i_off % MAX_ROAD_DIST == 0 {continue;}
		
		// there's a road to the left, connect to it
		if w >= 2 && roads[i_off*w + w-2] {
			roads[i_off*w + w-1] = true;
			continue;
		}
		
		// row roads
		let i_road1 = (i_off / MAX_ROAD_DIST) * MAX_ROAD_DIST;
		let i_road2 = i_road1 + MAX_ROAD_DIST;
		
		// closer to road1 or road2 is not in zone
		if i_road2 >= h || (i_road1 as isize - i_off as isize).abs() < (i_road2 as isize - i_off as isize).abs() {
			debug_assertq!(i_road1 < h && i_off < h);
			for i_use in i_road1..=i_off {
				roads[i_use*w + w-1] = true;
			}
		}else{
			debug_assertq!(i_off < h && i_road2 < h);
			for i_use in i_off..=i_road2 {
				roads[i_use*w + w-1] = true;
			}
		}
	} // i
	
	roads
}

pub enum ZonePlotType<'s,'b,'bt,'ut,'rt,'dt> {
	All {
		pstats: &'s Stats<'bt,'ut,'rt,'dt>,
		bldgs: &'b Vec<Bldg<'bt,'ut,'rt,'dt>>,
		owner_id: SmSvType
	},
	Single(f32, ZoneType) // plot single value (val, zone_type)
}

// if plot_zone is none, plot all, else plot only plot_zone
pub fn show_zone_demand_scales(roff: &mut i32, turn_col: i32, plot_type: ZonePlotType, dstate: &mut DispState) {
	macro_rules! mvclr{() => (dstate.mv(*roff, turn_col); *roff += 1; dstate.renderer.clrtoeol());}
	
	let scale_width = dstate.iface_settings.screen_sz.w  as isize - turn_col as isize;
	let scale_width_h = scale_width / 2;
	let step = 2./(scale_width as f32);
	
	macro_rules! plot_zone{($zone_type: expr, $val: expr) => {
		mvclr!();
			
		for i in 0..scale_width {
			if i == scale_width_h {
				dstate.addch(dstate.chars.vline_char);
				continue;	
			}
			let mid_val = step*(i as f32 - scale_width_h as f32);
			
			if i < scale_width_h && $val <= mid_val || // less than mid-point
			  (i+1) == scale_width_h && $val < 0. || // nearly at mid-point (below)
			  (i-1) == scale_width_h && $val > 0. || // nearly at mid-point (above)
			  i > scale_width_h && $val >= mid_val { // after midpoint
				dstate.plot_zone_type($zone_type); 
			}else {
				dstate.renderer.addch(' ' as chtype);
			}
		}
	};}
	
	match plot_type {
		ZonePlotType::Single(val, zone_type) => { // plot single value (local)
			plot_zone!(zone_type, val);
			
		} ZonePlotType::All{pstats, owner_id, bldgs} => { // plot all values (global)
			let zone_sum = &pstats.zone_demand_sum_map;
			
			for zt in 0..(ZoneType::N as usize) {
				let zone_type = ZoneType::from(zt);
				let val = zone_sum[zt].map_avg_zone_demand(zone_type, bldgs, owner_id);
				plot_zone!(zone_type, val);
			}
		}
	}
}
