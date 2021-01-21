use super::*;

pub const MIN_DIST_FRM_CITY_CENTER: usize = 3; // min radius of city
const BUFFER_AROUND_CITY: isize = 5; // additional distance outside of city walls to protect (`city_ul`, `city_lr`)

pub const GRID_SZ: usize = 20;

// returns (height offset, width offset)
pub fn city_hall_offset(grid_height: isize, grid_width: isize) -> (isize, isize) {
	(GRID_SZ as isize*grid_height/2, GRID_SZ as isize*grid_width/2)
}

impl <'bt,'ut,'rt,'dt> CityState <'bt,'ut,'rt,'dt> {
	pub fn new(loc: Coord, city_grid_height: usize, min_dist_frm_center: usize,
			bldg_template: &'bt BldgTemplate<'ut,'rt,'dt>,
			rng: &mut XorState, map_data: &mut MapData, map_sz: MapSz) -> Self {
		///////// road and zones
		let mut grid_actions = create_city_grid_actions(loc, city_grid_height, min_dist_frm_center, rng, map_data, map_sz);
		
		//////// city hall location
		let offsets = city_hall_offset(city_grid_height as isize, (city_grid_height*2) as isize);
		// update add_bldg() if this changes
		let ch_action_coord = Coord {y: loc.y + offsets.0,
					    x: loc.x + offsets.1}; // where we tell the worker to put the city hall
		
		let ch_c = Coord {y: ch_action_coord.y, x: ch_action_coord.x + 1}; // plus 1 in the x dimension because the worker will place the bldg one off from the location in the action
		let ch_coord = map_sz.coord_wrap(ch_c.y, ch_c.x).unwrap(); 
		let ch_action_coord = map_sz.coord_wrap(ch_action_coord.y, ch_action_coord.x).unwrap();
		
		//////// walls
		let (gate_loc, wall_action, defense_positions) = city_wall_build_action(&grid_actions, ch_c, map_sz, rng);
		
		// convert wall u64 coords to y,x coords to store later in the cities structure
		let mut wall_coords = Vec::with_capacity(wall_action.path_coords.len());
		for c in wall_action.path_coords.iter() {
			wall_coords.push(Coord::frm_ind(*c, map_sz));
		}
		
		// get boundary around city (used for defense)
		let city_ul = Coord {y: max(wall_coords.iter().map(|c| c.y).min().unwrap() - BUFFER_AROUND_CITY, 0),
					   x:     wall_coords.iter().map(|c| c.x).min().unwrap() - BUFFER_AROUND_CITY}.wrap(map_sz);
		
		let city_lr = Coord {y: min(wall_coords.iter().map(|c| c.y).max().unwrap() + BUFFER_AROUND_CITY, map_sz.h as isize - 1),
					   x:     wall_coords.iter().map(|c| c.x).max().unwrap() + BUFFER_AROUND_CITY}.wrap(map_sz);

		// put on worker todo list to build the grid and wall
		// (the wall is pushed second so it is popped off sooner)
		let mut worker_actions = Vec::with_capacity(grid_actions.len() + 2);
		
		worker_actions.append(&mut grid_actions);
		worker_actions.push(wall_action);
		
		//////////// city hall		
		worker_actions.push( ActionMeta { // city hall
			action_type: ActionType::WorkerBuildBldg{
						valid_placement: true,
						template: bldg_template,
						bldg_coord: None,
						doctrine_dedication: None
			},
			actions_req: 1.,
			path_coords: vec!{ch_action_coord; 1},
			action_meta_cont: None
		});
		
		Self {
			coord: ch_coord,
			gate_loc,
			wall_coords,
			damaged_wall_coords: Vec::new(),
			
			city_ul,
			city_lr,
			
			population_center_ind: None,
			boot_camp_ind: None,
			academy_ind: None,
			bonus_bldg_inds: Vec::new(),
			
			worker_actions,
			worker_inds: Vec::new(),
			
			explorer_inds: Vec::new(),
			
			defenders: Vec::new(),
			defense_positions,
			
			neighbors_possible: Neighbors::NotKnown
		}
	}
}

////// create road and zone placements for a city starting at `loc`
// see ~/af/arcane_fortune/python_tests/city_grid.py for a python implementation
fn create_city_grid_actions<'bt,'ut,'rt,'dt>(loc: Coord, city_grid_height: usize, min_dist_frm_center: usize,
		rng: &mut XorState, map_data: &mut MapData, map_sz: MapSz) -> Vec<ActionMeta<'bt,'ut,'rt,'dt>> {
	let city_grid_width = city_grid_height * 2;
	
	let mut grid_actions = Vec::new();
	
	const SKIP_PROB: f32 = 0.075; // prob of skipping line between vertices
	
	// columns to start and stop main city bounds
	let start_left = rng.usize_range_vec(0, (city_grid_width/2) - min_dist_frm_center, city_grid_height);
	let stop_right = rng.usize_range_vec((city_grid_width/2) + min_dist_frm_center, city_grid_width, city_grid_height);
	
	// rows to start and stop main city bounds
	let start_top = rng.usize_range_vec(0, (city_grid_height/2) - min_dist_frm_center, city_grid_width);
	let stop_bottom = rng.usize_range_vec((city_grid_height/2) + min_dist_frm_center, city_grid_height, city_grid_width);
	
	// vertices between grid locations to connect roads between
	let constraints_i = rng.usize_range_vec(0, GRID_SZ, (city_grid_height+1)*(city_grid_width+1));
	let constraints_j = rng.usize_range_vec(0, GRID_SZ, (city_grid_height+1)*(city_grid_width+1));
	
	// grids to skip
	let skip_cols = rng.f32b_vec(city_grid_height*city_grid_width);
	let skip_rows = rng.f32b_vec(city_grid_height*city_grid_width);
	
	// ActionMeta templates for creating roads and zones:
	let road_action = || -> ActionMeta {
		ActionMeta {
			action_type: ActionType::WorkerBuildStructure {structure_type: StructureType::Road, turns_expended: 0},
			actions_req: GRID_SZ as f32,
			path_coords: Vec::with_capacity(GRID_SZ),
			action_meta_cont: None
		}
	};
	
	let zone_action = |zone_prob| -> ActionMeta {
		let zone_type = if zone_prob < 0.5 {
						ZoneType::Residential
					}else if zone_prob < (0.5 + (1./6.)) {
						ZoneType::Agricultural
					}else if zone_prob < (0.5 + (2./6.)) {
						ZoneType::Business
					}else{
						ZoneType::Industrial
		};
		
		ActionMeta {
			action_type: ActionType::WorkerZoneCoords {zone_type},
			actions_req: GRID_SZ as f32,
			path_coords: Vec::with_capacity(GRID_SZ),
			action_meta_cont: None
		}
	};

	/* Line equation derivation between two vertices
		i: row, j: col (n = GRID_SZ)
		
		Linear equation of line:
		  i = c*j + b
		Notation of coordinate pair:
		 (i,j)
		
		Constraint points:
		 (1) (grid_i*n + ci, grid_j*n)
		 (2) (grid_i*n + ci2, (grid_j+1)*n)
		
		Equations at constraint points:
		 (1) grid_i*n + ci = c*grid_j*n + b
		 (2) grid_i*n + ci2 = c*(grid_j+1)*n + b
		
		Re-arranging (1):
		 b = grid_i*n + ci - c*grid_j*n
		
		Substitute b into (2):
		 grid_i*n + ci2 = c*(grid_j+1)*n + grid_i*n + ci - c*grid_j*n
		 grid_i*n + ci2 - (grid_i*n + ci) = c*[ (grid_j+1)*n - grid_j*n ]
		 c = (ci2 - ci)/n
		
		 b = grid_i*n + ci - grid_j*(ci2 - ci)
			
		 i = j*(ci2 - ci)/n + grid_i*n + ci - grid_j*(ci2 - ci) 
	*/

	// returns coordinates for a row or column for roads going down across columns or across rows
	let road_cross_cols = |j, grid_i, grid_j, constraints_i: &Vec<usize>| -> u64 {
		let ci = constraints_i[grid_i*(city_grid_width+1) + grid_j]; // left vertex
		let ci2 = constraints_i[grid_i*(city_grid_width+1) + grid_j+1]; // right vertex
		
		let i = (((ci2 as f32 - ci as f32) / GRID_SZ as f32)*j as f32).round() as isize + 
				(grid_i*GRID_SZ + ci) as isize + (grid_j as isize*(ci as isize - ci2 as isize));
			
		if let Some(coord) = map_sz.coord_wrap(loc.y + i as isize, loc.x + j as isize) {
			return coord;
		}else {panicq!("i {} j {} loc {} {} grid_i {} grid_j {} ci {} ci2 {} map_sz {}", i, j, loc.y, loc.x, grid_i, grid_j, ci, ci2, map_sz);}
	};
	
	let road_cross_rows = |i, grid_i, grid_j, constraints_j: &Vec<usize>| -> u64 {
		let cj = constraints_j[grid_i*(city_grid_width+1) + grid_j]; // top vertex
		let cj2 = constraints_j[(grid_i+1)*(city_grid_width+1) + grid_j]; // bottom vertex
		
		let j = (((cj2 as f32 - cj as f32) / GRID_SZ as f32)*i as f32).round() as isize +
				(grid_j*GRID_SZ + cj) as isize + (grid_i as isize *(cj as isize - cj2 as isize));
			
		if let Some(coord) = map_sz.coord_wrap(loc.y + i as isize, loc.x + j as isize) {
			return coord;
		}else {panicq!("i {} j {} loc {} {} grid_i {} grid_j {} cj {} cj2 {}", i, j, loc.y, loc.x, grid_i, grid_j, cj, cj2);}
	};
	
	// add road coordinates to road placing action for row-spanning and column-spanning roads
	enum RoadCross {Cols, Rows};
	let place_road = |skip_prob: &Vec<f32>, grid_k, grid_i, grid_j, road_cross, constraints, ck, worker_actions: &mut Vec<ActionMeta>| {
		if skip_prob[grid_i*city_grid_width + grid_j] <= SKIP_PROB {return;}
		
		let mut action = road_action();
		
		action.path_coords.push(
			match road_cross {
				RoadCross::Cols => {map_sz.coord_wrap(loc.y + (grid_i*GRID_SZ + ck) as isize, loc.x + (grid_j*GRID_SZ) as isize).unwrap()}
				RoadCross::Rows => {map_sz.coord_wrap(loc.y + (grid_i*GRID_SZ) as isize, loc.x + (grid_j*GRID_SZ + ck) as isize).unwrap()}
		});
		
		for k in (1 + grid_k*GRID_SZ)..((grid_k+1)*GRID_SZ) {
			action.path_coords.push(
				match road_cross {
					RoadCross::Cols => {road_cross_cols(k, grid_i, grid_j, constraints)}
					RoadCross::Rows => {road_cross_rows(k, grid_i, grid_j, constraints)}
			});
		}
					
		worker_actions.push(action);
	};
	
	let mut added = vec!{false; city_grid_height*city_grid_width};
	
	//endwin();
	let mut add_road_and_zone_actions = |grid_i_off, grid_j_off| {
		let grid_i = ((city_grid_height/2) as isize + grid_i_off) as usize;
		let grid_j = ((city_grid_width/2) as isize + grid_j_off) as usize;
		
		if grid_i >= city_grid_height || grid_j >= city_grid_width {return;}
		if added[grid_i*city_grid_width + grid_j] {return;}
		added[grid_i*city_grid_width + grid_j] = true;
		//println!("m[{}+{}, {}+{}] = 1", city_grid_height/2, grid_i_off, city_grid_width/2, grid_j_off);
		
		//debug_assertq!(grid_i < city_grid_height, "grid_i {}", grid_i);
		//debug_assertq!(grid_j < city_grid_width, "grid_j {}", grid_j);
		
		// line across columns
		let ci = constraints_i[grid_i*(city_grid_width+1) + grid_j]; // left vertex
		//let ci2 = constraints_i[grid_i*(city_grid_width+1) + grid_j+1]; // right vertex
		
		// line across rows
		let cj = constraints_j[grid_i*(city_grid_width+1) + grid_j]; // top vertex
		//let cj2 = constraints_j[(grid_i+1)*(city_grid_width+1) + grid_j]; // bottom vertex
		
		// check if the city has started or stopped
		if start_top[grid_j] >= grid_i || grid_i >= stop_bottom[grid_j] {return;}
		if start_left[grid_i] >= grid_j || grid_j >= stop_right[grid_i] {return;}
		
		place_road(&skip_cols, grid_j, grid_i, grid_j, RoadCross::Cols, &constraints_i, ci, &mut grid_actions);
		if (start_top[grid_j]+1) < grid_i {
			place_road(&skip_rows, grid_i, grid_i, grid_j, RoadCross::Rows, &constraints_j, cj, &mut grid_actions);
		}
		
		// zone
		if grid_i < (city_grid_height-1) && grid_j < (city_grid_width-1) { // chk not at edge
			let mut action = zone_action(rng.gen_f32b());
			
			for i in (grid_i*GRID_SZ)..=((grid_i+2)*GRID_SZ) {
				let cross_row1 = Coord::frm_ind(road_cross_rows(i, grid_i, grid_j, &constraints_j), map_sz);
				let cross_row2 = Coord::frm_ind(road_cross_rows(i, grid_i, grid_j+1, &constraints_j), map_sz);
				
				macro_rules! add_coord{($i: expr, $j: expr) => {
					let cross_col1 = Coord::frm_ind(road_cross_cols($j, grid_i, grid_j, &constraints_i), map_sz);
					let cross_col2 = Coord::frm_ind(road_cross_cols($j, grid_i+1, grid_j, &constraints_i), map_sz);
					
					// cross-row roads make the column zone boundaries
					// 	schematic:		 | cross_row1 | residential space | cross_row2 |
					if ($j as isize + loc.x) <= cross_row1.x || ($j as isize + loc.x) >= cross_row2.x {continue;}
					
					// cross-column roads make the row zone boundaries
					if ($i as isize + loc.y) <= cross_col1.y || ($i as isize + loc.y) >= cross_col2.y {continue;}
					
					if let Some(coord) = map_sz.coord_wrap(i as isize + loc.y, $j as isize + loc.x) {
						// if zone does not match the resource, change it to match
						if let Some(resource) = map_data.get(ZoomInd::Full, coord).get_resource(coord, map_data, map_sz) {
							if let ActionType::WorkerZoneCoords {ref mut zone_type} = &mut action.action_type {
								if *zone_type != resource.zone {*zone_type = resource.zone}
							}else{panicq!("action incorrectly set");}
						}
						action.path_coords.push(coord);
					}else{panicq!("invalid AI zone coord {} {}", $i,$j);}
				};};
				
				// zig-zag direction of zone placement depending on row we are at
				if (i % 2) == 0 {
					for j in (1 + grid_j*GRID_SZ)..=((grid_j+2)*GRID_SZ) {
						add_coord!(i,j);
					}
				}else{
					for j in ((1 + grid_j*GRID_SZ)..=((grid_j+2)*GRID_SZ)).rev() {
						add_coord!(i,j);
					}
				}
			} // i
			
			// append only if there is a region to zone
			if action.path_coords.len() > 0 {
				grid_actions.push(action);
			}
		} // zone
	};
	
	add_road_and_zone_actions(0,0);
	for i in 0..1 {
		add_road_and_zone_actions(0, 1+i); // right of center
		add_road_and_zone_actions(0, -(1+i)); // left of center
	}
	
	// create grid
	for height_offset in 1..(city_grid_height/2) as isize {
		for center_offset in 0..=(2*height_offset + 1) {
			// go twice as far on top and bottom rows (width)
			for i in 0..2 {
				add_road_and_zone_actions(-height_offset, 2*center_offset + i); // top right of center
				
				add_road_and_zone_actions(height_offset, -(2*center_offset + i)); // bottom left of center
				
				add_road_and_zone_actions(-height_offset, -(2*center_offset + i)); // top left of center
				
				add_road_and_zone_actions(height_offset, 2*center_offset + i); // bottom right of center			
				
				// left and right sides
				if center_offset <= height_offset {
					add_road_and_zone_actions(center_offset, 2*height_offset + i); // bottom of right
					
					add_road_and_zone_actions(center_offset, -(2*height_offset + i)); // bottom of left
					
					add_road_and_zone_actions(-center_offset, 2*height_offset + i); // top of right
					
					add_road_and_zone_actions(-center_offset, -(2*height_offset + i)); // top of left
				}
			}
		}	
	}
	//println!();
	/*for k in (0..=offset).rev() {
		add_road_and_zone_actions(k, offset); // across a column
		
		if k != 0 {
			add_road_and_zone_actions(-k, offset); // across a column
			add_road_and_zone_actions(offset, -k); // across a row
		}
		
		if offset != 0 { // don't repeat previous actions if offset == 0
			add_road_and_zone_actions(k, -offset);
			add_road_and_zone_actions(-offset, k);
			
			if k != 0 {
				add_road_and_zone_actions(-k, -offset);
				add_road_and_zone_actions(-offset, -k);
				add_road_and_zone_actions(offset, k); // across a row
			}
		}
	}} // dist from center*/
	let mut grid_actions_rev = Vec::with_capacity(grid_actions.len());
	for a in grid_actions.iter().cloned().rev() {
		grid_actions_rev.push(a);
	}
	grid_actions_rev
}

// grid_actions are where the zones and roads are in the city. ch_coord is the city hall coordinate
//  returns:
//		(gate_location, action to create walls, defense positions at gate)
fn city_wall_build_action<'bt,'ut,'rt,'dt>(grid_actions: &Vec<ActionMeta>, ch_c: Coord,
		map_sz: MapSz, rng: &mut XorState) -> (u64, ActionMeta<'bt,'ut,'rt,'dt>, Vec<u64>) {
	// convert the city grid u64s to y,x coords
	let mut city_coords = Vec::with_capacity(1000);
	for action in grid_actions.iter() {
		for coord in action.path_coords.iter() {
			city_coords.push(Coord::frm_ind(*coord, map_sz));
		}
	}
	
	// sort from greatest to least
	city_coords.sort_unstable_by(|b, a| a.y.partial_cmp(&b.y).unwrap());
	
	let new_wall_action = || -> ActionMeta {
		ActionMeta::with_capacity(
			ActionType::WorkerBuildStructure {structure_type: StructureType::Wall, turns_expended: 0},
			city_coords.len()
		)
	};
	
	let mut wall_action = new_wall_action();
	
	let line_eq = |x, c1: Coord, c2: Coord| -> isize {
		let c1x = c1.x as f32;
		let c2x = c2.x as f32;
		let c1y = c1.y as f32;
		let c2y = c2.y as f32;
		
		let slope = (c1y - c2y)/(c1x - c2x);
		(slope*(x as f32) + c2y - slope*c2x).round() as isize // aka `y`
	};
	
	let line_to = |pos_y_next: Coord, pos_y: Coord, offset: isize, wall_action_quad: &mut ActionMeta| {
		let mut prev_y_opt = None;
		
		macro_rules! add_x_pos{($x: expr) => {
			let y = line_eq($x, pos_y, pos_y_next);
			
			if prev_y_opt == None {prev_y_opt = Some(y);}
			
			// add corner and possible vertical line connecting previous and current walls
			if let Some(prev_y) = prev_y_opt {
				if y > prev_y { // going down
					for y_connect in prev_y..=y {
						if let Some(coord) = map_sz.coord_wrap(y_connect + offset, $x) {
							wall_action_quad.path_coords.push(coord);
						}
					}
				}else{ // going up (we should start at prev_y, but the loop must be in the opposite order)
					for y_connect in (y..=prev_y).rev() {
						if let Some(coord) = map_sz.coord_wrap(y_connect + offset, $x) {
							wall_action_quad.path_coords.push(coord);
						}
					}
				}
				wall_action_quad.actions_req += (y - prev_y).abs() as f32 + 1.;
			}
			
			prev_y_opt = Some(y);
		};};
		
		if pos_y.x < pos_y_next.x { // left to right (quad 1 & 4)
			for x in pos_y.x..=pos_y_next.x {add_x_pos!(x);}
		}else{ // right to left (quad 2 & 3)
			for x in (pos_y_next.x..=pos_y.x).rev() {add_x_pos!(x);}
		}
	};
	
	// move from max y location to next greatest y location until there are no more in the x direction x_dir
	// 	x_dir: -1 => move from max y to left; x_dir: 1 => move from max y to right
	//	reverse_order: when true reverse ordering and add to wall_action in opposite order
	let lower_quadrants = |x_dir: isize, reverse_order: bool, wall_action: &mut ActionMeta| { 
		let mut prev_start = 0;
		let mut max_y = city_coords[0];
		let mut wall_action_quad = new_wall_action();
		'quad: loop {
			if city_coords.len() <= prev_start {break;}
			
			// find next max y value location
			for (i, coord) in city_coords[prev_start..].iter().enumerate() {
				// because city_coords is sorted from greatest to least
				// all following values will be less, so we can stop here
				if (x_dir*coord.x) > (x_dir*max_y.x) { // decreasing x means we go from left to right; increasming means right to left
					prev_start += i + 1;
					
					// line from max_y to max_y_next
					line_to(*coord, max_y, 1, &mut wall_action_quad);
					max_y = *coord;
					continue 'quad; // find new max coord and draw a new line
				}
			}
			break;
		}
		
		///// nothing left to add
		
		// reverse order so worker moves in a single, continous path
		if reverse_order {
			wall_action_quad.path_coords.reverse();
		}
		
		// add to main action record
		wall_action.actions_req += wall_action_quad.actions_req;
		wall_action.path_coords.append(&mut wall_action_quad.path_coords);
	};
	
	// move from min y location to next least y location until there are no more in the x direction x_dir
	// 	x_dir: -1 => move from min y to left; x_dir: 1 => move from min y to right
	//	reverse_order: when true reverse ordering and add to wall_action in opposite order
	let upper_quadrants = |x_dir: isize, reverse_order: bool, wall_action: &mut ActionMeta| { 
		let mut prev_end = city_coords.len();
		let mut min_y = *city_coords.last().unwrap();
		let mut wall_action_quad = new_wall_action();
		'quad: loop {
			if 0 >= prev_end {break;}
			
			// find next min y value location
			for (i, coord) in city_coords[..prev_end].iter().rev().enumerate() {
				// because city_coords is sorted from greatest to least
				// all following values will be greater (because we're going in rev order), so we can stop here
				if (x_dir*coord.x) > (x_dir*min_y.x) { // decreasing x means we go from left to right; increasming means right to left
					prev_end -= i + 1;
					
					// line from max_y to max_y_next
					line_to(*coord, min_y, -1, &mut wall_action_quad);
					min_y = *coord;
					
					continue 'quad; // find new max coord and draw a new line
				}
			}
			break;
		}
		
		///// nothing left to add
		
		// reverse order so worker moves in a single, continous path
		if reverse_order {
			wall_action_quad.path_coords.reverse();
		}
		
		// add to main action record
		wall_action.actions_req += wall_action_quad.actions_req;
		wall_action.path_coords.append(&mut wall_action_quad.path_coords);
	};
	
	//////////// add counter-clockwise in direction (worker pops off values so last added
	//           is first to be completed -- so direction constructed is clockwise starting at quad 2)
	lower_quadrants(-1, true, &mut wall_action); // quad 3 -- left to right (x decreasing; then reversed order adding of default right to left)
	lower_quadrants(1, false, &mut wall_action); // quad 4 -- left to right (x increasing)
	
	// line connecting lower right quadrant to upper right quadrant (quad 4 to quad 1), then add quad 4
	{
		let lr = Coord::frm_ind(*wall_action.path_coords.last().unwrap(), map_sz);
		
		// save quadrant in tmp variable because we need to first get the end value (the first value added) to connect to the last value
		// of quadrant 4. we draw the line between the two then add the quadrant from the tmp var
		let mut wall_action_tmp = new_wall_action();
		upper_quadrants(1, true, &mut wall_action_tmp); // quad 1 -- right to left right (x increasing; then reversed order adding of default)
		
		// todo: cover this case better
		/*if wall_action_tmp.path_coords.len() == 0 {
			printlnq!("wall_action_tmp is empty ch_c {} map_sz {} wall_action len {}", ch_c, map_sz, wall_action.path_coords.len());
		}*/
		
		if let Some(first_quad1_coord) = wall_action_tmp.path_coords.first() { // first coord that was added for quad 1
			let ur = Coord::frm_ind(*first_quad1_coord, map_sz); 
			debug_assertq!(ur.x == lr.x);
			for y in (ur.y..=lr.y).rev() { // go down (y increases)
				wall_action.path_coords.push(map_sz.coord_wrap(y, ur.x+1).unwrap());
			}
			wall_action.path_coords.append(&mut wall_action_tmp.path_coords);
			wall_action.actions_req += wall_action_tmp.actions_req;
		}
	}
	
	upper_quadrants(-1, false, &mut wall_action); // quad 2 -- right to left (x decreasing)
	
	// line connecting lower left quadrant to upper left quadrant (quad4 to quad1)
	{
		let ul = Coord::frm_ind(*wall_action.path_coords.last().unwrap(), map_sz);
		let ll = Coord::frm_ind(wall_action.path_coords[0], map_sz);
		debug_assertq!(ul.x == ll.x); // should be the minima+1 of all x
		for y in ul.y..=ll.y { // go up (y decreases)
			wall_action.path_coords.push(map_sz.coord_wrap(y, ul.x-1).unwrap());
		}
	}
	
	// choose opening and remove it
	let wall_coords = &mut wall_action.path_coords;
	let gate_loc;
	'opening_loop: loop {
		let opening_ind = rng.usize_range(0, wall_coords.len());
		let c = Coord::frm_ind(wall_coords[opening_ind], map_sz);
		
		////////// check if gate is on a straight line
		
		// check if diagonal plots are filled
		{
			for i in [-1,1].iter() {
			for j in [-1,1].iter() {
				if wall_coords.contains(&map_sz.coord_wrap(c.y + i, c.x + j).unwrap()) {
					continue 'opening_loop;
				}
			}}
		}
		
		// return if top and bottom or left and right are both empty, continue otherwise
		{
			let gate_top = map_sz.coord_wrap(c.y - 1, c.x).unwrap();
			let gate_bottom = map_sz.coord_wrap(c.y + 1, c.x).unwrap();
			let gate_left = map_sz.coord_wrap(c.y, c.x - 1).unwrap();
			let gate_right = map_sz.coord_wrap(c.y, c.x + 1).unwrap();
			
			///// check gate orientation, continue if not either vertical or horizontal
			enum GateOrientation {Vertical, Horizontal};
			
			let gate_orientation = if !wall_coords.contains(&gate_left) && !wall_coords.contains(&gate_right) {
				GateOrientation::Vertical // wall is running up and down
			}else if !wall_coords.contains(&gate_top) && !wall_coords.contains(&gate_bottom) {
				GateOrientation::Horizontal // wall is running left and right
			}else {continue 'opening_loop;};
			
			gate_loc = wall_action.path_coords.remove(opening_ind);
			//gate_loc = wall_coords.remove(opening_ind);
			
			let mut defense_positions = Vec::with_capacity(4); // ordered by priority, first position most important
			defense_positions.push(gate_loc); // position at gate itself
			
			match gate_orientation {
				GateOrientation::Vertical => {
					let x_off = if c.x < ch_c.x {1} else {-1}; // is city hall to the left or right of the gate?
					for i_off in [0,1,-1].iter() { // up and down column
						defense_positions.push(map_sz.coord_wrap(c.y + i_off, c.x + x_off).unwrap());
					}
					defense_positions.push(map_sz.coord_wrap(c.y, c.x + 2*x_off).unwrap()); // most internal point
				} GateOrientation::Horizontal => {
					let y_off = if c.y < ch_c.y {1} else {-1}; // is city hall above or below the gate?
					for j_off in [0,1,-1].iter() { // left and right across row
						defense_positions.push(map_sz.coord_wrap(c.y + y_off, c.x + j_off).unwrap());
					}
					defense_positions.push(map_sz.coord_wrap(c.y + 2*y_off, c.x).unwrap()); // most internal point
				}
			}
			
			//endwin();println!("gate {} defense_positions {}", Coord::frm_ind(gate_loc, map_sz), defense_positions.len());
			
			return (gate_loc, wall_action, defense_positions);
		}
	}
}

