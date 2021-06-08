use super::*;

pub mod vars; pub use vars::*;
pub mod actions; pub use actions::*;

impl <'bt,'ut,'rt,'dt>EmpireState<'bt,'ut,'rt,'dt> {
	pub fn new(coord: Coord, chk_square_clear: bool,
			personality: AIPersonality, temps: &Templates<'bt,'ut,'rt,'dt,'_>, map_data: &mut MapData<'rt>,
			exf: &HashedMapEx, rng: &mut XorState) -> Option<Self> {
		let city_hall = BldgTemplate::frm_str(CITY_HALL_NM, temps.bldgs);
		if let Some(ai_state) = AIState::new(coord, chk_square_clear, CITY_GRID_HEIGHT, MIN_DIST_FRM_CITY_CENTER, city_hall, temps, map_data, exf, *map_data.map_szs.last().unwrap(), rng) {
			Some(Self {personality, ai_state})
		}else{None}
	}
}

//////// utilities
impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	// adds the city and worker actions into `ai_state.city_states`
	// `loc` is the upper left corner of the city boundary
	pub fn add_city_plan(&mut self, loc: Coord, rng: &mut XorState, map_data: &mut MapData,
			map_sz: MapSz, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>) {
		let bldg_template = BldgTemplate::frm_str(CITY_HALL_NM, bldg_templates);
		self.city_states.push(CityState::new(loc, CITY_GRID_HEIGHT, MIN_DIST_FRM_CITY_CENTER, bldg_template, rng, map_data, map_sz));
	}
}

pub fn current_war_advantage(owner_id: usize, players: &Vec<Player>, relations: &Relations) -> Option<isize> {
	if let Some(&offense_power) = players[owner_id].stats.offense_power_log.last() {
		Some((offense_power as isize) - relations.at_war_with(owner_id).iter()
			.map(|&enemy| *players[enemy].stats.defense_power_log.last().unwrap() as isize)
			.sum::<isize>())
	}else{
		None
	}
}

const KINGDOM_MILITARY_FACTOR: f32 = 2.;
pub fn willing_to_join_as_kingdom(candidate_kingdom_id: usize, candidate_empire_id: usize, players: &Vec<Player>) -> bool {
	if let Some(kingdom_power) = players[candidate_kingdom_id].stats.military_power() {
		if let Some(empire_power) = players[candidate_empire_id].stats.military_power() {
			return (kingdom_power as f32 * KINGDOM_MILITARY_FACTOR) < empire_power as f32;
		}
	}
	false
}

impl <'bt,'ut,'rt,'dt>CityState<'bt,'ut,'rt,'dt> {
	// find new city location near an existant city
	// todo: priortize options
	pub fn find_new_city_loc(&self, ai_states: &AIState, pstats: &Stats, map_data: &mut MapData, exf: &HashedMapEx, ai_config: &AIConfig, rng: &mut XorState, map_sz: MapSz) -> Option<Coord> {
		#[cfg(feature="profile")]
		let _g = Guard::new("find_new_city_loc");
		
		let valid_proposed_coord = |coord: Coord| {
			coord.y < (map_sz.h - CITY_HEIGHT - 3) as isize && coord.y >= 0 &&
			coord.x < (map_sz.w - CITY_WIDTH  - 3) as isize && coord.x >= 0
		};
		
		//////////// strategic resources
		// select location with no minimum distance between it and originating city
		// preference given to closer cities
		
		for strategic_resource in ai_config.strategic_resources.iter() {
			//if !pstats.resource_discov(strategic_resource) {continue;} // in the technological sense
			let resource_id = strategic_resource.id as usize;
			if pstats.resources_avail[resource_id] != 0 {continue;} // we already have this resource
			
			// loop over instances of `strategic_resource` that we have discovered on the map
			'resource_exemplars: for resource_coord in pstats.resources_discov_coords[resource_id].iter() {
				let mut resource_coord = Coord::frm_ind(*resource_coord, map_sz);
				if !valid_proposed_coord(resource_coord) {continue;}
				
				// check to make sure city not already founded here
				for city_state in ai_states.city_states.iter() {
					if manhattan_dist(resource_coord, Coord::frm_ind(city_state.coord, map_sz),
							map_sz) < CITY_WIDTH {
						continue 'resource_exemplars;
					}
				}
				
				// coordinate of proposed city will be near this resource
				resource_coord.y -= ((CITY_HEIGHT/2) + 10) as isize;
				resource_coord.x -= ((CITY_WIDTH/2) + 10) as isize;
				
				if valid_proposed_coord(resource_coord) {
					return Some(resource_coord);
				}
			}
		}
		
		//////////// find best nearby candidate location
		// (1) based on sampling known nearby resources
		// (2) random samples
		{
			let neighbor_c = Coord::frm_ind(self.coord, map_sz); // neighboring city the new city will be close to
			
			const N_ATTEMPTS: usize = 50;
			
			// in increments of the city dimension
			const MIN_DIST: usize = 1;
			const MAX_DIST: usize = 3;
			const MAX_RESOURCE_DIST: usize = 5*CITY_WIDTH;
			
			const ARABILITY_SZ: usize = 8;
			let arability_sz = ScreenSz {h: ARABILITY_SZ, w: ARABILITY_SZ*2, sz: 0};
			
			struct CandidateLocation {coord: u64, score: f32}
			
			let mut candidate_locations = Vec::with_capacity(N_ATTEMPTS);
			
			////////////////////////////////////////// discovered resources loop
			// (note: uses all resources even if they haven't technologically been discovered)
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("find_new_city_loc resource loop");
				
				debug_assertq!(pstats.resources_discov_coords.len() == ai_config.city_creation_resource_bonus.len());
				for (resource_coords, bonus) in pstats.resources_discov_coords.iter()
						.zip(ai_config.city_creation_resource_bonus.iter()) {
					if *bonus == 0 {continue;}
					
					// loop over exemplars for the given resource
					for resource_coord in resource_coords.iter() {
						let mut resource_coord = Coord::frm_ind(*resource_coord, map_sz);
						
						// resource is not too close or far away
						let dist = manhattan_dist_components(resource_coord, neighbor_c, map_sz);
						if dist.h > (MIN_DIST*CITY_HEIGHT) && dist.w > (MIN_DIST*CITY_HEIGHT) &&
								(dist.h + dist.w) < MAX_RESOURCE_DIST {
							resource_coord.y -= ((CITY_HEIGHT/2) + 10) as isize;
							resource_coord.x -= ((CITY_WIDTH/2) + 10) as isize;
							
							// chk if location is valid
							if let Some(map_coord) = map_sz.coord_wrap(resource_coord.y, resource_coord.x) {
								if square_clear(map_coord, ScreenSz{h: CITY_HEIGHT, w: CITY_WIDTH, sz: 0}, Quad::Lr, map_data, exf) == None ||
								   !valid_proposed_coord(resource_coord) {
									continue;
								}
								
								let score = (*bonus as f32) + arability_mean(resource_coord, arability_sz, map_data, map_sz);
								
								candidate_locations.push(CandidateLocation {coord: map_coord, score});
							}
						}
					}
				}
			}
			
			////////////////////////////////////////////// arability loop
			{
				#[cfg(feature="profile")]
				let _g = Guard::new("find_new_city_loc arability loop");
				
				//let mut max_arability_score = 0.;
				for _ in 0..N_ATTEMPTS {
					// find offset from neighbor_c (use equation for ellipse); store offsets as `y`, `x`
					
					// elipse: y = (y_lim/x_lim) * sqrt(x_lim^2 - x^2)
					let x_lim = rng.isize_range((MIN_DIST*CITY_WIDTH) as isize, (MAX_DIST*CITY_WIDTH) as isize);
					let y_lim = rng.isize_range((MIN_DIST*CITY_HEIGHT) as isize, (MAX_DIST*CITY_HEIGHT) as isize);
					
					let x = rng.isize_range(-x_lim, x_lim);
					let y = ((y_lim as f32 / x_lim as f32) * ((x_lim*x_lim - x*x) as f32).sqrt()).round() as isize;
					
					// add offsets to get `c_new` -- the upper left boundary of the city
					let c_new = if rng.usize_range(0,2) < 1 {
						Coord {y: neighbor_c.y + y, x: neighbor_c.x + x}
					}else{
						Coord {y: neighbor_c.y - y, x: neighbor_c.x + x}
					};
					
					// check that the city does not wrap around map -- can cause problems with
					// the city planning of worker actions
					if (c_new.x < 0) || ((c_new.x as usize + CITY_WIDTH) >= map_sz.w) {continue;}
					
					// check if coord on map & square is clear
					if let Some(map_coord) = map_sz.coord_wrap(c_new.y, c_new.x) {
						if square_clear(map_coord, ScreenSz{h: CITY_HEIGHT, w: CITY_WIDTH, sz: 0}, Quad::Lr, map_data, exf) == None || 
						   !valid_proposed_coord(c_new) {
							continue;
						}
						
						let score = arability_mean(c_new, arability_sz, map_data, map_sz);
						//if score > max_arability_score {max_arability_score = score;}
						
						candidate_locations.push(CandidateLocation {coord: map_coord, score});
					}
				}
			}
			
			// sort from greatest to least
			candidate_locations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Less));
			
			if let Some(candidate_location) = candidate_locations.first() {
				//printlnq!("max arability {} {}", max_arability_score, candidate_location.score);
				Some(Coord::frm_ind(candidate_location.coord, map_sz))
			}else {None}
		}
	}
}

impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	// find the closest city that the unit likely is in
	pub fn find_closest_city(&mut self, loc: Coord, map_sz: MapSz) -> Option<&mut CityState<'bt,'ut,'rt,'dt>> {
		const MAX_DIST_ALLOWED: usize = CITY_WIDTH + CITY_HEIGHT;
		if let Some(min_city) = self.city_states.iter_mut().min_by_key(|c| 
				manhattan_dist(Coord::frm_ind(c.coord, map_sz), loc, map_sz)) {
			if manhattan_dist(Coord::frm_ind(min_city.coord, map_sz), loc, map_sz) < MAX_DIST_ALLOWED {
				return Some(min_city);
			}
		}
		None
	}	
}

// `coord` specifies upper left coorner
fn arability_mean(coord: Coord, blank_spot: ScreenSz, map_data: &mut MapData, map_sz: MapSz) -> f32 {
	#[cfg(feature="profile")]
	let _g = Guard::new("arability_mean");
	
	let mut mean = 0.;
	
	for i_off in 0..blank_spot.h as isize {
	for j_off in 0..blank_spot.w as isize {
		let coord_chk = map_sz.coord_wrap(coord.y + i_off, coord.x + j_off).unwrap();
		let mfc = map_data.get(ZoomInd::Full, coord_chk);
		debug_assertq!(mfc.map_type == MapType::Land);
		mean += mfc.arability;
	}}
	
	mean / (blank_spot.h * blank_spot.w) as f32
}

