use super::*;

// for the purpose of recomputing zone demands and zone agnostic stats
impl ActionInterfaceMeta<'_,'_,'_,'_,'_> {
	pub fn too_far(&mut self, start_c: Coord, b: &Bldg, bldgs: &Vec<Bldg>, exf: &HashedMapEx, exs: &Vec<HashedMapEx>,
			map_data: &mut MapData, map_sz: MapSz) -> bool {
		// bldg too far to consider
		if manhattan_dist(start_c, Coord::frm_ind(b.coord, map_sz), map_sz) >= self.max_search_depth {return true;}
		
		let map_coord_end_use = find_closest_road(b.coord, map_data, exf, map_sz);
		
		if map_coord_end_use.is_none() {return true;} // not close enough to any road
		let end_c = Coord::frm_ind(map_coord_end_use.unwrap(), map_sz);
		
		// road too far to consider
		if manhattan_dist(start_c, end_c, map_sz) >= self.max_search_depth {return true;}
		
		// check if bldg connects to current position
		self.update_move_search(end_c, map_data, exs, MvVars::None, bldgs);
		
		// no connection
		if self.action.path_coords.len() == 0 {return true;}
		
		false
	}
}

