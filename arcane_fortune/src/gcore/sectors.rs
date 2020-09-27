use super::*;

#[derive(Clone, PartialEq)]
pub struct Rectangle {
	pub start: Coord,
	pub end: Coord
}

impl_saving!{Rectangle {start, end}}

impl Rectangle {
	pub fn perim(&self, map_sz: MapSz) -> Vec<u64> {
		let (coord, sz) = start_coord_use(self.start, self.end, map_sz);
		let h = sz.h as isize;
		let w = sz.w as isize;
		let mut coords = Vec::with_capacity((2*(h + w) + 4) as usize);
		
		// top row (left to right)
		for x_off in 0..=w {
			if let Some(coord) = map_sz.coord_wrap(coord.y, coord.x + x_off) {
				coords.push(coord);
			}
		}
		
		// right side (top to bottom)
		for y_off in 1..h {
			if let Some(coord) = map_sz.coord_wrap(coord.y + y_off, coord.x + w) {
				coords.push(coord);
			}
		}
		
		// bottom row (right to left)
		for x_off in (0..=w).rev() {
			if let Some(coord) = map_sz.coord_wrap(coord.y + h, coord.x + x_off) {
				coords.push(coord);
			}
		}
		
		// left side (bottom to top)
		for y_off in (1..h).rev() {
			if let Some(coord) = map_sz.coord_wrap(coord.y + y_off, coord.x) {
				coords.push(coord);
			}
		}
		coords
	}
}

// perimeter around a sector
#[derive(Clone, PartialEq)]
pub struct PerimCoords {
	pub coords: Vec<u64>,
	pub turn_computed: usize
}

impl_saving!{PerimCoords {coords, turn_computed}}

impl PerimCoords {
	pub fn new(turn_computed: usize, segments: &Vec<Rectangle>, map_sz: MapSz) -> Self {
		let mut coords = Vec::new();
		for segment in segments.iter() {
			coords.append(&mut segment.perim(map_sz));
		}
		
		PerimCoords {
			coords,
			turn_computed
		}
	}
}

// map sectors
#[derive(Clone, PartialEq)]
pub struct Sector {
	pub nm: String,
	pub segments: Vec<Rectangle>,
	pub perim_coords: PerimCoords
}

impl_saving!{Sector {nm, segments, perim_coords}}

impl Sector {
	pub fn add(&mut self, turn_computed: usize, segment: Rectangle, map_sz: MapSz) {
		self.perim_coords.turn_computed = turn_computed;
		self.perim_coords.coords.append(&mut segment.perim(map_sz));
		self.segments.push(segment);
	}
	
	pub fn average_coord(&self, map_sz: MapSz) -> u64 {
		let mut avg_coord = map_sz.average(self.segments[0].start, self.segments[0].end);
		for segment in self.segments.iter().skip(1) {
			let segment_avg_coord = map_sz.average(segment.start, segment.end);
			avg_coord = map_sz.average(avg_coord, segment_avg_coord);
		}
		
		if let Some(c) = map_sz.coord_wrap(avg_coord.y, avg_coord.x) {
			c
		}else{panicq!("could not average sector coordinates");}
	}
	
	pub fn contains(&self, c: &Coord, map_sz: MapSz) -> bool {
		// note does not handle coord wraping... todo
		for segment in self.segments.iter() {
			let (start_use, rect_sz) = start_coord_use(segment.start, segment.end, map_sz);
			
			if c.y >= start_use.y && c.y <= (start_use.y + rect_sz.h as isize) &&
			   c.x >= start_use.x && c.x <= (start_use.x + rect_sz.w as isize) {
				return true;
			}
		}
		false
	}
}

impl Stats<'_,'_,'_,'_> {
	pub fn new_sector_nm(&self, nms: &Nms) -> String {
		let mut nm_suffix = String::new();
		for i in 0..1000 {
			for nm in nms.sectors.iter() {
				let nm_txt = format!("{}{}", nm, nm_suffix);
				if !self.sectors.iter().any(|sector| sector.nm == nm_txt) {
					return nm_txt;
				}
			}
			nm_suffix = format!(" {}", i);
		}
		panicq!("could not create sector name; n_sectors: {}", self.sectors.len());
	}
	
	pub fn sector_nm_frm_coord(&self, coord: u64, map_sz: MapSz) -> Option<&str> {
		// note does not handle coord wraping... todo
		let c = Coord::frm_ind(coord, map_sz);
		for sector in self.sectors.iter() {
			if sector.contains(&c, map_sz) {
				return Some(&sector.nm);
			}
		}
		None
	}
	
	pub fn sector_frm_nm(&self, nm: &String) -> &Sector {
		if let Some(sector) = self.sectors.iter().find(|s| s.nm == *nm) {
			sector
		}else{
			panicq!("could not find sector: `{}`", nm);
		}
	}
	
	pub fn sector_frm_nm_checked(&self, nm: &String) -> Option<&Sector> {
		self.sectors.iter().find(|s| s.nm == *nm)
	}

	pub fn sector_frm_nm_mut(&mut self, nm: &String) -> &mut Sector {
		if let Some(sector) = self.sectors.iter_mut().find(|s| s.nm == *nm) {
			sector
		}else{
			panicq!("could not find sector: `{}`", nm);
		}
	}
}

