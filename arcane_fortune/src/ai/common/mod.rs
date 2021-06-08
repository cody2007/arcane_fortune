use super::*;

pub mod vars; pub use vars::*;
pub mod actions; pub use actions::*;
pub mod attack_fronts; pub use attack_fronts::*;
pub mod city_planning; pub use city_planning::*;
pub mod economy_planning; pub use economy_planning::*;

const WIDTH_PER_HEIGHT: usize = 2;

impl <'bt,'ut,'rt,'dt>AIState<'bt,'ut,'rt,'dt> {
	pub fn new(coord: Coord, chk_square_clear: bool, city_grid_height: usize, min_dist_frm_city_center: usize,
			taxable_template: &'bt BldgTemplate<'ut,'rt,'dt>, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			map_data: &mut MapData<'rt>, exf: &HashedMapEx, map_sz: MapSz, rng: &mut XorState) -> Option<Self> {
		
		let city_height = city_grid_height * GRID_SZ;
		let city_width = city_height * WIDTH_PER_HEIGHT;
		
		// check if location possible
		if chk_square_clear && (coord.y >= (map_sz.h - city_height - 3) as isize ||
		   coord.x >= (map_sz.w - city_width - 3) as isize ||
		   square_clear(coord.to_ind(map_sz) as u64, ScreenSz{h: city_height, w: city_width, sz: 0}, Quad::Lr, map_data, exf) == None) {
				/*printlnq!("{} {} {} {} {} {}",
					coord.y >= (map_sz.h - city_height - 3) as isize,
					coord.x >= (map_sz.w - city_width - 3) as isize,
					square_clear(coord.to_ind(map_sz) as u64, ScreenSz{h: city_height, w: city_width, sz: 0}, Quad::Lr, map_data, exf) == None,
					coord, city_height, city_width);*/
				return None;
		   }
		
		let city_state = CityState::new(coord, city_grid_height, min_dist_frm_city_center, taxable_template, rng, map_data, map_sz);
		
		Some(Self {
			city_states: vec![city_state],
			attack_fronts: Default::default(),
			icbm_inds: Vec::new(),
			damaged_wall_coords: Vec::new(),
			next_bonus_bldg: None,
			
			goal_doctrine: Some(&temps.doctrines[rng.usize_range(1, temps.doctrines.len())]),
			
			paused: false
		})
	}
	
	pub fn doctrine_txt(&self, l: &Localization) -> String {
		if let Some(doctrine) = self.goal_doctrine {
			doctrine.nm[l.lang_ind].clone()
		}else{
			l.no_doctrine.clone()
		}
	}
}

