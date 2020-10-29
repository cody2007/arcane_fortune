use super::vars::*;
use crate::disp::Coord;
use crate::movement::{movable_to, MvVarsAtZoom, Dest};
use crate::units::Unit;
use crate::buildings::Bldg;
use crate::doctrine::DoctrineTemplate;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx};
use crate::map::ZoomInd;
use crate::zones::{ZoneEx};
use crate::renderer::*;

impl MapSz {
	pub fn coord_wrap(&self, i: isize, j: isize) -> Option<u64> {
		if i < 0 || i >= (self.h as isize){
			return None;
		}
		
		let w = self.w as isize;
		
		let mj = match j {
			j if j < 0 => 
				w + (j % w),
			j if j >= w => 
				j % w,
			_ =>
				j
		};
		let ind = (i as u64)*(self.w as u64) + (mj as u64);
		debug_assertq!(ind < (self.sz as u64), "{}", self.sz);
		Some(ind)
	}
}

pub enum Unboard {Loc {coord: u64, carried_ind: usize}, None}

// is land next to current space and be used for unloading?
pub fn unboard_land_adj(unit_ind: usize, units: &Vec<Unit>, bldgs: &Vec<Bldg>, map_data: &mut MapData, exf: &HashedMapEx) -> Unboard {
	let u = &units[unit_ind];
	if let Some(units_carried) = &u.units_carried {
		let map_sz = *map_data.map_szs.last().unwrap();
		let mv_vars = MvVarsAtZoom::NonCivil {units, start_owner: u.owner_id, blind_undiscov: None};
		let c = Coord::frm_ind(u.return_coord(), map_sz);
		let offs = [-1,0,1];
		
		for i_off in &offs {
		for j_off in &offs {
			if *i_off == 0 && *j_off == 0 {continue;}
			if let Some(chk_coord) = map_sz.coord_wrap(c.y + *i_off, c.x + *j_off) {
				for (carried_ind, c) in units_carried.iter().enumerate() {
					if movable_to(u.return_coord(), chk_coord, &map_data.get(ZoomInd::Full, chk_coord), exf, mv_vars, bldgs, &Dest::NoAttack, c.template.movement_type) {
						return Unboard::Loc {coord: chk_coord, carried_ind};
					}
				}
			}
		}}
	}
	Unboard::None
}

pub trait ExFns<'bt,'ut,'rt,'dt> {
	fn create_if_empty(&mut self, coord: u64);
	fn update_or_insert(&mut self, coord: u64, ex_insert: MapEx<'bt,'ut,'rt,'dt>);
}

impl <'bt,'ut,'rt,'dt> ExFns <'bt,'ut,'rt,'dt> for HashedMapEx<'bt,'ut,'rt,'dt> {
	fn create_if_empty(&mut self, coord: u64) {
		if !self.contains_key(&coord) {
			self.insert(coord, MapEx::default());
		}
	}
	
	fn update_or_insert(&mut self, coord: u64, ex_insert: MapEx<'bt,'ut,'rt,'dt>){
		let ex_wrapped = self.get_mut(&coord);
		
		// use existant slot
		if let Some(ex) = ex_wrapped {
			debug_assertq!(ex.actual.ret_zone_type() == ex_insert.actual.ret_zone_type());
			// ^ if not true, stats[] zone demand sums are going to be corrupted
			//   (unless we are not at the max zoom lvl)
			
			*ex = ex_insert.clone();
		
		// slot must be allocated
		}else{
			self.insert(coord, ex_insert);
		}
	}
}

pub trait ZoneExFns {fn create_if_empty(&mut self, coord: u64, doctrine_templates: &Vec<DoctrineTemplate>);}

impl ZoneExFns for HashedMapZoneEx {
	fn create_if_empty(&mut self, coord: u64, doctrine_templates: &Vec<DoctrineTemplate>) {
		if !self.contains_key(&coord) {
			self.insert(coord, ZoneEx::default_init(doctrine_templates));
		}
	}
}

