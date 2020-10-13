use crate::saving::*;
use crate::disp_lib::endwin;
use crate::units::*;
use crate::buildings::*;
use crate::resources::*;
use crate::doctrine::DoctrineTemplate;

pub const MAX_BARBARIAN_UNITS: usize = 20;

#[derive(Clone, PartialEq, Debug)]
pub struct BarbarianState {
	// unit indices:
	pub defender_inds: Vec<usize>, // unit inds
	pub attacker_inds: Vec<usize>, // unit inds
	
	pub camp_ind: usize, // index into bldgs
}

impl_saving!{BarbarianState {defender_inds, attacker_inds, camp_ind}}

impl BarbarianState {
	pub fn add_unit(&mut self, unit_ind: usize, ut: &UnitTemplate) {
		debug_assertq!(!self.defender_inds.contains(&unit_ind) && !self.attacker_inds.contains(&unit_ind)); // id already contained
		
		const N_DEFENDERS: usize = 5;
		if self.defender_inds.len() >= N_DEFENDERS && ut.attack_per_turn != None {
			self.attacker_inds.push(unit_ind);
		}else{
			self.defender_inds.push(unit_ind);
		}
	}
	
	pub fn rm_unit(&mut self, unit_ind: usize) {
		if let Some(indx) = self.defender_inds.iter().position(|&r| r == unit_ind) {
			self.defender_inds.swap_remove(indx);
		}else if let Some(indx) = self.attacker_inds.iter().position(|&r| r == unit_ind) {
			self.attacker_inds.swap_remove(indx);
		}else{panicq!("id {} not contained in barbarian_state", unit_ind);}
	}
	
	pub fn chg_unit_ind(&mut self, frm_ind: usize, to_ind: usize) {
		if let Some(indx) = self.defender_inds.iter().position(|&r| r == frm_ind) {
			self.defender_inds[indx] = to_ind;
		}else if let Some(indx) = self.attacker_inds.iter().position(|&r| r == frm_ind) {
			self.attacker_inds[indx] = to_ind;
		}else{panicq!("chg_unit_ind frm {} to {}. frm not contained in barbarian_state. defenders & attackers lens {} {}", frm_ind, to_ind,
				self.defender_inds.len(), self.attacker_inds.len());}
	}
	
	pub fn n_units(&self) -> usize {
		self.attacker_inds.len() + self.defender_inds.len()
	}
	
	pub fn max_units(&self) -> bool {
		self.n_units() >= MAX_BARBARIAN_UNITS
	}
}

