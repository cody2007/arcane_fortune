use crate::disp_lib::*;
use crate::saving::*;
use crate::config_load::*;
use crate::map::{Stats, TechProg};
use crate::buildings::BldgTemplate;
use crate::disp::{IfaceSettings};
use crate::disp::window::{init_bldg_prod_windows, ProdOptions};
use crate::units::UnitTemplate;
use crate::localization::Localization;
use std::process::exit;

pub mod vars;
pub mod disp;

pub use vars::*;
pub use disp::*;

pub fn init_tech_templates(l: &Localization) -> Vec<TechTemplate> {
	let key_sets = config_parse(read_file("config/tech.txt"));
	chk_key_unique("nm", &key_sets);
	
	let mut tech_templates = Vec::new();
	let mut tech_req_nms: Vec<Option<Vec<String>>> = Vec::new();
	// ^ each tech can req. multiple techs
	
	for (id, keys) in key_sets.iter().enumerate() {
		let eng_nm = find_req_key("nm", keys);
		let nm = if let Some(nm) = l.tech_nms.iter().find(|nms| nms[0] == eng_nm) {
			nm.clone()
		}else{panicq!("could not find translations of tech `{}`. the localization file may need to be updated", eng_nm);};
		
		tech_templates.push( TechTemplate {
			id: id as SmSvType,
			nm,
			tech_req: None,
			research_req: find_req_key_parse("research_req", &keys)
		} );
		
		tech_req_nms.push(find_opt_key_vec_string("tech_req", &keys));
	}
	
	for i in 0..tech_templates.len() {
		if let Some(tech_req_nm) = &tech_req_nms[i] {
			let mut tech_req = Vec::new();

			for (i2, t) in tech_templates.iter().enumerate() {
				if tech_req_nm.contains(&t.nm[0]) {
					tech_req.push(i2 as SmSvType);
				}
			}
			
			tech_templates[i].tech_req = Some(tech_req);
			
			////////// error checking:
			if tech_templates[i].tech_req.as_ref().unwrap().len() == 0 {
				q!(format!("Could not find required tech for: \"{}\"", tech_templates[i].nm[0]));
			}
		}
	}
	
	// check ordering is correct
	#[cfg(any(feature="opt_debug", debug_assertions))]
	for (i, t) in tech_templates.iter().enumerate() {
		debug_assertq!(t.id == i as SmSvType);
	}
	
	tech_templates
}

fn find_index(key: SmSvType, vec: &Vec<SmSvType>) -> Option<usize> {
	vec.iter().position(|val| key == *val)
}

// research techs for all players
// update window production lists for bldgs & units, prompt player to select new tech
pub fn research_techs<'f,'bt,'ut,'rt,'dt>(stats: &mut Vec<Stats>, tech_templates: &Vec<TechTemplate>, 
		bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, cur_player: usize,
		production_options: &mut ProdOptions<'bt,'ut,'rt,'dt>,
		iface_settings: &mut IfaceSettings<'f,'bt,'ut,'rt,'dt>, l: &Localization, d: &mut DispState) {
	
	for (player, ps) in stats.iter_mut().enumerate() { // for all players
		if let Some(t_researching) = ps.techs_scheduled.last() { // if currently researching
			let t_researching = *t_researching as usize;
			
			// if progress is logged for this tech (should be)
			if let TechProg::Prog(ref mut prog) = ps.techs_progress[t_researching] {
				// increment research progress
				if (ps.research_per_turn + *prog) <= tech_templates[t_researching].research_req {
					*prog += ps.research_per_turn;
				
				// finish research
				}else{
					ps.techs_progress[t_researching] = TechProg::Finished;
					ps.techs_scheduled.pop();
					
					// update building unit production window options
					// prompt player to select new tech if nothing else has been scheduled, or show tech discovered window
					if player == cur_player {
						*production_options = init_bldg_prod_windows(bldg_templates, ps, l);
						
						// prompt to select new tech for research
						if ps.techs_scheduled.len() == 0 {
							iface_settings.create_tech_window(true, d);
						// tech discovered window
						}else{
							iface_settings.create_tech_discovered_window(t_researching, d);
						}
					}
				}
			} else {panicq!("Finished researching tech but still scheduled");}
		}
	}
}

impl Stats<'_,'_,'_,'_> {
	// max defensive unit the player can produce
	pub fn max_defensive_unit<'ut,'rt>(&self, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
		if let Some(ut) = unit_templates.iter().
				filter(|ut| self.unit_producable(&ut)).
				max_by_key(|ut| ut.max_health) {
			ut
		}else {panicq!("could not find max defensive unit")}
	}
	
	// max attack unit the player can produce
	pub fn max_attack_unit<'ut,'rt>(&self, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
		if let Some(ut) = unit_templates.iter().
				filter(|ut| self.unit_producable(&ut) && !ut.attack_per_turn.is_none()).
				max_by_key(|ut| ut.attack_per_turn.unwrap()) {
			ut
		}else {panicq!("could not find any attack units")}
	}
	
	// max seige unit the player can produce
	pub fn max_siege_unit<'ut,'rt>(&self, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
		if let Some(ut) = unit_templates.iter()
				.filter(|ut| 
						self.unit_producable(&ut) &&
						!ut.attack_per_turn.is_none() &&
						!ut.siege_bonus_per_turn.is_none())
				.max_by_key(|ut| ut.siege_bonus_per_turn.unwrap()) {
			ut
		}else {panicq!("could not find any siege units")}
	}

	// check tech requirements have been met for the building or unit to exist
	pub fn tech_met(&self, techs_req: &Option<Vec<usize>>) -> bool {
		if let Some(tech_reqs) = techs_req {
			'tech_req_loop: for tech_req in tech_reqs.iter() {
				match &self.techs_progress[*tech_req] {
					TechProg::Finished => {continue 'tech_req_loop;}
					TechProg::Prog(_) => {return false;}
				}
			}
		}
		true
	}
	
	// recursively schedule tech and all of its undiscovered research requirements
	fn schedule_undiscov_tech_req(&mut self, tech_req_ind: SmSvType, tech_templates: &Vec<TechTemplate>) {
		if let Some(tech_reqs) = &tech_templates[tech_req_ind as usize].tech_req {
			for tech_req_ind in tech_reqs {
				if self.techs_progress[*tech_req_ind as usize] != TechProg::Finished {
					// only add if not already scheduled
					if !self.techs_scheduled.contains(tech_req_ind) {
						self.techs_scheduled.push(*tech_req_ind);
						self.schedule_undiscov_tech_req(*tech_req_ind, tech_templates);
					}
				}
			}
		}
	}
	
	// recursively add tech and all of its undiscovered research requirements
	pub fn build_undiscov_tech_req_list(&self, tech_req_ind: SmSvType, tech_templates: &Vec<TechTemplate>, tech_reqs_list: &mut Vec<SmSvType>) {
		if let Some(tech_reqs) = &tech_templates[tech_req_ind as usize].tech_req {
			for tech_req_ind in tech_reqs {
				if self.techs_progress[*tech_req_ind as usize] != TechProg::Finished {
					// only add if not already scheduled
					if !tech_reqs_list.contains(tech_req_ind) {
						tech_reqs_list.push(*tech_req_ind);
						self.build_undiscov_tech_req_list(*tech_req_ind, tech_templates, tech_reqs_list);
					}
				}
			}
		}
	}

	// recursively discover tech and all of its undiscovered research requirements
	// ***** ASSUMES this is for the current player & updates `production_options`
	pub fn force_discover_undiscov_tech<'bt,'ut,'rt,'dt>(&mut self, tech_req_ind: SmSvType, tech_templates: &Vec<TechTemplate>,
			bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, production_options: &mut ProdOptions<'bt,'ut,'rt,'dt>,
			l: &Localization) {
		let mut techs_to_discover = Vec::with_capacity(tech_templates.len());
		techs_to_discover.push(tech_req_ind);
		self.build_undiscov_tech_req_list(tech_req_ind, tech_templates, &mut techs_to_discover);
		
		for tech in techs_to_discover.iter() {
			self.techs_progress[*tech as usize] = TechProg::Finished;
			
			// remove from scheduled
			for (sched_ind, tech_scheduled) in self.techs_scheduled.iter().enumerate() {
				if *tech_scheduled == *tech {
					self.techs_scheduled.remove(sched_ind);
					break;
				}
			}
		}
		
		*production_options = init_bldg_prod_windows(bldg_templates, self, l);
	}
	
	// tech_sel is the index into tech_templates
	pub fn start_researching(&mut self, tech_sel: SmSvType, tech_templates: &Vec<TechTemplate>) {
		if self.techs_progress[tech_sel as usize] == TechProg::Finished || // already discovered
		   self.techs_scheduled.contains(&tech_sel) // already scheduled
			{return;} // already discovered
		
		self.techs_scheduled.push(tech_sel);
		self.schedule_undiscov_tech_req(tech_sel, tech_templates);
	}
	
	pub fn stop_researching(&mut self, tech_sel: SmSvType, tech_templates: &Vec<TechTemplate>) {
		// remove all children requiring the tech to be removed
		fn chk_tech_children(pstats: &mut Stats, tech_sel: SmSvType, tech_templates: &Vec<TechTemplate>) {
			
			for (i, t) in tech_templates.iter().enumerate() {
				if let Some(tech_req) = &t.tech_req {
					let i = i as SmSvType;
					
					// check if tech actually scheduled
					if let Some(sched_ind) = find_index(i, &pstats.techs_scheduled) {
						// check if requires parent tech
						if tech_req.contains(&tech_sel) {
							pstats.techs_scheduled.swap_remove(sched_ind);
							chk_tech_children(pstats, i, tech_templates);
							// should probably check for infinite loops... in config
						}
					}
				}
			}
		}
		
		let sched_ind = find_index(tech_sel, &self.techs_scheduled).unwrap();
		// ^ fails if tech_sel is not actually scheduled
		
		self.techs_scheduled.swap_remove(sched_ind);
		
		chk_tech_children(self, tech_sel, tech_templates);
	}
}

