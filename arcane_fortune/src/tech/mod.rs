use crate::renderer::*;
use crate::saving::*;
use crate::config_load::*;
use crate::map::{TechProg};
use crate::player::{Stats, Player};
use crate::disp::window::*;
use crate::units::UnitTemplate;
use crate::gcore::Relations;
use crate::localization::Localization;
use crate::containers::*;
use std::process::exit;
#[cfg(feature="profile")]
use crate::gcore::profiling::*;

pub mod vars; pub use vars::*;
pub mod disp; pub use disp::*;

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
pub fn research_techs<'f,'bt,'ut,'rt,'dt>(players: &mut Vec<Player<'bt,'ut,'rt,'dt>>, relations: &mut Relations,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, disp: &mut Disp<'f,'_,'bt,'ut,'rt,'dt>) {
	#[cfg(feature="profile")]
	let _g = Guard::new("research_techs");
	
	struct TransferTech {frm_owner: usize, to_owner: usize}
	
	let mut transfer_techs = Vec::with_capacity(players.len()); // kingdoms transfer techs to the parent empire
	
	for player in players.iter_mut() { // for all players
		let tech_bonus_factor = player.tech_bonus_factor();
		let ps = &mut player.stats;
		if let Some(t_researching) = ps.techs_scheduled.last() { // if currently researching
			let t_researching = *t_researching as usize;
			
			// if progress is logged for this tech (should be)
			if let TechProg::Prog(ref mut prog) = ps.techs_progress[t_researching] {
				// increment research progress
				let research_per_turn = (ps.research_per_turn as f32 * tech_bonus_factor).round() as u32;
				if (research_per_turn + *prog) <= temps.techs[t_researching].research_req {
					*prog += research_per_turn;
				
				// finish research
				}else{
					ps.techs_progress[t_researching] = TechProg::Finished;
					ps.techs_scheduled.pop();
					
					// transfer tech from kingdoms to parent empire?
					if let Some(parent_empire) = relations.kingdom_of(player.id as usize) {
						transfer_techs.push(TransferTech {
							frm_owner: player.id as usize,
							to_owner: parent_empire
						});
					}
					
					// update building unit production window options
					// prompt player to select new tech if nothing else has been scheduled, or show tech discovered window
					if player.id == disp.state.iface_settings.cur_player {
						disp.state.production_options = init_bldg_prod_windows(temps.bldgs, ps, &disp.state.local);
						
						// prompt to select new tech for research
						if ps.techs_scheduled.len() == 0 {
							disp.create_tech_window(true);
						// tech discovered window
						}else{
							disp.create_tech_discovered_window(t_researching);
						}
					}
				}
			} else {panicq!("Finished researching tech but still scheduled");}
		}
	}
	
	// transfer techs from kingdoms to parent empires
	for transfer_tech in transfer_techs {
		transfer_undiscov_tech(transfer_tech.frm_owner, transfer_tech.to_owner, players, temps, &mut disp.state);
	}
}

impl <'bt,'ut,'rt,'dt>Stats<'bt,'ut,'rt,'dt> {
	pub fn discover_tech(&mut self, tech_ind: usize, temps: &Templates<'bt,'ut,'rt,'dt,'_>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) {
		self.techs_progress[tech_ind] = TechProg::Finished;
		// remove from schedule
		if let Some(tech_ind_rm) = self.techs_scheduled.iter().position(|&tech_scheduled| tech_scheduled == tech_ind as SmSvType) {
			self.techs_scheduled.swap_remove(tech_ind_rm);
		}
		
		// update building unit production window options
		if self.id == dstate.iface_settings.cur_player {
			dstate.production_options = init_bldg_prod_windows(temps.bldgs, self, &dstate.local);
		}
	}
	
	// max defensive unit the player can produce
	pub fn max_defensive_unit(&self, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
		if let Some(ut) = unit_templates.iter().
				filter(|ut| self.unit_producable(&ut)).
				max_by_key(|ut| ut.max_health) {
			ut
		}else {panicq!("could not find max defensive unit")}
	}
	
	// max attack unit the player can produce
	pub fn max_attack_unit(&self, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
		if let Some(ut) = unit_templates.iter().
				filter(|ut| self.unit_producable(&ut) && !ut.attack_per_turn.is_none()).
				max_by_key(|ut| ut.attack_per_turn.unwrap()) {
			ut
		}else {panicq!("could not find any attack units")}
	}
	
	// max seige unit the player can produce
	pub fn max_siege_unit(&self, unit_templates: &'ut Vec<UnitTemplate<'rt>>) -> &'ut UnitTemplate<'rt> {
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
	pub fn force_discover_undiscov_tech<'f>(&mut self, tech_req_ind: SmSvType, temps: &Templates<'bt,'ut,'rt,'dt,'_>,
			dstate: &mut DispState<'f,'_,'bt,'ut,'rt,'dt>) {
		let mut techs_to_discover = Vec::with_capacity(temps.techs.len());
		techs_to_discover.push(tech_req_ind);
		self.build_undiscov_tech_req_list(tech_req_ind, &temps.techs, &mut techs_to_discover);
		
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
		
		dstate.production_options = init_bldg_prod_windows(&temps.bldgs, self, &dstate.local);
	}
	
	// tech_sel is the index into tech_templates
	pub fn start_researching(&mut self, tech_sel: SmSvType, tech_templates: &Vec<TechTemplate>) {
		if self.techs_progress[tech_sel as usize] == TechProg::Finished || // already discovered
		   self.techs_scheduled.contains(&tech_sel) // already scheduled
			{return;} // already discovered
		
		self.techs_scheduled.clear();
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
	
	pub fn discovered_techs(&self) -> Vec<usize> {
		let mut tech_inds_discov = Vec::with_capacity(self.techs_progress.len());
		
		for (tech_ind, _tech_progress) in self.techs_progress.iter().enumerate()
				.filter(|(_tech_id, tech_progress)| **tech_progress == TechProg::Finished) {
			tech_inds_discov.push(tech_ind);
		}
		tech_inds_discov
	}
}

pub fn transfer_undiscov_tech<'bt,'ut,'rt,'dt>(frm_owner: usize, to_owner: usize,
		players: &mut Vec<Player<'bt,'ut,'rt,'dt>>,
		temps: &Templates<'bt,'ut,'rt,'dt,'_>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) {
	let techs_src = players[frm_owner].stats.discovered_techs();
	let player_dest = &mut players[to_owner].stats;
	let techs_dest = player_dest.discovered_techs();
	
	for tech_src in techs_src.iter().filter(|tech_src| !techs_dest.contains(tech_src)) {
		player_dest.discover_tech(*tech_src, temps, dstate);
	}
}

