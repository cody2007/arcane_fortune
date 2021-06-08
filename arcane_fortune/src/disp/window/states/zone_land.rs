// called when a worker is asked to zone land
// gets the zone type and density
use super::*;
use crate::units::{WORKER_NM, ActionType};

pub enum ZoneLandState {
	GetZoneType {mode: usize}, // step one
	GetZoneDensity {mode: usize, ztype: ZoneType} // step two
}

impl ZoneLandState {
	pub fn new() -> Self {Self::GetZoneType {mode: 0}}
	
	pub fn print<'bt,'ut,'rt,'dt>(&self, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let list_pos = match self {
			Self::GetZoneType {mode} => {
				let list = zone_types_list(&dstate.local);
				dstate.print_list_window(*mode, dstate.local.Select_a_zone_type.clone(), list, None, None, 0, None)
			} Self::GetZoneDensity {mode, ..} => {
				let list = zone_density_list(&dstate.local);
				dstate.print_list_window(*mode, dstate.local.Select_a_zone_density.clone(), list, None, None, 0, None)
			}
		};
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
	
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, units: &Vec<Unit<'bt,'ut,'rt,'dt>>, players: &Vec<Player<'bt,'ut,'rt,'dt>>, map_data: &mut MapData<'rt>,
			exf: &HashedMapEx<'bt,'ut,'rt,'dt>, dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		match self {
			Self::GetZoneType {ref mut mode} => {
				if zone_types_list(&dstate.local).list_mode_update_and_action(mode, dstate) {
					let ztype = match mode {
						0 => ZoneType::Agricultural,
						1 => ZoneType::Residential,
						2 => ZoneType::Business,
						3 => ZoneType::Industrial,
						_ => panicq!("unknown option")
					};
					
					*self = Self::GetZoneDensity {mode: 0, ztype};
				}
			} Self::GetZoneDensity {ref mut mode, ztype} => {
				if zone_density_list(&dstate.local).list_mode_update_and_action(mode, dstate) {
					let worker_unit_inds: Vec<usize> = {
						let unit_inds = dstate.iface_settings.sel_units_owned(&players[dstate.iface_settings.cur_player as usize].stats, units, map_data, exf);
						unit_inds.iter().cloned().filter(|&ind| units[ind].template.nm[0] == WORKER_NM).collect()
					};
					
					// zone if in build list mode or there is a worker selected
					if dstate.iface_settings.add_action_to.is_build_list() || worker_unit_inds.iter().any(|&ind| units[ind].template.nm[0] == WORKER_NM) {
						let act = {
							let density = match mode {
								0 => ZoneDensity::Low,
								1 => ZoneDensity::Medium,
								2 => ZoneDensity::High,
								_ => panicq!("unknown option")
							};
							
							ActionType::WorkerZone {
								valid_placement: false,
								zone: Zone {
									ztype: *ztype,
									density
								},
								start_coord: None,
								end_coord: None
							}
						};
						
						dstate.iface_settings.start_build_mv_mode(act, &worker_unit_inds, units, map_data);
					}
					return UIModeControl::Closed;
				}
			}
		}
		
		UIModeControl::UnChgd
	}
}

fn zone_types_list<'bt,'ut,'rt,'dt>(l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let nms = vec![
		l.Agricultural_menu.as_str(), 
		l.Residential_menu.as_str(),
		l.Business_menu.as_str(),
		l.Industrial_menu.as_str()
	];
	OptionsUI::new(&nms)
}

fn zone_density_list<'bt,'ut,'rt,'dt>(l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let nms = vec![
		l.Low_menu.as_str(), 
		l.Medium_menu.as_str(),
		l.High_menu.as_str()
	];
	OptionsUI::new(&nms)
}

