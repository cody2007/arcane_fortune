use super::*;
use crate::units::*;
pub struct SectorsWindowState {
	pub mode: usize,
	pub sector_action: SectorAction
}

pub enum SectorAction {
	GoTo, AddTo, Delete,
	SetBrigadeToRepairWalls(String) // the brigade name
}

impl SectorsWindowState {
	pub fn print<'bt,'ut,'rt,'dt>(&self, pstats: &Stats, map_data: &mut MapData,
			dstate: &mut DispState<'_,'_,'bt,'ut,'rt,'dt>) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let map_sz = *map_data.map_szs.last().unwrap();
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let sectors = sector_list(&pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
		
		let list_pos = dstate.print_list_window(self.mode, dstate.local.Select_map_sector.clone(), sectors, Some(w), label_txt_opt, 0, None);
		dstate.renderer.mv(list_pos.sel_loc.y as i32, list_pos.sel_loc.x as i32);
		
		UIModeControl::UnChgd
	}
	
	pub fn keys<'bt,'ut,'rt,'dt>(&mut self, players: &mut Vec<Player>, map_data: &mut MapData, map_sz: MapSz, dstate: &mut DispState) -> UIModeControl<'bt,'ut,'rt,'dt> {
		let mut w = 0;
		let mut label_txt_opt = None;
		let cursor_coord = dstate.iface_settings.cursor_to_map_coord_zoomed_in(map_data);
		
		let pstats = &mut players[dstate.iface_settings.cur_player as usize].stats;	
		let entries = sector_list(pstats, cursor_coord, &mut w, &mut label_txt_opt, map_sz, &dstate.local);
		let entries_present = entries.options.len() > 0;
		
		//////////////////////// handle keys
		macro_rules! enter_action {($mode: expr) => {
			// move cursor to entry
			let coord = match entries.options[$mode].arg {
				ArgOptionUI::SectorInd(sector_ind) => {
					let mode = $mode;
					if let Some(sector) = pstats.sectors.get_mut(sector_ind) {
						match &self.sector_action {
							SectorAction::GoTo => {
								sector.average_coord(map_sz)
							} SectorAction::AddTo => {
								dstate.iface_settings.add_action_to = AddActionTo::NoUnit {
									action: ActionMeta::new(
										ActionType::SectorCreation {
											nm: sector.nm.clone(),
											creation_type: SectorCreationType::AddTo,
											start_coord: None,
											end_coord: None
										}),
								};
								
								sector.average_coord(map_sz) // where the cursor / view will move to
							} SectorAction::Delete => {
								pstats.sectors.swap_remove(mode);
								return UIModeControl::Closed;
							} SectorAction::SetBrigadeToRepairWalls(brigade_nm) => {
								let sector_nm = sector.nm.clone();
								let brigade = pstats.brigade_frm_nm_mut(&brigade_nm);
								brigade.repair_sector_walls = Some(sector_nm);
								return UIModeControl::New(UIMode::BrigadesWindow(BrigadesWindowState {
									mode: 0,
									brigade_action: BrigadeAction::ViewBrigadeUnits {
										brigade_nm: brigade_nm.clone()
									}
								}));
							}
						}
					}else{return UIModeControl::Closed;}
				}
				_ => {panicq!("unit inventory list argument option not properly set");}
			};
			
			return UIModeControl::CloseAndGoTo(coord);
		};};
		if let Some(ind) = dstate.buttons.list_item_clicked(&dstate.mouse_event) {enter_action!(ind);}
		
		match dstate.key_pressed {
			k if entries_present && (dstate.kbd.down(k)) => {
				if (self.mode + 1) <= (entries.options.len()-1) {
					self.mode += 1;
				}else{
					self.mode = 0;
				}
			
			// up
			} k if entries_present && (dstate.kbd.up(k)) => {
				if self.mode > 0 {
					self.mode -= 1;
				}else{
					self.mode = entries.options.len() - 1;
				}
				
			// enter
			} k if entries_present && (k == dstate.kbd.enter) => {
				enter_action!(self.mode);
			} _ => {}
		}
		UIModeControl::UnChgd
	}
}
