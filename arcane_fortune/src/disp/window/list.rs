use super::*;
use crate::resources::N_RESOURCES_DISCOV_LOG;
use crate::gcore::{GameDifficulties, Log, LogType};
use crate::doctrine::available_doctrines;
use crate::localization::Localization;
use crate::player::{Stats, PlayerType, Player};
use crate::nobility::House;

pub fn noble_houses_list<'bt,'ut,'rt,'dt>(cur_player: usize, relations: &Relations,
		players: &Vec<Player>, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(players.len());
	
	for house_ind in relations.noble_houses(cur_player) {
		nms_string.push(l.House_of.replace("[]", &players[house_ind].personalization.nm));
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	register_shortcuts(&nms, &mut opts);
	
	opts
}

pub fn encyclopedia_bldg_list<'bt,'ut,'rt,'dt>(bldg_templates: &'bt Vec<BldgTemplate>,
		l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut taxable_nms = Vec::with_capacity(bldg_templates.len());
	let mut gov_nms = Vec::with_capacity(bldg_templates.len() + 2);
	let mut exemplar_nms_ref = Vec::with_capacity(bldg_templates.len() + 2);
	
	let mut taxable_inds = Vec::with_capacity(bldg_templates.len());
	let mut gov_inds = Vec::with_capacity(bldg_templates.len() + 2);
	
	gov_nms.push(String::from("\\Government funded"));
	gov_inds.push(None);
	
	for (bldg_ind, bt) in bldg_templates.iter().enumerate() {
		let nm = format!("  {}", bt.nm[l.lang_ind]);
		
		match bt.bldg_type {
			BldgType::Taxable(_) => {
				taxable_nms.push(nm);
				taxable_inds.push(Some(bldg_ind));
			}
			BldgType::Gov(_) => {
				gov_nms.push(nm);
				gov_inds.push(Some(bldg_ind));
			}
		}
	}
	
	gov_nms.push(String::from("\\Civilian (taxable)"));
	gov_inds.push(None);
	
	gov_nms.extend(taxable_nms);
	gov_inds.extend(taxable_inds);
	
	for exemplar_nm in gov_nms.iter() {
		exemplar_nms_ref.push(exemplar_nm.as_str());
	}
	
	let mut exemplar_options = OptionsUI {options: Vec::with_capacity(gov_nms.len()), max_strlen: 0};
	
	register_shortcuts(&exemplar_nms_ref, &mut exemplar_options);
	
	// associate owner_id w/ each menu entry
	for (i, opt) in exemplar_options.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::Ind(gov_inds[i]);
	}
	
	exemplar_options
}

// for creating list to display of units in `units_use`. w is set to be the width of the window to be created
pub fn unit_list_frm_vec<'bt,'ut,'rt,'dt>(unit_inds_use: &Vec<usize>, 
		units: &Vec<Unit<'bt,'ut,'rt,'dt>>, cur_coord: Coord, pstats: &Stats,
		w: &mut usize, label_txt_opt: &mut Option<String>, map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	struct UnitEntry {
		nm: String,
		brigade: String,
		health: String,
		cost: String,
		dist: String,
		action: String,
		unit_ind: usize
	}
	
	// get all units owned by player
	let mut unit_entries = Vec::with_capacity(unit_inds_use.len());
	let mut max_nm_len = 0;
	let mut max_brigade_len = 0;
	let mut max_health_len = 0;
	let mut max_cost_len = 0;
	let mut max_dist_len = 0;
	let mut max_action_len = 0;
	
	// get all units owned by player, gather costs, names
	for unit_ind in unit_inds_use.iter() {
		let u = &units[*unit_ind];
		let action = if let Some(action) = u.action.last() {
			action.action_type.nm(l)
		}else {String::from(&l.Idle)};
		
		let brigade = if let Some(brigade) = pstats.unit_brigade_nm(*unit_ind) {
			brigade.to_string()
		}else{String::from("None")};
		
		let entry = UnitEntry {
			nm: format!("{} ({})   ", u.nm, u.template.nm[l.lang_ind]),
			brigade,
			health: format!("{}%", u.health()),
			cost: format!("{}", u.template.upkeep),
			dist: direction_string(cur_coord, Coord::frm_ind(u.return_coord(), map_sz), map_sz),
			action,
			unit_ind: *unit_ind
		};
		
		// keep track of max width of text
		if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
		if max_brigade_len < entry.brigade.len() {max_brigade_len = entry.brigade.len();}
		if max_health_len < entry.health.len() {max_health_len = entry.health.len();}
		if max_cost_len < entry.cost.len() {max_cost_len = entry.cost.len();}
		if max_dist_len < entry.dist.len() {max_dist_len = entry.dist.len();}
		if max_action_len < entry.action.len() {max_action_len = entry.action.len();}
		
		unit_entries.push(entry);
	}
	
	///// format txt only if it player has units
	
	let mut label_txt = String::new(); // label before entries are printed
	if unit_entries.len() > 0 {
		*w = max_nm_len + max_brigade_len + max_health_len + max_cost_len + max_dist_len + max_action_len + 4*4 + 4;
		
		// format with gap between `name`, `cost`, `dist`, and `action`
		for entry in unit_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};};
			
			// brigade
			gap!(max_nm_len + max_brigade_len - entry.nm.len() - entry.brigade.len());
			entry.nm.push_str(&entry.brigade);
			
			// health
			gap!(4 + max_health_len - entry.health.len());
			entry.nm.push_str(&entry.health);
			
			// cost
			gap!(4 + max_cost_len - entry.cost.len());
			entry.nm.push_str(&entry.cost);
			
			// dist
			gap!(4 + max_dist_len - entry.dist.len());
			entry.nm.push_str(&entry.dist);
			
			// action
			gap!(4 + max_action_len - entry.action.len());
			entry.nm.push_str(&entry.action);
		}
		
		// label before entries are printed
		for _ in 0..(max_nm_len + max_brigade_len - l.Brigade.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Brigade);
		
		for _ in 0..(4 + max_health_len - l.Health.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Health);
		
		for _ in 0..(4+max_cost_len - l.Cost.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Cost);
		
		for _ in 0..max_dist_len { label_txt.push(' ');}
		label_txt.push_str(&l.Dist);
		
		for _ in 0..(max_action_len - 2) { label_txt.push(' ');}
		label_txt.push_str(&l.Action);

		*label_txt_opt = Some(label_txt);
	}else{
		*w = 29;
		*label_txt_opt = None;
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut unit_nms = Vec::with_capacity(unit_entries.len());
	
	for unit_entry in unit_entries.iter() {
		unit_nms.push(unit_entry.nm.as_str());
	}
	
	// call register_shortcuts
	let mut owned_units = OptionsUI {options: Vec::with_capacity(unit_nms.len()), max_strlen: 0};
	register_shortcuts(&unit_nms, &mut owned_units);
	
	// associate unit_ind w/ each menu entry
	for (opt, entry) in owned_units.options.iter_mut().zip(unit_entries.iter()) {
		opt.arg = ArgOptionUI::UnitInd(entry.unit_ind);
	}
	
	return owned_units;
}

// for creating list to display of player's owned units. w is set to be the width of the window to be created
pub fn owned_unit_list<'bt,'ut,'rt,'dt>(units: &Vec<Unit<'bt,'ut,'rt,'dt>>, cur_player: SmSvType, cur_coord: Coord,
		pstats: &Stats, w: &mut usize, label_txt_opt: &mut Option<String>,
		map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	
	let mut unit_inds_use = Vec::with_capacity(units.len());
	
	// get all units owned by player, gather costs, names
	for (unit_ind, _u) in units.iter().enumerate().filter(|(_, u)| u.owner_id == cur_player) {
		unit_inds_use.push(unit_ind);
	}
	
	unit_list_frm_vec(&unit_inds_use, units, cur_coord, pstats, w, label_txt_opt, map_sz, l)
}

// w is set to be the width of the window to be created
pub fn sector_list<'bt,'ut,'rt,'dt>(pstats: &Stats, cur_coord: Coord,
		w: &mut usize, label_txt_opt: &mut Option<String>, map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	struct SectorEntry {
		nm: String,
		dist: String,
	}
	
	// get all units owned by player
	let mut sector_entries = Vec::with_capacity(pstats.sectors.len());
	let mut max_nm_len = 0;
	let mut max_dist_len = l.Dist.len();
	
	// gather costs, names
	for sector in pstats.sectors.iter() {
		let avg_coord = sector.average_coord(map_sz);
			
		let entry = SectorEntry {
			nm: format!("{}   ", sector.nm),
			dist: direction_string(cur_coord, Coord::frm_ind(avg_coord, map_sz), map_sz),
		};
		
		// keep track of max width of text
		if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
		if max_dist_len < entry.dist.len() {max_dist_len = entry.dist.len();}
		
		sector_entries.push(entry);
	}
	
	///// format txt only if it player has any sectors
	
	let mut label_txt = String::new(); // label before entries are printed
	if sector_entries.len() > 0 {
		*w = max_nm_len + max_dist_len + 8 + 4;
		
		// format with gap between `name`, `dist`
		for entry in sector_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};};
			
			// dist
			gap!(*w - entry.nm.len() - entry.dist.len() - 4);
			entry.nm.push_str(&entry.dist);
		}
		
		// label before entries are printed
		for _ in 0..(*w - l.Dist.len() - 4) { label_txt.push(' ');}
		label_txt.push_str(&l.Dist);
		
		*label_txt_opt = Some(label_txt);
	}else{
		*w = 29;
		*label_txt_opt = None;
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut entries = Vec::with_capacity(sector_entries.len());
	
	for sector in sector_entries.iter() {
		entries.push(sector.nm.as_str());
	}
	
	// call register_shortcuts
	let mut owned_sectors = OptionsUI {options: Vec::with_capacity(entries.len()), max_strlen: 0};
	register_shortcuts(&entries, &mut owned_sectors);
	
	// associate unit_ind w/ each menu entry
	for (entry_ind, opt) in owned_sectors.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::SectorInd(entry_ind);
	}
	
	return owned_sectors;
}

// w is set to be the width of the window to be created
pub fn brigades_list<'bt,'ut,'rt,'dt>(pstats: &Stats, w: &mut usize, label_txt_opt: &mut Option<String>, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	struct BrigadeEntry {
		nm: String,
		n_units: String,
		brigade_ind: usize
	}
	
	let mut brigade_entries = Vec::with_capacity(pstats.brigades.len());
	let mut max_nm_len = 0;
	let mut max_n_units_len = l.N_units.len();
	
	// get all brigades, gather names and n_units
	for (brigade_ind, brigade) in pstats.brigades.iter().enumerate() {
		let entry = BrigadeEntry {
			nm: format!("{}   ", brigade.nm),
			n_units: format!("{}", brigade.unit_inds.len()),
			brigade_ind
		};
		
		// keep track of max width of text
		if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
		if max_n_units_len < entry.n_units.len() {max_n_units_len = entry.n_units.len();}
		
		brigade_entries.push(entry);
	}
	
	///// format txt only if it player has brigades
	
	let mut label_txt = String::new(); // label before entries are printed
	if brigade_entries.len() > 0 {
		*w = max_nm_len + max_n_units_len + 8 + 4;
		
		// format with gap between `name`, `n_units`
		for entry in brigade_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};};
			
			// cost
			gap!(*w - entry.nm.len() - entry.n_units.len() - 4);
			entry.nm.push_str(&entry.n_units);
		}
		
		// label before entries are printed
		for _ in 0..(*w - l.N_units.len() - 4) { label_txt.push(' ');}
		label_txt.push_str(&l.N_units);
		
		*label_txt_opt = Some(label_txt);
	}else{
		*w = 29;
		*label_txt_opt = None;
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut brigade_nms = Vec::with_capacity(brigade_entries.len());
	
	for entry in brigade_entries.iter() {
		brigade_nms.push(entry.nm.as_str());
	}
	
	// call register_shortcuts
	let mut owned_brigades = OptionsUI {options: Vec::with_capacity(brigade_nms.len()), max_strlen: 0};
	register_shortcuts(&brigade_nms, &mut owned_brigades);
	
	// associate brigade_ind w/ each menu entry
	for (opt, entry) in owned_brigades.options.iter_mut().zip(brigade_entries.iter()) {
		opt.arg = ArgOptionUI::BrigadeInd(entry.brigade_ind);
	}
	
	return owned_brigades;
}

// w is set to be the width of the window to be created
pub fn brigade_unit_list<'bt,'ut,'rt,'dt>(brigade_nm: &String, pstats: &Stats, units: &Vec<Unit<'bt,'ut,'rt,'dt>>,
		cur_coord: Coord, w: &mut usize, label_txt_opt: &mut Option<String>, map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let brigade = pstats.brigade_frm_nm(brigade_nm);
	unit_list_frm_vec(&brigade.unit_inds, units, cur_coord, pstats, w, label_txt_opt, map_sz, l)
}

pub fn brigade_build_list<'bt,'ut,'rt,'dt>(brigade_nm: &String, pstats: &Stats, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let brigade = pstats.brigade_frm_nm(brigade_nm);
	
	let mut nms_string = Vec::with_capacity(brigade.build_list.len());
	
	for build_entry in brigade.build_list.iter() {
		nms_string.push(build_entry.action_type.nm(l));
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut opts_ui = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut opts_ui);
	opts_ui
}

// for creating list to display of player's owned improvement buildings. w is set to be the width of the window to be created
pub fn owned_improvement_bldgs_list<'bt,'ut,'rt,'dt>(bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>,
		doctrine_templates: &Vec<DoctrineTemplate>, cur_player: SmSvType, cur_coord: Coord,
		w: &mut usize, label_txt_opt: &mut Option<String>, map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	
	struct CityEntry {
		nm: String,
		coord: Coord
	}
	
	let mut city_entries = Vec::with_capacity(bldgs.len());
	for b in bldgs.iter() {
		if b.owner_id != cur_player {continue;}
		if let BldgArgs::CityHall {nm, ..} = &b.args {
			city_entries.push(CityEntry {nm: nm.clone(), coord: Coord::frm_ind(b.coord, map_sz)});
		}
	}
	
	struct BldgEntry {
		nm: String,
		cost: String,
		dist: String,
		city_nm: String,
		dedication: String,
		bldg_ind: usize
	}
	
	// get all improvement bldgs owned by player
	let mut bldg_entries = Vec::with_capacity(bldgs.len());
	let mut max_nm_len = 0;
	let mut max_cost_len = 0;
	let mut max_dist_len = 0;
	let mut max_city_nm_len = 0;
	let mut max_dedication_len = 0;
	
	// get all improvement bldgs owned by player, gather costs, names
	for (bldg_ind, b) in bldgs.iter().enumerate() {
		if b.owner_id != cur_player {continue;}
		if let Some(_) = b.template.units_producable {continue;}
		
		match b.template.bldg_type {
			BldgType::Taxable {..} => {continue;}
			BldgType::Gov {..} => {
				let b_coord = Coord::frm_ind(b.coord, map_sz);
				let city_nm = if let Some(min_city) = city_entries.iter().min_by_key(|c| 
					manhattan_dist(c.coord, b_coord, map_sz)) {min_city.nm.clone()
				}else{l.None.clone()};
				
				let dedication = if b.template.doctrinality_bonus > 0. && b.doctrine_dedication != &doctrine_templates[0] {
					b.doctrine_dedication.nm[l.lang_ind].clone()
				}else{l.na.clone()};
				
				let entry = BldgEntry {
					nm: b.template.nm[l.lang_ind].clone(),
					cost: format!("{}", b.template.upkeep),
					dist: direction_string(cur_coord, b_coord, map_sz),
					city_nm,
					dedication,
					bldg_ind
				};
				
				// keep track of max width of text
				if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
				if max_cost_len < entry.cost.len() {max_cost_len = entry.cost.len();}
				if max_dist_len < entry.dist.len() {max_dist_len = entry.dist.len();}
				if max_city_nm_len < entry.city_nm.len() {max_city_nm_len = entry.city_nm.len();}
				if max_dedication_len < entry.dedication.len() {max_dedication_len = entry.dedication.len();}
				
				bldg_entries.push(entry);
			}
		}
	}
	
	///// format txt only if it player has bldgs
	
	let mut label_txt = String::new(); // label before entries are printed
	if bldg_entries.len() > 0 {
		max_cost_len = max(max_cost_len, l.Cost.len());
		max_dist_len = max(max_dist_len, l.Dist.len());
		max_city_nm_len = max(max_city_nm_len, l.City_nm.len());
		max_dedication_len = max(max_dedication_len, l.Dedication.len());
		
		*w = max_nm_len + max_cost_len + max_dist_len + max_city_nm_len + max_dedication_len + 4*4 + 4;
		
		// format with gap between `name`, `cost`, `dist`, `city name`, and `dedication`
		for entry in bldg_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};};
			
			// cost
			gap!(4 + max_nm_len + max_cost_len - entry.nm.len() - entry.cost.len());
			entry.nm.push_str(&entry.cost);
			
			// dist
			gap!(4 + max_dist_len - entry.dist.len());
			entry.nm.push_str(&entry.dist);
			
			// city name
			gap!(4 + max_city_nm_len - entry.city_nm.len());
			entry.nm.push_str(&entry.city_nm);
			
			// dedication
			gap!(4 + max_dedication_len - entry.dedication.len());
			entry.nm.push_str(&entry.dedication);
		}
		
		// label before entries are printed
		for _ in 0..(4 + max_nm_len + max_cost_len - l.Cost.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Cost);
		
		for _ in 0..(4 + max_dist_len - l.Dist.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Dist);
		
		for _ in 0..(4 + max_city_nm_len - l.City_nm.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.City_nm);
		
		for _ in 0..(4 + max_dedication_len - l.Dedication.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Dedication);

		*label_txt_opt = Some(label_txt);
	}else{
		*w = 29;
		*label_txt_opt = None;
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(bldg_entries.len());
	
	for entry in bldg_entries.iter() {
		nms.push(entry.nm.as_str());
	}
	
	// call register_shortcuts
	let mut owned_bldgs = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	register_shortcuts(&nms, &mut owned_bldgs);
	
	// associate unit_ind w/ each menu entry
	for (opt, entry) in owned_bldgs.options.iter_mut().zip(bldg_entries.iter()) {
		opt.arg = ArgOptionUI::BldgInd(entry.bldg_ind);
	}
	
	return owned_bldgs;
}

// for creating list to display of player's owned unit-producing (non-city hall) buildings. w is set to be the width of the window to be created
pub fn owned_military_bldgs_list<'bt,'ut,'rt,'dt>(bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, cur_player: SmSvType, cur_coord: Coord,
		w: &mut usize, label_txt_opt: &mut Option<String>, map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	
	struct CityEntry {
		nm: String,
		coord: Coord
	}
	
	let mut city_entries = Vec::with_capacity(bldgs.len());
	for b in bldgs.iter() {
		if b.owner_id != cur_player {continue;}
		if let BldgArgs::CityHall {nm, ..} = &b.args {
			city_entries.push(CityEntry {nm: nm.clone(), coord: Coord::frm_ind(b.coord, map_sz)});
		}
	}
	
	struct BldgEntry {
		nm: String,
		cost: String,
		dist: String,
		city_nm: String,
		producing: String,
		bldg_ind: usize
	}
	
	// get all improvement bldgs owned by player
	let mut bldg_entries = Vec::with_capacity(bldgs.len());
	let mut max_nm_len = 0;
	let mut max_cost_len = 0;
	let mut max_dist_len = 0;
	let mut max_city_nm_len = 0;
	let mut max_producing_len = 0;
	
	// get all non-city unit-producing bldgs owned by player, gather costs, names
	for (bldg_ind, b) in bldgs.iter().enumerate() {
		if b.owner_id != cur_player {continue;}
		if let BldgArgs::GenericProducable {production, ..} = &b.args {
			let b_coord = Coord::frm_ind(b.coord, map_sz);
			let city_nm = if let Some(min_city) = city_entries.iter().min_by_key(|c| 
				manhattan_dist(c.coord, b_coord, map_sz)) {min_city.nm.clone()
			}else{l.None.clone()};
			
			let producing = if let Some(ProductionEntry {production, ..}) = production.last() {
				production.nm[l.lang_ind].clone()
			}else{l.None.clone()};
			
			let entry = BldgEntry {
				nm: b.template.nm[l.lang_ind].clone(),
				cost: format!("{}", b.template.upkeep),
				dist: direction_string(cur_coord, b_coord, map_sz),
				city_nm,
				producing,
				bldg_ind
			};
			
			// keep track of max width of text
			if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
			if max_cost_len < entry.cost.len() {max_cost_len = entry.cost.len();}
			if max_dist_len < entry.dist.len() {max_dist_len = entry.dist.len();}
			if max_city_nm_len < entry.city_nm.len() {max_city_nm_len = entry.city_nm.len();}
			if max_producing_len < entry.producing.len() {max_producing_len = entry.producing.len();}
			
			bldg_entries.push(entry);
		}
	}
	
	///// format txt only if it player has bldgs
	
	let mut label_txt = String::new(); // label before entries are printed
	if bldg_entries.len() > 0 {
		max_cost_len = max(max_cost_len, l.Cost.len());
		max_dist_len = max(max_dist_len, l.Dist.len());
		max_city_nm_len = max(max_city_nm_len, l.City_nm.len());
		max_producing_len = max(max_producing_len, l.Producing.len());
		
		*w = max_nm_len + max_cost_len + max_dist_len + max_city_nm_len + max_producing_len + 4*4 + 4;
		
		// format with gap between `name`, `cost`, `dist`, `city name`, and `producing`
		for entry in bldg_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};};
			
			// cost
			gap!(4 + max_nm_len + max_cost_len - entry.nm.len() - entry.cost.len());
			entry.nm.push_str(&entry.cost);
			
			// dist
			gap!(4 + max_dist_len - entry.dist.len());
			entry.nm.push_str(&entry.dist);
			
			// city name
			gap!(4 + max_city_nm_len - entry.city_nm.len());
			entry.nm.push_str(&entry.city_nm);
			
			// producing
			gap!(4 + max_producing_len - entry.producing.len());
			entry.nm.push_str(&entry.producing);
		}
		
		// label before entries are printed
		for _ in 0..(4 + max_nm_len + max_cost_len - l.Cost.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Cost);
		
		for _ in 0..(4 + max_dist_len - l.Dist.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Dist);
		
		for _ in 0..(4 + max_city_nm_len - l.City_nm.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.City_nm);
		
		for _ in 0..(4 + max_producing_len - l.Producing.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Producing);

		*label_txt_opt = Some(label_txt);
	}else{
		*w = 29;
		*label_txt_opt = None;
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(bldg_entries.len());
	
	for entry in bldg_entries.iter() {
		nms.push(entry.nm.as_str());
	}
	
	// call register_shortcuts
	let mut owned_bldgs = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	register_shortcuts(&nms, &mut owned_bldgs);
	
	// associate unit_ind w/ each menu entry
	for (opt, entry) in owned_bldgs.options.iter_mut().zip(bldg_entries.iter()) {
		opt.arg = ArgOptionUI::BldgInd(entry.bldg_ind);
	}
	
	return owned_bldgs;
}

// for creating list to display of player's cities
pub fn owned_city_list<'bt,'ut,'rt,'dt>(bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, cur_player: SmSvType,
		cur_coord: Coord, w: &mut usize, label_txt_opt: &mut Option<String>,
		map_sz: MapSz, logs: &Vec<Log>, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	struct CityEntry {
		nm: String,
		population: String,
		dist: String,
		taxes: String,
		founded: String,
		producing: String,
		city_ind: usize
	}
	
	// get all cities owned by the player
	let mut city_entries = Vec::new();
	let mut max_nm_len = 0;
	let mut max_population_len = l.Population.len();
	let mut max_dist_len = l.Dist.len();
	let mut max_taxes_len = l.Taxes.len();
	let mut max_founded_len = l.Founded.len();
	let mut max_producing_len = l.Producing.len();
	
	for (bldg_ind, b) in bldgs.iter().enumerate().filter(|(_, b)| b.owner_id == cur_player) {
		if let BldgArgs::CityHall {nm, production, population, tax_rates} = &b.args {
			let producing = if let Some(entry) = production.last() {
				format!("{} ({}/{})", entry.production.nm[l.lang_ind], entry.progress, entry.production.production_req)
			}else{l.None.clone()};
			
			let get_founded = || {
				for log in logs.iter() {
					if let LogType::CityFounded {city_nm, ..} = &log.val {
						if *city_nm == *nm {
							return l.date_str(log.turn);
						}
					}
				}
				panicq!("could not find city founding log: {}", nm);
			};
			
			let entry = CityEntry {
				nm: nm.clone(),
				population: format!("{}", population),
				dist: direction_string(cur_coord, Coord::frm_ind(b.coord, map_sz), map_sz),
				taxes: format!("{}/{}/{}/{}%", tax_rates[0], tax_rates[1],
						tax_rates[2], tax_rates[3]),
				founded: get_founded(),
				producing,
				city_ind: bldg_ind
			};
			
			// keep track of max width of text
			if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
			if max_population_len < entry.population.len() {max_population_len = entry.population.len();}
			if max_dist_len < entry.dist.len() {max_dist_len = entry.dist.len();}
			if max_taxes_len < entry.taxes.len() {max_taxes_len = entry.taxes.len();}
			if max_founded_len < entry.founded.len() {max_founded_len = entry.founded.len();}
			if max_producing_len < entry.producing.len() {max_producing_len = entry.producing.len();}
			
			city_entries.push(entry);
		}
	}
	
	///// format txt only if player has any cities
	let mut label_txt = String::new(); // label before entries are printed
	if city_entries.len() > 0 {
		const SPACING: usize = 4;
		*w = max_nm_len + max_population_len + max_dist_len + max_taxes_len + max_founded_len + max_producing_len + SPACING*5 + 4;
		
		// format entries with gaps between columns
		for entry in city_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};};
			
			// population
			gap!(max_nm_len - entry.nm.len() + max_population_len - entry.population.len() + SPACING);
			entry.nm.push_str(&entry.population);
			
			// dist
			gap!(max_dist_len - entry.dist.len() + SPACING);
			entry.nm.push_str(&entry.dist);
			
			// taxes
			gap!(max_taxes_len - entry.taxes.len() + SPACING);
			entry.nm.push_str(&entry.taxes);
			
			// founded
			gap!(max_founded_len - entry.founded.len() + SPACING);
			entry.nm.push_str(&entry.founded);
			
			// producing
			gap!(max_producing_len - entry.producing.len() + SPACING);
			entry.nm.push_str(&entry.producing);
		}
		
		// label before entries are printed
		for _ in 0..(max_nm_len + SPACING) { label_txt.push(' ');}
		label_txt.push_str(&l.Population);
		
		for _ in 0..(max_dist_len + SPACING - l.Dist.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Dist);
		
		for _ in 0..(max_taxes_len + SPACING - l.Taxes.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Taxes);
		
		for _ in 0..(max_founded_len + SPACING - l.Founded.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Founded);
		
		for _ in 0..(max_producing_len + SPACING - l.Producing.len()) { label_txt.push(' ');}
		label_txt.push_str(&l.Producing);
		
		*label_txt_opt = Some(label_txt);
	}else{
		*w = 29;
		*label_txt_opt = None;
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut city_nms = Vec::with_capacity(city_entries.len());
	
	for entry in city_entries.iter() {	
		city_nms.push(entry.nm.as_str());
	}
	
	let mut owned_cities = OptionsUI {options: Vec::with_capacity(city_nms.len()), max_strlen: 0};
	
	register_shortcuts(&city_nms, &mut owned_cities);
	
	// associate unit_ind w/ each menu entry
	for (opt, entry) in owned_cities.options.iter_mut().zip(city_entries.iter()) {
		opt.arg = ArgOptionUI::CityInd(entry.city_ind);
	}
	
	return owned_cities;
}

// for creating list to display of player's cities
pub fn contacted_civilizations_list<'bt,'ut,'rt,'dt>(relations: &Relations, players: &Vec<Player>, cur_player: SmSvType, turn: usize) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let cur_player = cur_player as usize;
	
	// get all civs contacted by player
	let mut nms_string = Vec::with_capacity(players.len());
	let mut owner_ids = Vec::with_capacity(players.len());
	
	for (owner_id, player) in players.iter().enumerate() {
		if relations.discovered(cur_player, owner_id) && owner_id != cur_player && player.stats.alive {
			match player.ptype {
				PlayerType::AI {..} | PlayerType::Human {..} => {
					let nm_string = if let Some(_) = relations.peace_treaty_turns_remaining(cur_player as usize, owner_id, turn) {
						format!("{} (Peace treaty)", player.personalization.nm)
					}else if relations.at_war(cur_player as usize, owner_id) {
						format!("{} (WAR)", player.personalization.nm)
					}else {player.personalization.nm.clone()};
					
					nms_string.push(nm_string);
					owner_ids.push(owner_id);
				} PlayerType::Nobility {..} | PlayerType::Barbarian {..} => {}
			}
		}
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut contacted_civs = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut contacted_civs);
	
	// associate owner_id w/ each menu entry
	for (i, opt) in contacted_civs.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::OwnerInd(owner_ids[i]);
	}

	contacted_civs
}

// for creating list of all players
pub fn all_civilizations_list<'bt,'ut,'rt,'dt>(players: &Vec<Player>) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(players.len());
	let mut owner_ids = Vec::with_capacity(players.len());
	
	for (owner_id, player) in players.iter().enumerate() {
		match player.ptype {
			PlayerType::Barbarian {..} | PlayerType::Nobility {..} => {}
			PlayerType::AI {..} | PlayerType::Human {..} => {
				nms_string.push(player.personalization.nm.clone());
				owner_ids.push(owner_id);
			}
		}
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut civs = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut civs);
	
	// associate owner_id w/ each menu entry
	for (i, opt) in civs.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::OwnerInd(owner_ids[i]);
	}

	civs
}

pub fn undiscovered_tech_list<'bt,'ut,'rt,'dt>(pstats: &Stats, tech_templates: &Vec<TechTemplate>,
		l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(tech_templates.len());
	let mut tech_ids = Vec::with_capacity(tech_templates.len());
	
	for (tech_id, tech) in tech_templates.iter().enumerate() {
		if !pstats.tech_met(&Some(vec![tech_id])) {
			nms_string.push(tech.nm[l.lang_ind].clone());
			tech_ids.push(tech_id);
		}
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut tech_opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut tech_opts);
	
	// associate owner_id w/ each menu entry
	for (i, opt) in tech_opts.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::TechInd(tech_ids[i]);
	}
	
	tech_opts
}

pub fn all_resources_list<'bt,'ut,'rt,'dt>(resource_templates: &Vec<ResourceTemplate>,
		l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(resource_templates.len());
	
	for rt in resource_templates.iter() {
		nms_string.push(rt.nm[l.lang_ind].clone());
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut resource_opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut resource_opts);
	
	// associate owner_id w/ each menu entry
	for (i, opt) in resource_opts.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::ResourceInd(i);
	}
	
	resource_opts
}

pub fn game_difficulty_list<'bt,'ut,'rt,'dt>(game_difficulties: &GameDifficulties) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(game_difficulties.difficulties.len());
	
	for d in game_difficulties.difficulties.iter() {
		nms_string.push(d.nm.clone());
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut opts);
	
	opts
}

pub fn discovered_units_list<'bt,'ut,'rt,'dt>(pstats: &Stats, unit_templates: &'ut Vec<UnitTemplate<'rt>>,
		l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(unit_templates.len());
	let mut discov_templates = Vec::with_capacity(unit_templates.len());
	
	for ut in unit_templates.iter() {
		if pstats.tech_met(&ut.tech_req) {
			nms_string.push(ut.nm[l.lang_ind].clone());
			discov_templates.push(ut);
		}
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut discov_opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut discov_opts);
	
	// associate owner_id w/ each menu entry
	for (i, opt) in discov_opts.options.iter_mut().enumerate() {
		opt.arg = ArgOptionUI::UnitTemplate(Some(discov_templates[i]));
	}
	
	discov_opts
}

pub fn bldg_prod_list<'bt,'ut,'rt,'dt>(b: &Bldg<'bt,'ut,'rt,'dt>, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let (nms_string, unit_templates) = if let BldgArgs::CityHall {production, ..} |
					BldgArgs::GenericProducable {production} = &b.args {
		let mut nms_string = Vec::with_capacity(production.len());
		let mut unit_templates = Vec::with_capacity(production.len());
		for entry in production.iter() {
			nms_string.push(entry.production.nm[l.lang_ind].clone());
			unit_templates.push(entry.production);
		}
		(nms_string, unit_templates)
	}else{(Vec::new(), Vec::new())};
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	
	register_shortcuts(&nms, &mut opts);
	
	// associate owner_id w/ each menu entry
	for (opt, ut) in opts.options.iter_mut().zip(unit_templates.iter()) {
		opt.arg = ArgOptionUI::UnitTemplate(Some(ut));
	}
	
	opts
}

// `cur_coord` is in map coordinates
pub fn discovered_resources_list<'bt,'ut,'rt,'dt>(pstats: &Stats, cur_coord: Coord,
		resource_templates: &'rt Vec<ResourceTemplate>, map_sz: MapSz) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(N_RESOURCES_DISCOV_LOG * resource_templates.len());
	let mut resource_w_coords = Vec::with_capacity(N_RESOURCES_DISCOV_LOG * resource_templates.len());
	
	debug_assertq!(pstats.resources_discov_coords.len() == resource_templates.len());
	
	// discovered resources & their dist from the cursor
	for (res_disc_coords, rt) in pstats.resources_discov_coords.iter()
						.zip(resource_templates.iter()) {
		if !pstats.resource_discov(rt) {continue;}
		
		for res_disc_coord in res_disc_coords.iter() {
			let res_coord = Coord::frm_ind(*res_disc_coord, map_sz);
			nms_string.push(format!("{} ({})", rt.nm[0], direction_string(cur_coord, res_coord, map_sz)));
			resource_w_coords.push(ArgOptionUI::ResourceWCoord {
				rt,
				coord: *res_disc_coord
			});
		}
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut discov_opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	register_shortcuts(&nms, &mut discov_opts);
	
	// associate owner_id w/ each menu entry
	for (resource_w_coord, opt) in resource_w_coords.into_iter().zip(discov_opts.options.iter_mut()) {
		opt.arg = resource_w_coord;
	}
	
	discov_opts
}

pub fn doctrines_available_list<'bt,'ut,'rt,'dt>(pstats: &Stats, doctrine_templates: &'dt Vec<DoctrineTemplate>,
		l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let mut nms_string = Vec::with_capacity(doctrine_templates.len());
	
	let avail_doc = available_doctrines(pstats, doctrine_templates);
	for doc in avail_doc.iter() {
		nms_string.push(doc.nm[l.lang_ind].clone());
	}
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	register_shortcuts(&nms, &mut opts);
	
	// associate owner_id w/ each menu entry
	for (doc, opt) in avail_doc.iter().zip(opts.options.iter_mut()) {
		opt.arg = ArgOptionUI::DoctrineTemplate(Some(doc));
	}
	
	opts
}

// auto-explore options
pub fn explore_types_list<'bt,'ut,'rt,'dt>(l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	let nms_string = vec![l.Spiral_out.clone(), l.Random.clone()];
	
	// register_shortcuts takes [&str]s, so take references of all the strings
	let mut nms = Vec::with_capacity(nms_string.len());
	
	for nm_string in nms_string.iter() {
		nms.push(nm_string.as_str());
	}
	
	let mut opts = OptionsUI {options: Vec::with_capacity(nms.len()), max_strlen: 0};
	register_shortcuts(&nms, &mut opts);
	
	opts
}

// if owners supplied, print player colors
// returns (top-left, top-right) most point the window is printed at
pub fn print_list_window(mut mode: usize, top_txt: &str, mut options: OptionsUI,
		iface_settings: &IfaceSettings, disp_chars: &DispChars,
		w_opt: Option<usize>, label_txt: Option<String>, n_gap_lines: usize,
		owners_opt: Option<&Vec<Player>>, l: &Localization,
		buttons: &mut Buttons, d: &mut DispState) -> (Coord, Coord) {
	let n_orig_options = options.options.len();
	let mode_orig = mode;
	
	let mut w = if let Some(w_use) = w_opt {w_use as i32} else {29};
	let h = min(iface_settings.screen_sz.h, n_orig_options + 5);
	
	let n_rows_plot = min(n_orig_options, iface_settings.screen_sz.h - 6);
	
	let mut start_ind = 0;
	
	// add additional width if scrolling & crop options
	let show_scroll_bars = if n_rows_plot < n_orig_options {
		w += 1;
		
		options.options = if mode < n_rows_plot {
			options.options[..n_rows_plot].to_vec()
		}else{
			start_ind = mode;
			let cropped = options.options[(mode - (n_rows_plot - 1))..=mode].to_vec();
			mode = n_rows_plot - 1;
			cropped
		};
		
		true
	}else{false};
	
	let y = (iface_settings.screen_sz.h as i32 - h as i32)/2;
	let x = (iface_settings.screen_sz.w as i32 - w as i32)/2;
	
	let mut row: i32 = 0;
	let (mut cy, mut cx) = (0_i32,0_i32);
	
	// top line
	d.mv(row + y, x); row += 1;
	d.addch(disp_chars.ulcorner_char);
	for _ in 0..(w-2) {d.addch(disp_chars.hline_char);}
	d.addch(disp_chars.urcorner_char);
	
	macro_rules! nln {() => (d.mv(row + y, x); row += 1; 
				 d.addch(disp_chars.vline_char); d.addch(' ' as chtype););};
	macro_rules! eln {() => (
		d.getyx(stdscr(), &mut cy, &mut cx);
		for _ in cx..=(w+x-2) {d.addch(' ' as chtype);}
		d.addch(disp_chars.vline_char); );};
	
	nln!(); buttons.Esc_to_close.print(None, l, d); eln!();
	
	// txt line
	nln!(); d.addstr(top_txt); eln!();
	
	// blank line
	nln!(); eln!();
	
	// gap lines
	for _ in 0..n_gap_lines {nln!(); eln!();}
	
	// labels before entries are printed
	if let Some(txt) = label_txt {
		nln!(); addstr_c(&txt, CGRAY, d); eln!();
	}
	
	///////////////////////////// print list options
	if options.options.len() > 0 {
		//let w = if show_scroll_bars {w - 1} else {w};
		print_menu_vstack(&options, y + row, x, w as usize, mode, disp_chars, true, owners_opt, start_ind, &mut None, buttons, d);
	}else{
		nln!();
		d.addstr(&l.None); eln!();
		
		// bottom line
		d.mv(row + y, x);
		d.addch(disp_chars.llcorner_char);
		for _ in 0..(w-2) {d.addch(disp_chars.hline_char);}
		d.addch(disp_chars.lrcorner_char);
	}
	
	//////// print scroll bars
	if show_scroll_bars {
		let h = h as i32;
		let w = x as i32 + w as i32 - 2;
		let scroll_track_h = n_rows_plot;
		let frac_covered = n_rows_plot as f32 / n_orig_options as f32;
		let scroll_bar_h = ((scroll_track_h as f32) * frac_covered).round() as i32;
		debug_assertq!(frac_covered <= 1.);
		
		let frac_at_numer = if mode_orig < n_rows_plot as usize {
			0
		} else {
			mode_orig - n_rows_plot + 3 //first_ln + 2//+ n_rows_plot as usize
		};
		
		let frac_at = frac_at_numer as f32 / n_orig_options as f32;
		let scroll_bar_start = ((scroll_track_h as f32) * frac_at).round() as i32;
		
		// blank space after each option
		for offset in 0..options.options.len() as i32 {
			d.mv(offset + 2 + LOG_START_ROW, w-1);
			d.addstr("  ");
		}
		
		d.mv(LOG_START_ROW, w);
		d.attron(COLOR_PAIR(CLOGO));
		d.addch(disp_chars.hline_char);
		for row in 0..scroll_bar_h-1 {
			d.mv(row + 1 + scroll_bar_start + LOG_START_ROW, w);
			d.addch(disp_chars.vline_char);
			//d.addch('#' as chtype);
		}
		d.mv(h-LOG_STOP_ROW, w);
		d.addch(disp_chars.hline_char);
		d.attroff(COLOR_PAIR(CLOGO));
	}
	
	(Coord {y: y as isize, x: x as isize},
	 Coord {y: y as isize, x: (x + w) as isize})
}

