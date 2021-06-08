use super::*;

// all the manors/cities owned by nobles in the player's empire
pub fn manors_list<'bt,'ut,'rt,'dt>(bldgs: &Vec<Bldg<'bt,'ut,'rt,'dt>>, cur_player: SmSvType, players: &Vec<Player>,
		gstate: &GameState, cur_coord: Coord, w: &mut usize, label_txt_opt: &mut Option<String>, map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {
	
	let houses = gstate.relations.noble_houses(cur_player as usize);
	
	struct CityEntry {
		nm: String,
		house: String,
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
	let mut max_house_len = l.House.len();
	let mut max_population_len = l.Population.len();
	let mut max_dist_len = l.Dist.len();
	let mut max_taxes_len = l.Taxes.len();
	let mut max_founded_len = l.Founded.len();
	let mut max_producing_len = l.Producing.len();
	
	for (bldg_ind, b) in bldgs.iter().enumerate().filter(|(_, b)| houses.contains(&(b.owner_id as usize))) {
		if let BldgArgs::PopulationCenter {nm, production, population, tax_rates} = &b.args {
			let producing = if let Some(entry) = production.last() {
				format!("{} ({}/{})", entry.production.nm[l.lang_ind], entry.progress, entry.production.production_req)
			}else{l.None.clone()};
			
			let get_founded = || {
				for log in gstate.logs.iter() {
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
				house: players[b.owner_id as usize].personalization.nm.clone(),
				population: format!("{}", population.iter().sum::<u32>()),
				dist: direction_string(cur_coord, Coord::frm_ind(b.coord, map_sz), map_sz),
				taxes: format!("{}/{}/{}/{}%", tax_rates[0], tax_rates[1],
						tax_rates[2], tax_rates[3]),
				founded: get_founded(),
				producing,
				city_ind: bldg_ind
			};
			
			// keep track of max width of text
			if max_nm_len < entry.nm.len() {max_nm_len = entry.nm.len();}
			if max_house_len < entry.house.len() {max_house_len = entry.house.len();}
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
		*w = max_nm_len + max_house_len + max_population_len + max_dist_len +
			max_taxes_len + max_founded_len + max_producing_len + SPACING*6 + 4;
		
		// format entries with gaps between columns
		for entry in city_entries.iter_mut() {
			macro_rules! gap{($len: expr) => {
				for _ in 0..$len {entry.nm.push(' ');}
			};}
			
			// house
			gap!(max_nm_len - entry.nm.len() + max_house_len - entry.house.len() + SPACING);
			entry.nm.push_str(&entry.house);
			
			// population
			gap!(max_population_len - entry.population.len() + SPACING);
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
		label_txt.push_str(&l.House);
		
		for _ in 0..(max_population_len + SPACING - l.Population.len()) { label_txt.push(' ');}
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
	
	let mut owned_cities = OptionsUI::new(&city_nms);
	
	// associate unit_ind w/ each menu entry
	for (opt, entry) in owned_cities.options.iter_mut().zip(city_entries.iter()) {
		opt.arg = ArgOptionUI::CityInd(entry.city_ind);
	}
	
	return owned_cities;
}

/*pub fn noble_brigades_list<'bt,'ut,'rt,'dt>(units: &Vec<Unit<'bt,'ut,'rt,'dt>>, cur_player: SmSvType, cur_coord: Coord,
		players: &Vec<Player>, relations: &Relations,
		w: &mut usize, label_txt_opt: &mut Option<String>,
		map_sz: MapSz, l: &Localization) -> OptionsUI<'bt,'ut,'rt,'dt> {

	let mut unit_inds_use = Vec::with_capacity(units.len());
	
	let houses = relations.noble_houses(cur_player as usize);
	
	// get all units owned by player, gather costs, names
	for (unit_ind, _u) in units.iter().enumerate().filter(|(_, u)| houses.contains(&(u.owner_id as usize))) {
		unit_inds_use.push(unit_ind);
	}
	
	unit_list_frm_vec(&unit_inds_use, units, cur_coord, players, w, label_txt_opt, map_sz, l)
}*/

