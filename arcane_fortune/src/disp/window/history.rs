use crate::gcore::{Log, LogType, Relations};

pub fn world_history_events(player_id: usize, relations: &Relations,
		logs: &Vec<Log>) -> Vec<Log> {
	let mut events = Vec::with_capacity(logs.len());
	for log in logs.iter()
			.filter(|log| log.visible(player_id, relations)) {
		match log.val {
			LogType::CivCollapsed {..} |
			LogType::CivDestroyed {..} |
			LogType::CityCaptured {..} |
			LogType::CityDisbanded {..} |
			LogType::CityDestroyed {..} |
			LogType::CityFounded {..} |
			LogType::CivDiscov {..} |
			LogType::WarDeclaration {..} |
			LogType::PeaceDeclaration {..} |
			LogType::PrevailingDoctrineChanged {..} |
			LogType::Rioting {..} |
			LogType::NobleHouseJoinedEmpire {..} |
			LogType::RiotersAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::ICBMDetonation {..} => {
				events.push(log.clone());
			}
			
			LogType::Debug {..} | 
			LogType::UnitDestroyed {..} |
			LogType::UnitAttacked {..} |
			LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} |
			LogType::StructureAttacked {..} => {}
		}
	}
	events
}

pub fn battle_history_events(player_id: usize, relations: &Relations,
		logs: &Vec<Log>) -> Vec<Log> {
	let mut events = Vec::with_capacity(logs.len());
	for log in logs.iter().filter(|log| log.visible(player_id, relations)) {
		match log.val {
			LogType::CivCollapsed {..} |
			LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} |
			LogType::CityDisbanded {..} |
			LogType::CityDestroyed {..} |
			LogType::CityFounded {..} |
			LogType::CivDiscov {..} |
			LogType::WarDeclaration {..} |
			LogType::PeaceDeclaration {..} |
			LogType::PrevailingDoctrineChanged {..} |
			LogType::Rioting {..} |
			LogType::RiotersAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::NobleHouseJoinedEmpire {..} |
			LogType::Debug {..} => {}
			
			LogType::CivDestroyed {..} |
			LogType::CityCaptured {..} |
			LogType::ICBMDetonation {..} |
			LogType::UnitDestroyed {..} |
			LogType::UnitAttacked {..} |
			LogType::StructureAttacked {..} => {
				events.push(log.clone());
			}
		}
	}
	events
}

pub fn economic_history_events(player_id: usize, relations: &Relations,
		logs: &Vec<Log>) -> Vec<Log> {
	let mut events = Vec::with_capacity(logs.len());
	for log in logs.iter()
			.filter(|log| log.visible(player_id, relations)) {
		match log.val {
			LogType::CivCollapsed {..} |
			LogType::CityDisbanded {..} |
			LogType::CityDestroyed {..} |
			LogType::CityFounded {..} |
			LogType::CivDiscov {..} |
			LogType::WarDeclaration {..} |
			LogType::Rioting {..} |
			LogType::RiotersAttacked {..} |
			LogType::PeaceDeclaration {..} |
			LogType::Debug {..} |
			LogType::CivDestroyed {..} |
			LogType::CityCaptured {..} |
			LogType::NobleHouseJoinedEmpire {..} |
			LogType::ICBMDetonation {..} |
			LogType::PrevailingDoctrineChanged {..} |
			LogType::UnitDestroyed {..} |
			LogType::UnitAttacked {..} |
			LogType::CitizenDemand {..} |
			LogType::StructureAttacked {..} => {}
			
			LogType::UnitDisbanded {..} |
			LogType::BldgDisbanded {..} => {
				events.push(log.clone());
			}
		}
	}
	events
}

