use crate::saving::*;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::config_load::{read_file, config_parse, find_key, find_req_key_parse};
use crate::map::{Stats, Owner};

#[derive(Clone, PartialEq)]
pub struct Bonuses {
	pub combat_factor: f32, // 1. = equality w/ human players
	pub production_factor: SmSvType,
	pub gold_per_day: f32,
}

impl_saving!{Bonuses {combat_factor, production_factor, gold_per_day}}

pub struct GameDifficulty {
	pub nm: String,
	pub ai_bonuses: Bonuses
}

pub struct GameDifficulties {
	pub difficulties: Vec<GameDifficulty>,
	pub default_ind: usize,
	pub longest_nm: usize
}

pub fn load_game_difficulties() -> GameDifficulties {
	let key_sets = config_parse(read_file("config/ai.txt"));
	let mut difficulty_ind_cur = 0;
	let mut difficulty_ind_sel = 0;
	let mut difficulties = Vec::with_capacity(20);
	let mut longest_nm = 0;
	
	for keys in key_sets.iter() {
		if let Some(nm) = find_key("difficulty_name", &keys) {
			if nm.len() > longest_nm {
				longest_nm = nm.len();
			}
			
			difficulties.push(GameDifficulty {
				nm,
				ai_bonuses: Bonuses {
					combat_factor: find_req_key_parse("combat_bonus_factor", &keys),
					production_factor: find_req_key_parse("production_factor", &keys),
					gold_per_day: find_req_key_parse("gold_bonus_per_day", &keys)
				}
			});
			
			if let Some(_) = find_key("default_option", &keys) {
				difficulty_ind_sel = difficulty_ind_cur;
			}
			difficulty_ind_cur += 1;
		}
	}
	
	GameDifficulties {
		difficulties,
		default_ind: difficulty_ind_sel,
		longest_nm
	}
}

impl GameDifficulties {
	pub fn cur_difficulty_ind(&self, stats: &Vec<Stats>, owners: &Vec<Owner>) -> usize {
		for (pstats, owner) in stats.iter().zip(owners.iter()) {
			if owner.player_type.is_human() {continue;}
			
			for (diff_ind, difficulty) in self.difficulties.iter().enumerate() {
				if difficulty.ai_bonuses == pstats.bonuses {
					return diff_ind;
				}
			}
		}
		0 // not found
	}
}

