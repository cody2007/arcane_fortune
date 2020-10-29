use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::config_load::{config_parse, read_file, find_opt_key_parse, find_opt_key_resources};
use crate::saving::*;
use crate::renderer::endwin;

#[derive(Clone, PartialEq)]
pub struct AIConfig<'rt> {
	pub city_creation_resource_bonus: Vec<SmSvType>, // [resource_ind] -- drives city creation towards these resources (indexed by resource id)
	pub strategic_resources: Vec<&'rt ResourceTemplate> // resources to actively seek out in city creation location
}

impl_saving!{AIConfig<'rt> {city_creation_resource_bonus, strategic_resources}}

pub fn init_ai_config<'rt>(resource_templates: &'rt Vec<ResourceTemplate>) -> AIConfig<'rt> {
	const AI_CONFIG_FILE: &str = "config/ai.txt";
	let key_sets = config_parse(read_file(AI_CONFIG_FILE));
	
	let city_creation_resource_bonus = {
		let mut city_creation_resource_bonus = Vec::with_capacity(resource_templates.len());
		
		'resource_loop: for rt in resource_templates.iter() {
			let key_nm = format!("{}_bonus", rt.nm[0]);
			for key_set in key_sets.iter() {
				if let Some(bonus) = find_opt_key_parse(&key_nm, key_set) {
					city_creation_resource_bonus.push(bonus);
					//printlnq!("{} {}", key_nm, bonus);
					continue 'resource_loop;
				}
			}
			
			// no resource found
			city_creation_resource_bonus.push(0);
		}
		assertq!(city_creation_resource_bonus.len() == resource_templates.len());
		city_creation_resource_bonus
	};
	
	let get_strategic_resources = || {
		const STRATEGIC_RESOURCES_KEY: &str = "strategic_resources";
		for key_set in key_sets.iter() {
			if let Some(strategic_resources) = find_opt_key_resources(STRATEGIC_RESOURCES_KEY, key_set, resource_templates) {
				return strategic_resources;
			}
		}
		panicq!("could not find entry `{}` in: {}", STRATEGIC_RESOURCES_KEY, AI_CONFIG_FILE);
	};
	
	AIConfig {
		city_creation_resource_bonus,
		strategic_resources: get_strategic_resources()
	}
}

