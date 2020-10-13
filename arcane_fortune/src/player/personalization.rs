use crate::saving::*;
use crate::doctrine::DoctrineTemplate;
use crate::resources::ResourceTemplate;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;

// formerly `Owner`
#[derive(Clone, PartialEq)]
pub struct Personalization {
	pub color: i32,
	pub nm: String, // of country
	
	pub gender_female: bool,
	pub ruler_nm: PersonName,
	
	pub doctrine_advisor_nm: PersonName,
	pub crime_advisor_nm: PersonName,
	pub pacifism_advisor_nm: PersonName,
	pub health_advisor_nm: PersonName,
	pub unemployment_advisor_nm: PersonName,
	
	pub city_nm_theme: usize, // index into nms.cities[city_nm_theme]
	pub motto: String,
	pub founded: usize
}

impl_saving!{Personalization {color, nm, gender_female,
	ruler_nm, doctrine_advisor_nm, crime_advisor_nm,
	pacifism_advisor_nm, health_advisor_nm, unemployment_advisor_nm,
	city_nm_theme, motto, founded}}

#[derive(Clone, PartialEq)]
pub struct PersonName {
	pub first: String,
	pub last: String
}

impl_saving!{PersonName {first, last}}

#[derive(Clone, PartialEq)]
pub struct Nms { // name list used to randomly name each city and unit (loaded from config files)
	pub cities: Vec<Vec<String>>, // cities[i][:] are all city names with theme i
	pub units: Vec<String>, // aka battalions
	pub brigades: Vec<String>,
	pub sectors: Vec<String>,
	pub noble_houses: Vec<String>,
	pub females: Vec<String>,
	pub males: Vec<String>,
}

impl_saving!{Nms {cities, units, brigades, sectors, noble_houses, females, males}}

