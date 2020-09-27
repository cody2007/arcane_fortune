use crate::buildings::*;
use crate::units::*;
use crate::saving::*;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;

#[derive(Clone, PartialEq)]
pub struct TechTemplate {
	pub id: SmSvType,
	pub nm: Vec<String>,
	
	pub tech_req: Option<Vec<SmSvType>>, // required tech index
	pub research_req: SmSvType
}

impl_saving!{ TechTemplate {id, nm, tech_req, research_req} }

