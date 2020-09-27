/*
To implement saving and loading [ .sv() and .ld() ]:

1. Reqs. Default implementation (if to be used in Vecs, Boxes, or Options:
				         if no default() possible [as w/ structs containing refs], 
					 custom implementation of SvO2 providing the default
					 with some owner[0], bldg_template[0], ...

2. Implementation of trait Sv (and for all nested types):

        - for structs: use impl_saving!()
		                          must implement default if used in containers(!)
	
	- for enums containing embeded {values}: custom sv() and ld() for the Sv trait implementation
	
	- for simple enums: use impl_frm_cast!(EnumNameToImplement, SmSvType);

3. See below for exception of templates (which require Vec<UnitTemplate<'rt>> and Vec<&UnitTemplate<'rt>>, for example,
   perform separate operations
   
*/

use std::mem::*;
use std::hash::{BuildHasherDefault};
use std::time::Instant;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::map::*;
use crate::units::*;
use crate::buildings::*;
use crate::doctrine::*;
use crate::disp::*;
use crate::gcore::hashing::{HashedMapEx, HashedMapZoneEx, HashedFogVars, HashedCoords, HashStruct64};
use crate::gcore::{LogType, WarStatus};
use crate::zones::*;
use crate::ai::{AttackFrontState, Neighbors};
use crate::resources::ResourceTemplate;
use crate::disp_lib::endwin;

pub mod defaults; pub use defaults::*;
pub mod save_game; pub use save_game::*;
pub mod save_debug; pub use save_debug::*;
pub mod snappy;

#[derive(PartialEq, Clone)]
pub enum GameState {New, NewOptions, Load(String), TitleScreen}

//////////////////////////////////////////////// trait defs
macro_rules! fn_headers{() => (
	fn sv(&self, res: &mut Vec<u8>); // saves value into res
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>);
	// ^ loads values from res
);}

pub trait Sv<'f,'bt,'ut,'rt,'dt> {fn_headers!();}
pub trait SvO<'f,'bt,'ut,'rt,'dt,T> {fn_headers!();} // Option<T>, Box<T>
pub trait SvO2<'f,'bt,'ut,'rt,'dt> {fn_headers!();} // Option<&Owner>, Option<&BldgTemplate>. 
// ^must be manually init because we need an existing ref, but can't get this in default() w/o param inputs

pub trait SvOT<'f,'bt,'ut,'rt,'dt> { // for templates, see description below
	fn sv_template(&self, res: &mut Vec<u8>);
	fn ld_template(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>);
}

//////////////////////////////////////////////// macros

// template specific fns: have to be named differently because rust doesn't
// know how to differentiate when we want to access the implementation of, for example,
// UnitTemplate and &UnitTemplate (and "&UnitTemplate" are used in fields in other structures we need to save)
macro_rules! impl_saving_template{($type:ty{ $($entry:ident),*} ) => (
	impl <'f,'bt,'ut,'rt,'dt> SvOT <'f,'bt,'ut,'rt,'dt> for $type{
		fn sv_template(&self, res: &mut Vec<u8>){ $(self.$entry.sv(res);)* }
		fn ld_template(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){ 
			$(self.$entry.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)* }});}

macro_rules! impl_saving{($type:ty{ $($entry:ident),*} ) => (
	impl <'f,'bt,'ut,'rt,'dt> Sv<'f,'bt,'ut,'rt,'dt> for $type{
		fn sv(&self, res: &mut Vec<u8>){ $(self.$entry.sv(res);)* }
		fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){ 
		$(self.$entry.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)* }});}

macro_rules! impl_ident{($type:ty) => (
	impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for $type {
		fn sv(&self, res: &mut Vec<u8>){ res.extend_from_slice(&self.to_le_bytes());}
		
		fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &Vec<BldgTemplate>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
			const SZ: usize = size_of::<$type>();
			let mut cropped: [u8; SZ] = Default::default();
			cropped.copy_from_slice(&res[*o..*o + SZ]); *o += SZ;
			
			*self = Self::from_le_bytes(cropped);  }})}

macro_rules! impl_as_cast{($type:ty, $to:ty) => (
	impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for $type {
		fn sv(&self, res: &mut Vec<u8>){ res.extend_from_slice(&(*self as $to).to_le_bytes());}

		fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &Vec<BldgTemplate>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
			const SZ: usize = size_of::<$to>();
			let mut cropped: [u8; SZ] = Default::default();
			cropped.copy_from_slice(&res[*o..*o + SZ]); *o += SZ;
			
			*self = <$to>::from_le_bytes(cropped) as $type;  }})}

macro_rules! impl_frm_cast{($type:ty, $to: ty) => (
	impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for $type {
		fn sv(&self, res: &mut Vec<u8>){ res.extend_from_slice(&(*self as $to).to_le_bytes());}

		fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &Vec<BldgTemplate>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
			const SZ: usize = size_of::<$to>();
			let mut cropped: [u8; SZ] = Default::default();
			cropped.copy_from_slice(&res[*o..*o + SZ]); *o += SZ;
			
			*self = Self::from(<$to>::from_le_bytes(cropped));  }})}

/////////////////////// saving/loading primitives

// should have a Self::from declaration
impl_ident!(u8);
impl_ident!(i8);
impl_ident!(i16);
impl_ident!(i32);
impl_ident!(u32);
impl_ident!(u64);
impl_as_cast!(char, u8);
impl_as_cast!(usize, u32);
impl_as_cast!(isize, i32);

// enums
impl_frm_cast!(MapType, SmSvType);
impl_frm_cast!(ZoneType, SmSvType);
impl_frm_cast!(StructureType, SmSvType);
impl_frm_cast!(Underlay, SmSvType);
impl_frm_cast!(MovementType, SmSvType);
impl_frm_cast!(Neighbors, SmSvType);
impl_frm_cast!(ViewMvMode, SmSvType);
impl_frm_cast!(AutoTurn, SmSvType);
impl_frm_cast!(ZoneOverlayMap, SmSvType);
impl_frm_cast!(ExploreType, SmSvType);
impl_frm_cast!(SectorUnitEnterAction, SmSvType);
impl_frm_cast!(SectorCreationType, SmSvType);
impl_frm_cast!(PacifismMilitarism, SmSvType);

////////////////////////// saving/loading non-castable (to integer) primitives

impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for f32 {
	fn sv(&self, res: &mut Vec<u8>){
		res.extend_from_slice(&self.to_bits().to_le_bytes());
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &Vec<BldgTemplate>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		const SZ: usize = size_of::<f32>();
		let mut a: [u8; SZ] = Default::default();
		a.copy_from_slice(&res[*o..*o + SZ]);
		*self = <f32>::from_bits(<u32>::from_le_bytes(a));
		*o += SZ; }}

impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for bool {
	fn sv(&self, res: &mut Vec<u8>){
		res.push(if *self {1} else {0}); 
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &Vec<BldgTemplate>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		*self = res[*o] == 1;
		*o += 1;
	}}

impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for String {
	fn sv(&self, res: &mut Vec<u8>){
		let s = self.clone();
		s.into_bytes().sv(res);
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut bytes: Vec<u8> = Vec::new();
		bytes.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		*self = String::from_utf8(bytes).unwrap();
	}}

////////////// refs -- save as index; load as index

////// &BldgTemplate (save as index)
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for &'bt BldgTemplate<'ut,'rt,'dt> {
	fn sv(&self, res: &mut Vec<u8>){
		res.extend_from_slice(&(self.id as SmSvType).to_le_bytes());
	}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		const SZ: usize = size_of::<SmSvType>();
		let mut a: [u8; SZ] = Default::default();
		a.copy_from_slice(&res[*o..*o + SZ]);
		*self = &bldg_templates[<SmSvType>::from_le_bytes(a) as usize];
		*o += SZ; }}

//// &UnitTemplate (save as index)
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for &'ut UnitTemplate<'rt> {
	fn sv(&self, res: &mut Vec<u8>){
		res.extend_from_slice(&(self.id as SmSvType).to_le_bytes());
	}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		const SZ: usize = size_of::<SmSvType>();
		let mut a: [u8; SZ] = Default::default();
		a.copy_from_slice(&res[*o..*o + SZ]);
		*self = &unit_templates[<SmSvType>::from_le_bytes(a) as usize];
		*o += SZ;
}}

//// &ResourceTemplate (save as index)
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for &'rt ResourceTemplate {
	fn sv(&self, res: &mut Vec<u8>){
		res.extend_from_slice(&(self.id as SmSvType).to_le_bytes());
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, _: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		const SZ: usize = size_of::<SmSvType>();
		let mut a: [u8; SZ] = Default::default();
		a.copy_from_slice(&res[*o..*o + SZ]);
		*self = &resource_templates[<SmSvType>::from_le_bytes(a) as usize];
		*o += SZ;
}}

///// &DoctrineTemplate (save as index)
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for &'dt DoctrineTemplate {
	fn sv(&self, res: &mut Vec<u8>){
		res.extend_from_slice(&(self.id as SmSvType).to_le_bytes());
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, _: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, _: &'ut Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		const SZ: usize = size_of::<SmSvType>();
		let mut a: [u8; SZ] = Default::default();
		a.copy_from_slice(&res[*o..*o + SZ]);
		*self = &doctrine_templates[<SmSvType>::from_le_bytes(a) as usize];
		*o += SZ;
}}

//////////////////////////////////////// Enums w/ embeded values
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for HappinessCategory {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			HappinessCategory::Doctrine => {res.push(0);}
			HappinessCategory::PacifismOrMilitarism(pacifism_militarism) => {
				res.push(1);
				pacifism_militarism.sv(res);}
			HappinessCategory::Health => {res.push(2);}
			HappinessCategory::Unemployment => {res.push(3);}
			HappinessCategory::Crime => {res.push(4);}
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => HappinessCategory::Doctrine,
			1 => {
				let mut d = HappinessCategory::PacifismOrMilitarism(PacifismMilitarism::N);
				if let HappinessCategory::PacifismOrMilitarism(ref mut pacifism_militarism) = d {
					ld_vals!(pacifism_militarism);
				}else{panicq!("invalid value");}
				d}
			2 => HappinessCategory::Health,
			3 => HappinessCategory::Unemployment,
			4 => HappinessCategory::Crime,
			_ => {panicq!("invalid HappinessCategory.id in file")}
};}}

//////////// SectorIdleAction
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for SectorIdleAction {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			SectorIdleAction::Sentry => {
				res.push(0);
			} SectorIdleAction::Patrol {dist_monitor, perim_coord_ind, perim_coord_turn_computed} => {
				res.push(1);
				dist_monitor.sv(res);
				perim_coord_ind.sv(res);
				perim_coord_turn_computed.sv(res);
			}
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => SectorIdleAction::Sentry,
			1 => {
				let mut d = SectorIdleAction::Patrol {dist_monitor: 0, perim_coord_ind: 0, perim_coord_turn_computed: 0};
				if let SectorIdleAction::Patrol {ref mut dist_monitor, ref mut perim_coord_ind, ref mut perim_coord_turn_computed} = d {
					ld_vals!(dist_monitor, perim_coord_ind, perim_coord_turn_computed);
				}else{panicq!("invalid value");}
				d
			} _ => {panicq!("invalid owner.id in file")}
};}}

//////////// FireTile
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for FireTile {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			FireTile::Smoke => {
				res.push(0);
			} FireTile::Fire {color} => {
				res.push(1);
				color.sv(res);
			} FireTile::None => {res.push(2);}
	}}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};

		*self = match res[*o-1] {
			0 => FireTile::Smoke,
			1 => {
				let mut d = FireTile::Fire {color: 0};
				if let FireTile::Fire {ref mut color} = d {
					ld_vals!(color);
				}else{panicq!("invalid value");}
				d
			} 2 => FireTile::None,
			_ => {panicq!("invalid owner.id in file")}
};}}

//////////// AttackFrontState
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for AttackFrontState {
	fn sv(&self, res: &mut Vec<u8>) {
		match self {
			AttackFrontState::Recruitment {unreachable_city_coords} => {
				res.push(0);
				unreachable_city_coords.sv(res);
			} AttackFrontState::AssembleToLocation {assemble_location, 
					target_city_coord, unreachable_city_coords} => {
				res.push(1);
				assemble_location.sv(res);
				target_city_coord.sv(res);
				unreachable_city_coords.sv(res);
			} AttackFrontState::WallAttack {target_city_coord, wall_coord, attacks_initiated} => {
				res.push(2);
				target_city_coord.sv(res);
				wall_coord.sv(res);
				attacks_initiated.sv(res);
			} AttackFrontState::CityAttack {target_city_coord, attacks_initiated} => {
				res.push(3);
				target_city_coord.sv(res);
				attacks_initiated.sv(res);
			}
		}
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => { 
				let mut d = AttackFrontState::Recruitment {unreachable_city_coords: Vec::new()};
				if let AttackFrontState::Recruitment {ref mut unreachable_city_coords} = d {
					ld_vals!(unreachable_city_coords);
				}
				d
			} 1 => {
				let mut d = AttackFrontState::AssembleToLocation {
					assemble_location: 0,
					target_city_coord: 0,
					unreachable_city_coords: Vec::new()
				};
				if let AttackFrontState::AssembleToLocation{ref mut assemble_location,
						ref mut target_city_coord, ref mut unreachable_city_coords} = d {
					ld_vals!(assemble_location);
					ld_vals!(target_city_coord);
					ld_vals!(unreachable_city_coords);
				}
				d
			} 2 => {
				let mut d = AttackFrontState::WallAttack {target_city_coord: 0, wall_coord: 0, attacks_initiated: false};
				if let AttackFrontState::WallAttack{ref mut target_city_coord, ref mut wall_coord, ref mut attacks_initiated} = d {
					ld_vals!(target_city_coord);
					ld_vals!(wall_coord);
					ld_vals!(attacks_initiated);
				}
				d
			} 3 => {
				let mut d = AttackFrontState::CityAttack {target_city_coord: 0, attacks_initiated: false};
				if let AttackFrontState::CityAttack {ref mut target_city_coord, ref mut attacks_initiated} = d {
					ld_vals!(target_city_coord);
					ld_vals!(attacks_initiated);
				}
				d
			} _ => {panicq!("unknown attack front state")}
		}
	}
}

///////////// WarStatus
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for WarStatus {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			WarStatus::Peace {turn_started} => {
				res.push(0);
				turn_started.sv(res);
			}
			WarStatus::War {turn_started} => {
				res.push(1);
				turn_started.sv(res);
			}
		}
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => {
				let mut d = WarStatus::Peace {turn_started: 0};
				if let WarStatus::Peace {ref mut turn_started} = d {
					ld_vals!(turn_started);
				}
				d
			}
			1 => {
				let mut d = WarStatus::War {turn_started: 0};
				if let WarStatus::War {ref mut turn_started} = d {
					ld_vals!(turn_started);
				}
				d
			}
			_ => {panicq!("unknown warstatus type")}
		}
	}
}

///////// PlayerType
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for PlayerType {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			PlayerType::Human => {
				res.push(0);
			} PlayerType::AI(personality) => {
				res.push(1);
				personality.sv(res);
			} PlayerType::Barbarian => {
				res.push(2);
			}
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => { PlayerType::Human
			} 1 => {
				let mut d = PlayerType::AI(AIPersonality::default());
				if let PlayerType::AI(ref mut personality) = d {
					ld_vals!(personality);
				}
				d
			} 2 => { PlayerType::Barbarian
			} _ => {panicq!("invalid player type in file")}
};}}

///////// BldgType
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for BldgType {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			BldgType::Taxable(zone_type)  => {
				res.push(0);
				zone_type.sv(res);
			} BldgType::Gov(bonuses) => {
				res.push(1);
				bonuses.sv(res);
			}
	}}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => {
				let mut d = BldgType::Taxable(ZoneType::N);
				if let BldgType::Taxable(ref mut zt) = d {
					ld_vals!(zt);
				}
				d
			} 1 => {
				let mut d = BldgType::Gov(Vec::new());
				if let BldgType::Gov(ref mut bonuses) = d {
					ld_vals!(bonuses);
				}
				d
			}
			_ => {panicq!("invalid owner.id in file")}
};}}

///////// TechProg
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for TechProg {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			TechProg::Prog(prog)  => {
				res.push(0);
				prog.sv(res);
			} TechProg::Finished => {res.push(1);}
	}}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};

		*self = match res[*o-1] {
			0 => {
				let mut d = TechProg::Prog(0);
				if let TechProg::Prog(ref mut prog) = d {
					ld_vals!(prog);
				}
				d
			} 1 => TechProg::Finished,
			_ => {panicq!("invalid owner.id in file")}
};}}

////////// LogType
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for LogType {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			LogType::CivCollapsed {owner_id} => {res.push(0);
				owner_id.sv(res);
			} LogType::CivDestroyed {owner_attackee_id, owner_attacker_id} => {res.push(1);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::UnitDestroyed {unit_attackee_nm, unit_attacker_nm, 
					unit_attackee_type_nm, unit_attacker_type_nm,
					owner_attackee_id, owner_attacker_id} => {res.push(2);
				unit_attackee_nm.sv(res);
				unit_attacker_nm.sv(res);
				unit_attackee_type_nm.sv(res);
				unit_attacker_type_nm.sv(res);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::CityCaptured {city_attackee_nm, owner_attackee_id, owner_attacker_id} => {res.push(3);
				city_attackee_nm.sv(res);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::UnitDisbanded {owner_id, unit_nm, unit_type_nm} => {res.push(4);
				owner_id.sv(res);
				unit_nm.sv(res);
				unit_type_nm.sv(res);
			} LogType:: BldgDisbanded {owner_id, bldg_nm} => {res.push(5);
				owner_id.sv(res);
				bldg_nm.sv(res);
			} LogType::CityDisbanded {owner_id, city_nm} => {res.push(6);
				owner_id.sv(res);
				city_nm.sv(res);
			} LogType::CityDestroyed {city_attackee_nm, owner_attackee_id, owner_attacker_id} => {res.push(7);
				city_attackee_nm.sv(res);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::CityFounded {owner_id, city_nm} => {res.push(8);
				owner_id.sv(res);
				city_nm.sv(res);
			} LogType::CivDiscov {discover_id, discovee_id} => {res.push(9);
				discover_id.sv(res);
				discovee_id.sv(res);
			} LogType::UnitAttacked {unit_attackee_nm, unit_attacker_nm,
					unit_attackee_type_nm, unit_attacker_type_nm,
					owner_attackee_id, owner_attacker_id} => {res.push(10);
				unit_attackee_nm.sv(res);
				unit_attacker_nm.sv(res);
				unit_attackee_type_nm.sv(res);
				unit_attacker_type_nm.sv(res);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::StructureAttacked {structure_coord, unit_attacker_nm,
					unit_attacker_type_nm, structure_type,
					owner_attackee_id, owner_attacker_id} => {res.push(11);
				structure_coord.sv(res);
				unit_attacker_nm.sv(res);
				unit_attacker_type_nm.sv(res);
				structure_type.sv(res);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::WarDeclaration {owner_attackee_id, owner_attacker_id} => {res.push(12);
				owner_attackee_id.sv(res);
				owner_attacker_id.sv(res);
			} LogType::PeaceDeclaration {owner1_id, owner2_id} => {res.push(13);
				owner1_id.sv(res);
				owner2_id.sv(res);
			} LogType::ICBMDetonation {owner_id} => {res.push(14);
				owner_id.sv(res);
			} LogType::PrevailingDoctrineChanged {owner_id, doctrine_frm_id, doctrine_to_id} => {res.push(15);
				owner_id.sv(res);
				doctrine_frm_id.sv(res);
				doctrine_to_id.sv(res);
			} LogType::Rioting {owner_id, city_nm} => {res.push(16);
				owner_id.sv(res);
				city_nm.sv(res);
			} LogType::RiotersAttacked {owner_id} => {res.push(17);
				owner_id.sv(res);
			} LogType::CitizenDemand {owner_id, reason} => {res.push(18);
				owner_id.sv(res);
				reason.sv(res);
			} LogType::Debug {txt, owner_id} => {res.push(19);
				txt.sv(res);
				owner_id.sv(res);
			}}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => {
				let mut d = LogType::CivCollapsed {owner_id: 0};
				if let LogType::CivCollapsed {ref mut owner_id} = d {
					ld_vals!(owner_id);
				}
				d
			} 1 => {
				let mut d = LogType::CivDestroyed {
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::CivDestroyed {ref mut owner_attackee_id, ref mut owner_attacker_id} = d {
					ld_vals!(owner_attackee_id, owner_attacker_id);
				}
				d
			} 2 => {
				let mut d = LogType::UnitDestroyed {
					unit_attackee_nm: String::new(),
					unit_attacker_nm: String::new(),
					unit_attackee_type_nm: String::new(),
					unit_attacker_type_nm: String::new(),
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::UnitDestroyed {ref mut unit_attackee_nm, ref mut unit_attacker_nm,
						ref mut unit_attackee_type_nm, ref mut unit_attacker_type_nm,
						ref mut owner_attackee_id, ref mut owner_attacker_id} = d {
					ld_vals!(unit_attackee_nm, unit_attacker_nm, 
							unit_attackee_type_nm, unit_attacker_type_nm,
							owner_attackee_id, owner_attacker_id);
				}
				d
			} 3 => {
				let mut d = LogType::CityCaptured {
					city_attackee_nm: String::new(),
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::CityCaptured {ref mut city_attackee_nm, ref mut owner_attackee_id, ref mut owner_attacker_id} = d {
					ld_vals!(city_attackee_nm, owner_attackee_id, owner_attacker_id);
				}
				d
			} 4 => {
				let mut d = LogType::UnitDisbanded {
					owner_id: 0,
					unit_nm: String::new(),
					unit_type_nm: String::new()
				};
				
				if let LogType::UnitDisbanded {ref mut owner_id, ref mut unit_nm, ref mut unit_type_nm} = d {
					ld_vals!(owner_id, unit_nm, unit_type_nm);
				}
				d
			} 5 => {
				let mut d = LogType::BldgDisbanded {
					owner_id: 0,
					bldg_nm: String::new(),
				};
				
				if let LogType::BldgDisbanded {ref mut owner_id, ref mut bldg_nm} = d {
					ld_vals!(owner_id, bldg_nm);
				}
				d
			} 6 => {
				let mut d = LogType::CityDisbanded {
					owner_id: 0,
					city_nm: String::new(),
				};
				
				if let LogType::CityDisbanded {ref mut owner_id, ref mut city_nm} = d {
					ld_vals!(owner_id, city_nm);
				}
				d
			} 7 => {
				let mut d = LogType::CityDestroyed {
					city_attackee_nm: String::new(),
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::CityDestroyed {ref mut city_attackee_nm, ref mut owner_attackee_id, ref mut owner_attacker_id} = d {
					ld_vals!(city_attackee_nm, owner_attackee_id, owner_attacker_id);
				}
				d
			} 8 => {
				let mut d = LogType::CityFounded {
					owner_id: 0,
					city_nm: String::new(),
				};
				
				if let LogType::CityFounded {ref mut owner_id, ref mut city_nm} = d {
					ld_vals!(owner_id, city_nm);
				}
				d
			} 9 => {
				let mut d = LogType::CivDiscov {discover_id: 0, discovee_id: 0};
				
				if let LogType::CivDiscov {ref mut discover_id, ref mut discovee_id} = d {
					ld_vals!(discover_id, discovee_id);
				}
				d
			} 10 => {
				let mut d = LogType::UnitAttacked {
					unit_attackee_nm: String::new(),
					unit_attacker_nm: String::new(),
					unit_attackee_type_nm: String::new(),
					unit_attacker_type_nm: String::new(),
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::UnitAttacked {ref mut unit_attackee_nm, ref mut unit_attacker_nm,
						ref mut unit_attackee_type_nm, ref mut unit_attacker_type_nm,
						ref mut owner_attackee_id, ref mut owner_attacker_id} = d {
					ld_vals!(unit_attackee_nm, unit_attacker_nm,
							unit_attackee_type_nm, unit_attacker_type_nm,
							owner_attackee_id, owner_attacker_id);
				}
				d
			} 11 => {
				let mut d = LogType::StructureAttacked {
					structure_coord: 0,
					unit_attacker_nm: String::new(),
					unit_attacker_type_nm: String::new(),
					structure_type: StructureType::N,
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::StructureAttacked {ref mut structure_coord, ref mut unit_attacker_nm,
						ref mut unit_attacker_type_nm, ref mut structure_type,
						ref mut owner_attackee_id, ref mut owner_attacker_id, ..} = d {
					ld_vals!(structure_coord, unit_attacker_nm,
							unit_attacker_type_nm, structure_type,
							owner_attackee_id, owner_attacker_id);
				}else{panicq!("invalid value");}
				d
			} 12 => {
				let mut d = LogType::WarDeclaration {
					owner_attackee_id: 0,
					owner_attacker_id: 0
				};
				
				if let LogType::WarDeclaration {ref mut owner_attackee_id, ref mut owner_attacker_id} = d {
					ld_vals!(owner_attackee_id, owner_attacker_id);
				}
				d
			} 13 => {
				let mut d = LogType::PeaceDeclaration {
					owner1_id: 0,
					owner2_id: 0
				};
				
				if let LogType::PeaceDeclaration {ref mut owner1_id, ref mut owner2_id} = d {
					ld_vals!(owner1_id, owner2_id);
				}
				d
			} 14 => {
				let mut d = LogType::ICBMDetonation {
					owner_id: 0,
				};
				
				if let LogType::ICBMDetonation {ref mut owner_id} = d {
					ld_vals!(owner_id);
				}
				d
			} 15 => {
				let mut d = LogType::PrevailingDoctrineChanged {
					owner_id: 0,
					doctrine_frm_id: 0,
					doctrine_to_id: 0
				};
				
				if let LogType::PrevailingDoctrineChanged {ref mut owner_id, ref mut doctrine_frm_id, ref mut doctrine_to_id} = d {
					ld_vals!(owner_id, doctrine_frm_id, doctrine_to_id);
				}else{panicq!("invalid value");}
				d
			} 16 => {
				let mut d = LogType::Rioting {owner_id: 0, city_nm: String::new()};
				if let LogType::Rioting {ref mut owner_id, ref mut city_nm} = d {
					ld_vals!(owner_id, city_nm);
				}else{panicq!("invalid log type");}
				d
			} 17 => {
				let mut d = LogType::RiotersAttacked {owner_id: 0};
				if let LogType::RiotersAttacked {ref mut owner_id} = d {
					ld_vals!(owner_id);
				}else{panicq!("invalid log type");}
				d
			} 18 => {
				let mut d = LogType::CitizenDemand {owner_id: 0, reason: HappinessCategory::Doctrine};
				if let LogType::CitizenDemand {ref mut owner_id, ref mut reason} = d {
					ld_vals!(owner_id, reason);
				}else{panicq!("invalid log type");}
				d
			} 19 => {
				let mut d = LogType::Debug {
					txt: String::new(),
					owner_id: None
				};
				
				if let LogType::Debug {ref mut txt, ref mut owner_id} = d {
					ld_vals!(txt, owner_id);
				}
				d
			} _ => {panicq!("invalid logType.id in file")}
};}}

///////// Dist
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for Dist {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			Dist::NotInit => {res.push(0);
			} Dist::NotPossible {turn_computed} => {res.push(1);
				turn_computed.sv(res);
			} Dist::Is {dist, bldg_ind} => {res.push(2);
				dist.sv(res);
				bldg_ind.sv(res);
			} Dist::ForceRecompute {dist, bldg_ind} => {res.push(3);
				dist.sv(res);
				bldg_ind.sv(res);
	}}}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &Vec<BldgTemplate>, unit_templates: &Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};

		*self = match res[*o-1] {
			0 => Dist::NotInit,
			1 => {
				let mut d = Dist::NotPossible {turn_computed: 0};
				if let Dist::NotPossible {ref mut turn_computed} = d {
					ld_vals!(turn_computed);
				}
				d
			} 2 => {
				let mut d = Dist::Is {dist: 0, bldg_ind: 0};
				if let Dist::Is {ref mut dist, ref mut bldg_ind} = d {
					ld_vals!(dist, bldg_ind);
				}
				d
			} 3 => {
				let mut d = Dist::ForceRecompute {dist: 0, bldg_ind: 0};
				if let Dist::ForceRecompute {ref mut dist, ref mut bldg_ind} = d {
					ld_vals!(dist, bldg_ind);
				}
				d
			} _ => {panicq!("invalid owner.id in file")}
};}}

///////// UIMode
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for UIMode<'bt,'ut,'rt,'dt> {
	fn sv(&self, _: &mut Vec<u8>){
		//debug_assertq!(UIMode::None == *self, "Expected no UI mode in write to file.");
		// no longer true because we don't want to clear UIMode in the current game when we're auto saving
	}
	
	fn ld(&mut self, _: &Vec<u8>, _: &mut usize, _: &Vec<BldgTemplate>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		*self = UIMode::None;
}}

////// BldgArgs
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for BldgArgs<'ut,'rt> {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			BldgArgs::CityHall {tax_rates, production, population, nm} => {
				res.push(0);
				tax_rates.sv(res);
				production.sv(res);
				population.sv(res);
				nm.sv(res);
			} BldgArgs::GenericProducable {production} => {
				res.push(1);
				production.sv(res);
			} BldgArgs::None => {
				res.push(2);
	}}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};

		*self = match res[*o-1] {
			0 => {
				let mut args = BldgArgs::CityHall {tax_rates: Vec::new().into_boxed_slice(), production: Vec::new(), 
					nm: String::new(), population: 0};
				if let BldgArgs::CityHall {ref mut tax_rates, ref mut production, ref mut population, ref mut nm} = args {
					ld_vals!(tax_rates, production, population, nm);
				}
				args
			} 1 => {
				let mut args = BldgArgs::GenericProducable {production: Vec::new()};
				if let BldgArgs::GenericProducable {ref mut production}  = args {
					ld_vals!(production);
				}
				args
			} 2 => {BldgArgs::None
			} _ => {panicq!("invalid BldArgs id")
}}}}

/////// ActionType
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for ActionType<'bt,'ut,'rt,'dt> {
	fn sv(&self, res: &mut Vec<u8>){
		match self {
			ActionType::Mv => {res.push(0);
			} ActionType::MvIgnoreWalls => {res.push(1);
			} ActionType::MvIgnoreOwnWalls => {res.push(2);
			} ActionType::CivilianMv => {res.push(3);
			} ActionType::AutoExplore {start_coord, explore_type} => {
				res.push(4);
				start_coord.sv(res);
				explore_type.sv(res);
			} ActionType::WorkerBuildStructure {structure_type, turns_expended} => {
				res.push(5);
				structure_type.sv(res);
				turns_expended.sv(res);
			} ActionType::WorkerRepairWall {wall_coord, turns_expended} => {
				res.push(6);
				wall_coord.sv(res);
				turns_expended.sv(res);
			} ActionType::WorkerBuildBldg {valid_placement, template, bldg_coord, doctrine_dedication} => {
				res.push(7);
				valid_placement.sv(res);
				template.sv(res);
				bldg_coord.sv(res);
				doctrine_dedication.sv(res);
			} ActionType::Attack {attack_coord, attackee, ignore_own_walls} => {
				res.push(8);
				attack_coord.sv(res);
				attackee.sv(res);
				ignore_own_walls.sv(res);
			} ActionType::Fortify {turn} => {
				res.push(9);
				turn.sv(res);
			} ActionType::WorkerZone {valid_placement, zone_type, start_coord, end_coord} => {
				res.push(10);
				valid_placement.sv(res);
				zone_type.sv(res);
				start_coord.sv(res);
				end_coord.sv(res);
			} ActionType::GroupMv {start_coord, end_coord} => {
				res.push(11);
				start_coord.sv(res);
				end_coord.sv(res);
			} ActionType::BrigadeCreation {nm, start_coord, end_coord} => {
				res.push(12);
				nm.sv(res);
				start_coord.sv(res);
				end_coord.sv(res);
			} ActionType::SectorCreation {nm, creation_type, start_coord, end_coord} => {
				res.push(13);
				nm.sv(res);
				creation_type.sv(res);
				start_coord.sv(res);
				end_coord.sv(res);
			} ActionType::WorkerZoneCoords {zone_type} => {
				res.push(14);
				zone_type.sv(res);
			} ActionType::UIWorkerAutomateCity => {res.push(15);
			} ActionType::BurnBuilding {coord} => {
				res.push(16);
				coord.sv(res);
			} ActionType::WorkerContinueBuildBldg => {res.push(17);
			} ActionType::MvWithCursor => {res.push(18);
			} ActionType::SectorAutomation {unit_enter_action, idle_action, sector_nm} => {
				res.push(19);
				unit_enter_action.sv(res);
				idle_action.sv(res);
				sector_nm.sv(res);
			} ActionType::WorkerRmZonesAndBldgs {start_coord, end_coord} => {
				res.push(20);
				start_coord.sv(res);
				end_coord.sv(res);
	}}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		
		macro_rules! ld_vals{($($val:ident),*) => ($($val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);)*);};
		
		*self = match res[*o-1] {
			0 => {ActionType::Mv
			} 1 => {ActionType::MvIgnoreWalls
			} 2 => {ActionType::MvIgnoreOwnWalls
			} 3 => {ActionType::CivilianMv
			} 4 => {
				let mut at = ActionType::AutoExplore {start_coord: 0, explore_type: ExploreType::Random};
				if let ActionType::AutoExplore {ref mut start_coord, ref mut explore_type} = at {
					ld_vals!(start_coord, explore_type);
				}else{panicq!("invalid action type");}
				at
			} 5 => {
				let mut at = ActionType::WorkerBuildStructure {structure_type: StructureType::N, turns_expended: 0};
				if let ActionType::WorkerBuildStructure {ref mut structure_type, ref mut turns_expended} = at {
					ld_vals!(structure_type);
					ld_vals!(turns_expended);
				}
				at
			} 6 => {
				let mut at = ActionType::WorkerRepairWall {wall_coord: None, turns_expended: 0};
				if let ActionType::WorkerRepairWall {ref mut wall_coord, ref mut turns_expended} = at {
					ld_vals!(wall_coord);
					ld_vals!(turns_expended);
				}
				at
			} 7 => {
				let mut at = ActionType::WorkerBuildBldg{ valid_placement: false, template: &bldg_templates[0], bldg_coord: None, doctrine_dedication: None};
				if let ActionType::WorkerBuildBldg{ref mut valid_placement, ref mut template, ref mut bldg_coord, ref mut doctrine_dedication} = at {
					ld_vals!(valid_placement, template, bldg_coord, doctrine_dedication);
				}
				at
			} 8 => {
				let mut at = ActionType::Attack {attack_coord: None, attackee: None, ignore_own_walls: false};
				if let ActionType::Attack{ref mut attack_coord, ref mut attackee, ref mut ignore_own_walls} = at {
					ld_vals!(attack_coord, attackee, ignore_own_walls);
				}
				at
			} 9 => {
				let mut at = ActionType::Fortify {turn: 0};
				if let ActionType::Fortify{ref mut turn} = at {
					ld_vals!(turn);
				}
				at
			} 10 => {
				let mut at = ActionType::WorkerZone {valid_placement: false, zone_type: ZoneType::N, start_coord: None, end_coord: None};
				if let ActionType::WorkerZone{ref mut valid_placement, ref mut zone_type, ref mut start_coord, ref mut end_coord} = at {
					ld_vals!(valid_placement, zone_type, start_coord, end_coord);
				}
				at
			} 11 => {
				let mut at = ActionType::GroupMv {start_coord: None, end_coord: None};
				if let ActionType::GroupMv{ref mut start_coord, ref mut end_coord} = at {
					ld_vals!(start_coord, end_coord);
				}else{panicq!("load error");}
				at
			} 12 => {
				let mut at = ActionType::BrigadeCreation {nm: String::new(), start_coord: None, end_coord: None};
				if let ActionType::BrigadeCreation {ref mut nm,
						ref mut start_coord, ref mut end_coord} = at {
					ld_vals!(nm, start_coord, end_coord);
				}else{panicq!("load error");}
				at
			} 13 => {
				let mut at = ActionType::SectorCreation {nm: String::new(), creation_type: SectorCreationType::N, start_coord: None, end_coord: None};
				if let ActionType::SectorCreation {ref mut nm, ref mut creation_type,
						ref mut start_coord, ref mut end_coord} = at {
					ld_vals!(nm, creation_type, start_coord, end_coord);
				}else{panicq!("load error");}
				at
			} 14 => {
				let mut at = ActionType::WorkerZoneCoords {zone_type: ZoneType::N};
				if let ActionType::WorkerZoneCoords {ref mut zone_type} = at {
					ld_vals!(zone_type);
				}else{panicq!("load error");}
				at
			} 15 => {ActionType::UIWorkerAutomateCity
			} 16 => {
				let mut at = ActionType::BurnBuilding {coord: 0};
				if let ActionType::BurnBuilding {ref mut coord} = at {
					ld_vals!(coord);
				}else{panicq!("load error");}
				at
			} 17 => {ActionType::WorkerContinueBuildBldg
			} 18 => {ActionType::MvWithCursor
			} 19 => {
				let mut at = ActionType::SectorAutomation {
					unit_enter_action: SectorUnitEnterAction::Defense,
					idle_action: SectorIdleAction::Sentry,
					sector_nm: String::new()
				};
				if let ActionType::SectorAutomation {ref mut unit_enter_action, ref mut idle_action, ref mut sector_nm} = at {
					ld_vals!(unit_enter_action, idle_action, sector_nm);
				}else{panicq!("load error");}
				at
			} 20 => {
				let mut at = ActionType::WorkerRmZonesAndBldgs {
					start_coord: None,
					end_coord: None
				};
				if let ActionType::WorkerRmZonesAndBldgs {ref mut start_coord, ref mut end_coord} = at {
					ld_vals!(start_coord, end_coord);
				}else{panicq!("load error");}
				at
			} _ => {panicq!("unknown ActionType id {}", res[*o-1]) }};}}

////////////////////////////////// misc

//// Instant
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for Instant {
	fn sv(&self, _res: &mut Vec<u8>){}

	fn ld(&mut self, _res: &Vec<u8>, _o: &mut usize, _: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, _: &Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		*self = Instant::now();
}}


//////////////////////////////////// saving/loading Option<T>

macro_rules! impl_option_sv{() => (
	fn sv(&self, res: &mut Vec<u8>){
		(!self.is_none()).sv(res);
		if let Some(v) = self {
			(*v).sv(res);
}});}

macro_rules! impl_option_sv_ld{() => (
	impl_option_sv!();
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut val = false;
		val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		if val {
			*self = Some(Default::default());
			if let Some(ref mut val) = self {
				(*val).ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
		}else{
			*self = None;
}});}

// \/ ---  allow generic type T which implements Sv, has a default value, and, if embedded in a Vec or Box, also implements clone
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Option<T> { impl_option_sv_ld!(); }
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Option<Vec<T>> { impl_option_sv_ld!(); }
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Option<Box<T>> { impl_option_sv_ld!(); }

impl <'f,'bt,'ut,'rt,'dt>  SvO2  <'f,'bt,'ut,'rt,'dt> for Option<Vec<Unit<'bt,'ut,'rt,'dt>>> { impl_option_sv_ld!(); }

// ^ uses the Default::default() val then calls load method on it


// \/ ---------
// uses owners[0] as the initial value, then calls the load
// method on it (difficult to have a default() implementation because it req.s
// an existant Owner reference lying around without giving default() a parameter
// same for bldg_template. SvO2 is needed because SvO can't be overloaded on T

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Option<&'bt BldgTemplate<'ut,'rt,'dt>> { impl_option_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		if res[*o-1] == 1 {
			*self = Some(&bldg_templates[0]);
			if let Some(ref mut val) = self {
				val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
		}else{
			*self = None;
}}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Option<&'ut UnitTemplate<'rt>> { impl_option_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		if res[*o-1] == 1 {
			*self = Some(&unit_templates[0]);
			if let Some(ref mut val) = self {
				val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
		}else{
			*self = None;
}}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Option<&'rt ResourceTemplate> { impl_option_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		if res[*o-1] == 1 {
			*self = Some(&resource_templates[0]);
			if let Some(ref mut val) = self {
				val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
		}else{
			*self = None;
}}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Option<&'dt DoctrineTemplate> { impl_option_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		if res[*o-1] == 1 {
			*self = Some(&doctrine_templates[0]);
			if let Some(ref mut val) = self {
				val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
		}else{
			*self = None;
}}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for AddActionTo<'f,'bt,'ut,'rt,'dt> {
	fn sv(&self, _: &mut Vec<u8>){}
	fn ld(&mut self, _: &Vec<u8>, _: &mut usize, _: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, _: &'ut Vec<UnitTemplate<'rt>>, _: &'rt Vec<ResourceTemplate>, _: &'dt Vec<DoctrineTemplate>){
		*self = AddActionTo::None;
}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Option<Vec<&'ut UnitTemplate<'rt>>> {
	fn sv(&self, res: &mut Vec<u8>){
		(!self.is_none()).sv(res);
		if let Some(v) = self {
			v.len().sv(res); // save length
			for d in v.iter() {
				d.sv(res);
	}}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*o += 1;
		if res[*o-1] == 1 {
			let mut sz: usize = 0;
			sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			
			let mut vals = vec!{&unit_templates[0]; sz};
			for v in vals.iter_mut() {
				v.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
			*self = Some(vals);
		}else{
			*self = None;
}}}

///////////////////////// saving/loading Boxes and Vecs

macro_rules! impl_iter_sv{() => (
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res); // save length
		
		for d in self.iter() { // save vals
			d.sv(res);
}});}

// box (convert vec to box)
macro_rules! impl_box_ld{() => (
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{Default::default(); sz}.into_boxed_slice();
		
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}});}

// vec
macro_rules! impl_vec_ld{() => (
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{Default::default(); sz};
		
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}});}

// vecDeque
macro_rules! impl_vecdeque_ld{() => (
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = VecDeque::with_capacity(sz);
		
		for _ in 0..sz {
			let mut d = ActionMeta::default();
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			self.push_back(d);
		}
});}

macro_rules! impl_box{() => (impl_iter_sv!(); impl_box_ld!(););}
macro_rules! impl_vec{() => (impl_iter_sv!(); impl_vec_ld!(););}

impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Box<[T]> {impl_box!();}
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Vec<T> {impl_vec!();}
//impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for VecDeque<T> {impl_iter_sv!(); impl_vecdeque_ld!();} // gives error about not being able to infer type
impl <'f,'bt,'ut,'rt,'dt>  Sv  <'f,'bt,'ut,'rt,'dt> for VecDeque<ActionMeta<'bt,'ut,'rt,'dt>> {impl_iter_sv!(); impl_vecdeque_ld!();}


impl <'f,'bt,'ut,'rt,'dt>  Sv  <'f,'bt,'ut,'rt,'dt> for Vec<&'rt ResourceTemplate> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res); // save length
		
		for d in self.iter() { // save vals
			d.sv(res);
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{&resource_templates[0]; sz};
		
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt>  Sv  <'f,'bt,'ut,'rt,'dt> for Vec<ResourceTemplate> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res); // save length
		
		for d in self.iter() { // save vals
			d.sv_template(res);
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{Default::default(); sz};
		
		for d in self.iter_mut() {
			d.ld_template(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt>  Sv  <'f,'bt,'ut,'rt,'dt> for Vec<UnitTemplate<'rt>> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res); // save length
		
		for d in self.iter() { // save vals
			d.sv_template(res);
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{Default::default(); sz};
		
		for d in self.iter_mut() {
			d.ld_template(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt>  Sv  <'f,'bt,'ut,'rt,'dt> for Vec<BldgTemplate<'ut,'rt,'dt>> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res); // save length
		
		for d in self.iter() { // save vals
			d.sv_template(res);
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{Default::default(); sz};
		
		for d in self.iter_mut() {
			d.ld_template(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt>  Sv  <'f,'bt,'ut,'rt,'dt> for Vec<DoctrineTemplate> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res); // save length
		
		for d in self.iter() { // save vals
			d.sv_template(res);
	}}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = vec!{Default::default(); sz};
		
		for d in self.iter_mut() {
			d.ld_template(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Box<[Option<T>]> {impl_box!();}
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Vec<Option<T>> {impl_vec!();}
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Box<[Box<[T]>]> {impl_box!();}
impl <'f,'bt,'ut,'rt,'dt, T: Sv<'f,'bt,'ut,'rt,'dt> + Default + Clone>  SvO  <'f,'bt,'ut,'rt,'dt, T> for Vec<Vec<T>> {impl_vec!();}
// ^ Box<[Option<T>]> implemented because generic recursion is difficult/impossible(?)
// therefore, implementations must manually be implemented for each supported recursion depth

////////////////////////////// values that can't have a default() [they req. references]
// Vec<Bldg>
impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Vec<Bldg<'bt,'ut,'rt,'dt>> {impl_iter_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let bldg_null = Bldg::default(0, 0, 0., None, &doctrine_templates[0], &bldg_templates[0], BldgArgs::None);
		*self = vec!{bldg_null; sz};
		
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Vec<Unit<'bt,'ut,'rt,'dt>> {impl_iter_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let unit_null = Unit::default(unit_templates);
		
		*self = vec!{unit_null; sz};
		
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Vec<Stats<'bt,'ut,'rt,'dt>> {impl_iter_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let spirituality_null = Stats::default(doctrine_templates);
		
		*self = vec!{spirituality_null; sz};
		
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

// ProductionEntry
impl <'f,'bt,'ut,'rt,'dt>  SvO2<'f,'bt,'ut,'rt,'dt> for Vec<ProductionEntry<'ut,'rt>> {impl_iter_sv!();
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		let null = ProductionEntry {production: &unit_templates[0], progress: 0};
		*self = vec![null; sz];
		for d in self.iter_mut() {
			d.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
}}}

///////////////////////// HashedCoords
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for HashedCoords {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res);
		for land_discov in self.iter() {
			land_discov.sv(res);
		}
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let s: BuildHasherDefault<HashStruct64> = Default::default();
		*self = HashSet::with_capacity_and_hasher(sz, s);
		for _ in 0..sz {
			let mut coord: u64 = 0;
			coord.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			self.insert(coord);
		}
	}
}

///////////////////////// MapData
impl <'f,'bt,'ut,'rt,'dt>  Sv<'f,'bt,'ut,'rt,'dt> for MapData<'rt> {
	fn sv(&self, res: &mut Vec<u8>){
		self.zoom_out[ZOOM_IND_ROOT].sv(res); // aka map_root in ld()
		self.map_szs[ZOOM_IND_ROOT].h.sv(res);
		self.map_szs[ZOOM_IND_ROOT].w.sv(res);
		let zoom_in_depth = self.map_szs.len() - N_EXPLICITLY_STORED_ZOOM_LVLS;
		zoom_in_depth.sv(res);
		self.max_zoom_in_buffer_sz.sv(res);
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut map_root: Vec<Map> = Vec::new();
		map_root.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let mut root_map_sz = MapSz {h: 0, w: 0, sz: 0};
		root_map_sz.h.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		root_map_sz.w.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let mut zoom_in_depth: usize = 0;
		zoom_in_depth.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		let mut max_zoom_in_buffer_sz: usize = 0;
		max_zoom_in_buffer_sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = MapData::default(map_root, root_map_sz.h, root_map_sz.w, zoom_in_depth, max_zoom_in_buffer_sz, resource_templates);
	}
}

//////////////////////// Vec<HashedMapEx>
/*impl <'f,'bt,'ut,'rt,'dt> SvO2<'f,'bt,'ut,'rt,'dt> for Vec<HashedMapEx<'bt,'ut,'rt,'dt>> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res);
		for ex in self.iter() {
			ex.len().sv(res); // save length
			
			for (key, val) in ex.iter() { // save vals
				key.sv(res);
				val.sv(res);
			}
		}
	}

	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		*self = Vec::new();
		let mut sz: usize = 0;
		sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
*/		
		/*for ex in self.iter_mut() {
			let s: BuildHasherDefault<HashStruct64> = Default::default();
			let mut ex = HashMap::with_capacity_and_hasher(sz, s);

			for i in 0..sz {
				let mut key;
				key.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
				let mut val;
				val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			}
		}*/
/*	}
}*/

//////// MapEx
fn sv_exs(exs: &Vec<HashedMapEx>, res: &mut Vec<u8>){
	exs.len().sv(res);
	for exz in exs.iter() {
		exz.len().sv(res); // save length
		
		for (key, val) in exz.iter() { // save vals
			key.sv(res);
			val.sv(res);
		}
	}
}

fn ld_exs<'bt,'ut,'rt,'dt>(exs: &mut Vec<HashedMapEx<'bt,'ut,'rt,'dt>>, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
	let mut n_zooms: usize = 0;
	n_zooms.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
	
	*exs = Vec::with_capacity(n_zooms);
	
	for _ in 0..n_zooms {
		let s: BuildHasherDefault<HashStruct64> = Default::default();
		let mut n_hash_entries: usize = 0;
		n_hash_entries.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		let mut exz = HashMap::with_capacity_and_hasher(n_hash_entries, s);
		
		for _ in 0..n_hash_entries {
			let mut key: u64 = 0;
			key.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			let mut val: MapEx = MapEx::default();
			val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			exz.insert(key, val);
		}
		exs.push(exz);
	}
}

////////// ZoneEx
fn sv_zone_exs(zone_exs_owners: &Vec<HashedMapZoneEx>, res: &mut Vec<u8>){
	zone_exs_owners.len().sv(res);
	for zone_exs in zone_exs_owners.iter() {
		zone_exs.len().sv(res); // save length
		
		for (key, val) in zone_exs.iter() { // save vals
			key.sv(res);
			val.sv(res);
		}
	}
}

fn ld_zone_exs<'bt,'ut,'rt,'dt>(zone_exs_owners: &mut Vec<HashedMapZoneEx>, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
	let mut n_owners: usize = 0;
	n_owners.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
	
	*zone_exs_owners = Vec::with_capacity(n_owners);
	
	for _ in 0..n_owners {
		let s: BuildHasherDefault<HashStruct64> = Default::default();
		let mut n_hash_entries: usize = 0;
		n_hash_entries.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		let mut zone_exs = HashMap::with_capacity_and_hasher(n_hash_entries, s);
		
		for _ in 0..n_hash_entries {
			let mut key: u64 = 0;
			key.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			let mut val: ZoneEx = ZoneEx::default();
			val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			zone_exs.insert(key, val);
		}
		
		zone_exs_owners.push(zone_exs);
	}
}

///////////// HashedFogVars
impl <'f,'bt,'ut,'rt,'dt> SvO2<'f,'bt,'ut,'rt,'dt> for Vec<HashedFogVars<'bt,'ut,'rt,'dt>> {
	fn sv(&self, res: &mut Vec<u8>){
		self.len().sv(res);
		for fog in self.iter() {
			fog.len().sv(res); // save length
			
			for (key, val) in fog.iter() { // save vals
				key.sv(res);
				val.sv(res);
			}
		}
	}
	
	fn ld(&mut self, res: &Vec<u8>, o: &mut usize, bldg_templates: &'bt Vec<BldgTemplate<'ut,'rt,'dt>>, unit_templates: &'ut Vec<UnitTemplate<'rt>>, resource_templates: &'rt Vec<ResourceTemplate>, doctrine_templates: &'dt Vec<DoctrineTemplate>){
		let mut n_zooms: usize = 0;
		n_zooms.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
		
		*self = Vec::with_capacity(n_zooms);
			
		for _ in 0..n_zooms {
			let mut sz: usize = 0;
			sz.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
			
			let s: BuildHasherDefault<HashStruct64> = Default::default();
			let mut fog_z = HashMap::with_capacity_and_hasher(sz, s);
			
			for _ in 0..sz {
				let mut key: u64 = 0;
				key.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
				
				let mut val = FogVars::default();
				val.ld(res, o, bldg_templates, unit_templates, resource_templates, doctrine_templates);
				
				debug_assertq!(!fog_z.contains_key(&key));
				fog_z.insert(key, val);
			}
			
			self.push(fog_z);
		}
	}
}


