use super::*;
use crate::doctrine::disp::DoctrineWindowState;
use crate::tech::disp::TechWindowState;
use crate::map::ZoneType;

pub enum UIMode<'bt,'ut,'rt,'dt> {
	None,
	SetTaxes(ZoneType),
	TextTab {mode: usize, loc: TextTabLoc}, // for screen readers -- text cursor is moved to printed text
	Menu {mode: Option<usize>, sub_mode: Option<usize>, prev_auto_turn: AutoTurn,
		sel_loc: (i32, i32) // sel_loc is the row and col on the screen where the active menu item is -- used to set the text cursor for screen readers
	},
	ProdListWindow(ProdListWindowState), 
	// ^ bldg unit production: when bldg under cursor
	//   worker bldg production: when unit under cursor
	CurrentBldgProd(CurrentBldgProdState),
	
	SelectBldgDoctrine(SelectBldgDoctrineState<'bt,'ut,'rt,'dt>),
	SelectExploreType(SelectExploreTypeState),
	SaveAsWindow(SaveAsWindowState),
	OpenWindow(OpenWindowState),
	SaveAutoFreqWindow(SaveAutoFreqWindowState),
	GetTextWindow(GetTextWindowState),
	TechWindow(TechWindowState), // the tech tree -- set with iface_settings.create_tech_window()
	DoctrineWindow(DoctrineWindowState), // the spirituality tree -- set with iface_settings.create_spirituality_window()
	PlotWindow(PlotWindowState),
	UnitsWindow(UnitsWindowState), // selection of unit
	NobleUnitsWindow(NobleUnitsWindowState), // show units owned by noble house (if house_nm is Some string)
	
	BrigadesWindow(BrigadesWindowState), // selection of brigade -- to view it or add a unit to it
	BrigadeBuildList(BrigadeBuildListState),
	
	SectorsWindow(SectorsWindowState), // selection of sector to jump to or delete
	CitiesWindow(CitiesWindowState), // selection of city hall
	ManorsWindow(ManorsWindowState), // selection of noble house manors
	BldgsWindow(BldgsWindowState), // selection of either military or improvement buildings
	ContactEmbassyWindow(ContactEmbassyWindowState),
	ContactNobilityWindow(ContactNobilityState),
	Trade(TradeState),
	CivilizationIntelWindow(CivilizationIntelWindowState), // selection of civilization
	SwitchToPlayerWindow(SwitchToPlayerWindowState),
	SetDifficultyWindow(SetDifficultyWindowState),
	TechDiscoveredWindow(TechDiscoveredWindowState), // iface_settings.create_tech_discovered_window(); shows what new units/bldgs/resources avail after discovery of tech
	DiscoverTechWindow(DiscoverTechWindowState), // omniprescent option, not part of normal gameplay
	ObtainResourceWindow(ObtainResourceWindowState),
	PlaceUnitWindow(PlaceUnitWindowState),
	ResourcesAvailableWindow(ResourcesAvailableWindowState), // show resources available (utilized)
	ResourcesDiscoveredWindow(ResourcesDiscoveredWindowState), // every resource the player has ever come across
	HistoryWindow(HistoryWindowState),
	WarStatusWindow(WarStatusWindowState), // heatmap of who is at war with who
	EncyclopediaWindow(EncyclopediaWindowState),
	GoToCoordinateWindow(GoToCoordinateWindowState),
	MvWithCursorNoActionsRemainAlert(MvWithCursorNoActionsRemainAlertState),
	PrevailingDoctrineChangedWindow(PrevailingDoctrineChangedWindowState), // when the residents adopt a new doctrine
	RiotingAlert(RiotingAlertState), // when rioting breaks out
	GenericAlert(GenericAlertState),
	CitizenDemandAlert(CitizenDemandAlertState),
	CreateSectorAutomation(CreateSectorAutomationState),
	CivicAdvisorsWindow(CivicAdvisorsWindowState),
	PublicPollingWindow(PublicPollingWindowState),
	InitialGameWindow(InitialGameWindowState),
	EndGameWindow(EndGameWindowState),
	AboutWindow(AboutWindowState),
	UnmovedUnitsNotification(UnmovedUnitsNotificationState),
	NoblePedigree(NoblePedigreeState),
	AcceptNobilityIntoEmpire(AcceptNobilityIntoEmpireState), // nobility asks to join the empire
	ForeignUnitInSectorAlert(ForeignUnitInSectorAlertState)
}

pub enum TextTabLoc {
	BottomStats,
	RightSide
}

impl UIMode<'_,'_,'_,'_> {
	pub fn is_none(&self) -> bool {
		match self {
			UIMode::TextTab {..} | UIMode::None => {true}
			_ => {false}
		}
	}
}

/*impl fmt::Display for UIMode<'_,'_,'_,'_> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			UIMode::None => "None",
			UIMode::SetTaxes(_) => "SetTaxes",
			UIMode::Menu {..} => "Menu",
			UIMode::GenericAlert {..} => "GenericAlert",
			UIMode::PublicPollingWindow => "PublicPollingWindow",
			UIMode::ProdListWindow {..} => "ProdListWindow",
			UIMode::CurrentBldgProd {..} => "CurrentBldgProd",
			UIMode::CitizenDemandAlert {..} => "CitizenDemandAlert",
			UIMode::SelectBldgDoctrine {..} => "SelectBldgDoctrine",
			UIMode::SelectExploreType {..} => "SelectExploreType",
			UIMode::SaveAsWindow {..} => "SaveAsWindow",
			UIMode::OpenWindow {..} => "OpenWindow",
			UIMode::CreateSectorAutomation {..} => "CreateSectorAutomation",
			UIMode::SaveAutoFreqWindow {..} => "SaveAutoFreqWindow",
			UIMode::GetTextWindow {..} => "GetTextWindow",
			UIMode::TechWindow {..} => "TechWindow",
			UIMode::DoctrineWindow {..} => "DoctrineWindow",
			UIMode::PlotWindow {..} => "PlotWindow",
			UIMode::UnitsWindow {..} => "UnitsWindow",
			UIMode::BrigadesWindow {..} => "BrigadesWindow",
			UIMode::BrigadeBuildList {..} => "BrigadeBuildList",
			UIMode::SectorsWindow {..} => "SectorsWindow",
			UIMode::MilitaryBldgsWindow {..} => "MilitaryBldgsWindow",
			UIMode::ImprovementBldgsWindow {..} => "ImprovementBldgsWindow",
			UIMode::CitiesWindow {..} => "CitiesWindow",
			UIMode::ContactEmbassyWindow {..} => "ContactEmbassyWindow",
			UIMode::CivilizationIntelWindow {..} => "CivilizationIntelWindow",
			UIMode::SwitchToPlayerWindow {..} => "SwitchToPlayerWindow",
			UIMode::SetDifficultyWindow {..} => "SetDifficulty",
			UIMode::TechDiscoveredWindow {..} => "TechDiscoveredWindow",
			UIMode::DiscoverTechWindow {..} => "DiscoverTechWindow",
			UIMode::ObtainResourceWindow {..} => "ObtainResourceWindow",
			UIMode::PlaceUnitWindow {..} => "PlaceUnitWindow",
			UIMode::ResourcesAvailableWindow => "ResourcesAvailableWindow",
			UIMode::ResourcesDiscoveredWindow {..} => "ResourcesDiscoveredWindow",
			UIMode::WorldHistoryWindow {..} => "WorldHistoryWindow",
			UIMode::BattleHistoryWindow {..} => "BattleHistoryWindow",
			UIMode::EconomicHistoryWindow {..} => "EconomicHistoryWindow",
			UIMode::WarStatusWindow => "WarStatusWindow",
			UIMode::EncyclopediaWindow {..} => "EncyclopediaWindow",
			UIMode::GoToCoordinateWindow {..} => "GoToCoordinateWindow",
			UIMode::InitialGameWindow => "InitialGameWindow",
			UIMode::PrevailingDoctrineChangedWindow => "PrevailingDoctrineChangedWindow",
			UIMode::RiotingAlert {..} => "RiotingAlert",
			UIMode::EndGameWindow => "EndGameWindow",
			UIMode::AboutWindow => "AboutWindow",
			UIMode::CivicAdvisorsWindow => "CivicAdvisorsWindow",
			UIMode::UnmovedUnitsNotification => "UnmovedUnitsNotification",
			UIMode::MvWithCursorNoActionsRemainAlert {..} => "MvWithCursorNoActionsRemainAlert",
			UIMode::ForeignUnitInSectorAlert {..} => "ForeignUnitInSectorAlert",
			UIMode::NoblePedigree {..} => "NoblePedigree"
		})
	}
}*/


