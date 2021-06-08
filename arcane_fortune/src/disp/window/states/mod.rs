use super::*;
pub mod current_bldg_prod; pub use current_bldg_prod::*;
pub mod end_game; pub use end_game::*;
pub mod go_to_coordinate; pub use go_to_coordinate::*;
pub mod initial_game_window; pub use initial_game_window::*;
pub mod noble_pedigree; pub use noble_pedigree::*;
pub mod noble_units; pub use noble_units::*;
pub mod prod_list; pub use prod_list::*;
pub mod save_as; pub use save_as::*;
pub mod save_auto_freq; pub use save_auto_freq::*;
pub mod accept_nobility_into_empire; pub use accept_nobility_into_empire::*;
pub mod plot_window; pub use plot_window::*;
pub mod open; pub use open::*;
pub mod encyclopedia; pub use encyclopedia::*;
pub mod prevailing_doctrine_changed; pub use prevailing_doctrine_changed::*;
pub mod brigades; pub use brigades::*;
pub mod brigade_build_list; pub use brigade_build_list::*;
pub mod sectors; pub use sectors::*;
pub mod create_sector_automation; pub use create_sector_automation::*;
pub mod contact_embassy; pub use contact_embassy::*;
pub mod contact_nobility; pub use contact_nobility::*;
pub mod resources_available; pub use resources_available::*;
pub mod resources_discovered; pub use resources_discovered::*;
pub mod tech_discovered; pub use tech_discovered::*; // when a tech is discovered
pub mod discover_tech; pub use discover_tech::*; // for omnipresence mode
pub mod obtain_resource; pub use obtain_resource::*;
pub mod war_status; pub use war_status::*;
pub mod select_explore_type; pub use select_explore_type::*;
pub mod civilization_intel; pub use civilization_intel::*;
pub mod switch_to_player; pub use switch_to_player::*;
pub mod foreign_units_in_sector_alert; pub use foreign_units_in_sector_alert::*;
pub mod unmoved_units_notification; pub use unmoved_units_notification::*;
pub mod no_actions_remain_alert; pub use no_actions_remain_alert::*;
pub mod public_polling; pub use public_polling::*;
pub mod civic_advisors; pub use civic_advisors::*;
pub mod generic_alert; pub use generic_alert::*;
pub mod rioting_alert; pub use rioting_alert::*;
pub mod citizen_demand; pub use citizen_demand::*;
pub mod manors; pub use manors::*;
pub mod units; pub use units::*;
pub mod cities; pub use cities::*;
pub mod history; pub use history::*;
pub mod get_text; pub use get_text::*;
pub mod select_bldg_doctrine; pub use select_bldg_doctrine::*;
pub mod buildings; pub use buildings::*;
pub mod set_difficulty; pub use set_difficulty::*;
pub mod place_unit; pub use place_unit::*;
pub mod trade; pub use trade::*;
pub mod view_trade; pub use view_trade::*;
pub mod friends_and_foes; pub use friends_and_foes::*;
pub mod nobility_request; pub use nobility_request::*; // nobility requests main player do something
pub mod intro_nobility_join_options; pub use intro_nobility_join_options::*; // list of nobility that could join the empire. shown at the start of the game
pub mod nobility_declares_independence; pub use nobility_declares_independence::*;
pub mod set_noble_tax; pub use set_noble_tax::*; // set the noble's dues to the human's empire
pub mod zone_land; pub use zone_land::*;
pub mod budget; pub use budget::*;

pub enum UIModeControl<'bt,'ut,'rt,'dt> {
	UnChgd,
	Closed,
	CloseAndGoTo(u64),
	ChgGameControl,
	New(UIMode<'bt,'ut,'rt,'dt>)
}
