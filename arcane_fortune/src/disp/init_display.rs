use super::*;
use crate::disp::menus::*;
use crate::keyboard::KeyboardMap;
use crate::localization::Localization;

impl <'f,'r,'bt,'ut,'rt,'dt>Disp<'f,'r,'bt,'ut,'rt,'dt> {
	pub fn new(mut renderer: &'r mut Renderer) -> Self {
		// should support 256 colors, the code below doesn't seem to detect correctly that it does
		#[cfg(target_os = "windows")]
		let terminal = TerminalSettings {limit_colors: false, limit_schars: false };
		
		#[cfg(target_os = "linux")]
		let terminal = if !has_colors()|| !can_change_color() || COLOR_PAIRS() < 256 {
			TerminalSettings { limit_colors: true, limit_schars: screen_reader_mode() }
		}else{
			TerminalSettings { limit_colors: false, limit_schars: screen_reader_mode() }
		};
		
		#[cfg(target_os = "macos")]
		let terminal = if !has_colors()|| !can_change_color() || COLOR_PAIRS() < 256 {
			TerminalSettings { limit_colors: true, limit_schars: true }
		}else{
			TerminalSettings { limit_colors: false, limit_schars: true }
		};
		
		let chars = init_color_pairs(&terminal, &mut renderer);
		let iface_settings = IfaceSettings::default("".to_string(), 0);
		let kbd = KeyboardMap::new();
		let local = Localization::new();
		let buttons = Buttons::new(&kbd, &local);
		
		let mut disp = Disp {
			ui_mode: UIMode::None,
			state: DispState {
				iface_settings,
				terminal,
				chars,
				menu_options: OptionsUI {options: Vec::new(), max_strlen: 0},
				production_options: ProdOptions {
					bldgs: Box::new([]),
					worker: OptionsUI {options: Vec::new(), max_strlen: 0}
				},
				txt_list: TxtList::new(),
				buttons,
				local,
				kbd,
				
				key_pressed: 0_i32,
				mouse_event: None,
				
				renderer
			}
		};
		
		init_menus(&mut disp.state, &Vec::new());
		disp
	}
}

