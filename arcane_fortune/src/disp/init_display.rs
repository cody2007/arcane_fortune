use super::*;

pub fn init_display() -> (DispState, DispSettings) {
	let d = setup_disp_lib();
	
	#[cfg(target_os = "windows")]
	{ // should support 256 colors, the code below doesn't seem to detect correctly that it does
		return (d, DispSettings {limit_colors: false, limit_schars: false });
	}
	
	#[cfg(target_os = "linux")]
	{
		if !has_colors()|| !can_change_color() || COLOR_PAIRS() < 256 {
			(d, DispSettings { limit_colors: true, limit_schars: false })
		}else{
			(d, DispSettings { limit_colors: false, limit_schars: false })
		}
	}
	
	#[cfg(target_os = "macos")]
	{
		if !has_colors()|| !can_change_color() || COLOR_PAIRS() < 256 {
			(d, DispSettings { limit_colors: true, limit_schars: true })
		}else{
			(d, DispSettings { limit_colors: false, limit_schars: true })
		}
	}

}

