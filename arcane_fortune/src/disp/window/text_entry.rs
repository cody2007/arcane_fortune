macro_rules! do_txt_entry_keys{($key_pressed: expr, $curs_col: expr, $input_txt: expr,
					  $printable_type: expr, $iface_settings: expr, $d: expr) => {
	match $key_pressed {
		KEY_ESC => {end_window($iface_settings, $d);}
		k if k == '\n' as i32 => {
			// should be handled by outer function
		}
		KEY_LEFT => {if *$curs_col != 0 {*$curs_col -= 1;}}
		KEY_RIGHT => {
			if *$curs_col < ($input_txt.len() as isize) {
				*$curs_col += 1;
			}
		}
		
		KEY_HOME | KEY_UP => {*$curs_col = 0;}
		
		// end key
		KEY_DOWN | 0x166 | 0602 => {*$curs_col = $input_txt.len() as isize;}
		
		// backspace
		KEY_BACKSPACE | 127 | 0x8  => {
			if *$curs_col != 0 {
				*$curs_col -= 1;
				$input_txt.remove(*$curs_col as usize);
			}
		}
		
		// delete
		KEY_DC => {
			if *$curs_col != $input_txt.len() as isize {
				$input_txt.remove(*$curs_col as usize);
			}
		}
		_ => { // insert character
			if $input_txt.len() < (min(MAX_SAVE_AS_W, $iface_settings.screen_sz.w)-5) {
				if let Result::Ok(c) = u8::try_from($key_pressed) {
					if let Result::Ok(ch) = char::try_from(c) {
						if is_printable(ch, $printable_type) {
							$input_txt.insert(*$curs_col as usize, ch);
							*$curs_col += 1;
						}
					}
				}
			}
		} // remainder characters
	} // match
}}

