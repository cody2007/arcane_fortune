//use crate::disp_lib::*;

macro_rules! debug_assertq {
	($val: expr, $txt: expr, $($p_args:expr),* ) => {
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			if !$val {
				endwin();
				panic!($txt, $($p_args),*);
			}
		}
	};
	($val: expr, $txt: expr) => {
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			if !$val {
				endwin();
				panic!($txt);
			}
		}
	};
	($val: expr) => {
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			if !$val {
				endwin();
				assert!($val);
			}
		}
	}
}

macro_rules! assertq {
	($val: expr, $txt: expr, $($p_args:expr),* ) => {
		if !$val {
			endwin();
			panic!($txt, $($p_args),*);
		}
	};
	($val: expr, $txt: expr) => {
		if !$val {
			endwin();
			panic!($txt);
		}
	};
	($val: expr) => {
		if !$val {
			endwin();
			assert!($val);
		}
	}
}

macro_rules! panicq {
	($txt: expr, $($p_args: expr),*) => {{
			endwin();
			panic!($txt, $($p_args),*)
	}};
	($txt: expr) => {{
			endwin();
			panic!($txt)
	}};
	() => {{
		endwin();
		panic!();
	}}
}

macro_rules! debug_printlnq {
	($txt: expr, $($p_args: expr),*) => {
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			endwin();
			println!($txt, $($p_args),*);
		}
	};
	($txt: expr) => {
		#[cfg(any(feature="opt_debug", debug_assertions))]
		{
			endwin();
			println!($txt);
		}
	}
}

macro_rules! printlnq {
	($txt: expr, $($p_args: expr),*) => {{
		endwin();
		println!($txt, $($p_args),*);
	}};
	($txt: expr) => {{
		endwin();
		println!($txt);
	}}
}

