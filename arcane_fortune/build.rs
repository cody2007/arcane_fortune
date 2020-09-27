use std::env::var;
use std::process::Command;

fn main() {
	// attempt to get the binary to search in the current directory for the library
	//println!("cargo:rustc-flags=-Wl,-rpath,$ORIGIN");
	
	println!("cargo:rustc-env=TARGET={}", var("TARGET").unwrap());
	println!("cargo:rustc-env=PROFILE={}", var("PROFILE").unwrap());
	println!("cargo:rustc-env=HOST={}", var("HOST").unwrap());
	println!("cargo:rustc-env=OPT_LEVEL={}", var("OPT_LEVEL").unwrap());

	///// set rust version as environmental variable
	let mut rust_v = Command::new("rustc");
	rust_v.arg("-V");
	let txt = rust_v.output().expect("process failed to execute rustc");
	
	println!("cargo:rustc-env=RUSTV={}", String::from_utf8(txt.stdout).unwrap());
	
	//// build time
	let mut d = Command::new("date");
	let txt = d.output().expect("process failed to execute date");
	
	println!("cargo:rustc-env=COMPILE_DATE={}", String::from_utf8(txt.stdout).unwrap());
}

