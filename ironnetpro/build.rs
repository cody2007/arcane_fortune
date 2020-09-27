use std::env::var;
use std::process::Command;

fn main() {
	// cuda path
	println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
	
	///////////////////// for embedding strings of these values in the binary
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

