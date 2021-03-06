/*
 num_cpus crate
 https://github.com/seanmonstar/num_cpus/blob/master/LICENSE-MIT
 Accessed April 8, 2020

Copyright (c) 2015

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

// Accessed April 8, 2020
// https://raw.githubusercontent.com/seanmonstar/num_cpus/master/src/lib.rs
*/

//! A crate with utilities to determine the number of CPUs available on the
//! current system.
//!
//! Sometimes the CPU will exaggerate the number of CPUs it contains, because it can use
//! [processor tricks] to deliver increased performance when there are more threads. This 
//! crate provides methods to get both the logical and physical numbers of cores.
//!
//! This information can be used as a guide to how many tasks can be run in parallel.
//! There are many properties of the system architecture that will affect parallelism,
//! for example memory access speeds (for all the caches and RAM) and the physical
//! architecture of the processor, so the number of CPUs should be used as a rough guide
//! only.
//!
//!
//! ## Examples
//!
//! Fetch the number of logical CPUs.
//!
//! ```
//! let cpus = num_cpus::get();
//! ```
//!
//! See [`rayon::Threadpool`] for an example of where the number of CPUs could be
//! used when setting up parallel jobs (Where the threadpool example uses a fixed
//! number 8, it could use the number of CPUs).
//!
//! [processor tricks]: https://en.wikipedia.org/wiki/Simultaneous_multithreading
//! [`rayon::ThreadPool`]: https://docs.rs/rayon/1.*/rayon/struct.ThreadPool.html
#![cfg_attr(test, deny(warnings))]
#![deny(missing_docs)]
#![doc(html_root_url = "https://docs.rs/num_cpus/1.12.0")]
#![allow(non_snake_case)]

#[cfg(target_os = "hermit")]
extern crate hermit_abi;

/// Returns the number of available CPUs of the current system.
///
/// This function will get the number of logical cores. Sometimes this is different from the number
/// of physical cores (See [Simultaneous multithreading on Wikipedia][smt]).
///
/// # Examples
///
/// ```
/// let cpus = num_cpus::get();
/// if cpus > 1 {
///     println!("We are on a multicore system with {} CPUs", cpus);
/// } else {
///     println!("We are on a single core system");
/// }
/// ```
///
/// # Note
///
/// This will check [sched affinity] on Linux, showing a lower number of CPUs if the current 
/// thread does not have access to all the computer's CPUs. 
///
/// [smt]: https://en.wikipedia.org/wiki/Simultaneous_multithreading
/// [sched affinity]: http://www.gnu.org/software/libc/manual/html_node/CPU-Affinity.html
#[inline]
pub fn get() -> usize {
    get_num_cpus()
}

/// Returns the number of physical cores of the current system.
///
/// # Note
///
/// Physical count is supported only on Linux, mac OS and Windows platforms.
/// On other platforms, or if the physical count fails on supported platforms,
/// this function returns the same as [`get()`], which is the number of logical
/// CPUS.
///
/// # Examples
///
/// ```
/// let logical_cpus = num_cpus::get();
/// let physical_cpus = num_cpus::get_physical();
/// if logical_cpus > physical_cpus {
///     println!("We have simultaneous multithreading with about {:.2} \
///               logical cores to 1 physical core.", 
///               (logical_cpus as f64) / (physical_cpus as f64));
/// } else if logical_cpus == physical_cpus {
///     println!("Either we don't have simultaneous multithreading, or our \
///               system doesn't support getting the number of physical CPUs.");
/// } else {
///     println!("We have less logical CPUs than physical CPUs, maybe we only have access to \
///               some of the CPUs on our system.");
/// }
/// ```
///
/// [`get()`]: fn.get.html
#[inline]
pub fn get_physical() -> usize {
    get_num_physical_cpus()
}


#[cfg(not(any(target_os = "linux", target_os = "windows", target_os="macos", target_os="openbsd")))]
#[inline]
fn get_num_physical_cpus() -> usize {
    // Not implemented, fall back
    get_num_cpus()
}

#[cfg(target_os = "windows")]
fn get_num_physical_cpus() -> usize {
    match get_num_physical_cpus_windows() {
        Some(num) => num,
        None => get_num_cpus()
    }
}

#[cfg(target_os = "windows")]
fn get_num_physical_cpus_windows() -> Option<usize> {
    // Inspired by https://msdn.microsoft.com/en-us/library/ms683194

    use std::ptr;
    use std::mem;

    #[allow(non_upper_case_globals)]
    const RelationProcessorCore: u32 = 0;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct SYSTEM_LOGICAL_PROCESSOR_INFORMATION {
        mask: usize,
        relationship: u32,
        _unused: [u64; 2]
    }

    extern "system" {
        fn GetLogicalProcessorInformation(
            info: *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
            length: &mut u32
        ) -> u32;
    }

    // First we need to determine how much space to reserve.

    // The required size of the buffer, in bytes.
    let mut needed_size = 0;

    unsafe {
        GetLogicalProcessorInformation(ptr::null_mut(), &mut needed_size);
    }

    let struct_size = mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>() as u32;

    // Could be 0, or some other bogus size.
    if needed_size == 0 || needed_size < struct_size || needed_size % struct_size != 0 {
        return None;
    }

    let count = needed_size / struct_size;

    // Allocate some memory where we will store the processor info.
    let mut buf = Vec::with_capacity(count as usize);

    let result;

    unsafe {
        result = GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut needed_size);
    }

    // Failed for any reason.
    if result == 0 {
        return None;
    }

    let count = needed_size / struct_size;

    unsafe {
        buf.set_len(count as usize);
    }

    let phys_proc_count = buf.iter()
        // Only interested in processor packages (physical processors.)
        .filter(|proc_info| proc_info.relationship == RelationProcessorCore)
        .count();

    if phys_proc_count == 0 {
        None
    } else {
        Some(phys_proc_count)
    }
}

#[cfg(target_os = "linux")]
fn get_num_physical_cpus() -> usize {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = match File::open("/proc/cpuinfo") {
        Ok(val) => val,
        Err(_) => return get_num_cpus(),
    };
    let reader = BufReader::new(file);
    let mut map = HashMap::new();
    let mut physid: u32 = 0;
    let mut cores: usize = 0;
    let mut chgcount = 0;
    for line in reader.lines().filter_map(|result| result.ok()) {
        let mut it = line.split(':');
        let (key, value) = match (it.next(), it.next()) {
            (Some(key), Some(value)) => (key.trim(), value.trim()),
            _ => continue,
        };
        if key == "physical id" {
            match value.parse() {
                Ok(val) => physid = val,
                Err(_) => break,
            };
            chgcount += 1;
        }
        if key == "cpu cores" {
            match value.parse() {
                Ok(val) => cores = val,
                Err(_) => break,
            };
            chgcount += 1;
        }
        if chgcount == 2 {
            map.insert(physid, cores);
            chgcount = 0;
        }
    }
    let count = map.into_iter().fold(0, |acc, (_, cores)| acc + cores);

    if count == 0 {
        get_num_cpus()
    } else {
        count
    }
}

#[cfg(windows)]
fn get_num_cpus() -> usize {
    #[repr(C)]
    struct SYSTEM_INFO {
        wProcessorArchitecture: u16,
        wReserved: u16,
        dwPageSize: u32,
        lpMinimumApplicationAddress: *mut u8,
        lpMaximumApplicationAddress: *mut u8,
        dwActiveProcessorMask: *mut u8,
        dwNumberOfProcessors: u32,
        dwProcessorType: u32,
        dwAllocationGranularity: u32,
        wProcessorLevel: u16,
        wProcessorRevision: u16,
    }

    extern "system" {
        fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
    }

    unsafe {
        let mut sysinfo: SYSTEM_INFO = std::mem::zeroed();
        GetSystemInfo(&mut sysinfo);
        sysinfo.dwNumberOfProcessors as usize
    }
}

#[cfg(target_os = "macos")]
fn get_num_physical_cpus() -> usize {
    use std::ffi::CStr;
    use std::ptr;

    let mut cpus: i32 = 0;
    let mut cpus_size = std::mem::size_of_val(&cpus);

    let sysctl_name = CStr::from_bytes_with_nul(b"hw.physicalcpu\0")
        .expect("byte literal is missing NUL");

    unsafe {
        if 0 != libc::sysctlbyname(sysctl_name.as_ptr(),
                                   &mut cpus as *mut _ as *mut _,
                                   &mut cpus_size as *mut _ as *mut _,
                                   ptr::null_mut(),
                                   0) {
            return get_num_cpus();
        }
    }
    cpus as usize
}

#[cfg(target_os = "linux")]
fn get_num_cpus() -> usize {
    let mut set:  libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) } == 0 {
        let mut count: u32 = 0;
        for i in 0..libc::CPU_SETSIZE as usize {
            if unsafe { libc::CPU_ISSET(i, &set) } {
                count += 1
            }
        }
        count as usize
    } else {
        let cpus = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if cpus < 1 {
            1
        } else {
            cpus as usize
        }
    }
}

#[cfg(any(
    target_os = "nacl",
    target_os = "macos",
    target_os = "ios",
    target_os = "android",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "fuchsia")
)]
fn get_num_cpus() -> usize {
    // On ARM targets, processors could be turned off to save power.
    // Use `_SC_NPROCESSORS_CONF` to get the real number.
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    const CONF_NAME: libc::c_int = libc::_SC_NPROCESSORS_CONF;
    #[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
    const CONF_NAME: libc::c_int = libc::_SC_NPROCESSORS_ONLN;

    let cpus = unsafe { libc::sysconf(CONF_NAME) };
    if cpus < 1 {
        1
    } else {
        cpus as usize
    }
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    windows,
)))]
fn get_num_cpus() -> usize {
    1
}

