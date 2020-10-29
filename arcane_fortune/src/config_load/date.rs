// Source: https://docs.rs/crate/httpdate/0.3.2/source/src/lib.rs
// Jan-31-2020

/*
Copyright (c) 2016 Pyfisch

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
*/

#[allow(unused_imports)]
use std::fmt::{self, Display, Formatter};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::renderer::endwin;

#[derive(Copy, Clone, Debug)]
pub struct HttpDate {
    /// 0...59
    sec: u8,
    /// 0...59
    min: u8,
    /// 0...23
    hour: u8,
    /// 1...31
    day: u8,
    /// 1...12
    mon: u8,
    /// 1970...9999
    year: u16,
    /// 1...7
    wday: u8,
}

impl From<SystemTime> for HttpDate {
    fn from(v: SystemTime) -> HttpDate {
        let dur = v.duration_since(UNIX_EPOCH)
            .expect("all times should be after the epoch");
        let secs_since_epoch = dur.as_secs();

        if secs_since_epoch >= 253402300800 { // year 9999
            panicq!("date must be before year 9999");
        }

        /* 2000-03-01 (mod 400 year, immediately after feb29 */
        const LEAPOCH: i64 = 11017;
        const DAYS_PER_400Y: i64 = 365*400 + 97;
        const DAYS_PER_100Y: i64 = 365*100 + 24;
        const DAYS_PER_4Y: i64 = 365*4 + 1;

        let days = (secs_since_epoch / 86400) as i64 - LEAPOCH;
        let secs_of_day = secs_since_epoch % 86400;

        let mut qc_cycles = days / DAYS_PER_400Y;
        let mut remdays = days % DAYS_PER_400Y;

        if remdays < 0 {
            remdays += DAYS_PER_400Y;
            qc_cycles -= 1;
        }

        let mut c_cycles = remdays / DAYS_PER_100Y;
        if c_cycles == 4 { c_cycles -= 1; }
        remdays -= c_cycles * DAYS_PER_100Y;

        let mut q_cycles = remdays / DAYS_PER_4Y;
        if q_cycles == 25 { q_cycles -= 1; }
        remdays -= q_cycles * DAYS_PER_4Y;

        let mut remyears = remdays / 365;
        if remyears == 4 { remyears -= 1; }
        remdays -= remyears * 365;

        let mut year = 2000 +
            remyears + 4*q_cycles + 100*c_cycles + 400*qc_cycles;

        let months = [31,30,31,30,31,31,30,31,30,31,31,29];
        let mut mon = 0;
        for mon_len in months.iter() {
            mon += 1;
            if remdays < *mon_len {
                break;
            }
            remdays -= *mon_len;
        }
        let mday = remdays+1;
        let mon = if mon + 2 > 12 {
            year += 1;
            mon - 10
        } else {
            mon + 2
        };

        let mut wday = (3+days)%7;
        if wday <= 0 {
            wday += 7
        };

        HttpDate {
            sec: (secs_of_day % 60) as u8,
            min: ((secs_of_day % 3600) / 60) as u8,
            hour: (secs_of_day / 3600) as u8,
            day: mday as u8,
            mon: mon as u8,
            year: year as u16,
            wday: wday as u8,
        }
    }
}

impl HttpDate {
	pub fn string(&self) -> String {
		let wday = match self.wday {
			1 => "Mon",
			2 => "Tue",
			3 => "Wed",
			4 => "Thu",
			5 => "Fri",
			6 => "Sat",
			7 => "Sun",
			_ => unreachable!(),
		};
		
		let (hour, am) = if self.hour == 0 {
			(12, "AM")
		} else if self.hour <= 12 {
			(self.hour, "AM")
		} else {
			((self.hour as i8 - 12) as u8, "PM")
		};
		
		let mon = match self.mon {
			1 => "Jan",
			2 => "Feb",
			3 => "Mar",
			4 => "Apr",
			5 => "May",
			6 => "Jun",
			7 => "Jul",
			8 => "Aug",
			9 => "Sep",
			10 => "Oct",
			11 => "Nov",
			12 => "Dec",
			_ => unreachable!(),
		};
		
		let h_sp = if hour < 10 {" "} else {""};
		//let m_sp = if self.mon < 10 {" "} else {""};
		let d_sp = if self.day < 10 {"0"} else {""};
		let mi_sp = if self.min < 10 {"0"} else {""};
		
		//format!("{} {}{}-{}{}-{}  {}{}:{}{} {} (GMT)", wday, m_sp, mon, d_sp, self.day, self.year, h_sp, hour, mi_sp, self.min, am)
		format!("{} {}-{}{}-{}  {}{}:{}{} {} (GMT)", wday, mon, d_sp, self.day, self.year, h_sp, hour, mi_sp, self.min, am)

	}
}

/*impl Display for HttpDate {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let wday = match self.wday {
            1 => b"Mon",
            2 => b"Tue",
            3 => b"Wed",
            4 => b"Thu",
            5 => b"Fri",
            6 => b"Sat",
            7 => b"Sun",
            _ => unreachable!(),
        };
        let mon = match self.mon {
            1 => b"Jan",
            2 => b"Feb",
            3 => b"Mar",
            4 => b"Apr",
            5 => b"May",
            6 => b"Jun",
            7 => b"Jul",
            8 => b"Aug",
            9 => b"Sep",
            10 => b"Oct",
            11 => b"Nov",
            12 => b"Dec",
            _ => unreachable!(),
        };
        let mut buf: [u8; 29] = [
            // Too long to write as: b"Thu, 01 Jan 1970 00:00:00 GMT"
            b' ', b' ', b' ', b',', b' ',
            b'0', b'0', b' ', b' ', b' ', b' ', b' ',
            b'0', b'0', b'0', b'0', b' ',
            b'0', b'0', b':', b'0', b'0', b':', b'0', b'0',
            b' ', b'G', b'M', b'T',
        ];
        buf[0] = wday[0];
        buf[1] = wday[1];
        buf[2] = wday[2];
        buf[5] = b'0' + (self.day / 10) as u8;
        buf[6] = b'0' + (self.day % 10) as u8;
        buf[8] = mon[0];
        buf[9] = mon[1];
        buf[10] = mon[2];
        buf[12] = b'0' + (self.year / 1000) as u8;
        buf[13] = b'0' + (self.year / 100 % 10) as u8;
        buf[14] = b'0' + (self.year / 10 % 10) as u8;
        buf[15] = b'0' + (self.year % 10) as u8;
        buf[17] = b'0' + (self.hour / 10) as u8;
        buf[18] = b'0' + (self.hour % 10) as u8;
        buf[20] = b'0' + (self.min / 10) as u8;
        buf[21] = b'0' + (self.min % 10) as u8;
        buf[23] = b'0' + (self.sec / 10) as u8;
        buf[24] = b'0' + (self.sec % 10) as u8;
        f.write_str(unsafe { from_utf8_unchecked(&buf[..]) })
    }
}
*/

