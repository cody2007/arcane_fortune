pub fn corr(x1_v: &Vec<f32>, x2_v: &Vec<f32>) -> f32 {
	debug_assert!(x1_v.len() == x2_v.len());
	let x1_mean = x1_v.iter().sum::<f32>() / x1_v.len() as f32;
	let x2_mean = x2_v.iter().sum::<f32>() / x2_v.len() as f32;
	
	let mut numer: f32 = 0.;
	let mut denom_x1: f32 = 0.;
	let mut denom_x2: f32 = 0.;
	for (x1, x2) in x1_v.iter().zip(x2_v.iter()) {
		let x1z = x1 - x1_mean;
		let x2z = x2 - x2_mean;
		
		numer += x1z * x2z;
		denom_x1 += x1z * x1z;
		denom_x2 += x2z * x2z;
	}
	
	numer / (denom_x1 * denom_x2).sqrt() 
}

use std::cmp::Ordering;
pub fn accuracy(x1_v: &Vec<f32>, x2_v: &Vec<f32>, n_categories: usize) -> f32 {
	debug_assert!(x1_v.len() == x2_v.len());
	debug_assert!((x1_v.len() % n_categories) == 0);
	
	let n_imgs = x1_v.len() / n_categories;
	let mut accuracy = 0.;
	for img in 0..n_imgs {
		let max_ind1 = x1_v.iter().skip(img*n_categories).take(n_categories).
			enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(index, _)| index);
		
		let max_ind2 = x2_v.iter().skip(img*n_categories).take(n_categories).
			enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|(index, _)| index);
		if max_ind1 == max_ind2 {
			accuracy += 1.;
		}
	}
	accuracy / n_imgs as f32
}

