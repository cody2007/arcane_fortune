use super::*;
use crate::layers::{ImgParams, MulParams, SumType, SumReduceParams};

// computes the correlation across categories for each image 
// between the predictions and one-hot-labels. then sums across images
// producing a scalar loss which should be maximized (1.0 is a perfect correlation)
//
//	pred and imgs shape: [images, categories, 1, 1]
//
//	output =  
//           sum(pearson(pred[img, :], labels[img, :]) ### (output is scalar)
impl Model {
	pub fn add_correlation_loss(&mut self, data_type: cudnnDataType_t) {
		debug_assert!(self.layers.len() > 0);
		
		let pred_ind = self.layers.len() - 1;
		let pred = self.layers[pred_ind].y.tensor();
		let n_categories = pred.shape.c as f32;
		let labels_shape = pred.shape;
		
		let sum123 = SumReduceParams {
				sum_type: SumType::Axes(vec!{1,2,3}),
				data_type
		};
		
		let sub_mean = AddParams {
				alpha1: -1./n_categories,
				alpha2: 1.,
				data_type
		};
		
		debug_assert!(pred.mem.dataType == data_type);
		debug_assert!(pred.shape.c != 1 && pred.shape.w == 1 && pred.shape.h == 1,
				"correlation_loss inputs must be [n, c, 1, 1] where c != 1");
		
		// mean across categories for each img
		// pred_mean[img] =
		//		pred[img, :].mean() (output shape: [imgs, 1])
		self.add_sum_reduce(sum123.clone());
		
		// predictions - mean
		//    pred_zmean[img] =
		//		pred[img,:] - pred_mean[img]   (output shape: [imgs, cat])
		self.add_add(pred_ind, sub_mean);
		let pred_zmean = self.layers.len() - 1;
		
		// zero mean sq sum (output shape: [imgs, 1])
		//	pred_zmean_sq_sum[img] = 
		//		sum(pred_zmean[img, :]**2)
		self.add_pow(PowParams {alpha: 2., data_type});
		self.add_sum_reduce(sum123.clone());
		let pred_zmean_sq_sum = self.layers.len() - 1;
		
		///////////////////////////////
		// labels
		self.add_imgs(ImgParams {shape: labels_shape, data_type});
		let labels_ind = self.layers.len() - 1;
		
		// mean across categories for each img
		// labels_mean[img] =
		//		labels[img, :].mean() (output shape: [imgs, 1])
		self.add_sum_reduce(sum123.clone());
		
		// labels - mean
		//    labels_zmean[img] =
		//		labels[img,:] - labels_mean[img]   (output shape: [imgs, cat])
		self.add_add(labels_ind, sub_mean);
		let labels_zmean = self.layers.len() - 1;
		
		///////////
		// numerator of correlation
		// 	numer[img] = sum(pred_zmean[img,:] * labels_zmean[img,:])
		//
		// (output shape: [imgs, 1])
		self.add_mul(pred_zmean, MulParams {data_type});
		self.add_sum_reduce(sum123.clone());
		let numer = self.layers.len() - 1;
		
		// zero mean sq sum (output shape: [imgs, 1])
		//	labels_zmean_sq_sum[img] = 
		//		sum(labels_zmean[img, :]**2)
		self.add_pow_layer_ind(PowParams {alpha: 2., data_type}, labels_zmean);
		self.add_sum_reduce(sum123.clone());
		
		///////////////
		// denominator of correlation
		// 	denom[img] = 
		//		1 / sqrt(pred_zmean_sq_sum[img] * labels_zmean[img])i
		//
		//	(output shape: [imgs, 1])
		self.add_mul(pred_zmean_sq_sum, MulParams {data_type});
		self.add_pow(PowParams {alpha: -0.5, data_type});
		
		////////////////
		// correlation for each image shape: [imgs, 1]
		//
		// 	r[img] = numerator[img] * denominator[img]
		self.add_mul(numer, MulParams {data_type});
		
		// sum correlations across images (scalar output)
		self.add_sum_reduce(SumReduceParams {sum_type: SumType::All, data_type});
	}
}

