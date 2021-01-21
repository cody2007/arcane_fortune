use super::*;
use crate::layers::{ImgParams, SumType, SumReduceParams};

// computes:
//
//	pred and labels shape: [images, c, h, w]
//
//	output =  
//           sum((pred[img, :] - labels[img, :])**2) ### (output is scalar)
impl Model {
	pub fn add_least_square_loss(&mut self, data_type: cudnnDataType_t) {
		debug_assert!(self.layers.len() > 0);
		
		let pred_ind = self.layers.len() - 1;
		let pred = self.layers[pred_ind].y.tensor();
		
		debug_assert!(pred.mem.dataType == data_type);
		let labels_shape = pred.shape;
		
		///////////////////////////////
		// labels
		self.add_imgs(ImgParams {shape: labels_shape, data_type});
		
		// predictions - mean
		//    pred_zmean[img] =
		//		pred[img,:] - pred_mean[img]   (output shape: [imgs, cat])
		let sub = AddParams {
				alpha1: -1.,
				alpha2: 1.,
				data_type
		};
		
		self.add_add(pred_ind, sub);
		
		self.add_pow(PowParams {alpha: 2., data_type});
		
		// sum correlations across images (scalar output)
		self.add_sum_reduce(SumReduceParams {sum_type: SumType::All, data_type});
	}
}

