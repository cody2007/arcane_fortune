use super::*;
use crate::layers::{ImgParams, SoftmaxParams, MulParams, SumType, SumReduceParams};

// computes the softmax across categories for each image, (prev layer should be: [imgs, categories, 1, 1])
// then takes the point-wise log of category predictions, then multiplies
// with the one-hot labels and sums across all images for a scalar 
// (which should be maximized: 
//		* 0 indicates perfect classification (the maximum)
//		* negative values indicate less-than-perfect classification)
//
// sum_all( label_one_hot[img,cat] * log(softmax(pred[img,cat])) )

impl Model {
	pub fn add_softmax_cross_entropy_loss(&mut self, data_type: cudnnDataType_t) {
		debug_assert!(self.layers.len() > 0);
		
		//// log(softmax(pred))
		self.add_softmax_log(SoftmaxParams {data_type});
		
		let pred_ind = self.layers.len() - 1;
		let pred = self.layers[pred_ind].y.tensor();
		
		debug_assert!(pred.mem.dataType == data_type);
		
		//// labels
		let labels_shape = pred.shape;
		self.add_imgs(ImgParams {shape: labels_shape, data_type});
		
		// sum(labels * log(softmax(pred)))
		self.add_mul(pred_ind, MulParams {data_type});
		self.add_sum_reduce(SumReduceParams {
				sum_type: SumType::All,
				data_type
		});
	}
}

