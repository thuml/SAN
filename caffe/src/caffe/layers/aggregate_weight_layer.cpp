#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/aggregate_weight_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AggregateWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //bias_term_ = this->layer_param_.inner_product_param().bias_term();
  num_output_ = top.size();
  class_blob_.Reshape(num_output_, 1, 1, 1);
  caffe_gpu_set(num_output_, Dtype(0), class_blob_.mutable_gpu_data());
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  num_data_ = bottom[0]->shape(0);
  for(int i = 0; i < num_output_; i++){
    top[i]->Reshape(num_data_+1, 1,1,1);
  }
  weight_blob_.Reshape(num_output_, num_data_+1,1,1);
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AggregateWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(AggregateWeightLayer);
#endif

INSTANTIATE_CLASS(AggregateWeightLayer);
REGISTER_LAYER_CLASS(AggregateWeight);

}  // namespace caffe
