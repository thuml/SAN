#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    if(bottom.size() > 2){
      caffe_gpu_memcpy(bottom[2]->count()*sizeof(Dtype), bottom[2]->gpu_data(), bottom[2]->mutable_cpu_data());
    }
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int data_num = bottom[0]->shape(0);
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, Dtype(1) * loss_weight / num, bottom_diff);
    if (bottom.size() > 2){
      accumulating_weight_ = bottom[2]->cpu_data()[data_num];
    }
    if(bottom.size() > 2){
      for (int i = 0; i < data_num; i++){
        caffe_gpu_scal(bottom[0]->count(1), bottom[2]->cpu_data()[i], bottom_diff+i*bottom[0]->count(1));
      }
      caffe_gpu_scal(count, accumulating_weight_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
